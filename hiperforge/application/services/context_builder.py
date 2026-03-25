"""
ContextBuilder — Construye el contexto completo para cada llamada al LLM.

Tiene dos responsabilidades distintas pero relacionadas:

  1. CONSTRUCCIÓN DEL PROMPT DE SISTEMA:
     El prompt de sistema es las "instrucciones de trabajo" del agente.
     Le dice al LLM quién es, qué puede hacer, y exactamente cómo debe
     comunicarse. Se genera una vez al inicio de cada subtask.

     Un prompt de sistema bien construido es la diferencia entre un agente
     que usa las tools correctamente y uno que inventa resultados o usa
     el formato incorrecto.

  2. GESTIÓN DEL HISTORIAL DE MENSAJES:
     A medida que el loop ReAct itera, el historial crece: mensajes del
     sistema, del usuario, del agente, y resultados de tools. En algún
     punto el historial supera el context window del LLM y hay que truncar.

     La estrategia de truncado es crítica — una mala estrategia puede
     hacer que el agente "olvide" el objetivo original o pierda contexto
     de un error anterior que explica el estado actual.

FORMATO DE COMUNICACIÓN DEL AGENTE:
  El agente se comunica con el LLM usando JSON estructurado en el content.
  Este módulo define ese protocolo y lo documenta en el prompt de sistema.

  El LLM SIEMPRE responde con uno de estos tres formatos:

    Ejecutar una tool:
      {"action": "tool_call", "tool": "<nombre>", "arguments": {<args>}}

    Razonar sin actuar (pensar en voz alta):
      {"action": "think", "content": "<razonamiento>"}

    Indicar que la subtask está completa:
      {"action": "complete", "summary": "<qué se logró>"}

  Este protocolo funciona igual con Anthropic, OpenAI, Groq y Ollama
  porque no depende del tool use nativo de ningún proveedor — vive
  completamente en el contenido del mensaje.

ESTRATEGIA DE TRUNCADO DEL HISTORIAL:
  Cuando el historial excede el espacio disponible en el context window:

    PRESERVAR SIEMPRE:
      - Mensaje de sistema (instrucciones del agente)
      - Los N mensajes más recientes (contexto inmediato)

    ELIMINAR PRIMERO:
      - Mensajes más antiguos del historial conversacional
      - Siempre en pares (user + assistant) para mantener coherencia

    NUNCA ELIMINAR:
      - El mensaje de sistema — sin él el agente pierde su identidad
      - Los últimos 4 mensajes — son el contexto inmediato más relevante
"""

from __future__ import annotations

import os
from typing import Any

from hiperforge.core.constants import LLM_CONTEXT_RESPONSE_RESERVE
from hiperforge.core.logging import get_logger
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.tools.base import ToolRegistry, ToolSchema

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Prompt de sistema base del agente
#
# Este texto define la personalidad, el comportamiento y el protocolo
# de comunicación del agente. Es lo más importante que recibe el LLM.
#
# Principios de diseño de este prompt:
#   - Claro y sin ambigüedad — el LLM no debe interpretar, debe seguir.
#   - El protocolo JSON está completamente documentado con ejemplos.
#   - Las reglas críticas están explícitas y marcadas como OBLIGATORIAS.
#   - El contexto de la subtask actual se inyecta dinámicamente al final.
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT_BASE = """\
Eres HiperForge, un agente autónomo de desarrollo.
Completa la subtask con las herramientas disponibles y responde SOLO JSON.

PROTOCOLO:
1. Tool:
{{"action":"tool_call","tool":"<nombre>","arguments":{{<args>}}}}
2. Think:
{{"action":"think","content":"<razonamiento breve>"}}
3. Complete:
{{"action":"complete","summary":"<qué hiciste y cómo lo verificaste>"}}

REGLAS:
- Usa la herramienta más específica posible.
- Si ya conoces el archivo objetivo, actúa directamente sobre él; no explores el repo sin necesidad.
- No uses listados amplios (`ls`, `find`, `file list`) salvo que la ubicación sea realmente desconocida.
- No releas completo un archivo que acabas de escribir o parchear, salvo que necesites una línea puntual.
- Tras un cambio exitoso, verifica una vez; si ya quedó comprobado, responde con "complete".
- No repitas la misma acción fallida sin cambiar algo.
- No repitas verificaciones equivalentes que ya pasaron.
- Lee errores completos antes de reintentar.
- Usa "complete" solo si verificaste el resultado o si explicas claramente el bloqueo.
- Intenta como máximo {max_retries} estrategias distintas.
"""

# Plantilla para la sección de tools disponibles
_TOOLS_SECTION_HEADER = "HERRAMIENTAS:"

# Plantilla para el contexto de la subtask actual
_SUBTASK_CONTEXT_TEMPLATE = """\
CONTEXTO:
- Tarea: {task_prompt}
- Subtask: {subtask_description}
- Directorio: {working_dir}

Completa la subtask y responde SOLO con JSON válido.\
"""


class ContextBuilder:
    """
    Construye el contexto de mensajes para cada llamada al LLM.

    Responsabilidades:
      1. Generar el mensaje de sistema completo para una subtask.
      2. Truncar el historial cuando se acerca al context window.

    Parámetros:
        registry: ToolRegistry con las tools disponibles.
                  Se usa para incluir los schemas en el prompt de sistema.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # Construcción del prompt de sistema
    # ------------------------------------------------------------------

    def build_system_message(
        self,
        subtask_description: str,
        task_prompt: str,
        *,
        working_dir: str | None = None,
        max_retries: int = 3,
    ) -> Message:
        """
        Construye el mensaje de sistema completo para una subtask.

        El mensaje de sistema es la base de todo el contexto del agente.
        Incluye en orden:
          1. Instrucciones base del agente y protocolo de comunicación
          2. Lista de herramientas disponibles con sus parámetros
          3. Contexto específico de la task y subtask actuales

        Parámetros:
            subtask_description: Qué debe hacer el agente en esta subtask.
            task_prompt:         Instrucción original del usuario (contexto global).
            working_dir:         Directorio de trabajo del proyecto.
                                 None = usa el directorio actual del proceso.
            max_retries:         Máximo de estrategias que el agente intenta
                                 antes de rendirse. Se inyecta en el prompt.

        Returns:
            Message con role=SYSTEM listo para incluir en el historial.
        """
        effective_dir = working_dir or os.getcwd()
        compact_task_prompt = task_prompt.strip()[:240]
        compact_subtask_description = subtask_description.strip()[:280]

        sections: list[str] = []

        # Sección 1: instrucciones base con el max_retries inyectado
        sections.append(
            _SYSTEM_PROMPT_BASE.format(max_retries=max_retries)
        )

        # Sección 2: herramientas disponibles
        tools_section = self._build_tools_section()
        if tools_section:
            sections.append(tools_section)

        # Sección 3: contexto de la subtask actual
        sections.append(
            _SUBTASK_CONTEXT_TEMPLATE.format(
                task_prompt=compact_task_prompt,
                subtask_description=compact_subtask_description,
                working_dir=effective_dir,
            )
        )

        full_prompt = "\n\n".join(sections)

        return Message.system(full_prompt)

    def _build_tools_section(self) -> str:
        """
        Construye la sección del prompt que describe las tools disponibles.

        Para cada tool incluye:
          - Nombre (el string exacto que debe ir en "tool")
          - Descripción de para qué sirve y cuándo usarla
          - Parámetros con tipo, descripción y si son requeridos

        La calidad de esta sección impacta directamente en qué tan bien
        el agente usa las tools. Una descripción ambigua lleva a tool calls
        con argumentos incorrectos.

        Returns:
            String con la sección de tools formateada.
            String vacío si no hay tools registradas.
        """
        schemas = self._registry.get_schemas()

        if not schemas:
            return ""

        lines: list[str] = [_TOOLS_SECTION_HEADER]

        for schema in schemas:
            # Encabezado de la tool
            lines.append(f'- {schema.name}: {schema.description.split(".")[0].strip()[:72]}')

            # Parámetros
            props: dict[str, Any] = schema.parameters.get("properties", {})
            required: list[str] = schema.parameters.get("required", [])

            if props:
                lines.append("  Params:")
                for param_name, param_info in props.items():
                    is_req = "*" if param_name in required else ""
                    param_type = param_info.get("type", "any")
                    param_desc = param_info.get("description", "")
                    first_line = param_desc.split("\n")[0].strip()[:48]
                    lines.append(
                        f"    - {param_name}{is_req} ({param_type}): {first_line}"
                    )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Gestión del historial de mensajes
    # ------------------------------------------------------------------

    def truncate_messages_for_context_window(
        self,
        messages: list[Message],
        context_window_size: int,
        max_tokens_response: int,
    ) -> list[Message]:
        """
        Trunca el historial para que quepa en el context window del LLM.

        ALGORITMO:
          1. Calcular tokens disponibles para el historial.
          2. Si el historial cabe, devolverlo sin cambios.
          3. Si no cabe:
             a. Separar mensaje de sistema del historial conversacional.
             b. Identificar los últimos 4 mensajes como "contexto protegido".
             c. Eliminar mensajes del inicio del historial hasta que quepa.
             d. Nunca eliminar el mensaje de sistema.
             e. Nunca eliminar los últimos 4 mensajes.

        ESTIMACIÓN DE TOKENS:
          Usamos 4 caracteres ≈ 1 token como heurística.
          Es suficientemente precisa para decisiones de truncado.
          Si necesitas precisión exacta, el adapter del LLM puede
          sobreescribir este método con el tokenizer real del proveedor.

        Parámetros:
            messages:            Historial completo (sistema + conversacional).
            context_window_size: Tamaño máximo del context window del modelo.
            max_tokens_response: Tokens reservados para la respuesta del LLM.

        Returns:
            Lista de mensajes que cabe en el context window disponible.
            Siempre incluye el mensaje de sistema si existe.
        """
        # Tokens disponibles para el historial de mensajes
        available_tokens = (
            context_window_size
            - max_tokens_response
            - LLM_CONTEXT_RESPONSE_RESERVE
        )

        if available_tokens <= 0:
            # Configuración extrema — preservamos solo el sistema
            logger.warning(
                "context window insuficiente — preservando solo mensaje de sistema",
                context_window=context_window_size,
                max_tokens_response=max_tokens_response,
                reserve=LLM_CONTEXT_RESPONSE_RESERVE,
            )
            return [m for m in messages if m.role == Role.SYSTEM]

        # Si el historial completo cabe, no truncamos nada
        if self._estimate_tokens(messages) <= available_tokens:
            return messages

        # Separamos el sistema (intocable) del historial conversacional
        system_msgs = [m for m in messages if m.role == Role.SYSTEM]
        conv_msgs = [m for m in messages if m.role != Role.SYSTEM]

        # Los últimos 4 mensajes son el contexto inmediato — siempre se preservan.
        # 4 = [user_subtask, assistant_think, user_tool_result, assistant_think]
        # Este contexto es crítico para que el agente entienda dónde está en la subtask.
        protected_count = min(4, len(conv_msgs))
        protected = conv_msgs[-protected_count:] if protected_count > 0 else []
        truncatable = conv_msgs[:-protected_count] if protected_count > 0 else conv_msgs

        # Eliminamos desde el inicio hasta que quepa
        while truncatable and (
            self._estimate_tokens(system_msgs + truncatable + protected) > available_tokens
        ):
            truncatable.pop(0)

        truncated = system_msgs + truncatable + protected

        removed_count = len(messages) - len(truncated)
        logger.info(
            "historial truncado para context window",
            original_count=len(messages),
            truncated_count=len(truncated),
            removed_count=removed_count,
            estimated_tokens=self._estimate_tokens(truncated),
            available_tokens=available_tokens,
        )

        return truncated

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_tokens(messages: list[Message]) -> int:
        """
        Estima el número de tokens del historial.

        Heurística: 4 caracteres ≈ 1 token.
        Agrega un overhead fijo de 4 tokens por mensaje para los metadatos
        que los proveedores añaden internamente (role, separadores, etc.).
        """
        CHARS_PER_TOKEN = 4
        OVERHEAD_PER_MESSAGE = 4

        total_chars = sum(len(m.content) for m in messages)
        total_overhead = len(messages) * OVERHEAD_PER_MESSAGE

        return (total_chars // CHARS_PER_TOKEN) + total_overhead
