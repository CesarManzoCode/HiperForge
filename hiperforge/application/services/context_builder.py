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

  El LLM SIEMPRE responde con uno de estos dos formatos:

    Ejecutar una tool:
      {"action": "tool_call", "tool": "<nombre>", "arguments": {<args>}}

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
import re
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
Completa la subtask con las herramientas disponibles.
Responde SOLO con UN bloque JSON por turno. No generes múltiples bloques JSON.

PROTOCOLO (elige exactamente uno):
1. Ejecutar herramienta:
{{"action":"tool_call","tool":"<nombre>","arguments":{{<args>}}}}
2. Subtask completada:
{{"action":"complete","summary":"<qué hiciste y cómo lo verificaste>"}}

REGLAS CRÍTICAS:
- Actúa directo. Si sabes qué hacer, hazlo sin explorar.
- NUNCA releas un archivo que acabas de escribir — ya conoces su contenido.
- Tras un cambio exitoso, verifica UNA vez y responde "complete".
- No repitas acciones fallidas sin cambiar la estrategia.
- No uses listados amplios (ls, find, file list) salvo que la ubicación sea desconocida.
- Si una tool falla, lee el error completo y cambia de enfoque.
- Máximo {max_retries} estrategias distintas antes de completar con lo que tengas.
- Responde con "complete" en cuanto la verificación pase.
- NUNCA inventes valores para parámetros opcionales. Si no necesitas un parámetro, OMÍTELO. Los defaults del sistema son correctos para la mayoría de casos.
- Si una tool rechaza tus argumentos, OMITE el parámetro problemático en vez de probar otro valor.
- Para crear o sobrescribir archivos usa file con operation="write". NO inventes operation="create".
- Para editar archivos existentes usa file con operation="patch" o "append" según corresponda.
- NO uses shell con redirecciones (>, >>, heredoc) para crear archivos si file puede hacerlo.
"""

# Plantilla para la sección de tools disponibles
_TOOLS_SECTION_HEADER = "HERRAMIENTAS:"

# Plantilla para el contexto de la subtask actual
_SUBTASK_CONTEXT_TEMPLATE = """\
CONTEXTO:
- Tarea: {task_prompt}
- Subtask: {subtask_description}
- Dir: {working_dir}
{previous_context}
Completa la subtask y responde SOLO con UN bloque JSON.\
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
        previous_subtask_summary: str | None = None,
    ) -> Message:
        """
        Construye el mensaje de sistema completo para una subtask.

        El mensaje de sistema es la base de todo el contexto del agente.
        Incluye en orden:
          1. Instrucciones base del agente y protocolo de comunicación
          2. Lista de herramientas disponibles con sus parámetros
          3. Contexto específico de la task y subtask actuales
          4. Resumen de la subtask anterior (si existe) para continuidad

        Parámetros:
            subtask_description:      Qué debe hacer el agente en esta subtask.
            task_prompt:              Instrucción original del usuario (contexto global).
            working_dir:              Directorio de trabajo del proyecto.
                                      None = usa el directorio actual del proceso.
            max_retries:              Máximo de estrategias que el agente intenta
                                      antes de rendirse. Se inyecta en el prompt.
            previous_subtask_summary: Resumen de la subtask anterior completada.
                                      Permite continuidad entre subtasks sin
                                      pasar historial completo.

        Returns:
            Message con role=SYSTEM listo para incluir en el historial.
        """
        effective_dir = working_dir or os.getcwd()
        compact_task_prompt = task_prompt.strip()[:240]
        compact_subtask_description = subtask_description.strip()[:280]

        # Construir contexto de subtask anterior si existe
        previous_context = ""
        if previous_subtask_summary:
            # Limitar a 200 chars para no inflar el prompt
            trimmed = previous_subtask_summary.strip()[:200]
            previous_context = f"- Paso anterior completó: {trimmed}"

        sections: list[str] = []

        # Sección 1: instrucciones base con el max_retries inyectado
        sections.append(
            _SYSTEM_PROMPT_BASE.format(max_retries=max_retries)
        )

        # Sección 2: herramientas disponibles
        tools_section = self._build_tools_section()
        if tools_section:
            sections.append(tools_section)

        # Sección 3: contexto de la subtask actual + resumen de anterior
        sections.append(
            _SUBTASK_CONTEXT_TEMPLATE.format(
                task_prompt=compact_task_prompt,
                subtask_description=compact_subtask_description,
                working_dir=effective_dir,
                previous_context=previous_context,
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
          - Parámetros requeridos con tipo
          - Parámetros opcionales con constraints resumidos

        NOTA SOBRE CONSTRAINTS: Es crítico que los parámetros con límites
        (como timeout con máximo de 120s) se comuniquen explícitamente.
        Sin esta información, el LLM inventa valores fuera de rango y
        entra en bucles de rechazo de argumentos. El costo en tokens de
        incluir una línea extra por parámetro es insignificante comparado
        con el costo de 5+ iteraciones desperdiciadas.

        Returns:
            String con la sección de tools formateada.
            String vacío si no hay tools registradas.
        """
        schemas = self._registry.get_schemas()

        if not schemas:
            return ""

        lines: list[str] = [_TOOLS_SECTION_HEADER]

        for schema in schemas:
            # Encabezado compacto de la tool
            lines.append(
                f"- {schema.name}: {self._summarize_tool_description(schema.description)}"
            )

            # Solo parámetros requeridos para reducir tokens
            props: dict[str, Any] = schema.parameters.get("properties", {})
            required: list[str] = schema.parameters.get("required", [])

            if required:
                req_parts = []
                for param_name in required:
                    param_info = props.get(param_name, {})
                    param_type = self._format_param_type(param_info)
                    req_parts.append(f"{param_name}({param_type})")
                lines.append(f"  Requeridos: {', '.join(req_parts)}")

            # Parámetros opcionales — incluir constraint resumido cuando existe.
            # Esto evita que el LLM invente valores fuera de rango.
            optional = [p for p in props if p not in required]
            if optional:
                opt_parts = []
                for param_name in optional:
                    param_info = props.get(param_name, {})
                    desc = param_info.get("description", "")
                    param_label = param_name
                    formatted_type = self._format_param_type(param_info)
                    if formatted_type != "any":
                        param_label = f"{param_name}({formatted_type})"
                    # Extraer constraint clave de la descripción (primera frase útil)
                    constraint_hint = self._extract_constraint_hint(desc)
                    if constraint_hint:
                        opt_parts.append(f"{param_label} [{constraint_hint}]")
                    else:
                        opt_parts.append(param_label)
                lines.append(f"  Opcionales: {', '.join(opt_parts)}")

        return "\n".join(lines)

    @staticmethod
    def _summarize_tool_description(description: str) -> str:
        """
        Resume la descripción de una tool preservando señales críticas.

        A diferencia de truncar por la primera oración, conservamos
        instrucciones útiles como enums válidos o advertencias clave.
        """
        compact = " ".join(description.split())
        if len(compact) <= 160:
            return compact

        trimmed = compact[:157].rstrip(" ,.;:")
        return f"{trimmed}..."

    @staticmethod
    def _format_param_type(param_info: dict[str, Any]) -> str:
        """
        Formatea el tipo de parámetro incluyendo enums pequeños cuando existen.
        """
        enum_values = param_info.get("enum")
        if isinstance(enum_values, list) and 0 < len(enum_values) <= 8:
            return "|".join(str(value) for value in enum_values)

        return str(param_info.get("type", "any"))

    @staticmethod
    def _extract_constraint_hint(description: str) -> str:
        """
        Extrae un hint compacto de constraints desde la descripción de un parámetro.

        Busca patrones como "RANGO VÁLIDO: 1 a 120", "Default: 30s", "PROHIBIDO"
        y los compacta en una línea corta para el system prompt.

        Parámetros:
            description: Descripción completa del parámetro del schema.

        Returns:
            String compacto con el constraint (ej: "1-120s, default 30s"),
            o string vacío si no se detectan constraints.
        """
        if not description:
            return ""

        desc_lower = description.lower()

        # Detectar rangos numéricos con "rango válido" o "range"
        if "rango válido" in desc_lower or "range" in desc_lower:
            # Buscar patrón "N a M" o "N to M"
            range_match = re.search(r"(\d+)\s+(?:a|to)\s+(\d+)", description)
            if range_match:
                lo, hi = range_match.group(1), range_match.group(2)
                # Buscar default
                default_match = re.search(r"[Dd]efault[:\s]+(\d+)", description)
                default_hint = f", default {default_match.group(1)}" if default_match else ""
                return f"{lo}-{hi}{default_hint}"

        # Detectar "máximo" o "max"
        max_match = re.search(r"(?:máximo|max|MÁXIMO|MAX)[:\s]+(\d+)", description)
        if max_match:
            return f"max {max_match.group(1)}"

        return ""

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
