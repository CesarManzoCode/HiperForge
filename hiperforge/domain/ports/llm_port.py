"""
Port: LLMPort

Define el contrato que cualquier proveedor de LLM debe cumplir
para funcionar con el agente ReAct de HiperForge.

¿Qué operaciones necesita el agente del LLM?
  El agente ReAct interactúa con el LLM de dos formas distintas:

  1. complete() — respuesta completa de una vez.
     Usado en: generación del plan de subtasks, resumen final de la task.
     El agente espera a que el LLM termine antes de continuar.

  2. stream() — respuesta en fragmentos (tokens) a medida que se generan.
     Usado en: el loop ReAct principal (razonar → actuar → observar).
     El usuario ve la respuesta aparecer progresivamente en la terminal,
     lo que da la sensación de que el agente "está pensando en vivo".

  Ambos métodos reciben la misma lista de mensajes y devuelven
  LLMResponse — la diferencia es cuándo llega la respuesta.

IMPLEMENTACIONES ESPERADAS:
  AnthropicAdapter  →  usa anthropic-sdk, modelos claude-*
  OpenAIAdapter     →  usa openai-sdk, modelos gpt-* y o1-*
  OllamaAdapter     →  usa ollama REST API, modelos locales

USO TÍPICO (desde un service):
  class PlannerService:
      def __init__(self, llm: LLMPort) -> None:
          self._llm = llm   # recibe el adapter por DI

      def generate_plan(self, messages: list[Message]) -> LLMResponse:
          # No sabe si está hablando con Claude, GPT o Llama
          return self._llm.complete(messages)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import AsyncIterator

from hiperforge.domain.value_objects.message import Message
from hiperforge.domain.value_objects.token_usage import TokenUsage


@dataclass(frozen=True)
class LLMResponse:
    """
    Respuesta completa de una llamada al LLM.

    Contiene tanto el contenido de la respuesta como los metadatos
    de uso (tokens, modelo) necesarios para tracking de costos.

    Atributos:
        content:     Texto completo de la respuesta del LLM.
        token_usage: Tokens consumidos en esta llamada.
        model:       Nombre exacto del modelo que respondió.
                     Puede diferir del solicitado (ej: el proveedor
                     redirige a una versión específica del modelo).
        finish_reason: Por qué el LLM dejó de generar tokens.
                       "stop"        → terminó naturalmente.
                       "length"      → alcanzó max_tokens.
                       "tool_use"    → quiere ejecutar una tool (ReAct).
                       "content_filter" → bloqueado por filtro de contenido.
    """

    content: str
    token_usage: TokenUsage
    model: str
    finish_reason: str = "stop"

    @property
    def was_truncated(self) -> bool:
        """
        True si la respuesta fue cortada por límite de tokens.

        Una respuesta truncada puede contener un plan o razonamiento
        incompleto — el agente debe detectar esto y manejar el error.
        """
        return self.finish_reason == "length"

    @property
    def wants_tool_use(self) -> bool:
        """
        True si el LLM indica que quiere ejecutar una tool.

        En el loop ReAct, esta señal le dice al executor que debe
        parsear el contenido para extraer el tool call solicitado.
        """
        return self.finish_reason == "tool_use"

    def __str__(self) -> str:
        """
        Ejemplo: LLMResponse(claude-sonnet-4-6, 350 tokens, stop)
                 preview: "Voy a instalar las dependencias necesarias..."
        """
        preview = self.content[:80] + "..." if len(self.content) > 80 else self.content
        return (
            f"LLMResponse({self.model}, {self.token_usage.total_tokens} tokens,"
            f" {self.finish_reason})\n  preview: {preview!r}"
        )


class LLMPort(ABC):
    """
    Contrato abstracto para todos los proveedores de LLM.

    El agente ReAct usa este port para:
      - Generar el plan inicial de subtasks (complete)
      - Ejecutar el loop Razonar → Actuar en cada subtask (stream)
      - Generar el resumen final de la task (complete)
    """

    # ------------------------------------------------------------------
    # Métodos principales del contrato
    # ------------------------------------------------------------------

    @abstractmethod
    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        stop_sequences: list[str] | None = None,
    ) -> LLMResponse:
        """
        Envía mensajes al LLM y espera la respuesta completa.

        Usado cuando necesitamos la respuesta entera antes de continuar,
        como al generar el plan de subtasks (necesitamos todas las subtasks
        antes de empezar a ejecutar ninguna).

        Parámetros:
            messages:        Historial de la conversación.
            max_tokens:      Límite de tokens en la respuesta. Default 4096.
            temperature:     Creatividad del modelo (0.0 = determinista,
                             1.0 = muy creativo). Default 0.2 para el agente
                             porque queremos respuestas consistentes y precisas.
            stop_sequences:  Strings que detienen la generación si aparecen.
                             Útil para delimitar respuestas estructuradas.

        Returns:
            LLMResponse con el contenido completo y metadatos de uso.

        Raises:
            LLMConnectionError:  Si no se puede conectar al proveedor.
            LLMRateLimitError:   Si se alcanzó el límite de requests.
            LLMResponseError:    Si la respuesta tiene formato inesperado.
        """
        ...

    @abstractmethod
    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = 4096,
        temperature: float = 0.2,
        stop_sequences: list[str] | None = None,
    ) -> AsyncIterator[str]:
        """
        Envía mensajes al LLM y devuelve los tokens a medida que se generan.

        Usado en el loop ReAct principal para mostrar el razonamiento
        del agente en tiempo real en la terminal — el usuario ve al
        agente "pensar" sin esperar a que termine.

        Parámetros:
            Idénticos a complete().

        Yields:
            Fragmentos de texto (tokens o chunks) a medida que el LLM
            los genera. El caller acumula los chunks para reconstruir
            la respuesta completa.

        Raises:
            LLMConnectionError:  Si la conexión se interrumpe durante el stream.
            LLMRateLimitError:   Si se alcanza el límite durante el stream.
            LLMResponseError:    Si el stream devuelve datos inesperados.

        Ejemplo de uso:
            chunks = []
            async for chunk in llm.stream(messages):
                print(chunk, end="", flush=True)  # imprime en tiempo real
                chunks.append(chunk)
            full_response = "".join(chunks)
        """
        ...

    # ------------------------------------------------------------------
    # Métodos de información del proveedor
    # ------------------------------------------------------------------

    @abstractmethod
    def get_model_id(self) -> str:
        """
        Devuelve el ID del modelo configurado en este adapter.

        Ejemplo: "claude-sonnet-4-6", "gpt-4o", "llama3"

        Usado por TokenUsage para calcular el costo estimado
        y por los logs para identificar qué modelo se usó.
        """
        ...

    @abstractmethod
    def get_provider_name(self) -> str:
        """
        Devuelve el nombre del proveedor.

        Ejemplo: "anthropic", "openai", "ollama"

        Usado en los errores LLMError para identificar qué
        adapter falló sin necesidad de inspeccionar el tipo.
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """
        Verifica si el proveedor está disponible y la API key es válida.

        Hace una llamada mínima de verificación al proveedor.
        Usado al iniciar HiperForge para detectar problemas de configuración
        antes de que el usuario intente ejecutar una task.

        Returns:
            True si el proveedor responde correctamente.
            False si hay problemas de conexión o autenticación.

        Nota: No lanza excepciones — devuelve False en cualquier error.
        Esto permite al caller mostrar un mensaje amigable en vez de
        propagar una excepción en el arranque.
        """
        ...

    # ------------------------------------------------------------------
    # Método con implementación por defecto — los adapters pueden sobreescribir
    # ------------------------------------------------------------------

    def get_context_window_size(self) -> int:
        """
        Devuelve el tamaño máximo del context window en tokens.

        Los adapters sobreescriben este método con el valor real del modelo.
        Default conservador de 8192 para modelos desconocidos.

        Usado por context_builder.py para truncar el historial de mensajes
        cuando se acerca al límite del modelo.
        """
        return 8_192