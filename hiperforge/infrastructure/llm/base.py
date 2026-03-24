"""
BaseLLMAdapter — Adapter base para todos los proveedores de LLM.

Todo proveedor (Anthropic, OpenAI, Ollama) hereda de esta clase.
El base se encarga de todo lo transversal — retry, logging, eventos,
token tracking, truncado de contexto — para que cada adapter concreto
solo implemente lo específico de su API.

ARQUITECTURA DEL ADAPTER:

  BaseLLMAdapter (esta clase)
  ├── complete()         ← punto de entrada público, nunca sobreescribir
  │   ├── _truncate_messages_if_needed()
  │   ├── _emit_request_event()
  │   ├── _complete_impl()   ← SOBREESCRIBIR en cada adapter
  │   ├── _emit_response_event()
  │   └── _log_call()
  │
  ├── stream()           ← punto de entrada público, nunca sobreescribir
  │   ├── _truncate_messages_if_needed()
  │   ├── _emit_request_event()
  │   ├── _stream_impl()     ← SOBREESCRIBIR en cada adapter
  │   └── (chunks llegan via callback al caller)
  │
  └── format_tool_result()  ← SOBREESCRIBIR si el proveedor usa formato distinto

TOOL USE HÍBRIDO:
  Anthropic/OpenAI → tool use nativo (JSON estructurado en la respuesta)
  Ollama           → prompt engineering (XML en el content, parseado por el adapter)

  El executor siempre recibe LLMResponse con tool_calls: list[ToolCallRequest]
  sin importar cómo llegó por debajo. La abstracción es perfecta.

LLMResponse EXTENDIDO:
  Comparado con el LLMResponse del port, este módulo define
  ToolCallRequest — el objeto que representa una tool solicitada
  por el LLM en modo nativo o parseada desde texto.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from dataclasses import dataclass, field
from typing import Any, AsyncIterator

from hiperforge.core.constants import (
    LLM_CONTEXT_RESPONSE_RESERVE,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_TEMPERATURE,
)
from hiperforge.core.events import AgentEvent, EventType, get_event_bus
from hiperforge.core.logging import get_logger
from hiperforge.core.utils.retry import retry_call
from hiperforge.domain.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from hiperforge.domain.ports.llm_port import LLMPort, LLMResponse
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.domain.value_objects.token_usage import TokenUsage

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# ToolCallRequest — extensión de LLMResponse para tool use nativo
#
# Cuando el LLM decide usar una tool, la respuesta incluye uno o más
# ToolCallRequest. El executor los lee, los despacha al ToolDispatcher
# y devuelve los resultados al LLM en la siguiente iteración.
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ToolCallRequest:
    """
    Solicitud de ejecución de tool generada por el LLM.

    En tool use nativo (Anthropic/OpenAI), el LLM devuelve esto
    directamente como JSON estructurado — sin parsing frágil.

    En modo prompt engineering (Ollama), el adapter parsea el XML/JSON
    del content y construye este objeto. El executor nunca ve la diferencia.

    Atributos:
        tool_call_id: ID asignado por el proveedor para correlacionar
                      el result con esta request. En Ollama lo generamos
                      nosotros con generate_id().
        tool_name:    Nombre de la tool a ejecutar. Debe coincidir
                      exactamente con ToolPort.name del registry.
        arguments:    Argumentos parseados como dict. Ya validados
                      contra el JSON Schema del tool antes de llegar aquí.
    """

    tool_call_id: str
    tool_name: str
    arguments: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardarlo en el historial de la sesión."""
        return {
            "tool_call_id": self.tool_call_id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
        }

    def __str__(self) -> str:
        args_preview = str(self.arguments)[:80]
        return f"ToolCallRequest({self.tool_name}, args={args_preview})"


@dataclass(frozen=True)
class RichLLMResponse:
    """
    Respuesta extendida del LLM que soporta tool use nativo.

    Extiende el LLMResponse del domain/ports con tool_calls —
    el campo que hace posible el loop ReAct con tool use nativo.

    Atributos:
        content:      Texto de la respuesta. Puede estar vacío si
                      el LLM respondió solo con tool calls.
        tool_calls:   Tools que el LLM solicita ejecutar.
                      Lista vacía = el LLM respondió con texto puro.
        token_usage:  Tokens consumidos en esta llamada.
        model:        Modelo exacto que respondió.
        finish_reason: Por qué terminó la generación.
                       "end_turn"    → terminó naturalmente (Anthropic)
                       "stop"        → terminó naturalmente (OpenAI)
                       "tool_use"    → quiere ejecutar tools
                       "max_tokens"  → alcanzó el límite
    """

    content: str
    tool_calls: list[ToolCallRequest]
    token_usage: TokenUsage
    model: str
    finish_reason: str = "stop"

    @property
    def has_tool_calls(self) -> bool:
        """True si el LLM solicita ejecutar al menos una tool."""
        return len(self.tool_calls) > 0

    @property
    def has_content(self) -> bool:
        """True si el LLM generó texto además de (o en vez de) tool calls."""
        return bool(self.content.strip())

    @property
    def was_truncated(self) -> bool:
        """True si la respuesta fue cortada por límite de tokens."""
        return self.finish_reason in ("max_tokens", "length")

    def to_llm_response(self) -> LLMResponse:
        """
        Convierte a LLMResponse del domain port para compatibilidad.

        Útil cuando el caller solo necesita el texto y no le importan
        los tool calls (ej: generación del plan, resumen final).
        """
        return LLMResponse(
            content=self.content,
            token_usage=self.token_usage,
            model=self.model,
            finish_reason=self.finish_reason,
        )

    def __str__(self) -> str:
        tools_label = f", {len(self.tool_calls)} tool calls" if self.has_tool_calls else ""
        preview = self.content[:60] + "..." if len(self.content) > 60 else self.content
        return (
            f"RichLLMResponse({self.model}{tools_label}, "
            f"{self.token_usage.total_tokens} tokens) "
            f"preview={preview!r}"
        )


# ---------------------------------------------------------------------------
# BaseLLMAdapter
# ---------------------------------------------------------------------------

class BaseLLMAdapter(LLMPort):
    """
    Clase base para todos los adapters de LLM.

    Implementa el patrón Template Method:
      - Los métodos públicos (complete, stream) definen el algoritmo completo.
      - Los métodos _impl (abstractos) son los que cada adapter concreto implementa.
      - Todo lo transversal (retry, logging, eventos, truncado) vive aquí.

    LOS ADAPTERS CONCRETOS DEBEN IMPLEMENTAR:
      - _complete_impl()        → llamada real a la API del proveedor
      - _stream_impl()          → streaming real de la API del proveedor
      - get_model_id()          → ID del modelo configurado
      - get_provider_name()     → nombre del proveedor
      - is_available()          → verificación de salud
      - get_context_window_size() → tamaño del context window
      - format_tool_result()    → cómo formatear el resultado de una tool
                                  para devolverlo al LLM en la siguiente iteración
      - build_tools_payload()   → cómo serializar ToolSchemas para la API del proveedor
    """

    def __init__(self) -> None:
        # Contexto opcional para enriquecer eventos y logs
        # El executor lo setea con task_id y subtask_id al iniciar el loop
        self._task_id: str | None = None
        self._subtask_id: str | None = None

    # ------------------------------------------------------------------
    # API pública — NUNCA sobreescribir en adapters concretos
    # ------------------------------------------------------------------

    def complete(
        self,
        messages: list[Message],
        *,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        stop_sequences: list[str] | None = None,
        tools: list[ToolSchema] | None = None,
    ) -> RichLLMResponse:
        """
        Envía mensajes al LLM y devuelve la respuesta completa.

        Wrapper completo con: truncado de contexto → eventos → retry →
        llamada real → logging → eventos de respuesta.

        Parámetros:
            messages:        Historial de la conversación.
            max_tokens:      Límite de tokens en la respuesta.
            temperature:     Temperatura de generación.
            stop_sequences:  Strings que detienen la generación.
            tools:           Schemas de las tools disponibles para el LLM.
                             Si None, el LLM no puede solicitar tools.

        Returns:
            RichLLMResponse con content, tool_calls y métricas.
        """
        # Paso 1: truncar el historial si se acerca al límite del context window
        truncated_messages = self._truncate_messages_if_needed(
            messages=messages,
            max_tokens=max_tokens,
        )

        # Paso 2: emitir evento de request al bus (CLI y logger reaccionan)
        self._emit_request_event(truncated_messages)

        start_time = time.monotonic()

        # Paso 3: llamada real con retry automático en errores transitorios
        response = retry_call(
            fn=lambda: self._complete_impl(
                messages=truncated_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                tools=tools,
            ),
            max_attempts=3,
            retryable_exceptions=(LLMRateLimitError, LLMConnectionError),
            on_retry=self._on_retry,
        )

        duration = time.monotonic() - start_time

        # Paso 4: logging estructurado de la llamada
        self._log_call(response=response, duration=duration)

        # Paso 5: emitir evento de respuesta al bus
        self._emit_response_event(response=response, duration=duration)

        return response

    async def stream(
        self,
        messages: list[Message],
        *,
        max_tokens: int = LLM_DEFAULT_MAX_TOKENS,
        temperature: float = LLM_DEFAULT_TEMPERATURE,
        stop_sequences: list[str] | None = None,
        tools: list[ToolSchema] | None = None,
        on_chunk: callable | None = None,
    ) -> RichLLMResponse:
        """
        Envía mensajes al LLM y procesa la respuesta en streaming.

        Los chunks llegan via on_chunk callback para que la CLI pueda
        mostrarlos en tiempo real mientras el adapter los acumula.

        Parámetros:
            messages:    Historial de la conversación.
            on_chunk:    Callback opcional llamado por cada chunk de texto.
                         Signature: on_chunk(chunk: str) -> None
                         DEBE ser rápido — no hacer I/O dentro.
            (resto igual que complete())

        Returns:
            RichLLMResponse completo al terminar el stream.
            El content es la concatenación de todos los chunks.
        """
        truncated_messages = self._truncate_messages_if_needed(
            messages=messages,
            max_tokens=max_tokens,
        )

        self._emit_request_event(truncated_messages)

        start_time = time.monotonic()

        # Acumulamos chunks y los enviamos al callback y al EventBus
        chunks: list[str] = []

        async for chunk in self._stream_impl(
            messages=truncated_messages,
            max_tokens=max_tokens,
            temperature=temperature,
            stop_sequences=stop_sequences,
            tools=tools,
        ):
            chunks.append(chunk)

            # Callback para la CLI (actualiza buffer en RAM)
            if on_chunk is not None:
                on_chunk(chunk)

            # Evento de chunk al bus — los listeners deben ser ultra rápidos
            get_event_bus().emit(
                AgentEvent.llm_streaming_chunk(
                    task_id=self._task_id or "",
                    subtask_id=self._subtask_id or "",
                    chunk=chunk,
                )
            )

        # Construimos la respuesta completa a partir de los chunks acumulados
        response = self._build_response_from_stream(chunks)

        duration = time.monotonic() - start_time

        self._log_call(response=response, duration=duration)
        self._emit_response_event(response=response, duration=duration)

        return response

    # ------------------------------------------------------------------
    # Métodos que los adapters concretos DEBEN implementar
    # ------------------------------------------------------------------

    @abstractmethod
    def _complete_impl(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        tools: list[ToolSchema] | None,
    ) -> RichLLMResponse:
        """
        Implementación real de la llamada al proveedor.

        Aquí va la lógica específica de cada API:
          - AnthropicAdapter: llama a self._client.messages.create()
          - OpenAIAdapter:    llama a self._client.chat.completions.create()
          - OllamaAdapter:    llama a requests.post(self._base_url + "/api/chat")

        DEBE:
          - Convertir los Message a el formato del proveedor.
          - Convertir los ToolSchema al formato del proveedor.
          - Parsear la respuesta y construir un RichLLMResponse.
          - Lanzar LLMRateLimitError, LLMConnectionError o LLMResponseError
            según el tipo de error — nunca dejar pasar excepciones del SDK.

        NO DEBE:
          - Hacer retry (lo hace el base).
          - Loggear (lo hace el base).
          - Emitir eventos (lo hace el base).
        """
        ...

    @abstractmethod
    async def _stream_impl(
        self,
        messages: list[Message],
        *,
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        tools: list[ToolSchema] | None,
    ) -> AsyncIterator[str]:
        """
        Implementación real del streaming del proveedor.

        Yields fragmentos de texto (chunks) a medida que el LLM los genera.

        DEBE:
          - Conectarse al endpoint de streaming del proveedor.
          - Yielding cada chunk de texto recibido.
          - Al terminar el stream, el base construye RichLLMResponse
            a partir de los chunks acumulados via _build_response_from_stream().

        NOTA SOBRE TOOL CALLS EN STREAMING:
          En streaming, los tool calls llegan fragmentados en múltiples chunks.
          El adapter debe acumularlos internamente y yielding solo los chunks
          de texto. Los tool calls se incluyen en _build_response_from_stream().
        """
        ...

    @abstractmethod
    def _build_response_from_stream(self, chunks: list[str]) -> RichLLMResponse:
        """
        Construye el RichLLMResponse final a partir de los chunks acumulados.

        Llamado por stream() después de que termina el stream.
        El adapter tiene acceso al estado interno (tool calls acumulados,
        token usage del stream) para construir la respuesta completa.
        """
        ...

    @abstractmethod
    def format_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        output: str,
        success: bool,
    ) -> Message:
        """
        Formatea el resultado de una tool como Message para el LLM.

        Cada proveedor usa un formato distinto:
          Anthropic → Message con role='user' y content type='tool_result'
          OpenAI    → Message con role='tool' y tool_call_id
          Ollama    → Message con role='user' formateado en texto

        El executor llama siempre este método — nunca construye
        el mensaje de resultado manualmente.

        Parámetros:
            tool_call_id: ID de la ToolCallRequest original.
            tool_name:    Nombre de la tool ejecutada.
            output:       Output de la tool (stdout, contenido, etc.).
            success:      Si la tool terminó exitosamente.

        Returns:
            Message listo para agregar al historial y enviarlo al LLM.
        """
        ...

    @abstractmethod
    def build_tools_payload(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """
        Serializa los ToolSchemas al formato que espera la API del proveedor.

        Anthropic usa:  [{"name": ..., "description": ..., "input_schema": ...}]
        OpenAI usa:     [{"type": "function", "function": {"name": ..., "parameters": ...}}]
        Ollama usa:     No tiene formato nativo → retorna lista vacía (usa prompt engineering)

        El base llama este método en _complete_impl y _stream_impl antes
        de hacer la request al proveedor.
        """
        ...

    # ------------------------------------------------------------------
    # Context binding — el executor setea el contexto al inicio de cada subtask
    # ------------------------------------------------------------------

    def bind_context(self, task_id: str, subtask_id: str | None = None) -> None:
        """
        Vincula el adapter al contexto de una task/subtask específica.

        El executor llama esto al inicio de cada subtask para que todos
        los eventos y logs emitidos por el adapter lleven el task_id
        y subtask_id correctos automáticamente.

        Parámetros:
            task_id:    ID de la task que se está ejecutando.
            subtask_id: ID de la subtask activa. None durante el planning.
        """
        self._task_id = task_id
        self._subtask_id = subtask_id

        logger.debug(
            "contexto del adapter actualizado",
            task_id=task_id,
            subtask_id=subtask_id,
            provider=self.get_provider_name(),
            model=self.get_model_id(),
        )

    # ------------------------------------------------------------------
    # Métodos internos del base — lógica transversal compartida
    # ------------------------------------------------------------------

    def _truncate_messages_if_needed(
        self,
        messages: list[Message],
        max_tokens: int,
    ) -> list[Message]:
        """
        Trunca el historial de mensajes si se acerca al límite del context window.

        ESTRATEGIA DE TRUNCADO:
          1. El mensaje de sistema (role=system) SIEMPRE se preserva.
             Contiene las instrucciones del agente — perderlo es catastrófico.
          2. Los mensajes más recientes tienen prioridad sobre los antiguos.
             El LLM necesita el contexto inmediato más que el histórico lejano.
          3. Se elimina de a un mensaje desde el más antiguo hasta que
             el historial cabe en el context window disponible.

        ESTIMACIÓN DE TOKENS:
          Usamos la heurística de 4 caracteres ≈ 1 token, que es suficientemente
          precisa para decidir cuándo truncar. Si necesitamos precisión exacta,
          el adapter concreto puede sobreescribir este método con el tokenizer
          real del proveedor.

        Parámetros:
            messages:   Historial completo de mensajes.
            max_tokens: Tokens reservados para la respuesta del LLM.

        Returns:
            Lista de mensajes truncada si era necesario, original si no.
        """
        context_window = self.get_context_window_size()

        # Tokens disponibles para el historial de mensajes
        # Reservamos espacio para la respuesta del LLM
        available_tokens = context_window - max_tokens - LLM_CONTEXT_RESPONSE_RESERVE

        if available_tokens <= 0:
            # max_tokens demasiado grande para este modelo — devolvemos solo system
            logger.warning(
                "max_tokens excede el context window disponible",
                context_window=context_window,
                max_tokens=max_tokens,
                provider=self.get_provider_name(),
            )
            # Preservamos al menos el mensaje de sistema
            return [m for m in messages if m.role == Role.SYSTEM]

        # Estimación rápida: 1 token ≈ 4 caracteres
        def estimate_tokens(msgs: list[Message]) -> int:
            total_chars = sum(len(m.content) for m in msgs)
            return total_chars // 4

        # Si cabe completo, no truncamos nada
        if estimate_tokens(messages) <= available_tokens:
            return messages

        # Separamos el sistema (siempre se preserva) del resto
        system_messages = [m for m in messages if m.role == Role.SYSTEM]
        non_system = [m for m in messages if m.role != Role.SYSTEM]

        # Eliminamos mensajes antiguos (del inicio) hasta que quepa
        # Siempre en pares (user + assistant) para mantener coherencia
        while non_system and estimate_tokens(system_messages + non_system) > available_tokens:
            # Eliminamos el mensaje más antiguo del historial no-sistema
            non_system.pop(0)

        truncated = system_messages + non_system

        logger.info(
            "historial truncado por límite de contexto",
            original_count=len(messages),
            truncated_count=len(truncated),
            provider=self.get_provider_name(),
            task_id=self._task_id,
        )

        return truncated

    def _emit_request_event(self, messages: list[Message]) -> None:
        """Emite LLM_REQUEST_SENT al EventBus antes de llamar al proveedor."""
        get_event_bus().emit(
            AgentEvent.llm_request_sent(
                task_id=self._task_id or "",
                subtask_id=self._subtask_id or "",
                provider=self.get_provider_name(),
                model=self.get_model_id(),
                message_count=len(messages),
            )
        )

    def _emit_response_event(self, response: RichLLMResponse, duration: float) -> None:
        """Emite LLM_RESPONSE_RECEIVED al EventBus después de recibir la respuesta."""
        get_event_bus().emit(
            AgentEvent.llm_response_received(
                task_id=self._task_id or "",
                subtask_id=self._subtask_id or "",
                provider=self.get_provider_name(),
                model=response.model,
                input_tokens=response.token_usage.input_tokens,
                output_tokens=response.token_usage.output_tokens,
                finish_reason=response.finish_reason,
                duration_seconds=round(duration, 3),
            )
        )

    def _log_call(self, response: RichLLMResponse, duration: float) -> None:
        """Logging estructurado de cada llamada al LLM."""
        logger.info(
            "llamada al LLM completada",
            provider=self.get_provider_name(),
            model=response.model,
            input_tokens=response.token_usage.input_tokens,
            output_tokens=response.token_usage.output_tokens,
            estimated_cost_usd=response.token_usage.estimated_cost_usd,
            duration_seconds=round(duration, 3),
            finish_reason=response.finish_reason,
            has_tool_calls=response.has_tool_calls,
            tool_calls_count=len(response.tool_calls),
            task_id=self._task_id,
            subtask_id=self._subtask_id,
        )

    def _on_retry(self, attempt: int, error: Exception, wait_seconds: float) -> None:
        """
        Callback llamado por retry_call antes de cada reintento.

        Emite evento al bus para que la CLI pueda mostrar
        "Rate limit alcanzado, reintentando en 8.3s..." en tiempo real.
        """
        logger.warning(
            "reintentando llamada al LLM",
            attempt=attempt,
            error_type=type(error).__name__,
            error_message=str(error),
            wait_seconds=round(wait_seconds, 1),
            provider=self.get_provider_name(),
            task_id=self._task_id,
        )

        get_event_bus().emit(
            AgentEvent.retry_scheduled(
                task_id=self._task_id or "",
                subtask_id=self._subtask_id or "",
                attempt=attempt,
                max_attempts=3,
                wait_seconds=round(wait_seconds, 1),
                reason=str(error),
            )
        )