"""
AnthropicAdapter — Adapter concreto para la API de Anthropic.

Implementa BaseLLMAdapter para los modelos Claude.
Se encarga únicamente de:
  1. Convertir Message[] al formato que espera la API de Anthropic.
  2. Llamar a la API y manejar sus errores específicos.
  3. Convertir la respuesta de Anthropic a RichLLMResponse.
  4. Parsear tool calls desde el JSON en el content.

TODO lo demás (retry, logging, eventos, truncado) lo hereda del base.

TOOL USE EN HIPERFORGE:
  En vez de usar el tool use nativo de Anthropic (que complica el historial
  con tipos especiales de mensajes), usamos JSON en el content.

  El prompt de sistema le enseña al LLM este formato:
    Si necesitas ejecutar una tool, responde EXACTAMENTE así:
    {
      "action": "tool_call",
      "tool": "shell",
      "arguments": {"command": "pytest tests/"}
    }

  Si el LLM quiere responder con texto normal (razonamiento):
    {
      "action": "think",
      "content": "Necesito primero instalar las dependencias..."
    }

  Si el LLM considera la subtask completada:
    {
      "action": "complete",
      "summary": "Los tests pasan exitosamente."
    }

  Este enfoque funciona igual en Anthropic, OpenAI y Ollama — cero divergencia.

MANEJO DE ERRORES DE ANTHROPIC:
  anthropic.RateLimitError    → LLMRateLimitError (con retry_after si viene en headers)
  anthropic.APIConnectionError → LLMConnectionError
  anthropic.APIStatusError    → LLMResponseError (401, 500, etc.)
  anthropic.APITimeoutError   → LLMConnectionError
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import anthropic

from hiperforge.core.logging import get_logger
from hiperforge.domain.exceptions import (
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.domain.value_objects.token_usage import TokenUsage
from hiperforge.infrastructure.llm.base import (
    BaseLLMAdapter,
    RichLLMResponse,
    ToolCallRequest,
)
from hiperforge.core.utils.ids import generate_id

logger = get_logger(__name__)

# Modelos de Anthropic soportados con sus context windows reales
_CONTEXT_WINDOWS: dict[str, int] = {
    "claude-opus-4-6":   200_000,
    "claude-sonnet-4-6": 200_000,
    "claude-haiku-4-5":  200_000,
}

# Nombre del campo JSON que indica el tipo de acción del LLM
_ACTION_FIELD = "action"
_ACTION_TOOL_CALL = "tool_call"
_ACTION_THINK = "think"
_ACTION_COMPLETE = "complete"


class AnthropicAdapter(BaseLLMAdapter):
    """
    Adapter para la API de Anthropic (modelos Claude).

    Instanciar via LLMRegistry, no directamente.

    Parámetros:
        api_key:  API key de Anthropic. Requerida.
        model_id: Modelo a usar. Default: claude-sonnet-4-6.
    """

    def __init__(self, api_key: str, model_id: str = "claude-sonnet-4-6") -> None:
        super().__init__()
        self._model_id = model_id

        # Cliente oficial de Anthropic — maneja connection pooling internamente
        self._client = anthropic.Anthropic(api_key=api_key)

        # Estado interno para reconstruir RichLLMResponse después del stream
        # Se resetea al inicio de cada llamada a _stream_impl
        self._stream_accumulated_content: str = ""
        self._stream_token_usage: TokenUsage | None = None

        logger.debug(
            "AnthropicAdapter inicializado",
            model=model_id,
        )

    # ------------------------------------------------------------------
    # Implementación del contrato LLMPort
    # ------------------------------------------------------------------

    def get_model_id(self) -> str:
        return self._model_id

    def get_provider_name(self) -> str:
        return "anthropic"

    def get_context_window_size(self) -> int:
        """Devuelve el context window real del modelo configurado."""
        return _CONTEXT_WINDOWS.get(self._model_id, 200_000)

    def is_available(self) -> bool:
        """
        Verifica disponibilidad haciendo una llamada mínima a la API.

        Usa max_tokens=1 para minimizar costo y latencia de la verificación.
        Devuelve False ante cualquier error — nunca lanza excepción.
        """
        try:
            self._client.messages.create(
                model=self._model_id,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True
        except Exception:
            return False

    # ------------------------------------------------------------------
    # Implementación de los métodos abstractos del base
    # ------------------------------------------------------------------

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
        Llamada real a la API de Anthropic messages.create().

        Convierte nuestros tipos internos al formato de Anthropic,
        llama a la API y convierte la respuesta de vuelta.
        """
        try:
            # Separamos el mensaje de sistema del historial
            # Anthropic lo espera como parámetro separado, no dentro de messages
            system_content, api_messages = self._split_system_message(messages)

            # Construimos los kwargs de la llamada
            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages,
            }

            if system_content:
                kwargs["system"] = system_content

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            # Llamada a la API
            api_response = self._client.messages.create(**kwargs)

            # Convertimos la respuesta al tipo interno
            return self._parse_api_response(api_response)

        except anthropic.RateLimitError as exc:
            # Extraemos retry_after del header si viene en la respuesta
            retry_after = self._extract_retry_after(exc)
            raise LLMRateLimitError(
                provider="anthropic",
                retry_after_seconds=retry_after,
            ) from exc

        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(
                provider="anthropic",
                reason=str(exc),
            ) from exc

        except anthropic.APITimeoutError as exc:
            raise LLMConnectionError(
                provider="anthropic",
                reason=f"timeout: {exc}",
            ) from exc

        except anthropic.APIStatusError as exc:
            raise LLMResponseError(
                provider="anthropic",
                reason=f"HTTP {exc.status_code}: {exc.message}",
                raw_response=str(exc.response),
            ) from exc

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
        Streaming real de la API de Anthropic.

        Acumula el content completo internamente para poder construir
        el RichLLMResponse en _build_response_from_stream().
        """
        # Reseteamos el estado del stream anterior
        self._stream_accumulated_content = ""
        self._stream_token_usage = None

        try:
            system_content, api_messages = self._split_system_message(messages)

            kwargs: dict[str, Any] = {
                "model": self._model_id,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": api_messages,
            }

            if system_content:
                kwargs["system"] = system_content

            if stop_sequences:
                kwargs["stop_sequences"] = stop_sequences

            # Usamos el context manager de streaming de Anthropic
            with self._client.messages.stream(**kwargs) as stream:
                for text_chunk in stream.text_stream:
                    self._stream_accumulated_content += text_chunk
                    yield text_chunk

                # Al terminar el stream, guardamos el usage final
                final_message = stream.get_final_message()
                self._stream_token_usage = TokenUsage(
                    input_tokens=final_message.usage.input_tokens,
                    output_tokens=final_message.usage.output_tokens,
                    model=self._model_id,
                )

        except anthropic.RateLimitError as exc:
            retry_after = self._extract_retry_after(exc)
            raise LLMRateLimitError(
                provider="anthropic",
                retry_after_seconds=retry_after,
            ) from exc

        except anthropic.APIConnectionError as exc:
            raise LLMConnectionError(
                provider="anthropic",
                reason=str(exc),
            ) from exc

        except anthropic.APIStatusError as exc:
            raise LLMResponseError(
                provider="anthropic",
                reason=f"HTTP {exc.status_code}: {exc.message}",
            ) from exc

    def _build_response_from_stream(self, chunks: list[str]) -> RichLLMResponse:
        """
        Construye el RichLLMResponse final después de terminar el stream.

        El content completo ya está en self._stream_accumulated_content.
        Lo parseamos en busca de tool calls y construimos la respuesta.
        """
        full_content = self._stream_accumulated_content

        # Intentamos parsear si el LLM devolvió una acción JSON
        tool_calls, parsed_content, finish_reason = self._parse_action_from_content(full_content)

        # Usamos el usage acumulado durante el stream
        # Si por algún error no lo tenemos, estimamos desde los chunks
        token_usage = self._stream_token_usage or TokenUsage(
            input_tokens=0,
            output_tokens=sum(len(c) for c in chunks) // 4,
            model=self._model_id,
        )

        return RichLLMResponse(
            content=parsed_content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            model=self._model_id,
            finish_reason=finish_reason,
        )

    def format_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        output: str,
        success: bool,
    ) -> Message:
        """
        Formatea el resultado de una tool como mensaje de usuario.

        Como usamos JSON en el content para tool calls, el resultado
        también se devuelve como mensaje de usuario con JSON estructurado.
        Esto mantiene consistencia en todo el historial.

        Formato:
          {
            "action": "tool_result",
            "tool_call_id": "...",
            "tool_name": "shell",
            "success": true,
            "output": "5 passed in 0.42s"
          }
        """
        result_payload = {
            "action": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "success": success,
            "output": output,
        }

        return Message.user(json.dumps(result_payload, ensure_ascii=False))

    def build_tools_payload(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """
        En el enfoque JSON-en-content no necesitamos el payload nativo de tools.
        Las tools se describen en el prompt de sistema via planner.py.
        Retornamos lista vacía — el base no la usa en las kwargs de la API.
        """
        return []

    # ------------------------------------------------------------------
    # Helpers privados de AnthropicAdapter
    # ------------------------------------------------------------------

    def _split_system_message(
        self,
        messages: list[Message],
    ) -> tuple[str, list[dict[str, Any]]]:
        """
        Separa el mensaje de sistema del historial y convierte al formato de Anthropic.

        Anthropic espera:
          - system: string con las instrucciones del sistema
          - messages: lista de {role: user|assistant, content: string}

        Returns:
            Tupla (system_content, api_messages).
            system_content es string vacío si no hay mensaje de sistema.
        """
        system_content = ""
        api_messages: list[dict[str, Any]] = []

        for msg in messages:
            if msg.role == Role.SYSTEM:
                # Si hay múltiples mensajes de sistema, los concatenamos
                system_content = f"{system_content}\n{msg.content}".strip()
            else:
                api_messages.append({
                    "role": msg.role.value,  # "user" o "assistant"
                    "content": msg.content,
                })

        return system_content, api_messages

    def _parse_api_response(self, api_response: Any) -> RichLLMResponse:
        """
        Convierte la respuesta de la API de Anthropic a RichLLMResponse.

        Extrae el content, parsea si hay una acción JSON, y construye
        el TokenUsage desde el usage reportado por Anthropic.
        """
        # Extraemos el texto de la respuesta
        raw_content = ""
        for block in api_response.content:
            if hasattr(block, "text"):
                raw_content += block.text

        # Parseamos si el LLM devolvió una acción JSON
        tool_calls, parsed_content, finish_reason = self._parse_action_from_content(raw_content)

        # El finish_reason de Anthropic puede ser "end_turn", "max_tokens", "stop_sequence"
        # Normalizamos a nuestro vocabulario interno
        api_finish_reason = api_response.stop_reason or "end_turn"
        if finish_reason == "stop":
            # Solo sobreescribimos si no detectamos una acción específica
            finish_reason = self._normalize_finish_reason(api_finish_reason)

        token_usage = TokenUsage(
            input_tokens=api_response.usage.input_tokens,
            output_tokens=api_response.usage.output_tokens,
            model=self._model_id,
        )

        return RichLLMResponse(
            content=parsed_content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            model=self._model_id,
            finish_reason=finish_reason,
        )

    def _parse_action_from_content(
        self,
        content: str,
    ) -> tuple[list[ToolCallRequest], str, str]:
        """
        Parsea el contenido del LLM buscando una acción JSON.

        El LLM puede responder con tres formatos:
          1. JSON con action=tool_call  → devuelve ToolCallRequest
          2. JSON con action=complete  → devuelve finish_reason="complete"
          3. JSON con action=think     → devuelve el content del pensamiento
          4. Texto libre               → lo devuelve tal cual (razonamiento libre)

        Returns:
            Tupla (tool_calls, content_limpio, finish_reason).
            tool_calls está vacío si no hay tool call.
        """
        stripped = content.strip()

        # Si no parece JSON, es texto libre — razonamiento del agente
        if not stripped.startswith("{"):
            return [], content, "stop"

        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            # JSON malformado — tratamos como texto libre
            # Loggeamos para debug pero no rompemos el flujo
            logger.warning(
                "LLM devolvió JSON malformado, tratando como texto",
                content_preview=stripped[:200],
                provider="anthropic",
                task_id=self._task_id,
            )
            return [], content, "stop"

        action = parsed.get(_ACTION_FIELD, "")

        if action == _ACTION_TOOL_CALL:
            # El LLM quiere ejecutar una tool
            tool_name = parsed.get("tool", "")
            arguments = parsed.get("arguments", {})

            if not tool_name:
                logger.warning(
                    "LLM solicitó tool_call sin nombre de tool",
                    parsed=parsed,
                    task_id=self._task_id,
                )
                return [], content, "stop"

            tool_call = ToolCallRequest(
                tool_call_id=generate_id(),
                tool_name=tool_name,
                arguments=arguments if isinstance(arguments, dict) else {},
            )

            return [tool_call], "", "tool_use"

        if action == _ACTION_COMPLETE:
            # El LLM indica que la subtask está completa
            summary = parsed.get("summary", "")
            return [], summary, "complete"

        if action == _ACTION_THINK:
            # Razonamiento explícito del agente
            think_content = parsed.get("content", "")
            return [], think_content, "stop"

        # Acción desconocida — tratamos como texto libre
        logger.warning(
            "LLM devolvió acción desconocida",
            action=action,
            task_id=self._task_id,
        )
        return [], content, "stop"

    def _normalize_finish_reason(self, anthropic_reason: str) -> str:
        """
        Normaliza el finish_reason de Anthropic a nuestro vocabulario.

        Anthropic usa: "end_turn", "max_tokens", "stop_sequence"
        Nosotros usamos: "stop", "max_tokens", "stop"
        """
        mapping = {
            "end_turn":      "stop",
            "max_tokens":    "max_tokens",
            "stop_sequence": "stop",
        }
        return mapping.get(anthropic_reason, "stop")

    def _extract_retry_after(self, exc: anthropic.RateLimitError) -> float | None:
        """
        Extrae el tiempo de espera del header Retry-After de Anthropic.

        Anthropic incluye este header en respuestas 429 para indicar
        cuántos segundos esperar antes de reintentar.

        Returns:
            Segundos a esperar, o None si el header no está presente.
        """
        try:
            # El header puede estar en la respuesta HTTP subyacente
            response = getattr(exc, "response", None)
            if response is None:
                return None

            headers = getattr(response, "headers", {})
            retry_after = headers.get("retry-after") or headers.get("Retry-After")

            if retry_after is not None:
                return float(retry_after)

        except (AttributeError, ValueError):
            pass

        return None
