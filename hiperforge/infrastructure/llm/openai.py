"""
OpenAIAdapter — Adapter concreto para la API de OpenAI.

Implementa BaseLLMAdapter para los modelos GPT y o1.
La diferencia principal con AnthropicAdapter:

  - OpenAI acepta role='system' dentro del array de messages
    (Anthropic lo espera como parámetro separado)
  - El finish_reason de OpenAI usa "stop" en vez de "end_turn"
  - Los modelos o1 no soportan temperature ni system messages
    (tienen su propio sistema de razonamiento interno)
  - El streaming usa un generador distinto al de Anthropic

Todo lo demás — parseo de JSON actions, retry, logging, eventos —
es idéntico porque hereda del base. Esa es exactamente la ventaja
de tener BaseLLMAdapter.

MODELOS SOPORTADOS:
  GPT:  gpt-4o, gpt-4o-mini, gpt-4-turbo
  O1:   o1, o1-mini (sin temperature, sin system message)

GROQ:
  GroqAdapter hereda de OpenAIAdapter porque Groq usa una API
  100% compatible con OpenAI. Solo cambia base_url y los modelos.
  Ver groq.py para la implementación.
"""

from __future__ import annotations

import json
from typing import Any, AsyncIterator

import openai

from hiperforge.core.logging import get_logger
from hiperforge.core.utils.ids import generate_id
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

logger = get_logger(__name__)

# Modelos y sus context windows reales
_CONTEXT_WINDOWS: dict[str, int] = {
    "gpt-4o":        128_000,
    "gpt-4o-mini":   128_000,
    "gpt-4-turbo":   128_000,
    "o1":            200_000,
    "o1-mini":       128_000,
}

# Modelos o1 — tienen restricciones especiales:
# no soportan temperature ni streaming ni system messages
_O1_MODELS: frozenset[str] = frozenset({"o1", "o1-mini"})

# Campos JSON de acciones — idénticos a Anthropic para consistencia
_ACTION_FIELD    = "action"
_ACTION_TOOL_CALL = "tool_call"
_ACTION_THINK    = "think"
_ACTION_COMPLETE = "complete"


class OpenAIAdapter(BaseLLMAdapter):
    """
    Adapter para la API de OpenAI (modelos GPT y o1).

    También sirve como base para GroqAdapter — ver groq.py.

    Parámetros:
        api_key:  API key de OpenAI. Requerida.
        model_id: Modelo a usar. Default: gpt-4o.
        base_url: URL base de la API. Default: API oficial de OpenAI.
                  GroqAdapter sobreescribe esto con la URL de Groq.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = "gpt-4o",
        base_url: str | None = None,
    ) -> None:
        super().__init__()
        self._model_id = model_id

        # base_url permite que GroqAdapter reutilice todo este adapter
        # apuntando a la API de Groq que es compatible con OpenAI
        self._client = openai.OpenAI(
            api_key=api_key,
            base_url=base_url,  # None = URL oficial de OpenAI
        )

        # Estado interno para reconstruir respuesta después del stream
        self._stream_accumulated_content: str = ""
        self._stream_input_tokens: int = 0
        self._stream_output_tokens: int = 0

        logger.debug(
            "OpenAIAdapter inicializado",
            model=model_id,
            base_url=base_url or "https://api.openai.com",
        )

    # ------------------------------------------------------------------
    # Implementación del contrato LLMPort
    # ------------------------------------------------------------------

    def get_model_id(self) -> str:
        return self._model_id

    def get_provider_name(self) -> str:
        return "openai"

    def get_context_window_size(self) -> int:
        return _CONTEXT_WINDOWS.get(self._model_id, 128_000)

    def is_available(self) -> bool:
        """Verificación mínima de disponibilidad."""
        try:
            kwargs = self._build_kwargs(
                api_messages=[{"role": "user", "content": "ping"}],
                max_tokens=1,
                temperature=0.0,
                stop_sequences=None,
            )
            self._client.chat.completions.create(**kwargs)
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
        Llamada real a la API de OpenAI chat.completions.create().

        Maneja las restricciones especiales de los modelos o1:
          - No soportan temperature (lo ignoramos silenciosamente)
          - No soportan system messages (los convertimos a user messages)
          - No soportan streaming (el base ya maneja esto via complete())
        """
        try:
            api_messages = self._convert_messages(messages)
            kwargs = self._build_kwargs(
                api_messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
            )

            api_response = self._client.chat.completions.create(**kwargs)
            return self._parse_api_response(api_response)

        except openai.RateLimitError as exc:
            retry_after = self._extract_retry_after(exc)
            raise LLMRateLimitError(
                provider=self.get_provider_name(),
                retry_after_seconds=retry_after,
            ) from exc

        except openai.APIConnectionError as exc:
            raise LLMConnectionError(
                provider=self.get_provider_name(),
                reason=str(exc),
            ) from exc

        except openai.APITimeoutError as exc:
            raise LLMConnectionError(
                provider=self.get_provider_name(),
                reason=f"timeout: {exc}",
            ) from exc

        except openai.APIStatusError as exc:
            raise LLMResponseError(
                provider=self.get_provider_name(),
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
        Streaming real de la API de OpenAI.

        Los modelos o1 no soportan streaming — para ellos hacemos
        una llamada completa y simulamos el stream yielding el
        content completo de una vez. El caller no nota la diferencia.
        """
        # Reseteamos estado del stream anterior
        self._stream_accumulated_content = ""
        self._stream_input_tokens = 0
        self._stream_output_tokens = 0

        # Los modelos o1 no soportan streaming
        if self._is_o1_model():
            response = self._complete_impl(
                messages=messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                tools=tools,
            )
            self._stream_accumulated_content = response.content
            self._stream_input_tokens = response.token_usage.input_tokens
            self._stream_output_tokens = response.token_usage.output_tokens
            yield response.content
            return

        try:
            api_messages = self._convert_messages(messages)
            kwargs = self._build_kwargs(
                api_messages=api_messages,
                max_tokens=max_tokens,
                temperature=temperature,
                stop_sequences=stop_sequences,
                stream=True,
            )

            # stream=True activa el streaming en OpenAI
            stream = self._client.chat.completions.create(**kwargs)

            for chunk in stream:
                # Cada chunk puede tener o no tener content
                delta = chunk.choices[0].delta if chunk.choices else None
                if delta and delta.content:
                    self._stream_accumulated_content += delta.content
                    yield delta.content

                # El último chunk trae el usage (con stream_options)
                if chunk.usage:
                    self._stream_input_tokens = chunk.usage.prompt_tokens
                    self._stream_output_tokens = chunk.usage.completion_tokens

        except openai.RateLimitError as exc:
            retry_after = self._extract_retry_after(exc)
            raise LLMRateLimitError(
                provider=self.get_provider_name(),
                retry_after_seconds=retry_after,
            ) from exc

        except openai.APIConnectionError as exc:
            raise LLMConnectionError(
                provider=self.get_provider_name(),
                reason=str(exc),
            ) from exc

        except openai.APIStatusError as exc:
            raise LLMResponseError(
                provider=self.get_provider_name(),
                reason=f"HTTP {exc.status_code}: {exc.message}",
            ) from exc

    def _build_response_from_stream(self, chunks: list[str]) -> RichLLMResponse:
        """Construye RichLLMResponse al terminar el stream."""
        full_content = self._stream_accumulated_content
        tool_calls, parsed_content, finish_reason, deferred_summary = (
            self._parse_action_from_content(full_content)
        )

        token_usage = TokenUsage(
            input_tokens=self._stream_input_tokens,
            output_tokens=self._stream_output_tokens,
            model=self._model_id,
        )

        return RichLLMResponse(
            content=parsed_content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            model=self._model_id,
            finish_reason=finish_reason,
            deferred_completion_summary=deferred_summary,
        )

    def format_tool_result(
        self,
        tool_call_id: str,
        tool_name: str,
        output: str,
        success: bool,
    ) -> Message:
        """
        Idéntico a AnthropicAdapter — JSON como mensaje de usuario.
        Consistencia total en el historial entre providers.
        """
        result_payload = {
            "action": "tool_result",
            "tool_call_id": tool_call_id,
            "tool_name": tool_name,
            "success": success,
            "output": self._compress_tool_output_for_llm(output),
        }
        return Message.user(json.dumps(result_payload, ensure_ascii=False))

    def build_tools_payload(self, tools: list[ToolSchema]) -> list[dict[str, Any]]:
        """
        Igual que Anthropic — no usamos tool use nativo, usamos JSON en content.
        """
        return []

    # ------------------------------------------------------------------
    # Helpers privados de OpenAIAdapter
    # ------------------------------------------------------------------

    def _convert_messages(self, messages: list[Message]) -> list[dict[str, Any]]:
        """
        Convierte Message[] al formato de OpenAI.

        A diferencia de Anthropic, OpenAI acepta role='system' dentro
        del array de messages — no necesitamos separarlo.

        Excepción: los modelos o1 no soportan system messages.
        Los convertimos a user messages con un prefijo claro.
        """
        api_messages = []

        for msg in messages:
            if msg.role == Role.SYSTEM and self._is_o1_model():
                # o1 no soporta system — convertimos a user con marcador
                api_messages.append({
                    "role": "user",
                    "content": f"[Instrucciones del sistema]\n{msg.content}",
                })
            else:
                api_messages.append({
                    "role": msg.role.value,
                    "content": msg.content,
                })

        return api_messages

    def _build_kwargs(
        self,
        *,
        api_messages: list[dict[str, Any]],
        max_tokens: int,
        temperature: float,
        stop_sequences: list[str] | None,
        stream: bool = False,
    ) -> dict[str, Any]:
        """
        Construye los kwargs para la llamada a la API.

        Maneja las restricciones de o1:
          - No acepta temperature (la omitimos)
          - Usa max_completion_tokens en vez de max_tokens
          - No acepta stream=True
        """
        uses_max_completion_tokens = self._uses_max_completion_tokens()

        kwargs: dict[str, Any] = {
            "model": self._model_id,
            "messages": api_messages,
        }

        # Algunas familias nuevas de OpenAI usan max_completion_tokens.
        if uses_max_completion_tokens:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens
            kwargs["temperature"] = temperature

        if stop_sequences and self._supports_stop_sequences():
            kwargs["stop"] = stop_sequences

        if stream and not self._is_o1_model():
            kwargs["stream"] = True
            # stream_options permite recibir usage en el último chunk
            kwargs["stream_options"] = {"include_usage": True}

        return kwargs

    def _parse_api_response(self, api_response: Any) -> RichLLMResponse:
        """Convierte la respuesta de OpenAI a RichLLMResponse."""
        choice = api_response.choices[0] if api_response.choices else None

        raw_content = ""
        if choice and choice.message and choice.message.content:
            raw_content = choice.message.content

        tool_calls, parsed_content, finish_reason, deferred_summary = (
            self._parse_action_from_content(raw_content)
        )

        # Normalizamos el finish_reason de OpenAI
        if finish_reason == "stop" and choice:
            finish_reason = self._normalize_finish_reason(
                choice.finish_reason or "stop"
            )

        token_usage = TokenUsage(
            input_tokens=api_response.usage.prompt_tokens if api_response.usage else 0,
            output_tokens=api_response.usage.completion_tokens if api_response.usage else 0,
            model=self._model_id,
        )

        return RichLLMResponse(
            content=parsed_content,
            tool_calls=tool_calls,
            token_usage=token_usage,
            model=self._model_id,
            finish_reason=finish_reason,
            deferred_completion_summary=deferred_summary,
        )

    def _parse_action_from_content(
        self,
        content: str,
    ) -> tuple[list[ToolCallRequest], str, str, str | None]:
        """
        Parsea TODOS los bloques JSON de acción del contenido del LLM.

        OPTIMIZACIÓN CLAVE: Cuando el LLM genera múltiples bloques tool_call
        consecutivos, los parseamos TODOS y los retornamos como lista de
        ToolCallRequests. Antes se descartaban todos menos el primero,
        desperdiciando los tokens de output y forzando al LLM a re-generar
        las acciones descartadas en iteraciones posteriores.

        Ejemplo de entrada que ahora se maneja correctamente:
          {"action":"tool_call","tool":"file","arguments":{...}}
          {"action":"tool_call","tool":"shell","arguments":{...}}
          {"action":"complete","summary":"..."}

        → Retorna 2 ToolCallRequests + deferred_summary del complete.
        """
        stripped = content.strip()

        if not stripped.startswith("{"):
            return [], content, "stop", None

        blocks, trailing_content = self._extract_json_object_blocks(stripped)
        if not blocks:
            logger.warning(
                "LLM devolvió JSON malformado, tratando como texto",
                content_preview=stripped[:200],
                provider=self.get_provider_name(),
                task_id=self._task_id,
            )
            return [], content, "stop", None

        # Recolectar TODAS las tool_calls y buscar un complete diferido
        tool_calls: list[ToolCallRequest] = []
        deferred_summary: str | None = None
        first_non_tool_action: str | None = None
        first_non_tool_content: str | None = None

        for block in blocks:
            if not isinstance(block, dict):
                continue

            action = block.get(_ACTION_FIELD, "")

            if action == _ACTION_TOOL_CALL:
                tool_name = block.get("tool", "")
                arguments = block.get("arguments", {})
                if tool_name:
                    tool_calls.append(ToolCallRequest(
                        tool_call_id=generate_id(),
                        tool_name=tool_name,
                        arguments=arguments if isinstance(arguments, dict) else {},
                    ))

            elif action == _ACTION_COMPLETE:
                summary = str(block.get("summary", "")).strip()
                if tool_calls:
                    # Si hay tool_calls antes, el complete es diferido
                    deferred_summary = summary or None
                else:
                    # Es el único bloque — complete directo
                    return [], summary, "complete", None

            elif action == _ACTION_THINK:
                if first_non_tool_action is None:
                    first_non_tool_action = "think"
                    first_non_tool_content = block.get("content", "")

        # Si recolectamos tool_calls, retornarlas todas
        if tool_calls:
            if len(tool_calls) > 1:
                logger.info(
                    "LLM generó múltiples tool_calls — ejecutando todas",
                    count=len(tool_calls),
                    tools=[tc.tool_name for tc in tool_calls],
                    provider=self.get_provider_name(),
                    task_id=self._task_id,
                )
            return tool_calls, "", "tool_use", deferred_summary

        # Si solo hubo un think
        if first_non_tool_action == "think":
            return [], first_non_tool_content or "", "stop", None

        # Fallback: primer bloque como acción única (backward compat)
        parsed = blocks[0]
        action = parsed.get(_ACTION_FIELD, "")

        if action == _ACTION_COMPLETE:
            return [], parsed.get("summary", ""), "complete", None
        if action == _ACTION_THINK:
            return [], parsed.get("content", ""), "stop", None

        return [], content, "stop", None

    def _normalize_finish_reason(self, openai_reason: str) -> str:
        """
        Normaliza el finish_reason de OpenAI a nuestro vocabulario.

        OpenAI usa: "stop", "length", "content_filter", "tool_calls"
        Nosotros:   "stop", "max_tokens", "stop",        "tool_use"
        """
        mapping = {
            "stop":           "stop",
            "length":         "max_tokens",
            "content_filter": "stop",
            "tool_calls":     "tool_use",
        }
        return mapping.get(openai_reason, "stop")

    def _extract_retry_after(self, exc: openai.RateLimitError) -> float | None:
        """Extrae retry-after del header de la respuesta 429 de OpenAI."""
        try:
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

    def _is_o1_model(self) -> bool:
        """True si el modelo activo es de la familia o1."""
        return self._model_id in _O1_MODELS

    def _uses_max_completion_tokens(self) -> bool:
        """
        True si el modelo espera `max_completion_tokens` en vez de `max_tokens`.

        OpenAI ya exige esto en familias como `o1` y `gpt-5`.
        """
        return self._is_o1_model() or self._model_id.startswith("gpt-5")

    def _supports_stop_sequences(self) -> bool:
        """True si el modelo acepta el parámetro `stop`."""
        return not self._is_o1_model() and not self._model_id.startswith("gpt-5")
