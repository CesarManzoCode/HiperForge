"""
GroqAdapter — Adapter para la API de Groq.

Groq ofrece inferencia ultra-rápida (Low Latency Language Processing)
usando hardware especializado (LPU). Su API es 100% compatible con
la de OpenAI — mismos endpoints, mismo formato de request y response.

Por eso GroqAdapter hereda directamente de OpenAIAdapter:
solo cambia la base_url y los modelos disponibles.
Toda la lógica de llamada, parseo, retry y streaming es idéntica.

MODELOS DE GROQ (velocidad de inferencia muy superior a OpenAI/Anthropic):
  llama-3.3-70b-versatile     → modelo general de alta calidad
  llama-3.1-8b-instant        → modelo ultra-rápido para tasks simples
  mixtral-8x7b-32768          → contexto largo (32k tokens)
  gemma2-9b-it                → modelo compacto de Google

¿Cuándo usar Groq vs Anthropic/OpenAI?
  - Groq es ideal para el loop ReAct donde la velocidad importa mucho.
    Cada iteración espera la respuesta del LLM — menos latencia = agente más rápido.
  - Anthropic/OpenAI tienen modelos más capaces para tasks complejas.
  - El usuario elige su proveedor en la configuración — el agente funciona igual.

LIMITACIÓN IMPORTANTE:
  Groq tiene rate limits más estrictos que OpenAI/Anthropic en su tier gratuito.
  El retry con backoff del BaseLLMAdapter lo maneja automáticamente.
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.infrastructure.llm.openai import OpenAIAdapter

logger = get_logger(__name__)

# URL base oficial de la API de Groq
_GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# Modelos de Groq con sus context windows reales
_GROQ_CONTEXT_WINDOWS: dict[str, int] = {
    "llama-3.3-70b-versatile":  128_000,
    "llama-3.1-8b-instant":     128_000,
    "mixtral-8x7b-32768":        32_768,
    "gemma2-9b-it":               8_192,
}

# Modelo por defecto de Groq — balance entre velocidad y calidad
_GROQ_DEFAULT_MODEL = "llama-3.3-70b-versatile"


class GroqAdapter(OpenAIAdapter):
    """
    Adapter para la API de Groq.

    Hereda toda la lógica de OpenAIAdapter — solo sobreescribe
    la base_url, el provider name, y el context window size.

    Parámetros:
        api_key:  API key de Groq. Obtenida en console.groq.com.
        model_id: Modelo de Groq a usar. Default: llama-3.3-70b-versatile.
    """

    def __init__(
        self,
        api_key: str,
        model_id: str = _GROQ_DEFAULT_MODEL,
    ) -> None:
        # Pasamos la base_url de Groq al OpenAIAdapter
        # El cliente OpenAI apunta a los servidores de Groq automáticamente
        super().__init__(
            api_key=api_key,
            model_id=model_id,
            base_url=_GROQ_BASE_URL,
        )

        logger.debug(
            "GroqAdapter inicializado",
            model=model_id,
            base_url=_GROQ_BASE_URL,
        )

    # ------------------------------------------------------------------
    # Solo sobreescribimos lo que cambia respecto a OpenAI
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        """Groq es un proveedor distinto a OpenAI aunque comparta la API."""
        return "groq"

    def get_context_window_size(self) -> int:
        """
        Devuelve el context window real del modelo de Groq configurado.

        Los modelos de Groq tienen context windows distintos a los de OpenAI
        aunque compartan nombres similares.
        """
        return _GROQ_CONTEXT_WINDOWS.get(self._model_id, 8_192)