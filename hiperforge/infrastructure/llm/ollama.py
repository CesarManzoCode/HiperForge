"""
OllamaAdapter — Adapter para modelos locales via Ollama.

Ollama permite correr modelos LLM localmente sin internet ni API keys.
Es ideal para developers que:
  - Trabajan con código propietario que no puede salir a la nube
  - Quieren cero costo de tokens durante desarrollo/testing
  - Necesitan trabajar offline

DIFERENCIAS CON ANTHROPIC/OPENAI:
  1. Sin API key — el servidor local no requiere autenticación
  2. Latencia variable — depende del hardware del developer
  3. Capacidad variable — modelos locales son menos capaces que GPT-4o/Claude
  4. La API REST de Ollama es compatible con OpenAI en su endpoint /v1

Por eso OllamaAdapter hereda de OpenAIAdapter usando la URL local de Ollama.
La lógica de llamada, parseo y retry es idéntica.

REQUISITO:
  Ollama debe estar corriendo localmente:
    brew install ollama && ollama serve
    ollama pull llama3.3   # descargar el modelo

MODELOS RECOMENDADOS PARA CÓDIGO:
  llama3.3        → mejor balance calidad/velocidad para código
  codellama       → especializado en código, bueno para tasks técnicas
  deepseek-coder  → excelente para generación y análisis de código
  mistral         → rápido y versátil
"""

from __future__ import annotations

from hiperforge.core.constants import ENV_OLLAMA_BASE_URL
from hiperforge.core.logging import get_logger
from hiperforge.infrastructure.llm.openai import OpenAIAdapter

logger = get_logger(__name__)

# URL por defecto del servidor Ollama local
_OLLAMA_DEFAULT_BASE_URL = "http://localhost:11434/v1"

# Context windows aproximados — Ollama no siempre reporta esto
# correctamente, así que usamos valores conservadores
_OLLAMA_CONTEXT_WINDOWS: dict[str, int] = {
    "llama3.3":       128_000,
    "llama3.1":       128_000,
    "llama3":           8_192,
    "codellama":        16_384,
    "deepseek-coder":   16_384,
    "mistral":          32_768,
    "mixtral":          32_768,
    "gemma2":            8_192,
    "phi3":              4_096,
}

_OLLAMA_DEFAULT_MODEL = "llama3.3"


class OllamaAdapter(OpenAIAdapter):
    """
    Adapter para modelos locales via Ollama.

    Hereda de OpenAIAdapter porque Ollama expone un endpoint
    compatible con la API de OpenAI en /v1.

    Parámetros:
        model_id: Nombre del modelo Ollama a usar.
                  Debe estar descargado localmente con `ollama pull <model>`.
        base_url: URL del servidor Ollama. Default: http://localhost:11434/v1
    """

    def __init__(
        self,
        model_id: str = _OLLAMA_DEFAULT_MODEL,
        base_url: str = _OLLAMA_DEFAULT_BASE_URL,
    ) -> None:
        # Ollama no requiere API key — pasamos un placeholder
        # El cliente OpenAI requiere un valor no vacío aunque no se use
        super().__init__(
            api_key="ollama",   # placeholder — Ollama no valida esto
            model_id=model_id,
            base_url=base_url,
        )

        logger.debug(
            "OllamaAdapter inicializado",
            model=model_id,
            base_url=base_url,
        )

    # ------------------------------------------------------------------
    # Sobreescribimos solo lo que cambia respecto a OpenAI
    # ------------------------------------------------------------------

    def get_provider_name(self) -> str:
        return "ollama"

    def get_context_window_size(self) -> int:
        """
        Context window aproximado del modelo Ollama.

        Ollama no siempre reporta el context window correctamente.
        Usamos la tabla de valores conocidos con fallback conservador.
        """
        # Normalizamos el nombre para manejar variantes como "llama3:8b"
        base_name = self._model_id.split(":")[0].lower()
        return _OLLAMA_CONTEXT_WINDOWS.get(base_name, 4_096)

    def is_available(self) -> bool:
        """
        Verifica que el servidor Ollama esté corriendo y el modelo disponible.

        A diferencia de los providers cloud, Ollama puede no estar iniciado.
        Damos un mensaje de error más descriptivo en ese caso.
        """
        try:
            # Intentamos una llamada mínima al servidor local
            self._client.chat.completions.create(
                model=self._model_id,
                max_tokens=1,
                messages=[{"role": "user", "content": "ping"}],
            )
            return True

        except Exception as exc:
            # Logeamos el error específico para que el developer sepa qué hacer
            error_msg = str(exc).lower()

            if "connection refused" in error_msg or "connection error" in error_msg:
                logger.warning(
                    "Ollama no está corriendo. Iniciarlo con: ollama serve",
                    base_url=self._client.base_url,
                )
            elif "model" in error_msg and "not found" in error_msg:
                logger.warning(
                    f"Modelo '{self._model_id}' no descargado. "
                    f"Descargarlo con: ollama pull {self._model_id}",
                )

            return False