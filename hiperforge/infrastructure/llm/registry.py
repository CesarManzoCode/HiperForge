"""
LLMRegistry — Registro centralizado de adapters LLM.

El registry es el único lugar donde se instancian los adapters.
El resto del sistema pide un adapter por nombre de proveedor
y el registry se encarga de construirlo con la configuración correcta.

¿POR QUÉ UN REGISTRY?
  Sin registry, cada parte del código que necesita un LLM tendría que:
    1. Leer la configuración
    2. Saber qué adapter usar según el proveedor
    3. Instanciarlo con los parámetros correctos

  Con registry:
    adapter = LLMRegistry.get_adapter()  # una línea, siempre correcto

PATRÓN:
  container.py llama a LLMRegistry.get_adapter() una vez al arrancar.
  El adapter resultante se inyecta en todos los use cases y services
  que lo necesiten. Un solo adapter por proceso.
"""

from __future__ import annotations

from hiperforge.core.config import get_settings
from hiperforge.core.logging import get_logger
from hiperforge.domain.exceptions import LLMConnectionError
from hiperforge.infrastructure.llm.anthropic import AnthropicAdapter
from hiperforge.infrastructure.llm.base import BaseLLMAdapter
from hiperforge.infrastructure.llm.groq import GroqAdapter
from hiperforge.infrastructure.llm.ollama import OllamaAdapter
from hiperforge.infrastructure.llm.openai import OpenAIAdapter

logger = get_logger(__name__)


class LLMRegistry:
    """
    Fábrica de adapters LLM.

    Todos los métodos son estáticos — el registry no tiene estado propio,
    solo sabe cómo construir adapters a partir de la configuración.
    """

    @staticmethod
    def get_adapter(provider: str | None = None) -> BaseLLMAdapter:
        """
        Construye y devuelve el adapter del proveedor especificado.

        Parámetros:
            provider: Nombre del proveedor. Si None, usa el configurado
                      en settings (HIPERFORGE_LLM_PROVIDER o default).

        Returns:
            Adapter listo para usar, ya instanciado con sus credenciales.

        Raises:
            LLMConnectionError: Si el proveedor no está soportado o
                                 si le falta la API key requerida.
        """
        settings = get_settings()
        effective_provider = (provider or settings.llm_provider).lower()
        model_id = settings.effective_llm_model

        logger.info(
            "inicializando adapter LLM",
            provider=effective_provider,
            model=model_id,
        )

        # Mapa de constructores — agregar un provider nuevo = agregar una entrada aquí
        builders = {
            "anthropic": lambda: LLMRegistry._build_anthropic(model_id, settings),
            "openai":    lambda: LLMRegistry._build_openai(model_id, settings),
            "groq":      lambda: LLMRegistry._build_groq(model_id, settings),
            "ollama":    lambda: LLMRegistry._build_ollama(model_id, settings),
        }

        builder = builders.get(effective_provider)

        if builder is None:
            supported = ", ".join(sorted(builders.keys()))
            raise LLMConnectionError(
                provider=effective_provider,
                reason=(
                    f"Proveedor '{effective_provider}' no soportado. "
                    f"Opciones válidas: {supported}"
                ),
            )

        return builder()

    @staticmethod
    def list_supported_providers() -> list[str]:
        """Devuelve la lista de proveedores soportados."""
        return sorted(["anthropic", "openai", "groq", "ollama"])

    # ------------------------------------------------------------------
    # Constructores privados por proveedor
    # ------------------------------------------------------------------

    @staticmethod
    def _build_anthropic(model_id: str, settings: any) -> AnthropicAdapter:
        """Construye el adapter de Anthropic validando que tenga API key."""
        api_key = settings.anthropic_api_key

        if not api_key:
            raise LLMConnectionError(
                provider="anthropic",
                reason=(
                    "ANTHROPIC_API_KEY no configurada. "
                    "Agrégala al .env o como variable de entorno."
                ),
            )

        return AnthropicAdapter(api_key=api_key, model_id=model_id)

    @staticmethod
    def _build_openai(model_id: str, settings: any) -> OpenAIAdapter:
        """Construye el adapter de OpenAI validando que tenga API key."""
        api_key = settings.openai_api_key

        if not api_key:
            raise LLMConnectionError(
                provider="openai",
                reason=(
                    "OPENAI_API_KEY no configurada. "
                    "Agrégala al .env o como variable de entorno."
                ),
            )

        return OpenAIAdapter(api_key=api_key, model_id=model_id)

    @staticmethod
    def _build_groq(model_id: str, settings: any) -> GroqAdapter:
        """Construye el adapter de Groq validando que tenga API key."""
        # Groq usa su propia variable de entorno
        api_key = getattr(settings, "groq_api_key", None)

        if not api_key:
            raise LLMConnectionError(
                provider="groq",
                reason=(
                    "GROQ_API_KEY no configurada. "
                    "Agrégala al .env o como variable de entorno. "
                    "Obtén una key gratis en console.groq.com"
                ),
            )

        return GroqAdapter(api_key=api_key, model_id=model_id)

    @staticmethod
    def _build_ollama(model_id: str, settings: any) -> OllamaAdapter:
        """
        Construye el adapter de Ollama.

        Ollama no requiere API key — solo necesita el servidor corriendo.
        Usamos la base_url de la configuración o el default local.
        """
        base_url = settings.ollama_base_url

        return OllamaAdapter(
            model_id=model_id,
            base_url=f"{base_url.rstrip('/')}/v1",
        )
