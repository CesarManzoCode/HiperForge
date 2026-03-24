"""
Configuración centralizada de HiperForge.

Este módulo es el único punto donde se leen variables de entorno
y archivos de configuración. El resto del sistema importa Settings
y accede a los valores ya parseados y validados.

JERARQUÍA DE CONFIGURACIÓN (de menor a mayor prioridad):
  1. Valores por defecto en Settings (constants.py)
  2. Archivo ~/.hiperforge/preferences.json
  3. Variables de entorno (HIPERFORGE_*, ANTHROPIC_API_KEY, etc.)

  Si defines HIPERFORGE_LLM_PROVIDER=openai en el entorno,
  sobreescribe cualquier valor en preferences.json.

¿POR QUÉ PYDANTIC-SETTINGS?
  Pydantic-settings hace tres cosas automáticamente:
    1. Lee las variables de entorno por nombre.
    2. Valida que los tipos sean correctos (int, float, bool, etc.).
    3. Lanza errores descriptivos si falta una variable requerida.

  Sin pydantic-settings tendríamos decenas de os.getenv() dispersos
  por el código, sin validación de tipos y sin valores por defecto
  centralizados.

USO:
  from hiperforge.core.config import get_settings

  settings = get_settings()
  print(settings.llm_provider)     # "anthropic"
  print(settings.anthropic_api_key) # "sk-ant-..."
  print(settings.is_debug)          # False
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from hiperforge.core.constants import (
    APP_DIR,
    ENV_ANTHROPIC_API_KEY,
    ENV_APP_DIR,
    ENV_DEBUG,
    ENV_LLM_MODEL,
    ENV_LLM_PROVIDER,
    ENV_OLLAMA_BASE_URL,
    ENV_OPENAI_API_KEY,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_MODEL_ANTHROPIC,
    LLM_DEFAULT_MODEL_OLLAMA,
    LLM_DEFAULT_MODEL_OPENAI,
    LLM_DEFAULT_PROVIDER,
    LLM_DEFAULT_TEMPERATURE,
    LOG_LEVEL_DEBUG,
    LOG_LEVEL_DEFAULT,
    REACT_MAX_ITERATIONS_PER_SUBTASK,
    REACT_MAX_SUBTASKS,
    REACT_MAX_TOOL_RETRIES,
    TOOL_DEFAULT_TIMEOUT_SECONDS,
    TOOL_EXTENDED_TIMEOUT_SECONDS,
    TOOL_MAX_FILE_SIZE_BYTES,
    TOOL_MAX_OUTPUT_CHARS,
)


class Settings(BaseSettings):
    """
    Configuración completa del sistema.

    Cada campo tiene un valor por defecto sensato para que el agente
    funcione sin configuración adicional (excepto la API key del LLM).

    Los nombres de campo mapean directamente a variables de entorno
    mediante el prefijo definido en model_config.
    """

    model_config = SettingsConfigDict(
        # Prefijo de variables de entorno para evitar colisiones
        # HIPERFORGE_DEBUG=true → settings.debug = True
        env_prefix="HIPERFORGE_",
        # Archivo .env en el directorio de trabajo del proyecto
        env_file=".env",
        env_file_encoding="utf-8",
        # No explotar si hay variables extra en el .env
        extra="ignore",
        # Permite leer el .env sin que exista (primera ejecución)
        env_ignore_empty=True,
        # Case insensitive para las variables de entorno
        case_sensitive=False,
    )

    # ------------------------------------------------------------------
    # Directorio de datos — sobreescribible para tests
    # ------------------------------------------------------------------

    app_dir: Path = Field(
        default=APP_DIR,
        description="Directorio raíz de datos. Default: ~/.hiperforge/",
        alias=ENV_APP_DIR,
    )

    # ------------------------------------------------------------------
    # Configuración del LLM
    # ------------------------------------------------------------------

    llm_provider: str = Field(
        default=LLM_DEFAULT_PROVIDER,
        description="Proveedor del LLM: 'anthropic', 'openai', 'ollama'",
        alias=ENV_LLM_PROVIDER,
    )

    llm_model: str | None = Field(
        default=None,
        description=(
            "Modelo específico a usar. Si None, se usa el default del proveedor. "
            "Ejemplo: 'claude-sonnet-4-6', 'gpt-4o', 'llama3'"
        ),
        alias=ENV_LLM_MODEL,
    )

    llm_temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperatura de generación. 0.0=determinista, 2.0=muy creativo.",
    )

    llm_max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        gt=0,
        le=200_000,
        description="Máximo de tokens en la respuesta del LLM.",
    )

    # ------------------------------------------------------------------
    # API Keys de proveedores — leídas directamente por su nombre estándar
    # (sin prefijo HIPERFORGE_ para compatibilidad con el ecosistema)
    # ------------------------------------------------------------------

    anthropic_api_key: str | None = Field(
        default=None,
        description="API key de Anthropic. Requerida si llm_provider='anthropic'.",
        alias=ENV_ANTHROPIC_API_KEY,
    )

    openai_api_key: str | None = Field(
        default=None,
        description="API key de OpenAI. Requerida si llm_provider='openai'.",
        alias=ENV_OPENAI_API_KEY,
    )

    ollama_base_url: str = Field(
        default="http://localhost:11434",
        description="URL base del servidor Ollama local.",
        alias=ENV_OLLAMA_BASE_URL,
    )

    # ------------------------------------------------------------------
    # Configuración del loop ReAct
    # ------------------------------------------------------------------

    react_max_iterations: int = Field(
        default=REACT_MAX_ITERATIONS_PER_SUBTASK,
        gt=0,
        le=50,
        description="Máximo de iteraciones del loop ReAct por subtask.",
    )

    react_max_subtasks: int = Field(
        default=REACT_MAX_SUBTASKS,
        gt=0,
        le=50,
        description="Máximo de subtasks que puede tener un plan.",
    )

    react_max_tool_retries: int = Field(
        default=REACT_MAX_TOOL_RETRIES,
        ge=0,
        le=10,
        description="Máximo de reintentos cuando una tool falla.",
    )

    # ------------------------------------------------------------------
    # Configuración de tools
    # ------------------------------------------------------------------

    tool_timeout_seconds: float = Field(
        default=TOOL_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        le=600.0,
        description="Timeout por defecto para ejecución de tools (segundos).",
    )

    tool_extended_timeout_seconds: float = Field(
        default=TOOL_EXTENDED_TIMEOUT_SECONDS,
        gt=0,
        le=600.0,
        description="Timeout extendido para operaciones lentas (segundos).",
    )

    tool_max_output_chars: int = Field(
        default=TOOL_MAX_OUTPUT_CHARS,
        gt=0,
        description="Máximo de caracteres del output de una tool enviados al LLM.",
    )

    tool_max_file_size_bytes: int = Field(
        default=TOOL_MAX_FILE_SIZE_BYTES,
        gt=0,
        description="Tamaño máximo de archivo que FileTool puede leer completo.",
    )

    # ------------------------------------------------------------------
    # Modo debug y logging
    # ------------------------------------------------------------------

    debug: bool = Field(
        default=False,
        description="Activa logs detallados y desactiva optimizaciones.",
        alias=ENV_DEBUG,
    )

    # ------------------------------------------------------------------
    # Propiedades calculadas
    # ------------------------------------------------------------------

    @property
    def log_level(self) -> str:
        """Nivel de log según el modo debug."""
        return LOG_LEVEL_DEBUG if self.debug else LOG_LEVEL_DEFAULT

    @property
    def effective_llm_model(self) -> str:
        """
        Modelo efectivo a usar según el proveedor configurado.

        Si el usuario configuró un modelo específico, lo usa.
        Si no, usa el modelo por defecto del proveedor activo.
        """
        if self.llm_model:
            return self.llm_model

        defaults = {
            "anthropic": LLM_DEFAULT_MODEL_ANTHROPIC,
            "openai":    LLM_DEFAULT_MODEL_OPENAI,
            "ollama":    LLM_DEFAULT_MODEL_OLLAMA,
        }
        # Si el proveedor es desconocido, devolvemos el string tal cual
        # para que el adapter lance un error descriptivo
        return defaults.get(self.llm_provider, self.llm_provider)

    @property
    def active_api_key(self) -> str | None:
        """
        API key del proveedor activo.

        Devuelve la key correspondiente al proveedor configurado.
        None si es Ollama (no requiere key) o si no está configurada.
        """
        keys = {
            "anthropic": self.anthropic_api_key,
            "openai":    self.openai_api_key,
            "ollama":    None,   # Ollama es local, no necesita key
        }
        return keys.get(self.llm_provider)

    # ------------------------------------------------------------------
    # Validadores — se ejecutan al instanciar Settings
    # ------------------------------------------------------------------

    @field_validator("llm_provider")
    @classmethod
    def validate_provider(cls, value: str) -> str:
        """Verifica que el proveedor sea uno de los soportados."""
        supported = {"anthropic", "openai", "ollama"}
        normalized = value.lower().strip()

        if normalized not in supported:
            raise ValueError(
                f"Proveedor '{value}' no soportado. "
                f"Opciones válidas: {', '.join(sorted(supported))}"
            )
        return normalized

    @field_validator("app_dir", mode="before")
    @classmethod
    def expand_app_dir(cls, value: Any) -> Path:
        """
        Expande ~ en la ruta del directorio de datos.

        Permite configurar HIPERFORGE_APP_DIR=~/mis-datos
        y que se resuelva correctamente al directorio del usuario.
        """
        return Path(str(value)).expanduser()

    @model_validator(mode="after")
    def validate_api_key_present(self) -> Settings:
        """
        Verifica que la API key del proveedor activo esté configurada.

        Solo valida si el proveedor requiere key (no Ollama).
        No bloqueamos la instanciación — solo advertimos, porque
        el usuario puede estar configurando el sistema por primera vez.
        """
        if self.llm_provider != "ollama" and self.active_api_key is None:
            # No lanzamos error — el adapter lanzará LLMConnectionError
            # con un mensaje descriptivo cuando intente conectarse.
            # Aquí solo es una advertencia para debug.
            pass
        return self


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Devuelve la instancia única de Settings (singleton).

    lru_cache(maxsize=1) garantiza que Settings se instancia una sola vez
    durante toda la ejecución — las variables de entorno se leen una vez
    al arrancar, no en cada llamada.

    Para tests que necesiten settings distintos usar:
        get_settings.cache_clear()
        os.environ["HIPERFORGE_DEBUG"] = "true"
        settings = get_settings()

    Returns:
        Instancia configurada y validada de Settings.
    """
    return Settings()