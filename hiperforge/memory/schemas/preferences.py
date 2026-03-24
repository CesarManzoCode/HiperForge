"""
Schema: Preferencias del usuario

Define la estructura exacta de preferences.json en disco.
Hay dos niveles de preferencias:

  1. Globales: ~/.hiperforge/preferences.json
     Se aplican a todos los workspaces.

  2. Por workspace: ~/.hiperforge/workspaces/{id}/preferences.json
     Sobreescriben las globales para ese workspace específico.

CASCADA DE PREFERENCIAS (de menor a mayor prioridad):
  Defaults en código → Globales → Workspace → Variables de entorno

  Ejemplo:
    Global:    llm_provider = "anthropic"
    Workspace: llm_provider = "groq"      ← sobreescribe para este workspace
    Resultado: se usa groq en este workspace, anthropic en los demás

USO:
  from hiperforge.memory.schemas.preferences import UserPrefsSchema, LLMProfileSchema

  # Crear preferencias con defaults
  prefs = UserPrefsSchema()

  # Serializar para guardar
  data = prefs.model_dump()

  # Cargar desde disco
  prefs = UserPrefsSchema.model_validate(data)
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from hiperforge.core.constants import (
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_MODEL_ANTHROPIC,
    LLM_DEFAULT_PROVIDER,
    LLM_DEFAULT_TEMPERATURE,
    REACT_MAX_ITERATIONS_PER_SUBTASK,
    REACT_MAX_SUBTASKS,
    SCHEMA_VERSION_PREFERENCES,
    TOOL_DEFAULT_TIMEOUT_SECONDS,
)


class LLMProfileSchema(BaseModel):
    """
    Configuración del LLM para un workspace o globalmente.

    Todos los campos tienen defaults sensatos — el usuario solo
    necesita configurar lo que quiere cambiar del comportamiento base.
    """

    provider: str = Field(
        default=LLM_DEFAULT_PROVIDER,
        description="Proveedor del LLM: anthropic, openai, groq, ollama",
    )

    model: str | None = Field(
        default=None,
        description=(
            "Modelo específico. None = usar el default del proveedor. "
            "Ejemplos: claude-sonnet-4-6, gpt-4o, llama-3.3-70b-versatile"
        ),
    )

    temperature: float = Field(
        default=LLM_DEFAULT_TEMPERATURE,
        ge=0.0,
        le=2.0,
        description="Temperatura de generación. Bajo = más determinista.",
    )

    max_tokens: int = Field(
        default=LLM_DEFAULT_MAX_TOKENS,
        gt=0,
        description="Máximo de tokens en la respuesta del LLM.",
    )


class AgentBehaviorSchema(BaseModel):
    """
    Configuración del comportamiento del agente ReAct.

    Controla cómo el agente planea y ejecuta las tasks.
    """

    max_react_iterations: int = Field(
        default=REACT_MAX_ITERATIONS_PER_SUBTASK,
        gt=0,
        le=50,
        description="Máximo de iteraciones del loop ReAct por subtask.",
    )

    max_subtasks: int = Field(
        default=REACT_MAX_SUBTASKS,
        gt=0,
        le=50,
        description="Máximo de subtasks que puede tener un plan.",
    )

    tool_timeout_seconds: float = Field(
        default=TOOL_DEFAULT_TIMEOUT_SECONDS,
        gt=0,
        description="Timeout por defecto para tools en segundos.",
    )

    auto_confirm_plan: bool = Field(
        default=False,
        description=(
            "Si True, el agente ejecuta el plan sin pedir confirmación. "
            "Útil para usuarios avanzados que confían en el agente."
        ),
    )

    show_reasoning: bool = Field(
        default=True,
        description="Mostrar el razonamiento del agente en la terminal.",
    )


class UIPrefsSchema(BaseModel):
    """Preferencias de la interfaz de terminal."""

    show_token_usage: bool = Field(
        default=True,
        description="Mostrar uso de tokens y costo estimado al terminar.",
    )

    show_timestamps: bool = Field(
        default=False,
        description="Mostrar timestamps en los logs de la terminal.",
    )

    verbose: bool = Field(
        default=False,
        description="Modo verbose: muestra más detalle en la terminal.",
    )


class UserPrefsSchema(BaseModel):
    """
    Schema completo de preferencias del usuario.

    Estructura del JSON en disco:
      {
        "schema_version": 1,
        "llm": { "provider": "anthropic", "model": null, ... },
        "agent": { "max_react_iterations": 15, ... },
        "ui": { "show_token_usage": true, ... }
      }
    """

    schema_version: int = Field(
        default=SCHEMA_VERSION_PREFERENCES,
        description="Versión del schema. Usado por migrations.py.",
    )

    llm: LLMProfileSchema = Field(
        default_factory=LLMProfileSchema,
        description="Configuración del LLM.",
    )

    agent: AgentBehaviorSchema = Field(
        default_factory=AgentBehaviorSchema,
        description="Comportamiento del agente ReAct.",
    )

    ui: UIPrefsSchema = Field(
        default_factory=UIPrefsSchema,
        description="Preferencias de la interfaz de terminal.",
    )

    def merge_with(self, override: UserPrefsSchema) -> UserPrefsSchema:
        """
        Combina estas preferencias con las de override.

        Los campos de override sobreescriben los de self.
        Usado para implementar la cascada global → workspace.

        Parámetros:
            override: Preferencias con mayor prioridad (workspace).

        Returns:
            Nuevas preferencias combinadas.
        """
        # Combinamos campo por campo para manejar defaults correctamente
        base_data = self.model_dump()
        override_data = override.model_dump(exclude_defaults=True)

        # Merge recursivo de dicts anidados
        merged = _deep_merge(base_data, override_data)

        return UserPrefsSchema.model_validate(merged)


def _deep_merge(base: dict, override: dict) -> dict:
    """
    Combina dos dicts recursivamente.

    Los valores de override sobreescriben los de base.
    Para dicts anidados, hace merge recursivo en vez de reemplazar.
    """
    result = dict(base)

    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value

    return result
