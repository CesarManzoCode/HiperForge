"""
DTOs — Data Transfer Objects de la capa de aplicación.

Los DTOs son los contratos de entrada y salida de cada use case.
Son la frontera entre la CLI y el dominio — la CLI solo conoce DTOs,
nunca las entidades del dominio directamente.

¿POR QUÉ DTOs Y NO PASAR LAS ENTIDADES DIRECTAMENTE?
  Las entidades del dominio (Task, Project, Workspace) contienen lógica
  de negocio compleja y no deben exponerse a la capa de presentación.
  Los DTOs son objetos planos que:

    1. Desacoplan la CLI del dominio.
       La CLI no necesita saber cómo funciona Task internamente.
       Solo necesita el id, status y summary para mostrarlo.

    2. Validan la entrada en el borde del sistema.
       Si el usuario pasa un prompt vacío, el DTO lo rechaza
       antes de que llegue al use case o al dominio.

    3. Simplifican el output.
       RunTaskOutput tiene exactamente los campos que la CLI necesita
       mostrar — sin campos internos del dominio que no tienen sentido
       para el usuario.

CONVENCIÓN DE NOMBRES:
  - Input DTOs:  <UseCase>Input  (ej: RunTaskInput)
  - Output DTOs: <UseCase>Output (ej: RunTaskOutput)
  - Resúmenes:   <Entidad>Summary (ej: ProjectSummary, WorkspaceSummary)

INMUTABILIDAD:
  Todos los DTOs son frozen dataclasses.
  Una vez creados no se modifican — si el resultado cambia,
  se crea un nuevo DTO.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


# ---------------------------------------------------------------------------
# Run Task — use case principal del agente
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RunTaskInput:
    """
    Entrada para ejecutar una instrucción con el agente ReAct.

    Atributos:
        prompt:       Instrucción del usuario. Requerida, no puede estar vacía.
        project_id:   Proyecto al que pertenece la task. None = task suelta.
        workspace_id: Workspace donde ejecutar. None = usar el activo global.
        auto_confirm: Si True, ejecuta el plan sin pedir confirmación al usuario.
                      Útil para pipelines automatizados o modo --yes de la CLI.
    """

    prompt: str
    project_id: str | None = None
    workspace_id: str | None = None
    auto_confirm: bool = False

    def __post_init__(self) -> None:
        # Validación en el borde — antes de llegar al dominio
        if not self.prompt or not self.prompt.strip():
            raise ValueError(
                "El prompt no puede estar vacío. "
                "Describe qué quieres que el agente haga."
            )


@dataclass(frozen=True)
class RunTaskOutput:
    """
    Resultado de ejecutar una task completa con el agente.

    Contiene exactamente la información que la CLI necesita mostrar
    al usuario al finalizar — ni más ni menos.

    Atributos:
        task_id:            ID de la task creada.
        status:             Estado final: "completed", "failed" o "cancelled".
        summary:            Resumen generado por el agente al terminar.
        subtasks_completed: Cuántas subtasks terminaron exitosamente.
        subtasks_total:     Total de subtasks en el plan.
        total_tokens:       Tokens consumidos en toda la task.
        estimated_cost_usd: Costo estimado en dólares.
        duration_seconds:   Tiempo total de ejecución.
        error_message:      Descripción del error si status == "failed".
    """

    task_id: str
    status: str
    summary: str
    subtasks_completed: int
    subtasks_total: int
    total_tokens: int
    estimated_cost_usd: float
    duration_seconds: float
    error_message: str | None = None

    @property
    def succeeded(self) -> bool:
        """True si la task terminó exitosamente."""
        return self.status == "completed"

    @property
    def progress_pct(self) -> float:
        """Porcentaje de subtasks completadas. 0.0 si no hay subtasks."""
        if self.subtasks_total == 0:
            return 0.0
        return round((self.subtasks_completed / self.subtasks_total) * 100, 1)


# ---------------------------------------------------------------------------
# Create Task / Task Summary
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CreateTaskInput:
    """
    Entrada para crear una task en estado PENDING sin ejecutarla.

    Atributos:
        prompt:       Instrucción del usuario. Requerida, no puede estar vacía.
        project_id:   Proyecto al que pertenecerá la task. None = task suelta.
        workspace_id: Workspace donde crear la task. None = usar el activo.
    """

    prompt: str
    project_id: str | None = None
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        if not self.prompt or not self.prompt.strip():
            raise ValueError(
                "El prompt no puede estar vacío. "
                "Describe qué quieres que la task haga."
            )


@dataclass(frozen=True)
class TaskSummary:
    """
    Resumen de una task para listados y confirmaciones de la CLI.

    Contiene el mínimo de datos que la capa de presentación necesita
    para mostrar estado, progreso, costo y contexto.
    """

    id: str
    prompt: str
    status: str
    project_id: str | None
    subtask_count: int
    completed_subtasks: int
    total_tokens: int
    estimated_cost_usd: float
    created_at: datetime
    completed_at: datetime | None = None

    @property
    def prompt_preview(self) -> str:
        """Versión corta del prompt para tablas y paneles."""
        trimmed = self.prompt.strip()
        if len(trimmed) <= 80:
            return trimmed
        return trimmed[:77] + "..."

    @property
    def progress_pct(self) -> float:
        """Porcentaje de subtasks completadas. 0.0 si no hay subtasks."""
        if self.subtask_count == 0:
            return 0.0
        return round((self.completed_subtasks / self.subtask_count) * 100, 1)


# ---------------------------------------------------------------------------
# Create Project
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CreateProjectInput:
    """
    Entrada para crear un nuevo proyecto dentro de un workspace.

    Atributos:
        name:         Nombre del proyecto. Debe ser único en el workspace.
        workspace_id: Workspace donde se crea el proyecto.
        description:  Descripción opcional del propósito del proyecto.
        tags:         Etiquetas opcionales para categorizar y filtrar.
    """

    name: str
    workspace_id: str
    description: str | None = None
    tags: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("El nombre del proyecto no puede estar vacío.")
        if not self.workspace_id or not self.workspace_id.strip():
            raise ValueError("workspace_id no puede estar vacío.")


@dataclass(frozen=True)
class ProjectSummary:
    """
    Resumen de un proyecto para mostrar en listados de la CLI.

    Contiene los datos mínimos necesarios para identificar un proyecto
    y mostrar su estado sin cargar sus tasks completas.
    """

    id: str
    name: str
    description: str | None
    status: str
    tags: list[str]
    task_count: int
    completed_tasks: int
    created_at: datetime
    updated_at: datetime

    @property
    def completion_pct(self) -> float:
        """Porcentaje de tasks completadas."""
        if self.task_count == 0:
            return 0.0
        return round((self.completed_tasks / self.task_count) * 100, 1)

    @property
    def status_icon(self) -> str:
        """Icono de texto para mostrar el estado en la CLI."""
        icons = {
            "active":   "●",
            "archived": "○",
            "deleted":  "✕",
        }
        return icons.get(self.status, "?")


# ---------------------------------------------------------------------------
# Create Workspace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CreateWorkspaceInput:
    """
    Entrada para crear un nuevo workspace.

    Atributos:
        name:        Nombre del workspace. Debe ser único globalmente.
        description: Descripción opcional del propósito del workspace.
    """

    name: str
    description: str | None = None

    def __post_init__(self) -> None:
        if not self.name or not self.name.strip():
            raise ValueError("El nombre del workspace no puede estar vacío.")


@dataclass(frozen=True)
class WorkspaceSummary:
    """
    Resumen de un workspace para mostrar en listados de la CLI.

    Atributos:
        is_active: True si este es el workspace activo actualmente.
    """

    id: str
    name: str
    description: str | None
    status: str
    project_count: int
    is_active: bool
    created_at: datetime

    @property
    def status_icon(self) -> str:
        """Icono de texto para mostrar el estado en la CLI."""
        if self.is_active:
            return "▶"
        icons = {
            "active":   "●",
            "archived": "○",
            "deleted":  "✕",
        }
        return icons.get(self.status, "?")


# ---------------------------------------------------------------------------
# Switch Workspace
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SwitchWorkspaceInput:
    """
    Entrada para cambiar el workspace activo.

    Acepta tanto el ID como el nombre del workspace para mayor comodidad
    desde la CLI — el use case resuelve cuál usar.
    """

    workspace_id: str | None = None
    workspace_name: str | None = None

    def __post_init__(self) -> None:
        if not self.workspace_id and not self.workspace_name:
            raise ValueError(
                "Debes especificar workspace_id o workspace_name."
            )


# ---------------------------------------------------------------------------
# Manage Preferences
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class UpdatePreferencesInput:
    """
    Entrada para actualizar preferencias del sistema.

    Atributos:
        updates:      Campos a actualizar en notación de punto.
                      Ejemplo: {"llm.provider": "groq", "ui.verbose": True}
                      Solo los campos presentes se modifican.
                      Los campos ausentes conservan su valor actual.
        workspace_id: Si se especifica, actualiza prefs del workspace.
                      Si None, actualiza las prefs globales.
    """

    updates: dict[str, Any]
    workspace_id: str | None = None

    def __post_init__(self) -> None:
        if not self.updates:
            raise ValueError(
                "updates no puede estar vacío. "
                "Especifica al menos un campo a actualizar."
            )
