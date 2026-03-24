"""
Entidad: Project

Un Project agrupa Tasks relacionadas bajo un mismo contexto.
Es el nivel organizacional por encima de las Tasks.

¿Para qué sirve tener Projects?
  Sin Projects, todas las Tasks del usuario estarían mezcladas.
  Con Projects el usuario puede tener contextos separados:

  Proyecto: "API de pagos"
    ├── Task: "Implementar endpoint de cobro"
    ├── Task: "Agregar validación de tarjetas"
    └── Task: "Escribir documentación de la API"

  Proyecto: "Dashboard admin"
    ├── Task: "Crear componente de tabla"
    └── Task: "Agregar filtros de búsqueda"

  Cada Project vive dentro de un Workspace (multi-workspace).
  Un Workspace puede tener múltiples Projects.

CICLO DE VIDA:
  ACTIVE → ARCHIVED   (el usuario archiva proyectos terminados)
  ACTIVE → DELETED    (eliminación lógica, no borra los archivos)
  ARCHIVED → ACTIVE   (se puede reactivar un proyecto archivado)

  No existe COMPLETED porque un proyecto puede recibir Tasks
  nuevas en cualquier momento mientras esté ACTIVE.

USO TÍPICO:
  # Crear un proyecto nuevo
  project = Project.create(
      name="API de pagos",
      workspace_id="01HX4K...",
      description="Backend de procesamiento de pagos con Stripe",
  )

  # Agregar una task al proyecto
  project = project.add_task(task)

  # Archivar cuando ya no se use activamente
  project = project.archive()

  # Ver el resumen del proyecto
  print(project.task_count)         # 3
  print(project.completed_ratio)    # 0.66 (2 de 3 completadas)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.core.utils.ids import generate_id
from hiperforge.domain.entities.task import Task, TaskStatus


class ProjectStatus(str, Enum):
    """
    Estado del ciclo de vida de un Project.

    Transiciones válidas → ver _VALID_TRANSITIONS abajo.
    """

    ACTIVE = "active"       # Activo, recibe nuevas Tasks
    ARCHIVED = "archived"   # Archivado, solo lectura
    DELETED = "deleted"     # Eliminación lógica (los datos siguen en disco)


_VALID_TRANSITIONS: dict[ProjectStatus, frozenset[ProjectStatus]] = {
    ProjectStatus.ACTIVE:    frozenset({ProjectStatus.ARCHIVED, ProjectStatus.DELETED}),
    ProjectStatus.ARCHIVED:  frozenset({ProjectStatus.ACTIVE, ProjectStatus.DELETED}),
    ProjectStatus.DELETED:   frozenset(),   # terminal — eliminación es irreversible
}


@dataclass(frozen=True)
class Project:
    """
    Agrupador de Tasks relacionadas dentro de un Workspace.

    Atributos:
        id:           Identificador único (ULID).
        workspace_id: ID del Workspace al que pertenece.
        name:         Nombre del proyecto (único dentro del workspace).
        description:  Descripción opcional del propósito del proyecto.
        status:       Estado actual del ciclo de vida.
        tags:         Etiquetas para categorizar y filtrar proyectos.
                      Guardadas como tuple para mantener inmutabilidad.
        tasks:        Tasks del proyecto en orden cronológico de creación.
                      Guardadas como tuple para mantener inmutabilidad.
        created_at:   Cuándo fue creado (UTC).
        updated_at:   Última vez que se modificó el proyecto (UTC).
                      Se actualiza al agregar tasks o cambiar estado.
    """

    id: str
    workspace_id: str
    name: str
    description: str | None
    status: ProjectStatus
    tags: tuple[str, ...]
    tasks: tuple[Task, ...]
    created_at: datetime
    updated_at: datetime

    # ------------------------------------------------------------------
    # Constructor principal
    # ------------------------------------------------------------------

    @classmethod
    def create(
        cls,
        name: str,
        workspace_id: str,
        *,
        description: str | None = None,
        tags: list[str] | None = None,
    ) -> Project:
        """
        Crea un nuevo Project en estado ACTIVE.

        Parámetros:
            name:         Nombre del proyecto. Debe ser único en el workspace.
            workspace_id: ID del workspace al que pertenece.
            description:  Descripción opcional del proyecto.
            tags:         Etiquetas opcionales para categorizar.
        """
        now = datetime.now(timezone.utc)

        return cls(
            id=generate_id(),
            workspace_id=workspace_id,
            name=name.strip(),          # limpiamos espacios del nombre
            description=description,
            status=ProjectStatus.ACTIVE,
            tags=tuple(tags or []),
            tasks=(),
            created_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Métodos de transición de estado
    # ------------------------------------------------------------------

    def _transition_to(self, new_status: ProjectStatus) -> Project:
        """Valida y ejecuta una transición de estado."""
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"Project({self.id}, '{self.name}')",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            description=self.description,
            status=new_status,
            tags=self.tags,
            tasks=self.tasks,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def archive(self) -> Project:
        """
        Archiva el proyecto. Ya no recibirá nuevas Tasks.
        Se puede reactivar con reactivate() si se necesita.
        """
        return self._transition_to(ProjectStatus.ARCHIVED)

    def reactivate(self) -> Project:
        """Reactiva un proyecto archivado para que vuelva a recibir Tasks."""
        return self._transition_to(ProjectStatus.ACTIVE)

    def delete(self) -> Project:
        """
        Eliminación lógica del proyecto.

        Los datos siguen en disco — solo se marca como DELETED.
        Esto es irreversible desde el dominio.
        """
        return self._transition_to(ProjectStatus.DELETED)

    # ------------------------------------------------------------------
    # Métodos de actualización
    # ------------------------------------------------------------------

    def add_task(self, task: Task) -> Project:
        """
        Agrega una Task al proyecto.

        Solo se pueden agregar Tasks a proyectos ACTIVE.

        Raises:
            PermissionError: Si el proyecto no está activo.
            ValueError:      Si la Task no pertenece a este proyecto.
        """
        if self.status != ProjectStatus.ACTIVE:
            raise PermissionError(
                f"No se pueden agregar Tasks a un proyecto '{self.status.value}'. "
                f"El proyecto '{self.name}' debe estar ACTIVE."
            )

        # Verificamos consistencia: la task debe referenciar este proyecto
        if task.project_id is not None and task.project_id != self.id:
            raise ValueError(
                f"La Task '{task.id}' pertenece al proyecto '{task.project_id}', "
                f"no a '{self.id}'."
            )

        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            description=self.description,
            status=self.status,
            tags=self.tags,
            tasks=(*self.tasks, task),
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def update_task(self, updated_task: Task) -> Project:
        """
        Reemplaza una Task por su versión actualizada.

        Se llama durante el loop ReAct para reflejar el progreso
        de la task activa en el estado del proyecto.

        Raises:
            KeyError: Si la Task no existe en este proyecto.
        """
        found = False
        updated_tasks = []

        for task in self.tasks:
            if task.id == updated_task.id:
                updated_tasks.append(updated_task)
                found = True
            else:
                updated_tasks.append(task)

        if not found:
            raise KeyError(
                f"Task con id '{updated_task.id}' no encontrada "
                f"en Project('{self.name}')"
            )

        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            description=self.description,
            status=self.status,
            tags=self.tags,
            tasks=tuple(updated_tasks),
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def rename(self, new_name: str) -> Project:
        """
        Cambia el nombre del proyecto.

        El caller es responsable de verificar unicidad dentro del workspace
        antes de llamar este método.
        """
        if not new_name.strip():
            raise ValueError("El nombre del proyecto no puede estar vacío.")

        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=new_name.strip(),
            description=self.description,
            status=self.status,
            tags=self.tags,
            tasks=self.tasks,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def update_description(self, description: str | None) -> Project:
        """Actualiza la descripción del proyecto."""
        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            description=description,
            status=self.status,
            tags=self.tags,
            tasks=self.tasks,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def update_tags(self, tags: list[str]) -> Project:
        """Reemplaza las etiquetas del proyecto."""
        return Project(
            id=self.id,
            workspace_id=self.workspace_id,
            name=self.name,
            description=self.description,
            status=self.status,
            tags=tuple(tags),
            tasks=self.tasks,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True si el proyecto puede recibir nuevas Tasks."""
        return self.status == ProjectStatus.ACTIVE

    @property
    def task_count(self) -> int:
        """Número total de Tasks en el proyecto."""
        return len(self.tasks)

    @property
    def completed_tasks(self) -> tuple[Task, ...]:
        """Tasks que terminaron exitosamente."""
        return tuple(t for t in self.tasks if t.status == TaskStatus.COMPLETED)

    @property
    def active_tasks(self) -> tuple[Task, ...]:
        """Tasks que están siendo ejecutadas en este momento."""
        return tuple(
            t for t in self.tasks
            if t.status in {TaskStatus.PLANNING, TaskStatus.IN_PROGRESS}
        )

    @property
    def failed_tasks(self) -> tuple[Task, ...]:
        """Tasks que fallaron."""
        return tuple(t for t in self.tasks if t.status == TaskStatus.FAILED)

    @property
    def completed_ratio(self) -> float:
        """
        Ratio de Tasks completadas sobre el total.
        0.0 si no hay tasks. 1.0 si todas están completadas.
        """
        if not self.tasks:
            return 0.0
        return round(len(self.completed_tasks) / self.task_count, 2)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa para guardar en JSON.

        Nota: las Tasks se guardan como lista de IDs, no como objetos completos.
        Cada Task tiene su propio archivo JSON en tasks/{task_id}/task.json.
        Guardar las Tasks completas aquí duplicaría los datos.
        """
        return {
            "id": self.id,
            "workspace_id": self.workspace_id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            "tags": list(self.tags),
            # Solo guardamos IDs — los datos completos están en archivos separados
            "task_ids": [t.id for t in self.tasks],
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any], tasks: list[Task] | None = None) -> Project:
        """
        Reconstruye desde un diccionario leído del JSON.

        Parámetros:
            data:  Diccionario del project.json.
            tasks: Lista de Tasks ya cargadas desde sus archivos individuales.
                   None o lista vacía si solo necesitamos los metadatos del proyecto.
        """
        return cls(
            id=data["id"],
            workspace_id=data["workspace_id"],
            name=data["name"],
            description=data.get("description"),
            status=ProjectStatus(data["status"]),
            tags=tuple(data.get("tags", [])),
            tasks=tuple(tasks or []),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo: Project('API de pagos', ACTIVE, 3 tasks, 66% completado)
        """
        return (
            f"Project({self.name!r}, {self.status.value},"
            f" {self.task_count} tasks, {int(self.completed_ratio * 100)}% completado)"
        )