"""
Entidad: Workspace

El Workspace es el nivel más alto de organización en HiperForge.
Representa un contexto de trabajo completamente aislado.

¿Para qué sirve el multi-workspace?
  Un desarrollador puede tener contextos muy distintos que no deben mezclarse:

  Workspace: "Trabajo"
    ├── Project: "API de pagos"
    └── Project: "Dashboard admin"

  Workspace: "Personal"
    ├── Project: "Blog personal"
    └── Project: "CLI de utilidades"

  Workspace: "Cliente Acme"
    └── Project: "Migración de base de datos"

  Cada workspace tiene sus propias preferencias de LLM, tools habilitadas
  y configuración — completamente independiente de los otros.

¿Cómo funciona el switch entre workspaces?
  El sistema mantiene un "workspace activo" en el index.json global.
  Al hacer `hiperforge workspace switch <id>`, solo se cambia ese puntero.
  Los datos de cada workspace nunca se mezclan.

CICLO DE VIDA:
  (creación) → ACTIVE
  ACTIVE → ARCHIVED   (workspace pausado, solo lectura)
  ACTIVE → DELETED    (eliminación lógica)
  ARCHIVED → ACTIVE   (reactivación)

USO TÍPICO:
  # Crear un workspace nuevo
  ws = Workspace.create(name="Trabajo", description="Proyectos profesionales")

  # Agregar un proyecto
  ws = ws.add_project(project)

  # Obtener el proyecto activo por nombre
  proyecto = ws.get_project_by_name("API de pagos")

  # Archivar cuando ya no se use
  ws = ws.archive()
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.core.utils.ids import generate_id
from hiperforge.domain.entities.project import Project, ProjectStatus


class WorkspaceStatus(str, Enum):
    """
    Estado del ciclo de vida de un Workspace.

    Transiciones válidas → ver _VALID_TRANSITIONS abajo.
    """

    ACTIVE = "active"       # Activo, recibe nuevos Projects y Tasks
    ARCHIVED = "archived"   # Archivado, solo lectura
    DELETED = "deleted"     # Eliminación lógica — irreversible


_VALID_TRANSITIONS: dict[WorkspaceStatus, frozenset[WorkspaceStatus]] = {
    WorkspaceStatus.ACTIVE:   frozenset({WorkspaceStatus.ARCHIVED, WorkspaceStatus.DELETED}),
    WorkspaceStatus.ARCHIVED: frozenset({WorkspaceStatus.ACTIVE, WorkspaceStatus.DELETED}),
    WorkspaceStatus.DELETED:  frozenset(),  # terminal
}


@dataclass(frozen=True)
class Workspace:
    """
    Contexto de trabajo aislado. Nivel raíz de la jerarquía de datos.

    Jerarquía completa:
      Workspace → Project → Task → Subtask → ToolCall

    Atributos:
        id:            Identificador único (ULID).
        name:          Nombre del workspace (único globalmente entre workspaces).
        description:   Descripción opcional del propósito del workspace.
        status:        Estado actual del ciclo de vida.
        projects:      Projects del workspace en orden cronológico.
                       Guardados como tuple para mantener inmutabilidad.
        schema_version: Versión del schema de datos. Usada por migrations.py
                        para detectar si los archivos JSON necesitan migración.
        created_at:    Cuándo fue creado (UTC).
        updated_at:    Última modificación (UTC).
    """

    id: str
    name: str
    description: str | None
    status: WorkspaceStatus
    projects: tuple[Project, ...]
    schema_version: int
    created_at: datetime
    updated_at: datetime

    # Versión actual del schema. Incrementar cuando cambie la estructura
    # de los archivos JSON para que migrations.py pueda detectarlo.
    CURRENT_SCHEMA_VERSION: int = 1

    # ------------------------------------------------------------------
    # Constructor principal
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, name: str, *, description: str | None = None) -> Workspace:
        """
        Crea un nuevo Workspace en estado ACTIVE.

        Parámetros:
            name:        Nombre del workspace. Debe ser único globalmente.
            description: Descripción opcional del propósito del workspace.
        """
        now = datetime.now(timezone.utc)

        return cls(
            id=generate_id(),
            name=name.strip(),
            description=description,
            status=WorkspaceStatus.ACTIVE,
            projects=(),
            schema_version=cls.CURRENT_SCHEMA_VERSION,
            created_at=now,
            updated_at=now,
        )

    # ------------------------------------------------------------------
    # Métodos de transición de estado
    # ------------------------------------------------------------------

    def _transition_to(self, new_status: WorkspaceStatus) -> Workspace:
        """Valida y ejecuta una transición de estado."""
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"Workspace({self.id}, '{self.name}')",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        return Workspace(
            id=self.id,
            name=self.name,
            description=self.description,
            status=new_status,
            projects=self.projects,
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def archive(self) -> Workspace:
        """Archiva el workspace. Sus proyectos quedan en solo lectura."""
        return self._transition_to(WorkspaceStatus.ARCHIVED)

    def reactivate(self) -> Workspace:
        """Reactiva un workspace archivado."""
        return self._transition_to(WorkspaceStatus.ACTIVE)

    def delete(self) -> Workspace:
        """
        Eliminación lógica del workspace.

        Los datos siguen en disco. El repositorio decide
        cuándo limpiar los archivos físicamente.
        """
        return self._transition_to(WorkspaceStatus.DELETED)

    # ------------------------------------------------------------------
    # Métodos de gestión de Projects
    # ------------------------------------------------------------------

    def add_project(self, project: Project) -> Workspace:
        """
        Agrega un Project al workspace.

        Solo se pueden agregar Projects a workspaces ACTIVE.

        Raises:
            PermissionError: Si el workspace no está activo.
            ValueError:      Si el proyecto no pertenece a este workspace.
            DuplicateEntity: Si ya existe un proyecto con ese nombre.
        """
        from hiperforge.domain.exceptions import DuplicateEntity

        if self.status != WorkspaceStatus.ACTIVE:
            raise PermissionError(
                f"No se pueden agregar Projects a un workspace '{self.status.value}'. "
                f"El workspace '{self.name}' debe estar ACTIVE."
            )

        # Verificamos que el proyecto pertenece a este workspace
        if project.workspace_id != self.id:
            raise ValueError(
                f"El Project '{project.id}' pertenece al workspace '{project.workspace_id}', "
                f"no a '{self.id}'."
            )

        # Nombre único dentro del workspace (solo entre proyectos activos/archivados)
        existing_names = {
            p.name.lower()
            for p in self.projects
            if p.status != ProjectStatus.DELETED
        }
        if project.name.lower() in existing_names:
            raise DuplicateEntity(entity_type="Project", identifier=project.name)

        return Workspace(
            id=self.id,
            name=self.name,
            description=self.description,
            status=self.status,
            projects=(*self.projects, project),
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def update_project(self, updated_project: Project) -> Workspace:
        """
        Reemplaza un Project por su versión actualizada.

        Se llama cuando un proyecto cambia estado o se le agregan Tasks.

        Raises:
            KeyError: Si el Project no existe en este workspace.
        """
        found = False
        updated_projects = []

        for project in self.projects:
            if project.id == updated_project.id:
                updated_projects.append(updated_project)
                found = True
            else:
                updated_projects.append(project)

        if not found:
            raise KeyError(
                f"Project con id '{updated_project.id}' no encontrado "
                f"en Workspace('{self.name}')"
            )

        return Workspace(
            id=self.id,
            name=self.name,
            description=self.description,
            status=self.status,
            projects=tuple(updated_projects),
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def rename(self, new_name: str) -> Workspace:
        """
        Cambia el nombre del workspace.

        El caller es responsable de verificar unicidad global
        antes de llamar este método.
        """
        if not new_name.strip():
            raise ValueError("El nombre del workspace no puede estar vacío.")

        return Workspace(
            id=self.id,
            name=new_name.strip(),
            description=self.description,
            status=self.status,
            projects=self.projects,
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    def update_description(self, description: str | None) -> Workspace:
        """Actualiza la descripción del workspace."""
        return Workspace(
            id=self.id,
            name=self.name,
            description=description,
            status=self.status,
            projects=self.projects,
            schema_version=self.schema_version,
            created_at=self.created_at,
            updated_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Métodos de consulta
    # ------------------------------------------------------------------

    def get_project_by_id(self, project_id: str) -> Project:
        """
        Busca un proyecto por su ID.

        Raises:
            EntityNotFound: Si no existe ningún proyecto con ese ID.
        """
        from hiperforge.domain.exceptions import EntityNotFound

        for project in self.projects:
            if project.id == project_id:
                return project

        raise EntityNotFound(entity_type="Project", entity_id=project_id)

    def get_project_by_name(self, name: str) -> Project | None:
        """
        Busca un proyecto por su nombre (case-insensitive).

        Devuelve None si no existe — no lanza excepción porque
        "buscar por nombre" no garantiza existencia como buscar por ID.
        """
        name_lower = name.lower()
        for project in self.projects:
            if project.name.lower() == name_lower and project.status != ProjectStatus.DELETED:
                return project
        return None

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_active(self) -> bool:
        """True si el workspace puede recibir nuevos Projects."""
        return self.status == WorkspaceStatus.ACTIVE

    @property
    def active_projects(self) -> tuple[Project, ...]:
        """Projects que están activos (no archivados ni eliminados)."""
        return tuple(p for p in self.projects if p.status == ProjectStatus.ACTIVE)

    @property
    def project_count(self) -> int:
        """Número total de Projects (incluye archivados, excluye eliminados)."""
        return sum(1 for p in self.projects if p.status != ProjectStatus.DELETED)

    @property
    def needs_migration(self) -> bool:
        """
        True si el schema de este workspace está desactualizado.

        migrations.py usa esta propiedad para saber si debe migrar
        los archivos JSON antes de cargarlos.
        """
        return self.schema_version < self.CURRENT_SCHEMA_VERSION

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa para guardar en workspace.json.

        Al igual que Project, los Projects se guardan como IDs.
        Cada Project tiene su propio directorio y archivo JSON.
        """
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value,
            # Solo IDs — los datos completos están en projects/{project_id}/project.json
            "project_ids": [p.id for p in self.projects],
            "schema_version": self.schema_version,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        projects: list[Project] | None = None,
    ) -> Workspace:
        """
        Reconstruye desde un diccionario leído del workspace.json.

        Parámetros:
            data:     Diccionario del workspace.json.
            projects: Lista de Projects ya cargados desde sus archivos individuales.
                      None o lista vacía si solo necesitamos los metadatos.
        """
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
            status=WorkspaceStatus(data["status"]),
            projects=tuple(projects or []),
            schema_version=data.get("schema_version", 1),
            created_at=datetime.fromisoformat(data["created_at"]),
            updated_at=datetime.fromisoformat(data["updated_at"]),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo: Workspace('Trabajo', ACTIVE, 3 projects) [v1]
        """
        return (
            f"Workspace({self.name!r}, {self.status.value},"
            f" {self.project_count} projects) [v{self.schema_version}]"
        )
