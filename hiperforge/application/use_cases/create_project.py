"""
CreateProjectUseCase — Crea un nuevo proyecto dentro de un workspace.

Un proyecto agrupa tasks relacionadas bajo un mismo contexto.
Este use case garantiza que el proyecto quede correctamente vinculado
al workspace — tanto en la entidad en memoria como en los archivos
JSON de disco.

FLUJO:
  1. Verificar que el workspace existe y está ACTIVE.
  2. Verificar que no existe otro proyecto con el mismo nombre.
  3. Crear la entidad Project con Project.create().
  4. Vincular el proyecto al workspace con workspace.add_project().
  5. Persistir: project.json + workspace.json actualizado.
  6. Devolver ProjectSummary a la CLI.

¿POR QUÉ PERSISTIR TANTO EL PROJECT COMO EL WORKSPACE?
  El workspace mantiene una lista de sus project_ids.
  Sin actualizar workspace.json, el proyecto existiría en disco
  pero el workspace no sabría que tiene ese proyecto — quedaría
  huérfano e inaccesible para el resto del sistema.

  El orden de escritura importa:
    1. project.json primero   → si falla aquí, no hay inconsistencia
    2. workspace.json después → actualiza el índice cuando el proyecto ya existe

  Si el proceso muere entre 1 y 2, workspace_repo.save() tiene lógica
  para detectar proyectos huérfanos en disco y recuperarlos.
"""

from __future__ import annotations

from hiperforge.application.dto import CreateProjectInput, ProjectSummary
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.project import Project
from hiperforge.domain.entities.workspace import WorkspaceStatus
from hiperforge.domain.exceptions import DuplicateEntity, EntityNotFound
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)


class CreateProjectUseCase:
    """
    Crea un nuevo proyecto dentro de un workspace existente.

    Parámetros:
        store: MemoryStore para acceder a workspaces y proyectos.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def execute(self, input_data: CreateProjectInput) -> ProjectSummary:
        """
        Crea el proyecto, lo vincula al workspace y lo persiste.

        Parámetros:
            input_data: Nombre, workspace_id, descripción y tags del proyecto.

        Returns:
            ProjectSummary con los datos del proyecto recién creado.

        Raises:
            EntityNotFound:  Si el workspace_id no existe en disco.
            PermissionError: Si el workspace está archivado o eliminado.
            DuplicateEntity: Si ya existe un proyecto con ese nombre en el workspace.
        """
        # ── Paso 1: cargar el workspace completo ─────────────────────
        # Necesitamos cargarlo con proyectos para poder verificar unicidad
        # de nombre y vincular el nuevo proyecto via workspace.add_project()
        try:
            workspace = self._store.workspaces.find_by_id(
                input_data.workspace_id
            )
        except EntityNotFound:
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=input_data.workspace_id,
            )

        # ── Paso 2: verificar que el workspace acepta proyectos nuevos ─
        # Workspace.add_project() lo verifica también, pero hacerlo aquí
        # nos permite dar un mensaje de error más descriptivo antes de
        # intentar la operación de dominio
        if workspace.status != WorkspaceStatus.ACTIVE:
            raise PermissionError(
                f"No se pueden crear proyectos en el workspace '{workspace.name}' "
                f"porque está en estado '{workspace.status.value}'. "
                f"Solo los workspaces ACTIVE aceptan proyectos nuevos."
            )

        # ── Paso 3: verificar unicidad del nombre (case-insensitive) ──
        # Workspace.add_project() también lo verifica, pero adelantamos la
        # validación para dar el mensaje más claro posible al usuario
        name_lower = input_data.name.strip().lower()
        existing_names = {
            p.name.lower()
            for p in workspace.projects
            if p.status.value != "deleted"
        }

        if name_lower in existing_names:
            raise DuplicateEntity(
                entity_type="Project",
                identifier=input_data.name,
            )

        # ── Paso 4: crear la entidad Project ──────────────────────────
        project = Project.create(
            name=input_data.name,
            workspace_id=input_data.workspace_id,
            description=input_data.description,
            tags=input_data.tags,
        )

        # ── Paso 5: vincular proyecto al workspace ────────────────────
        # workspace.add_project() devuelve un nuevo workspace inmutable
        # con el proyecto en su lista interna
        updated_workspace = workspace.add_project(project)

        # ── Paso 6: persistir en el orden correcto ────────────────────
        # Primero el project.json, luego workspace.json actualizado
        self._store.projects.save(project)
        self._store.workspaces.save(updated_workspace)

        logger.info(
            "proyecto creado exitosamente",
            project_id=project.id,
            name=project.name,
            workspace_id=project.workspace_id,
            tags=list(project.tags),
        )

        # ── Paso 7: construir y devolver el summary ───────────────────
        return ProjectSummary(
            id=project.id,
            name=project.name,
            description=project.description,
            status=project.status.value,
            tags=list(project.tags),
            task_count=0,
            completed_tasks=0,
            created_at=project.created_at,
            updated_at=project.updated_at,
        )