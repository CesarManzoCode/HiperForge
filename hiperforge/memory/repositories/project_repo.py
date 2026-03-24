"""
ProjectRepository — CRUD de projects con migraciones automáticas.
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.project import Project, ProjectStatus
from hiperforge.domain.entities.task import Task
from hiperforge.domain.exceptions import EntityNotFound
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.migrations import migrate_project
from hiperforge.memory.schemas.project import ProjectSchema

logger = get_logger(__name__)


class ProjectRepository:
    """
    Acceso a datos de projects con migraciones automáticas.
    """

    def __init__(self, storage: JSONStorage, locator: WorkspaceLocator) -> None:
        self._storage = storage
        self._locator = locator

    def save(self, project: Project) -> None:
        """Persiste un project en disco."""
        self._storage.save_project(project)

    def find_by_id(self, workspace_id: str, project_id: str) -> Project:
        """
        Carga un project completo con todas sus tasks.

        Aplica migraciones si el schema está desactualizado.

        Raises:
            EntityNotFound: Si el project no existe.
        """
        if not self._locator.project_exists(workspace_id, project_id):
            raise EntityNotFound(
                entity_type="Project",
                entity_id=project_id,
            )

        path = self._locator.project_file(workspace_id, project_id)
        raw_data = self._storage.read_json(path)

        migrated_data = migrate_project(raw_data)

        # Guardamos si se migró
        if migrated_data.get("schema_version") != raw_data.get("schema_version"):
            self._storage.write_json(path, migrated_data)

        # Validamos con schema Pydantic
        ProjectSchema.model_validate(migrated_data)

        # Cargamos tasks referenciadas
        tasks = self._load_tasks_for_project(workspace_id, project_id, migrated_data.get("task_ids", []))

        return Project.from_dict(migrated_data, tasks=tasks)

    def find_by_id_meta(self, workspace_id: str, project_id: str) -> Project:
        """Carga solo metadatos del project sin tasks."""
        if not self._locator.project_exists(workspace_id, project_id):
            raise EntityNotFound(
                entity_type="Project",
                entity_id=project_id,
            )

        path = self._locator.project_file(workspace_id, project_id)
        raw_data = self._storage.read_json(path)
        migrated_data = migrate_project(raw_data)

        ProjectSchema.model_validate(migrated_data)

        return Project.from_dict(migrated_data, tasks=[])

    def find_all(self, workspace_id: str) -> list[Project]:
        """
        Carga todos los projects de un workspace en modo ligero.

        Excluye projects eliminados.
        """
        project_ids = self._locator.list_project_ids(workspace_id)
        projects = []

        for project_id in project_ids:
            try:
                project = self.find_by_id_meta(workspace_id, project_id)
                if project.status != ProjectStatus.DELETED:
                    projects.append(project)
            except Exception as exc:
                logger.warning(
                    "error cargando project — omitiendo",
                    project_id=project_id,
                    error=str(exc),
                )

        return projects

    def delete(self, workspace_id: str, project_id: str) -> None:
        """Eliminación física del project del disco."""
        self._storage.delete_project(workspace_id, project_id)
        logger.info(
            "project eliminado del disco",
            project_id=project_id,
            workspace_id=workspace_id,
        )

    def _load_tasks_for_project(
        self,
        workspace_id: str,
        project_id: str,
        task_ids: list[str],
    ) -> list[Task]:
        """Carga las tasks de un project desde sus task_ids."""
        from hiperforge.memory.repositories.task_repo import TaskRepository

        task_repo = TaskRepository(self._storage, self._locator)
        tasks = []

        for task_id in task_ids:
            try:
                task = task_repo.find_by_id(workspace_id, project_id, task_id)
                tasks.append(task)
            except EntityNotFound:
                logger.warning(
                    "task_id en project.json no encontrada en disco",
                    project_id=project_id,
                    task_id=task_id,
                )

        return tasks
