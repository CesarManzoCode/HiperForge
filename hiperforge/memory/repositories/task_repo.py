"""
TaskRepository — CRUD de tasks con migraciones automáticas.
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import Task
from hiperforge.domain.exceptions import EntityNotFound
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.migrations import migrate_task
from hiperforge.memory.schemas.project import TaskSchema

logger = get_logger(__name__)


class TaskRepository:
    """
    Acceso a datos de tasks con migraciones automáticas.
    """

    def __init__(self, storage: JSONStorage, locator: WorkspaceLocator) -> None:
        self._storage = storage
        self._locator = locator

    def save(self, task: Task, workspace_id: str) -> None:
        """
        Persiste una task completa en disco.

        Parámetros:
            task:         Task a persistir.
            workspace_id: ID del workspace que contiene el proyecto de la task.
        """
        if task.project_id is None:
            raise ValueError(
                f"Task '{task.id}' no tiene project_id — no se puede guardar"
            )

        path = self._locator.task_file(workspace_id, task.project_id, task.id)
        self._storage.write_json(path, task.to_dict())

        logger.debug(
            "task guardada",
            task_id=task.id,
            project_id=task.project_id,
            status=task.status.value,
        )

    def find_by_id(
        self,
        workspace_id: str,
        project_id: str,
        task_id: str,
    ) -> Task:
        """
        Carga una task completa con subtasks y tool calls.

        Aplica migraciones si el schema está desactualizado.

        Raises:
            EntityNotFound: Si la task no existe.
        """
        if not self._locator.task_exists(workspace_id, project_id, task_id):
            raise EntityNotFound(
                entity_type="Task",
                entity_id=task_id,
            )

        path = self._locator.task_file(workspace_id, project_id, task_id)
        raw_data = self._storage.read_json(path)

        migrated_data = migrate_task(raw_data)

        # Guardamos si se migró
        if migrated_data.get("schema_version") != raw_data.get("schema_version"):
            self._storage.write_json(path, migrated_data)

        # Validamos con schema Pydantic
        TaskSchema.model_validate(migrated_data)

        return Task.from_dict(migrated_data)

    def find_all(self, workspace_id: str, project_id: str) -> list[Task]:
        """Carga todas las tasks de un project."""
        task_ids = self._locator.list_task_ids(workspace_id, project_id)
        tasks = []

        for task_id in task_ids:
            try:
                task = self.find_by_id(workspace_id, project_id, task_id)
                tasks.append(task)
            except Exception as exc:
                logger.warning(
                    "error cargando task — omitiendo",
                    task_id=task_id,
                    error=str(exc),
                )

        return tasks

    def delete(self, workspace_id: str, project_id: str, task_id: str) -> None:
        """Eliminación física de la task del disco."""
        self._storage.delete_task(project_id, task_id)
        logger.info(
            "task eliminada del disco",
            task_id=task_id,
            project_id=project_id,
        )
