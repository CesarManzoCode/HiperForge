"""
JSONStorage — Implementación de StoragePort usando archivos JSON en disco.

Este es el adaptador concreto que implementa el contrato StoragePort
del dominio usando el sistema de archivos con archivos JSON.

RESPONSABILIDADES:
  1. Saber DÓNDE guardar cada entidad (delegado a WorkspaceLocator)
  2. Saber CÓMO serializar/deserializar cada entidad (to_dict/from_dict)
  3. Garantizar escrituras atómicas (delegado a BaseStorage)
  4. Manejar el índice global de workspaces activos

VALIDACIÓN AL ESCRIBIR, LECTURA RÁPIDA:
  - Al guardar: validamos que la entidad tiene todos los campos requeridos
    antes de escribir al disco. Si el dict está incompleto, fallamos rápido.
  - Al cargar: lectura directa sin re-validar. Confiamos en que lo que
    escribimos fue válido. Si está corrupto, BaseStorage lo detectará
    via checksum.

CARGA LAZY DE ENTIDADES:
  La jerarquía es Workspace → Project → Task → Subtask.
  Cargar un Workspace completo (con todos sus proyectos y todas sus tasks)
  puede ser costoso. JSONStorage soporta dos modos:

  Modo completo:   load_workspace() carga toda la jerarquía
  Modo ligero:     load_workspace_meta() carga solo workspace.json
                   sin los proyectos ni tasks

  El caller elige según lo que necesita.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.project import Project
from hiperforge.domain.entities.task import Task
from hiperforge.domain.entities.workspace import Workspace
from hiperforge.domain.exceptions import (
    EntityNotFound,
    StorageCorruptedError,
    StorageReadError,
    StorageWriteError,
)
from hiperforge.domain.ports.storage_port import StoragePort
from hiperforge.infrastructure.storage.base import BaseStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator

logger = get_logger(__name__)


class JSONStorage(StoragePort, BaseStorage):
    """
    Implementación de StoragePort con archivos JSON en disco.

    Hereda de:
      StoragePort  → contrato del dominio (qué operaciones existen)
      BaseStorage  → infraestructura (cómo escribir/leer de forma segura)

    Parámetros:
        locator: WorkspaceLocator que resuelve las rutas en disco.
                 Inyectado para permitir tests con directorios temporales.
    """

    def __init__(self, locator: WorkspaceLocator) -> None:
        BaseStorage.__init__(self)
        self._locator = locator

        # Aseguramos que los directorios base existen al instanciar
        self._locator.ensure_app_dirs()

        logger.debug(
            "JSONStorage inicializado",
            app_dir=str(self._locator.app_dir),
        )

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def save_workspace(self, workspace: Workspace) -> None:
        """
        Persiste un Workspace en disco.

        Guarda SOLO los metadatos del workspace (workspace.json).
        Los proyectos se guardan por separado con save_project().

        El workspace.json contiene:
          - Metadatos: id, name, description, status, tags
          - Lista de project_ids (no los proyectos completos)
          - schema_version para migraciones futuras
        """
        path = self._locator.workspace_file(workspace.id)

        self.write_json(path, workspace.to_dict())

        logger.info(
            "workspace guardado",
            workspace_id=workspace.id,
            workspace_name=workspace.name,
            path=str(path),
        )

    def load_workspace(self, workspace_id: str) -> Workspace:
        """
        Carga un Workspace completo con todos sus Projects y Tasks.

        FLUJO:
          1. Leer workspace.json → obtener lista de project_ids
          2. Para cada project_id → load_project()
          3. Reconstruir Workspace con la lista de Projects cargados

        Para carga ligera (sin projects), usar load_workspace_meta().
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        path = self._locator.workspace_file(workspace_id)
        data = self._safe_read(path)

        # Cargamos los proyectos referenciados en project_ids
        project_ids = data.get("project_ids", [])
        projects: list[Project] = []

        for project_id in project_ids:
            try:
                project = self.load_project(workspace_id, project_id)
                projects.append(project)
            except EntityNotFound:
                # El proyecto fue eliminado del disco pero sigue en el índice
                # Loggeamos el inconsistencia pero no fallamos — el workspace
                # sigue siendo válido sin ese proyecto
                logger.warning(
                    "project_id en workspace.json no encontrado en disco",
                    workspace_id=workspace_id,
                    project_id=project_id,
                )

        workspace = Workspace.from_dict(data, projects=projects)

        logger.debug(
            "workspace cargado",
            workspace_id=workspace_id,
            project_count=len(projects),
        )

        return workspace

    def load_workspace_meta(self, workspace_id: str) -> Workspace:
        """
        Carga solo los metadatos del workspace sin proyectos ni tasks.

        Más rápido que load_workspace() cuando solo necesitamos
        el nombre, estado o schema_version del workspace.

        Útil para: listar workspaces, verificar schema_version antes
        de migrar, mostrar resumen en la CLI.
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        path = self._locator.workspace_file(workspace_id)
        data = self._safe_read(path)

        # from_dict sin projects → carga solo metadatos
        return Workspace.from_dict(data, projects=[])

    def workspace_exists(self, workspace_id: str) -> bool:
        return self._locator.workspace_exists(workspace_id)

    def delete_workspace(self, workspace_id: str) -> None:
        """
        Elimina FÍSICAMENTE todos los archivos de un workspace.

        ATENCIÓN: operación destructiva e irreversible.
        El caller debe verificar que workspace.status == DELETED
        antes de llamar este método.
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        workspace_dir = self._locator.workspace_dir(workspace_id)
        self.delete_path(workspace_dir)

        # Eliminamos también el lock file si existe
        lock_file = self._locator.workspace_lock_file(workspace_id)
        if lock_file.exists():
            self.delete_path(lock_file)

        logger.info(
            "workspace eliminado del disco",
            workspace_id=workspace_id,
            path=str(workspace_dir),
        )

    def list_workspace_ids(self) -> list[str]:
        return self._locator.list_workspace_ids()

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    def save_project(self, project: Project) -> None:
        """
        Persiste un Project en disco.

        Guarda SOLO los metadatos del proyecto (project.json).
        Las tasks se guardan por separado con save_task().

        Resuelve workspace_id desde project.workspace_id.
        """
        path = self._locator.project_file(
            workspace_id=project.workspace_id,
            project_id=project.id,
        )

        self.write_json(path, project.to_dict())

        logger.debug(
            "project guardado",
            project_id=project.id,
            project_name=project.name,
            workspace_id=project.workspace_id,
        )

    def load_project(self, workspace_id: str, project_id: str) -> Project:
        """
        Carga un Project con todas sus Tasks.

        FLUJO:
          1. Leer project.json → obtener lista de task_ids
          2. Para cada task_id → load_task()
          3. Reconstruir Project con la lista de Tasks cargadas
        """
        if not self._locator.project_exists(workspace_id, project_id):
            raise EntityNotFound(
                entity_type="Project",
                entity_id=project_id,
            )

        path = self._locator.project_file(workspace_id, project_id)
        data = self._safe_read(path)

        # Cargamos las tasks referenciadas
        task_ids = data.get("task_ids", [])
        tasks: list[Task] = []

        for task_id in task_ids:
            try:
                task = self.load_task(project_id, task_id)
                tasks.append(task)
            except EntityNotFound:
                logger.warning(
                    "task_id en project.json no encontrada en disco",
                    project_id=project_id,
                    task_id=task_id,
                )

        return Project.from_dict(data, tasks=tasks)

    def project_exists(self, workspace_id: str, project_id: str) -> bool:
        return self._locator.project_exists(workspace_id, project_id)

    def delete_project(self, workspace_id: str, project_id: str) -> None:
        """Elimina físicamente todos los archivos de un proyecto."""
        if not self._locator.project_exists(workspace_id, project_id):
            raise EntityNotFound(
                entity_type="Project",
                entity_id=project_id,
            )

        project_dir = self._locator.project_dir(workspace_id, project_id)
        self.delete_path(project_dir)

        logger.info(
            "project eliminado del disco",
            project_id=project_id,
            workspace_id=workspace_id,
        )

    def list_project_ids(self, workspace_id: str) -> list[str]:
        return self._locator.list_project_ids(workspace_id)

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def save_task(self, task: Task) -> None:
        """
        Persiste una Task completa en disco.

        A diferencia de workspace y project, la task se guarda COMPLETA
        — incluyendo subtasks y tool calls — en un solo archivo task.json.

        Las subtasks y tool calls no tienen archivos propios porque:
          1. Una subtask no tiene sentido fuera de su task.
          2. El número de subtasks es acotado (max REACT_MAX_SUBTASKS).
          3. Un solo archivo es más fácil de leer/debuggear que
             un directorio con decenas de archivos pequeños.

        Resuelve workspace_id buscando el project en disco.
        """
        # Necesitamos workspace_id para construir la ruta
        # Lo obtenemos del project_id de la task
        workspace_id = self._resolve_workspace_id_for_task(task)

        path = self._locator.task_file(
            workspace_id=workspace_id,
            project_id=task.project_id,
            task_id=task.id,
        )

        self.write_json(path, task.to_dict())

        logger.debug(
            "task guardada",
            task_id=task.id,
            project_id=task.project_id,
            status=task.status.value,
            subtask_count=len(task.subtasks),
        )

    def load_task(self, project_id: str, task_id: str) -> Task:
        """
        Carga una Task completa con subtasks y tool calls.

        Busca la task en todos los workspaces que contengan el project_id.
        """
        workspace_id = self._find_workspace_id_for_project(project_id)

        if workspace_id is None:
            raise EntityNotFound(
                entity_type="Task",
                entity_id=task_id,
            )

        if not self._locator.task_exists(workspace_id, project_id, task_id):
            raise EntityNotFound(
                entity_type="Task",
                entity_id=task_id,
            )

        path = self._locator.task_file(workspace_id, project_id, task_id)
        data = self._safe_read(path)

        return Task.from_dict(data)

    def task_exists(self, project_id: str, task_id: str) -> bool:
        workspace_id = self._find_workspace_id_for_project(project_id)
        if workspace_id is None:
            return False
        return self._locator.task_exists(workspace_id, project_id, task_id)

    def delete_task(self, project_id: str, task_id: str) -> None:
        """Elimina físicamente todos los archivos de una task."""
        workspace_id = self._find_workspace_id_for_project(project_id)

        if workspace_id is None or not self._locator.task_exists(
            workspace_id, project_id, task_id
        ):
            raise EntityNotFound(
                entity_type="Task",
                entity_id=task_id,
            )

        task_dir = self._locator.task_dir(workspace_id, project_id, task_id)
        self.delete_path(task_dir)

        logger.info(
            "task eliminada del disco",
            task_id=task_id,
            project_id=project_id,
        )

    def list_task_ids(self, project_id: str) -> list[str]:
        workspace_id = self._find_workspace_id_for_project(project_id)
        if workspace_id is None:
            return []
        return self._locator.list_task_ids(workspace_id, project_id)

    # ------------------------------------------------------------------
    # Índice global — workspace activo
    # ------------------------------------------------------------------

    def load_active_workspace_id(self) -> str | None:
        """
        Lee el ID del workspace activo desde index.json.

        Devuelve None si es la primera vez que se usa HiperForge
        (el archivo no existe todavía).
        """
        index_path = self._locator.index_file

        if not index_path.exists():
            return None

        try:
            data = self._safe_read(index_path)
            return data.get("active_workspace_id")
        except (StorageReadError, StorageCorruptedError):
            # Si el índice está corrupto, tratamos como primera vez
            logger.warning(
                "index.json corrupto — tratando como primera ejecución",
                path=str(index_path),
            )
            return None

    def save_active_workspace_id(self, workspace_id: str) -> None:
        """
        Guarda el ID del workspace activo en index.json.

        También guarda la lista completa de workspace IDs conocidos
        para que el índice sea autocontenido.
        """
        index_path = self._locator.index_file

        # Leemos el índice existente para no perder datos previos
        existing_data: dict = {}
        if index_path.exists():
            try:
                existing_data = self._safe_read(index_path)
            except (StorageReadError, StorageCorruptedError):
                pass  # índice corrupto — lo recreamos

        # Actualizamos el workspace activo
        existing_data["active_workspace_id"] = workspace_id

        # Mantenemos la lista de todos los workspace IDs conocidos
        known_ids: list[str] = existing_data.get("workspace_ids", [])
        if workspace_id not in known_ids:
            known_ids.append(workspace_id)
        existing_data["workspace_ids"] = sorted(known_ids)

        self.write_json(index_path, existing_data)

        logger.info(
            "workspace activo actualizado",
            workspace_id=workspace_id,
        )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _safe_read(self, path: Path) -> dict[str, Any]:
        """
        Lee un JSON con intento de recuperación desde backup si falla.

        Si read_json() falla por corrupción, intenta restaurar desde .bak
        automáticamente antes de propagar el error.
        """
        try:
            return self.read_json(path)

        except StorageCorruptedError as exc:
            logger.error(
                "archivo JSON corrupto, intentando recuperar desde backup",
                path=str(path),
                error=str(exc),
            )

            # Intentamos restaurar el backup automáticamente
            restored = self.try_restore_from_backup(path)

            if restored:
                logger.info(
                    "recuperación desde backup exitosa",
                    path=str(path),
                )
                # Reintentamos la lectura con el archivo restaurado
                return self.read_json(path)

            # Sin backup válido — propagamos el error original
            raise

    def _resolve_workspace_id_for_task(self, task: Task) -> str:
        """
        Resuelve el workspace_id de una task buscando su project en disco.

        Las tasks pertenecen a un project, y los projects pertenecen
        a un workspace. Navegamos la jerarquía para encontrar el workspace.

        Raises:
            StorageWriteError: Si no se puede determinar el workspace.
        """
        if task.project_id is None:
            raise StorageWriteError(
                path="",
                reason=(
                    f"Task '{task.id}' no tiene project_id — "
                    "no se puede determinar dónde guardarla"
                ),
            )

        workspace_id = self._find_workspace_id_for_project(task.project_id)

        if workspace_id is None:
            raise StorageWriteError(
                path="",
                reason=(
                    f"No se encontró workspace que contenga "
                    f"el project '{task.project_id}'"
                ),
            )

        return workspace_id

    def _find_workspace_id_for_project(self, project_id: str) -> str | None:
        """
        Busca en qué workspace vive un project dado su ID.

        Itera los workspace IDs en disco y verifica si contienen
        el project. Retorna el primero que lo contenga.

        Esta operación es O(n) en número de workspaces — aceptable
        porque los usuarios raramente tienen más de 5-10 workspaces.

        Returns:
            workspace_id si se encontró, None si no existe.
        """
        for workspace_id in self._locator.list_workspace_ids():
            if self._locator.project_exists(workspace_id, project_id):
                return workspace_id

        return None