"""
WorkspaceRepository — CRUD de workspaces con migraciones automáticas.

El repositorio es la capa entre los schemas JSON y las entidades del dominio.
Su responsabilidad es:
  1. Cargar JSON → aplicar migraciones → validar con schema → construir entidad
  2. Serializar entidad → validar con schema → guardar JSON

Los use cases nunca tocan JSONStorage directamente — van por el repositorio
que garantiza que los datos siempre estén migrados y válidos.
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.project import Project
from hiperforge.domain.entities.workspace import Workspace, WorkspaceStatus
from hiperforge.domain.exceptions import (
    DuplicateEntity,
    EntityNotFound,
)
from hiperforge.memory.migrations import migrate_workspace
from hiperforge.memory.schemas.workspace import WorkspaceIndexSchema, WorkspaceSchema
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator

logger = get_logger(__name__)


class WorkspaceRepository:
    """
    Acceso a datos de workspaces con migraciones automáticas.

    Parámetros:
        storage: JSONStorage para operaciones atómicas en disco.
        locator: WorkspaceLocator para resolver rutas.
    """

    def __init__(self, storage: JSONStorage, locator: WorkspaceLocator) -> None:
        self._storage = storage
        self._locator = locator

    # ------------------------------------------------------------------
    # Índice global
    # ------------------------------------------------------------------

    def load_index(self) -> WorkspaceIndexSchema:
        """
        Carga el índice global de workspaces.

        Si el archivo no existe (primera ejecución), devuelve
        un índice vacío sin lanzar error.
        """
        index_path = self._locator.index_file

        if not index_path.exists():
            return WorkspaceIndexSchema()

        data = self._storage.read_json(index_path)
        return WorkspaceIndexSchema.model_validate(data)

    def save_index(self, index: WorkspaceIndexSchema) -> None:
        """Persiste el índice global."""
        index_path = self._locator.index_file
        self._storage.write_json(index_path, index.model_dump())

    def get_active_workspace_id(self) -> str | None:
        """Devuelve el ID del workspace activo. None si no hay ninguno."""
        return self.load_index().active_workspace_id

    def set_active_workspace(self, workspace_id: str) -> None:
        """
        Cambia el workspace activo.

        Raises:
            EntityNotFound: Si el workspace no existe en disco.
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        index = self.load_index()
        updated_index = index.set_active(workspace_id)
        self.save_index(updated_index)

        logger.info(
            "workspace activo cambiado",
            workspace_id=workspace_id,
        )

    # ------------------------------------------------------------------
    # CRUD de Workspaces
    # ------------------------------------------------------------------

    def save(self, workspace: Workspace) -> None:
        """
        Persiste un workspace y actualiza el índice global.

        Si es un workspace nuevo, lo agrega al índice.
        Si ya existe, actualiza sus datos.
        """
        # Guardamos el workspace.json
        self._storage.save_workspace(workspace)

        # Actualizamos el índice global con este workspace_id
        index = self.load_index()
        updated_index = index.add_workspace(workspace.id)
        self.save_index(updated_index)

        logger.debug(
            "workspace guardado",
            workspace_id=workspace.id,
            name=workspace.name,
        )

    def find_by_id(self, workspace_id: str) -> Workspace:
        """
        Carga un workspace completo con todos sus proyectos.

        Aplica migraciones automáticamente si el schema es viejo.

        Raises:
            EntityNotFound: Si el workspace no existe.
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        path = self._locator.workspace_file(workspace_id)
        raw_data = self._storage.read_json(path)

        # Migramos si el schema está desactualizado
        migrated_data = migrate_workspace(raw_data)

        # Si se migró, guardamos la versión actualizada
        if migrated_data.get("schema_version") != raw_data.get("schema_version"):
            self._storage.write_json(path, migrated_data)

        # Validamos con Pydantic antes de construir la entidad
        schema = WorkspaceSchema.model_validate(migrated_data)

        # Cargamos los proyectos referenciados
        projects = self._load_projects_for_workspace(workspace_id, schema.project_ids)

        return Workspace.from_dict(migrated_data, projects=projects)

    def find_by_id_meta(self, workspace_id: str) -> Workspace:
        """
        Carga solo los metadatos del workspace sin proyectos.

        Más rápido que find_by_id() para operaciones que no
        necesitan los datos completos.
        """
        if not self._locator.workspace_exists(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        path = self._locator.workspace_file(workspace_id)
        raw_data = self._storage.read_json(path)
        migrated_data = migrate_workspace(raw_data)

        WorkspaceSchema.model_validate(migrated_data)  # validación

        return Workspace.from_dict(migrated_data, projects=[])

    def find_all(self) -> list[Workspace]:
        """
        Carga todos los workspaces en modo ligero (sin proyectos).

        Usado por la CLI para listar workspaces disponibles.
        """
        workspace_ids = self._locator.list_workspace_ids()
        workspaces = []

        for workspace_id in workspace_ids:
            try:
                ws = self.find_by_id_meta(workspace_id)
                # Excluimos workspaces eliminados de la lista
                if ws.status != WorkspaceStatus.DELETED:
                    workspaces.append(ws)
            except Exception as exc:
                logger.warning(
                    "error cargando workspace — omitiendo",
                    workspace_id=workspace_id,
                    error=str(exc),
                )

        return workspaces

    def exists_by_name(self, name: str) -> bool:
        """
        Verifica si existe un workspace con ese nombre (case-insensitive).

        Usado para validar unicidad antes de crear un workspace nuevo.
        """
        name_lower = name.lower().strip()

        for workspace_id in self._locator.list_workspace_ids():
            try:
                ws = self.find_by_id_meta(workspace_id)
                if (
                    ws.name.lower() == name_lower
                    and ws.status != WorkspaceStatus.DELETED
                ):
                    return True
            except Exception:
                continue

        return False

    def delete(self, workspace_id: str) -> None:
        """
        Eliminación física del workspace del disco.

        Llamar solo cuando workspace.status == DELETED.
        También lo elimina del índice global.
        """
        self._storage.delete_workspace(workspace_id)

        # Actualizamos el índice removiendo el workspace eliminado
        index = self.load_index()
        updated_index = index.remove_workspace(workspace_id)
        self.save_index(updated_index)

        logger.info("workspace eliminado del disco e índice", workspace_id=workspace_id)

    # ------------------------------------------------------------------
    # Helper privado
    # ------------------------------------------------------------------

    def _load_projects_for_workspace(
        self,
        workspace_id: str,
        project_ids: list[str],
    ) -> list[Project]:
        """
        Carga los proyectos de un workspace desde sus project_ids.

        Omite proyectos que no existen en disco con un warning.
        """
        from hiperforge.memory.repositories.project_repo import ProjectRepository

        project_repo = ProjectRepository(self._storage, self._locator)
        projects = []

        for project_id in project_ids:
            try:
                project = project_repo.find_by_id(workspace_id, project_id)
                projects.append(project)
            except EntityNotFound:
                logger.warning(
                    "project_id en workspace.json no encontrado en disco",
                    workspace_id=workspace_id,
                    project_id=project_id,
                )

        return projects
