"""
MemoryStore — Fachada unificada de acceso a datos.

En vez de inyectar WorkspaceRepository, ProjectRepository y TaskRepository
por separado en cada use case, MemoryStore los agrupa en una sola clase.

¿POR QUÉ UNA FACHADA?
  Sin fachada, cada use case necesita 3-4 dependencias:
    class CreateProjectUseCase:
        def __init__(
            self,
            workspace_repo: WorkspaceRepository,
            project_repo: ProjectRepository,
            preferences_repo: PreferencesRepository,
        ):

  Con fachada, una sola dependencia lo da todo:
    class CreateProjectUseCase:
        def __init__(self, store: MemoryStore):

  El store es el único punto de entrada a todos los datos persistentes.
  container.py lo construye una vez y lo inyecta donde se necesite.

USO:
  store = MemoryStore(storage, locator)

  # Workspace
  ws = store.workspaces.find_by_id("01HX...")
  store.workspaces.save(ws)

  # Project
  project = store.projects.find_by_id(workspace_id, project_id)

  # Task
  task = store.tasks.find_by_id(workspace_id, project_id, task_id)

  # Preferences con cascada automática
  prefs = store.get_effective_preferences(workspace_id)
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.repositories.project_repo import ProjectRepository
from hiperforge.memory.repositories.task_repo import TaskRepository
from hiperforge.memory.repositories.workspace_repo import WorkspaceRepository
from hiperforge.memory.schemas.preferences import UserPrefsSchema
from hiperforge.memory.repositories.preferences_repo import PreferencesRepository

logger = get_logger(__name__)


class MemoryStore:
    """
    Fachada unificada de acceso a todos los repositorios.

    Punto de entrada único para todas las operaciones de datos.
    Los use cases solo dependen de MemoryStore — nunca de los
    repositorios individuales directamente.

    Parámetros:
        storage: JSONStorage para operaciones de disco.
        locator: WorkspaceLocator para resolver rutas.
    """

    def __init__(self, storage: JSONStorage, locator: WorkspaceLocator) -> None:
        self._storage = storage
        self._locator = locator

        # Repositorios individuales — accesibles como propiedades
        self._workspaces = WorkspaceRepository(storage, locator)
        self._projects = ProjectRepository(storage, locator)
        self._tasks = TaskRepository(storage, locator)
        self._preferences = PreferencesRepository(storage, locator)

    # ------------------------------------------------------------------
    # Acceso a repositorios individuales
    # ------------------------------------------------------------------

    @property
    def workspaces(self) -> WorkspaceRepository:
        """Repositorio de workspaces."""
        return self._workspaces

    @property
    def projects(self) -> ProjectRepository:
        """Repositorio de projects."""
        return self._projects

    @property
    def tasks(self) -> TaskRepository:
        """Repositorio de tasks."""
        return self._tasks

    @property
    def preferences(self) -> PreferencesRepository:
        """Repositorio de preferencias."""
        return self._preferences

    # ------------------------------------------------------------------
    # Operaciones de conveniencia que cruzan repositorios
    # ------------------------------------------------------------------

    def get_active_workspace_id(self) -> str | None:
        """
        Devuelve el ID del workspace activo.

        Shortcut para store.workspaces.get_active_workspace_id().
        Usado frecuentemente en los use cases.
        """
        return self._workspaces.get_active_workspace_id()

    def get_effective_preferences(
        self,
        workspace_id: str | None = None,
    ) -> UserPrefsSchema:
        """
        Devuelve las preferencias efectivas con cascada aplicada.

        Si workspace_id es None, devuelve solo las preferencias globales.
        Si workspace_id está presente, combina globales + workspace.

        Cascada: globales → workspace (workspace sobreescribe globales)

        Parámetros:
            workspace_id: ID del workspace activo. None para globales.

        Returns:
            UserPrefsSchema con la configuración efectiva.
        """
        global_prefs = self._preferences.load_global()

        if workspace_id is None:
            return global_prefs

        workspace_prefs = self._preferences.load_for_workspace(workspace_id)

        if workspace_prefs is None:
            return global_prefs

        # Combinamos: las preferencias del workspace sobreescriben las globales
        return global_prefs.merge_with(workspace_prefs)

    def __repr__(self) -> str:
        return f"MemoryStore(app_dir={self._locator.app_dir})"
