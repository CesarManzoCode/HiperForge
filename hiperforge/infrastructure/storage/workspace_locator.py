"""
WorkspaceLocator — Resuelve rutas del sistema de archivos.

Este módulo es el único lugar del proyecto donde se construyen
rutas de archivos. Ningún otro módulo hardcodea rutas — todos
piden la ruta al WorkspaceLocator.

¿POR QUÉ UN LOCATOR DEDICADO?
  Sin locator, las rutas estarían dispersas por todo el código:
    json_storage.py:    path = APP_DIR / "workspaces" / workspace_id / "projects" / ...
    workspace_repo.py:  path = Path.home() / ".hiperforge" / "workspaces" / ...
    migrations.py:      path = settings.app_dir / "workspaces" / ...

  Con locator, hay un único punto de verdad:
    locator.workspace_dir(workspace_id)
    locator.project_dir(workspace_id, project_id)
    locator.task_file(workspace_id, project_id, task_id)

  Si alguna vez cambia la estructura de directorios, se modifica
  solo este archivo y todo el resto funciona sin cambios.

ESTRUCTURA DE DIRECTORIOS QUE RESUELVE:
  ~/.hiperforge/
  ├── index.json                                    ← index_file()
  ├── preferences.json                              ← global_preferences_file()
  ├── workspaces/
  │   └── {workspace_id}/                           ← workspace_dir()
  │       ├── workspace.json                        ← workspace_file()
  │       ├── preferences.json                      ← workspace_preferences_file()
  │       ├── projects/
  │       │   └── {project_id}/                     ← project_dir()
  │       │       ├── project.json                  ← project_file()
  │       │       └── tasks/
  │       │           └── {task_id}/                ← task_dir()
  │       │               └── task.json             ← task_file()
  │       └── sessions/
  │           └── {session_id}.json                 ← session_file()
  ├── logs/
  │   └── hiperforge.log                            ← log_file()
  └── locks/
      └── {workspace_id}.lock                       ← workspace_lock_file()
"""

from __future__ import annotations

from pathlib import Path

from hiperforge.core.config import get_settings
from hiperforge.core.constants import (
    DIR_LOCKS,
    DIR_LOGS,
    DIR_WORKSPACES,
    FILENAME_PREFERENCES,
    FILENAME_PROJECT,
    FILENAME_SESSION,
    FILENAME_TASK,
    FILENAME_WORKSPACE,
    FILE_GLOBAL_PREFERENCES,
    FILE_INDEX,
    LOG_FILENAME,
    STORAGE_LOCK_EXTENSION,
)


class WorkspaceLocator:
    """
    Resuelve rutas absolutas para todos los archivos del sistema.

    Instanciar una vez en container.py e inyectar donde se necesite.
    Todos los métodos son puros — dado el mismo input, siempre
    devuelven el mismo output. Sin estado mutable.

    Parámetros:
        app_dir: Directorio raíz de datos. Default: settings.app_dir
                 (~/.hiperforge/ o el valor de HIPERFORGE_APP_DIR).
                 Sobreescribir en tests para usar un directorio temporal.
    """

    def __init__(self, app_dir: Path | None = None) -> None:
        # Permitimos inyectar app_dir para tests — sin esto los tests
        # escribirían en ~/.hiperforge/ del desarrollador
        self._app_dir = app_dir or get_settings().app_dir

    # ------------------------------------------------------------------
    # Directorio raíz y archivos globales
    # ------------------------------------------------------------------

    @property
    def app_dir(self) -> Path:
        """Directorio raíz de datos (~/.hiperforge/)."""
        return self._app_dir

    @property
    def index_file(self) -> Path:
        """~/.hiperforge/index.json — índice global de workspaces."""
        return self._app_dir / FILE_INDEX.name

    @property
    def global_preferences_file(self) -> Path:
        """~/.hiperforge/preferences.json — preferencias globales del usuario."""
        return self._app_dir / FILE_GLOBAL_PREFERENCES.name

    @property
    def workspaces_dir(self) -> Path:
        """~/.hiperforge/workspaces/"""
        return self._app_dir / DIR_WORKSPACES.name

    @property
    def logs_dir(self) -> Path:
        """~/.hiperforge/logs/"""
        return self._app_dir / DIR_LOGS.name

    @property
    def locks_dir(self) -> Path:
        """~/.hiperforge/locks/"""
        return self._app_dir / DIR_LOCKS.name

    @property
    def log_file(self) -> Path:
        """~/.hiperforge/logs/hiperforge.log"""
        return self.logs_dir / LOG_FILENAME

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    def workspace_dir(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/

        Directorio raíz de un workspace específico.
        """
        return self.workspaces_dir / workspace_id

    def workspace_file(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/workspace.json
        """
        return self.workspace_dir(workspace_id) / FILENAME_WORKSPACE

    def workspace_preferences_file(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/preferences.json

        Preferencias que sobreescriben las globales para este workspace.
        """
        return self.workspace_dir(workspace_id) / FILENAME_PREFERENCES

    def workspace_lock_file(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/locks/{workspace_id}.lock

        File lock para escrituras concurrentes en este workspace.
        Vive en el directorio de locks (fuera del workspace) para
        poder bloquearlo incluso cuando el workspace_dir no existe.
        """
        return self.locks_dir / f"{workspace_id}{STORAGE_LOCK_EXTENSION}"

    def workspace_sessions_dir(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/sessions/
        """
        return self.workspace_dir(workspace_id) / "sessions"

    def session_file(self, workspace_id: str, session_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/sessions/{session_id}.json
        """
        return self.workspace_sessions_dir(workspace_id) / f"{session_id}.json"

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    def projects_dir(self, workspace_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/
        """
        return self.workspace_dir(workspace_id) / "projects"

    def project_dir(self, workspace_id: str, project_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/{project_id}/
        """
        return self.projects_dir(workspace_id) / project_id

    def project_file(self, workspace_id: str, project_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/{project_id}/project.json
        """
        return self.project_dir(workspace_id, project_id) / FILENAME_PROJECT

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    def tasks_dir(self, workspace_id: str, project_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/{project_id}/tasks/
        """
        return self.project_dir(workspace_id, project_id) / "tasks"

    def task_dir(self, workspace_id: str, project_id: str, task_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/{project_id}/tasks/{task_id}/
        """
        return self.tasks_dir(workspace_id, project_id) / task_id

    def task_file(self, workspace_id: str, project_id: str, task_id: str) -> Path:
        """
        ~/.hiperforge/workspaces/{workspace_id}/projects/{project_id}/tasks/{task_id}/task.json
        """
        return self.task_dir(workspace_id, project_id, task_id) / FILENAME_TASK

    # ------------------------------------------------------------------
    # Métodos de utilidad
    # ------------------------------------------------------------------

    def ensure_app_dirs(self) -> None:
        """
        Crea todos los directorios base del sistema si no existen.

        Llamado una vez al arrancar HiperForge por primera vez.
        Idempotente — si los directorios ya existen, no hace nada.
        """
        dirs_to_create = [
            self._app_dir,
            self.workspaces_dir,
            self.logs_dir,
            self.locks_dir,
        ]

        for directory in dirs_to_create:
            directory.mkdir(parents=True, exist_ok=True)

    def list_workspace_ids(self) -> list[str]:
        """
        Lista los IDs de todos los workspaces en disco.

        Lee los subdirectorios de workspaces_dir — cada subdirectorio
        es el ID de un workspace. Ordenados por nombre (= orden cronológico
        si usamos ULIDs como IDs).

        Returns:
            Lista de workspace IDs. Lista vacía si no hay ninguno.
        """
        if not self.workspaces_dir.exists():
            return []

        return sorted(
            entry.name
            for entry in self.workspaces_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def list_project_ids(self, workspace_id: str) -> list[str]:
        """
        Lista los IDs de todos los proyectos de un workspace.

        Returns:
            Lista de project IDs. Lista vacía si el workspace no existe
            o no tiene proyectos.
        """
        projects_dir = self.projects_dir(workspace_id)

        if not projects_dir.exists():
            return []

        return sorted(
            entry.name
            for entry in projects_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def list_task_ids(self, workspace_id: str, project_id: str) -> list[str]:
        """
        Lista los IDs de todas las tasks de un proyecto.

        Returns:
            Lista de task IDs. Lista vacía si el proyecto no existe
            o no tiene tasks.
        """
        tasks_dir = self.tasks_dir(workspace_id, project_id)

        if not tasks_dir.exists():
            return []

        return sorted(
            entry.name
            for entry in tasks_dir.iterdir()
            if entry.is_dir() and not entry.name.startswith(".")
        )

    def workspace_exists(self, workspace_id: str) -> bool:
        """True si existe el directorio y el archivo workspace.json."""
        return self.workspace_file(workspace_id).exists()

    def project_exists(self, workspace_id: str, project_id: str) -> bool:
        """True si existe el directorio y el archivo project.json."""
        return self.project_file(workspace_id, project_id).exists()

    def task_exists(self, workspace_id: str, project_id: str, task_id: str) -> bool:
        """True si existe el directorio y el archivo task.json."""
        return self.task_file(workspace_id, project_id, task_id).exists()

    def __repr__(self) -> str:
        return f"WorkspaceLocator(app_dir={self._app_dir})"