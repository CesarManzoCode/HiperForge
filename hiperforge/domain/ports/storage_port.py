"""
Port: StoragePort

Define el contrato que cualquier implementación de almacenamiento
debe cumplir para funcionar con HiperForge.

¿Qué es un Port?
  Un Port es una interfaz abstracta (ABC) que vive en el dominio.
  Define QUÉ operaciones existen, sin decir CÓMO se implementan.
  La implementación concreta (JSON, SQLite, S3, etc.) vive en infrastructure/.

  Dominio                    Infraestructura
  ──────────────────         ──────────────────────────
  StoragePort (ABC)    ←──── JSONStorage (implementación)
  "necesito poder             "lo hago leyendo y escribiendo
   leer y escribir"            archivos .json en disco"

¿Por qué esta separación?
  - Los use cases solo dependen de StoragePort.
  - Si mañana queremos cambiar de JSON a SQLite, creamos SqliteStorage
    que implementa StoragePort y cambiamos una línea en container.py.
  - El resto del código no se toca.

OPERACIONES DEL CONTRATO:
  Workspace  → save, load, exists, delete, list_ids
  Project    → save, load, exists, delete, list_ids (dentro de un workspace)
  Task       → save, load, exists, delete, list_ids (dentro de un project)

USO TÍPICO (desde un use case):
  class CreateProjectUseCase:
      def __init__(self, storage: StoragePort) -> None:
          self._storage = storage   # recibe la implementación por DI

      def execute(self, ...) -> Project:
          project = Project.create(...)
          self._storage.save_project(project)  # no sabe si es JSON o SQLite
          return project
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from hiperforge.domain.entities.project import Project
from hiperforge.domain.entities.task import Task
from hiperforge.domain.entities.workspace import Workspace


class StoragePort(ABC):
    """
    Contrato abstracto para todas las operaciones de persistencia.

    Cualquier clase que herede de StoragePort y no implemente
    todos los métodos abstractos lanzará TypeError al instanciarse.
    Esto garantiza que ninguna implementación incompleta pueda usarse.
    """

    # ------------------------------------------------------------------
    # Workspace
    # ------------------------------------------------------------------

    @abstractmethod
    def save_workspace(self, workspace: Workspace) -> None:
        """
        Persiste un Workspace.

        Si el workspace ya existe, lo sobreescribe completamente.
        Si no existe, lo crea.

        Raises:
            StorageWriteError: Si no se puede escribir en disco.
        """
        ...

    @abstractmethod
    def load_workspace(self, workspace_id: str) -> Workspace:
        """
        Carga un Workspace por su ID con todos sus Projects.

        Parámetros:
            workspace_id: ID del workspace a cargar.

        Returns:
            Workspace completo con sus Projects cargados.

        Raises:
            EntityNotFound:   Si no existe ningún workspace con ese ID.
            StorageReadError: Si no se puede leer el archivo.
            StorageCorruptedError: Si el JSON está corrupto o no pasa validación.
        """
        ...

    @abstractmethod
    def workspace_exists(self, workspace_id: str) -> bool:
        """
        Verifica si existe un workspace con ese ID sin cargarlo completo.

        Más eficiente que load_workspace() cuando solo necesitamos
        saber si existe, no sus datos.
        """
        ...

    @abstractmethod
    def delete_workspace(self, workspace_id: str) -> None:
        """
        Elimina físicamente todos los archivos de un workspace del disco.

        ATENCIÓN: Esta operación es destructiva e irreversible.
        Solo debe llamarse cuando el workspace tiene status=DELETED
        y el usuario confirmó explícitamente.

        Raises:
            EntityNotFound:   Si no existe ningún workspace con ese ID.
            StorageWriteError: Si no se pueden eliminar los archivos.
        """
        ...

    @abstractmethod
    def list_workspace_ids(self) -> list[str]:
        """
        Devuelve los IDs de todos los workspaces existentes en disco.

        No carga los workspaces completos — solo lista los IDs.
        Útil para el índice global y para operaciones de migración.

        Returns:
            Lista de IDs ordenada por fecha de creación (más antiguo primero).
        """
        ...

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    @abstractmethod
    def save_project(self, project: Project) -> None:
        """
        Persiste un Project dentro de su workspace.

        La implementación es responsable de guardar el proyecto
        en la ruta correcta: workspaces/{workspace_id}/projects/{project_id}/

        Raises:
            StorageWriteError: Si no se puede escribir en disco.
        """
        ...

    @abstractmethod
    def load_project(self, workspace_id: str, project_id: str) -> Project:
        """
        Carga un Project por su ID con todas sus Tasks.

        Parámetros:
            workspace_id: ID del workspace que contiene el proyecto.
            project_id:   ID del proyecto a cargar.

        Returns:
            Project completo con sus Tasks cargadas.

        Raises:
            EntityNotFound:       Si no existe el proyecto.
            StorageReadError:     Si no se puede leer.
            StorageCorruptedError: Si el JSON está corrupto.
        """
        ...

    @abstractmethod
    def project_exists(self, workspace_id: str, project_id: str) -> bool:
        """Verifica si existe un proyecto sin cargarlo completo."""
        ...

    @abstractmethod
    def delete_project(self, workspace_id: str, project_id: str) -> None:
        """
        Elimina físicamente todos los archivos de un proyecto del disco.

        ATENCIÓN: Operación destructiva e irreversible.

        Raises:
            EntityNotFound:    Si no existe el proyecto.
            StorageWriteError: Si no se pueden eliminar los archivos.
        """
        ...

    @abstractmethod
    def list_project_ids(self, workspace_id: str) -> list[str]:
        """
        Devuelve los IDs de todos los proyectos de un workspace.

        Raises:
            EntityNotFound: Si el workspace no existe.
        """
        ...

    # ------------------------------------------------------------------
    # Task
    # ------------------------------------------------------------------

    @abstractmethod
    def save_task(self, task: Task) -> None:
        """
        Persiste una Task dentro de su proyecto.

        Ruta de guardado:
          workspaces/{workspace_id}/projects/{project_id}/tasks/{task_id}/

        La implementación resuelve workspace_id desde task.project_id.

        Raises:
            StorageWriteError: Si no se puede escribir.
        """
        ...

    @abstractmethod
    def load_task(self, project_id: str, task_id: str) -> Task:
        """
        Carga una Task con todas sus Subtasks y ToolCalls.

        Parámetros:
            project_id: ID del proyecto que contiene la task.
            task_id:    ID de la task a cargar.

        Returns:
            Task completa con subtasks y tool calls.

        Raises:
            EntityNotFound:       Si no existe la task.
            StorageReadError:     Si no se puede leer.
            StorageCorruptedError: Si el JSON está corrupto.
        """
        ...

    @abstractmethod
    def task_exists(self, project_id: str, task_id: str) -> bool:
        """Verifica si existe una task sin cargarla completa."""
        ...

    @abstractmethod
    def delete_task(self, project_id: str, task_id: str) -> None:
        """
        Elimina físicamente todos los archivos de una task.

        ATENCIÓN: Operación destructiva e irreversible.

        Raises:
            EntityNotFound:    Si no existe la task.
            StorageWriteError: Si no se pueden eliminar los archivos.
        """
        ...

    @abstractmethod
    def list_task_ids(self, project_id: str) -> list[str]:
        """
        Devuelve los IDs de todas las tasks de un proyecto.

        Raises:
            EntityNotFound: Si el proyecto no existe.
        """
        ...

    # ------------------------------------------------------------------
    # Índice global de workspaces
    # ------------------------------------------------------------------

    @abstractmethod
    def load_active_workspace_id(self) -> str | None:
        """
        Lee el ID del workspace activo desde el índice global.

        El índice global está en ~/.hiperforge/index.json.

        Returns:
            ID del workspace activo, o None si no hay ninguno configurado
            (primera vez que se usa la aplicación).
        """
        ...

    @abstractmethod
    def save_active_workspace_id(self, workspace_id: str) -> None:
        """
        Guarda el ID del workspace activo en el índice global.

        Se llama al crear el primer workspace o al hacer switch.

        Raises:
            StorageWriteError: Si no se puede escribir el índice.
        """
        ...
