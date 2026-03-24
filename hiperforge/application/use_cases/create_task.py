"""
CreateTaskUseCase — Crea una task en estado PENDING sin ejecutarla.

Parte del flujo de tres fases separadas:
  create_task → plan_task → run_task

Permite al usuario construir un backlog de tasks antes de ejecutarlas,
revisar y editar el prompt antes de comprometerse con la ejecución,
o simplemente registrar trabajo pendiente para después.

FLUJO:
  1. Resolver workspace activo (o crear default si no existe).
  2. Verificar que el project_id es válido si se especificó.
  3. Crear la entidad Task en estado PENDING.
  4. Persistir task.json en disco.
  5. Si tiene proyecto: actualizar project.json con el nuevo task_id.
  6. Devolver TaskSummary a la CLI.

DIFERENCIA CON RunTaskUseCase:
  RunTaskUseCase crea + planifica + ejecuta en un solo flujo.
  CreateTaskUseCase solo crea — sin LLM, sin planificación, sin ejecución.
  La task queda en PENDING lista para ser procesada cuando el usuario lo decida.
"""

from __future__ import annotations

from hiperforge.application.dto import CreateTaskInput, TaskSummary
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import Task
from hiperforge.domain.entities.workspace import Workspace
from hiperforge.domain.exceptions import EntityNotFound
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)


class CreateTaskUseCase:
    """
    Crea una task en estado PENDING sin planificar ni ejecutar.

    Parámetros:
        store:   MemoryStore para acceder a workspaces, proyectos y tasks.
        locator: WorkspaceLocator para verificar existencia de entidades.
    """

    def __init__(self, store: MemoryStore, locator: WorkspaceLocator) -> None:
        self._store = store
        self._locator = locator

    def execute(self, input_data: CreateTaskInput) -> TaskSummary:
        """
        Crea la task en disco y devuelve su resumen.

        Parámetros:
            input_data: Prompt, workspace_id y project_id opcionales.

        Returns:
            TaskSummary con los datos de la task recién creada.

        Raises:
            EntityNotFound:  Si el workspace_id o project_id especificados no existen.
            PermissionError: Si el proyecto está archivado o eliminado.
        """
        # ── Paso 1: resolver workspace activo ─────────────────────────
        workspace_id = self._resolve_workspace_id(input_data.workspace_id)

        # ── Paso 2: resolver y verificar project_id ───────────────────
        effective_project_id = self._resolve_project_id(
            project_id=input_data.project_id,
            workspace_id=workspace_id,
        )

        # ── Paso 3: crear la entidad Task ─────────────────────────────
        task = Task.create(
            prompt=input_data.prompt,
            project_id=effective_project_id,
        )

        # ── Paso 4: persistir task.json ───────────────────────────────
        self._store.tasks.save(task=task, workspace_id=workspace_id)

        # ── Paso 5: actualizar project.json si tiene proyecto ─────────
        if effective_project_id is not None:
            self._register_task_in_project(
                task=task,
                workspace_id=workspace_id,
                project_id=effective_project_id,
            )

        logger.info(
            "task creada en PENDING",
            task_id=task.id,
            project_id=effective_project_id,
            workspace_id=workspace_id,
            prompt_preview=task.prompt[:80],
        )

        # ── Paso 6: construir y devolver el summary ───────────────────
        return TaskSummary(
            id=task.id,
            prompt=task.prompt,
            status=task.status.value,
            project_id=task.project_id,
            subtask_count=0,
            completed_subtasks=0,
            total_tokens=0,
            estimated_cost_usd=0.0,
            created_at=task.created_at,
            completed_at=task.completed_at,
        )

    # ------------------------------------------------------------------
    # Helpers de resolución
    # ------------------------------------------------------------------

    def _resolve_workspace_id(self, workspace_id: str | None) -> str:
        """
        Determina el workspace_id efectivo.

        Si se especificó explícitamente, verifica que existe.
        Si no, usa el activo global.
        Si no hay activo, crea un workspace default.
        """
        if workspace_id:
            if not self._locator.workspace_exists(workspace_id):
                raise EntityNotFound(
                    entity_type="Workspace",
                    entity_id=workspace_id,
                )
            return workspace_id

        active_id = self._store.get_active_workspace_id()
        if active_id and self._locator.workspace_exists(active_id):
            return active_id

        # Primera ejecución — crear workspace default
        logger.info("no hay workspace activo — creando workspace 'default'")
        return self._create_default_workspace()

    def _create_default_workspace(self) -> str:
        """Crea el workspace por defecto y lo activa."""
        default_ws = Workspace.create(
            name="default",
            description=(
                "Workspace creado automáticamente en la primera ejecución de HiperForge."
            ),
        )
        self._store.workspaces.save(default_ws)
        self._store.workspaces.set_active_workspace(default_ws.id)

        logger.info(
            "workspace default creado",
            workspace_id=default_ws.id,
        )
        return default_ws.id

    def _resolve_project_id(
        self,
        project_id: str | None,
        workspace_id: str,
    ) -> str | None:
        """
        Verifica que el project_id es válido y acepta nuevas tasks.

        Raises:
            EntityNotFound:  Si el project_id no existe en el workspace.
            PermissionError: Si el proyecto está archivado o eliminado.
        """
        if project_id is None:
            return None

        if not self._locator.project_exists(workspace_id, project_id):
            raise EntityNotFound(
                entity_type="Project",
                entity_id=project_id,
            )

        # Cargamos el proyecto para verificar su estado
        project = self._store.projects.find_by_id_meta(workspace_id, project_id)

        if project.status.value == "archived":
            raise PermissionError(
                f"No se pueden crear tasks en el proyecto '{project.name}' "
                f"porque está archivado. Reactívalo primero con: "
                f"hiperforge project reactivate {project_id}"
            )

        if project.status.value == "deleted":
            raise PermissionError(
                f"El proyecto con id '{project_id}' fue eliminado y no puede recibir tasks."
            )

        return project_id

    def _register_task_in_project(
        self,
        task: Task,
        workspace_id: str,
        project_id: str,
    ) -> None:
        """
        Agrega el task_id al índice de tasks del proyecto.

        Carga el proyecto, agrega la task, y persiste.
        Si falla, loggea el error pero no interrumpe la creación
        de la task — la task ya fue guardada y es recuperable.
        """
        try:
            project = self._store.projects.find_by_id_meta(workspace_id, project_id)
            updated_project = project.add_task(task)
            self._store.projects.save(updated_project)
        except Exception as exc:
            # Fallo no fatal — la task existe en disco pero el proyecto
            # no sabe de ella todavía. Loggeamos para debug.
            logger.warning(
                "no se pudo registrar la task en el proyecto — task sigue siendo válida",
                task_id=task.id,
                project_id=project_id,
                error=str(exc),
            )