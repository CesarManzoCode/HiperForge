"""
PlanTaskUseCase — Genera el plan de subtasks para una task en PENDING.

Parte del flujo de tres fases separadas:
  create_task → plan_task → run_task

Al terminar, la task queda en estado IN_PROGRESS con todas sus subtasks
en PENDING — lista para ser ejecutada cuando el usuario lo decida.

¿POR QUÉ SEPARAR PLAN DE EJECUCIÓN?
  En el flujo rápido (hiperforge run "..."), planificación y ejecución
  ocurren sin interrupción. Pero hay casos donde separarlos tiene valor:

  1. REVISIÓN HUMANA DEL PLAN:
     El usuario quiere ver las subtasks antes de comprometerse con
     la ejecución. Puede revisar, ajustar el prompt si el plan no
     le convence, o confirmar que el agente entendió correctamente.

  2. PLANIFICACIÓN BATCH:
     El usuario crea 5 tasks con create_task, las planifica todas
     con plan_task para revisar sus planes, y luego ejecuta las
     que le parecen correctas con run_task.

  3. DEBUGGING:
     Si el agente falla, el usuario puede inspeccionar el plan
     generado para entender si el problema fue de planificación
     o de ejecución.

FLUJO:
  1. Cargar la task por task_id.
  2. Verificar que está en estado PENDING (única transición válida).
  3. Llamar al planner para generar las subtasks.
  4. Transicionar: PENDING → PLANNING → IN_PROGRESS con subtasks.
  5. Persistir la task actualizada.
  6. Devolver TaskSummary con el plan generado.
"""

from __future__ import annotations

from hiperforge.application.dto import TaskSummary
from hiperforge.application.services.planner import PlannerService
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import Task, TaskStatus
from hiperforge.domain.exceptions import EntityNotFound, InvalidStatusTransition, PlanError
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)


class PlanTaskUseCase:
    """
    Genera el plan de subtasks para una task existente en PENDING.

    Después de este use case, la task está en IN_PROGRESS con subtasks
    en PENDING — lista para ejecutarse con RunTaskUseCase.

    Parámetros:
        planner: PlannerService que llama al LLM para generar el plan.
        store:   MemoryStore para leer y persistir la task.
        locator: WorkspaceLocator para resolver rutas y verificar existencia.
    """

    def __init__(
        self,
        planner: PlannerService,
        store: MemoryStore,
        locator: WorkspaceLocator,
    ) -> None:
        self._planner = planner
        self._store = store
        self._locator = locator

    def execute(
        self,
        task_id: str,
        workspace_id: str | None = None,
    ) -> TaskSummary:
        """
        Genera el plan para la task indicada.

        Parámetros:
            task_id:      ID de la task a planificar.
            workspace_id: Workspace donde reside la task.
                          None = usar el activo global.

        Returns:
            TaskSummary con el plan generado (subtasks en PENDING).

        Raises:
            EntityNotFound:        Si la task no se encuentra en el workspace.
            InvalidStatusTransition: Si la task no está en PENDING.
            PlanError:             Si el LLM no puede generar un plan válido.
        """
        # ── Paso 1: resolver workspace activo ─────────────────────────
        effective_workspace_id = (
            workspace_id or self._store.get_active_workspace_id()
        )

        if not effective_workspace_id:
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id="(activo)",
            )

        # ── Paso 2: localizar y cargar la task ────────────────────────
        task, project_id = self._find_task(
            task_id=task_id,
            workspace_id=effective_workspace_id,
        )

        # ── Paso 3: verificar estado PENDING ──────────────────────────
        if task.status != TaskStatus.PENDING:
            raise InvalidStatusTransition(
                entity=f"Task({task.id})",
                from_status=task.status.value,
                to_status=TaskStatus.PLANNING.value,
            )

        logger.info(
            "planificando task",
            task_id=task.id,
            workspace_id=effective_workspace_id,
            project_id=project_id,
            prompt_preview=task.prompt[:80],
        )

        # ── Paso 4: transicionar a PLANNING ───────────────────────────
        task = task.start_planning()

        # ── Paso 5: generar el plan con el LLM ───────────────────────
        # PlanError se propaga al caller — es información útil para el usuario
        subtasks = self._planner.generate_plan(task)

        # ── Paso 6: registrar el plan — PLANNING → IN_PROGRESS ────────
        task = task.start_execution(subtasks)

        # ── Paso 7: persistir la task actualizada ─────────────────────
        self._store.tasks.save(task=task, workspace_id=effective_workspace_id)

        logger.info(
            "plan generado exitosamente",
            task_id=task.id,
            subtask_count=len(subtasks),
            subtask_descriptions=[st.description[:60] for st in subtasks],
        )

        # ── Paso 8: construir y devolver el summary ───────────────────
        return TaskSummary(
            id=task.id,
            prompt=task.prompt,
            status=task.status.value,
            project_id=task.project_id,
            subtask_count=len(task.subtasks),
            completed_subtasks=len(task.completed_subtasks),
            total_tokens=task.token_usage.total_tokens,
            estimated_cost_usd=task.token_usage.estimated_cost_usd,
            created_at=task.created_at,
            completed_at=task.completed_at,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _find_task(
        self,
        task_id: str,
        workspace_id: str,
    ) -> tuple[Task, str | None]:
        """
        Busca una task por ID en todos los proyectos del workspace.

        Devuelve la task y el project_id donde se encontró.

        Raises:
            EntityNotFound: Si la task no existe en ningún proyecto del workspace.
        """
        try:
            workspace = self._store.workspaces.find_by_id_meta(workspace_id)
        except EntityNotFound:
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

        # Buscar en todos los proyectos del workspace
        for project in workspace.projects:
            if self._locator.task_exists(workspace_id, project.id, task_id):
                try:
                    task = self._store.tasks.find_by_id(
                        workspace_id=workspace_id,
                        project_id=project.id,
                        task_id=task_id,
                    )
                    return task, project.id
                except Exception:
                    continue

        raise EntityNotFound(
            entity_type="Task",
            entity_id=task_id,
        )