"""
RunTaskUseCase — Orquesta el flujo completo de ejecución de una task.

Es el use case principal de HiperForge — el que invoca el comando
`hiperforge run "..."`. Todo el ciclo de vida de una task pasa por aquí:

════════════════════════════════════════════════════════════
FLUJO COMPLETO
════════════════════════════════════════════════════════════

  RunTaskInput (prompt, workspace_id, project_id, auto_confirm)
       │
       ▼
  1. _resolve_workspace()          → workspace activo o crea default
       │
       ▼
  2. Task.create()                 → entidad Task en estado PENDING
       │
       ▼
  3. SessionFlusher.__enter__()    → registra signal handlers SIGINT/SIGTERM
       │                             garantiza persistencia ante cualquier fallo
       ▼
  4. task.start_planning()         → PENDING → PLANNING
  5. planner.generate_plan()       → genera la lista de Subtasks
  6. task.start_execution()        → PLANNING → IN_PROGRESS
       │
       ▼
  7. executor.execute_plan()       → loop ReAct subtask por subtask
       │                             callbacks para confirmación y límites
       ▼
  8. SessionFlusher.__exit__()     → flush completo a disco
       │
       ▼
  9. RunTaskOutput                 → métricas + resumen para la CLI

════════════════════════════════════════════════════════════
GARANTÍAS DE PERSISTENCIA
════════════════════════════════════════════════════════════

  El SessionFlusher como context manager garantiza que la task
  se persiste a disco en TODOS los escenarios posibles:

    Escenario A — Éxito:
      with SessionFlusher → executor completa → __exit__ → flush completo

    Escenario B — Fallo en planificación:
      with SessionFlusher → planner lanza error → except → task.fail()
      → session actualizada → __exit__ → flush con estado FAILED

    Escenario C — Fallo en ejecución:
      with SessionFlusher → executor detecta fallo → task.fail()
      → __exit__ → flush con estado FAILED

    Escenario D — Ctrl+C del usuario:
      with SessionFlusher → signal SIGINT → flush_on_interrupt()
      → task.cancel() → flush con estado CANCELLED

    Escenario E — SIGTERM del sistema:
      with SessionFlusher → signal handler → flush_on_interrupt()
      → task.cancel() → flush con estado CANCELLED

════════════════════════════════════════════════════════════
WORKSPACE AUTOMÁTICO EN PRIMERA EJECUCIÓN
════════════════════════════════════════════════════════════

  Si el usuario nunca ha configurado HiperForge, no hay workspace activo.
  En vez de fallar con "EntityNotFound", creamos un workspace "default"
  automáticamente la primera vez.

  Esto hace que `hiperforge run "..."` funcione sin ninguna configuración
  previa — experiencia de zero-setup para el usuario.

════════════════════════════════════════════════════════════
VINCULACIÓN CON PROYECTOS
════════════════════════════════════════════════════════════

  Si se especifica project_id, la task se vincula al proyecto.
  El use case verifica que el proyecto existe y pertenece al workspace.
  Si no existe, la task se crea como "task suelta" (sin proyecto).

  Tareas sueltas son válidas para exploración rápida sin proyecto.
"""

from __future__ import annotations

import time
from typing import Callable

from hiperforge.application.dto import RunTaskInput, RunTaskOutput
from hiperforge.application.services.executor import ExecutorService, LimitDecision
from hiperforge.application.services.planner import PlannerService
from hiperforge.core.config import Settings
from hiperforge.core.events import AgentEvent, get_event_bus
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import Task, TaskStatus
from hiperforge.domain.entities.workspace import Workspace
from hiperforge.domain.exceptions import EntityNotFound, PlanError
from hiperforge.infrastructure.session.in_memory_session import InMemorySession
from hiperforge.infrastructure.session.session_flusher import SessionFlusher
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)


class RunTaskUseCase:
    """
    Orquesta el flujo completo de creación y ejecución de una task.

    Responsabilidades:
      - Resolver el workspace activo (o crear uno por defecto).
      - Crear la entidad Task y registrarla en la sesión.
      - Coordinar planificación y ejecución con sus servicios.
      - Garantizar persistencia a disco con SessionFlusher.
      - Construir el RunTaskOutput con las métricas finales.

    Parámetros:
        planner:  Genera el plan de subtasks con el LLM.
        executor: Ejecuta el plan con el loop ReAct.
        store:    Acceso a datos persistentes.
        locator:  Resuelve rutas del sistema de archivos.
        settings: Configuración del sistema.
    """

    def __init__(
        self,
        planner: PlannerService,
        executor: ExecutorService,
        store: MemoryStore,
        locator: WorkspaceLocator,
        settings: Settings,
    ) -> None:
        self._planner = planner
        self._executor = executor
        self._store = store
        self._locator = locator
        self._settings = settings

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def execute(
        self,
        input_data: RunTaskInput,
        *,
        on_confirm_plan: Callable[[Task], bool] | None = None,
        on_subtask_limit_reached: Callable[[object, int], LimitDecision] | None = None,
        on_subtask_started: Callable[[object, int, int], None] | None = None,
    ) -> RunTaskOutput:
        """
        Ejecuta el flujo completo de una task del agente.

        Parámetros:
            input_data:               Datos de entrada validados.
            on_confirm_plan:          Callback para mostrar el plan y pedir confirmación.
                                      Signature: (task: Task) -> bool
                                      True = proceder, False = cancelar.
                                      None = ejecutar sin confirmación (modo --yes).
            on_subtask_limit_reached: Callback cuando el loop ReAct agota iteraciones.
                                      Signature: (subtask, iterations_used) -> LimitDecision
                                      None = fallar la subtask automáticamente.
            on_subtask_started:       Callback informativo al inicio de cada subtask.
                                      Signature: (subtask, current_index, total) -> None
                                      None = sin notificaciones de progreso.

        Returns:
            RunTaskOutput con el estado final, el resumen y las métricas.

        Raises:
            No lanza excepciones — todos los errores se capturan y se
            reflejan en RunTaskOutput.status == "failed" con error_message.
        """
        start_time = time.monotonic()

        # ── Paso 1: resolver el workspace activo ──────────────────────
        workspace_id = self._resolve_workspace_id(input_data)

        # ── Modo B: ejecutar una task ya existente (planificada) ──────
        if input_data.task_id is not None:
            return self._execute_existing_task(
                task_id=input_data.task_id,
                workspace_id=workspace_id,
                auto_confirm=input_data.auto_confirm,
                on_confirm_plan=on_confirm_plan,
                on_subtask_limit_reached=on_subtask_limit_reached,
                on_subtask_started=on_subtask_started,
                start_time=start_time,
            )

        # ── Modo A: crear y ejecutar una task nueva ───────────────────
        logger.info(
            "iniciando RunTaskUseCase (nueva task)",
            workspace_id=workspace_id,
            project_id=input_data.project_id,
            prompt_preview=(input_data.prompt or "")[:80],
            auto_confirm=input_data.auto_confirm,
        )

        # ── Paso 2: resolver project_id válido ────────────────────────
        effective_project_id = self._resolve_project_id(
            project_id=input_data.project_id,
            workspace_id=workspace_id,
        )

        # ── Paso 3: crear la entidad Task ─────────────────────────────
        task = Task.create(
            prompt=input_data.prompt or "",
            project_id=effective_project_id,
        )

        logger.info(
            "task creada",
            task_id=task.id,
            project_id=effective_project_id,
            workspace_id=workspace_id,
        )

        # Emitir evento de inicio
        get_event_bus().emit(
            AgentEvent.task_started(
                task_id=task.id,
                prompt=task.prompt,
                project_id=task.project_id,
            )
        )

        # ── Paso 4: sesión en memoria + flusher ───────────────────────
        # El flusher garantiza persistencia en cualquier escenario
        session = InMemorySession(task=task, workspace_id=workspace_id)

        with SessionFlusher(
            session=session,
            storage=self._store._storage,
            locator=self._locator,
        ):
            try:
                task = self._run_planning_and_execution(
                    task=task,
                    session=session,
                    auto_confirm=input_data.auto_confirm,
                    on_confirm_plan=on_confirm_plan,
                    on_subtask_limit_reached=on_subtask_limit_reached,
                    on_subtask_started=on_subtask_started,
                )

                # Actualizar la sesión con el estado final de la task
                session.update_task(task)

            except KeyboardInterrupt:
                # Ctrl+C — el SessionFlusher ya registró el signal handler,
                # pero si llega aquí es porque ocurrió dentro del loop de Python.
                # Cancelamos la task y dejamos que el flusher persista.
                logger.warning("KeyboardInterrupt en RunTaskUseCase — cancelando task")
                if not task.is_terminal:
                    task = task.cancel()
                session.update_task(task)
                raise  # Repropagamos para que la CLI maneje la salida correctamente

            except Exception as exc:
                # Error inesperado — marcamos la task como fallida y persistimos
                logger.error(
                    "error inesperado en RunTaskUseCase",
                    task_id=task.id,
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                if not task.is_terminal:
                    task = task.fail()
                session.update_task(task)
                # No repropagamos — construimos el output con el error

        # ── Paso 5: construir y devolver el output ────────────────────
        duration = time.monotonic() - start_time

        return self._build_output(task=task, duration_seconds=duration)

    # ------------------------------------------------------------------
    # Planificación y ejecución
    # ------------------------------------------------------------------

    def _run_planning_and_execution(
        self,
        task: Task,
        session: InMemorySession,
        auto_confirm: bool,
        on_confirm_plan: Callable[[Task], bool] | None,
        on_subtask_limit_reached: Callable[[object, int], LimitDecision] | None,
        on_subtask_started: Callable[[object, int, int], None] | None,
    ) -> Task:
        """
        Ejecuta las fases de planificación y ejecución en secuencia.

        Soporta dos puntos de entrada:
          - Task en PENDING     → planifica primero, luego ejecuta.
          - Task en IN_PROGRESS → el plan ya existe, ejecuta directamente.
            (Ocurre cuando se usó plan_task antes de run_task.)

        Si la planificación falla, la task queda en estado FAILED.

        Returns:
            Task en estado terminal (COMPLETED, FAILED o CANCELLED).
        """
        # ── Caso 1: task ya planificada (IN_PROGRESS con subtasks) ────
        # Ocurre cuando el usuario usó `hiperforge task plan` antes de run.
        # Saltamos la planificación y vamos directo a la ejecución.
        if task.status == TaskStatus.IN_PROGRESS and task.subtasks:
            logger.info(
                "task ya planificada — saltando fase de planificación",
                task_id=task.id,
                subtask_count=len(task.subtasks),
            )
            session.update_task(task)

        else:
            # ── Caso 2: task en PENDING — planificar primero ──────────
            task = task.start_planning()
            get_event_bus().emit(AgentEvent.task_planning(task_id=task.id))
            session.update_task(task)

            try:
                subtasks = self._planner.generate_plan(task)
            except PlanError as exc:
                logger.error(
                    "planificación falló",
                    task_id=task.id,
                    error=str(exc),
                )
                # Transicionamos directamente a FAILED desde PLANNING.
                # Las transiciones válidas son PLANNING → IN_PROGRESS → FAILED,
                # así que necesitamos pasar por IN_PROGRESS con lista vacía.
                task = task.start_execution([])
                task = task.fail()
                session.update_task(task)
                return task

            # Registrar el plan generado en la task
            task = task.start_execution(subtasks)
            session.update_task(task)

        logger.info(
            "plan listo para ejecutar",
            task_id=task.id,
            subtask_count=len(task.subtasks),
        )

        # ── Fase de ejecución ─────────────────────────────────────────
        # Si auto_confirm está activo, no pedimos confirmación del plan
        effective_confirm_callback = (
            None if auto_confirm or self._settings.debug
            else on_confirm_plan
        )

        task = self._executor.execute_plan(
            task=task,
            on_confirm_plan=effective_confirm_callback,
            on_subtask_limit_reached=on_subtask_limit_reached,
            on_subtask_started=on_subtask_started,
        )

        session.update_task(task)

        return task

    # ------------------------------------------------------------------
    # Resolución de workspace
    # ------------------------------------------------------------------

    def _resolve_workspace_id(self, input_data: RunTaskInput) -> str:
        """
        Determina el workspace_id a usar para esta task.

        PRIORIDAD:
          1. workspace_id explícito en el input → se usa directamente.
          2. Workspace activo en el índice global → se usa si existe.
          3. No hay workspace → se crea uno "default" automáticamente.

        La creación automática garantiza zero-setup en la primera ejecución.
        El workspace "default" es funcional — el usuario puede usarlo
        indefinidamente o migrar sus tasks a un workspace nombrado después.

        Returns:
            workspace_id garantizando que existe en disco.

        Raises:
            EntityNotFound: Si se especificó un workspace_id que no existe.
        """
        # Caso 1: workspace explícito en el input
        if input_data.workspace_id:
            if not self._locator.workspace_exists(input_data.workspace_id):
                raise EntityNotFound(
                    entity_type="Workspace",
                    entity_id=input_data.workspace_id,
                )
            return input_data.workspace_id

        # Caso 2: workspace activo global
        active_id = self._store.get_active_workspace_id()
        if active_id and self._locator.workspace_exists(active_id):
            return active_id

        # Caso 3: primera ejecución — crear workspace default
        logger.info("no hay workspace activo — creando workspace 'default'")
        return self._create_default_workspace()

    def _create_default_workspace(self) -> str:
        """
        Crea el workspace por defecto para la primera ejecución.

        El workspace se llama "default" y queda como activo globalmente.
        El usuario puede renombrarlo o crear otros más adelante.

        Returns:
            ID del workspace recién creado.
        """
        default_ws = Workspace.create(
            name="default",
            description=(
                "Workspace creado automáticamente en la primera ejecución de HiperForge. "
                "Puedes renombrarlo con: hiperforge workspace rename default <nuevo-nombre>"
            ),
        )

        self._store.workspaces.save(default_ws)
        self._store.workspaces.set_active_workspace(default_ws.id)

        logger.info(
            "workspace default creado y activado",
            workspace_id=default_ws.id,
        )

        return default_ws.id

    # ------------------------------------------------------------------
    # Resolución de proyecto
    # ------------------------------------------------------------------

    def _resolve_project_id(
        self,
        project_id: str | None,
        workspace_id: str,
    ) -> str | None:
        """
        Verifica que el project_id especificado es válido.

        Si project_id es None, devuelve None — la task será suelta.
        Si project_id no existe en el workspace, devuelve None también
        (y loggea un warning) — preferimos una task suelta a un error
        que bloquee completamente la ejecución.

        Returns:
            project_id si es válido, None si no lo es o no se especificó.
        """
        if project_id is None:
            return None

        if self._locator.project_exists(workspace_id, project_id):
            return project_id

        logger.warning(
            "project_id especificado no existe — creando task suelta",
            project_id=project_id,
            workspace_id=workspace_id,
        )

        return None

    # ------------------------------------------------------------------
    # Construcción del output
    # ------------------------------------------------------------------

    def _build_output(self, task: Task, duration_seconds: float) -> RunTaskOutput:
        """
        Construye el RunTaskOutput a partir del estado final de la task.

        Extrae exactamente los campos que la CLI necesita para mostrar
        el resultado al usuario — sin exponer entidades del dominio.

        Parámetros:
            task:             Task en estado terminal.
            duration_seconds: Tiempo total del proceso (incluyendo planificación).
        """
        completed_count = len(task.completed_subtasks)
        total_count = len(task.subtasks)

        # El error_message solo se incluye cuando la task falló
        error_message: str | None = None
        if task.status.value == "failed":
            error_message = (
                "La task no se pudo completar. Revisa el historial de subtasks "
                "para ver qué salió mal. Puedes intentar de nuevo con una "
                "instrucción más específica o en pasos más pequeños."
            )
        elif task.status.value == "cancelled":
            error_message = "La task fue cancelada antes de completarse."

        return RunTaskOutput(
            task_id=task.id,
            status=task.status.value,
            summary=task.summary or "",
            subtasks_completed=completed_count,
            subtasks_total=total_count,
            total_tokens=task.token_usage.total_tokens,
            estimated_cost_usd=task.token_usage.estimated_cost_usd,
            duration_seconds=round(duration_seconds, 3),
            error_message=error_message,
        )

    def _execute_existing_task(
        self,
        task_id: str,
        workspace_id: str,
        auto_confirm: bool,
        on_confirm_plan: Callable[[Task], bool] | None,
        on_subtask_limit_reached: Callable[[object, int], LimitDecision] | None,
        on_subtask_started: Callable[[object, int, int], None] | None,
        start_time: float,
    ) -> RunTaskOutput:
        """
        Ejecuta una task ya existente en disco (ya planificada).

        Carga la task desde disco, verifica que está en IN_PROGRESS
        con subtasks en PENDING, y la pasa directamente al executor
        saltando la fase de planificación.

        Parámetros:
            task_id:      ID de la task a ejecutar.
            workspace_id: Workspace donde reside la task.
        """
        logger.info(
            "iniciando RunTaskUseCase (task existente)",
            task_id=task_id,
            workspace_id=workspace_id,
        )

        # Buscar la task en todos los proyectos del workspace
        task = self._load_task_from_workspace(
            task_id=task_id,
            workspace_id=workspace_id,
        )

        if task is None:
            # Task no encontrada — construimos un output de fallo
            return RunTaskOutput(
                task_id=task_id,
                status="failed",
                summary="",
                subtasks_completed=0,
                subtasks_total=0,
                total_tokens=0,
                estimated_cost_usd=0.0,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error_message=(
                    f"Task '{task_id}' no encontrada en el workspace '{workspace_id}'. "
                    f"Verifica el ID con: hiperforge task list"
                ),
            )

        # Verificar que la task puede ejecutarse
        if task.status == TaskStatus.COMPLETED:
            return RunTaskOutput(
                task_id=task.id,
                status="completed",
                summary=task.summary or "Task ya completada anteriormente.",
                subtasks_completed=len(task.completed_subtasks),
                subtasks_total=len(task.subtasks),
                total_tokens=task.token_usage.total_tokens,
                estimated_cost_usd=task.token_usage.estimated_cost_usd,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error_message=None,
            )

        if task.is_terminal:
            return RunTaskOutput(
                task_id=task.id,
                status=task.status.value,
                summary="",
                subtasks_completed=len(task.completed_subtasks),
                subtasks_total=len(task.subtasks),
                total_tokens=task.token_usage.total_tokens,
                estimated_cost_usd=task.token_usage.estimated_cost_usd,
                duration_seconds=round(time.monotonic() - start_time, 3),
                error_message=(
                    f"La task está en estado '{task.status.value}' y no puede ejecutarse. "
                    f"Crea una nueva task con el mismo prompt si quieres reintentar."
                ),
            )

        # Ejecutar con el mismo flujo que una task nueva
        session = InMemorySession(task=task, workspace_id=workspace_id)
        get_event_bus().emit(
            AgentEvent.task_started(
                task_id=task.id,
                prompt=task.prompt,
                project_id=task.project_id,
            )
        )

        with SessionFlusher(
            session=session,
            storage=self._store._storage,
            locator=self._locator,
        ):
            try:
                task = self._run_planning_and_execution(
                    task=task,
                    session=session,
                    auto_confirm=auto_confirm,
                    on_confirm_plan=on_confirm_plan,
                    on_subtask_limit_reached=on_subtask_limit_reached,
                    on_subtask_started=on_subtask_started,
                )
                session.update_task(task)
            except KeyboardInterrupt:
                if not task.is_terminal:
                    task = task.cancel()
                session.update_task(task)
                raise
            except Exception as exc:
                logger.error(
                    "error inesperado ejecutando task existente",
                    task_id=task.id,
                    error=str(exc),
                )
                if not task.is_terminal:
                    task = task.fail()
                session.update_task(task)

        return self._build_output(task=task, duration_seconds=time.monotonic() - start_time)

    def _load_task_from_workspace(
        self,
        task_id: str,
        workspace_id: str,
    ) -> Task | None:
        """
        Busca una task por ID dentro de todos los proyectos del workspace.

        Devuelve None si no se encuentra en ningún proyecto.
        """
        try:
            workspace = self._store.workspaces.find_by_id_meta(workspace_id)
        except EntityNotFound:
            return None

        for project_id in [p.id for p in workspace.projects]:
            try:
                if self._locator.task_exists(workspace_id, project_id, task_id):
                    return self._store.tasks.find_by_id(
                        workspace_id=workspace_id,
                        project_id=project_id,
                        task_id=task_id,
                    )
            except Exception:
                continue

        return None