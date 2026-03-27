"""
ExecutorService — El corazón del agente HiperForge.

Implementa el ciclo completo de Razonar → Actuar → Observar para cada
subtask del plan hasta completarla o agotar los recursos disponibles.

════════════════════════════════════════════════════════════
ARQUITECTURA DEL LOOP REACT
════════════════════════════════════════════════════════════

  execute_plan(task)
  └── para cada subtask en orden:
      └── _execute_subtask(task, subtask)
          ├── Crear sesión en memoria con contexto inicial
          ├── Loop ReAct (máx REACT_MAX_ITERATIONS_PER_SUBTASK):
          │   ├── _call_llm()          → obtiene respuesta del LLM
          │   ├── _process_response()  → interpreta la acción
          │   │   ├── action=think     → razonamiento, continuar loop
          │   │   ├── action=tool_call → _execute_tool_call()
          │   │   └── action=complete  → marcar subtask done, salir loop
          │   └── actualizar estado en sesión y task
          └── Si límite alcanzado → _handle_limit_reached()

════════════════════════════════════════════════════════════
DETECCIÓN DE BUCLES Y PATRONES REPETITIVOS
════════════════════════════════════════════════════════════

  Un agente puede quedar atascado repitiendo las mismas acciones:
    Iteración 3: tool_call shell {"command": "pytest"}
    Iteración 5: tool_call shell {"command": "pytest"}  ← mismo error
    Iteración 7: tool_call shell {"command": "pytest"}  ← sigue igual

  El executor detecta esto comparando las últimas N tool calls.
  Si detecta un bucle, inyecta un mensaje de "intervención" en el
  historial que fuerza al LLM a tomar una estrategia diferente.

════════════════════════════════════════════════════════════
REINTENTOS INTELIGENTES ANTE TIMEOUTS
════════════════════════════════════════════════════════════

  Cuando una tool tiene timeout, el executor no falla inmediatamente.
  Primero intenta con extended_timeout automáticamente:

    Intento 1: shell {"command": "npm install"}           → TIMEOUT (30s)
    Intento 2: shell {"command": "npm install",
                       "extended_timeout": true}          → OK (45s)

  Solo si el reintento con extended_timeout también falla,
  el error se propaga a la subtask.

════════════════════════════════════════════════════════════
LIMPIEZA DE HISTORIAL ENTRE SUBTASKS
════════════════════════════════════════════════════════════

  Cada subtask comienza con un historial limpio.
  Esto tiene dos ventajas:
    1. El LLM no se confunde con el contexto de subtasks anteriores.
    2. El context window se usa eficientemente en cada subtask.

  Excepción: si la subtask actual depende explícitamente del resultado
  de una anterior (indicado en su descripción), se puede pasar un
  resumen del resultado previo como contexto inicial.

════════════════════════════════════════════════════════════
PERSISTENCIA INCREMENTAL
════════════════════════════════════════════════════════════

  La task se persiste a disco después de cada subtask completada.
  Si el proceso muere durante la subtask 4 de 6, las subtasks
  1-3 ya están guardadas y no se pierden.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from hiperforge.core.constants import (
    LLM_DEFAULT_MAX_TOKENS_REACT,
    REACT_MAX_ITERATIONS_PER_SUBTASK,
    REACT_MAX_ITERATIONS_SIMPLE,
    REACT_MAX_ITERATIONS_MEDIUM,
    REACT_MAX_TOOL_RETRIES,
    REACT_RETRY_DELAY_SECONDS,
)
from hiperforge.core.events import AgentEvent, get_event_bus
from hiperforge.core.logging import get_agent_logger, get_logger
from hiperforge.core.utils.datetime import seconds_since
from hiperforge.domain.entities.task import Subtask, SubtaskStatus, Task, TaskStatus
from hiperforge.domain.exceptions import ToolTimeoutError
from hiperforge.domain.ports.session_port import EventType as SessionEventType
from hiperforge.domain.value_objects.message import Message
from hiperforge.infrastructure.llm.base import BaseLLMAdapter, RichLLMResponse, ToolCallRequest
from hiperforge.infrastructure.session.in_memory_session import InMemorySession
from hiperforge.memory.store import MemoryStore
from hiperforge.application.services.context_builder import ContextBuilder
from hiperforge.application.services.tool_dispatcher import ToolDispatcher
from hiperforge.core.config import Settings

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Constantes internas del executor
# ---------------------------------------------------------------------------

# Cuántas tool calls recientes comparar para detectar bucles
_LOOP_DETECTION_WINDOW = 3

# Ventana reducida para detección POST-intervención — si tras inyectar una
# intervención el agente repite el mismo error 2 veces más, cortamos antes.
_POST_INTERVENTION_DETECTION_WINDOW = 2

# Ventana para detectar exploración o verificación redundante
_EFFICIENCY_WINDOW = 4

# Después de cuántas iteraciones sin progreso inyectar una intervención
_STALL_THRESHOLD = 4

# Mensaje genérico de intervención cuando el agente está atascado
_STALL_INTERVENTION_MSG = """\
{"action": "think", "content": "ATENCION: estoy en un bucle. Debo cambiar de estrategia ahora. No debo repetir la misma tool con el mismo objetivo si ya fallo. Si una ruta es un directorio, usare list/shell en vez de read. Si acabo de escribir un archivo, no debo releerlo completo salvo que sea imprescindible. Mi siguiente respuesta debe ser una sola accion util y distinta."}"""

# Intervención específica para errores de argumentos inválidos repetidos.
# Este es el caso más frecuente: el LLM envía timeout=100000 y no aprende
# del error de validación. La intervención le dice EXACTAMENTE qué hacer.
_ARG_VALIDATION_INTERVENTION_MSG = """\
{"action": "think", "content": "ATENCION: estoy enviando argumentos invalidos repetidamente. El parametro timeout que envie excede el maximo del sistema. SOLUCION: debo OMITIR completamente el campo timeout de mis argumentos para usar el default del sistema, o usar extended_timeout=true si necesito mas tiempo. No debo inventar valores de timeout."}"""

# Intervención para errores de tool no encontrada repetidos
_TOOL_NOT_FOUND_INTERVENTION_MSG = """\
{"action": "think", "content": "ATENCION: estoy usando un nombre de tool que no existe. Debo revisar las herramientas disponibles en mis instrucciones de sistema y usar exactamente uno de esos nombres."}"""

_EFFICIENCY_INTERVENTION_MSG = """\
{"action": "think", "content": "Ya tengo evidencia suficiente o estoy reverificando de forma redundante. No debo volver a listar, releer o ejecutar la misma verificacion si ya paso. Si el resultado ya fue comprobado al menos una vez, ahora debo responder con action=complete. Solo usare otra tool si aporta evidencia NUEVA."}"""


# ---------------------------------------------------------------------------
# Tipos de decisión ante el límite de iteraciones
# ---------------------------------------------------------------------------

class LimitDecision(str, Enum):
    """
    Decisión del usuario cuando el loop ReAct alcanza el límite.
    """
    RETRY   = "retry"   # reintentar la subtask desde cero
    SKIP    = "skip"    # omitir esta subtask y continuar con la siguiente
    CANCEL  = "cancel"  # cancelar toda la task


# ---------------------------------------------------------------------------
# Tipos internos del executor
# ---------------------------------------------------------------------------

@dataclass
class IterationResult:
    """
    Resultado de una iteración del loop ReAct.

    Indica qué ocurrió en esta iteración y si el loop debe continuar.
    """
    subtask: Subtask        # subtask actualizada con el resultado
    task: Task              # task actualizada con tokens y subtask
    should_continue: bool   # True = continuar el loop, False = salir
    action_type: str        # "think", "tool_call", "complete", "text", "error"


@dataclass(frozen=True)
class ToolOutcomeSnapshot:
    """Resumen semántico de una tool call reciente para detectar patrones."""
    fingerprint: str
    family: str
    success: bool
    tool_name: str
    is_mutation: bool
    is_observation: bool
    is_verification: bool


@dataclass
class SubtaskExecutionResult:
    """
    Resultado completo de ejecutar una subtask.

    Contiene el estado final de la subtask y la task, junto con
    métricas de la ejecución para logging y debugging.
    """
    subtask: Subtask
    task: Task
    iterations_used: int
    duration_seconds: float
    loop_detections: int    # cuántos bucles se detectaron y corrigieron


# ---------------------------------------------------------------------------
# ExecutorService
# ---------------------------------------------------------------------------

class ExecutorService:
    """
    Ejecuta el plan de subtasks con el loop ReAct completo.

    Orquesta el ciclo Razonar → Actuar → Observar para cada subtask,
    gestionando el historial de mensajes, los eventos al EventBus,
    la detección de bucles, los reintentos y la persistencia incremental.

    Parámetros:
        llm:             Adapter del LLM configurado.
        tool_dispatcher: Dispatcher que resuelve y ejecuta las tools.
        context_builder: Construye el prompt de sistema.
        store:           MemoryStore para persistir progreso.
        settings:        Configuración del sistema.
    """

    def __init__(
        self,
        llm: BaseLLMAdapter,
        tool_dispatcher: ToolDispatcher,
        context_builder: ContextBuilder,
        store: MemoryStore,
        settings: Settings,
    ) -> None:
        self._llm = llm
        self._dispatcher = tool_dispatcher
        self._context_builder = context_builder
        self._store = store
        self._settings = settings

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def execute_plan(
        self,
        task: Task,
        *,
        on_confirm_plan: Callable[[Task], bool] | None = None,
        on_subtask_limit_reached: Callable[[Subtask, int], LimitDecision] | None = None,
        on_subtask_started: Callable[[Subtask, int, int], None] | None = None,
    ) -> Task:
        """
        Ejecuta todas las subtasks del plan en orden.

        Parámetros:
            task:                     Task con subtasks ya generadas por el planner.
            on_confirm_plan:          Callback para mostrar el plan al usuario y pedir
                                      confirmación antes de ejecutar.
                                      Signature: (task: Task) -> bool
                                      True = confirmar, False = cancelar.
                                      None = ejecutar sin confirmación.
            on_subtask_limit_reached: Callback cuando el loop ReAct agota sus iteraciones.
                                      Signature: (subtask, iterations_used) -> LimitDecision
                                      None = fallar la subtask automáticamente.
            on_subtask_started:       Callback informativo al inicio de cada subtask.
                                      Signature: (subtask, current_index, total) -> None
                                      Útil para la CLI para mostrar progreso.

        Returns:
            Task con el estado final: COMPLETED, FAILED o CANCELLED.
        """
        log = get_agent_logger(
            task_id=task.id,
            provider=self._llm.get_provider_name(),
        )

        log.info(
            "iniciando ejecución del plan",
            subtask_count=len(task.subtasks),
            model=self._llm.get_model_id(),
        )

        # Confirmación del plan por el usuario (si se configuró)
        if on_confirm_plan is not None:
            confirmed = on_confirm_plan(task)
            if not confirmed:
                log.info("plan cancelado por el usuario")
                cancelled_task = task.cancel()
                get_event_bus().emit(AgentEvent.task_cancelled(task_id=task.id))
                return cancelled_task

        # Emitir evento de inicio de ejecución
        get_event_bus().emit(
            AgentEvent.task_executing(
                task_id=task.id,
                subtask_count=len(task.subtasks),
            )
        )

        total_subtasks = len(task.subtasks)

        # Resumen de la subtask anterior — se pasa como contexto a la siguiente
        previous_subtask_summary: str | None = None

        # Máximo de reintentos por subtask cuando el usuario elige "Retry".
        # Esto evita un loop infinito de reintentos si la subtask es imposible.
        _MAX_USER_RETRIES_PER_SUBTASK = 3

        # Ejecutar cada subtask en el orden del plan
        for index, subtask in enumerate(task.subtasks):
            # Notificar a la CLI del inicio de la subtask (informativo)
            if on_subtask_started is not None:
                on_subtask_started(subtask, index, total_subtasks)

            # ── RETRY LOOP ──────────────────────────────────────────────
            # Si el usuario elige "Retry" cuando una subtask agota sus
            # iteraciones, re-ejecutamos la subtask desde cero.
            # Limitamos a _MAX_USER_RETRIES_PER_SUBTASK para evitar
            # reintentos infinitos en subtasks imposibles.
            user_retries = 0
            while True:
                execution_result = self._execute_subtask(
                    task=task,
                    subtask=subtask,
                    on_limit_reached=on_subtask_limit_reached,
                    previous_subtask_summary=previous_subtask_summary,
                )

                # Actualizar referencias locales con el estado más reciente
                task = execution_result.task
                subtask = execution_result.subtask

                # Si la subtask llegó a un estado terminal, salimos del retry loop
                if subtask.is_terminal:
                    break

                # La subtask NO está en estado terminal — el usuario eligió RETRY.
                # Verificar que no excedamos el máximo de reintentos.
                user_retries += 1
                if user_retries >= _MAX_USER_RETRIES_PER_SUBTASK:
                    log.warning(
                        "máximo de reintentos de usuario alcanzado — fallando subtask",
                        subtask_id=subtask.id,
                        user_retries=user_retries,
                    )
                    subtask = subtask.fail()
                    task = task.update_subtask(subtask)
                    break

                # Reintentar: resetear la subtask a PENDING para re-ejecutarla
                log.info(
                    "reintentando subtask por decisión del usuario",
                    subtask_id=subtask.id,
                    user_retry=user_retries,
                    max_retries=_MAX_USER_RETRIES_PER_SUBTASK,
                )
                subtask = subtask.reset_to_pending()
                task = task.update_subtask(subtask)

            # ── FIN DEL RETRY LOOP ──────────────────────────────────────

            # Capturar resumen de esta subtask para pasarlo a la siguiente
            if subtask.status == SubtaskStatus.COMPLETED:
                previous_subtask_summary = subtask.reasoning or subtask.description
            else:
                previous_subtask_summary = None

            log.info(
                "subtask finalizada",
                subtask_id=subtask.id,
                status=subtask.status.value,
                iterations=execution_result.iterations_used,
                duration_seconds=round(execution_result.duration_seconds, 2),
                loop_detections=execution_result.loop_detections,
                progress_pct=task.progress_percentage,
            )

            # Persistir el progreso después de cada subtask completada
            # Si el proceso muere aquí, las subtasks anteriores ya están guardadas
            self._persist_task_progress(task)

            # Evaluar si debemos continuar según el estado de la subtask
            if subtask.status == SubtaskStatus.FAILED:
                log.error(
                    "subtask falló — abortando ejecución del plan",
                    subtask_id=subtask.id,
                    description=subtask.description,
                )
                failed_task = task.fail()
                get_event_bus().emit(
                    AgentEvent.task_failed(
                        task_id=task.id,
                        reason=f"Subtask '{subtask.description[:60]}' falló tras agotar reintentos.",
                    )
                )
                return failed_task

            if task.status == TaskStatus.CANCELLED:
                log.info("task cancelada durante la ejecución")
                return task

        # Todas las subtasks completadas exitosamente
        summary = self._generate_completion_summary(task)
        completed_task = task.complete(summary=summary)

        get_event_bus().emit(
            AgentEvent.task_completed(
                task_id=completed_task.id,
                duration_seconds=completed_task.duration_seconds or 0.0,
                total_tokens=completed_task.token_usage.total_tokens,
                estimated_cost_usd=completed_task.token_usage.estimated_cost_usd,
            )
        )

        log.info(
            "task completada exitosamente",
            total_tokens=completed_task.token_usage.total_tokens,
            cost_usd=completed_task.token_usage.estimated_cost_usd,
            duration_seconds=completed_task.duration_seconds,
        )

        return completed_task

    # ------------------------------------------------------------------
    # Ejecución de una subtask individual
    # ------------------------------------------------------------------

    def _execute_subtask(
        self,
        task: Task,
        subtask: Subtask,
        on_limit_reached: Callable[[Subtask, int], LimitDecision] | None,
        previous_subtask_summary: str | None = None,
    ) -> SubtaskExecutionResult:
        """
        Ejecuta una subtask completa con su loop ReAct.

        Gestiona el historial de mensajes, los eventos, la detección
        de bucles y el manejo del límite de iteraciones.

        Parámetros:
            previous_subtask_summary: Resumen de la subtask anterior completada.
                                      Permite continuidad sin historial completo.

        Returns:
            SubtaskExecutionResult con el estado final y las métricas.
        """
        subtask_start = time.monotonic()
        log = get_agent_logger(
            task_id=task.id,
            subtask_id=subtask.id,
            provider=self._llm.get_provider_name(),
        )

        log.info(
            "iniciando subtask",
            order=subtask.order,
            description=subtask.description,
        )

        # Transicionar subtask a IN_PROGRESS
        subtask = subtask.mark_running()
        task = task.update_subtask(subtask)

        # Bindear contexto al LLM y a todas las tools
        self._llm.bind_context(task_id=task.id, subtask_id=subtask.id)
        self._dispatcher._registry.bind_context_to_all(
            task_id=task.id,
            subtask_id=subtask.id,
        )

        # Emitir evento de inicio de subtask
        get_event_bus().emit(
            AgentEvent.subtask_started(
                task_id=task.id,
                subtask_id=subtask.id,
                order=subtask.order,
                description=subtask.description,
            )
        )

        # Crear la sesión en memoria con el contexto inicial de esta subtask
        session = self._initialize_subtask_session(
            task=task,
            subtask=subtask,
            previous_subtask_summary=previous_subtask_summary,
        )

        # Calcular iteraciones máximas dinámicamente según complejidad del plan
        max_iterations = self._max_iterations_for_plan(len(task.subtasks))

        # Estado interno del loop
        iteration = 0
        loop_detections = 0
        interventions_applied = 0
        recent_tool_outcomes: list[ToolOutcomeSnapshot] = []

        # ──────────────────────────────────────────────────────────────
        # LOOP REACT
        # ──────────────────────────────────────────────────────────────
        while iteration < max_iterations:
            iteration += 1

            log.debug(
                "iteración ReAct",
                iteration=iteration,
                max=max_iterations,
                messages_in_history=session.message_count,
            )

            get_event_bus().emit(
                AgentEvent.react_iteration_started(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    iteration=iteration,
                )
            )

            # ── DETECCIÓN DE BUCLES ──────────────────────────────────
            # Usamos ventana normal para la primera detección y ventana
            # agresiva (más corta) después de ya haber intervenido.
            effective_window = (
                _POST_INTERVENTION_DETECTION_WINDOW
                if interventions_applied > 0
                else _LOOP_DETECTION_WINDOW
            )
            if self._is_stuck_in_loop(recent_tool_outcomes, window=effective_window):
                log.warning(
                    "bucle detectado — inyectando intervención",
                    iteration=iteration,
                    interventions_applied=interventions_applied,
                    recent_calls=[
                        item.fingerprint
                        for item in recent_tool_outcomes[-effective_window:]
                    ],
                )

                # Seleccionar intervención específica según el patrón de error
                intervention_msg = self._select_intervention_message(
                    recent_tool_outcomes
                )
                session.push_message(
                    Message.user(intervention_msg)
                )
                loop_detections += 1
                interventions_applied += 1

                # NO limpiar recent_tool_outcomes por completo — solo eliminar
                # los últimos N para evitar re-detección inmediata, pero conservar
                # historial suficiente para que la detección post-intervención
                # funcione con ventana reducida.
                if len(recent_tool_outcomes) >= effective_window:
                    recent_tool_outcomes = recent_tool_outcomes[:-effective_window]

            elif self._is_wasting_iterations(recent_tool_outcomes):
                log.warning(
                    "exploración/reverificación redundante detectada",
                    iteration=iteration,
                    recent_calls=[
                        item.fingerprint
                        for item in recent_tool_outcomes[-_EFFICIENCY_WINDOW:]
                    ],
                )
                self._inject_efficiency_intervention(session)
                loop_detections += 1

            # Llamar al LLM con el historial actual
            response = self._call_llm_safe(session=session, task=task)

            if response is None:
                # Error fatal en la llamada al LLM — falla la subtask
                subtask = subtask.fail()
                task = task.update_subtask(subtask)
                break

            # Acumular tokens en la task y en la sesión
            task = task.add_token_usage(response.token_usage)
            session.accumulate_tokens(response.token_usage)
            session.update_task(task)

            # Registrar la llamada al LLM en la sesión
            session.record_event(
                SessionEventType.LLM_CALLED,
                {
                    "iteration": iteration,
                    "tokens": response.token_usage.total_tokens,
                    "finish_reason": response.finish_reason,
                },
            )

            # Procesar la respuesta del LLM
            iteration_result = self._process_response(
                response=response,
                task=task,
                subtask=subtask,
                session=session,
                iteration=iteration,
                recent_tool_outcomes=recent_tool_outcomes,
                log=log,
            )

            # Actualizar estado con el resultado de la iteración
            subtask = iteration_result.subtask
            task = iteration_result.task
            session.update_task(task)

            get_event_bus().emit(
                AgentEvent.react_iteration_completed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    iteration=iteration,
                )
            )

            # Salir del loop si la subtask terminó (complete o failed)
            if not iteration_result.should_continue:
                break

        # ──────────────────────────────────────────────────────────────
        # FIN DEL LOOP
        # ──────────────────────────────────────────────────────────────

        # Si el loop terminó sin que la subtask llegara a un estado terminal
        if not subtask.is_terminal:
            log.warning(
                "subtask no completada — límite de iteraciones alcanzado",
                iteration=iteration,
                max=max_iterations,
            )
            subtask, task = self._handle_limit_reached(
                task=task,
                subtask=subtask,
                iterations_used=iteration,
                on_limit_reached=on_limit_reached,
                log=log,
            )

        # Emitir evento de fin de subtask
        duration = time.monotonic() - subtask_start
        self._emit_subtask_end_event(
            task=task,
            subtask=subtask,
            iterations=iteration,
            duration=duration,
        )

        return SubtaskExecutionResult(
            subtask=subtask,
            task=task,
            iterations_used=iteration,
            duration_seconds=duration,
            loop_detections=loop_detections,
        )

    # ------------------------------------------------------------------
    # Procesamiento de la respuesta del LLM
    # ------------------------------------------------------------------

    def _process_response(
        self,
        response: RichLLMResponse,
        task: Task,
        subtask: Subtask,
        session: InMemorySession,
        iteration: int,
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
        log: Any,
    ) -> IterationResult:
        """
        Interpreta la respuesta del LLM y ejecuta la acción correspondiente.

        Maneja los tres tipos de respuesta del protocolo JSON:
          - action=complete   → subtask terminada
          - action=tool_call  → ejecutar tool
          - action=think      → razonamiento del agente, continuar
          - texto libre       → agente olvidó el protocolo, reconducir
        """
        # ── CASE 1: El agente indica que terminó la subtask ────────────
        if response.finish_reason == "complete":
            log.info(
                "agente indica subtask completa",
                summary_preview=response.content[:100],
            )

            subtask, task = self._complete_subtask(
                task=task,
                subtask=subtask,
                session=session,
                summary=response.content,
                iteration=iteration,
            )

            return IterationResult(
                subtask=subtask,
                task=task,
                should_continue=False,
                action_type="complete",
            )

        # ── CASE 2: El agente solicita ejecutar una tool ───────────────
        if response.has_tool_calls:
            subtask, task, all_succeeded = self._execute_tool_calls(
                tool_requests=response.tool_calls,
                task=task,
                subtask=subtask,
                session=session,
                recent_tool_outcomes=recent_tool_outcomes,
                iteration=iteration,
                log=log,
            )

            # Verificar si la subtask falló durante la ejecución de tools
            if subtask.status == SubtaskStatus.FAILED:
                return IterationResult(
                    subtask=subtask,
                    task=task,
                    should_continue=False,
                    action_type="tool_call",
                )

            if all_succeeded and response.deferred_completion_summary:
                completed = self._complete_subtask(
                    task=task,
                    subtask=subtask,
                    session=session,
                    summary=response.deferred_completion_summary,
                    iteration=iteration,
                )
                return IterationResult(
                    subtask=completed[0],
                    task=completed[1],
                    should_continue=False,
                    action_type="complete",
                )

            return IterationResult(
                subtask=subtask,
                task=task,
                should_continue=True,
                action_type="tool_call",
            )

        # ── CASE 3: El agente está razonando (action=think) ────────────
        # OPTIMIZACIÓN: No agregamos el contenido del think al historial.
        # El LLM ya gastó los tokens de output generándolo, pero evitamos
        # que esos tokens se re-lean como input en la siguiente iteración.
        # Esto ahorra ~200-500 tokens de input por cada think innecesario.
        if response.has_content:
            log.debug(
                "agente razonando (no persistido en historial)",
                iteration=iteration,
                content_preview=response.content[:100],
            )

            session.record_event(
                SessionEventType.REACT_ITERATION,
                {
                    "iteration": iteration,
                    "action_type": "think",
                    "content_preview": response.content[:200],
                },
            )

            return IterationResult(
                subtask=subtask,
                task=task,
                should_continue=True,
                action_type="think",
            )

        # ── CASE 4: Respuesta vacía o inválida ─────────────────────────
        # El LLM devolvió algo que no podemos interpretar.
        # Inyectamos un recordatorio del protocolo para reconducirlo.
        log.warning(
            "respuesta del LLM no interpretable — inyectando recordatorio",
            finish_reason=response.finish_reason,
            content_preview=response.content[:100],
            iteration=iteration,
        )

        reminder_msg = Message.user(
            '{"action": "think", "content": "Recuerda: DEBES responder SOLO con JSON '
            'válido. Usa action=think para razonar, action=tool_call para ejecutar '
            'herramientas, o action=complete para indicar que terminaste."}'
        )
        session.push_message(reminder_msg)

        return IterationResult(
            subtask=subtask,
            task=task,
            should_continue=True,
            action_type="error",
        )

    # ------------------------------------------------------------------
    # Ejecución de tool calls
    # ------------------------------------------------------------------

    def _execute_tool_calls(
        self,
        tool_requests: list[ToolCallRequest],
        task: Task,
        subtask: Subtask,
        session: InMemorySession,
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
        iteration: int,
        log: Any,
    ) -> tuple[Subtask, Task, bool]:
        """
        Ejecuta todas las tool calls solicitadas en esta iteración.

        Un solo mensaje del LLM puede solicitar múltiples tools.
        Las ejecutamos en orden y agregamos cada resultado al historial.

        Maneja reintentos ante timeout automáticamente.
        """
        all_succeeded = True
        unique_requests = self._dedupe_tool_requests(tool_requests)

        for request in unique_requests:
            subtask, task, succeeded = self._execute_single_tool_call(
                request=request,
                task=task,
                subtask=subtask,
                session=session,
                recent_tool_outcomes=recent_tool_outcomes,
                iteration=iteration,
                log=log,
            )
            all_succeeded = all_succeeded and succeeded

            # Si la subtask falló durante esta tool, no ejecutamos las siguientes
            if subtask.status == SubtaskStatus.FAILED:
                break

        return subtask, task, all_succeeded

    @staticmethod
    def _dedupe_tool_requests(
        tool_requests: list[ToolCallRequest],
    ) -> list[ToolCallRequest]:
        """
        Elimina tool calls idénticas dentro de una misma respuesta del LLM.

        Algunos modelos repiten accidentalmente el mismo bloque JSON dos veces.
        Ejecutarlo dos veces solo consume iteraciones y puede contaminar el
        historial sin aportar evidencia nueva.
        """
        unique_requests: list[ToolCallRequest] = []
        seen: set[tuple[str, str]] = set()

        for request in tool_requests:
            try:
                args_key = json.dumps(
                    request.arguments,
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                )
            except (TypeError, ValueError):
                args_key = repr(request.arguments)

            fingerprint = (request.tool_name, args_key)
            if fingerprint in seen:
                continue

            seen.add(fingerprint)
            unique_requests.append(request)

        return unique_requests

    def _execute_single_tool_call(
        self,
        request: ToolCallRequest,
        task: Task,
        subtask: Subtask,
        session: InMemorySession,
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
        iteration: int,
        log: Any,
    ) -> tuple[Subtask, Task, bool]:
        """
        Ejecuta una sola tool call con manejo de timeout inteligente.

        ESTRATEGIA DE TIMEOUT:
          Si la tool falla por timeout, intentamos automáticamente con
          extended_timeout=true antes de rendirnos. Solo ShellTool soporta
          este parámetro, pero el dispatcher maneja el caso gracefully
          si otra tool lo recibe.

        TRACKING DE BUCLES:
          Registramos cada tool call en recent_tool_calls para que
          _is_stuck_in_loop() pueda detectar si el agente repite
          las mismas acciones sin progreso.
        """
        log.debug(
            "ejecutando tool call",
            tool_name=request.tool_name,
            iteration=iteration,
        )

        session.record_event(
            SessionEventType.TOOL_CALLED,
            {
                "tool_name": request.tool_name,
                "arguments_preview": str(request.arguments)[:150],
                "iteration": iteration,
            },
        )

        try:
            dispatch_result, formatted_message = self._dispatcher.dispatch_and_format_for_llm(
                request=request,
                task_id=task.id,
                subtask_id=subtask.id,
                format_result_fn=self._llm.format_tool_result,
            )
            self._remember_tool_outcome(
                recent_tool_outcomes=recent_tool_outcomes,
                request=request,
                succeeded=dispatch_result.succeeded,
                error=dispatch_result.result.error_message,
            )

        except ToolTimeoutError:
            self._remember_tool_outcome(
                recent_tool_outcomes=recent_tool_outcomes,
                request=request,
                succeeded=False,
                error="timeout",
            )
            # Reintento automático con extended_timeout
            log.warning(
                "timeout en tool — reintentando con extended_timeout",
                tool_name=request.tool_name,
                iteration=iteration,
            )

            extended_request = ToolCallRequest(
                tool_call_id=request.tool_call_id,
                tool_name=request.tool_name,
                arguments={**request.arguments, "extended_timeout": True},
            )

            try:
                dispatch_result, formatted_message = self._dispatcher.dispatch_and_format_for_llm(
                    request=extended_request,
                    task_id=task.id,
                    subtask_id=subtask.id,
                    format_result_fn=self._llm.format_tool_result,
                )
                self._remember_tool_outcome(
                    recent_tool_outcomes=recent_tool_outcomes,
                    request=extended_request,
                    succeeded=dispatch_result.succeeded,
                    error=dispatch_result.result.error_message,
                )
            except ToolTimeoutError:
                self._remember_tool_outcome(
                    recent_tool_outcomes=recent_tool_outcomes,
                    request=extended_request,
                    succeeded=False,
                    error="extended-timeout",
                )
                # Timeout incluso con extended — falla la subtask
                log.error(
                    "timeout persistente tras extended_timeout — fallando subtask",
                    tool_name=request.tool_name,
                )
                subtask = subtask.fail()
                task = task.update_subtask(subtask)
                return subtask, task, False

        # Registrar el tool call completado en la subtask
        subtask = subtask.add_tool_call(dispatch_result.tool_call)
        task = task.update_subtask(subtask)

        # Agregar el resultado formateado al historial del LLM
        session.push_message(formatted_message)

        session.record_event(
            SessionEventType.TOOL_RESULT_RECEIVED,
            {
                "tool_name": request.tool_name,
                "success": dispatch_result.succeeded,
                "duration_seconds": dispatch_result.duration_seconds,
                "output_preview": dispatch_result.result.output[:200],
                "iteration": iteration,
            },
        )

        log.debug(
            "tool call completada",
            tool_name=request.tool_name,
            success=dispatch_result.succeeded,
            duration_seconds=dispatch_result.duration_seconds,
        )

        return subtask, task, dispatch_result.succeeded

    def _complete_subtask(
        self,
        task: Task,
        subtask: Subtask,
        session: InMemorySession,
        summary: str,
        iteration: int,
    ) -> tuple[Subtask, Task]:
        """Marca la subtask como completada y registra el resumen final."""
        completed_subtask = subtask.update_reasoning(summary)
        completed_subtask = completed_subtask.complete()
        updated_task = task.update_subtask(completed_subtask)

        session.push_message(Message.assistant(summary))
        session.record_event(
            SessionEventType.SUBTASK_COMPLETED,
            {
                "iteration": iteration,
                "summary_preview": summary[:200],
            },
        )

        return completed_subtask, updated_task

    # ------------------------------------------------------------------
    # Llamada al LLM con manejo de errores
    # ------------------------------------------------------------------

    def _call_llm_safe(
        self,
        session: InMemorySession,
        task: Task,
    ) -> RichLLMResponse | None:
        """
        Llama al LLM con el historial actual, manejando errores.

        Trunca el historial si es necesario antes de llamar.
        Devuelve None si la llamada falla de forma no recuperable,
        lo que hace que el executor falle la subtask activa.
        """
        # Truncar historial si se acerca al context window
        react_max_tokens = min(
            self._settings.llm_max_tokens,
            LLM_DEFAULT_MAX_TOKENS_REACT,
        )
        messages = self._context_builder.truncate_messages_for_context_window(
            messages=session.get_messages(),
            context_window_size=self._llm.get_context_window_size(),
            max_tokens_response=react_max_tokens,
        )

        try:
            return self._llm.complete(
                messages=messages,
                max_tokens=react_max_tokens,
                temperature=self._settings.llm_temperature,
            )
        except Exception as exc:
            logger.error(
                "error fatal llamando al LLM",
                task_id=task.id,
                error_type=type(exc).__name__,
                error=str(exc),
            )
            return None

    # ------------------------------------------------------------------
    # Gestión del límite de iteraciones
    # ------------------------------------------------------------------

    def _handle_limit_reached(
        self,
        task: Task,
        subtask: Subtask,
        iterations_used: int,
        on_limit_reached: Callable[[Subtask, int], LimitDecision] | None,
        log: Any,
    ) -> tuple[Subtask, Task]:
        """
        Maneja el caso cuando el loop ReAct agota su límite de iteraciones.

        Si hay callback del usuario, le pregunta qué hacer.
        Si no hay callback, falla la subtask automáticamente.

        El callback recibe la subtask y el número de iteraciones usadas,
        y devuelve una LimitDecision que indica qué hacer.
        """
        if on_limit_reached is not None:
            decision = on_limit_reached(subtask, iterations_used)
        else:
            failed = subtask.fail()
            task = task.update_subtask(failed)
            get_event_bus().emit(
                AgentEvent.subtask_failed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    reason=f"Límite de {REACT_MAX_ITERATIONS_PER_SUBTASK} iteraciones alcanzado.",
                )
            )
            log.warning(
                "límite de iteraciones alcanzado",
                decision="fail_current",
                iterations_used=iterations_used,
            )
            return failed, task

        log.warning(
            "límite de iteraciones alcanzado",
            decision=decision.value,
            iterations_used=iterations_used,
        )

        if decision == LimitDecision.RETRY:
            # El caller se encargará de reintentar — devolvemos sin cambios
            # para que el caller decida cómo reiniciar la subtask
            return subtask, task

        elif decision == LimitDecision.SKIP:
            # ══════════════════════════════════════════════════════════════
            # CORRECCIÓN: La subtask está en IN_PROGRESS cuando llegamos aquí.
            # Las transiciones válidas desde IN_PROGRESS son solo COMPLETED
            # y FAILED. Usar skip() causaba InvalidStatusTransition porque
            # SKIPPED solo es válido desde PENDING.
            #
            # Usamos fail() que es la transición correcta para "no se completó".
            # ══════════════════════════════════════════════════════════════
            failed = subtask.fail()
            task = task.update_subtask(failed)
            get_event_bus().emit(
                AgentEvent.subtask_failed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    reason=f"Omitida por decisión del usuario tras {iterations_used} iteraciones.",
                )
            )
            return failed, task

        elif decision == LimitDecision.CANCEL:
            # Igual que SKIP: usamos fail() porque la subtask está IN_PROGRESS
            failed = subtask.fail()
            task = task.update_subtask(failed)
            task = task.cancel()
            get_event_bus().emit(AgentEvent.task_cancelled(task_id=task.id))
            return failed, task

        else:
            # LimitDecision desconocida o ausente — fallamos la subtask
            failed = subtask.fail()
            task = task.update_subtask(failed)
            get_event_bus().emit(
                AgentEvent.subtask_failed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    reason=f"Límite de {REACT_MAX_ITERATIONS_PER_SUBTASK} iteraciones alcanzado.",
                )
            )
            return failed, task

    # ------------------------------------------------------------------
    # Iteraciones dinámicas según complejidad del plan
    # ------------------------------------------------------------------

    @staticmethod
    def _max_iterations_for_plan(subtask_count: int) -> int:
        """
        Calcula el máximo de iteraciones ReAct por subtask según la complejidad del plan.

        Un plan con pocas subtasks (1-2) indica una tarea simple donde cada
        subtask debería resolverse en 2-3 iteraciones. Permitir 15 iteraciones
        en este caso solo invita al agente a vagabundear, explorar y reverificar
        sin necesidad.

        Un plan con muchas subtasks (6+) indica tarea compleja donde cada
        subtask puede requerir más intentos para resolver problemas inesperados.

        MAPEO:
          1-2 subtasks → REACT_MAX_ITERATIONS_SIMPLE (8)
          3-5 subtasks → REACT_MAX_ITERATIONS_MEDIUM  (12)
          6+  subtasks → REACT_MAX_ITERATIONS_PER_SUBTASK (15)

        Parámetros:
            subtask_count: Número total de subtasks en el plan.

        Returns:
            Máximo de iteraciones permitidas por subtask.
        """
        if subtask_count <= 2:
            return REACT_MAX_ITERATIONS_SIMPLE
        if subtask_count <= 5:
            return REACT_MAX_ITERATIONS_MEDIUM
        return REACT_MAX_ITERATIONS_PER_SUBTASK

    # ------------------------------------------------------------------
    # Detección de bucles
    # ------------------------------------------------------------------

    @staticmethod
    def _is_stuck_in_loop(
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
        window: int = _LOOP_DETECTION_WINDOW,
    ) -> bool:
        """
        Detecta si el agente está repitiendo las mismas tool calls SIN PROGRESO.

        Un bucle se detecta cuando:
          a) Las últimas N tool calls son idénticas Y al menos una falló.
             (Dos verificaciones exitosas idénticas NO son un bucle — el agente
             puede legítimamente ejecutar py_compile dos veces para confirmar.)
          b) Las últimas N tool calls son todas fallos de la misma familia.

        El parámetro `window` permite reducir la ventana de detección
        después de ya haber intervenido — si el agente sigue repitiendo
        tras una intervención, lo detectamos más rápido (window=2).

        Parámetros:
            recent_tool_outcomes: Lista de resultados recientes de tools.
            window: Cuántas tool calls recientes comparar (default: 3,
                    post-intervención: 2).

        Returns:
            True si se detecta un bucle, False si el agente progresa.
        """
        if len(recent_tool_outcomes) < window:
            return False

        last_n = recent_tool_outcomes[-window:]

        # ── Detección por fingerprint exacto ──────────────────────────
        # Solo si al menos una call FALLÓ. Dos calls exitosos idénticos
        # son legítimos (ej: py_compile verificando que el archivo compila).
        has_failures = any(not item.success for item in last_n)
        if has_failures and len({item.fingerprint for item in last_n}) == 1:
            return True

        # ── Detección por familia de error ────────────────────────────
        # Todas las calls en la ventana son fallos de la misma familia
        # (misma tool, distinto detalle menor en los argumentos)
        failure_families = [
            item.family
            for item in last_n
            if not item.success
        ]
        return (
            len(failure_families) == window
            and len(set(failure_families)) == 1
        )

    @staticmethod
    def _is_wasting_iterations(
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
    ) -> bool:
        """
        Detecta exploración o reverificación redundante tras haber avanzado.

        Caso típico:
          1. El agente modifica un archivo o genera una salida.
          2. La verifica con éxito una vez.
          3. Sigue listando, leyendo o ejecutando la misma verificación
             una y otra vez sin aportar evidencia nueva.
        """
        if len(recent_tool_outcomes) < _EFFICIENCY_WINDOW:
            return False

        window = recent_tool_outcomes[-_EFFICIENCY_WINDOW:]
        if not all(item.success for item in window):
            return False

        if any(item.is_mutation for item in window):
            return False

        if not any(item.is_mutation and item.success for item in recent_tool_outcomes[:-1]):
            return False

        if not all(item.is_observation or item.is_verification for item in window):
            return False

        repeated_families = {item.family for item in window}
        verification_count = sum(1 for item in window if item.is_verification)
        observation_count = sum(1 for item in window if item.is_observation)

        return (
            len(repeated_families) <= 2
            or verification_count >= 3
            or observation_count >= 3
        )

    def _inject_loop_intervention(self, session: InMemorySession) -> None:
        """
        Inyecta un mensaje genérico para romper el bucle del agente.

        NOTA: Preferir _select_intervention_message() para intervenciones
        context-aware que son más efectivas. Este método se conserva
        para backward compatibility.
        """
        session.push_message(
            Message.user(_STALL_INTERVENTION_MSG)
        )

    @staticmethod
    def _select_intervention_message(
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
    ) -> str:
        """
        Selecciona el mensaje de intervención más efectivo según el patrón
        de error detectado en las tool calls recientes.

        ESTRATEGIA:
          En vez de siempre inyectar el mensaje genérico "cambiar de estrategia",
          analizamos los últimos errores para dar una indicación ESPECÍFICA al LLM
          sobre qué debe cambiar. Esto es mucho más efectivo porque los modelos
          pequeños suelen ignorar indicaciones genéricas pero sí responden a
          instrucciones concretas y accionables.

        PATRONES RECONOCIDOS:
          - Argumentos inválidos repetidos → decirle exactamente qué parámetro omitir
          - Tool no encontrada repetida → recordarle las tools disponibles
          - Mismo comando fallido repetido → sugerir cambio de estrategia concreto
          - Default → intervención genérica

        Parámetros:
            recent_tool_outcomes: Lista de resultados recientes de tools.

        Returns:
            String JSON del mensaje de intervención a inyectar.
        """
        if not recent_tool_outcomes:
            return _STALL_INTERVENTION_MSG

        # Analizar los últimos errores para detectar el patrón
        recent_errors = [
            item for item in recent_tool_outcomes[-4:]
            if not item.success
        ]

        if not recent_errors:
            return _STALL_INTERVENTION_MSG

        # Patrón 1: Errores de argumentos inválidos repetidos
        # Detectado por el fingerprint que contiene "invalid" o "timeout"
        arg_validation_errors = [
            e for e in recent_errors
            if "timeout" in e.fingerprint.lower()
            or "inválid" in e.fingerprint.lower()
            or "invalid" in e.fingerprint.lower()
        ]
        if len(arg_validation_errors) >= 2:
            return _ARG_VALIDATION_INTERVENTION_MSG

        # Patrón 2: Tool no encontrada repetida
        tool_not_found_errors = [
            e for e in recent_errors
            if "not-found" in e.fingerprint.lower()
            or "no encontrada" in e.fingerprint.lower()
        ]
        if len(tool_not_found_errors) >= 2:
            return _TOOL_NOT_FOUND_INTERVENTION_MSG

        # Default: intervención genérica
        return _STALL_INTERVENTION_MSG

    def _inject_efficiency_intervention(self, session: InMemorySession) -> None:
        """Le indica al agente que deje de explorar y cierre la subtask."""
        session.push_message(Message.user(_EFFICIENCY_INTERVENTION_MSG))

    def _remember_tool_outcome(
        self,
        recent_tool_outcomes: list[ToolOutcomeSnapshot],
        request: ToolCallRequest,
        succeeded: bool,
        error: str | None,
    ) -> None:
        """Guarda una fingerprint compacta del resultado de una tool call."""
        family = self._tool_family(request)
        base = f"{request.tool_name}:{self._arg_fingerprint(request.arguments)}"
        if succeeded:
            fingerprint = f"ok|{base}"
        else:
            fingerprint = f"fail|{request.tool_name}|{self._error_fingerprint(error)}|{base}"

        snapshot = ToolOutcomeSnapshot(
            fingerprint=fingerprint,
            family=family,
            success=succeeded,
            tool_name=request.tool_name,
            is_mutation=self._is_mutating_request(request),
            is_observation=self._is_observation_request(request),
            is_verification=self._is_verification_request(request),
        )

        recent_tool_outcomes.append(snapshot)
        if len(recent_tool_outcomes) > _EFFICIENCY_WINDOW * 3:
            recent_tool_outcomes.pop(0)

    # ------------------------------------------------------------------
    # Inicialización de la sesión por subtask
    # ------------------------------------------------------------------

    def _initialize_subtask_session(
        self,
        task: Task,
        subtask: Subtask,
        previous_subtask_summary: str | None = None,
    ) -> InMemorySession:
        """
        Crea e inicializa la sesión en memoria para una subtask.

        El historial inicial contiene exactamente dos mensajes:
          1. Sistema: instrucciones del agente + tools disponibles + contexto
             + resumen de la subtask anterior (si existe)
          2. Usuario: descripción de la subtask a completar

        El resumen de la subtask anterior permite continuidad entre subtasks
        sin pasar el historial completo — el agente sabe qué archivos se
        crearon y qué se verificó sin tener que explorar de nuevo.
        """
        workspace_id = self._store.get_active_workspace_id() or ""

        session = InMemorySession(task=task, workspace_id=workspace_id)
        session.set_active_subtask(subtask.id)

        # Mensaje 1: sistema con instrucciones completas + contexto anterior
        system_msg = self._context_builder.build_system_message(
            subtask_description=subtask.description,
            task_prompt=task.prompt,
            working_dir=os.getcwd(),
            max_retries=REACT_MAX_TOOL_RETRIES,
            previous_subtask_summary=previous_subtask_summary,
        )
        session.push_message(system_msg)

        # Mensaje 2: la subtask actual como instrucción del usuario
        subtask_msg = Message.user(
            f"Subtask {subtask.order + 1} de {len(task.subtasks)}: "
            f"{subtask.description}"
        )
        session.push_message(subtask_msg)

        session.record_event(
            SessionEventType.SUBTASK_STARTED,
            {
                "subtask_id": subtask.id,
                "order": subtask.order,
                "description": subtask.description,
            },
        )

        return session

    # ------------------------------------------------------------------
    # Generación del resumen final
    # ------------------------------------------------------------------

    def _generate_completion_summary(self, task: Task) -> str:
        """
        Genera un resumen del resultado de la task completa.

        Construye el resumen a partir de los razonamientos finales
        de cada subtask. No llama al LLM — usa lo que el agente
        ya generó durante la ejecución.

        Esto evita un costo adicional de tokens solo para el resumen.
        """
        completed = task.completed_subtasks

        if not completed:
            return "Task completada sin subtasks registradas."

        lines = [f"Task completada: {task.prompt[:100]}"]
        lines.append("")
        lines.append(f"Plan ejecutado ({len(completed)} pasos):")

        for subtask in completed:
            # Usamos el último razonamiento del agente como resumen del paso
            step_summary = subtask.reasoning or subtask.description
            # Tomamos solo la primera línea si es muy largo
            first_line = step_summary.split("\n")[0][:120]
            lines.append(f"  ✓ [{subtask.order + 1}] {first_line}")

        lines.append("")
        lines.append(
            f"Tokens totales: {task.token_usage.total_tokens:,} "
            f"(~${task.token_usage.estimated_cost_usd:.4f} USD)"
        )

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Persistencia incremental
    # ------------------------------------------------------------------

    def _persist_task_progress(self, task: Task) -> None:
        """
        Persiste el estado actual de la task a disco.

        Llamado después de cada subtask completada para que el trabajo
        no se pierda si el proceso termina inesperadamente.

        No falla si no puede persistir — el trabajo en RAM sigue siendo
        válido. El error se loggea para debug pero no interrumpe la ejecución.
        """
        if task.project_id is None:
            # Task suelta sin proyecto — no tenemos workspace_id para la ruta
            return

        try:
            workspace_id = self._store.get_active_workspace_id()
            if workspace_id:
                self._store.tasks.save(task=task, workspace_id=workspace_id)
                logger.debug(
                    "progreso de task persistido",
                    task_id=task.id,
                    progress_pct=task.progress_percentage,
                )
        except Exception as exc:
            # Fallo de persistencia no es fatal — el trabajo sigue en RAM
            logger.warning(
                "no se pudo persistir el progreso de la task",
                task_id=task.id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Emisión de eventos de fin de subtask
    # ------------------------------------------------------------------

    def _emit_subtask_end_event(
        self,
        task: Task,
        subtask: Subtask,
        iterations: int,
        duration: float,
    ) -> None:
        """Emite el evento correcto según el estado final de la subtask."""
        if subtask.status == SubtaskStatus.COMPLETED:
            get_event_bus().emit(
                AgentEvent.subtask_completed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    duration_seconds=round(duration, 3),
                    react_iterations=iterations,
                )
            )
        elif subtask.status == SubtaskStatus.FAILED:
            get_event_bus().emit(
                AgentEvent.subtask_failed(
                    task_id=task.id,
                    subtask_id=subtask.id,
                    reason="Subtask falló durante la ejecución.",
                )
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _arg_fingerprint(arguments: dict[str, Any]) -> str:
        """
        Genera un fingerprint corto de los argumentos de una tool call.

        Usado para detectar tool calls repetidas en _is_stuck_in_loop().
        No necesita ser 100% único — solo suficientemente distintivo
        para detectar cuando el agente repite exactamente los mismos argumentos.
        """
        def normalize(value: Any, key: str | None = None) -> Any:
            if isinstance(value, dict):
                return {
                    str(k): normalize(v, str(k))
                    for k, v in sorted(value.items())
                    if str(k) not in {"content", "patch"}
                }
            if isinstance(value, list):
                return [normalize(v) for v in value[:6]]
            if isinstance(value, str):
                text = value.strip()
                if key in {"path", "working_dir"}:
                    return os.path.basename(text.rstrip("/")) or text[-24:]
                if key == "command":
                    return text[:48]
                return text[:32]
            return value

        try:
            # Serialización ordenada para que el mismo dict siempre
            # produzca el mismo fingerprint
            serialized = json.dumps(normalize(arguments), sort_keys=True)
            # Tomamos solo los primeros 64 chars para mantenerlo corto
            return serialized[:64]
        except (TypeError, ValueError):
            return str(arguments)[:64]

    @staticmethod
    def _error_fingerprint(error: str | None) -> str:
        """Reduce errores de tool a familias cortas para detectar repeticiones."""
        if not error:
            return "unknown"

        normalized = error.lower()
        if "no es un archivo" in normalized:
            return "path-not-file"
        if "operación" in normalized and "inválida" in normalized:
            return "invalid-operation"
        if "timeout" in normalized:
            return "timeout"
        if "exit code" in normalized:
            return "command-failed"
        return normalized[:32]

    @staticmethod
    def _tool_family(request: ToolCallRequest) -> str:
        """Agrupa tool calls en familias semánticas para detectar repeticiones."""
        arguments = request.arguments

        if request.tool_name == "file":
            operation = str(arguments.get("operation", ""))
            path = str(arguments.get("path", "")).rstrip("/")
            return f"file:{operation}:{os.path.basename(path) or path}"

        if request.tool_name == "shell":
            command = str(arguments.get("command", "")).strip()
            tokens = command.split()
            if not tokens:
                return "shell:empty"

            first = tokens[0]
            if first.startswith("python") and len(tokens) > 1:
                for token in tokens[1:]:
                    if token.endswith(".py"):
                        return f"shell:{first}:{os.path.basename(token)}"
                return f"shell:{first}"

            if first in {"sed", "cat", "head", "tail"} and tokens:
                for token in reversed(tokens):
                    if "." in token and not token.startswith("-"):
                        return f"shell:{first}:{os.path.basename(token)}"

            return f"shell:{first}"

        return request.tool_name

    @staticmethod
    def _is_mutating_request(request: ToolCallRequest) -> bool:
        """Heurística conservadora para detectar acciones que cambian estado."""
        arguments = request.arguments

        if request.tool_name == "file":
            return str(arguments.get("operation", "")) in {"write", "append", "patch", "delete"}

        if request.tool_name != "shell":
            return False

        command = str(arguments.get("command", "")).lower()
        mutating_markers = (
            " --output ",
            " --out ",
            ">",
            ">>",
            "tee ",
            "sed -i",
            "mkdir ",
            "touch ",
            "mv ",
            "cp ",
            "rm ",
            "apply_patch",
            "git apply",
            "patch ",
        )
        return any(marker in command for marker in mutating_markers)

    @staticmethod
    def _is_observation_request(request: ToolCallRequest) -> bool:
        """Acciones de exploración/lectura que no suelen aportar progreso nuevo."""
        arguments = request.arguments

        if request.tool_name == "file":
            return str(arguments.get("operation", "")) in {"read", "list", "exists"}

        if request.tool_name != "shell":
            return False

        command = str(arguments.get("command", "")).strip().lower()
        observation_prefixes = (
            "ls", "find", "tree", "pwd", "cat", "sed", "head", "tail",
            "rg", "grep", "fd", "stat",
        )
        return command.startswith(observation_prefixes)

    @staticmethod
    def _is_verification_request(request: ToolCallRequest) -> bool:
        """Acciones que suelen servir para verificar que un cambio ya funciona."""
        if request.tool_name != "shell":
            return False

        command = str(request.arguments.get("command", "")).strip().lower()
        verification_prefixes = (
            "python ", "python3 ", "pytest", "uv run", "cargo test",
            "npm test", "pnpm test", "yarn test",
        )
        return command.startswith(verification_prefixes)

    @staticmethod
    def _task_duration(task: Task) -> float:
        """Duración total de la task en segundos. 0.0 si no tiene timestamps."""
        if task.completed_at is None or task.created_at is None:
            return 0.0
        delta = task.completed_at - task.created_at
        return round(delta.total_seconds(), 3)
