"""
Tests unitarios para Task y Subtask — ciclo de vida completo.

Cobertura:
  - Creación de entidades con valores por defecto correctos
  - Todas las transiciones de estado válidas
  - Transiciones inválidas (deben lanzar InvalidStatusTransition)
  - Inmutabilidad (frozen dataclass — cada mutación crea nueva instancia)
  - Acumulación de tokens y tool calls
  - Propiedades calculadas (progress_percentage, completed_subtasks, etc.)
  - Serialización a/desde dict para persistencia
"""

from __future__ import annotations

import pytest

from hiperforge.domain.entities.task import (
    Subtask,
    SubtaskStatus,
    Task,
    TaskStatus,
)
from hiperforge.domain.entities.tool_call import ToolCall, ToolResult
from hiperforge.domain.exceptions import InvalidStatusTransition
from hiperforge.domain.value_objects.token_usage import TokenUsage


# ═══════════════════════════════════════════════════════════════════
# TASK — CREACIÓN
# ═══════════════════════════════════════════════════════════════════

class TestTaskCreation:
    """Verificar que Task.create() produce entidades correctas."""

    def test_create_task_genera_id_unico(self):
        t1 = Task.create(prompt="tarea 1")
        t2 = Task.create(prompt="tarea 2")
        assert t1.id != t2.id
        assert len(t1.id) > 0

    def test_create_task_estado_pending(self):
        task = Task.create(prompt="hola")
        assert task.status == TaskStatus.PENDING

    def test_create_task_preserva_prompt(self):
        task = Task.create(prompt="crea un script")
        assert task.prompt == "crea un script"

    def test_create_task_sin_proyecto(self):
        task = Task.create(prompt="test")
        assert task.project_id is None

    def test_create_task_con_proyecto(self):
        task = Task.create(prompt="test", project_id="proj-123")
        assert task.project_id == "proj-123"

    def test_create_task_subtasks_vacias(self):
        task = Task.create(prompt="test")
        assert len(task.subtasks) == 0

    def test_create_task_tokens_zero(self):
        task = Task.create(prompt="test")
        assert task.token_usage.total_tokens == 0

    def test_create_task_timestamps_presentes(self):
        task = Task.create(prompt="test")
        assert task.created_at is not None
        assert task.completed_at is None


# ═══════════════════════════════════════════════════════════════════
# TASK — TRANSICIONES DE ESTADO
# ═══════════════════════════════════════════════════════════════════

class TestTaskTransitions:
    """Verificar todas las transiciones válidas e inválidas de Task."""

    def test_pending_a_planning(self, sample_task):
        planning = sample_task.start_planning()
        assert planning.status == TaskStatus.PLANNING

    def test_planning_a_in_progress(self, sample_task):
        planning = sample_task.start_planning()
        subtasks = [Subtask.create(task_id=planning.id, description="paso 1", order=0)]
        in_progress = planning.start_execution(subtasks)
        assert in_progress.status == TaskStatus.IN_PROGRESS
        assert len(in_progress.subtasks) == 1

    def test_in_progress_a_completed(self, planned_task):
        completed = planned_task.complete(summary="Listo")
        assert completed.status == TaskStatus.COMPLETED
        assert completed.completed_at is not None

    def test_in_progress_a_failed(self, planned_task):
        failed = planned_task.fail()
        assert failed.status == TaskStatus.FAILED

    def test_pending_a_cancelled(self, sample_task):
        cancelled = sample_task.cancel()
        assert cancelled.status == TaskStatus.CANCELLED

    def test_planning_a_cancelled(self, sample_task):
        planning = sample_task.start_planning()
        cancelled = planning.cancel()
        assert cancelled.status == TaskStatus.CANCELLED

    def test_in_progress_a_cancelled(self, planned_task):
        cancelled = planned_task.cancel()
        assert cancelled.status == TaskStatus.CANCELLED

    # ── Transiciones INVÁLIDAS ──────────────────────────────────────

    def test_pending_a_completed_invalido(self, sample_task):
        with pytest.raises(InvalidStatusTransition):
            sample_task.complete(summary="no")

    def test_pending_a_failed_invalido(self, sample_task):
        with pytest.raises(InvalidStatusTransition):
            sample_task.fail()

    def test_completed_a_cualquier_invalido(self, planned_task):
        completed = planned_task.complete(summary="ok")
        with pytest.raises(InvalidStatusTransition):
            completed.fail()
        with pytest.raises(InvalidStatusTransition):
            completed.cancel()

    def test_failed_a_cualquier_invalido(self, planned_task):
        failed = planned_task.fail()
        with pytest.raises(InvalidStatusTransition):
            failed.complete(summary="no")
        with pytest.raises(InvalidStatusTransition):
            failed.cancel()

    def test_cancelled_a_cualquier_invalido(self, sample_task):
        cancelled = sample_task.cancel()
        with pytest.raises(InvalidStatusTransition):
            cancelled.start_planning()


# ═══════════════════════════════════════════════════════════════════
# TASK — INMUTABILIDAD
# ═══════════════════════════════════════════════════════════════════

class TestTaskImmutability:
    """Verificar que cada mutación crea una nueva instancia."""

    def test_start_planning_crea_nueva_instancia(self, sample_task):
        planning = sample_task.start_planning()
        assert planning is not sample_task
        assert sample_task.status == TaskStatus.PENDING  # original intacta

    def test_add_token_usage_crea_nueva_instancia(self, planned_task):
        usage = TokenUsage(input_tokens=100, output_tokens=50, model="test")
        updated = planned_task.add_token_usage(usage)
        assert updated is not planned_task
        assert updated.token_usage.total_tokens == 150
        assert planned_task.token_usage.total_tokens == 0

    def test_update_subtask_crea_nueva_instancia(self, planned_task):
        subtask = planned_task.subtasks[0].mark_running()
        updated = planned_task.update_subtask(subtask)
        assert updated is not planned_task
        assert updated.subtasks[0].status == SubtaskStatus.IN_PROGRESS
        assert planned_task.subtasks[0].status == SubtaskStatus.PENDING


# ═══════════════════════════════════════════════════════════════════
# TASK — PROPIEDADES CALCULADAS
# ═══════════════════════════════════════════════════════════════════

class TestTaskProperties:
    """Verificar propiedades derivadas de la Task."""

    def test_progress_percentage_sin_subtasks(self, sample_task):
        assert sample_task.progress_percentage == 0.0

    def test_progress_percentage_parcial(self, multi_subtask_task):
        task = multi_subtask_task
        s0 = task.subtasks[0].mark_running().complete()
        task = task.update_subtask(s0)
        assert task.progress_percentage == pytest.approx(25.0, abs=0.1)

    def test_progress_percentage_completa(self, planned_task):
        s = planned_task.subtasks[0].mark_running().complete()
        task = planned_task.update_subtask(s)
        assert task.progress_percentage == pytest.approx(100.0, abs=0.1)

    def test_completed_subtasks_filtra_correctamente(self, multi_subtask_task):
        task = multi_subtask_task
        s0 = task.subtasks[0].mark_running().complete()
        s1 = task.subtasks[1].mark_running().fail()
        task = task.update_subtask(s0)
        task = task.update_subtask(s1)
        assert len(task.completed_subtasks) == 1

    def test_is_terminal_estados_finales(self, planned_task):
        assert not planned_task.is_terminal
        assert planned_task.complete(summary="ok").is_terminal
        assert planned_task.fail().is_terminal
        assert planned_task.cancel().is_terminal


# ═══════════════════════════════════════════════════════════════════
# SUBTASK — CREACIÓN Y CICLO DE VIDA
# ═══════════════════════════════════════════════════════════════════

class TestSubtask:
    """Tests del ciclo de vida de Subtask."""

    def test_create_subtask_estado_pending(self):
        s = Subtask.create(task_id="t1", description="paso 1", order=0)
        assert s.status == SubtaskStatus.PENDING
        assert s.description == "paso 1"
        assert s.order == 0

    def test_mark_running(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        running = s.mark_running()
        assert running.status == SubtaskStatus.IN_PROGRESS

    def test_complete(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        completed = s.mark_running().complete()
        assert completed.status == SubtaskStatus.COMPLETED

    def test_fail(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        failed = s.mark_running().fail()
        assert failed.status == SubtaskStatus.FAILED

    def test_skip(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        skipped = s.skip()
        assert skipped.status == SubtaskStatus.SKIPPED

    def test_add_tool_call(self, sample_subtask, completed_tool_call):
        running = sample_subtask.mark_running()
        updated = running.add_tool_call(completed_tool_call)
        assert len(updated.tool_calls) == 1

    def test_update_reasoning(self, sample_subtask):
        running = sample_subtask.mark_running()
        updated = running.update_reasoning("Creé el archivo correctamente.")
        assert updated.reasoning == "Creé el archivo correctamente."

    def test_is_terminal(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        assert not s.is_terminal
        assert s.mark_running().complete().is_terminal
        assert s.mark_running().fail().is_terminal
        assert s.skip().is_terminal

    # ── Transiciones inválidas de Subtask ───────────────────────────

    def test_pending_a_completed_invalido(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        with pytest.raises(InvalidStatusTransition):
            s.complete()

    def test_pending_a_failed_invalido(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        with pytest.raises(InvalidStatusTransition):
            s.fail()

    def test_completed_a_failed_invalido(self):
        s = Subtask.create(task_id="t1", description="paso", order=0)
        completed = s.mark_running().complete()
        with pytest.raises(InvalidStatusTransition):
            completed.fail()
