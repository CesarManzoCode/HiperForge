"""
Tests unitarios para EventBus — emisión, listeners, aislamiento de errores.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from hiperforge.core.events import AgentEvent, EventType, get_event_bus, EventBus


class TestEventBus:

    def test_emit_llama_listener(self):
        bus = EventBus()
        listener = MagicMock()
        bus.subscribe(EventType.TASK_STARTED, listener)
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        bus.emit(event)
        listener.assert_called_once()

    def test_listener_recibe_evento(self):
        bus = EventBus()
        captured = []
        bus.subscribe(EventType.TASK_STARTED, lambda e: captured.append(e))
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        bus.emit(event)
        assert len(captured) == 1
        assert captured[0].data["task_id"] == "t1"

    def test_multiples_listeners(self):
        bus = EventBus()
        l1, l2 = MagicMock(), MagicMock()
        bus.subscribe(EventType.TASK_STARTED, l1)
        bus.subscribe(EventType.TASK_STARTED, l2)
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        bus.emit(event)
        l1.assert_called_once()
        l2.assert_called_once()

    def test_listener_de_otro_tipo_no_recibe(self):
        bus = EventBus()
        listener = MagicMock()
        bus.subscribe(EventType.TASK_COMPLETED, listener)
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        bus.emit(event)
        listener.assert_not_called()

    def test_listener_que_explota_no_rompe_el_bus(self):
        bus = EventBus()
        bad_listener = MagicMock(side_effect=Exception("boom"))
        good_listener = MagicMock()
        bus.subscribe(EventType.TASK_STARTED, bad_listener)
        bus.subscribe(EventType.TASK_STARTED, good_listener)
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        bus.emit(event)  # no debe lanzar excepción
        good_listener.assert_called_once()


class TestAgentEvent:

    def test_task_started_tiene_tipo(self):
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        assert event.event_type == EventType.TASK_STARTED

    def test_task_completed_tiene_datos(self):
        event = AgentEvent.task_completed(
            task_id="t1",
            duration_seconds=10.5,
            total_tokens=1500,
            estimated_cost_usd=0.005,
        )
        assert event.data["total_tokens"] == 1500

    def test_subtask_started_tiene_orden(self):
        event = AgentEvent.subtask_started(
            task_id="t1", subtask_id="s1", order=0, description="paso 1",
        )
        assert event.data["order"] == 0

    def test_evento_tiene_timestamp(self):
        event = AgentEvent.task_started(task_id="t1", prompt="test", project_id=None)
        assert event.timestamp is not None
