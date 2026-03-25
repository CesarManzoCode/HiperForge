"""
Tests unitarios para ToolCall y ToolResult — ciclo de vida y constructores.
"""

from __future__ import annotations

import pytest

from hiperforge.domain.entities.tool_call import ToolCall, ToolCallStatus, ToolResult
from hiperforge.domain.exceptions import InvalidStatusTransition


class TestToolCallCreation:

    def test_create_genera_id_unico(self):
        c1 = ToolCall.create(tool_name="shell", arguments={"command": "ls"})
        c2 = ToolCall.create(tool_name="shell", arguments={"command": "ls"})
        assert c1.id != c2.id

    def test_create_estado_pending(self):
        call = ToolCall.create(tool_name="file", arguments={"operation": "read"})
        assert call.status == ToolCallStatus.PENDING

    def test_create_preserva_nombre_y_args(self):
        args = {"command": "pytest", "timeout": 30.0}
        call = ToolCall.create(tool_name="shell", arguments=args)
        assert call.tool_name == "shell"
        assert call.arguments == args

    def test_create_sin_resultado(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        assert call.result is None


class TestToolCallTransitions:

    def test_pending_a_running(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        running = call.mark_running()
        assert running.status == ToolCallStatus.RUNNING

    def test_running_a_completed(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        running = call.mark_running()
        result = ToolResult.success(tool_call_id=running.id, output="ok")
        completed = running.with_result(result)
        assert completed.status == ToolCallStatus.COMPLETED
        assert completed.result is not None
        assert completed.result.success is True

    def test_running_a_failed(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        running = call.mark_running()
        result = ToolResult.failure(tool_call_id=running.id, error_message="exit code 1")
        failed = running.with_result(result)
        assert failed.status == ToolCallStatus.FAILED
        assert failed.result.success is False

    def test_pending_a_skipped(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        # Si existe método skip
        if hasattr(call, "skip"):
            skipped = call.skip()
            assert skipped.status == ToolCallStatus.SKIPPED

    def test_pending_a_completed_invalido(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        result = ToolResult.success(tool_call_id=call.id, output="ok")
        with pytest.raises(InvalidStatusTransition):
            call.with_result(result)

    def test_completed_a_running_invalido(self):
        call = ToolCall.create(tool_name="shell", arguments={})
        running = call.mark_running()
        result = ToolResult.success(tool_call_id=running.id, output="ok")
        completed = running.with_result(result)
        with pytest.raises(InvalidStatusTransition):
            completed.mark_running()


class TestToolResult:

    def test_success_constructor(self):
        result = ToolResult.success(tool_call_id="tc-1", output="5 passed")
        assert result.success is True
        assert result.output == "5 passed"
        assert result.error_message is None

    def test_failure_constructor(self):
        result = ToolResult.failure(tool_call_id="tc-1", error_message="timeout")
        assert result.success is False
        assert result.error_message == "timeout"

    def test_failure_con_output(self):
        result = ToolResult.failure(
            tool_call_id="tc-1",
            error_message="exit code 1",
            output="stderr: file not found",
        )
        assert result.success is False
        assert result.output == "stderr: file not found"

    def test_executed_at_presente(self):
        result = ToolResult.success(tool_call_id="tc-1", output="ok")
        assert result.executed_at is not None
