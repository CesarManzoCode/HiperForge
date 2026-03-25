"""
Tests unitarios para ToolDispatcher — dispatch, ciclo de vida, errores.
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, PropertyMock

from hiperforge.application.services.tool_dispatcher import ToolDispatcher, DispatchResult
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.exceptions import ToolNotFound, ToolTimeoutError
from hiperforge.infrastructure.llm.base import ToolCallRequest
from hiperforge.tools.base import ToolRegistry


@pytest.fixture
def mock_registry():
    registry = MagicMock(spec=ToolRegistry)
    mock_tool = MagicMock()
    mock_tool.execute_safe.return_value = ToolResult.success(
        tool_call_id="tc-1",
        output="Archivo creado exitosamente.",
    )
    registry.get.return_value = mock_tool
    registry.tool_names = ["shell", "file"]
    return registry


@pytest.fixture
def dispatcher(mock_registry):
    return ToolDispatcher(registry=mock_registry)


class TestDispatch:

    def test_dispatch_exitoso(self, dispatcher):
        request = ToolCallRequest(
            tool_call_id="tc-1",
            tool_name="file",
            arguments={"operation": "write", "path": "test.py", "content": "pass"},
        )
        result = dispatcher.dispatch(request, task_id="t1", subtask_id="s1")
        assert isinstance(result, DispatchResult)
        assert result.succeeded

    def test_dispatch_tool_no_encontrada(self, dispatcher, mock_registry):
        mock_registry.get.side_effect = ToolNotFound("super_tool")
        request = ToolCallRequest(
            tool_call_id="tc-1",
            tool_name="super_tool",
            arguments={},
        )
        result = dispatcher.dispatch(request, task_id="t1", subtask_id="s1")
        assert not result.succeeded

    def test_dispatch_timeout_se_propaga(self, dispatcher, mock_registry):
        mock_tool = MagicMock()
        mock_tool.execute_safe.side_effect = ToolTimeoutError(
            tool_name="shell", timeout_seconds=30.0,
        )
        mock_registry.get.return_value = mock_tool
        request = ToolCallRequest(
            tool_call_id="tc-1",
            tool_name="shell",
            arguments={"command": "sleep 100"},
        )
        with pytest.raises(ToolTimeoutError):
            dispatcher.dispatch(request, task_id="t1", subtask_id="s1")

    def test_dispatch_registra_duracion(self, dispatcher):
        request = ToolCallRequest(
            tool_call_id="tc-1",
            tool_name="file",
            arguments={"operation": "read", "path": "x.py"},
        )
        result = dispatcher.dispatch(request, task_id="t1", subtask_id="s1")
        assert result.duration_seconds >= 0.0


class TestDispatchAndFormat:

    def test_dispatch_and_format_retorna_mensaje(self, dispatcher):
        request = ToolCallRequest(
            tool_call_id="tc-1",
            tool_name="file",
            arguments={"operation": "write", "path": "test.py", "content": "pass"},
        )

        def mock_format(tool_call_id, tool_name, output, success):
            from hiperforge.domain.value_objects.message import Message
            return Message.user(f'{{"tool_result": "{output}"}}')

        result, msg = dispatcher.dispatch_and_format_for_llm(
            request=request,
            task_id="t1",
            subtask_id="s1",
            format_result_fn=mock_format,
        )
        assert result.succeeded
        assert msg is not None
        assert msg.role.value == "user"


class TestDispatchResult:

    def test_succeeded_true(self):
        result = DispatchResult(
            tool_call=MagicMock(),
            result=ToolResult.success(tool_call_id="tc-1", output="ok"),
            duration_seconds=0.5,
        )
        assert result.succeeded

    def test_succeeded_false(self):
        result = DispatchResult(
            tool_call=MagicMock(),
            result=ToolResult.failure(tool_call_id="tc-1", error_message="error"),
            duration_seconds=0.1,
        )
        assert not result.succeeded

    def test_was_timeout(self):
        result = DispatchResult(
            tool_call=MagicMock(),
            result=ToolResult.failure(tool_call_id="tc-1", error_message="timeout"),
            duration_seconds=30.0,
            was_timeout=True,
        )
        assert result.was_timeout
