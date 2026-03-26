from __future__ import annotations

import importlib
from unittest.mock import patch

from hiperforge.application.services.executor import ExecutorService
from hiperforge.domain.entities.task import Subtask, Task, TaskStatus, SubtaskStatus
from hiperforge.tools.file_ops import FileTool
from hiperforge.tools.shell import ShellTool


def test_shell_tool_usa_tool_call_id_activo_en_resultado() -> None:
    tool = ShellTool()

    with patch("subprocess.run") as run_mock:
        run_mock.return_value.returncode = 0
        run_mock.return_value.stdout = "ok"
        run_mock.return_value.stderr = ""

        result = tool.execute_safe(
            {"command": "ls"},
            tool_call_id="tc-real-123",
        )

    assert result.success
    assert result.tool_call_id == "tc-real-123"


def test_shell_tool_bloquea_operadores_de_control() -> None:
    tool = ShellTool()

    assert not tool.is_safe_to_run({"command": "ls && pwd"})
    assert not tool.is_safe_to_run({"command": "echo hola > x.txt"})
    assert not tool.is_safe_to_run({"command": "cat file | grep todo"})


def test_file_tool_usa_tool_call_id_activo_en_resultado() -> None:
    tool = FileTool()

    result = tool.execute_safe(
        {"operation": "exists", "path": "."},
        tool_call_id="tc-file-1",
    )

    assert result.tool_call_id == "tc-file-1"


def test_handle_limit_reached_default_falla_subtask() -> None:
    service = ExecutorService.__new__(ExecutorService)

    task = Task.create(prompt="x").start_planning()
    task = task.start_execution([Subtask.create(task_id=task.id, description="s", order=0)])
    subtask = task.subtasks[0].mark_running()
    task = task.update_subtask(subtask)

    log = type("DummyLog", (), {"warning": lambda *args, **kwargs: None})()

    updated_subtask, updated_task = service._handle_limit_reached(
        task=task,
        subtask=subtask,
        iterations_used=5,
        on_limit_reached=None,
        log=log,
    )

    assert updated_subtask.status == SubtaskStatus.FAILED
    assert updated_task.status == TaskStatus.IN_PROGRESS


def test_setup_logging_es_idempotente() -> None:
    logging_module = importlib.import_module("hiperforge.core.logging")
    logging_module = importlib.reload(logging_module)

    with patch.object(logging_module.logging, "basicConfig") as basic_config_mock, patch.object(
        logging_module, "_setup_file_logging"
    ) as file_logging_mock:
        logging_module.setup_logging(debug=False)
        logging_module.setup_logging(debug=True)

    assert basic_config_mock.call_count == 1
    assert file_logging_mock.call_count == 1
