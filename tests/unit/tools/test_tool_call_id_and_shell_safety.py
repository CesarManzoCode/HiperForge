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


def test_shell_tool_bloquea_redirects() -> None:
    """Redirects (>, <) siempre bloqueados — pueden escribir a archivos arbitrarios."""
    tool = ShellTool()

    assert not tool.is_safe_to_run({"command": "echo hola > x.txt"})
    assert not tool.is_safe_to_run({"command": "cat < /etc/passwd"})
    assert not tool.is_safe_to_run({"command": "python script.py >> log.txt"})


def test_shell_tool_permite_chains_seguros() -> None:
    """Chains con && de comandos seguros deben ser permitidos."""
    tool = ShellTool()

    # cd && command se auto-transforma a working_dir
    args = {"command": "cd /tmp && ls"}
    assert tool.is_safe_to_run(args)

    # Chains de solo lectura
    assert tool.is_safe_to_run({"command": "ls && pwd"})


def test_shell_tool_bloquea_chains_peligrosos() -> None:
    """Chains con algún segmento peligroso deben ser bloqueados."""
    tool = ShellTool()

    assert not tool.is_safe_to_run({"command": "ls && rm -rf /"})
    assert not tool.is_safe_to_run({"command": "echo test && sudo apt install evil"})


def test_shell_tool_ejecuta_heredoc_seguro(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tool = ShellTool()

    with patch("subprocess.run") as run_mock:
        result = tool.execute_safe(
            {"command": "cat > sample.csv <<'CSV'\nname,age\nalice,30\nCSV"},
            tool_call_id="tc-heredoc-1",
        )

    assert result.success
    assert result.tool_call_id == "tc-heredoc-1"
    assert (tmp_path / "sample.csv").read_text(encoding="utf-8") == "name,age\nalice,30\n"
    run_mock.assert_not_called()


def test_shell_tool_corrige_working_dir_con_case_incorrecto(tmp_path) -> None:
    project_dir = tmp_path / "HiperForge"
    project_dir.mkdir()
    tool = ShellTool()

    args = {"working_dir": str(tmp_path / "hiperforge"), "command": "pwd"}
    errors = tool.validate_arguments(args)

    assert errors == []
    assert args["working_dir"] == str(project_dir)


def test_file_tool_usa_tool_call_id_activo_en_resultado() -> None:
    tool = FileTool()

    result = tool.execute_safe(
        {"operation": "exists", "path": "."},
        tool_call_id="tc-file-1",
    )

    assert result.tool_call_id == "tc-file-1"


def test_file_tool_normaliza_create_a_write(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    tool = FileTool()

    result = tool.execute_safe(
        {
            "operation": "create",
            "path": "report.py",
            "content": "print('ok')\n",
        },
        tool_call_id="tc-file-create",
    )

    assert result.success
    assert (tmp_path / "report.py").read_text(encoding="utf-8") == "print('ok')\n"


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
