"""
Configuración global de pytest para HiperForge.

Fixtures compartidas entre TODOS los tests del proyecto.
Cada subdirectorio puede tener su propio conftest.py con fixtures específicas.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from hiperforge.domain.entities.task import Subtask, SubtaskStatus, Task, TaskStatus
from hiperforge.domain.entities.tool_call import ToolCall, ToolCallStatus, ToolResult
from hiperforge.domain.entities.project import Project, ProjectStatus
from hiperforge.domain.entities.workspace import Workspace, WorkspaceStatus
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.domain.value_objects.token_usage import TokenUsage
from hiperforge.infrastructure.llm.base import RichLLMResponse, ToolCallRequest


# ──────────────────────────────────────────────────────────────────────
# Directorio temporal — se limpia automáticamente al terminar cada test
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_dir(tmp_path: Path) -> Path:
    """Directorio temporal para tests que necesitan I/O en disco."""
    return tmp_path


@pytest.fixture
def app_dir(tmp_path: Path) -> Path:
    """Simula ~/.hiperforge en un directorio temporal."""
    app = tmp_path / ".hiperforge"
    app.mkdir()
    (app / "workspaces").mkdir()
    (app / "logs").mkdir()
    return app


# ──────────────────────────────────────────────────────────────────────
# Entidades de dominio prefabricadas
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_task() -> Task:
    """Task en estado PENDING lista para planificar."""
    return Task.create(
        prompt="crea un script Python que lea un CSV y genere un reporte en HTML",
        project_id=None,
    )


@pytest.fixture
def planned_task(sample_task: Task) -> Task:
    """Task con plan generado, lista para ejecutar."""
    task = sample_task.start_planning()
    subtasks = [
        Subtask.create(
            task_id=task.id,
            description="Crear script report.py con csv.DictReader y generación HTML",
            order=0,
        ),
    ]
    return task.start_execution(subtasks)


@pytest.fixture
def multi_subtask_task(sample_task: Task) -> Task:
    """Task con múltiples subtasks para tests de ejecución compleja."""
    task = sample_task.start_planning()
    subtasks = [
        Subtask.create(task_id=task.id, description="Instalar dependencias", order=0),
        Subtask.create(task_id=task.id, description="Crear módulo auth", order=1),
        Subtask.create(task_id=task.id, description="Escribir tests", order=2),
        Subtask.create(task_id=task.id, description="Verificar cobertura", order=3),
    ]
    return task.start_execution(subtasks)


@pytest.fixture
def sample_subtask(planned_task: Task) -> Subtask:
    """Subtask individual en estado PENDING."""
    return planned_task.subtasks[0]


@pytest.fixture
def sample_tool_call() -> ToolCall:
    """ToolCall en estado PENDING."""
    return ToolCall.create(
        tool_name="shell",
        arguments={"command": "pytest tests/", "timeout": 30.0},
    )


@pytest.fixture
def completed_tool_call(sample_tool_call: ToolCall) -> ToolCall:
    """ToolCall completado exitosamente."""
    running = sample_tool_call.mark_running()
    result = ToolResult.success(
        tool_call_id=running.id,
        output="5 passed in 0.42s",
    )
    return running.with_result(result)


@pytest.fixture
def sample_project() -> Project:
    """Project en estado ACTIVE."""
    return Project.create(
        name="API de pagos",
        workspace_id="test-workspace-id",
        description="Backend de procesamiento de pagos con Stripe",
    )


@pytest.fixture
def sample_workspace() -> Workspace:
    """Workspace en estado ACTIVE."""
    return Workspace.create(
        name="desarrollo",
        description="Workspace de desarrollo",
    )


# ──────────────────────────────────────────────────────────────────────
# Value objects prefabricados
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_messages() -> list[Message]:
    """Historial típico de conversación system + user + assistant."""
    return [
        Message.system("Eres HiperForge, un agente autónomo."),
        Message.user("Subtask 1 de 1: Crear script report.py"),
        Message.assistant('{"action":"tool_call","tool":"file","arguments":{"operation":"write","path":"report.py","content":"# script"}}'),
    ]


@pytest.fixture
def sample_token_usage() -> TokenUsage:
    """TokenUsage típico de una llamada al LLM."""
    return TokenUsage(input_tokens=500, output_tokens=200, model="gpt-4o")


@pytest.fixture
def zero_token_usage() -> TokenUsage:
    """TokenUsage vacío para acumulación."""
    return TokenUsage.zero()


# ──────────────────────────────────────────────────────────────────────
# Mocks del LLM
# ──────────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm_response() -> RichLLMResponse:
    """Respuesta típica del LLM con tool_call."""
    return RichLLMResponse(
        content="",
        tool_calls=[
            ToolCallRequest(
                tool_call_id="tc-001",
                tool_name="file",
                arguments={"operation": "write", "path": "test.py", "content": "print('ok')"},
            ),
        ],
        token_usage=TokenUsage(input_tokens=500, output_tokens=100, model="gpt-4o"),
        model="gpt-4o",
        finish_reason="tool_use",
    )


@pytest.fixture
def mock_llm_complete_response() -> RichLLMResponse:
    """Respuesta del LLM indicando subtask completada."""
    return RichLLMResponse(
        content="Script report.py creado y verificado con py_compile.",
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=300, output_tokens=50, model="gpt-4o"),
        model="gpt-4o",
        finish_reason="complete",
    )


@pytest.fixture
def mock_llm_think_response() -> RichLLMResponse:
    """Respuesta del LLM con acción think."""
    return RichLLMResponse(
        content="Necesito crear el archivo primero y luego verificar.",
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=300, output_tokens=30, model="gpt-4o"),
        model="gpt-4o",
        finish_reason="stop",
    )
