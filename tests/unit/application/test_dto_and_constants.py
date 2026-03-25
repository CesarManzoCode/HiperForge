"""
Tests unitarios para DTOs de entrada/salida y constantes globales.
"""

from __future__ import annotations

import pytest

from hiperforge.application.dto import RunTaskInput
from hiperforge.core.constants import (
    APP_NAME,
    APP_VERSION,
    LLM_DEFAULT_PROVIDER,
    LLM_DEFAULT_TEMPERATURE,
    LLM_DEFAULT_MAX_TOKENS,
    LLM_DEFAULT_MAX_TOKENS_PLANNING,
    LLM_DEFAULT_MAX_TOKENS_REACT,
    REACT_MAX_ITERATIONS_PER_SUBTASK,
    REACT_MAX_ITERATIONS_SIMPLE,
    REACT_MAX_ITERATIONS_MEDIUM,
    REACT_MAX_SUBTASKS,
    REACT_MAX_TOOL_RETRIES,
    TOOL_DEFAULT_TIMEOUT_SECONDS,
    TOOL_MAX_OUTPUT_CHARS,
    LLM_TOOL_RESULT_MAX_CHARS,
)


# ═══════════════════════════════════════════════════════════════════
# DTOs
# ═══════════════════════════════════════════════════════════════════

class TestRunTaskInput:

    def test_con_prompt_valido(self):
        inp = RunTaskInput(prompt="crea un script")
        assert inp.prompt == "crea un script"
        assert inp.task_id is None

    def test_con_task_id_valido(self):
        inp = RunTaskInput(task_id="01HX4K123")
        assert inp.task_id == "01HX4K123"
        assert inp.prompt is None

    def test_ambos_prompt_y_task_id_error(self):
        with pytest.raises(ValueError, match="exactamente uno"):
            RunTaskInput(prompt="hola", task_id="t1")

    def test_ninguno_prompt_ni_task_id_error(self):
        with pytest.raises(ValueError, match="exactamente uno"):
            RunTaskInput()

    def test_prompt_vacio_error(self):
        with pytest.raises(ValueError):
            RunTaskInput(prompt="   ")

    def test_auto_confirm_default_false(self):
        inp = RunTaskInput(prompt="test")
        assert inp.auto_confirm is False

    def test_auto_confirm_true(self):
        inp = RunTaskInput(prompt="test", auto_confirm=True)
        assert inp.auto_confirm is True

    def test_workspace_id_opcional(self):
        inp = RunTaskInput(prompt="test", workspace_id="ws-1")
        assert inp.workspace_id == "ws-1"

    def test_project_id_opcional(self):
        inp = RunTaskInput(prompt="test", project_id="proj-1")
        assert inp.project_id == "proj-1"


# ═══════════════════════════════════════════════════════════════════
# CONSTANTES
# ═══════════════════════════════════════════════════════════════════

class TestConstants:
    """Verificar que las constantes tienen valores sensatos y consistentes."""

    def test_app_name(self):
        assert APP_NAME == "hiperforge"

    def test_app_version_format(self):
        parts = APP_VERSION.split(".")
        assert len(parts) == 3

    def test_default_provider(self):
        assert LLM_DEFAULT_PROVIDER in ("anthropic", "openai", "groq", "ollama")

    def test_temperature_rango(self):
        assert 0.0 <= LLM_DEFAULT_TEMPERATURE <= 2.0

    def test_max_tokens_positivos(self):
        assert LLM_DEFAULT_MAX_TOKENS > 0
        assert LLM_DEFAULT_MAX_TOKENS_PLANNING > 0
        assert LLM_DEFAULT_MAX_TOKENS_REACT > 0

    def test_planning_tokens_menor_que_default(self):
        assert LLM_DEFAULT_MAX_TOKENS_PLANNING <= LLM_DEFAULT_MAX_TOKENS

    def test_react_tokens_menor_que_default(self):
        assert LLM_DEFAULT_MAX_TOKENS_REACT <= LLM_DEFAULT_MAX_TOKENS

    def test_iteraciones_dinamicas_orden(self):
        """SIMPLE < MEDIUM < FULL."""
        assert REACT_MAX_ITERATIONS_SIMPLE < REACT_MAX_ITERATIONS_MEDIUM
        assert REACT_MAX_ITERATIONS_MEDIUM < REACT_MAX_ITERATIONS_PER_SUBTASK

    def test_iteraciones_simple_razonable(self):
        assert 3 <= REACT_MAX_ITERATIONS_SIMPLE <= 8

    def test_iteraciones_medium_razonable(self):
        assert 5 <= REACT_MAX_ITERATIONS_MEDIUM <= 12

    def test_max_subtasks_razonable(self):
        assert REACT_MAX_SUBTASKS >= 10
        assert REACT_MAX_SUBTASKS <= 50

    def test_tool_retries_razonable(self):
        assert 1 <= REACT_MAX_TOOL_RETRIES <= 5

    def test_tool_timeout_razonable(self):
        assert TOOL_DEFAULT_TIMEOUT_SECONDS >= 10.0

    def test_tool_output_chars_mayor_que_llm_result(self):
        """El output máximo de una tool debe ser >= lo que se reinyecta al LLM."""
        assert TOOL_MAX_OUTPUT_CHARS >= LLM_TOOL_RESULT_MAX_CHARS
