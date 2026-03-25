"""
Tests unitarios para ExecutorService — loop ReAct, detección de bucles, iteraciones dinámicas.

Cobertura:
  - Iteraciones dinámicas según número de subtasks
  - Detección de bucles repetitivos
  - Detección de exploración redundante
  - Procesamiento de respuestas (complete, tool_call, think)
  - Helpers de fingerprint y familias de tools
  - Contexto entre subtasks (previous_subtask_summary)
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock, patch, PropertyMock

from hiperforge.application.services.executor import (
    ExecutorService,
    ToolOutcomeSnapshot,
    _LOOP_DETECTION_WINDOW,
    _EFFICIENCY_WINDOW,
)
from hiperforge.core.constants import (
    REACT_MAX_ITERATIONS_PER_SUBTASK,
    REACT_MAX_ITERATIONS_SIMPLE,
    REACT_MAX_ITERATIONS_MEDIUM,
)
from hiperforge.infrastructure.llm.base import ToolCallRequest


# ═══════════════════════════════════════════════════════════════════
# ITERACIONES DINÁMICAS
# ═══════════════════════════════════════════════════════════════════

class TestDynamicIterations:
    """Verificar _max_iterations_for_plan asigna correctamente."""

    def test_1_subtask_usa_simple(self):
        assert ExecutorService._max_iterations_for_plan(1) == REACT_MAX_ITERATIONS_SIMPLE

    def test_2_subtasks_usa_simple(self):
        assert ExecutorService._max_iterations_for_plan(2) == REACT_MAX_ITERATIONS_SIMPLE

    def test_3_subtasks_usa_medium(self):
        assert ExecutorService._max_iterations_for_plan(3) == REACT_MAX_ITERATIONS_MEDIUM

    def test_5_subtasks_usa_medium(self):
        assert ExecutorService._max_iterations_for_plan(5) == REACT_MAX_ITERATIONS_MEDIUM

    def test_6_subtasks_usa_full(self):
        assert ExecutorService._max_iterations_for_plan(6) == REACT_MAX_ITERATIONS_PER_SUBTASK

    def test_20_subtasks_usa_full(self):
        assert ExecutorService._max_iterations_for_plan(20) == REACT_MAX_ITERATIONS_PER_SUBTASK

    def test_simple_menor_que_medium(self):
        assert REACT_MAX_ITERATIONS_SIMPLE < REACT_MAX_ITERATIONS_MEDIUM

    def test_medium_menor_que_full(self):
        assert REACT_MAX_ITERATIONS_MEDIUM < REACT_MAX_ITERATIONS_PER_SUBTASK


# ═══════════════════════════════════════════════════════════════════
# DETECCIÓN DE BUCLES
# ═══════════════════════════════════════════════════════════════════

class TestLoopDetection:
    """Verificar _is_stuck_in_loop detecta patrones repetitivos."""

    @staticmethod
    def _make_outcome(fingerprint: str, success: bool = True, family: str = "shell:ls") -> ToolOutcomeSnapshot:
        return ToolOutcomeSnapshot(
            fingerprint=fingerprint,
            family=family,
            success=success,
            tool_name="shell",
            is_mutation=False,
            is_observation=True,
            is_verification=False,
        )

    def test_no_detecta_con_pocos_outcomes(self):
        outcomes = [self._make_outcome("a")]
        assert not ExecutorService._is_stuck_in_loop(outcomes)

    def test_no_detecta_con_fingerprints_distintos(self):
        outcomes = [
            self._make_outcome("a"),
            self._make_outcome("b"),
            self._make_outcome("c"),
        ]
        assert not ExecutorService._is_stuck_in_loop(outcomes)

    def test_detecta_fingerprints_identicos(self):
        outcomes = [self._make_outcome("same")] * _LOOP_DETECTION_WINDOW
        assert ExecutorService._is_stuck_in_loop(outcomes)

    def test_detecta_fallos_misma_familia(self):
        outcomes = [
            self._make_outcome(f"fail-{i}", success=False, family="shell:pytest")
            for i in range(_LOOP_DETECTION_WINDOW)
        ]
        assert ExecutorService._is_stuck_in_loop(outcomes)

    def test_no_detecta_fallos_familias_distintas(self):
        outcomes = [
            self._make_outcome(f"fail-{i}", success=False, family=f"shell:cmd{i}")
            for i in range(_LOOP_DETECTION_WINDOW)
        ]
        assert not ExecutorService._is_stuck_in_loop(outcomes)


class TestWastingIterations:
    """Verificar _is_wasting_iterations detecta exploración redundante."""

    @staticmethod
    def _make_outcome(
        *,
        success: bool = True,
        is_mutation: bool = False,
        is_observation: bool = False,
        is_verification: bool = False,
        family: str = "file:read:test.py",
    ) -> ToolOutcomeSnapshot:
        return ToolOutcomeSnapshot(
            fingerprint=f"ok|{family}",
            family=family,
            success=success,
            tool_name="file",
            is_mutation=is_mutation,
            is_observation=is_observation,
            is_verification=is_verification,
        )

    def test_no_detecta_sin_historial_suficiente(self):
        outcomes = [self._make_outcome(is_observation=True)]
        assert not ExecutorService._is_wasting_iterations(outcomes)

    def test_detecta_4_observaciones_tras_mutacion(self):
        outcomes = [
            self._make_outcome(is_mutation=True),  # escribió algo
        ] + [
            self._make_outcome(is_observation=True, family="file:read:test.py")
            for _ in range(_EFFICIENCY_WINDOW)
        ]
        assert ExecutorService._is_wasting_iterations(outcomes)

    def test_no_detecta_si_hay_mutacion_reciente(self):
        outcomes = [
            self._make_outcome(is_mutation=True),
            self._make_outcome(is_observation=True),
            self._make_outcome(is_mutation=True),  # mutación en la ventana
            self._make_outcome(is_observation=True),
        ]
        assert not ExecutorService._is_wasting_iterations(outcomes)


# ═══════════════════════════════════════════════════════════════════
# HELPERS DE FINGERPRINT
# ═══════════════════════════════════════════════════════════════════

class TestArgFingerprint:

    def test_mismos_args_mismo_fingerprint(self):
        args = {"command": "pytest tests/", "timeout": 30.0}
        fp1 = ExecutorService._arg_fingerprint(args)
        fp2 = ExecutorService._arg_fingerprint(args)
        assert fp1 == fp2

    def test_args_distintos_diferente_fingerprint(self):
        fp1 = ExecutorService._arg_fingerprint({"command": "ls"})
        fp2 = ExecutorService._arg_fingerprint({"command": "pwd"})
        assert fp1 != fp2

    def test_normaliza_paths_a_basename(self):
        fp = ExecutorService._arg_fingerprint({"path": "/home/user/proyecto/main.py"})
        assert "main.py" in fp

    def test_maneja_dict_vacio(self):
        fp = ExecutorService._arg_fingerprint({})
        assert isinstance(fp, str)


class TestToolFamily:

    def test_file_operation(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "write", "path": "/x/main.py"})
        family = ExecutorService._tool_family(req)
        assert "file:write:main.py" == family

    def test_shell_python(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "python3 test.py"})
        family = ExecutorService._tool_family(req)
        assert "shell:python3:test.py" == family

    def test_shell_simple(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "ls -la"})
        family = ExecutorService._tool_family(req)
        assert "shell:ls" == family

    def test_tool_desconocida(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="web", arguments={"url": "http://x"})
        family = ExecutorService._tool_family(req)
        assert family == "web"


class TestMutationDetection:

    def test_file_write_es_mutacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "write"})
        assert ExecutorService._is_mutating_request(req)

    def test_file_read_no_es_mutacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "read"})
        assert not ExecutorService._is_mutating_request(req)

    def test_shell_redirect_es_mutacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "echo x > file.txt"})
        assert ExecutorService._is_mutating_request(req)

    def test_shell_ls_no_es_mutacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "ls -la"})
        assert not ExecutorService._is_mutating_request(req)


class TestObservationDetection:

    def test_file_read_es_observacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "read"})
        assert ExecutorService._is_observation_request(req)

    def test_file_write_no_es_observacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "write"})
        assert not ExecutorService._is_observation_request(req)

    def test_shell_grep_es_observacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "grep -r TODO ."})
        assert ExecutorService._is_observation_request(req)

    def test_shell_cat_es_observacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "cat main.py"})
        assert ExecutorService._is_observation_request(req)


class TestVerificationDetection:

    def test_pytest_es_verificacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "pytest tests/"})
        assert ExecutorService._is_verification_request(req)

    def test_python_script_es_verificacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "python3 test.py"})
        assert ExecutorService._is_verification_request(req)

    def test_ls_no_es_verificacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="shell", arguments={"command": "ls"})
        assert not ExecutorService._is_verification_request(req)

    def test_file_tool_no_es_verificacion(self):
        req = ToolCallRequest(tool_call_id="t1", tool_name="file", arguments={"operation": "read"})
        assert not ExecutorService._is_verification_request(req)
