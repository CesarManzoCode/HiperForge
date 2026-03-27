"""
Tests unitarios para PlannerService — clasificación, validación, parsing.

Cobertura:
  - Clasificación de complejidad (SIMPLE, MEDIUM, COMPLEX)
  - Detección de tareas single-file
  - Parsing de JSON del LLM (limpio, markdown, comentarios)
  - Validación semántica del plan (subtasks vagas, duplicadas, límites)
  - Reintentos ante planes inválidos
  - Generación del plan completo con mock del LLM
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from hiperforge.application.services.planner import (
    PlannerService,
    TaskComplexity,
    _MAX_SUBTASKS_BY_COMPLEXITY,
    _MAX_SUBTASK_DESCRIPTION_LENGTH,
)
from hiperforge.domain.entities.task import Task
from hiperforge.domain.exceptions import EmptyPlanError, InvalidPlanError
from hiperforge.domain.ports.llm_port import LLMResponse
from hiperforge.domain.value_objects.token_usage import TokenUsage
from hiperforge.infrastructure.llm.base import RichLLMResponse


# ═══════════════════════════════════════════════════════════════════
# FIXTURES LOCALES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def mock_llm():
    """Mock del LLM adapter."""
    llm = MagicMock()
    llm.bind_context = MagicMock()
    return llm


@pytest.fixture
def mock_store():
    """Mock del MemoryStore."""
    return MagicMock()


@pytest.fixture
def planner(mock_llm, mock_store) -> PlannerService:
    """PlannerService con dependencias mockeadas."""
    return PlannerService(llm=mock_llm, store=mock_store)


def make_llm_plan_response(subtasks: list[str], summary: str = "Plan generado") -> RichLLMResponse:
    """Helper: crea una respuesta del LLM con un plan JSON válido."""
    plan = {
        "subtasks": [
            {"description": desc, "order": i}
            for i, desc in enumerate(subtasks)
        ],
        "summary": summary,
    }
    return RichLLMResponse(
        content=json.dumps(plan),
        tool_calls=[],
        token_usage=TokenUsage(input_tokens=300, output_tokens=100, model="test"),
        model="test",
        finish_reason="stop",
    )


# ═══════════════════════════════════════════════════════════════════
# CLASIFICACIÓN DE COMPLEJIDAD
# ═══════════════════════════════════════════════════════════════════

class TestComplexityClassification:
    """Verificar que _classify_complexity asigna correctamente."""

    def test_tarea_simple_crear_archivo(self, planner):
        c = planner._classify_complexity("crea un script Python para sumar números")
        assert c == TaskComplexity.SIMPLE

    def test_tarea_simple_single_file(self, planner):
        c = planner._classify_complexity("crea un archivo config.py")
        assert c == TaskComplexity.SIMPLE

    def test_tarea_simple_arreglar(self, planner):
        c = planner._classify_complexity("arregla el bug en main.py")
        assert c == TaskComplexity.SIMPLE

    def test_tarea_media_implementar(self, planner):
        c = planner._classify_complexity("implementa un endpoint REST para usuarios")
        assert c == TaskComplexity.MEDIUM

    def test_tarea_media_prompt_largo(self, planner):
        prompt = "agrega validación de formularios en el componente " + "x " * 20
        c = planner._classify_complexity(prompt)
        assert c == TaskComplexity.MEDIUM

    def test_tarea_compleja_multiples_keywords(self, planner):
        c = planner._classify_complexity("refactoriza la arquitectura del sistema de pagos completo")
        assert c == TaskComplexity.COMPLEX

    def test_tarea_compleja_prompt_muy_largo(self, planner):
        prompt = " ".join(["palabra"] * 60)
        c = planner._classify_complexity(prompt)
        assert c == TaskComplexity.COMPLEX

    def test_single_file_pattern_crea_script(self, planner):
        c = planner._classify_complexity("crea un script que genere reportes")
        assert c == TaskComplexity.SIMPLE

    def test_single_file_pattern_escribe_programa(self, planner):
        c = planner._classify_complexity("escribe un programa de calculadora")
        assert c == TaskComplexity.SIMPLE

    def test_limites_max_subtasks(self):
        """Verificar los límites configurados por complejidad."""
        assert _MAX_SUBTASKS_BY_COMPLEXITY[TaskComplexity.SIMPLE] <= 3
        assert _MAX_SUBTASKS_BY_COMPLEXITY[TaskComplexity.MEDIUM] <= 8


# ═══════════════════════════════════════════════════════════════════
# PARSING DE RESPUESTA DEL LLM
# ═══════════════════════════════════════════════════════════════════

class TestPlanParsing:
    """Verificar _parse_plan_response con diferentes formatos de LLM."""

    def test_json_limpio(self, planner):
        content = json.dumps({
            "subtasks": [{"description": "Crear archivo main.py", "order": 0}],
            "summary": "Crear script",
        })
        raw = planner._parse_plan_response(content, task_id="t1")
        assert len(raw.subtask_descriptions) == 1
        assert raw.subtask_descriptions[0] == "Crear archivo main.py"

    def test_json_en_markdown(self, planner):
        content = '```json\n{"subtasks": [{"description": "Paso 1", "order": 0}], "summary": "ok"}\n```'
        raw = planner._parse_plan_response(content, task_id="t1")
        assert len(raw.subtask_descriptions) == 1

    def test_json_con_texto_antes(self, planner):
        content = 'Aquí está el plan:\n\n{"subtasks": [{"description": "Paso 1", "order": 0}], "summary": "ok"}'
        raw = planner._parse_plan_response(content, task_id="t1")
        assert len(raw.subtask_descriptions) == 1

    def test_json_con_comentarios(self, planner):
        content = '{"subtasks": [{"description": "Paso 1", "order": 0}], "summary": "ok"} // esto es un plan'
        raw = planner._parse_plan_response(content, task_id="t1")
        assert len(raw.subtask_descriptions) == 1

    def test_contenido_vacio_lanza_error(self, planner):
        with pytest.raises(EmptyPlanError):
            planner._parse_plan_response("", task_id="t1")

    def test_json_sin_subtasks_lanza_error(self, planner):
        content = json.dumps({"summary": "hola"})
        with pytest.raises(InvalidPlanError):
            planner._parse_plan_response(content, task_id="t1")

    def test_subtasks_vacia_lanza_error(self, planner):
        content = json.dumps({"subtasks": [], "summary": "vacío"})
        with pytest.raises(EmptyPlanError):
            planner._parse_plan_response(content, task_id="t1")

    def test_no_json_lanza_error(self, planner):
        with pytest.raises(InvalidPlanError):
            planner._parse_plan_response("esto no es JSON para nada", task_id="t1")

    def test_normaliza_descripcion_demasiado_larga(self, planner):
        long_desc = (
            "Crear report.py con soporte para CSV, HTML, estilos inline, argumentos CLI, "
            "escape de contenido, manejo de errores, encabezados configurables y salida opcional "
            "en stdout. Verificar ejecutando py_compile y una corrida con CSV de ejemplo."
        )
        content = json.dumps({
            "subtasks": [{"description": long_desc, "order": 0}],
            "summary": "ok",
        })

        raw = planner._parse_plan_response(content, task_id="t1")

        assert len(raw.subtask_descriptions[0]) <= _MAX_SUBTASK_DESCRIPTION_LENGTH
        assert "Verificar" in raw.subtask_descriptions[0]


# ═══════════════════════════════════════════════════════════════════
# VALIDACIÓN SEMÁNTICA
# ═══════════════════════════════════════════════════════════════════

class TestPlanValidation:
    """Verificar _validate_plan detecta problemas semánticos."""

    def _make_raw_plan(self, descriptions: list[str]):
        from hiperforge.application.services.planner import RawPlan
        return RawPlan(
            subtask_descriptions=descriptions,
            summary="test",
            raw_json={"subtasks": [{"description": d} for d in descriptions]},
        )

    def test_plan_valido_sin_errores(self, planner):
        plan = self._make_raw_plan(["Crear archivo main.py con función principal"])
        errors = planner._validate_plan(plan, max_subtasks=5)
        assert errors == []

    def test_descripcion_demasiado_corta(self, planner):
        plan = self._make_raw_plan(["hola"])
        errors = planner._validate_plan(plan, max_subtasks=5)
        assert len(errors) > 0
        assert "corta" in errors[0].lower()

    def test_excede_max_subtasks(self, planner):
        descs = [f"Paso {i}: hacer algo concreto y verificable" for i in range(10)]
        plan = self._make_raw_plan(descs)
        errors = planner._validate_plan(plan, max_subtasks=3)
        assert any("máximo" in e.lower() for e in errors)

    def test_descripcion_vaga_detectada(self, planner):
        plan = self._make_raw_plan(["Asegurarse de que todo funciona correctamente"])
        errors = planner._validate_plan(plan, max_subtasks=5)
        assert any("vaga" in e.lower() for e in errors)


# ═══════════════════════════════════════════════════════════════════
# GENERACIÓN COMPLETA CON MOCK
# ═══════════════════════════════════════════════════════════════════

class TestPlanGeneration:
    """Verificar generate_plan end-to-end con LLM mockeado."""

    def test_genera_plan_simple(self, planner, mock_llm):
        mock_llm.complete.return_value = make_llm_plan_response(
            ["Crear script report.py con csv.DictReader y html.escape, verificar con py_compile"],
        )
        task = Task.create(prompt="crea un script Python que lea CSV y genere HTML")
        subtasks = planner.generate_plan(task)
        assert len(subtasks) >= 1
        assert all(s.task_id == task.id for s in subtasks)

    def test_genera_plan_con_multiples_pasos(self, planner, mock_llm):
        mock_llm.complete.return_value = make_llm_plan_response([
            "Instalar dependencias FastAPI y uvicorn",
            "Crear estructura del proyecto con main.py y routers/",
            "Implementar endpoint GET /health",
        ])
        task = Task.create(prompt="implementa un servidor FastAPI básico")
        subtasks = planner.generate_plan(task)
        assert len(subtasks) == 3
        assert subtasks[0].order == 0
        assert subtasks[2].order == 2

    def test_reintenta_ante_plan_invalido(self, planner, mock_llm):
        """Si el primer plan es inválido, reintenta con feedback."""
        bad_response = make_llm_plan_response(["ok"])  # muy corta
        good_response = make_llm_plan_response([
            "Crear archivo main.py con servidor HTTP básico usando http.server"
        ])
        mock_llm.complete.side_effect = [bad_response, good_response]
        task = Task.create(prompt="crea un servidor HTTP")
        subtasks = planner.generate_plan(task)
        assert len(subtasks) >= 1
        assert mock_llm.complete.call_count == 2
