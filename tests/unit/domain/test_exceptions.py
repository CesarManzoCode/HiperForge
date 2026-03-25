"""
Tests para la jerarquía de excepciones del dominio.
Verifica herencia, mensajes y contexto de cada tipo de error.
"""

from __future__ import annotations

import pytest

from hiperforge.domain.exceptions import (
    HiperForgeError,
    DomainError,
    InvalidStatusTransition,
    EntityNotFound,
    DuplicateEntity,
    PlanError,
    EmptyPlanError,
    InvalidPlanError,
    ToolError,
    ToolNotFound,
    ToolExecutionError,
    ToolTimeoutError,
    LLMError,
    LLMConnectionError,
    LLMRateLimitError,
    LLMResponseError,
)


class TestExceptionHierarchy:
    """Verificar que la herencia es correcta."""

    def test_domain_error_hereda_de_hiperforge(self):
        assert issubclass(DomainError, HiperForgeError)

    def test_plan_error_hereda_de_hiperforge(self):
        assert issubclass(PlanError, HiperForgeError)

    def test_tool_error_hereda_de_hiperforge(self):
        assert issubclass(ToolError, HiperForgeError)

    def test_llm_error_hereda_de_hiperforge(self):
        assert issubclass(LLMError, HiperForgeError)

    def test_invalid_status_hereda_de_domain(self):
        assert issubclass(InvalidStatusTransition, DomainError)

    def test_empty_plan_hereda_de_plan(self):
        assert issubclass(EmptyPlanError, PlanError)

    def test_tool_timeout_hereda_de_tool(self):
        assert issubclass(ToolTimeoutError, ToolError)

    def test_llm_rate_limit_hereda_de_llm(self):
        assert issubclass(LLMRateLimitError, LLMError)


class TestExceptionContext:
    """Verificar que cada excepción tiene contexto útil."""

    def test_invalid_status_transition_context(self):
        exc = InvalidStatusTransition("Task", "pending", "completed")
        assert "Task" in str(exc)
        assert "pending" in str(exc)
        assert "completed" in str(exc)
        assert exc.context["entity"] == "Task"

    def test_entity_not_found_context(self):
        exc = EntityNotFound("Project", "proj-123")
        assert "proj-123" in str(exc)
        assert exc.context["entity_type"] == "Project"

    def test_duplicate_entity_context(self):
        exc = DuplicateEntity("Workspace", "mi-workspace")
        assert "mi-workspace" in str(exc)

    def test_tool_not_found(self):
        exc = ToolNotFound("super_tool")
        assert "super_tool" in str(exc)

    def test_llm_connection_error(self):
        exc = LLMConnectionError(provider="openai", reason="timeout")
        assert "openai" in str(exc)

    def test_llm_rate_limit_con_retry(self):
        exc = LLMRateLimitError(provider="anthropic", retry_after_seconds=8.5)
        assert exc.retry_after_seconds == 8.5


class TestExceptionCatch:
    """Verificar que except HiperForgeError atrapa todo."""

    def test_catch_domain_error(self):
        with pytest.raises(HiperForgeError):
            raise EntityNotFound("Task", "t1")

    def test_catch_plan_error(self):
        with pytest.raises(HiperForgeError):
            raise EmptyPlanError(task_prompt="test")

    def test_catch_tool_error(self):
        with pytest.raises(HiperForgeError):
            raise ToolTimeoutError(tool_name="shell", timeout_seconds=30.0)

    def test_catch_llm_error(self):
        with pytest.raises(HiperForgeError):
            raise LLMRateLimitError(provider="openai")
