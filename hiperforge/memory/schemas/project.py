"""
Schema: Project y Task

Define la estructura de:
  - project.json: metadatos del proyecto + lista de task_ids
  - task.json:    task completa con subtasks y tool calls

Las tasks se guardan completas en un solo archivo porque:
  1. Una subtask no tiene sentido fuera de su task
  2. El número de subtasks es acotado (max REACT_MAX_SUBTASKS)
  3. Un solo archivo es más fácil de debuggear manualmente
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field, field_validator

from hiperforge.core.constants import SCHEMA_VERSION_PROJECT, SCHEMA_VERSION_TASK


class ProjectSchema(BaseModel):
    """
    Schema del archivo project.json.

    Contiene metadatos del proyecto y lista de task_ids.
    Las tasks completas están en tasks/{task_id}/task.json.
    """

    id: str
    workspace_id: str
    name: str
    description: str | None = None
    status: str = "active"
    tags: list[str] = Field(default_factory=list)
    task_ids: list[str] = Field(default_factory=list)
    schema_version: int = SCHEMA_VERSION_PROJECT
    created_at: str
    updated_at: str

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"active", "archived", "deleted"}
        if v not in valid:
            raise ValueError(f"status inválido: '{v}'. Válidos: {valid}")
        return v

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("el nombre del proyecto no puede estar vacío")
        return v.strip()


class ToolResultSchema(BaseModel):
    """Schema del resultado de una tool call."""

    tool_call_id: str
    output: str
    success: bool
    error_message: str | None = None
    executed_at: str


class ToolCallSchema(BaseModel):
    """Schema de una tool call completa con su resultado."""

    id: str
    tool_name: str
    arguments: dict[str, Any]
    status: str
    created_at: str
    result: ToolResultSchema | None = None


class SubtaskSchema(BaseModel):
    """Schema de una subtask con su historial de tool calls."""

    id: str
    task_id: str
    description: str
    order: int
    status: str
    tool_calls: list[ToolCallSchema] = Field(default_factory=list)
    reasoning: str | None = None
    created_at: str
    completed_at: str | None = None


class TokenUsageSchema(BaseModel):
    """Schema del uso de tokens de una task."""

    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    model: str | None = None


class TaskSchema(BaseModel):
    """
    Schema del archivo task.json.

    La task se guarda COMPLETA — incluyendo subtasks y tool calls.
    Es la unidad de persistencia más importante del sistema.
    """

    id: str
    project_id: str | None = None
    prompt: str
    status: str
    subtasks: list[SubtaskSchema] = Field(default_factory=list)
    summary: str | None = None
    token_usage: TokenUsageSchema = Field(default_factory=TokenUsageSchema)
    schema_version: int = SCHEMA_VERSION_TASK
    created_at: str
    completed_at: str | None = None

    @field_validator("status")
    @classmethod
    def validate_status(cls, v: str) -> str:
        valid = {"pending", "planning", "in_progress", "completed", "failed", "cancelled"}
        if v not in valid:
            raise ValueError(f"status inválido: '{v}'")
        return v

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("el prompt no puede estar vacío")
        return v
