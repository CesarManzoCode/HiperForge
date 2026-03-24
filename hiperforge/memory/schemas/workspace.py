"""
Schema: Workspace e índice global

Define la estructura de dos archivos JSON:

  1. index.json — índice global de workspaces
     { "active_workspace_id": "01HX...", "workspace_ids": ["01HX...", ...] }

  2. workspace.json — metadatos de un workspace específico
     { "id": "01HX...", "name": "Trabajo", "status": "active", ... }

Los schemas de Pydantic son la fuente de verdad del formato JSON.
Si el JSON en disco no coincide con el schema, la validación falla
con un error descriptivo en vez de corromper datos silenciosamente.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field, field_validator

from hiperforge.core.constants import SCHEMA_VERSION_WORKSPACE


class WorkspaceIndexSchema(BaseModel):
    """
    Schema del índice global: ~/.hiperforge/index.json

    Contiene el workspace activo y la lista de todos los workspaces.
    Es el primer archivo que se lee al arrancar HiperForge.
    """

    active_workspace_id: str | None = Field(
        default=None,
        description="ID del workspace activo. None si es primera ejecución.",
    )

    workspace_ids: list[str] = Field(
        default_factory=list,
        description="IDs de todos los workspaces registrados.",
    )

    def add_workspace(self, workspace_id: str) -> WorkspaceIndexSchema:
        """Agrega un workspace al índice si no existe ya."""
        if workspace_id in self.workspace_ids:
            return self

        return WorkspaceIndexSchema(
            active_workspace_id=self.active_workspace_id,
            workspace_ids=sorted([*self.workspace_ids, workspace_id]),
        )

    def set_active(self, workspace_id: str) -> WorkspaceIndexSchema:
        """Cambia el workspace activo."""
        return WorkspaceIndexSchema(
            active_workspace_id=workspace_id,
            workspace_ids=self.workspace_ids,
        )

    def remove_workspace(self, workspace_id: str) -> WorkspaceIndexSchema:
        """
        Elimina un workspace del índice.

        Si era el workspace activo, lo limpia también.
        """
        new_ids = [wid for wid in self.workspace_ids if wid != workspace_id]
        new_active = (
            None if self.active_workspace_id == workspace_id
            else self.active_workspace_id
        )

        return WorkspaceIndexSchema(
            active_workspace_id=new_active,
            workspace_ids=new_ids,
        )


class WorkspaceSchema(BaseModel):
    """
    Schema del archivo workspace.json.

    Contiene los metadatos del workspace pero NO los proyectos completos —
    solo sus IDs. Cada proyecto tiene su propio project.json.
    """

    id: str = Field(description="ULID único del workspace.")
    name: str = Field(description="Nombre del workspace.")
    description: str | None = Field(default=None)
    status: str = Field(default="active")
    project_ids: list[str] = Field(default_factory=list)
    schema_version: int = Field(default=SCHEMA_VERSION_WORKSPACE)
    created_at: str = Field(description="ISO 8601 UTC.")
    updated_at: str = Field(description="ISO 8601 UTC.")

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
            raise ValueError("el nombre del workspace no puede estar vacío")
        return v.strip()

    def to_datetime(self, field: str) -> datetime:
        """Convierte un campo de fecha string a datetime."""
        value = getattr(self, field)
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
