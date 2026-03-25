"""
Capa de aplicación de HiperForge.

Expone únicamente lo que la CLI necesita importar directamente.
Todo lo demás se accede a través del Container.
"""

from hiperforge.application.container import Container
from hiperforge.application.dto import (
    CreateProjectInput,
    CreateTaskInput,
    CreateWorkspaceInput,
    ProjectSummary,
    RunTaskInput,
    RunTaskOutput,
    SwitchWorkspaceInput,
    TaskSummary,
    UpdatePreferencesInput,
    WorkspaceSummary,
)

__all__ = [
    "Container",
    "RunTaskInput",
    "RunTaskOutput",
    "CreateTaskInput",
    "CreateProjectInput",
    "CreateWorkspaceInput",
    "SwitchWorkspaceInput",
    "UpdatePreferencesInput",
    "TaskSummary",
    "ProjectSummary",
    "WorkspaceSummary",
]
