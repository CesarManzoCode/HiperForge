"""
Capa de aplicación de HiperForge.

Expone exactamente lo que la CLI necesita importar:
  - Container: el punto de entrada para construir todas las dependencias.
  - DTOs:      los contratos de entrada y salida de cada use case.

Los services y use cases internos NO se exponen aquí — la CLI
los obtiene a través del Container, nunca directamente.

USO CORRECTO EN LA CLI:
  from hiperforge.application import Container, RunTaskInput

  container = Container.build()
  output = container.run_task.execute(RunTaskInput(prompt="..."))

USO INCORRECTO (no hacer esto):
  from hiperforge.application.services.executor import ExecutorService
  from hiperforge.application.use_cases.run_task import RunTaskUseCase
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
    # Container — único punto de construcción del sistema
    "Container",

    # DTOs de entrada
    "RunTaskInput",
    "CreateTaskInput",
    "CreateProjectInput",
    "CreateWorkspaceInput",
    "SwitchWorkspaceInput",
    "UpdatePreferencesInput",

    # DTOs de salida / resúmenes
    "RunTaskOutput",
    "TaskSummary",
    "ProjectSummary",
    "WorkspaceSummary",
]