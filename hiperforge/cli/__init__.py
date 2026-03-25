"""
Capa de presentación de la CLI de HiperForge.

Expone los componentes de UI que los comandos de la CLI necesitan
importar directamente. Todos los componentes usan Rich internamente.

COMPONENTES DISPONIBLES:
  Renderer     → motor de presentación para todos los outputs de la CLI.
  AgentSpinner → indicador de progreso que escucha el EventBus.
  PlanView     → vista interactiva del plan con confirmación del usuario.
  Confirm      → diálogos de confirmación para operaciones críticas.
  console      → consola Rich global compartida (stdout).

USO ESTÁNDAR EN COMANDOS:
  from hiperforge.cli.ui import Renderer, AgentSpinner, PlanView, Confirm

  renderer = Renderer(verbose=verbose)
  spinner = AgentSpinner()
  plan_view = PlanView(renderer=renderer, auto_confirm=yes)
"""

from hiperforge.cli.ui.confirm import Confirm, ConfirmResult
from hiperforge.cli.ui.plan_view import PlanView
from hiperforge.cli.ui.renderer import Renderer, _stdout_console as console
from hiperforge.cli.ui.spinner import AgentSpinner

__all__ = [
    "Renderer",
    "AgentSpinner",
    "PlanView",
    "Confirm",
    "ConfirmResult",
    "console",
]