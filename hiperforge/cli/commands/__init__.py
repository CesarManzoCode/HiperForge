"""
Comandos de la CLI de HiperForge.

Expone los grupos de comandos Typer que main.py registra en el
entrypoint principal. Cada grupo encapsula un dominio completo
de operaciones de la CLI.

GRUPOS DE COMANDOS:
  run_command    → hiperforge run "<prompt>"  (comando raíz, no es grupo)
  task_app       → hiperforge task *
  workspace_app  → hiperforge workspace *
  project_app    → hiperforge project *
  config_app     → hiperforge config *
"""

from hiperforge.cli.commands.config import config_app
from hiperforge.cli.commands.project import project_app
from hiperforge.cli.commands.run import run_command, task_app
from hiperforge.cli.commands.workspace import workspace_app

__all__ = [
    "run_command",
    "task_app",
    "workspace_app",
    "project_app",
    "config_app",
]
