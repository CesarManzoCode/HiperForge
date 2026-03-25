"""
Capa CLI de HiperForge.

Expone únicamente el entrypoint `app` que pyproject.toml necesita
referenciar y la función `cli_entry` que actúa como wrapper del proceso.

El resto de los componentes de la CLI (comandos, UI, error handler) no
se exponen aquí — la CLI es una capa de presentación interna y sus
componentes no forman parte de la API pública del paquete.

USO EN pyproject.toml:
  [project.scripts]
  hiperforge = "hiperforge.cli.main:cli_entry"

USO PROGRAMÁTICO (tests de integración e2e):
  from hiperforge.cli import app
  from typer.testing import CliRunner

  runner = CliRunner()
  result = runner.invoke(app, ["workspace", "create", "test-ws"])
  assert result.exit_code == 0
"""

from hiperforge.cli.main import app, cli_entry

__all__ = [
    "app",
    "cli_entry",
]