"""
main.py — Entrypoint principal de la CLI de HiperForge.

Este módulo es el punto de entrada único de toda la CLI.
Responsabilidades exclusivas de este módulo:

  1. CONSTRUIR el árbol de comandos Typer registrando todos los grupos.
  2. INICIALIZAR el sistema de logging antes del primer comando.
  3. CONFIGURAR el comportamiento global de la CLI (versión, help, etc.).
  4. EXPONER el callback de la app raíz para flags globales (--version, --debug).
  5. SER el target del entry_point en pyproject.toml.

════════════════════════════════════════════════════════════
ÁRBOL DE COMANDOS
════════════════════════════════════════════════════════════

  hiperforge [--version] [--debug] [--verbose]
  │
  ├── run "<prompt>"              → Flujo completo del agente (comando más usado)
  │
  ├── task
  │   ├── create "<prompt>"      → Crea task en PENDING
  │   ├── plan   <task_id>       → Genera el plan sin ejecutar
  │   ├── run    <task_id>       → Ejecuta un plan ya generado
  │   └── list                   → Lista tasks del workspace activo
  │
  ├── workspace
  │   ├── create <nombre>        → Crea nuevo workspace
  │   ├── list                   → Lista todos los workspaces
  │   ├── switch <id_o_nombre>   → Cambia el workspace activo
  │   ├── show   [id_o_nombre]   → Detalle del workspace activo
  │   ├── rename <id_o_nombre> <nuevo_nombre>
  │   ├── archive    <id_o_nombre>
  │   ├── reactivate <id_o_nombre>
  │   └── delete     <id_o_nombre>
  │
  ├── project
  │   ├── create <nombre>        → Crea proyecto en workspace activo
  │   ├── list                   → Lista proyectos del workspace activo
  │   ├── show   <id_o_nombre>   → Detalle de un proyecto
  │   ├── rename <id_o_nombre> <nuevo_nombre>
  │   ├── tag    <id_o_nombre> <tags...>
  │   ├── untag  <id_o_nombre> <tags...>
  │   ├── archive    <id_o_nombre>
  │   ├── reactivate <id_o_nombre>
  │   └── delete     <id_o_nombre>
  │
  └── config
      ├── get     [--workspace] [--global]
      ├── set     <campo> <valor> [--workspace]
      ├── unset   <campo> [--workspace]
      ├── reset   [--workspace] [--yes]
      └── fields  [--section]

════════════════════════════════════════════════════════════
INICIALIZACIÓN DEL SISTEMA
════════════════════════════════════════════════════════════

  El callback raíz de la app (@app.callback) se ejecuta ANTES de
  cualquier subcomando. Aquí es donde inicializamos el sistema:

    1. Inicializar structlog con el modo correcto (debug/prod).
    2. Verificar que el directorio de datos existe (~/.hiperforge/).
    3. Emitir el evento de inicio de sesión al EventBus.

  La inicialización ocurre una sola vez por invocación del proceso.
  Si el usuario ejecuta 100 comandos en un script, la inicialización
  ocurre 100 veces — una por proceso, que es lo correcto.

════════════════════════════════════════════════════════════
FLAGS GLOBALES
════════════════════════════════════════════════════════════

  --version / -V
    Muestra la versión instalada de HiperForge y termina.
    Formato: "HiperForge 0.1.0"

  --debug / -d
    Activa structlog en modo DEBUG con formato colorido en terminal.
    Los errores incluyen el traceback completo de Rich.
    Equivalente a HIPERFORGE_DEBUG=true.

  --verbose / -v
    Activa el modo verbose en el Renderer — muestra IDs, timestamps
    y detalles adicionales en los outputs de la CLI.
    No afecta al nivel de logging.

════════════════════════════════════════════════════════════
ENTRY POINT EN pyproject.toml
════════════════════════════════════════════════════════════

  [project.scripts]
  hiperforge = "hiperforge.cli.main:app"

  Al instalar el paquete con `pip install -e .`, este entry point
  crea el ejecutable `hiperforge` en el PATH del sistema.
  `pip install -e .` durante desarrollo crea el mismo ejecutable
  apuntando al código fuente para cambios en caliente.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.text import Text

from hiperforge.cli.commands import (
    config_app,
    project_app,
    run_command,
    task_app,
    workspace_app,
)
from hiperforge.core.constants import (
    APP_DESCRIPTION,
    APP_DISPLAY_NAME,
    APP_NAME,
    APP_VERSION,
    APP_DIR,
)

# ---------------------------------------------------------------------------
# Consola para el entrypoint — mensajes de versión e inicialización
# ---------------------------------------------------------------------------

_console = Console()

# ---------------------------------------------------------------------------
# Aplicación raíz de Typer
# ---------------------------------------------------------------------------

app = typer.Typer(
    name=APP_NAME,
    help=(
        f"[bold cyan]{APP_DISPLAY_NAME}[/bold cyan] — {APP_DESCRIPTION}\n\n"
        "Agente de IA para desarrolladores que entiende tu código,\n"
        "planifica las tareas y las ejecuta paso a paso.\n\n"
        "[dim]Documentación: https://hiperforge.dev/docs[/dim]"
    ),
    # Typer no invoca el help automáticamente si no hay subcomando
    # — queremos mostrar el banner en ese caso
    invoke_without_command=True,
    # Permitir que el callback se ejecute siempre, incluso con subcomandos
    no_args_is_help=False,
    # Rich markup en el texto del help
    rich_markup_mode="rich",
    # No agregar el comando "help" implícito de Typer — usamos --help nativo
    add_completion=True,
    pretty_exceptions_enable=False,  # Manejamos excepciones nosotros con ErrorHandler
    pretty_exceptions_show_locals=False,
    pretty_exceptions_short=False,
)

# ---------------------------------------------------------------------------
# Registro de subgrupos de comandos
# ---------------------------------------------------------------------------

# Cada grupo es un Typer app registrado como subcomando de la app raíz.
# El nombre de registro (primer argumento) es el prefijo del comando en la CLI.

app.add_typer(task_app,      name="task")
app.add_typer(workspace_app, name="workspace")
app.add_typer(project_app,   name="project")
app.add_typer(config_app,    name="config")

# El comando `run` es especial — no es un grupo sino un comando directo.
# Se registra directamente en la app raíz para que sea `hiperforge run "..."`.
app.command(
    name="run",
    help=(
        "Ejecuta una instrucción con el agente HiperForge.\n\n"
        "El agente planificará y ejecutará la tarea paso a paso.\n\n"
        "[bold]Ejemplo:[/bold]\n\n"
        "  [cyan]hiperforge run \"agrega tests al módulo auth\"[/cyan]"
    ),
    rich_markup_mode="rich",
)(run_command)


# ---------------------------------------------------------------------------
# Callback raíz — se ejecuta antes de CUALQUIER subcomando
# ---------------------------------------------------------------------------

@app.callback()
def main_callback(
    ctx: typer.Context,
    version: bool = typer.Option(
        False,
        "--version", "-V",
        help="Muestra la versión de HiperForge y termina.",
        is_eager=True,      # Se procesa antes que cualquier otro flag
        is_flag=True,
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Activa logs detallados y traceback completo en errores.",
        envvar="HIPERFORGE_DEBUG",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Muestra IDs, timestamps y detalles adicionales en los outputs.",
    ),
) -> None:
    """
    HiperForge — Agente de IA para desarrolladores.

    Usa [bold]hiperforge run \"tu instrucción\"[/bold] para empezar.

    Para ayuda sobre un comando específico:
      [cyan]hiperforge <comando> --help[/cyan]
    """
    # ── Flag --version: mostrar versión y salir ────────────────────────
    # is_eager=True garantiza que se procesa antes que cualquier subcomando
    if version:
        _print_version()
        raise typer.Exit(0)

    # ── Inicializar el sistema de logging ─────────────────────────────
    # Debe ocurrir antes de que cualquier subcomando haga su primera llamada
    # al logger. Si no se inicializa aquí, structlog usaría la configuración
    # por defecto (sin formato, sin archivo, sin nivel correcto).
    _initialize_logging(debug=debug)

    # ── Garantizar que el directorio de datos existe ───────────────────
    # En la primera ejecución, ~/.hiperforge/ no existe todavía.
    # Lo creamos aquí — antes de que cualquier repositorio intente leer de él.
    _ensure_app_directory()

    # ── Sin subcomando: mostrar el banner de bienvenida ────────────────
    # Si el usuario ejecutó `hiperforge` sin nada más, mostramos el banner
    # con las instrucciones básicas de uso.
    if ctx.invoked_subcommand is None:
        _print_welcome_banner()


# ---------------------------------------------------------------------------
# Helpers privados del entrypoint
# ---------------------------------------------------------------------------

def _initialize_logging(*, debug: bool) -> None:
    """
    Inicializa structlog con la configuración apropiada según el modo.

    Llamado UNA SOLA VEZ por invocación del proceso, desde el callback
    raíz de la app. Inicializar dos veces structlog tiene efectos no
    deseados — los procesadores se apilarían duplicados.

    El modo debug activa:
      - Nivel DEBUG (todos los logs internos visibles).
      - Formato colorido y legible en la terminal.
      - Traceback completo en el ErrorHandler.

    El modo producción (default) activa:
      - Nivel INFO (solo lo relevante para el usuario).
      - Formato JSON por línea para procesamiento externo.
      - Rotación diaria del archivo de logs.

    Parámetros:
        debug: Si True, activa el modo de desarrollo.
    """
    from hiperforge.core.logging import setup_logging

    setup_logging(debug=debug)


def _ensure_app_directory() -> None:
    """
    Garantiza que el directorio de datos de HiperForge existe en disco.

    En la primera ejecución, ~/.hiperforge/ no existe. Todos los
    repositorios asumen que el directorio existe para leer/escribir.
    Sin esta creación preventiva, la primera operación de cualquier
    repositorio fallaría con FileNotFoundError.

    Crea también los subdirectorios necesarios:
      ~/.hiperforge/workspaces/
      ~/.hiperforge/logs/
      ~/.hiperforge/locks/

    La operación es idempotente — si el directorio ya existe, no hace nada.
    """
    from hiperforge.core.constants import DIR_LOCKS, DIR_LOGS, DIR_WORKSPACES

    directories = [APP_DIR, DIR_WORKSPACES, DIR_LOGS, DIR_LOCKS]

    for directory in directories:
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            # Fallo crítico — sin el directorio de datos no podemos funcionar
            _console.print(
                f"[red]✗ No se pudo crear el directorio de datos:[/red] {directory}\n"
                f"[dim]Error: {exc}[/dim]\n"
                f"[dim]Verifica que tienes permisos de escritura en: {APP_DIR.parent}[/dim]"
            )
            raise typer.Exit(1)


def _print_version() -> None:
    """
    Imprime la versión de HiperForge y termina.

    Formato estándar compatible con scripts que parsean versiones:
      HiperForge 0.1.0

    También incluye información del entorno Python para facilitar
    el diagnóstico de problemas de compatibilidad.
    """
    import sys

    version_text = Text()
    version_text.append(APP_DISPLAY_NAME, style="bold cyan")
    version_text.append(f" {APP_VERSION}", style="bold white")

    _console.print(version_text)
    _console.print(
        f"[dim]Python {sys.version.split()[0]} · "
        f"{sys.platform} · "
        f"{APP_DIR}[/dim]"
    )


def _print_welcome_banner() -> None:
    """
    Muestra el banner de bienvenida cuando el usuario ejecuta `hiperforge` sin comandos.

    Diseñado para ser informativo y accionable:
      1. Identifica el producto y su versión.
      2. Muestra el comando más importante primero.
      3. Lista los grupos de comandos disponibles.
      4. Sugiere cómo obtener más ayuda.

    Este banner es la primera impresión del producto para usuarios nuevos
    y una referencia rápida para usuarios frecuentes.
    """
    from hiperforge.application import Container

    # Encabezado con nombre y versión
    _console.print(
        f"\n  [bold cyan]{APP_DISPLAY_NAME}[/bold cyan] "
        f"[dim]v{APP_VERSION}[/dim]  "
        f"[dim]— {APP_DESCRIPTION}[/dim]\n"
    )

    # Comando más importante primero — el que usará el 80% del tiempo
    _console.print(
        "  [dim]Uso principal:[/dim]\n"
        "    [bold cyan]hiperforge run[/bold cyan] [green]\"tu instrucción\"[/green]\n"
    )

    # Comandos disponibles organizados por grupo
    _console.print("  [dim]Comandos disponibles:[/dim]\n")

    commands = [
        ("run",       "Ejecutar una instrucción con el agente [dim](flujo completo)[/dim]"),
        ("task",      "Gestionar tasks: create, plan, run, list"),
        ("workspace", "Gestionar workspaces: create, list, switch, ..."),
        ("project",   "Gestionar proyectos: create, list, show, ..."),
        ("config",    "Configurar el agente: get, set, reset, fields"),
    ]

    for cmd_name, cmd_desc in commands:
        _console.print(
            f"    [bold cyan]{cmd_name:<12}[/bold cyan] {cmd_desc}"
        )

    # Workspace activo actual — contexto inmediato para el usuario
    _print_active_workspace_hint()

    # Instrucción de ayuda
    _console.print(
        "\n  [dim]Ayuda detallada:[/dim]\n"
        "    [dim]hiperforge --help[/dim]\n"
        "    [dim]hiperforge <comando> --help[/dim]\n"
    )


def _print_active_workspace_hint() -> None:
    """
    Muestra el workspace activo actual en el banner de bienvenida.

    Proporciona contexto inmediato al usuario sobre dónde operarán
    sus comandos. Si no hay workspace activo, sugiere cómo crear uno.

    Falla silenciosamente — si no puede leer el workspace activo
    (primera ejecución, datos corruptos, etc.), simplemente no muestra
    nada. El banner no debe fallar por un dato informativo.

    La importación del Container es tardía e intencional: en el momento
    en que main.py se importa como módulo (ej: autocompletion de shell),
    no queremos instanciar el Container ni leer disco. Solo lo hacemos
    cuando el usuario ejecuta `hiperforge` sin subcomandos.
    """
    try:
        from hiperforge.application import Container

        container = Container.build()
        active_id = container.store.get_active_workspace_id()

        if active_id:
            ws = container.store.workspaces.find_by_id_meta(active_id)
            _console.print(
                f"\n  [dim]Workspace activo:[/dim] "
                f"[bold]{ws.name}[/bold]"
                + (f" [dim]({ws.project_count} proyecto(s))[/dim]" if ws.project_count > 0 else "")
            )
        else:
            _console.print(
                "\n  [yellow]⚠[/yellow] [dim]Sin workspace activo. "
                "Crea uno con:[/dim] "
                "[bold cyan]hiperforge workspace create <nombre>[/bold cyan]"
            )

    except Exception:
        # Fallo silencioso — el banner informativo no debe bloquear la CLI
        pass


# ---------------------------------------------------------------------------
# Entry point del proceso
# ---------------------------------------------------------------------------

def cli_entry() -> None:
    """
    Función invocada por el entry_point de pyproject.toml.

    Configura la app Typer para que maneje Ctrl+C limpiamente
    antes de pasar el control al loop de Typer/Click.

    En pyproject.toml:
      [project.scripts]
      hiperforge = "hiperforge.cli.main:cli_entry"
    """
    app()


if __name__ == "__main__":
    # Soporte para ejecución directa durante desarrollo:
    #   python -m hiperforge.cli.main run "test"
    # Equivalente al entry_point en producción.
    cli_entry()