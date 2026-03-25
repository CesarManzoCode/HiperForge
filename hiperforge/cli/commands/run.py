"""
Comando `run` — Punto de entrada principal del agente HiperForge.

Es el comando más usado de toda la CLI. Representa el flujo completo
del agente: recibe una instrucción del desarrollador, la planifica,
la ejecuta con el loop ReAct, y muestra el resultado.

════════════════════════════════════════════════════════════
SUBCOMANDOS
════════════════════════════════════════════════════════════

  hiperforge run "<prompt>"
    Flujo completo: crea task → planifica → ejecuta.
    Es el comando rápido para la mayoría de los casos.

  hiperforge task create "<prompt>"
    Solo crea la task en estado PENDING. Sin planificar ni ejecutar.
    Útil para construir un backlog de trabajo.

  hiperforge task plan <task_id>
    Genera el plan de subtasks para una task en PENDING.
    El usuario puede revisar el plan antes de ejecutar.

  hiperforge task run <task_id>
    Ejecuta una task que ya tiene su plan generado (IN_PROGRESS).
    Complementa task plan para el flujo de tres fases.

  hiperforge task list
    Lista todas las tasks del workspace activo o de un proyecto.

════════════════════════════════════════════════════════════
FLAGS GLOBALES DEL COMANDO RUN
════════════════════════════════════════════════════════════

  --project  / -p   ID o nombre del proyecto al que vincular la task.
  --workspace / -w   ID o nombre del workspace. Default: activo global.
  --yes / -y         Confirmar el plan automáticamente sin mostrar preview.
  --verbose / -v     Mostrar IDs, timestamps y detalles adicionales.
  --debug / -d       Modo debug: logs detallados y traceback en errores.
  --provider         Sobreescribir el proveedor LLM para esta ejecución.
  --model            Sobreescribir el modelo LLM para esta ejecución.

════════════════════════════════════════════════════════════
FLUJO INTERNO DEL COMANDO run
════════════════════════════════════════════════════════════

  1. Construir Container con las dependencias del sistema.
  2. Verificar disponibilidad del LLM (con aviso si falla, no error).
  3. Construir PlanView con los callbacks de confirmación.
  4. Construir AgentSpinner y conectarlo al EventBus.
  5. Ejecutar RunTaskUseCase dentro del spinner.
  6. Mostrar el resultado final con el Renderer.

════════════════════════════════════════════════════════════
VARIACIONES DE COMPORTAMIENTO SEGÚN FLAGS
════════════════════════════════════════════════════════════

  Con --yes:
    El agente ejecuta el plan sin mostrar el preview ni pedir confirmación.
    Equivalente a responder 'S' siempre en el prompt de plan.

  Sin --yes (default):
    El agente genera el plan, lo muestra al usuario, y espera
    confirmación antes de ejecutar. Si el usuario cancela, la task
    queda en estado IN_PROGRESS para ejecutarse después.

  Con --debug:
    Activa logs estructurados de structlog en la terminal.
    Los errores muestran el traceback completo de Rich.
    Útil para diagnosticar fallos del LLM o de las tools.
"""

from __future__ import annotations

import sys
from typing import Optional

import typer
from rich.console import Console

from hiperforge.application import (
    Container,
    CreateTaskInput,
    RunTaskInput,
    TaskSummary,
)
from hiperforge.cli.error_handler import EXIT_SUCCESS, ErrorHandler
from hiperforge.cli.ui import AgentSpinner, Confirm, PlanView, Renderer
from hiperforge.core.events import get_event_bus
from hiperforge.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App Typer para el grupo de comandos `task`
# ---------------------------------------------------------------------------

# El app de `run` no tiene subcomandos propios — es un comando simple.
# El grupo de subcomandos de `task` vive en su propio app de Typer.
task_app = typer.Typer(
    name="task",
    help="Gestión de tasks del agente (crear, planificar, ejecutar, listar).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# Comando principal: hiperforge run "<prompt>"
# ---------------------------------------------------------------------------

def run_command(
    prompt: str = typer.Argument(
        ...,
        help="Instrucción para el agente. Describe qué quieres que haga.",
        metavar="PROMPT",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="ID o nombre del proyecto al que vincular la task.",
        metavar="PROJECT",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Confirmar el plan automáticamente sin mostrar el preview.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar IDs, timestamps y detalles adicionales.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Modo debug: logs detallados y traceback en errores.",
        envvar="HIPERFORGE_DEBUG",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Sobreescribir el proveedor LLM (anthropic, openai, groq, ollama).",
        metavar="PROVIDER",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Sobreescribir el modelo LLM para esta ejecución.",
        metavar="MODEL",
    ),
) -> None:
    """
    Ejecuta una instrucción completa con el agente HiperForge.

    El agente planificará la tarea en pasos concretos y los ejecutará
    usando sus herramientas disponibles (shell, archivos, git, web, código).

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge run "agrega tests unitarios al módulo auth"[/cyan]

      [cyan]hiperforge run "refactoriza la clase UserService para usar async" --yes[/cyan]

      [cyan]hiperforge run "instala y configura FastAPI" --project mi-api[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)
    console = Console()

    with error_handler.context("run"):
        # ── Construir el container con las dependencias ────────────────
        container = _build_container(
            debug=debug,
            provider_override=provider,
            model_override=model,
        )

        # ── Pre-flight: verificar disponibilidad del LLM ──────────────
        # Es un warning, no un error fatal — el usuario puede querer continuar
        # incluso si la verificación falla (ej: conectividad intermitente)
        _check_llm_availability(container, renderer, provider or container.settings.llm_provider)

        # ── Resolver el workspace activo para contexto de la UI ────────
        workspace_name, project_name = _resolve_context_names(
            container=container,
            workspace_override=workspace,
            project_override=project,
        )

        # ── Construir los componentes de UI ───────────────────────────
        plan_view = PlanView(
            renderer=renderer,
            console=console,
            workspace_name=workspace_name,
            project_name=project_name,
            model_name=container.settings.effective_llm_model,
            auto_confirm=yes,
        )
        spinner = AgentSpinner()

        # ── Resolver el project_id desde el nombre o ID ───────────────
        # Si el usuario pasó un nombre, lo resolvemos al ID
        resolved_project_id = _resolve_project_id(
            container=container,
            project_arg=project,
        )

        # ── Construir el input del use case ───────────────────────────
        input_data = RunTaskInput(
            prompt=prompt,
            project_id=resolved_project_id,
            workspace_id=_resolve_workspace_id(container, workspace),
            auto_confirm=yes,
        )

        # ── Ejecutar con spinner activo ───────────────────────────────
        with spinner.attach(get_event_bus()):
            output = container.run_task.execute(
                input_data,
                on_confirm_plan=plan_view.confirm_callback,
                on_subtask_limit_reached=plan_view.limit_callback,
            )

        # ── Mostrar resultado final ───────────────────────────────────
        renderer.render_task_result(output)

        # ── Exit code correcto según el resultado ─────────────────────
        if not output.succeeded:
            sys.exit(1)


# ---------------------------------------------------------------------------
# Subcomandos del grupo `task`
# ---------------------------------------------------------------------------

@task_app.command("create")
def task_create(
    prompt: str = typer.Argument(
        ...,
        help="Instrucción para el agente. La task quedará en estado PENDING.",
        metavar="PROMPT",
    ),
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="ID o nombre del proyecto al que vincular la task.",
        metavar="PROJECT",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar el ID de la task creada y detalles adicionales.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Crea una task en estado PENDING sin planificarla ni ejecutarla.

    La task queda pendiente para ser planificada y ejecutada cuando
    el usuario lo decida con [bold]task plan[/bold] y [bold]task run[/bold].

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge task create "implementar autenticación JWT"[/cyan]
      [cyan]hiperforge task plan <task_id>[/cyan]
      [cyan]hiperforge task run <task_id>[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("task create"):
        container = _build_container(debug=debug)

        input_data = CreateTaskInput(
            prompt=prompt,
            workspace_id=_resolve_workspace_id(container, workspace),
            project_id=_resolve_project_id(container, project),
        )

        summary = container.create_task.execute(input_data)
        renderer.render_task_created(summary)


@task_app.command("plan")
def task_plan(
    task_id: str = typer.Argument(
        ...,
        help="ID de la task a planificar. Debe estar en estado PENDING.",
        metavar="TASK_ID",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar detalles del plan generado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Sobreescribir el proveedor LLM para generar el plan.",
        metavar="PROVIDER",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Sobreescribir el modelo LLM para generar el plan.",
        metavar="MODEL",
    ),
) -> None:
    """
    Genera el plan de ejecución para una task en estado PENDING.

    El plan queda guardado — la task pasa a estado IN_PROGRESS.
    Usa [bold]task run <task_id>[/bold] para ejecutar el plan.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge task plan 01HX4K2J8QNVR0SBPZ1Y3W9D6E[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("task plan"):
        container = _build_container(
            debug=debug,
            provider_override=provider,
            model_override=model,
        )

        workspace_id = _resolve_workspace_id(container, workspace)

        summary = container.plan_task.execute(
            task_id=task_id,
            workspace_id=workspace_id,
        )

        # Mostrar el plan generado
        renderer.render_success(
            f"Plan generado para la task [cyan]{task_id[:16]}...[/cyan]"
        )

        # Mostrar las subtasks del plan si verbose
        if verbose and summary.subtask_count > 0:
            renderer.render_info(
                f"{summary.subtask_count} pasos planificados. "
                f"Ejecuta con: [bold]hiperforge task run {task_id}[/bold]"
            )
        else:
            renderer.render_info(
                f"{summary.subtask_count} pasos. "
                f"Ejecuta con: [bold]hiperforge task run {task_id}[/bold]"
            )


@task_app.command("run")
def task_run(
    task_id: str = typer.Argument(
        ...,
        help="ID de la task a ejecutar. Debe estar en estado IN_PROGRESS.",
        metavar="TASK_ID",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Ejecutar sin mostrar el plan previo.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar detalles adicionales del resultado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
    provider: Optional[str] = typer.Option(
        None,
        "--provider",
        help="Sobreescribir el proveedor LLM para la ejecución.",
        metavar="PROVIDER",
    ),
    model: Optional[str] = typer.Option(
        None,
        "--model",
        help="Sobreescribir el modelo LLM para la ejecución.",
        metavar="MODEL",
    ),
) -> None:
    """
    Ejecuta el plan de una task que ya fue planificada.

    La task debe estar en estado IN_PROGRESS (ya fue planificada
    con [bold]task plan[/bold]).

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge task run 01HX4K2J8QNVR0SBPZ1Y3W9D6E[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)
    console = Console()

    with error_handler.context("task run"):
        container = _build_container(
            debug=debug,
            provider_override=provider,
            model_override=model,
        )

        _check_llm_availability(
            container, renderer,
            provider or container.settings.llm_provider,
        )

        workspace_name, _ = _resolve_context_names(
            container=container,
            workspace_override=workspace,
            project_override=None,
        )

        plan_view = PlanView(
            renderer=renderer,
            console=console,
            workspace_name=workspace_name,
            model_name=container.settings.effective_llm_model,
            auto_confirm=yes,
        )
        spinner = AgentSpinner()

        input_data = RunTaskInput(
            task_id=task_id,
            workspace_id=_resolve_workspace_id(container, workspace),
            auto_confirm=yes,
        )

        with spinner.attach(get_event_bus()):
            output = container.run_task.execute(
                input_data,
                on_confirm_plan=plan_view.confirm_callback,
                on_subtask_limit_reached=plan_view.limit_callback,
            )

        renderer.render_task_result(output)

        if not output.succeeded:
            sys.exit(1)


@task_app.command("list")
def task_list(
    project: Optional[str] = typer.Option(
        None,
        "--project", "-p",
        help="Filtrar por proyecto (ID o nombre).",
        metavar="PROJECT",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status", "-s",
        help="Filtrar por estado (pending, in_progress, completed, failed).",
        metavar="STATUS",
    ),
    limit: int = typer.Option(
        20,
        "--limit", "-l",
        help="Número máximo de tasks a mostrar.",
        min=1,
        max=200,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar IDs y detalles adicionales.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug", "-d",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Lista las tasks del workspace activo o de un proyecto específico.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge task list[/cyan]
      [cyan]hiperforge task list --project mi-api --status completed[/cyan]
      [cyan]hiperforge task list --limit 50[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("task list"):
        container = _build_container(debug=debug)
        workspace_id = _resolve_workspace_id(container, workspace)

        # Cargar tasks del workspace/proyecto
        tasks = _load_tasks(
            container=container,
            workspace_id=workspace_id,
            project_arg=project,
            status_filter=status,
            limit=limit,
        )

        renderer.render_task_list(tasks)


# ---------------------------------------------------------------------------
# Helpers privados compartidos entre subcomandos
# ---------------------------------------------------------------------------

def _build_container(
    debug: bool = False,
    provider_override: str | None = None,
    model_override: str | None = None,
) -> Container:
    """
    Construye el Container con posibles overrides de proveedor y modelo.

    Si se especifican provider_override o model_override, los inyecta
    en el entorno antes de que get_settings() los lea. Esto permite
    que el flag --provider sobreescriba la configuración sin modificar
    el archivo .env ni las preferencias persistidas.

    Parámetros:
        debug:             Activa el modo debug en el container.
        provider_override: Proveedor LLM a usar en vez del configurado.
        model_override:    Modelo LLM a usar en vez del configurado.

    Returns:
        Container completamente construido con las dependencias.
    """
    import os

    # Inyectamos overrides en las variables de entorno del proceso.
    # get_settings() usa pydantic-settings que lee variables de entorno,
    # así que cualquier override aquí se refleja en Settings.
    # IMPORTANTE: solo afecta al proceso actual, no al .env del usuario.
    if provider_override:
        os.environ["HIPERFORGE_LLM_PROVIDER"] = provider_override

    if model_override:
        os.environ["HIPERFORGE_LLM_MODEL"] = model_override

    if debug:
        os.environ["HIPERFORGE_DEBUG"] = "true"

    # Forzamos recarga de settings para que tome los overrides
    from hiperforge.core.config import get_settings
    get_settings.cache_clear()

    return Container.build()


def _check_llm_availability(
    container: Container,
    renderer: Renderer,
    provider_name: str,
) -> None:
    """
    Verifica que el LLM configurado está disponible.

    Si no está disponible, muestra un warning pero NO cancela la ejecución.
    El usuario puede querer continuar igual (ej: conectividad intermitente).

    Parámetros:
        container:     Container con el LLM adapter ya configurado.
        renderer:      Renderer para mostrar el warning.
        provider_name: Nombre del proveedor para el mensaje.
    """
    available = container.check_llm_availability()

    if not available:
        renderer.render_warning(
            f"No se pudo verificar la disponibilidad de {provider_name}. "
            f"Verifica tu API key y conexión. La ejecución continuará de todas formas."
        )
        logger.warning(
            "LLM no disponible antes de ejecutar",
            provider=provider_name,
        )


def _resolve_workspace_id(
    container: Container,
    workspace_arg: str | None,
) -> str | None:
    """
    Resuelve el workspace_id a partir del argumento del usuario.

    Si el argumento parece un ULID (26 chars), lo usa como ID directamente.
    Si parece un nombre, busca el workspace por nombre en el store.
    Si es None, devuelve None — el use case usará el workspace activo.

    Parámetros:
        container:     Container con acceso al store.
        workspace_arg: Argumento --workspace del usuario (ID o nombre) o None.

    Returns:
        workspace_id si se pudo resolver, None si no se especificó.
    """
    if workspace_arg is None:
        return None

    # Heurística: los ULIDs tienen exactamente 26 caracteres alfanuméricos
    if len(workspace_arg) == 26 and workspace_arg.isalnum():
        return workspace_arg

    # Es un nombre — buscamos por nombre
    workspaces = container.store.workspaces.find_all()
    for ws in workspaces:
        if ws.name.lower() == workspace_arg.lower():
            return ws.id

    # No encontramos por nombre — devolvemos tal cual y dejamos
    # que el use case lo maneje con EntityNotFound
    return workspace_arg


def _resolve_project_id(
    container: Container,
    project_arg: str | None,
) -> str | None:
    """
    Resuelve el project_id a partir del argumento del usuario.

    Usa la misma heurística que _resolve_workspace_id:
    ULID = ID directo, texto = búsqueda por nombre en el workspace activo.

    Parámetros:
        container:   Container con acceso al store.
        project_arg: Argumento --project del usuario (ID o nombre) o None.

    Returns:
        project_id si se pudo resolver, None si no se especificó.
    """
    if project_arg is None:
        return None

    # Heurística ULID
    if len(project_arg) == 26 and project_arg.isalnum():
        return project_arg

    # Buscar por nombre en el workspace activo
    workspace_id = container.store.get_active_workspace_id()
    if not workspace_id:
        return project_arg  # Sin workspace activo, devolvemos tal cual

    try:
        projects = container.store.projects.find_all(workspace_id)
        for proj in projects:
            if proj.name.lower() == project_arg.lower():
                return proj.id
    except Exception:
        pass  # Si falla la búsqueda, devolvemos el argumento tal cual

    return project_arg


def _resolve_context_names(
    container: Container,
    workspace_override: str | None,
    project_override: str | None,
) -> tuple[str, str]:
    """
    Resuelve los nombres del workspace y proyecto activos para la UI.

    Estos nombres se usan en el PlanView para dar contexto al usuario
    sobre dónde se ejecutará la task — no son críticos para la ejecución,
    solo para la UI.

    Returns:
        Tupla (workspace_name, project_name).
        Strings vacíos si no se puede resolver.
    """
    workspace_name = ""
    project_name = ""

    try:
        workspace_id = (
            _resolve_workspace_id(container, workspace_override)
            or container.store.get_active_workspace_id()
        )
        if workspace_id:
            ws = container.store.workspaces.find_by_id_meta(workspace_id)
            workspace_name = ws.name
    except Exception:
        pass

    try:
        if project_override and workspace_name:
            project_id = _resolve_project_id(container, project_override)
            if project_id:
                workspace_id = container.store.get_active_workspace_id() or ""
                proj = container.store.projects.find_by_id_meta(
                    workspace_id, project_id
                )
                project_name = proj.name
    except Exception:
        pass

    return workspace_name, project_name


def _load_tasks(
    container: Container,
    workspace_id: str | None,
    project_arg: str | None,
    status_filter: str | None,
    limit: int,
) -> list[TaskSummary]:
    """
    Carga tasks del workspace con filtros opcionales y las convierte a TaskSummary.

    Parámetros:
        container:     Container con acceso al store.
        workspace_id:  ID del workspace a consultar.
        project_arg:   Filtro por proyecto (ID o nombre) o None para todos.
        status_filter: Filtro por estado o None para todos.
        limit:         Número máximo de tasks a devolver.

    Returns:
        Lista de TaskSummary ordenadas de más recientes a más antiguas.
    """
    effective_workspace_id = workspace_id or container.store.get_active_workspace_id()
    if not effective_workspace_id:
        return []

    project_id = _resolve_project_id(container, project_arg)

    # Recolectamos tasks de todos los proyectos o del proyecto específico
    all_tasks = []

    try:
        if project_id:
            # Solo las tasks del proyecto indicado
            raw_tasks = container.store.tasks.find_all(
                workspace_id=effective_workspace_id,
                project_id=project_id,
            )
            all_tasks.extend(raw_tasks)
        else:
            # Tasks de todos los proyectos del workspace
            projects = container.store.projects.find_all(effective_workspace_id)
            for proj in projects:
                try:
                    raw_tasks = container.store.tasks.find_all(
                        workspace_id=effective_workspace_id,
                        project_id=proj.id,
                    )
                    all_tasks.extend(raw_tasks)
                except Exception:
                    continue

    except Exception as exc:
        logger.warning("error cargando tasks", error=str(exc))
        return []

    # Aplicar filtro de estado si se especificó
    if status_filter:
        all_tasks = [t for t in all_tasks if t.status.value == status_filter]

    # Ordenar de más reciente a más antigua y aplicar límite
    all_tasks.sort(key=lambda t: t.created_at, reverse=True)
    all_tasks = all_tasks[:limit]

    # Convertir a TaskSummary para la CLI
    return [
        TaskSummary(
            id=task.id,
            prompt=task.prompt,
            status=task.status.value,
            project_id=task.project_id,
            subtask_count=len(task.subtasks),
            completed_subtasks=len(task.completed_subtasks),
            total_tokens=task.token_usage.total_tokens,
            estimated_cost_usd=task.token_usage.estimated_cost_usd,
            created_at=task.created_at,
            completed_at=task.completed_at,
        )
        for task in all_tasks
    ]