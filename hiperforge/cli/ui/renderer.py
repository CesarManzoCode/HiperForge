"""
Renderer — Motor de presentación visual de la CLI de HiperForge.

Centraliza toda la lógica de renderizado con Rich. Ningún comando
de la CLI imprime directamente — todos pasan por el Renderer.

════════════════════════════════════════════════════════════
FILOSOFÍA DE DISEÑO
════════════════════════════════════════════════════════════

  La CLI de HiperForge es la interfaz principal con el usuario.
  Cada output debe ser:

    CLARO     → el usuario entiende qué pasó en 2 segundos.
    ACCIONABLE → el usuario sabe qué hacer a continuación.
    CONSISTENTE → el mismo tipo de información siempre se ve igual.
    EFICIENTE  → no hay ruido visual — cada elemento tiene propósito.

  El Renderer garantiza estas propiedades centralizando todo el
  formateo. Si necesitas cambiar cómo se muestra algo, lo cambias
  en un solo lugar.

════════════════════════════════════════════════════════════
SISTEMA DE COLORES Y ESTILOS
════════════════════════════════════════════════════════════

  HiperForge usa un sistema de colores semántico consistente:

    ÉXITO    → green  (✓ completado, operaciones exitosas)
    ERROR    → red    (✗ fallos, errores críticos)
    AVISO    → yellow (⚠ advertencias, estados de cuidado)
    INFO     → cyan   (● información neutral, IDs, métricas)
    ACTIVO   → bold   (workspace activo, elemento seleccionado)
    INACTIVO → dim    (archivado, metadatos secundarios)

    Los iconos son consistentes:
      ● = activo / en progreso
      ✓ = completado exitosamente
      ✗ = fallido
      ○ = inactivo / archivado / pendiente
      ▶ = seleccionado / activo actualmente
      ◐ = en proceso (planning)
      ⊘ = cancelado

════════════════════════════════════════════════════════════
MÉTODOS DISPONIBLES
════════════════════════════════════════════════════════════

  Resultados de ejecución:
    render_task_result(output)      → panel de resultado de una task
    render_task_summary(summary)    → fila en listado de tasks

  Workspaces:
    render_workspace_created(ws)    → confirmación de workspace creado
    render_workspace_switched(ws)   → confirmación de switch
    render_workspace_list(workspaces) → tabla de workspaces

  Proyectos:
    render_project_created(proj)    → confirmación de proyecto creado
    render_project_list(projects)   → tabla de proyectos

  Tasks:
    render_task_created(summary)    → confirmación de task creada
    render_plan_preview(task)       → preview del plan antes de confirmar

  Configuración:
    render_prefs(prefs, level)      → tabla de preferencias actuales
    render_prefs_updated(updates)   → confirmación de actualización

  Utilidades:
    render_success(msg)             → mensaje de éxito genérico
    render_warning(msg)             → mensaje de advertencia
    render_info(msg)                → mensaje informativo
    render_id(entity, id_str)       → ID copiable con contexto

USO EN COMANDOS:
  renderer = Renderer()
  renderer.render_workspace_list(workspaces)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from hiperforge.application.dto import (
    ProjectSummary,
    RunTaskOutput,
    TaskSummary,
    WorkspaceSummary,
)
from hiperforge.memory.schemas.preferences import UserPrefsSchema

# ---------------------------------------------------------------------------
# Consola global — única instancia por proceso
# ---------------------------------------------------------------------------

# stdout para output normal de la CLI (puede ser redirigido por scripts)
_stdout_console = Console(highlight=True)

# stderr para mensajes de estado que no deben contaminar el output
_stderr_console = Console(stderr=True, highlight=False)


# ---------------------------------------------------------------------------
# Constantes de estilo — sistema de colores semántico
# ---------------------------------------------------------------------------

# Colores por estado de entidades
_STATUS_COLORS: dict[str, str] = {
    "active":      "green",
    "in_progress": "cyan",
    "completed":   "green",
    "pending":     "dim",
    "planning":    "yellow",
    "failed":      "red",
    "cancelled":   "yellow",
    "archived":    "dim",
    "deleted":     "red dim",
}

# Iconos por estado de entidades
_STATUS_ICONS: dict[str, str] = {
    "active":      "●",
    "in_progress": "●",
    "completed":   "✓",
    "pending":     "○",
    "planning":    "◐",
    "failed":      "✗",
    "cancelled":   "⊘",
    "archived":    "○",
    "deleted":     "✕",
}

# Ancho máximo para descripciones en tablas (para evitar tablas muy anchas)
_MAX_DESC_WIDTH = 60

# Formato de fechas en la CLI
_DATE_FORMAT = "%d/%m/%Y %H:%M"


class Renderer:
    """
    Motor de presentación visual de la CLI de HiperForge.

    Centraliza todo el renderizado con Rich para garantizar
    consistencia visual en toda la interfaz.

    Parámetros:
        console: Consola Rich para output. None = usar la consola global.
        verbose: Si True, incluye información adicional (IDs, timestamps, etc.).
    """

    def __init__(
        self,
        console: Console | None = None,
        verbose: bool = False,
    ) -> None:
        self._console = console or _stdout_console
        self._verbose = verbose

    # ──────────────────────────────────────────────────────────────────
    # RESULTADOS DE EJECUCIÓN DE TASKS
    # ──────────────────────────────────────────────────────────────────

    def render_task_result(self, output: RunTaskOutput) -> None:
        """
        Renderiza el resultado final de ejecutar una task con el agente.

        Muestra un panel con:
          - Estado final con ícono y color semántico
          - Resumen generado por el agente
          - Métricas: subtasks, tokens, costo, duración
          - ID de la task para referencia futura

        Parámetros:
            output: RunTaskOutput con el resultado completo de la ejecución.
        """
        if output.succeeded:
            self._render_task_success(output)
        elif output.status == "cancelled":
            self._render_task_cancelled(output)
        else:
            self._render_task_failure(output)

    def _render_task_success(self, output: RunTaskOutput) -> None:
        """Renderiza el resultado de una task completada exitosamente."""
        # Barra de progreso estática (100%)
        progress_bar = self._build_static_progress_bar(
            completed=output.subtasks_completed,
            total=output.subtasks_total,
            color="green",
        )

        # Métricas de la ejecución
        metrics = self._build_execution_metrics(output)

        # Contenido del panel
        content_lines: list[str] = []

        if output.summary:
            # Resumen del agente — el valor más importante para el usuario
            content_lines.append(f"[white]{output.summary}[/white]")
            content_lines.append("")

        content_lines.append(progress_bar)
        content_lines.append("")
        content_lines.append(metrics)

        if self._verbose:
            content_lines.append("")
            content_lines.append(f"[dim]Task ID: {output.task_id}[/dim]")

        self._console.print(
            Panel(
                "\n".join(content_lines),
                title="[bold green]✓ Task completada[/bold green]",
                border_style="green",
                padding=(1, 2),
            )
        )

    def _render_task_failure(self, output: RunTaskOutput) -> None:
        """Renderiza el resultado de una task que falló."""
        content_lines: list[str] = []

        if output.error_message:
            content_lines.append(f"[red]{output.error_message}[/red]")
            content_lines.append("")

        # Métricas incluso en fallo — útil para debug
        if output.subtasks_total > 0:
            progress_bar = self._build_static_progress_bar(
                completed=output.subtasks_completed,
                total=output.subtasks_total,
                color="red",
            )
            content_lines.append(progress_bar)
            content_lines.append("")

        metrics = self._build_execution_metrics(output)
        content_lines.append(metrics)
        content_lines.append("")
        content_lines.append(
            "[dim]Sugerencias:[/dim]\n"
            "[dim]  • Intenta con una instrucción más específica.[/dim]\n"
            "[dim]  • Divide la tarea en pasos más pequeños.[/dim]\n"
            "[dim]  • Ejecuta con --debug para ver el log detallado.[/dim]"
        )

        if self._verbose:
            content_lines.append("")
            content_lines.append(f"[dim]Task ID: {output.task_id}[/dim]")

        self._console.print(
            Panel(
                "\n".join(content_lines),
                title="[bold red]✗ Task falló[/bold red]",
                border_style="red",
                padding=(1, 2),
            )
        )

    def _render_task_cancelled(self, output: RunTaskOutput) -> None:
        """Renderiza el resultado de una task cancelada por el usuario."""
        content_lines = [
            "[yellow]La task fue cancelada antes de completarse.[/yellow]",
            "",
            self._build_execution_metrics(output),
        ]

        if self._verbose:
            content_lines.append("")
            content_lines.append(f"[dim]Task ID: {output.task_id}[/dim]")

        self._console.print(
            Panel(
                "\n".join(content_lines),
                title="[bold yellow]⊘ Task cancelada[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

    # ──────────────────────────────────────────────────────────────────
    # WORKSPACES
    # ──────────────────────────────────────────────────────────────────

    def render_workspace_created(self, workspace: WorkspaceSummary) -> None:
        """
        Renderiza la confirmación de un workspace recién creado.

        Muestra el nombre, estado, si quedó como activo y el ID
        para referencia futura.
        """
        active_label = (
            " [green](activo)[/green]" if workspace.is_active else ""
        )

        lines = [
            f"[bold]{workspace.name}[/bold]{active_label}",
        ]

        if workspace.description:
            lines.append(f"[dim]{workspace.description}[/dim]")

        lines.append("")
        lines.append(f"[dim]ID: {workspace.id}[/dim]")
        lines.append(f"[dim]Creado: {self._format_datetime(workspace.created_at)}[/dim]")

        if workspace.is_active:
            lines.append("")
            lines.append(
                "[dim]Este workspace quedó como activo. "
                "Todas las operaciones operarán en él por defecto.[/dim]"
            )

        self._console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]✓ Workspace creado[/bold green]",
                border_style="green",
                padding=(1, 2),
                expand=False,
            )
        )

    def render_workspace_switched(self, workspace: WorkspaceSummary) -> None:
        """Renderiza la confirmación de haber cambiado el workspace activo."""
        self._console.print(
            f"[green]▶[/green] Workspace activo: [bold]{workspace.name}[/bold]"
            + (f" [dim]({workspace.id[:16]}...)[/dim]" if self._verbose else "")
        )

        if workspace.project_count > 0:
            self._console.print(
                f"  [dim]{workspace.project_count} proyecto(s) disponibles[/dim]"
            )

    def render_workspace_list(self, workspaces: list[WorkspaceSummary]) -> None:
        """
        Renderiza la lista de todos los workspaces en una tabla formateada.

        El workspace activo se marca con ▶ y aparece en negrita.
        Los workspaces archivados aparecen en dim.

        Parámetros:
            workspaces: Lista de WorkspaceSummary a mostrar.
        """
        if not workspaces:
            self.render_info(
                "No hay workspaces creados todavía.\n"
                "Crea uno con: [bold]hiperforge workspace create <nombre>[/bold]"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
            show_edge=True,
        )

        table.add_column("", width=2, no_wrap=True)           # Icono de estado
        table.add_column("Nombre", style="bold", min_width=15)
        table.add_column("Estado", width=12)
        table.add_column("Proyectos", justify="right", width=10)
        table.add_column("Creado", width=16)

        if self._verbose:
            table.add_column("ID", width=26, style="dim")

        # Ordenamos: activo primero, luego alfabéticamente
        sorted_workspaces = sorted(
            workspaces,
            key=lambda w: (0 if w.is_active else 1, w.name.lower()),
        )

        for ws in sorted_workspaces:
            icon = "▶" if ws.is_active else _STATUS_ICONS.get(ws.status, "?")
            icon_style = "bold green" if ws.is_active else _STATUS_COLORS.get(ws.status, "")

            name_style = "bold" if ws.is_active else ("dim" if ws.status == "archived" else "")
            status_color = _STATUS_COLORS.get(ws.status, "")

            row_data = [
                Text(icon, style=icon_style),
                Text(ws.name, style=name_style),
                Text(ws.status, style=status_color),
                Text(str(ws.project_count), style="dim"),
                Text(self._format_datetime(ws.created_at), style="dim"),
            ]

            if self._verbose:
                row_data.append(Text(ws.id, style="dim"))

            table.add_row(*row_data)

        self._console.print(table)
        self._console.print(
            f"[dim]{len(workspaces)} workspace(s) total[/dim]"
        )

    # ──────────────────────────────────────────────────────────────────
    # PROYECTOS
    # ──────────────────────────────────────────────────────────────────

    def render_project_created(self, project: ProjectSummary) -> None:
        """Renderiza la confirmación de un proyecto recién creado."""
        lines = [
            f"[bold]{project.name}[/bold]",
        ]

        if project.description:
            lines.append(f"[dim]{project.description}[/dim]")

        if project.tags:
            tags_str = "  ".join(f"[cyan]#{t}[/cyan]" for t in project.tags)
            lines.append(tags_str)

        lines.append("")
        lines.append(f"[dim]ID: {project.id}[/dim]")
        lines.append(f"[dim]Creado: {self._format_datetime(project.created_at)}[/dim]")
        lines.append("")
        lines.append(
            "[dim]Ejecuta tasks en este proyecto con:\n"
            f"  hiperforge run \"tu instrucción\" --project {project.id}[/dim]"
        )

        self._console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]✓ Proyecto creado[/bold green]",
                border_style="green",
                padding=(1, 2),
                expand=False,
            )
        )

    def render_project_list(self, projects: list[ProjectSummary]) -> None:
        """
        Renderiza la lista de proyectos con métricas de progreso.

        Parámetros:
            projects: Lista de ProjectSummary a mostrar.
        """
        if not projects:
            self.render_info(
                "No hay proyectos en este workspace.\n"
                "Crea uno con: [bold]hiperforge project create <nombre>[/bold]"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
            show_edge=True,
        )

        table.add_column("", width=2, no_wrap=True)
        table.add_column("Nombre", style="bold", min_width=15)
        table.add_column("Tasks", justify="right", width=8)
        table.add_column("Progreso", width=12)
        table.add_column("Tags", width=20)
        table.add_column("Actualizado", width=16)

        if self._verbose:
            table.add_column("ID", width=26, style="dim")

        for proj in sorted(projects, key=lambda p: p.name.lower()):
            icon = _STATUS_ICONS.get(proj.status, "?")
            icon_style = _STATUS_COLORS.get(proj.status, "")
            name_style = "dim" if proj.status == "archived" else ""

            # Barra de progreso compacta
            progress_str = self._build_compact_progress(
                completed=proj.completed_tasks,
                total=proj.task_count,
            )

            tags_str = " ".join(f"#{t}" for t in proj.tags[:3])
            if len(proj.tags) > 3:
                tags_str += f" +{len(proj.tags) - 3}"

            row_data = [
                Text(icon, style=icon_style),
                Text(proj.name, style=name_style),
                Text(
                    f"{proj.completed_tasks}/{proj.task_count}" if proj.task_count > 0 else "—",
                    style="dim",
                ),
                Text(progress_str, style="green" if proj.completion_pct == 100 else "dim"),
                Text(tags_str, style="cyan dim"),
                Text(self._format_datetime(proj.updated_at), style="dim"),
            ]

            if self._verbose:
                row_data.append(Text(proj.id, style="dim"))

            table.add_row(*row_data)

        self._console.print(table)
        self._console.print(f"[dim]{len(projects)} proyecto(s) total[/dim]")

    # ──────────────────────────────────────────────────────────────────
    # TASKS
    # ──────────────────────────────────────────────────────────────────

    def render_task_created(self, summary: TaskSummary) -> None:
        """
        Renderiza la confirmación de una task creada en estado PENDING.

        Incluye los próximos pasos para planificar y ejecutar la task.
        """
        lines = [
            f"[bold]{summary.prompt_preview}[/bold]",
            "",
            f"[dim]Estado: [/dim]{_STATUS_ICONS.get(summary.status, '?')} "
            f"[{_STATUS_COLORS.get(summary.status, '')}]{summary.status}[/]",
        ]

        if summary.project_id:
            lines.append(f"[dim]Proyecto: {summary.project_id}[/dim]")

        lines.append(f"[dim]ID: {summary.id}[/dim]")
        lines.append("")
        lines.append(
            "[dim]Próximos pasos:\n"
            f"  Generar plan:  [bold]hiperforge task plan {summary.id}[/bold]\n"
            f"  Ejecutar:      [bold]hiperforge task run {summary.id}[/bold][/dim]"
        )

        self._console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]✓ Task creada[/bold green]",
                border_style="green",
                padding=(1, 2),
                expand=False,
            )
        )

    def render_plan_preview(
        self,
        task_prompt: str,
        subtasks: list,
        *,
        workspace_name: str = "",
        project_name: str = "",
    ) -> None:
        """
        Renderiza el plan de subtasks antes de pedir confirmación al usuario.

        Muestra cada subtask numerada con su descripción completa.
        Es el output más crítico de la CLI — el usuario decide aquí
        si el agente entendió correctamente la instrucción.

        Parámetros:
            task_prompt:    Prompt original del usuario.
            subtasks:       Lista de Subtask del dominio.
            workspace_name: Nombre del workspace activo (para contexto).
            project_name:   Nombre del proyecto (para contexto).
        """
        # Árbol de Rich para mostrar el plan de forma jerárquica
        tree = Tree(
            f"[bold cyan]Plan de ejecución[/bold cyan] "
            f"[dim]({len(subtasks)} pasos)[/dim]",
            guide_style="dim",
        )

        for subtask in sorted(subtasks, key=lambda s: s.order):
            step_num = f"[dim]{subtask.order + 1:02d}.[/dim]"
            tree.add(f"{step_num} {subtask.description}")

        # Contexto de la instrucción original
        context_parts: list[str] = []
        if workspace_name:
            context_parts.append(f"Workspace: [bold]{workspace_name}[/bold]")
        if project_name:
            context_parts.append(f"Proyecto: [bold]{project_name}[/bold]")

        context_str = "  •  ".join(context_parts) if context_parts else ""

        header = Text()
        header.append("Instrucción: ", style="dim")
        header.append(
            task_prompt[:100] + ("..." if len(task_prompt) > 100 else ""),
            style="bold white",
        )

        lines = [str(header)]
        if context_str:
            lines.append(f"[dim]{context_str}[/dim]")
        lines.append("")

        self._console.print(
            Panel(
                "\n".join(lines),
                border_style="cyan",
                padding=(0, 2),
                expand=False,
            )
        )
        self._console.print(tree)

    def render_task_list(self, tasks: list[TaskSummary]) -> None:
        """
        Renderiza la lista de tasks con su estado y progreso.

        Parámetros:
            tasks: Lista de TaskSummary a mostrar.
        """
        if not tasks:
            self.render_info(
                "No hay tasks en este proyecto.\n"
                "Ejecuta una con: [bold]hiperforge run \"tu instrucción\"[/bold]"
            )
            return

        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
            show_edge=True,
        )

        table.add_column("", width=2, no_wrap=True)
        table.add_column("Prompt", min_width=30, max_width=_MAX_DESC_WIDTH)
        table.add_column("Estado", width=14)
        table.add_column("Progreso", width=12)
        table.add_column("Tokens", justify="right", width=10)
        table.add_column("Costo", justify="right", width=8)
        table.add_column("Creada", width=16)

        if self._verbose:
            table.add_column("ID", width=26, style="dim")

        # Ordenamos más recientes primero
        sorted_tasks = sorted(tasks, key=lambda t: t.created_at, reverse=True)

        for task in sorted_tasks:
            icon = _STATUS_ICONS.get(task.status, "?")
            icon_style = _STATUS_COLORS.get(task.status, "")
            status_color = _STATUS_COLORS.get(task.status, "")

            progress_str = self._build_compact_progress(
                completed=task.completed_subtasks,
                total=task.subtask_count,
            ) if task.subtask_count > 0 else "—"

            cost_str = (
                f"${task.estimated_cost_usd:.3f}"
                if task.estimated_cost_usd > 0
                else "—"
            )

            row_data = [
                Text(icon, style=icon_style),
                Text(task.prompt_preview, overflow="ellipsis"),
                Text(task.status, style=status_color),
                Text(progress_str, style="dim"),
                Text(f"{task.total_tokens:,}" if task.total_tokens > 0 else "—", style="dim"),
                Text(cost_str, style="dim"),
                Text(self._format_datetime(task.created_at), style="dim"),
            ]

            if self._verbose:
                row_data.append(Text(task.id, style="dim"))

            table.add_row(*row_data)

        self._console.print(table)
        self._console.print(f"[dim]{len(tasks)} task(s) total[/dim]")

    # ──────────────────────────────────────────────────────────────────
    # CONFIGURACIÓN / PREFERENCIAS
    # ──────────────────────────────────────────────────────────────────

    def render_prefs(
        self,
        prefs: UserPrefsSchema,
        *,
        level: str = "global",
        workspace_name: str = "",
    ) -> None:
        """
        Renderiza las preferencias actuales en una tabla organizada por sección.

        Parámetros:
            prefs:          UserPrefsSchema con los valores actuales.
            level:          "global" o "workspace" — para el título del panel.
            workspace_name: Nombre del workspace si level="workspace".
        """
        # Sección LLM
        llm_table = Table(
            show_header=False,
            border_style="dim",
            box=None,
            padding=(0, 2),
        )
        llm_table.add_column("Campo", style="dim", width=25)
        llm_table.add_column("Valor", style="bold")

        llm_table.add_row("proveedor", prefs.llm.provider)
        llm_table.add_row(
            "modelo",
            prefs.llm.model or f"[dim](default del proveedor)[/dim]"
        )
        llm_table.add_row("temperatura", str(prefs.llm.temperature))
        llm_table.add_row("max_tokens", str(prefs.llm.max_tokens))

        # Sección Agent
        agent_table = Table(
            show_header=False,
            border_style="dim",
            box=None,
            padding=(0, 2),
        )
        agent_table.add_column("Campo", style="dim", width=25)
        agent_table.add_column("Valor", style="bold")

        agent_table.add_row(
            "max_react_iterations",
            str(prefs.agent.max_react_iterations)
        )
        agent_table.add_row("max_subtasks", str(prefs.agent.max_subtasks))
        agent_table.add_row(
            "tool_timeout_seconds",
            f"{prefs.agent.tool_timeout_seconds}s"
        )
        agent_table.add_row(
            "auto_confirm_plan",
            "[green]Sí[/green]" if prefs.agent.auto_confirm_plan else "[dim]No[/dim]"
        )
        agent_table.add_row(
            "show_reasoning",
            "[green]Sí[/green]" if prefs.agent.show_reasoning else "[dim]No[/dim]"
        )

        # Sección UI
        ui_table = Table(
            show_header=False,
            border_style="dim",
            box=None,
            padding=(0, 2),
        )
        ui_table.add_column("Campo", style="dim", width=25)
        ui_table.add_column("Valor", style="bold")

        ui_table.add_row(
            "show_token_usage",
            "[green]Sí[/green]" if prefs.ui.show_token_usage else "[dim]No[/dim]"
        )
        ui_table.add_row(
            "show_timestamps",
            "[green]Sí[/green]" if prefs.ui.show_timestamps else "[dim]No[/dim]"
        )
        ui_table.add_row(
            "verbose",
            "[green]Sí[/green]" if prefs.ui.verbose else "[dim]No[/dim]"
        )

        # Título del panel según el nivel
        if level == "workspace" and workspace_name:
            title_str = f"Configuración del workspace [bold]{workspace_name}[/bold]"
        else:
            title_str = "Configuración global"

        # Imprimimos cada sección con su encabezado
        self._console.print(f"\n[bold cyan]{title_str}[/bold cyan]\n")

        self._console.print("[bold dim]── LLM ──[/bold dim]")
        self._console.print(llm_table)

        self._console.print("\n[bold dim]── Agente ──[/bold dim]")
        self._console.print(agent_table)

        self._console.print("\n[bold dim]── Interfaz ──[/bold dim]")
        self._console.print(ui_table)

        self._console.print(
            "\n[dim]Modifica con: "
            "[bold]hiperforge config set <campo> <valor>[/bold][/dim]\n"
        )

    def render_prefs_updated(
        self,
        updates: dict[str, Any],
        level: str = "global",
    ) -> None:
        """
        Renderiza la confirmación de una actualización de preferencias.

        Parámetros:
            updates: Dict de campo → valor actualizado.
            level:   "global" o "workspace".
        """
        scope = "globales" if level == "global" else f"del workspace ({level})"
        lines = [f"[dim]Preferencias {scope} actualizadas:[/dim]", ""]

        for field_path, value in updates.items():
            lines.append(f"  [cyan]{field_path}[/cyan] → [bold]{value!r}[/bold]")

        self._console.print(
            Panel(
                "\n".join(lines),
                title="[bold green]✓ Configuración actualizada[/bold green]",
                border_style="green",
                padding=(1, 2),
                expand=False,
            )
        )

    def render_prefs_fields(self, fields: dict[str, dict[str, str]]) -> None:
        """
        Renderiza la tabla de campos configurables disponibles.

        Usada por `hiperforge config fields`.

        Parámetros:
            fields: Dict de campo → {"type": tipo, "description": descripción}
        """
        table = Table(
            show_header=True,
            header_style="bold cyan",
            border_style="dim",
            expand=False,
        )

        table.add_column("Campo", style="bold", min_width=30)
        table.add_column("Tipo", width=8, style="dim")
        table.add_column("Descripción", min_width=40)

        # Agrupamos por sección (llm.*, agent.*, ui.*)
        sections: dict[str, list[tuple[str, dict]]] = {}
        for field_path, info in sorted(fields.items()):
            section = field_path.split(".")[0]
            sections.setdefault(section, []).append((field_path, info))

        for section, section_fields in sections.items():
            # Separador de sección
            table.add_row(
                Text(f"── {section.upper()} ──", style="bold dim"),
                "", "",
            )
            for field_path, info in section_fields:
                table.add_row(
                    field_path,
                    info.get("type", ""),
                    info.get("description", ""),
                )

        self._console.print(table)
        self._console.print(
            "\n[dim]Uso: [bold]hiperforge config set <campo> <valor>[/bold][/dim]\n"
        )

    # ──────────────────────────────────────────────────────────────────
    # MENSAJES UTILITARIOS
    # ──────────────────────────────────────────────────────────────────

    def render_success(self, message: str) -> None:
        """Renderiza un mensaje de éxito genérico con ícono verde."""
        self._console.print(f"[green]✓[/green] {message}")

    def render_warning(self, message: str) -> None:
        """Renderiza un mensaje de advertencia con ícono amarillo."""
        self._console.print(f"[yellow]⚠[/yellow] {message}")

    def render_info(self, message: str) -> None:
        """Renderiza un mensaje informativo con formato neutro."""
        self._console.print(f"[cyan]●[/cyan] {message}")

    def render_id(self, entity_type: str, id_str: str) -> None:
        """
        Renderiza un ID de entidad de forma copiable.

        Muestra el ID completo para que el usuario pueda copiarlo
        para usarlo en comandos posteriores.

        Parámetros:
            entity_type: Nombre de la entidad (ej: "Task", "Workspace").
            id_str:      El ID completo a mostrar.
        """
        self._console.print(
            f"[dim]{entity_type} ID:[/dim] [bold cyan]{id_str}[/bold cyan]"
        )

    def render_divider(self, label: str = "") -> None:
        """Renderiza un separador horizontal con etiqueta opcional."""
        if label:
            self._console.rule(f"[dim]{label}[/dim]", style="dim")
        else:
            self._console.rule(style="dim")

    # ──────────────────────────────────────────────────────────────────
    # HELPERS PRIVADOS
    # ──────────────────────────────────────────────────────────────────

    def _build_execution_metrics(self, output: RunTaskOutput) -> str:
        """
        Construye la línea de métricas de ejecución de una task.

        Formato:
          4/4 subtasks  ·  1,234 tokens  ·  $0.0023  ·  12.3s

        Parámetros:
            output: RunTaskOutput con las métricas a mostrar.
        """
        parts: list[str] = []

        # Subtasks completadas
        if output.subtasks_total > 0:
            parts.append(
                f"{output.subtasks_completed}/{output.subtasks_total} subtasks"
            )

        # Tokens consumidos
        if output.total_tokens > 0:
            parts.append(f"{output.total_tokens:,} tokens")

        # Costo estimado
        if output.estimated_cost_usd > 0:
            parts.append(f"${output.estimated_cost_usd:.4f}")

        # Duración
        parts.append(f"{output.duration_seconds:.1f}s")

        return "[dim]" + "  ·  ".join(parts) + "[/dim]"

    def _build_static_progress_bar(
        self,
        completed: int,
        total: int,
        color: str = "green",
        width: int = 30,
    ) -> str:
        """
        Construye una barra de progreso estática como string Rich.

        Parámetros:
            completed: Número de elementos completados.
            total:     Total de elementos.
            color:     Color de la barra (Rich color name).
            width:     Ancho en caracteres de la barra.

        Returns:
            String formateado con Rich markup.
        """
        if total == 0:
            return "[dim]Sin subtasks[/dim]"

        pct = completed / total
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)

        pct_str = f"{pct * 100:.0f}%"

        return (
            f"[{color}]{bar}[/{color}] "
            f"[bold]{completed}/{total}[/bold] "
            f"[dim]({pct_str})[/dim]"
        )

    def _build_compact_progress(self, completed: int, total: int) -> str:
        """
        Construye una representación compacta del progreso para tablas.

        Formato: ████░░░░ 3/8

        Parámetros:
            completed: Elementos completados.
            total:     Total de elementos.
        """
        if total == 0:
            return "—"

        width = 8
        pct = completed / total
        filled = int(pct * width)
        bar = "█" * filled + "░" * (width - filled)

        return f"{bar} {completed}/{total}"

    @staticmethod
    def _format_datetime(dt: datetime | None) -> str:
        """
        Formatea un datetime para mostrar en la CLI.

        Si el datetime es None o no tiene timezone, lo maneja gracefully.
        Usa el formato DD/MM/YYYY HH:MM en hora local del sistema.

        Parámetros:
            dt: Datetime a formatear. None devuelve "—".
        """
        if dt is None:
            return "—"

        try:
            # Convertir a hora local si tiene timezone
            if dt.tzinfo is not None:
                local_dt = dt.astimezone()
            else:
                local_dt = dt

            return local_dt.strftime(_DATE_FORMAT)
        except (ValueError, OSError):
            return str(dt)[:16]