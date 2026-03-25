"""
PlanView — Vista interactiva del plan de ejecución generado por el agente.

Es la pantalla más importante de la interacción humano-agente.
Aquí el usuario decide si el agente entendió correctamente la instrucción
antes de comprometer recursos (tokens, tiempo, cambios en el código).

════════════════════════════════════════════════════════════
¿POR QUÉ ESTA VISTA ES CRÍTICA?
════════════════════════════════════════════════════════════

  Un plan mal entendido desperdicia tokens, tiempo y potencialmente
  daña el código del proyecto. La vista del plan es la última
  oportunidad del usuario de corregir el rumbo antes de ejecutar.

  Casos donde el plan puede ser incorrecto:
    • El prompt fue ambiguo — el agente interpretó algo diferente.
    • El agente no tiene suficiente contexto del proyecto.
    • La tarea es demasiado compleja y el plan está incompleto.
    • El modelo LLM tiene limitaciones para este tipo de tarea.

════════════════════════════════════════════════════════════
INFORMACIÓN QUE MUESTRA
════════════════════════════════════════════════════════════

  1. ENCABEZADO: contexto completo de la task
     - Instrucción original del usuario (prompt completo)
     - Workspace y proyecto donde se ejecutará
     - ID de la task para referencia futura
     - Modelo LLM que generó el plan

  2. PLAN DE EJECUCIÓN: lista numerada de subtasks
     - Descripción completa de cada paso
     - Estimación de complejidad por subtask
     - Advertencias si el plan parece incompleto o sospechoso

  3. ESTIMACIÓN DE RECURSOS
     - Número de subtasks y estimación de iteraciones
     - Estimación de tokens y costo aproximado
     - Tiempo estimado de ejecución

  4. OPCIONES DE ACCIÓN
     - [S] Sí, ejecutar el plan
     - [N] No, cancelar
     - [D] Detalles (mostrar información adicional en verbose)

════════════════════════════════════════════════════════════
INTEGRACIÓN CON EL EXECUTOR
════════════════════════════════════════════════════════════

  PlanView genera el callback on_confirm_plan que el ExecutorService
  espera. El callback recibe la Task con las subtasks ya generadas
  y devuelve True (ejecutar) o False (cancelar).

  Uso en run.py:
    plan_view = PlanView(renderer=renderer)
    output = container.run_task.execute(
        input_data,
        on_confirm_plan=plan_view.confirm_callback,
        on_subtask_limit_reached=plan_view.limit_callback,
    )
"""

from __future__ import annotations

from typing import Callable

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Confirm, Prompt
from rich.table import Table
from rich.text import Text
from rich.tree import Tree

from hiperforge.application.services.executor import LimitDecision
from hiperforge.cli.ui.renderer import Renderer, _STATUS_COLORS, _STATUS_ICONS
from hiperforge.core.constants import REACT_MAX_ITERATIONS_PER_SUBTASK
from hiperforge.domain.entities.task import Subtask, Task

# ---------------------------------------------------------------------------
# Estimaciones para el pre-flight del plan
# ---------------------------------------------------------------------------

# Estimación de tokens por subtask (muy conservador — es solo para orientar al usuario)
_EST_TOKENS_PER_SUBTASK = 2_500

# Estimación de segundos por subtask (depende de la complejidad real)
_EST_SECONDS_PER_SUBTASK = 45

# Costo estimado por 1K tokens (modelo mediano como claude-sonnet)
_EST_COST_PER_1K_TOKENS_USD = 0.003

# Número de subtasks a partir del cual emitimos una advertencia de plan extenso
_LARGE_PLAN_WARNING_THRESHOLD = 8


class PlanView:
    """
    Vista interactiva del plan de ejecución generado por el agente.

    Muestra el plan completo al usuario y solicita confirmación
    antes de que el ExecutorService comience la ejecución.

    Integra con el ExecutorService a través de callbacks:
      - confirm_callback    → on_confirm_plan del executor
      - limit_callback      → on_subtask_limit_reached del executor

    Parámetros:
        renderer:        Renderer de la CLI para output consistente.
        console:         Consola Rich para prompts interactivos.
        workspace_name:  Nombre del workspace activo (para contexto).
        project_name:    Nombre del proyecto activo (para contexto).
        model_name:      Nombre del modelo LLM que generó el plan.
        auto_confirm:    Si True, confirma automáticamente sin mostrar el plan.
                         Equivalente al flag --yes / --auto-confirm de la CLI.
    """

    def __init__(
        self,
        renderer: Renderer,
        console: Console | None = None,
        workspace_name: str = "",
        project_name: str = "",
        model_name: str = "",
        auto_confirm: bool = False,
    ) -> None:
        self._renderer = renderer
        self._console = console or Console()
        self._workspace_name = workspace_name
        self._project_name = project_name
        self._model_name = model_name
        self._auto_confirm = auto_confirm

    # ------------------------------------------------------------------
    # Callbacks para el ExecutorService
    # ------------------------------------------------------------------

    @property
    def confirm_callback(self) -> Callable[[Task], bool]:
        """
        Devuelve el callback on_confirm_plan para el ExecutorService.

        El callback recibe la Task con las subtasks ya generadas y
        devuelve True si el usuario confirma la ejecución, False si cancela.

        Uso en comandos de la CLI:
            output = container.run_task.execute(
                input_data,
                on_confirm_plan=plan_view.confirm_callback,
            )
        """
        def _callback(task: Task) -> bool:
            return self.show_and_confirm(task)
        return _callback

    @property
    def limit_callback(self) -> Callable[[Subtask, int], LimitDecision]:
        """
        Devuelve el callback on_subtask_limit_reached para el ExecutorService.

        Se llama cuando el loop ReAct de una subtask agota su límite de
        iteraciones sin completarse. Muestra el contexto al usuario y
        le pregunta cómo proceder.

        Uso en comandos de la CLI:
            output = container.run_task.execute(
                input_data,
                on_subtask_limit_reached=plan_view.limit_callback,
            )
        """
        def _callback(subtask: Subtask, iterations_used: int) -> LimitDecision:
            return self.handle_limit_reached(subtask, iterations_used)
        return _callback

    # ------------------------------------------------------------------
    # Confirmación del plan
    # ------------------------------------------------------------------

    def show_and_confirm(self, task: Task) -> bool:
        """
        Muestra el plan de ejecución completo y solicita confirmación.

        Si auto_confirm=True, devuelve True directamente sin mostrar nada.

        Parámetros:
            task: La Task con las subtasks ya generadas por el planner.

        Returns:
            True si el usuario confirma la ejecución.
            False si el usuario cancela.
        """
        if self._auto_confirm:
            return True

        if not task.subtasks:
            # Sin plan no hay nada que confirmar — esto no debería ocurrir
            # pero lo manejamos gracefully
            return True

        # Renderizar el encabezado con el contexto completo de la task
        self._render_plan_header(task)

        # Renderizar el árbol de subtasks con análisis de calidad
        self._render_subtasks_tree(list(task.subtasks))

        # Renderizar la estimación de recursos
        self._render_resource_estimate(list(task.subtasks))

        # Advertencias si el plan tiene características sospechosas
        self._render_plan_warnings(list(task.subtasks))

        # Solicitar confirmación
        return self._prompt_confirmation()

    def _render_plan_header(self, task: Task) -> None:
        """
        Renderiza el encabezado con el contexto completo de la task.

        Incluye: instrucción original, workspace/proyecto, ID de task
        y el modelo LLM que generó el plan.
        """
        # Metadatos de contexto
        meta_parts: list[str] = []

        if self._workspace_name:
            meta_parts.append(f"Workspace: [bold]{self._workspace_name}[/bold]")
        if self._project_name:
            meta_parts.append(f"Proyecto: [bold]{self._project_name}[/bold]")
        if self._model_name:
            meta_parts.append(f"Modelo: [dim]{self._model_name}[/dim]")

        meta_line = "  ·  ".join(meta_parts) if meta_parts else ""

        # Instrucción completa del usuario
        # No truncamos aquí — el usuario debe ver exactamente qué pidió
        prompt_text = Text()
        prompt_text.append("\"", style="dim")
        prompt_text.append(task.prompt, style="bold white")
        prompt_text.append("\"", style="dim")

        content_lines: list[str] = [str(prompt_text)]

        if meta_line:
            content_lines.append("")
            content_lines.append(meta_line)

        content_lines.append("")
        content_lines.append(f"[dim]Task ID: {task.id}[/dim]")

        self._console.print(
            Panel(
                "\n".join(content_lines),
                title="[bold cyan]Plan de ejecución generado por el agente[/bold cyan]",
                border_style="cyan",
                padding=(1, 2),
            )
        )

    def _render_subtasks_tree(self, subtasks: list[Subtask]) -> None:
        """
        Renderiza las subtasks en formato de árbol numerado.

        Cada subtask muestra su descripción completa y un indicador
        visual de complejidad estimada basado en la longitud y palabras clave
        de la descripción.

        Parámetros:
            subtasks: Lista de Subtask en orden de ejecución.
        """
        tree = Tree(
            f"[bold]{len(subtasks)} paso(s) a ejecutar:[/bold]",
            guide_style="dim cyan",
        )

        for subtask in sorted(subtasks, key=lambda s: s.order):
            step_num = f"[dim cyan]{subtask.order + 1:02d}.[/dim cyan]"
            complexity = self._estimate_subtask_complexity(subtask.description)
            complexity_indicator = self._complexity_indicator(complexity)

            # Nodo del árbol con el número, descripción y complejidad
            node_text = Text()
            node_text.append(f"{subtask.order + 1:02d}. ", style="dim cyan")
            node_text.append(subtask.description)
            node_text.append(f"  {complexity_indicator}", style="dim")

            tree.add(node_text)

        self._console.print(tree)
        self._console.print()

    def _render_resource_estimate(self, subtasks: list[Subtask]) -> None:
        """
        Renderiza una estimación de los recursos que consumirá la ejecución.

        Las estimaciones son conservadoras e indicativas — no son exactas.
        Su propósito es que el usuario tenga una idea del orden de magnitud
        antes de ejecutar.

        FÓRMULAS:
          tokens estimados = num_subtasks × _EST_TOKENS_PER_SUBTASK
          costo estimado   = (tokens / 1000) × _EST_COST_PER_1K_TOKENS_USD
          tiempo estimado  = num_subtasks × _EST_SECONDS_PER_SUBTASK

        Parámetros:
            subtasks: Lista de subtasks del plan.
        """
        num_subtasks = len(subtasks)
        est_tokens = num_subtasks * _EST_TOKENS_PER_SUBTASK
        est_cost = (est_tokens / 1000) * _EST_COST_PER_1K_TOKENS_USD
        est_seconds = num_subtasks * _EST_SECONDS_PER_SUBTASK

        # Formatear tiempo estimado de forma legible
        if est_seconds < 60:
            time_str = f"~{est_seconds}s"
        elif est_seconds < 3600:
            time_str = f"~{est_seconds // 60}m"
        else:
            time_str = f"~{est_seconds // 3600}h {(est_seconds % 3600) // 60}m"

        table = Table(
            show_header=False,
            box=None,
            padding=(0, 2),
            show_edge=False,
        )
        table.add_column("Métrica", style="dim", width=22)
        table.add_column("Valor", style="bold")
        table.add_column("Nota", style="dim")

        table.add_row(
            "Pasos del plan",
            str(num_subtasks),
            f"máx {REACT_MAX_ITERATIONS_PER_SUBTASK} iteraciones ReAct c/u",
        )
        table.add_row(
            "Tokens estimados",
            f"~{est_tokens:,}",
            "estimación conservadora",
        )
        table.add_row(
            "Costo estimado",
            f"~${est_cost:.3f} USD",
            "varía según el modelo",
        )
        table.add_row(
            "Tiempo estimado",
            time_str,
            "depende de las tools usadas",
        )

        self._console.print(
            Panel(
                table,
                title="[dim]Estimación de recursos[/dim]",
                border_style="dim",
                padding=(0, 1),
                expand=False,
            )
        )

    def _render_plan_warnings(self, subtasks: list[Subtask]) -> None:
        """
        Emite advertencias si el plan tiene características que merecen atención.

        Las advertencias no bloquean la ejecución — solo informan al usuario
        de situaciones que pueden indicar que el plan no es óptimo.

        ADVERTENCIAS POSIBLES:
          - Plan extenso (>8 subtasks): puede ser innecesariamente complejo.
          - Subtask con descripción muy corta (<15 chars): puede ser vaga.
          - Subtasks con descripciones muy similares: posible duplicación.
          - Primera subtask involucra eliminación: operación de alto riesgo.

        Parámetros:
            subtasks: Lista de subtasks del plan a analizar.
        """
        warnings: list[str] = []

        # Plan demasiado extenso
        if len(subtasks) >= _LARGE_PLAN_WARNING_THRESHOLD:
            warnings.append(
                f"El plan tiene {len(subtasks)} pasos. Los planes muy largos pueden "
                f"indicar una instrucción demasiado compleja. Considera dividirla en "
                f"tareas más pequeñas para mejor control."
            )

        # Subtasks con descripciones sospechosamente cortas
        short_subtasks = [
            s for s in subtasks
            if len(s.description.strip()) < 15
        ]
        if short_subtasks:
            steps = ", ".join(str(s.order + 1) for s in short_subtasks)
            warnings.append(
                f"Los pasos {steps} tienen descripciones muy cortas. "
                f"Pueden ser demasiado vagos para que el agente los ejecute correctamente."
            )

        # Primera subtask involucra eliminación — alto riesgo
        if subtasks:
            first_desc = subtasks[0].description.lower()
            destructive_keywords = {
                "eliminar", "borrar", "delete", "remove", "drop",
                "rm ", "truncate", "reset", "limpiar",
            }
            if any(kw in first_desc for kw in destructive_keywords):
                warnings.append(
                    "⚠ El primer paso parece involucrar una operación de eliminación. "
                    "Asegúrate de tener un backup antes de continuar."
                )

        if warnings:
            self._console.print()
            for warning in warnings:
                self._console.print(f"  [yellow]⚠[/yellow] [dim]{warning}[/dim]")
            self._console.print()

    def _prompt_confirmation(self) -> bool:
        """
        Muestra las opciones disponibles y solicita la decisión del usuario.

        Las opciones son:
          [S] Sí     → ejecutar el plan tal como está
          [N] No     → cancelar sin ejecutar nada
          [D] Detalle → mostrar información adicional (equivale a --verbose)

        Returns:
            True si el usuario confirma la ejecución.
            False si cancela.
        """
        self._console.print(
            "[dim]¿Ejecutar este plan?[/dim]  "
            "[[bold green]S[/bold green]]í  "
            "[[bold red]N[/bold red]]o"
        )
        self._console.print("[dim]Escribe S o N y presiona Enter.[/dim]")

        try:
            answer = Prompt.ask(
                "Respuesta",
                choices=["s", "n", "si", "sí", "no", "y", "yes"],
                default="s",
                show_choices=False,
                show_default=False,
                console=self._console,
            ).lower().strip()

            return answer in {"s", "si", "sí", "y", "yes"}

        except (EOFError, KeyboardInterrupt):
            # Sin terminal interactiva o Ctrl+C durante el prompt
            # Optamos por no ejecutar — más seguro que ejecutar sin confirmación
            self._console.print()
            return False

    # ------------------------------------------------------------------
    # Manejo del límite de iteraciones ReAct
    # ------------------------------------------------------------------

    def handle_limit_reached(
        self,
        subtask: Subtask,
        iterations_used: int,
    ) -> LimitDecision:
        """
        Muestra el contexto de la subtask atascada y solicita una decisión.

        Se llama cuando el loop ReAct de una subtask agotó su límite de
        iteraciones sin completarse. El usuario decide cómo proceder.

        OPCIONES:
          [R] Reintentar  → el executor intenta la subtask desde cero.
          [O] Omitir      → salta esta subtask y continúa con la siguiente.
          [C] Cancelar    → detiene toda la ejecución de la task.

        Parámetros:
            subtask:         La subtask que no se completó.
            iterations_used: Cuántas iteraciones se usaron antes del límite.

        Returns:
            LimitDecision que el executor usará para decidir qué hacer.
        """
        # Mostrar contexto de la subtask atascada
        content_lines = [
            f"[bold]Paso {subtask.order + 1}:[/bold] {subtask.description}",
            "",
            f"[dim]El agente usó {iterations_used} iteraciones sin completar este paso.[/dim]",
        ]

        # Mostrar el último razonamiento del agente si existe
        if subtask.reasoning:
            last_reasoning = subtask.reasoning.split("\n")[0][:150]
            content_lines.append("")
            content_lines.append(
                f"[dim]Último razonamiento del agente:[/dim]\n"
                f"[dim italic]  \"{last_reasoning}\"[/dim italic]"
            )

        # Mostrar las últimas tool calls intentadas si existen
        if subtask.tool_calls:
            last_calls = list(subtask.tool_calls)[-3:]  # últimas 3
            content_lines.append("")
            content_lines.append("[dim]Últimas acciones del agente:[/dim]")
            for tc in last_calls:
                status_icon = "✓" if tc.status.value == "completed" else "✗"
                content_lines.append(
                    f"  [dim]{status_icon} {tc.tool_name}[/dim]"
                )

        self._console.print(
            Panel(
                "\n".join(content_lines),
                title="[bold yellow]⚠ Subtask no completada[/bold yellow]",
                border_style="yellow",
                padding=(1, 2),
            )
        )

        # Mostrar opciones disponibles
        self._console.print(
            "\n[dim]¿Cómo proceder?[/dim]  "
            "[[bold cyan]R[/bold cyan]]eintentar  "
            "[[bold yellow]O[/bold yellow]]mitir este paso  "
            "[[bold red]C[/bold red]]ancelar todo"
        )

        try:
            answer = Prompt.ask(
                "",
                choices=["r", "o", "c", "reintentar", "omitir", "cancelar"],
                default="r",
                show_choices=False,
                show_default=False,
                console=self._console,
            ).lower().strip()

            if answer in {"r", "reintentar"}:
                self._console.print(
                    "[cyan]↺[/cyan] Reintentando la subtask..."
                )
                return LimitDecision.RETRY

            elif answer in {"o", "omitir"}:
                self._console.print(
                    f"[yellow]→[/yellow] Omitiendo paso {subtask.order + 1} y continuando..."
                )
                return LimitDecision.SKIP

            else:
                self._console.print(
                    "[red]⊘[/red] Cancelando la ejecución..."
                )
                return LimitDecision.CANCEL

        except (EOFError, KeyboardInterrupt):
            # Sin terminal interactiva o Ctrl+C — cancelamos por seguridad
            self._console.print()
            return LimitDecision.CANCEL

    # ------------------------------------------------------------------
    # Helpers privados de análisis del plan
    # ------------------------------------------------------------------

    @staticmethod
    def _estimate_subtask_complexity(description: str) -> str:
        """
        Estima la complejidad de una subtask basándose en su descripción.

        Usa heurísticas simples basadas en palabras clave y longitud:
          - "alta" si involucra múltiples sistemas o operaciones complejas.
          - "media" si es un paso concreto de desarrollo estándar.
          - "baja" si es una operación simple y directa.

        NOTA: Esta es una estimación heurística, no un análisis real.
        Su propósito es dar al usuario una orientación visual, no ser exacta.

        Parámetros:
            description: Texto de la descripción de la subtask.

        Returns:
            "alta", "media" o "baja".
        """
        desc_lower = description.lower()

        # Indicadores de alta complejidad
        high_keywords = {
            "implementar", "refactorizar", "migrar", "arquitectura",
            "sistema", "módulo", "integrate", "refactor", "implement",
            "diseñar", "crear el", "desarrollar",
        }

        # Indicadores de baja complejidad
        low_keywords = {
            "agregar", "añadir", "actualizar", "cambiar", "renombrar",
            "eliminar", "borrar", "add", "update", "delete", "rename",
            "instalar", "configurar", "verificar", "revisar",
        }

        high_count = sum(1 for kw in high_keywords if kw in desc_lower)
        low_count = sum(1 for kw in low_keywords if kw in desc_lower)

        # La longitud también es indicador de complejidad
        word_count = len(description.split())

        if high_count >= 2 or word_count > 25:
            return "alta"
        elif low_count >= 1 or word_count < 8:
            return "baja"
        else:
            return "media"

    @staticmethod
    def _complexity_indicator(complexity: str) -> str:
        """
        Devuelve el indicador visual Rich para un nivel de complejidad.

        Parámetros:
            complexity: "alta", "media" o "baja".

        Returns:
            String con markup Rich para el indicador.
        """
        indicators = {
            "alta":  "[red]●●●[/red]",
            "media": "[yellow]●●○[/yellow]",
            "baja":  "[green]●○○[/green]",
        }
        return indicators.get(complexity, "[dim]●○○[/dim]")
