"""
AgentSpinner — Indicador de progreso del agente durante la ejecución.

El spinner se conecta al EventBus y reacciona a los eventos del agente
para mostrar al usuario qué está ocurriendo en tiempo real.

DISEÑO:
  El spinner es un listener del EventBus — no tiene estado propio
  sobre la ejecución. El agente emite eventos y el spinner los traduce
  a mensajes visuales en la terminal.

  Esta separación es fundamental: el executor no sabe que existe
  una terminal. Solo emite eventos. El spinner los interpreta.

EVENTOS QUE ESCUCHA:
  TASK_STARTED       → "Iniciando task..."
  TASK_PLANNING      → "Generando plan..."
  SUBTASK_STARTED    → "Ejecutando: <descripción>"
  REACT_ITERATION_*  → actualiza el contador de iteraciones
  TOOL_CALLED        → "→ <tool_name>(<args_preview>)"
  TOOL_RESULT_*      → "✓ <tool_name>" o "✗ <tool_name>"
  TASK_COMPLETED     → cierra el spinner con éxito
  TASK_FAILED        → cierra el spinner con error

THREAD SAFETY:
  El EventBus es síncrono y el loop ReAct es single-thread.
  Los listeners se ejecutan en el mismo thread que el executor.
  El spinner no necesita locks ni concurrencia.

USO:
  spinner = AgentSpinner()
  with spinner.attach(get_event_bus()):
      container.run_task.execute(input_data)
  # Al salir del context manager, el spinner se desconecta del bus
"""

from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Generator

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.spinner import Spinner
from rich.text import Text

from hiperforge.core.events import AgentEvent, EventBus, EventType, get_event_bus

# Consola Rich compartida — una sola instancia por proceso
console = Console(stderr=False)


class AgentSpinner:
    """
    Indicador visual de progreso del agente.

    Se conecta al EventBus y actualiza la vista en tiempo real
    a medida que el agente ejecuta subtasks y tool calls.

    Parámetros:
        console: Instancia de Rich Console a usar.
                 Default: consola global del módulo.
    """

    def __init__(self, rich_console: Console | None = None) -> None:
        self._console = rich_console or console

        # Estado interno del spinner — actualizado por los listeners
        self._phase: str = "Iniciando..."
        self._subtask_desc: str = ""
        self._subtask_order: int = 0
        self._subtask_total: int = 0
        self._iteration: int = 0
        self._last_tool: str = ""
        self._last_tool_success: bool = True
        self._task_id: str = ""

        # Live display de Rich — actualizado en cada evento
        self._live: Live | None = None

        # Listeners registrados — guardamos referencias para poder desregistrarlos
        self._registered_listeners: list[tuple[EventType, callable]] = []

    # ------------------------------------------------------------------
    # Context manager — attach/detach al EventBus
    # ------------------------------------------------------------------

    @contextmanager
    def attach(self, bus: EventBus | None = None) -> Generator[AgentSpinner, None, None]:
        """
        Conecta el spinner al EventBus y muestra el indicador de progreso.

        Uso:
            spinner = AgentSpinner()
            with spinner.attach():
                container.run_task.execute(input_data)

        Al entrar al context manager:
          - Registra los listeners en el EventBus.
          - Inicia el Live display de Rich.

        Al salir:
          - Desregistra los listeners.
          - Detiene el Live display.
        """
        effective_bus = bus or get_event_bus()

        self._register_listeners(effective_bus)

        with Live(
            self._render(),
            console=self._console,
            refresh_per_second=10,
            transient=False,
        ) as live:
            self._live = live
            try:
                yield self
            finally:
                self._live = None
                self._deregister_listeners(effective_bus)

    # ------------------------------------------------------------------
    # Registro de listeners
    # ------------------------------------------------------------------

    def _register_listeners(self, bus: EventBus) -> None:
        """Registra todos los listeners del spinner en el EventBus."""
        listeners = [
            (EventType.TASK_STARTED,            self._on_task_started),
            (EventType.TASK_PLANNING,            self._on_task_planning),
            (EventType.TASK_EXECUTING,           self._on_task_executing),
            (EventType.TASK_COMPLETED,           self._on_task_completed),
            (EventType.TASK_FAILED,              self._on_task_failed),
            (EventType.TASK_CANCELLED,           self._on_task_cancelled),
            (EventType.SUBTASK_STARTED,          self._on_subtask_started),
            (EventType.SUBTASK_COMPLETED,        self._on_subtask_completed),
            (EventType.SUBTASK_FAILED,           self._on_subtask_failed),
            (EventType.REACT_ITERATION_STARTED,  self._on_iteration_started),
            (EventType.TOOL_CALLED,              self._on_tool_called),
            (EventType.TOOL_RESULT_RECEIVED,     self._on_tool_result),
        ]

        for event_type, listener in listeners:
            bus.subscribe(event_type, listener)
            self._registered_listeners.append((event_type, listener))

    def _deregister_listeners(self, bus: EventBus) -> None:
        """Desregistra todos los listeners del spinner del EventBus."""
        for event_type, listener in self._registered_listeners:
            bus.unsubscribe(event_type, listener)
        self._registered_listeners.clear()

    # ------------------------------------------------------------------
    # Handlers de eventos
    # ------------------------------------------------------------------

    def _on_task_started(self, event: AgentEvent) -> None:
        self._task_id = event.data.get("task_id", "")
        prompt_preview = event.data.get("prompt_preview", "")
        self._phase = f"Iniciando: {prompt_preview[:60]}"
        self._refresh()

    def _on_task_planning(self, event: AgentEvent) -> None:
        self._phase = "Generando plan de ejecución..."
        self._refresh()

    def _on_task_executing(self, event: AgentEvent) -> None:
        self._subtask_total = event.data.get("subtask_count", 0)
        self._phase = f"Ejecutando plan ({self._subtask_total} pasos)..."
        self._refresh()

    def _on_task_completed(self, event: AgentEvent) -> None:
        duration = event.data.get("duration_seconds", 0.0)
        tokens = event.data.get("total_tokens", 0)
        self._phase = (
            f"✓ Completado en {duration:.1f}s "
            f"({tokens:,} tokens)"
        )
        self._refresh()

    def _on_task_failed(self, event: AgentEvent) -> None:
        reason = event.data.get("reason", "Error desconocido")
        self._phase = f"✗ Falló: {reason[:80]}"
        self._refresh()

    def _on_task_cancelled(self, event: AgentEvent) -> None:
        self._phase = "⊘ Cancelado por el usuario"
        self._refresh()

    def _on_subtask_started(self, event: AgentEvent) -> None:
        self._subtask_desc = event.data.get("description", "")
        self._subtask_order = event.data.get("order", 0) + 1  # 1-indexed
        self._iteration = 0
        self._last_tool = ""
        self._phase = "ejecutando"
        self._refresh()

    def _on_subtask_completed(self, event: AgentEvent) -> None:
        iterations = event.data.get("react_iterations", 0)
        duration = event.data.get("duration_seconds", 0.0)
        self._console.print(
            f"  [green]✓[/green] [{self._subtask_order}/{self._subtask_total}] "
            f"{self._subtask_desc[:70]} "
            f"[dim]({iterations} iter, {duration:.1f}s)[/dim]"
        )
        self._refresh()

    def _on_subtask_failed(self, event: AgentEvent) -> None:
        reason = event.data.get("reason", "")
        self._console.print(
            f"  [red]✗[/red] [{self._subtask_order}/{self._subtask_total}] "
            f"{self._subtask_desc[:70]} "
            f"[dim]({reason[:50]})[/dim]"
        )
        self._refresh()

    def _on_iteration_started(self, event: AgentEvent) -> None:
        self._iteration = event.data.get("iteration", 0)
        self._refresh()

    def _on_tool_called(self, event: AgentEvent) -> None:
        self._last_tool = event.data.get("tool_name", "")
        args_preview = event.data.get("arguments_preview", "")
        self._phase = f"→ {self._last_tool}({args_preview[:40]})"
        self._refresh()

    def _on_tool_result(self, event: AgentEvent) -> None:
        self._last_tool_success = event.data.get("success", True)
        tool_name = event.data.get("tool_name", self._last_tool)
        duration = event.data.get("duration_seconds", 0.0)
        icon = "✓" if self._last_tool_success else "✗"
        color = "green" if self._last_tool_success else "red"
        self._phase = f"[{color}]{icon}[/{color}] {tool_name} ({duration:.2f}s)"
        self._refresh()

    # ------------------------------------------------------------------
    # Renderizado
    # ------------------------------------------------------------------

    def _render(self) -> Panel:
        """
        Construye el panel de Rich con el estado actual del spinner.

        Se llama en cada evento y también por el Live display a 10fps.
        Debe ser rápido — sin I/O ni cálculos costosos.
        """
        # Línea principal con el spinner animado
        spinner_widget = Spinner("dots", style="cyan")

        # Información de progreso de la subtask actual
        progress_info = ""
        if self._subtask_total > 0 and self._subtask_order > 0:
            progress_info = (
                f"[dim]Paso {self._subtask_order}/{self._subtask_total}[/dim] "
                f"[dim]· iteración {self._iteration}[/dim]"
            )

        # Descripción de la subtask activa
        subtask_line = ""
        if self._subtask_desc:
            subtask_line = f"\n  [bold]{self._subtask_desc[:80]}[/bold]"

        # Estado actual
        status_line = f"\n  {self._phase}"

        content = Text.assemble(
            ("HiperForge ", "bold cyan"),
            (progress_info + subtask_line + status_line),
        )

        return Panel(
            content,
            border_style="cyan",
            padding=(0, 1),
        )

    def _refresh(self) -> None:
        """Actualiza el Live display si está activo."""
        if self._live is not None:
            self._live.update(self._render())