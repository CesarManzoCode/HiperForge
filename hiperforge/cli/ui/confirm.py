"""
Confirm — Diálogos de confirmación interactivos para operaciones críticas de la CLI.

Este módulo centraliza todos los patrones de confirmación interactiva
que la CLI usa antes de ejecutar operaciones potencialmente destructivas
o de alto impacto.

════════════════════════════════════════════════════════════
¿POR QUÉ UN MÓDULO DEDICADO PARA CONFIRMACIONES?
════════════════════════════════════════════════════════════

  Las confirmaciones son más complejas de lo que parecen. Deben:

    1. IDENTIFICAR el impacto de la operación — no todas las operaciones
       son igual de peligrosas. Eliminar un workspace con proyectos
       activos es mucho más grave que renombrarlo.

    2. COMUNICAR el costo — cuántos proyectos se perderán, cuántas tasks,
       cuántos bytes de datos. El usuario necesita contexto concreto.

    3. MANEJAR non-TTY — cuando la CLI se ejecuta en CI/CD o pipelines,
       no hay terminal interactiva. El comportamiento default debe ser
       seguro (no ejecutar la operación destructiva).

    4. RESPETAR --yes / --force — el usuario avanzado puede querer
       saltarse las confirmaciones en scripts automatizados.

    5. DOBLE CONFIRMACIÓN para operaciones irreversibles — algunas
       operaciones son tan graves que merecen que el usuario las
       confirme dos veces, o que escriba el nombre de la entidad.

════════════════════════════════════════════════════════════
TIPOS DE CONFIRMACIÓN DISPONIBLES
════════════════════════════════════════════════════════════

  SIMPLE:
    confirm_action(msg)
    → Sí/No simple. Para operaciones moderadamente importantes.

  CON CONSECUENCIAS:
    confirm_destructive(msg, consequences)
    → Muestra las consecuencias antes de preguntar.
      Para operaciones que afectan datos existentes.

  CON NOMBRE (double-confirm):
    confirm_by_name(entity_type, entity_name)
    → El usuario debe escribir el nombre exacto de la entidad.
      Para operaciones completamente irreversibles (eliminar workspace).

  SWITCH DE WORKSPACE:
    confirm_workspace_switch(current, target)
    → Muestra workspace actual y destino antes de cambiar.

  RESET DE CONFIGURACIÓN:
    confirm_config_reset(workspace_name)
    → Advierte sobre la pérdida de configuración personalizada.

════════════════════════════════════════════════════════════
MANEJO DE ENTORNOS NO INTERACTIVOS
════════════════════════════════════════════════════════════

  Cuando la CLI se ejecuta en un entorno sin terminal (CI/CD, pipes,
  scripts), las confirmaciones deben fallar de forma segura.

  Estrategia:
    - Detectamos si sys.stdin es un TTY con `sys.stdin.isatty()`.
    - Si no es TTY y no se pasó --yes, devolvemos False (no ejecutar).
    - Loggeamos un warning para que el desarrollador sepa que necesita
      agregar --yes a su script si quiere ejecutar la operación.

  Excepción: operaciones de solo lectura nunca necesitan confirmación.

USO EN COMANDOS:
  from hiperforge.cli.ui.confirm import Confirm as HFConfirm

  # Confirmación simple
  if not HFConfirm.action("¿Archivar este workspace?"):
      raise typer.Abort()

  # Confirmación destructiva con consecuencias
  if not HFConfirm.destructive(
      "¿Eliminar el workspace?",
      consequences=["5 proyectos se eliminarán", "32 tasks se perderán"],
  ):
      raise typer.Abort()

  # Double-confirm por nombre
  if not HFConfirm.by_name("workspace", workspace.name):
      raise typer.Abort()
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any

from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from rich.text import Text

from hiperforge.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Consola dedicada para prompts interactivos — siempre a stderr
# ---------------------------------------------------------------------------
# Usamos stderr para los prompts de confirmación porque stdout puede estar
# siendo redirigido por un script o pipeline. Los prompts deben verse
# en la terminal del usuario, no en el archivo de salida del script.
_prompt_console = Console(stderr=True, highlight=False)


@dataclass(frozen=True)
class ConfirmResult:
    """
    Resultado de una confirmación interactiva.

    Encapsula la decisión del usuario junto con el contexto de cómo
    se tomó esa decisión — útil para logging y auditoría.

    Atributos:
        confirmed:    True si el usuario confirmó la operación.
        was_forced:   True si se confirmó automáticamente por --yes/--force.
        was_non_tty:  True si no había terminal interactiva disponible.
        input_value:  El texto que escribió el usuario (en double-confirm).
    """
    confirmed: bool
    was_forced: bool = False
    was_non_tty: bool = False
    input_value: str = ""


class Confirm:
    """
    Diálogos de confirmación interactivos para la CLI de HiperForge.

    Todos los métodos son estáticos — no hay estado interno.
    La clase actúa como namespace para organizar los tipos de confirmación.

    CONVENCIÓN DE NOMBRES:
      Todos los métodos devuelven bool para facilitar el patrón:
        if not Confirm.action("¿Continuar?"):
            raise typer.Abort()
    """

    # ------------------------------------------------------------------
    # Confirmación simple — Sí/No
    # ------------------------------------------------------------------

    @staticmethod
    def action(
        message: str,
        *,
        default: bool = False,
        force: bool = False,
        console: Console | None = None,
    ) -> bool:
        """
        Muestra una confirmación simple de Sí/No al usuario.

        El valor por defecto es False (no ejecutar) para que las
        operaciones importantes requieran una confirmación explícita.

        Parámetros:
            message: Descripción clara de la operación a confirmar.
                     Ejemplo: "¿Archivar el workspace 'Trabajo'?"
            default: Valor por defecto si el usuario presiona Enter sin escribir.
                     False = presionar Enter sin escribir = no ejecutar (más seguro).
            force:   Si True, confirma automáticamente sin mostrar el prompt.
                     Usado cuando el usuario pasa --yes en la CLI.
            console: Consola Rich a usar. None = usar la consola global de prompts.

        Returns:
            True si el usuario confirmó.
            False si no confirmó, canceló, o no hay terminal interactiva.
        """
        con = console or _prompt_console

        result = Confirm._get_confirmation(
            message=message,
            default=default,
            force=force,
            console=con,
        )

        if result.was_non_tty and not force:
            logger.warning(
                "confirmación requerida pero no hay TTY — operación cancelada",
                message=message[:100],
                hint="Agrega --yes para ejecutar en entornos no interactivos.",
            )

        return result.confirmed

    # ------------------------------------------------------------------
    # Confirmación destructiva — con lista de consecuencias
    # ------------------------------------------------------------------

    @staticmethod
    def destructive(
        message: str,
        consequences: list[str],
        *,
        force: bool = False,
        console: Console | None = None,
    ) -> bool:
        """
        Muestra las consecuencias de la operación antes de pedir confirmación.

        Diseñado para operaciones que eliminan o modifican datos existentes
        de forma que puede ser difícil revertir.

        PATRÓN VISUAL:
          ┌─────────────────────────────────────────────┐
          │ ⚠ Esta operación no se puede deshacer       │
          │                                              │
          │ Consecuencias:                               │
          │   • Se eliminarán 5 proyectos               │
          │   • Se perderán 32 tasks y su historial     │
          │   • Las preferencias del workspace se       │
          │     eliminarán permanentemente              │
          └─────────────────────────────────────────────┘
          ¿Continuar? [s/N]

        Parámetros:
            message:      Descripción de la operación.
            consequences: Lista de consecuencias concretas a mostrar.
                          Deben ser específicas: "5 proyectos" no "proyectos".
            force:        Si True, confirma automáticamente.
            console:      Consola Rich a usar.

        Returns:
            True solo si el usuario confirmó explícitamente.
        """
        con = console or _prompt_console

        if force:
            logger.warning(
                "operación destructiva confirmada automáticamente por --force",
                message=message[:100],
                consequences=consequences,
            )
            return True

        # Mostrar el panel de consecuencias
        cons_lines = ["[bold red]⚠ Esta operación no se puede deshacer[/bold red]", ""]
        cons_lines.append("[dim]Consecuencias:[/dim]")

        for consequence in consequences:
            cons_lines.append(f"  [red]•[/red] {consequence}")

        con.print(
            Panel(
                "\n".join(cons_lines),
                title=f"[bold red]{message}[/bold red]",
                border_style="red",
                padding=(1, 2),
                expand=False,
            )
        )

        result = Confirm._get_confirmation(
            message="¿Confirmar esta operación?",
            default=False,  # Siempre False para operaciones destructivas
            force=False,    # force ya fue manejado arriba
            console=con,
        )

        if result.was_non_tty:
            logger.warning(
                "confirmación destructiva requerida pero no hay TTY — cancelada",
                message=message[:100],
            )

        return result.confirmed

    # ------------------------------------------------------------------
    # Double-confirm por nombre de entidad — para operaciones irreversibles
    # ------------------------------------------------------------------

    @staticmethod
    def by_name(
        entity_type: str,
        entity_name: str,
        *,
        force: bool = False,
        console: Console | None = None,
    ) -> bool:
        """
        Requiere que el usuario escriba el nombre exacto de la entidad.

        El patrón más estricto de confirmación. Diseñado para operaciones
        completamente irreversibles donde la pérdida de datos es total
        y permanente.

        Inspirado en el patrón de GitHub para eliminar repositorios:
        el usuario debe escribir el nombre exacto del repositorio.

        PATRÓN VISUAL:
          Para confirmar, escribe el nombre del workspace:
          "mi-workspace-de-trabajo"
          > _

        Si el usuario escribe algo diferente al nombre exacto,
        la operación se cancela — sin segunda oportunidad.

        Parámetros:
            entity_type: Tipo de entidad para el mensaje.
                         Ejemplo: "workspace", "proyecto".
            entity_name: Nombre exacto que el usuario debe escribir.
            force:       Si True, confirma automáticamente sin prompt.
            console:     Consola Rich a usar.

        Returns:
            True solo si el usuario escribió el nombre exacto.
        """
        con = console or _prompt_console

        if force:
            logger.warning(
                "double-confirm por nombre omitido por --force",
                entity_type=entity_type,
                entity_name=entity_name,
            )
            return True

        # Verificar TTY antes de mostrar el prompt
        if not sys.stdin.isatty():
            logger.warning(
                "double-confirm requerida pero no hay TTY — cancelada",
                entity_type=entity_type,
                entity_name=entity_name,
            )
            con.print(
                f"[red]✗[/red] Se requiere confirmación interactiva para eliminar "
                f"'{entity_name}'. Usa --force para omitir (operación destructiva)."
            )
            return False

        # Instrucciones claras al usuario
        con.print(
            f"\n[dim]Para confirmar la eliminación del {entity_type}, "
            f"escribe su nombre exacto:[/dim]"
        )
        con.print(f'  [bold dim]"{entity_name}"[/bold dim]\n')

        try:
            user_input = Prompt.ask(
                "Nombre",
                console=con,
                default="",
                show_default=False,
            ).strip()

            if user_input == entity_name:
                return True
            else:
                # Nombre incorrecto — cancelamos inmediatamente
                # No damos otra oportunidad para evitar que el usuario
                # confirme por desesperación en lugar de conscientemente
                con.print(
                    f"\n[yellow]⊘[/yellow] Nombre incorrecto. "
                    f"Operación cancelada para proteger tus datos."
                )
                return False

        except (EOFError, KeyboardInterrupt):
            con.print()
            return False

    # ------------------------------------------------------------------
    # Confirmación de cambio de workspace
    # ------------------------------------------------------------------

    @staticmethod
    def workspace_switch(
        current_name: str | None,
        target_name: str,
        *,
        force: bool = False,
        console: Console | None = None,
    ) -> bool:
        """
        Muestra el workspace actual y el destino antes de cambiar.

        Permite al usuario verificar que está cambiando al workspace
        correcto antes de que todas las operaciones futuras operen
        en el nuevo contexto.

        Parámetros:
            current_name: Nombre del workspace activo actual. None si no hay ninguno.
            target_name:  Nombre del workspace destino.
            force:        Si True, confirma automáticamente.
            console:      Consola Rich a usar.

        Returns:
            True si el usuario confirma el cambio.
        """
        con = console or _prompt_console

        if force:
            return True

        if current_name is None:
            # Sin workspace activo — cambio seguro, no necesita doble confirmación
            return True

        if current_name == target_name:
            # Ya es el workspace activo — confirmar es innecesario
            con.print(
                f"[yellow]●[/yellow] El workspace '[bold]{target_name}[/bold]' "
                f"ya es el activo."
            )
            return False

        # Mostrar el contexto del cambio
        con.print(
            f"\n  [dim]Activo actual:[/dim] [bold]{current_name}[/bold]\n"
            f"  [dim]Nuevo activo: [/dim] [bold cyan]{target_name}[/bold cyan]\n"
        )

        return Confirm.action(
            f"¿Cambiar al workspace '{target_name}'?",
            default=True,  # Default True — el usuario ya eligió el workspace
            force=force,
            console=con,
        )

    # ------------------------------------------------------------------
    # Confirmación de reset de configuración
    # ------------------------------------------------------------------

    @staticmethod
    def config_reset(
        workspace_name: str | None = None,
        *,
        force: bool = False,
        console: Console | None = None,
    ) -> bool:
        """
        Advierte sobre la pérdida de configuración personalizada al resetear.

        Parámetros:
            workspace_name: Si se especifica, el reset es del workspace.
                            Si None, el reset es de la configuración global.
            force:          Si True, confirma automáticamente.
            console:        Consola Rich a usar.

        Returns:
            True si el usuario confirma el reset.
        """
        if workspace_name:
            scope = f"del workspace '[bold]{workspace_name}[/bold]'"
        else:
            scope = "global"

        consequences = [
            f"Toda la configuración {scope} se reemplazará por los valores por defecto.",
            "Las API keys y modelos configurados se perderán.",
            "Esta acción no se puede deshacer.",
        ]

        return Confirm.destructive(
            message=f"¿Resetear la configuración {scope}?",
            consequences=consequences,
            force=force,
            console=console,
        )

    # ------------------------------------------------------------------
    # Helper interno — lógica central de prompts
    # ------------------------------------------------------------------

    @staticmethod
    def _get_confirmation(
        message: str,
        default: bool,
        force: bool,
        console: Console,
    ) -> ConfirmResult:
        """
        Implementación central de la lógica de confirmación.

        Maneja tres escenarios:
          1. force=True  → confirmación automática sin prompt.
          2. Non-TTY     → cancelación automática (entorno no interactivo).
          3. TTY normal  → muestra el prompt y espera respuesta del usuario.

        FORMATO DEL PROMPT:
          El prompt muestra la opción por defecto en mayúsculas:
            default=True  → [S/n]
            default=False → [s/N]

          Esto sigue la convención estándar de Unix.

        Parámetros:
            message: Mensaje del prompt.
            default: Valor si el usuario presiona Enter.
            force:   Si True, confirma sin prompt.
            console: Consola Rich para mostrar el prompt.

        Returns:
            ConfirmResult con la decisión y cómo se tomó.
        """
        # Escenario 1: confirmación automática por --yes/--force
        if force:
            logger.debug("confirmación automática por --force", message=message[:80])
            return ConfirmResult(confirmed=True, was_forced=True)

        # Escenario 2: entorno no interactivo — cancelar por seguridad
        if not sys.stdin.isatty():
            return ConfirmResult(confirmed=False, was_non_tty=True)

        # Escenario 3: prompt interactivo normal
        # Opciones s/n con la opción por defecto en mayúsculas
        if default:
            options_display = "[[bold green]S[/bold green]/n]"
            choices = ["s", "n", "si", "sí", "no", "y", "yes"]
            default_choice = "s"
        else:
            options_display = "[s/[bold red]N[/bold red]]"
            choices = ["s", "n", "si", "sí", "no", "y", "yes"]
            default_choice = "n"

        console.print(f"\n{message} {options_display}")

        try:
            answer = Prompt.ask(
                "",
                choices=choices,
                default=default_choice,
                show_choices=False,
                show_default=False,
                console=console,
            ).lower().strip()

            confirmed = answer in {"s", "si", "sí", "y", "yes"}
            return ConfirmResult(confirmed=confirmed, input_value=answer)

        except (EOFError, KeyboardInterrupt):
            # Ctrl+C durante el prompt o pipe cerrado
            console.print()
            return ConfirmResult(confirmed=False, was_non_tty=True)