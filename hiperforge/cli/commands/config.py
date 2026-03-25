"""
Comandos `config` — Gestión de la configuración y preferencias del sistema.

HiperForge tiene un sistema de configuración en dos niveles con cascada:

  NIVEL GLOBAL   (~/.hiperforge/preferences.json)
    Aplica a todos los workspaces que no tengan configuración propia.

  NIVEL WORKSPACE (~/.hiperforge/workspaces/{id}/preferences.json)
    Sobreescribe las globales para ese workspace específico.

La cascada es: defaults del código → globales → workspace → variables de entorno.

════════════════════════════════════════════════════════════
COMANDOS DISPONIBLES
════════════════════════════════════════════════════════════

  hiperforge config get
    Muestra la configuración efectiva con la cascada aplicada.
    Incluye: proveedor LLM, modelo, temperatura, timeouts, flags de UI.

  hiperforge config get --global
    Muestra solo las preferencias globales, sin combinar con workspace.

  hiperforge config get --workspace <id_o_nombre>
    Muestra solo las preferencias del workspace especificado.

  hiperforge config set <campo> <valor>
    Actualiza un campo específico de la configuración global.
    Usa notación de punto: llm.provider, agent.max_react_iterations, etc.

  hiperforge config set <campo> <valor> --workspace <id_o_nombre>
    Actualiza un campo de la configuración de un workspace específico.

  hiperforge config unset <campo>
    Elimina una sobreescritura de configuración (vuelve al default).
    Solo aplica a configuración de workspace — las globales se resetean
    a su valor por defecto.

  hiperforge config reset
    Resetea la configuración global a los valores por defecto del código.
    Pide confirmación antes de ejecutar.

  hiperforge config reset --workspace <id_o_nombre>
    Resetea las preferencias del workspace específico.

  hiperforge config fields
    Lista todos los campos configurables con su tipo y descripción.
    Referencia rápida para saber qué se puede cambiar y cómo.

════════════════════════════════════════════════════════════
CONVERSIÓN DE VALORES EN LA CLI
════════════════════════════════════════════════════════════

  Los valores en la CLI son siempre strings. El comando `set`
  los convierte automáticamente al tipo correcto según el campo:

    Campo de tipo bool:
      hiperforge config set agent.auto_confirm_plan true
      hiperforge config set agent.auto_confirm_plan false
      Acepta: true/false, 1/0, yes/no, on/off (case-insensitive)

    Campo de tipo int:
      hiperforge config set agent.max_react_iterations 20

    Campo de tipo float:
      hiperforge config set llm.temperature 0.5
      hiperforge config set agent.tool_timeout_seconds 60.0

    Campo de tipo str:
      hiperforge config set llm.provider groq
      hiperforge config set llm.model claude-sonnet-4-6

    Valor especial "null" o "none":
      hiperforge config set llm.model null
      → Limpia el valor (usa el default del proveedor)

════════════════════════════════════════════════════════════
EJEMPLOS DE USO TÍPICO
════════════════════════════════════════════════════════════

  # Cambiar al proveedor Groq globalmente
  hiperforge config set llm.provider groq

  # Configurar un modelo específico para el workspace "trabajo"
  hiperforge config set llm.model claude-sonnet-4-6 --workspace trabajo

  # Aumentar el límite de iteraciones ReAct para tareas complejas
  hiperforge config set agent.max_react_iterations 25

  # Activar confirmación automática del plan (modo experto)
  hiperforge config set agent.auto_confirm_plan true

  # Ver qué campos se pueden configurar
  hiperforge config fields

  # Ver la configuración actual del workspace "trabajo"
  hiperforge config get --workspace trabajo
"""

from __future__ import annotations

from typing import Optional

import typer

from hiperforge.application import Container, UpdatePreferencesInput
from hiperforge.cli.error_handler import ErrorHandler
from hiperforge.cli.ui import Confirm, Renderer
from hiperforge.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App Typer del grupo config
# ---------------------------------------------------------------------------

config_app = typer.Typer(
    name="config",
    help="Gestión de la configuración y preferencias del agente.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# ---------------------------------------------------------------------------
# Valores booleanos aceptados como strings en la CLI
# ---------------------------------------------------------------------------

_TRUE_VALUES:  frozenset[str] = frozenset({"true",  "1", "yes", "on",  "sí", "si"})
_FALSE_VALUES: frozenset[str] = frozenset({"false", "0", "no",  "off"})

# Valores que representan "limpiar el campo" (None/null)
_NULL_VALUES: frozenset[str] = frozenset({"null", "none", "nil", "~", ""})


# ---------------------------------------------------------------------------
# config get
# ---------------------------------------------------------------------------

@config_app.command("get")
def config_get(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help=(
            "Mostrar preferencias de este workspace específico. "
            "Sin --workspace: muestra la configuración efectiva con cascada."
        ),
        metavar="ID_O_NOMBRE",
    ),
    global_only: bool = typer.Option(
        False,
        "--global", "-g",
        help="Mostrar solo las preferencias globales, sin cascada de workspace.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar los valores por defecto del código junto a los configurados.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Muestra la configuración actual del sistema.

    Sin opciones: muestra la configuración efectiva con la cascada completa
    aplicada (globals + workspace activo).

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge config get[/cyan]
      [cyan]hiperforge config get --workspace trabajo[/cyan]
      [cyan]hiperforge config get --global[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("config get"):
        container = Container.build()

        if global_only:
            # Solo las preferencias globales, sin cascada
            prefs = container.manage_prefs.get_global()
            renderer.render_prefs(prefs, level="global")
            return

        if workspace:
            # Preferencias de un workspace específico (sin cascada)
            workspace_id = _resolve_workspace_id(container, workspace)
            workspace_name = _get_workspace_name(container, workspace_id)
            ws_prefs = container.manage_prefs.get_for_workspace(workspace_id)

            if ws_prefs is None:
                renderer.render_info(
                    f"El workspace '[bold]{workspace_name}[/bold]' no tiene "
                    f"configuración propia. Usa la configuración global."
                )
                # Mostramos las globales como referencia
                prefs = container.manage_prefs.get_global()
                renderer.render_prefs(prefs, level="global")
            else:
                renderer.render_prefs(
                    ws_prefs,
                    level="workspace",
                    workspace_name=workspace_name,
                )
            return

        # Configuración efectiva: globals + workspace activo con cascada
        active_workspace_id = container.store.get_active_workspace_id()
        prefs = container.manage_prefs.get_effective(active_workspace_id)
        workspace_name = _get_workspace_name(container, active_workspace_id) if active_workspace_id else ""

        renderer.render_prefs(
            prefs,
            level="workspace" if active_workspace_id else "global",
            workspace_name=workspace_name,
        )


# ---------------------------------------------------------------------------
# config set
# ---------------------------------------------------------------------------

@config_app.command("set")
def config_set(
    field: str = typer.Argument(
        ...,
        help="Campo a configurar en notación de punto. Ejemplo: llm.provider",
        metavar="CAMPO",
    ),
    value: str = typer.Argument(
        ...,
        help="Nuevo valor para el campo.",
        metavar="VALOR",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help=(
            "Actualizar la configuración de este workspace específico. "
            "Sin --workspace: actualiza la configuración global."
        ),
        metavar="ID_O_NOMBRE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar la configuración completa después de actualizar.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Actualiza un campo de configuración.

    El tipo del valor se detecta automáticamente según el campo:
    booleans (true/false), enteros, decimales o strings.
    Usa [bold]null[/bold] para limpiar un campo opcional.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge config set llm.provider groq[/cyan]
      [cyan]hiperforge config set llm.temperature 0.3[/cyan]
      [cyan]hiperforge config set agent.auto_confirm_plan true[/cyan]
      [cyan]hiperforge config set llm.model null[/cyan]
      [cyan]hiperforge config set llm.provider anthropic --workspace trabajo[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("config set"):
        container = Container.build()

        # Resolver workspace_id si se especificó
        workspace_id: str | None = None
        workspace_name: str = "global"

        if workspace:
            workspace_id = _resolve_workspace_id(container, workspace)
            workspace_name = _get_workspace_name(container, workspace_id)

        # Convertir el valor string al tipo correcto para el campo
        typed_value = _coerce_value(field, value)

        input_data = UpdatePreferencesInput(
            updates={field: typed_value},
            workspace_id=workspace_id,
        )

        updated_prefs = container.manage_prefs.update(input_data)

        # Confirmar la actualización
        renderer.render_prefs_updated(
            updates={field: typed_value},
            level=workspace_name,
        )

        # Si verbose, mostrar la configuración completa resultante
        if verbose:
            renderer.render_prefs(
                updated_prefs,
                level="workspace" if workspace_id else "global",
                workspace_name=workspace_name,
            )


# ---------------------------------------------------------------------------
# config unset
# ---------------------------------------------------------------------------

@config_app.command("unset")
def config_unset(
    field: str = typer.Argument(
        ...,
        help="Campo a limpiar/resetear a su valor por defecto.",
        metavar="CAMPO",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help=(
            "Limpiar este campo solo en el workspace especificado. "
            "Sin --workspace: limpia el campo en la configuración global "
            "(lo resetea al valor por defecto del código)."
        ),
        metavar="ID_O_NOMBRE",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Limpia un campo de configuración y lo vuelve al valor por defecto.

    Para campos de workspace, elimina la sobreescritura y vuelve
    a usar el valor de la configuración global.

    Para campos globales, lo resetea al valor por defecto del código.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge config unset llm.model[/cyan]
      [cyan]hiperforge config unset llm.provider --workspace trabajo[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("config unset"):
        container = Container.build()

        workspace_id: str | None = None
        workspace_name: str = "global"

        if workspace:
            workspace_id = _resolve_workspace_id(container, workspace)
            workspace_name = _get_workspace_name(container, workspace_id)

        # Para campos opcionales como llm.model, unset = poner a None
        # Para campos requeridos, unset = poner al valor por defecto
        # Usamos None para indicar "limpiar" — el use case lo maneja
        input_data = UpdatePreferencesInput(
            updates={field: None},
            workspace_id=workspace_id,
        )

        container.manage_prefs.update(input_data)

        scope_label = (
            f"del workspace '[bold]{workspace_name}[/bold]'"
            if workspace_id
            else "global"
        )
        renderer.render_success(
            f"Campo [cyan]{field}[/cyan] eliminado de la configuración {scope_label}. "
            f"Se usará el valor por defecto."
        )


# ---------------------------------------------------------------------------
# config reset
# ---------------------------------------------------------------------------

@config_app.command("reset")
def config_reset(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help=(
            "Resetear la configuración de este workspace específico. "
            "Sin --workspace: resetea la configuración global."
        ),
        metavar="ID_O_NOMBRE",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Confirmar el reset automáticamente.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Resetea la configuración a los valores por defecto del código.

    [bold red]⚠ Esta operación reemplaza toda la configuración personalizada.[/bold red]

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge config reset[/cyan]
      [cyan]hiperforge config reset --workspace trabajo[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("config reset"):
        container = Container.build()

        workspace_id: str | None = None
        workspace_name: str | None = None

        if workspace:
            workspace_id = _resolve_workspace_id(container, workspace)
            workspace_name = _get_workspace_name(container, workspace_id)

        # Confirmación destructiva antes de resetear
        if not yes:
            if not Confirm.config_reset(
                workspace_name=workspace_name,
                force=yes,
            ):
                raise typer.Abort()

        container.manage_prefs.reset(workspace_id=workspace_id)

        scope_label = (
            f"del workspace '[bold]{workspace_name}[/bold]'"
            if workspace_name
            else "global"
        )
        renderer.render_success(
            f"Configuración {scope_label} reseteada a los valores por defecto."
        )


# ---------------------------------------------------------------------------
# config fields
# ---------------------------------------------------------------------------

@config_app.command("fields")
def config_fields(
    section: Optional[str] = typer.Option(
        None,
        "--section", "-s",
        help="Filtrar por sección: llm, agent, ui.",
        metavar="SECCIÓN",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Lista todos los campos configurables con su tipo y descripción.

    Referencia rápida para saber qué campos se pueden modificar
    con [bold]config set[/bold] y cuál es su tipo esperado.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge config fields[/cyan]
      [cyan]hiperforge config fields --section llm[/cyan]
      [cyan]hiperforge config fields --section agent[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("config fields"):
        container = Container.build()

        all_fields = container.manage_prefs.list_available_fields()

        # Filtrar por sección si se especificó
        if section:
            section_lower = section.lower()
            all_fields = {
                k: v for k, v in all_fields.items()
                if k.startswith(f"{section_lower}.")
            }

            if not all_fields:
                renderer.render_warning(
                    f"No se encontraron campos para la sección '{section}'. "
                    f"Secciones disponibles: llm, agent, ui"
                )
                raise typer.Exit(0)

        renderer.render_prefs_fields(all_fields)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _resolve_workspace_id(container: Container, workspace_arg: str) -> str:
    """
    Resuelve el workspace_id desde un argumento de nombre o ID.

    Aplica la heurística estándar de la CLI:
      - 26 chars alfanuméricos → ULID, resolución directa.
      - Texto → búsqueda case-insensitive por nombre.

    Parámetros:
        container:     Container con acceso al store de workspaces.
        workspace_arg: Argumento del usuario — nombre o ID.

    Returns:
        workspace_id resuelto.

    Raises:
        EntityNotFound: Si el workspace no existe.
    """
    from hiperforge.domain.exceptions import EntityNotFound

    # Heurística ULID: 26 caracteres alfanuméricos
    if len(workspace_arg) == 26 and workspace_arg.isalnum():
        return workspace_arg

    # Búsqueda por nombre
    workspaces = container.store.workspaces.find_all()
    target_lower = workspace_arg.strip().lower()

    # Búsqueda exacta primero (case-insensitive)
    for ws in workspaces:
        if ws.name.lower() == target_lower:
            return ws.id

    # Búsqueda parcial como fallback si es única
    partial = [ws for ws in workspaces if target_lower in ws.name.lower()]
    if len(partial) == 1:
        return partial[0].id

    raise EntityNotFound(
        entity_type="Workspace",
        entity_id=workspace_arg,
    )


def _get_workspace_name(container: Container, workspace_id: str | None) -> str:
    """
    Obtiene el nombre de un workspace por su ID para mensajes de UI.

    Devuelve "global" si workspace_id es None o si no puede resolver el nombre.
    Nunca lanza excepciones — es solo para mensajes de UI.

    Parámetros:
        container:    Container con acceso al store.
        workspace_id: ID del workspace o None.

    Returns:
        Nombre del workspace o "global".
    """
    if not workspace_id:
        return "global"

    try:
        ws = container.store.workspaces.find_by_id_meta(workspace_id)
        return ws.name
    except Exception:
        return workspace_id[:16] + "..."


def _coerce_value(field: str, raw_value: str) -> object:
    """
    Convierte el valor string de la CLI al tipo Python correcto para el campo.

    La CLI siempre recibe strings. Este método detecta el tipo esperado
    para el campo y convierte el valor en consecuencia.

    CONVERSIONES APLICADAS:
      bool:  "true"/"false"/"1"/"0"/"yes"/"no"/"on"/"off" → True/False
      int:   "20" → 20
      float: "0.5" → 0.5
      str:   se mantiene como string
      null:  "null"/"none"/"nil"/"~"/"" → None (limpiar campo)

    Parámetros:
        field:     Nombre del campo en notación de punto (ej: "llm.provider").
        raw_value: Valor como string recibido desde la CLI.

    Returns:
        El valor convertido al tipo apropiado.

    Raises:
        ValueError: Si el valor no puede convertirse al tipo esperado.
    """
    # Valor null/none para limpiar campos opcionales
    if raw_value.strip().lower() in _NULL_VALUES:
        return None

    # Importamos el mapa de campos válidos del use case para conocer el tipo
    from hiperforge.application.use_cases.manage_prefs import _VALID_PREF_FIELDS

    if field not in _VALID_PREF_FIELDS:
        # Campo desconocido — devolvemos el string tal cual y dejamos que
        # el use case lo valide con su mensaje de error específico
        return raw_value

    expected_type, _ = _VALID_PREF_FIELDS[field]
    value_stripped = raw_value.strip()

    # ── Conversión a bool ──────────────────────────────────────────────
    if expected_type is bool:
        value_lower = value_stripped.lower()
        if value_lower in _TRUE_VALUES:
            return True
        if value_lower in _FALSE_VALUES:
            return False
        raise ValueError(
            f"Valor inválido para el campo booleano '{field}': '{raw_value}'\n"
            f"Valores válidos: true, false, 1, 0, yes, no"
        )

    # ── Conversión a int ───────────────────────────────────────────────
    if expected_type is int:
        try:
            # Aceptamos también floats enteros como "20.0" → 20
            float_val = float(value_stripped)
            if float_val != int(float_val):
                raise ValueError(
                    f"'{raw_value}' no es un entero válido para '{field}'."
                )
            return int(float_val)
        except (ValueError, TypeError):
            raise ValueError(
                f"Valor inválido para el campo entero '{field}': '{raw_value}'\n"
                f"Ejemplo de valor válido: 20"
            )

    # ── Conversión a float ─────────────────────────────────────────────
    if expected_type is float:
        try:
            return float(value_stripped)
        except (ValueError, TypeError):
            raise ValueError(
                f"Valor inválido para el campo decimal '{field}': '{raw_value}'\n"
                f"Ejemplo de valor válido: 0.5"
            )

    # ── String — sin conversión ────────────────────────────────────────
    return value_stripped