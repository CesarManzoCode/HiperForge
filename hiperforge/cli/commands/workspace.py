"""
Comandos `workspace` — Gestión completa del ciclo de vida de workspaces.

Los workspaces son el nivel más alto de organización en HiperForge.
Este grupo de comandos expone todas las operaciones sobre workspaces:
crear, listar, cambiar, renombrar, archivar, reactivar y eliminar.

════════════════════════════════════════════════════════════
COMANDOS DISPONIBLES
════════════════════════════════════════════════════════════

  hiperforge workspace create <nombre>
    Crea un nuevo workspace. Si es el primero, queda activo automáticamente.

  hiperforge workspace list
    Lista todos los workspaces con su estado y número de proyectos.
    El workspace activo aparece marcado con ▶.

  hiperforge workspace switch <id_o_nombre>
    Cambia el workspace activo. Acepta ID o nombre (fuzzy match incluido).

  hiperforge workspace rename <id_o_nombre> <nuevo_nombre>
    Renombra un workspace. Verifica unicidad antes de persistir.

  hiperforge workspace archive <id_o_nombre>
    Archiva un workspace. Queda en modo solo lectura.
    Sus proyectos y tasks se conservan pero no se pueden agregar nuevos.

  hiperforge workspace reactivate <id_o_nombre>
    Reactiva un workspace archivado.

  hiperforge workspace delete <id_o_nombre>
    Eliminación lógica del workspace. Requiere confirmación por nombre.
    Los datos se marcan como eliminados pero permanecen en disco.

  hiperforge workspace show [id_o_nombre]
    Muestra el detalle del workspace activo o del especificado.

════════════════════════════════════════════════════════════
RESOLUCIÓN DE WORKSPACE POR NOMBRE O ID
════════════════════════════════════════════════════════════

  Todos los comandos que reciben un argumento <id_o_nombre> aceptan:
    - El ULID completo: 01HX4K2J8QNVR0SBPZ1Y3W9D6E
    - El nombre exacto: "trabajo"
    - Un prefix del nombre (fuzzy, solo si es único): "trab"

  La resolución ocurre en _resolve_workspace() que usa
  SwitchWorkspaceUseCase._resolve_by_name() internamente.

════════════════════════════════════════════════════════════
OPERACIONES DESTRUCTIVAS Y SUS CONFIRMACIONES
════════════════════════════════════════════════════════════

  archive:
    → Confirmación simple (Sí/No) con descripción del impacto.
    → Reversible con reactivate.

  delete:
    → Muestra las consecuencias concretas (N proyectos, M tasks).
    → Double-confirm: el usuario debe escribir el nombre exacto.
    → Irreversible desde la CLI (los archivos siguen en disco).
    → Si el workspace es el activo, se desactiva y no hay activo.
"""

from __future__ import annotations

from typing import Optional

import typer

from hiperforge.application import Container, CreateWorkspaceInput, SwitchWorkspaceInput
from hiperforge.cli.error_handler import ErrorHandler
from hiperforge.cli.ui import Confirm, Renderer
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.workspace import WorkspaceStatus
from hiperforge.domain.exceptions import DuplicateEntity, EntityNotFound

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App Typer del grupo workspace
# ---------------------------------------------------------------------------

workspace_app = typer.Typer(
    name="workspace",
    help="Gestión de workspaces (contextos de trabajo aislados).",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# workspace create
# ---------------------------------------------------------------------------

@workspace_app.command("create")
def workspace_create(
    name: str = typer.Argument(
        ...,
        help="Nombre del workspace. Debe ser único globalmente.",
        metavar="NOMBRE",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "--desc", "-d",
        help="Descripción opcional del propósito del workspace.",
        metavar="DESC",
    ),
    activate: bool = typer.Option(
        False,
        "--activate", "-a",
        help="Activar el workspace inmediatamente después de crearlo.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar el ID del workspace creado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Crea un nuevo workspace.

    Si es el primer workspace del sistema, queda activo automáticamente.
    Para workspaces adicionales, usa [bold]--activate[/bold] para activarlo
    al crearlo, o [bold]workspace switch[/bold] después.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge workspace create trabajo[/cyan]
      [cyan]hiperforge workspace create "Cliente Acme" --desc "Proyectos para Acme Corp" --activate[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("workspace create"):
        container = Container.build()

        input_data = CreateWorkspaceInput(
            name=name,
            description=description,
        )

        summary = container.create_workspace.execute(
            input_data,
            set_as_active=activate,
        )

        renderer.render_workspace_created(summary)


# ---------------------------------------------------------------------------
# workspace list
# ---------------------------------------------------------------------------

@workspace_app.command("list")
def workspace_list(
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar IDs completos y fechas de creación.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Lista todos los workspaces con su estado y número de proyectos.

    El workspace activo aparece marcado con [bold green]▶[/bold green].
    Los workspaces archivados aparecen en gris.
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("workspace list"):
        container = Container.build()

        workspaces_raw = container.store.workspaces.find_all()
        active_id = container.store.get_active_workspace_id()

        # Convertir entidades de dominio a WorkspaceSummary para el renderer
        from hiperforge.application.dto import WorkspaceSummary
        summaries = [
            WorkspaceSummary(
                id=ws.id,
                name=ws.name,
                description=ws.description,
                status=ws.status.value,
                project_count=ws.project_count,
                is_active=(ws.id == active_id),
                created_at=ws.created_at,
            )
            for ws in workspaces_raw
        ]

        renderer.render_workspace_list(summaries)


# ---------------------------------------------------------------------------
# workspace switch
# ---------------------------------------------------------------------------

@workspace_app.command("switch")
def workspace_switch(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del workspace al que cambiar.",
        metavar="ID_O_NOMBRE",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Confirmar el cambio automáticamente.",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar detalles del workspace activado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Cambia el workspace activo del sistema.

    Acepta tanto el ID como el nombre del workspace (incluso parcial
    si es único). Todas las operaciones posteriores usarán el nuevo workspace.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge workspace switch trabajo[/cyan]
      [cyan]hiperforge workspace switch 01HX4K2J8QNVR0SBPZ1Y3W9D6E[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("workspace switch"):
        container = Container.build()

        # Obtener el nombre del workspace activo actual para la confirmación
        current_name: str | None = None
        current_id = container.store.get_active_workspace_id()
        if current_id:
            try:
                current_ws = container.store.workspaces.find_by_id_meta(current_id)
                current_name = current_ws.name
            except Exception:
                pass

        # Construir el input según si es ID o nombre
        switch_input = _build_switch_input(target)

        # Confirmar el cambio si no es automático
        if not yes:
            # Necesitamos el nombre del destino para la confirmación
            target_name = _resolve_target_name(container, target)
            if not Confirm.workspace_switch(
                current_name=current_name,
                target_name=target_name,
                force=yes,
            ):
                raise typer.Abort()

        summary = container.switch_workspace.execute(switch_input)
        renderer.render_workspace_switched(summary)


# ---------------------------------------------------------------------------
# workspace rename
# ---------------------------------------------------------------------------

@workspace_app.command("rename")
def workspace_rename(
    target: str = typer.Argument(
        ...,
        help="ID o nombre actual del workspace.",
        metavar="ID_O_NOMBRE",
    ),
    new_name: str = typer.Argument(
        ...,
        help="Nuevo nombre para el workspace.",
        metavar="NUEVO_NOMBRE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar detalles del workspace renombrado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Renombra un workspace existente.

    Verifica que el nuevo nombre sea único antes de aplicar el cambio.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge workspace rename trabajo "Trabajo Freelance"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("workspace rename"):
        container = Container.build()

        # Resolver el workspace
        workspace = _resolve_workspace(container, target)

        # Verificar que el nuevo nombre no existe ya
        if container.store.workspaces.exists_by_name(new_name):
            # Verificar que no es el mismo workspace (renombrar a sí mismo)
            if new_name.lower().strip() != workspace.name.lower():
                raise DuplicateEntity(
                    entity_type="Workspace",
                    identifier=new_name,
                )

        # Aplicar el rename en el dominio y persistir
        renamed = workspace.rename(new_name)
        container.store.workspaces.save(renamed)

        renderer.render_success(
            f"Workspace renombrado: "
            f"[bold]{workspace.name}[/bold] → [bold cyan]{new_name}[/bold cyan]"
        )

        if verbose:
            renderer.render_id("Workspace", workspace.id)


# ---------------------------------------------------------------------------
# workspace archive
# ---------------------------------------------------------------------------

@workspace_app.command("archive")
def workspace_archive(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del workspace a archivar.",
        metavar="ID_O_NOMBRE",
    ),
    yes: bool = typer.Option(
        False,
        "--yes", "-y",
        help="Confirmar el archivado automáticamente.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Archiva un workspace. Sus proyectos y tasks quedan en solo lectura.

    El workspace archivado no puede recibir nuevos proyectos ni tasks.
    Usa [bold]workspace reactivate[/bold] para revertir este cambio.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge workspace archive "Cliente Antiguo"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("workspace archive"):
        container = Container.build()

        workspace = _resolve_workspace(container, target)

        # Verificar que no está ya archivado
        if workspace.status == WorkspaceStatus.ARCHIVED:
            renderer.render_warning(
                f"El workspace '[bold]{workspace.name}[/bold]' ya está archivado."
            )
            raise typer.Exit(0)

        if workspace.status == WorkspaceStatus.DELETED:
            renderer.render_warning(
                f"El workspace '[bold]{workspace.name}[/bold]' fue eliminado."
            )
            raise typer.Exit(1)

        # Confirmación con consecuencias claras
        if not yes:
            consequences = [
                f"El workspace '{workspace.name}' quedará en modo solo lectura.",
                f"{workspace.project_count} proyecto(s) no podrán recibir nuevas tasks.",
                "No podrás crear nuevos proyectos en este workspace.",
                "Reversible con: hiperforge workspace reactivate",
            ]
            if not Confirm.destructive(
                message=f"¿Archivar el workspace '{workspace.name}'?",
                consequences=consequences,
                force=yes,
            ):
                raise typer.Abort()

        # Aplicar el archivado
        archived = workspace.archive()
        container.store.workspaces.save(archived)

        # Si era el workspace activo, desactivarlo
        active_id = container.store.get_active_workspace_id()
        if active_id == workspace.id:
            _deactivate_workspace(container, renderer, workspace.name)

        renderer.render_success(
            f"Workspace '[bold]{workspace.name}[/bold]' archivado. "
            f"Reactívalo con: [bold]hiperforge workspace reactivate {workspace.id}[/bold]"
        )


# ---------------------------------------------------------------------------
# workspace reactivate
# ---------------------------------------------------------------------------

@workspace_app.command("reactivate")
def workspace_reactivate(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del workspace a reactivar.",
        metavar="ID_O_NOMBRE",
    ),
    activate: bool = typer.Option(
        False,
        "--activate", "-a",
        help="Activar el workspace como activo al reactivarlo.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Reactiva un workspace archivado para que pueda recibir nuevos proyectos.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge workspace reactivate "Cliente Antiguo" --activate[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("workspace reactivate"):
        container = Container.build()

        workspace = _resolve_workspace(container, target)

        if workspace.status == WorkspaceStatus.ACTIVE:
            renderer.render_warning(
                f"El workspace '[bold]{workspace.name}[/bold]' ya está activo."
            )
            raise typer.Exit(0)

        if workspace.status == WorkspaceStatus.DELETED:
            renderer.render_warning(
                f"El workspace '[bold]{workspace.name}[/bold]' fue eliminado "
                f"y no puede reactivarse."
            )
            raise typer.Exit(1)

        # Reactivar en el dominio y persistir
        reactivated = workspace.reactivate()
        container.store.workspaces.save(reactivated)

        # Activar si se solicitó
        if activate:
            container.store.workspaces.set_active_workspace(workspace.id)
            renderer.render_success(
                f"Workspace '[bold]{workspace.name}[/bold]' reactivado y activado."
            )
        else:
            renderer.render_success(
                f"Workspace '[bold]{workspace.name}[/bold]' reactivado. "
                f"Actívalo con: [bold]hiperforge workspace switch {workspace.id}[/bold]"
            )


# ---------------------------------------------------------------------------
# workspace delete
# ---------------------------------------------------------------------------

@workspace_app.command("delete")
def workspace_delete(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del workspace a eliminar.",
        metavar="ID_O_NOMBRE",
    ),
    force: bool = typer.Option(
        False,
        "--force", "-f",
        help="Forzar la eliminación sin confirmación. [red]¡PELIGROSO![/red]",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Elimina lógicamente un workspace.

    [bold red]⚠ OPERACIÓN IRREVERSIBLE[/bold red]

    Los datos se marcan como eliminados. Para confirmarlo, deberás
    escribir el nombre exacto del workspace.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge workspace delete "Proyecto Viejo"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("workspace delete"):
        container = Container.build()

        workspace = _resolve_workspace(container, target)

        if workspace.status == WorkspaceStatus.DELETED:
            renderer.render_warning(
                f"El workspace '[bold]{workspace.name}[/bold]' ya fue eliminado."
            )
            raise typer.Exit(0)

        # Construir las consecuencias concretas con datos reales
        consequences = _build_delete_consequences(workspace)

        # Mostrar consecuencias y pedir confirmación destructiva
        if not force:
            if not Confirm.destructive(
                message=f"¿Eliminar el workspace '{workspace.name}'?",
                consequences=consequences,
                force=False,
            ):
                raise typer.Abort()

            # Double-confirm: el usuario debe escribir el nombre exacto
            if not Confirm.by_name(
                entity_type="workspace",
                entity_name=workspace.name,
                force=False,
            ):
                raise typer.Abort()

        # Aplicar eliminación lógica en el dominio
        deleted = workspace.delete()
        container.store.workspaces.save(deleted)

        # Eliminar físicamente si el estado es DELETED
        # (workspace_repo.save() ya no lo mostrará en find_all())
        try:
            container.store.workspaces.delete(workspace.id)
        except Exception as exc:
            logger.warning(
                "no se pudo eliminar físicamente el workspace",
                workspace_id=workspace.id,
                error=str(exc),
            )
            # No es fatal — la eliminación lógica ya ocurrió

        # Si era el workspace activo, limpiar el índice
        active_id = container.store.get_active_workspace_id()
        if active_id == workspace.id:
            _deactivate_workspace(container, renderer, workspace.name)

        renderer.render_success(
            f"Workspace '[bold]{workspace.name}[/bold]' eliminado."
        )


# ---------------------------------------------------------------------------
# workspace show
# ---------------------------------------------------------------------------

@workspace_app.command("show")
def workspace_show(
    target: Optional[str] = typer.Argument(
        None,
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="ID_O_NOMBRE",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar todos los detalles incluyendo ID completo.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Muestra el detalle del workspace activo o del especificado.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge workspace show[/cyan]
      [cyan]hiperforge workspace show trabajo --verbose[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("workspace show"):
        container = Container.build()
        active_id = container.store.get_active_workspace_id()

        if target:
            workspace = _resolve_workspace(container, target)
        else:
            if not active_id:
                renderer.render_info(
                    "No hay workspace activo. "
                    "Crea uno con: [bold]hiperforge workspace create <nombre>[/bold]"
                )
                raise typer.Exit(0)
            workspace = container.store.workspaces.find_by_id_meta(active_id)

        # Mostrar detalle del workspace
        from hiperforge.application.dto import WorkspaceSummary
        summary = WorkspaceSummary(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            status=workspace.status.value,
            project_count=workspace.project_count,
            is_active=(workspace.id == active_id),
            created_at=workspace.created_at,
        )

        renderer.render_workspace_created(summary)


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _resolve_workspace(container: Container, target: str):
    """
    Resuelve un workspace por ID o nombre usando SwitchWorkspaceUseCase.

    Reutiliza la lógica de resolución implementada en el use case
    para garantizar consistencia en todos los comandos workspace.

    Parámetros:
        container: Container con acceso al store.
        target:    ID o nombre del workspace.

    Returns:
        Workspace en modo meta (sin proyectos cargados).

    Raises:
        EntityNotFound: Si el workspace no existe.
    """
    # Intentar resolución directa si parece un ULID
    if len(target) == 26 and target.isalnum():
        return container.store.workspaces.find_by_id_meta(target)

    # Buscar por nombre usando la lógica del use case
    # (búsqueda exacta primero, parcial como fallback)
    all_workspaces = container.store.workspaces.find_all()
    target_lower = target.strip().lower()

    # Búsqueda exacta
    for ws in all_workspaces:
        if ws.name.lower() == target_lower:
            return ws

    # Búsqueda parcial
    partial = [ws for ws in all_workspaces if target_lower in ws.name.lower()]
    if len(partial) == 1:
        return partial[0]

    if len(partial) > 1:
        names = ", ".join(f"'{ws.name}'" for ws in partial)
        raise EntityNotFound(
            entity_type="Workspace",
            entity_id=target,
        )

    raise EntityNotFound(
        entity_type="Workspace",
        entity_id=target,
    )


def _resolve_target_name(container: Container, target: str) -> str:
    """
    Obtiene el nombre del workspace destino para mensajes de confirmación.

    Devuelve el target original si no puede resolver el nombre — la
    confirmación se mostrará con el target tal cual.
    """
    try:
        ws = _resolve_workspace(container, target)
        return ws.name
    except Exception:
        return target


def _build_switch_input(target: str) -> SwitchWorkspaceInput:
    """
    Construye el SwitchWorkspaceInput según si target es ID o nombre.

    Heurística: ULIDs tienen exactamente 26 caracteres alfanuméricos.

    Parámetros:
        target: ID o nombre del workspace destino.

    Returns:
        SwitchWorkspaceInput con el campo correcto rellenado.
    """
    if len(target) == 26 and target.isalnum():
        return SwitchWorkspaceInput(workspace_id=target)
    return SwitchWorkspaceInput(workspace_name=target)


def _build_delete_consequences(workspace) -> list[str]:
    """
    Construye la lista de consecuencias concretas de eliminar un workspace.

    Usa los datos reales del workspace para ser específico — no mensajes
    genéricos sino información concreta sobre lo que se perderá.

    Parámetros:
        workspace: Entidad Workspace con project_count y datos.

    Returns:
        Lista de strings describiendo las consecuencias del delete.
    """
    consequences = [
        f"El workspace '[bold]{workspace.name}[/bold]' se eliminará permanentemente.",
    ]

    if workspace.project_count > 0:
        consequences.append(
            f"{workspace.project_count} proyecto(s) y todas sus tasks se eliminarán."
        )
    else:
        consequences.append("El workspace no tiene proyectos.")

    consequences.extend([
        "Las preferencias del workspace se perderán.",
        "Esta operación no se puede deshacer desde la CLI.",
        "Los archivos en disco se eliminarán definitivamente.",
    ])

    return consequences


def _deactivate_workspace(
    container: Container,
    renderer: Renderer,
    workspace_name: str,
) -> None:
    """
    Limpia el workspace activo del índice global cuando el activo es eliminado o archivado.

    Cuando el workspace activo es eliminado o archivado, debe haber un
    workspace activo o ninguno. Aquí limpiamos el puntero del índice
    y sugerimos al usuario que active otro.

    Parámetros:
        container:      Container con acceso al store.
        renderer:       Renderer para mostrar el aviso al usuario.
        workspace_name: Nombre del workspace que se está desactivando.
    """
    try:
        # Cargar el índice y remover el workspace activo
        index = container.store.workspaces.load_index()
        updated_index = index.set_active(None)  # type: ignore[arg-type]
        container.store.workspaces.save_index(updated_index)
    except Exception as exc:
        logger.warning(
            "no se pudo limpiar el workspace activo del índice",
            error=str(exc),
        )

    renderer.render_warning(
        f"'{workspace_name}' era el workspace activo. "
        f"Activa otro con: [bold]hiperforge workspace switch <nombre>[/bold]"
    )