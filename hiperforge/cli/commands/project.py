"""
Comandos `project` — Gestión completa del ciclo de vida de proyectos.

Los proyectos agrupan tasks relacionadas bajo un mismo contexto dentro
de un workspace. Este módulo expone todas las operaciones disponibles
sobre proyectos: crear, listar, mostrar, renombrar, etiquetar,
archivar, reactivar y eliminar.

════════════════════════════════════════════════════════════
COMANDOS DISPONIBLES
════════════════════════════════════════════════════════════

  hiperforge project create <nombre>
    Crea un nuevo proyecto en el workspace activo.
    Acepta descripción opcional y tags de categorización.

  hiperforge project list
    Lista todos los proyectos del workspace activo con métricas
    de progreso (tasks completadas / total).

  hiperforge project show <id_o_nombre>
    Muestra el detalle completo de un proyecto: descripción,
    tags, estado y métricas de sus tasks.

  hiperforge project rename <id_o_nombre> <nuevo_nombre>
    Renombra un proyecto. Verifica unicidad en el workspace.

  hiperforge project tag <id_o_nombre> <tag1> [tag2...]
    Añade tags a un proyecto para categorización y filtrado.

  hiperforge project untag <id_o_nombre> <tag1> [tag2...]
    Elimina tags de un proyecto.

  hiperforge project archive <id_o_nombre>
    Archiva un proyecto. Ya no puede recibir nuevas tasks.
    Reversible con project reactivate.

  hiperforge project reactivate <id_o_nombre>
    Reactiva un proyecto archivado.

  hiperforge project delete <id_o_nombre>
    Eliminación lógica del proyecto. Requiere confirmación por nombre.
    Irreversible desde la CLI.

════════════════════════════════════════════════════════════
RESOLUCIÓN DE PROYECTOS
════════════════════════════════════════════════════════════

  Todos los comandos que aceptan <id_o_nombre> usan la misma
  heurística que workspace:
    - ULID de 26 chars → resolución directa por ID.
    - Texto → búsqueda case-insensitive por nombre.
    - Texto parcial → fuzzy match si el resultado es único.

  La búsqueda siempre opera dentro del workspace activo,
  a menos que se especifique --workspace explícitamente.

════════════════════════════════════════════════════════════
CONFIRMACIONES Y SEGURIDAD
════════════════════════════════════════════════════════════

  archive:
    → Confirmación simple (Sí/No) con descripción del impacto.
    → Reversible con reactivate.

  delete:
    → Muestra las consecuencias (N tasks afectadas).
    → Double-confirm: el usuario debe escribir el nombre exacto.
    → Irreversible desde la CLI.
"""

from __future__ import annotations

from typing import Optional

import typer

from hiperforge.application import Container, CreateProjectInput
from hiperforge.application.dto import ProjectSummary
from hiperforge.cli.error_handler import ErrorHandler
from hiperforge.cli.ui import Confirm, Renderer
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.project import ProjectStatus
from hiperforge.domain.exceptions import DuplicateEntity, EntityNotFound

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# App Typer del grupo project
# ---------------------------------------------------------------------------

project_app = typer.Typer(
    name="project",
    help="Gestión de proyectos dentro del workspace activo.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)


# ---------------------------------------------------------------------------
# project create
# ---------------------------------------------------------------------------

@project_app.command("create")
def project_create(
    name: str = typer.Argument(
        ...,
        help="Nombre del proyecto. Debe ser único en el workspace.",
        metavar="NOMBRE",
    ),
    description: Optional[str] = typer.Option(
        None,
        "--description", "--desc", "-d",
        help="Descripción del propósito del proyecto.",
        metavar="DESC",
    ),
    tags: Optional[str] = typer.Option(
        None,
        "--tags", "-t",
        help="Tags separados por coma. Ejemplo: --tags backend,api,python",
        metavar="TAGS",
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
        help="Mostrar el ID del proyecto creado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Crea un nuevo proyecto en el workspace activo.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge project create "API de pagos"[/cyan]
      [cyan]hiperforge project create "Dashboard" --desc "Panel de admin" --tags frontend,react[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("project create"):
        container = Container.build()

        # Resolver workspace activo
        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning(
                "No hay workspace activo. "
                "Crea uno con: [bold]hiperforge workspace create <nombre>[/bold]"
            )
            raise typer.Exit(1)

        # Parsear tags separados por coma
        parsed_tags = _parse_tags(tags)

        input_data = CreateProjectInput(
            name=name,
            workspace_id=workspace_id,
            description=description,
            tags=parsed_tags,
        )

        summary = container.create_project.execute(input_data)
        renderer.render_project_created(summary)


# ---------------------------------------------------------------------------
# project list
# ---------------------------------------------------------------------------

@project_app.command("list")
def project_list(
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    status: Optional[str] = typer.Option(
        None,
        "--status", "-s",
        help="Filtrar por estado (active, archived).",
        metavar="STATUS",
    ),
    tag: Optional[str] = typer.Option(
        None,
        "--tag",
        help="Filtrar proyectos que tengan este tag.",
        metavar="TAG",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Mostrar IDs completos y fechas.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Lista los proyectos del workspace activo con métricas de progreso.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge project list[/cyan]
      [cyan]hiperforge project list --status active --tag backend[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("project list"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_info(
                "No hay workspace activo. "
                "Crea uno con: [bold]hiperforge workspace create <nombre>[/bold]"
            )
            raise typer.Exit(0)

        # Cargar proyectos del workspace en modo ligero (sin tasks)
        projects_raw = container.store.projects.find_all(workspace_id)

        # Aplicar filtros
        if status:
            projects_raw = [p for p in projects_raw if p.status.value == status]
        if tag:
            tag_lower = tag.lower()
            projects_raw = [
                p for p in projects_raw
                if any(t.lower() == tag_lower for t in p.tags)
            ]

        # Convertir a ProjectSummary para el renderer
        summaries = [
            ProjectSummary(
                id=proj.id,
                name=proj.name,
                description=proj.description,
                status=proj.status.value,
                tags=list(proj.tags),
                task_count=proj.task_count,
                completed_tasks=len(proj.completed_tasks),
                created_at=proj.created_at,
                updated_at=proj.updated_at,
            )
            for proj in projects_raw
        ]

        renderer.render_project_list(summaries)


# ---------------------------------------------------------------------------
# project show
# ---------------------------------------------------------------------------

@project_app.command("show")
def project_show(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto.",
        metavar="ID_O_NOMBRE",
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
        help="Mostrar ID completo y todos los metadatos.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Muestra el detalle completo de un proyecto.

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge project show "API de pagos"[/cyan]
      [cyan]hiperforge project show 01HX4K2J8QNVR0SBPZ1Y3W9D6E --verbose[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("project show"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        summary = ProjectSummary(
            id=project.id,
            name=project.name,
            description=project.description,
            status=project.status.value,
            tags=list(project.tags),
            task_count=project.task_count,
            completed_tasks=len(project.completed_tasks),
            created_at=project.created_at,
            updated_at=project.updated_at,
        )

        renderer.render_project_created(summary)

        # Mostrar estadísticas adicionales
        if project.task_count > 0:
            renderer.render_info(
                f"Progreso: {len(project.completed_tasks)}/{project.task_count} tasks "
                f"completadas ({project.completed_ratio * 100:.0f}%)"
            )

        if project.active_tasks:
            renderer.render_info(
                f"{len(project.active_tasks)} task(s) en ejecución actualmente."
            )

        if project.failed_tasks:
            renderer.render_warning(
                f"{len(project.failed_tasks)} task(s) fallaron. "
                f"Revísalas con: [bold]hiperforge task list --project {project.id} --status failed[/bold]"
            )


# ---------------------------------------------------------------------------
# project rename
# ---------------------------------------------------------------------------

@project_app.command("rename")
def project_rename(
    target: str = typer.Argument(
        ...,
        help="ID o nombre actual del proyecto.",
        metavar="ID_O_NOMBRE",
    ),
    new_name: str = typer.Argument(
        ...,
        help="Nuevo nombre para el proyecto.",
        metavar="NUEVO_NOMBRE",
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
        help="Mostrar el ID del proyecto renombrado.",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Renombra un proyecto existente.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge project rename "API vieja" "API v2"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer(verbose=verbose)

    with error_handler.context("project rename"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        # Verificar unicidad del nuevo nombre en el workspace
        existing_projects = container.store.projects.find_all(workspace_id)
        for existing in existing_projects:
            if (
                existing.name.lower() == new_name.strip().lower()
                and existing.id != project.id
                and existing.status != ProjectStatus.DELETED
            ):
                raise DuplicateEntity(
                    entity_type="Project",
                    identifier=new_name,
                )

        # Aplicar rename en el dominio y persistir
        renamed = project.rename(new_name)
        container.store.projects.save(renamed)

        renderer.render_success(
            f"Proyecto renombrado: "
            f"[bold]{project.name}[/bold] → [bold cyan]{new_name}[/bold cyan]"
        )

        if verbose:
            renderer.render_id("Project", project.id)


# ---------------------------------------------------------------------------
# project tag
# ---------------------------------------------------------------------------

@project_app.command("tag")
def project_tag(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto.",
        metavar="ID_O_NOMBRE",
    ),
    new_tags: list[str] = typer.Argument(
        ...,
        help="Tags a añadir al proyecto.",
        metavar="TAG",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Añade tags a un proyecto para categorización y filtrado.

    Los tags que ya existen en el proyecto son ignorados (idempotente).

    [bold]Ejemplos:[/bold]

      [cyan]hiperforge project tag "API de pagos" backend python stripe[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("project tag"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        # Normalizar los tags: minúsculas, sin espacios
        normalized_tags = [t.lower().strip() for t in new_tags if t.strip()]

        # Combinar con los tags existentes (sin duplicados, preservando orden)
        existing_tags = list(project.tags)
        merged_tags = existing_tags + [
            t for t in normalized_tags if t not in existing_tags
        ]

        updated = project.update_tags(merged_tags)
        container.store.projects.save(updated)

        added = [t for t in normalized_tags if t not in existing_tags]
        if added:
            renderer.render_success(
                f"Tags añadidos a '[bold]{project.name}[/bold]': "
                + " ".join(f"[cyan]#{t}[/cyan]" for t in added)
            )
        else:
            renderer.render_info(
                f"Los tags ya existían en '[bold]{project.name}[/bold]'. Sin cambios."
            )


# ---------------------------------------------------------------------------
# project untag
# ---------------------------------------------------------------------------

@project_app.command("untag")
def project_untag(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto.",
        metavar="ID_O_NOMBRE",
    ),
    tags_to_remove: list[str] = typer.Argument(
        ...,
        help="Tags a eliminar del proyecto.",
        metavar="TAG",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Elimina tags de un proyecto.

    Los tags que no existen en el proyecto son ignorados (idempotente).

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge project untag "API de pagos" deprecated[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("project untag"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        tags_to_remove_lower = {t.lower().strip() for t in tags_to_remove}
        remaining_tags = [
            t for t in project.tags
            if t.lower() not in tags_to_remove_lower
        ]

        removed = [t for t in project.tags if t.lower() in tags_to_remove_lower]

        updated = project.update_tags(remaining_tags)
        container.store.projects.save(updated)

        if removed:
            renderer.render_success(
                f"Tags eliminados de '[bold]{project.name}[/bold]': "
                + " ".join(f"[dim]#{t}[/dim]" for t in removed)
            )
        else:
            renderer.render_info(
                f"Ninguno de los tags especificados existía en '[bold]{project.name}[/bold]'."
            )


# ---------------------------------------------------------------------------
# project archive
# ---------------------------------------------------------------------------

@project_app.command("archive")
def project_archive(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto a archivar.",
        metavar="ID_O_NOMBRE",
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
    Archiva un proyecto. Ya no puede recibir nuevas tasks.

    Reversible con [bold]project reactivate[/bold].

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge project archive "Proyecto Terminado"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("project archive"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        if project.status == ProjectStatus.ARCHIVED:
            renderer.render_warning(
                f"El proyecto '[bold]{project.name}[/bold]' ya está archivado."
            )
            raise typer.Exit(0)

        if project.status == ProjectStatus.DELETED:
            renderer.render_warning(
                f"El proyecto '[bold]{project.name}[/bold]' fue eliminado."
            )
            raise typer.Exit(1)

        # Confirmación con consecuencias
        if not yes:
            active_count = len(project.active_tasks)
            consequences = [
                f"El proyecto '{project.name}' quedará en modo solo lectura.",
                f"{project.task_count} task(s) no podrán modificarse.",
            ]
            if active_count > 0:
                consequences.append(
                    f"⚠ {active_count} task(s) están en ejecución activa — "
                    f"serán interrumpidas."
                )
            consequences.append(
                "Reversible con: hiperforge project reactivate"
            )

            if not Confirm.destructive(
                message=f"¿Archivar el proyecto '{project.name}'?",
                consequences=consequences,
                force=yes,
            ):
                raise typer.Abort()

        archived = project.archive()
        container.store.projects.save(archived)

        renderer.render_success(
            f"Proyecto '[bold]{project.name}[/bold]' archivado. "
            f"Reactívalo con: [bold]hiperforge project reactivate {project.id}[/bold]"
        )


# ---------------------------------------------------------------------------
# project reactivate
# ---------------------------------------------------------------------------

@project_app.command("reactivate")
def project_reactivate(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto a reactivar.",
        metavar="ID_O_NOMBRE",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
    ),
    debug: bool = typer.Option(
        False,
        "--debug",
        help="Modo debug.",
        envvar="HIPERFORGE_DEBUG",
    ),
) -> None:
    """
    Reactiva un proyecto archivado para que pueda recibir nuevas tasks.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge project reactivate "Proyecto Antiguo"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("project reactivate"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        if project.status == ProjectStatus.ACTIVE:
            renderer.render_warning(
                f"El proyecto '[bold]{project.name}[/bold]' ya está activo."
            )
            raise typer.Exit(0)

        if project.status == ProjectStatus.DELETED:
            renderer.render_warning(
                f"El proyecto '[bold]{project.name}[/bold]' fue eliminado "
                f"y no puede reactivarse."
            )
            raise typer.Exit(1)

        reactivated = project.reactivate()
        container.store.projects.save(reactivated)

        renderer.render_success(
            f"Proyecto '[bold]{project.name}[/bold]' reactivado."
        )


# ---------------------------------------------------------------------------
# project delete
# ---------------------------------------------------------------------------

@project_app.command("delete")
def project_delete(
    target: str = typer.Argument(
        ...,
        help="ID o nombre del proyecto a eliminar.",
        metavar="ID_O_NOMBRE",
    ),
    workspace: Optional[str] = typer.Option(
        None,
        "--workspace", "-w",
        help="ID o nombre del workspace. Default: workspace activo.",
        metavar="WORKSPACE",
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
    Elimina lógicamente un proyecto y todas sus tasks.

    [bold red]⚠ OPERACIÓN IRREVERSIBLE[/bold red]

    Deberás escribir el nombre exacto del proyecto para confirmar.

    [bold]Ejemplo:[/bold]

      [cyan]hiperforge project delete "Proyecto Viejo"[/cyan]
    """
    error_handler = ErrorHandler.from_settings(debug=debug)
    renderer = Renderer()

    with error_handler.context("project delete"):
        container = Container.build()

        workspace_id = _resolve_workspace_id(container, workspace)
        if not workspace_id:
            renderer.render_warning("No hay workspace activo.")
            raise typer.Exit(1)

        project = _resolve_project(container, workspace_id, target)

        if project.status == ProjectStatus.DELETED:
            renderer.render_warning(
                f"El proyecto '[bold]{project.name}[/bold]' ya fue eliminado."
            )
            raise typer.Exit(0)

        # Construir consecuencias concretas
        consequences = _build_delete_consequences(project)

        if not force:
            if not Confirm.destructive(
                message=f"¿Eliminar el proyecto '{project.name}'?",
                consequences=consequences,
                force=False,
            ):
                raise typer.Abort()

            # Double-confirm por nombre
            if not Confirm.by_name(
                entity_type="proyecto",
                entity_name=project.name,
                force=False,
            ):
                raise typer.Abort()

        # Eliminación lógica en el dominio
        deleted = project.delete()
        container.store.projects.save(deleted)

        # Eliminación física de los archivos del proyecto
        try:
            container.store.projects.delete(workspace_id, project.id)
        except Exception as exc:
            logger.warning(
                "no se pudo eliminar físicamente el proyecto",
                project_id=project.id,
                error=str(exc),
            )

        # Actualizar el workspace para que no referencie el proyecto eliminado
        try:
            workspace_entity = container.store.workspaces.find_by_id(workspace_id)
            updated_ws = workspace_entity.update_project(deleted)
            container.store.workspaces.save(updated_ws)
        except Exception as exc:
            logger.warning(
                "no se pudo actualizar el workspace tras eliminar el proyecto",
                project_id=project.id,
                error=str(exc),
            )

        renderer.render_success(
            f"Proyecto '[bold]{project.name}[/bold]' eliminado."
        )


# ---------------------------------------------------------------------------
# Helpers privados
# ---------------------------------------------------------------------------

def _resolve_workspace_id(
    container: Container,
    workspace_arg: str | None,
) -> str | None:
    """
    Resuelve el workspace_id efectivo para las operaciones de proyecto.

    Si se especificó un workspace explícito, lo resuelve por ID o nombre.
    Si no se especificó, devuelve el workspace activo global.

    Parámetros:
        container:     Container con acceso al store.
        workspace_arg: Argumento --workspace del usuario o None.

    Returns:
        workspace_id resuelto, o None si no hay workspace activo.
    """
    if workspace_arg is None:
        return container.store.get_active_workspace_id()

    # Heurística ULID
    if len(workspace_arg) == 26 and workspace_arg.isalnum():
        return workspace_arg

    # Búsqueda por nombre
    workspaces = container.store.workspaces.find_all()
    for ws in workspaces:
        if ws.name.lower() == workspace_arg.lower():
            return ws.id

    return workspace_arg


def _resolve_project(container: Container, workspace_id: str, target: str):
    """
    Resuelve un proyecto por ID o nombre dentro de un workspace específico.

    Aplica la misma heurística ULID + búsqueda por nombre que workspace_commands.
    La búsqueda por nombre es case-insensitive y aplica fuzzy match si el
    resultado es único.

    Parámetros:
        container:    Container con acceso al store de proyectos.
        workspace_id: ID del workspace donde buscar.
        target:       ID o nombre del proyecto.

    Returns:
        Project en modo meta (sin tasks cargadas).

    Raises:
        EntityNotFound: Si el proyecto no se encuentra en el workspace.
    """
    # Resolución directa por ULID
    if len(target) == 26 and target.isalnum():
        return container.store.projects.find_by_id_meta(workspace_id, target)

    # Búsqueda por nombre en el workspace
    all_projects = container.store.projects.find_all(workspace_id)
    target_lower = target.strip().lower()

    # Búsqueda exacta primero (case-insensitive)
    for proj in all_projects:
        if proj.name.lower() == target_lower:
            return proj

    # Búsqueda parcial como fallback si es única
    partial_matches = [
        p for p in all_projects
        if target_lower in p.name.lower()
    ]

    if len(partial_matches) == 1:
        logger.debug(
            "proyecto resuelto por coincidencia parcial",
            query=target,
            matched=partial_matches[0].name,
        )
        return partial_matches[0]

    if len(partial_matches) > 1:
        # Ambiguo — no podemos decidir
        raise EntityNotFound(
            entity_type="Project",
            entity_id=target,
        )

    raise EntityNotFound(
        entity_type="Project",
        entity_id=target,
    )


def _parse_tags(tags_str: str | None) -> list[str]:
    """
    Parsea el string de tags separados por coma.

    Normaliza: minúsculas, sin espacios, sin duplicados, sin vacíos.

    Parámetros:
        tags_str: String de tags separados por coma o None.

    Returns:
        Lista de tags normalizados. Lista vacía si tags_str es None.
    """
    if not tags_str:
        return []

    seen: set[str] = set()
    result: list[str] = []

    for raw_tag in tags_str.split(","):
        normalized = raw_tag.strip().lower()
        if normalized and normalized not in seen:
            seen.add(normalized)
            result.append(normalized)

    return result


def _build_delete_consequences(project) -> list[str]:
    """
    Construye la lista de consecuencias concretas de eliminar un proyecto.

    Usa datos reales del proyecto para ser específico — el usuario
    necesita saber exactamente qué perderá antes de confirmar.

    Parámetros:
        project: Entidad Project con task_count y metadatos.

    Returns:
        Lista de strings describiendo las consecuencias del delete.
    """
    consequences = [
        f"El proyecto '[bold]{project.name}[/bold]' se eliminará permanentemente.",
    ]

    if project.task_count > 0:
        completed_count = len(project.completed_tasks)
        active_count = len(project.active_tasks)
        failed_count = len(project.failed_tasks)

        consequences.append(
            f"{project.task_count} task(s) y todo su historial de ejecución se eliminarán."
        )

        if completed_count > 0:
            consequences.append(
                f"  • {completed_count} task(s) completadas — su trabajo se perderá."
            )
        if active_count > 0:
            consequences.append(
                f"  • ⚠ {active_count} task(s) en ejecución activa serán interrumpidas."
            )
        if failed_count > 0:
            consequences.append(
                f"  • {failed_count} task(s) falladas también se eliminarán."
            )
    else:
        consequences.append("El proyecto no tiene tasks registradas.")

    if project.tags:
        tags_str = ", ".join(f"#{t}" for t in project.tags)
        consequences.append(f"Tags asociados ({tags_str}) se perderán.")

    consequences.append("Esta operación no se puede deshacer.")

    return consequences