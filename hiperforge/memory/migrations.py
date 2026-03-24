"""
Migrations — Sistema de migraciones de schema JSON.

Cuando HiperForge evoluciona, la estructura de los archivos JSON
puede cambiar — se agregan campos, se renombran, se reorganizan.

Sin migraciones, los archivos JSON viejos serían incompatibles
con la nueva versión del código. El usuario perdería todos sus datos.

Con migraciones, los archivos JSON viejos se actualizan automáticamente
al formato actual la primera vez que se leen.

¿CÓMO FUNCIONA?
  Cada archivo JSON tiene un campo schema_version.
  Al leer un archivo, comparamos su versión con CURRENT_VERSION.
  Si es menor, aplicamos las migraciones necesarias en secuencia.

  Ejemplo: archivo en v1, código en v3
    → aplicar migración v1→v2
    → aplicar migración v2→v3
    → guardar el archivo actualizado a v3
    → devolver los datos migrados

AGREGAR UNA MIGRACIÓN NUEVA:
  1. Incrementar la constante SCHEMA_VERSION_* en constants.py
  2. Agregar la función de migración en este archivo:
       def _migrate_workspace_v1_to_v2(data: dict) -> dict:
           data["nuevo_campo"] = "valor_default"
           return data
  3. Registrarla en el mapa _WORKSPACE_MIGRATIONS:
       _WORKSPACE_MIGRATIONS = {1: _migrate_workspace_v1_to_v2}

  El sistema aplica todas las migraciones pendientes automáticamente.

SEGURIDAD:
  Las migraciones NUNCA eliminan datos — solo agregan o transforman.
  Si una migración necesita eliminar un campo, lo renombra a
  _deprecated_{campo} para preservarlo por si acaso.
"""

from __future__ import annotations

from typing import Any, Callable

from hiperforge.core.constants import (
    SCHEMA_VERSION_PREFERENCES,
    SCHEMA_VERSION_PROJECT,
    SCHEMA_VERSION_TASK,
    SCHEMA_VERSION_WORKSPACE,
)
from hiperforge.core.logging import get_logger

logger = get_logger(__name__)

# Tipo de una función de migración
MigrationFn = Callable[[dict[str, Any]], dict[str, Any]]

# ---------------------------------------------------------------------------
# Mapas de migraciones por tipo de entidad
# Estructura: { version_origen: función_que_migra_a_version_siguiente }
# ---------------------------------------------------------------------------

# Workspace: actualmente en v1, sin migraciones pendientes
_WORKSPACE_MIGRATIONS: dict[int, MigrationFn] = {
    # Ejemplo de cómo agregar una migración en el futuro:
    # 1: _migrate_workspace_v1_to_v2,
}

# Project: actualmente en v1
_PROJECT_MIGRATIONS: dict[int, MigrationFn] = {}

# Task: actualmente en v1
_TASK_MIGRATIONS: dict[int, MigrationFn] = {}

# Preferences: actualmente en v1
_PREFERENCES_MIGRATIONS: dict[int, MigrationFn] = {}


# ---------------------------------------------------------------------------
# Motor de migraciones
# ---------------------------------------------------------------------------

def migrate_workspace(data: dict[str, Any]) -> dict[str, Any]:
    """
    Migra un workspace.json al schema actual si es necesario.

    Parámetros:
        data: Contenido del workspace.json leído del disco.

    Returns:
        Data migrada al schema_version actual.
        Si ya está en la versión actual, devuelve data sin cambios.
    """
    return _migrate(
        data=data,
        entity_type="Workspace",
        migrations=_WORKSPACE_MIGRATIONS,
        target_version=SCHEMA_VERSION_WORKSPACE,
    )


def migrate_project(data: dict[str, Any]) -> dict[str, Any]:
    """Migra un project.json al schema actual si es necesario."""
    return _migrate(
        data=data,
        entity_type="Project",
        migrations=_PROJECT_MIGRATIONS,
        target_version=SCHEMA_VERSION_PROJECT,
    )


def migrate_task(data: dict[str, Any]) -> dict[str, Any]:
    """Migra un task.json al schema actual si es necesario."""
    return _migrate(
        data=data,
        entity_type="Task",
        migrations=_TASK_MIGRATIONS,
        target_version=SCHEMA_VERSION_TASK,
    )


def migrate_preferences(data: dict[str, Any]) -> dict[str, Any]:
    """Migra un preferences.json al schema actual si es necesario."""
    return _migrate(
        data=data,
        entity_type="Preferences",
        migrations=_PREFERENCES_MIGRATIONS,
        target_version=SCHEMA_VERSION_PREFERENCES,
    )


def _migrate(
    data: dict[str, Any],
    entity_type: str,
    migrations: dict[int, MigrationFn],
    target_version: int,
) -> dict[str, Any]:
    """
    Motor genérico de migraciones.

    Aplica migraciones secuencialmente desde la versión actual del archivo
    hasta la versión target. Si ya está en target, no hace nada.

    Parámetros:
        data:          Contenido del JSON a migrar.
        entity_type:   Nombre de la entidad para logging ("Workspace", etc.).
        migrations:    Mapa de migraciones disponibles para este tipo.
        target_version: Versión a la que queremos llegar.

    Returns:
        Data migrada con schema_version actualizado.
    """
    current_version = data.get("schema_version", 1)

    # Ya está en la versión actual — no hay nada que migrar
    if current_version >= target_version:
        return data

    logger.info(
        "migrando schema JSON",
        entity_type=entity_type,
        from_version=current_version,
        to_version=target_version,
    )

    migrated = dict(data)

    # Aplicamos migraciones en secuencia: v1→v2, v2→v3, etc.
    for version in range(current_version, target_version):
        migration_fn = migrations.get(version)

        if migration_fn is None:
            # No hay migración definida para este salto de versión
            # Esto puede ocurrir si se saltaron versiones en el desarrollo
            # Loggeamos advertencia pero continuamos — puede ser intencional
            logger.warning(
                "no hay migración definida para esta versión",
                entity_type=entity_type,
                version=version,
            )
            continue

        try:
            migrated = migration_fn(migrated)
            logger.debug(
                "migración aplicada",
                entity_type=entity_type,
                version=f"v{version}→v{version + 1}",
            )
        except Exception as exc:
            # Si falla una migración, loggeamos pero devolvemos
            # los datos hasta donde llegamos — es mejor que nada
            logger.error(
                "fallo en migración",
                entity_type=entity_type,
                version=version,
                error=str(exc),
            )
            break

    # Actualizamos la versión en el dato migrado
    migrated["schema_version"] = target_version

    return migrated


# ---------------------------------------------------------------------------
# Funciones de migración concretas (ejemplos comentados para referencia)
# ---------------------------------------------------------------------------

# Ejemplo de migración futura de Workspace v1 → v2:
# def _migrate_workspace_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
#     """
#     v2 agrega el campo 'color' para identificación visual en la CLI.
#     Default: None (sin color asignado).
#     """
#     data["color"] = None
#     return data

# Ejemplo de migración futura de Task v1 → v2:
# def _migrate_task_v1_to_v2(data: dict[str, Any]) -> dict[str, Any]:
#     """
#     v2 renombra 'plan' a 'subtasks' para mayor claridad.
#     """
#     if "plan" in data and "subtasks" not in data:
#         data["subtasks"] = data.pop("plan")
#     return data