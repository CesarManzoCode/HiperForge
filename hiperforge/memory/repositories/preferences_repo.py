"""
PreferencesRepository — Lectura y escritura de preferencias del usuario.

Maneja dos niveles de preferencias:
  - Globales: ~/.hiperforge/preferences.json
  - Por workspace: ~/.hiperforge/workspaces/{id}/preferences.json

La cascada (global → workspace) la resuelve MemoryStore.get_effective_preferences().
Este repositorio solo lee/escribe cada nivel por separado.
"""

from __future__ import annotations

from hiperforge.core.logging import get_logger
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.migrations import migrate_preferences
from hiperforge.memory.schemas.preferences import UserPrefsSchema

logger = get_logger(__name__)


class PreferencesRepository:
    """
    Acceso a preferencias globales y por workspace.
    """

    def __init__(self, storage: JSONStorage, locator: WorkspaceLocator) -> None:
        self._storage = storage
        self._locator = locator

    def load_global(self) -> UserPrefsSchema:
        """
        Carga las preferencias globales.

        Si el archivo no existe, devuelve los defaults sin error —
        es normal en la primera ejecución.
        """
        path = self._locator.global_preferences_file

        if not path.exists():
            return UserPrefsSchema()

        try:
            raw_data = self._storage.read_json(path)
            migrated = migrate_preferences(raw_data)

            # Guardamos si se migró
            if migrated.get("schema_version") != raw_data.get("schema_version"):
                self._storage.write_json(path, migrated)

            return UserPrefsSchema.model_validate(migrated)

        except Exception as exc:
            logger.warning(
                "error cargando preferencias globales — usando defaults",
                error=str(exc),
            )
            return UserPrefsSchema()

    def save_global(self, prefs: UserPrefsSchema) -> None:
        """Persiste las preferencias globales."""
        path = self._locator.global_preferences_file
        self._storage.write_json(path, prefs.model_dump())

        logger.debug("preferencias globales guardadas")

    def load_for_workspace(self, workspace_id: str) -> UserPrefsSchema | None:
        """
        Carga las preferencias de un workspace específico.

        Returns:
            UserPrefsSchema si el archivo existe.
            None si el workspace no tiene preferencias propias —
            el caller usará solo las globales.
        """
        path = self._locator.workspace_preferences_file(workspace_id)

        if not path.exists():
            return None

        try:
            raw_data = self._storage.read_json(path)
            migrated = migrate_preferences(raw_data)

            if migrated.get("schema_version") != raw_data.get("schema_version"):
                self._storage.write_json(path, migrated)

            return UserPrefsSchema.model_validate(migrated)

        except Exception as exc:
            logger.warning(
                "error cargando preferencias del workspace — usando globales",
                workspace_id=workspace_id,
                error=str(exc),
            )
            return None

    def save_for_workspace(
        self,
        workspace_id: str,
        prefs: UserPrefsSchema,
    ) -> None:
        """Persiste las preferencias de un workspace."""
        path = self._locator.workspace_preferences_file(workspace_id)
        self._storage.write_json(path, prefs.model_dump())

        logger.debug(
            "preferencias del workspace guardadas",
            workspace_id=workspace_id,
        )
