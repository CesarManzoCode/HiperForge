"""
CreateWorkspaceUseCase — Crea un nuevo workspace en el sistema.

El workspace es el nivel más alto de organización en HiperForge.
Crear uno correctamente implica más que simplemente escribir un archivo:
hay que garantizar unicidad del nombre, inicializar los directorios,
registrarlo en el índice global y opcionalmente activarlo.

════════════════════════════════════════════════════════════
FLUJO
════════════════════════════════════════════════════════════

  1. Validar que el nombre no está vacío ni tiene caracteres inválidos.
  2. Verificar unicidad del nombre (case-insensitive) entre todos
     los workspaces existentes no eliminados.
  3. Crear la entidad Workspace.create().
  4. Persistir workspace.json + actualizar index.json.
  5. Si es el primer workspace o set_as_active=True → activarlo.
  6. Crear las preferencias del workspace con los defaults.
  7. Devolver WorkspaceSummary a la CLI.

════════════════════════════════════════════════════════════
NOMBRES VÁLIDOS
════════════════════════════════════════════════════════════

  Los nombres de workspace se usan como identificadores humanos
  en la CLI y potencialmente como nombres de directorio en
  algunas integraciones futuras. Por eso imponemos restricciones:

    ✓ "Trabajo"
    ✓ "cliente-acme"
    ✓ "proyectos_personales"
    ✓ "API v2"
    ✗ ""           (vacío)
    ✗ "  "         (solo espacios)
    ✗ "../etc"     (path traversal)
    ✗ "abc/def"    (separador de ruta)
    ✗ nombre > 100 chars (demasiado largo para mostrarse en CLI)

════════════════════════════════════════════════════════════
ACTIVACIÓN AUTOMÁTICA
════════════════════════════════════════════════════════════

  Si es el primer workspace del sistema, se activa automáticamente.
  El usuario no tiene que hacer `workspace switch` después de crear
  el primero — puede empezar a trabajar de inmediato.

  Si ya hay workspaces, el nuevo se crea inactivo por defecto.
  El usuario puede activarlo explícitamente con set_as_active=True
  o con `hiperforge workspace switch <id>` después.
"""

from __future__ import annotations

import re

from hiperforge.application.dto import CreateWorkspaceInput, WorkspaceSummary
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.workspace import Workspace
from hiperforge.domain.exceptions import DuplicateEntity
from hiperforge.memory.schemas.preferences import UserPrefsSchema
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Validación de nombres de workspace
# ---------------------------------------------------------------------------

# Longitud máxima permitida para el nombre
_MAX_NAME_LENGTH = 100

# Patrón de caracteres prohibidos en nombres de workspace
# Prohibimos: separadores de ruta, caracteres de control, null bytes
_INVALID_CHARS_PATTERN = re.compile(r"[/\\<>:\"|?*\x00-\x1f]")

# Nombres reservados que el sistema usa internamente
_RESERVED_NAMES: frozenset[str] = frozenset({
    "default", "system", "admin", "root", "config", "settings",
    "hiperforge", "tmp", "temp", "cache", "logs",
})


class CreateWorkspaceUseCase:
    """
    Crea un nuevo workspace con validaciones completas.

    Parámetros:
        store: MemoryStore para verificar unicidad y persistir.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def execute(
        self,
        input_data: CreateWorkspaceInput,
        *,
        set_as_active: bool = False,
    ) -> WorkspaceSummary:
        """
        Crea el workspace y lo registra en el sistema.

        Parámetros:
            input_data:    Nombre y descripción del workspace.
            set_as_active: Si True, activa el workspace al crearlo.
                           Si False (default), solo se activa automáticamente
                           cuando es el primer workspace del sistema.

        Returns:
            WorkspaceSummary con los datos del workspace creado.

        Raises:
            ValueError:      Si el nombre tiene caracteres inválidos o es demasiado largo.
            DuplicateEntity: Si ya existe un workspace con ese nombre.
        """
        clean_name = input_data.name.strip()

        # ── Paso 1: validar el nombre ──────────────────────────────────
        self._validate_name(clean_name)

        # ── Paso 2: verificar unicidad del nombre ─────────────────────
        # exists_by_name es case-insensitive internamente
        if self._store.workspaces.exists_by_name(clean_name):
            raise DuplicateEntity(
                entity_type="Workspace",
                identifier=clean_name,
            )

        # ── Paso 3: crear la entidad Workspace ────────────────────────
        workspace = Workspace.create(
            name=clean_name,
            description=input_data.description,
        )

        # ── Paso 4: persistir workspace.json + actualizar index.json ──
        # workspace_repo.save() hace ambas cosas atómicamente:
        # escribe workspace.json y agrega el id al index.json
        self._store.workspaces.save(workspace)

        logger.info(
            "workspace creado",
            workspace_id=workspace.id,
            name=workspace.name,
        )

        # ── Paso 5: determinar si activar ─────────────────────────────
        is_first_workspace = self._is_first_workspace(workspace.id)
        should_activate = set_as_active or is_first_workspace

        if should_activate:
            self._store.workspaces.set_active_workspace(workspace.id)
            logger.info(
                "workspace activado",
                workspace_id=workspace.id,
                reason="primer_workspace" if is_first_workspace else "set_as_active=True",
            )

        # ── Paso 6: inicializar preferencias del workspace ────────────
        # Creamos un preferences.json vacío con los defaults.
        # Esto permite que el usuario pueda hacer `workspace config set ...`
        # inmediatamente sin que el sistema tenga que manejar el caso
        # de "preferencias no inicializadas".
        self._initialize_workspace_preferences(workspace.id)

        # ── Paso 7: construir y devolver el summary ───────────────────
        return WorkspaceSummary(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            status=workspace.status.value,
            project_count=0,
            is_active=should_activate,
            created_at=workspace.created_at,
        )

    # ------------------------------------------------------------------
    # Validación del nombre
    # ------------------------------------------------------------------

    def _validate_name(self, name: str) -> None:
        """
        Valida que el nombre del workspace cumple todas las restricciones.

        Raises:
            ValueError: Si el nombre es inválido, con un mensaje descriptivo
                        que indica exactamente qué está mal.
        """
        # Longitud mínima (el DTO ya verifica que no esté vacío,
        # pero verificamos de nuevo después del strip())
        if not name:
            raise ValueError(
                "El nombre del workspace no puede estar vacío ni ser solo espacios."
            )

        # Longitud máxima
        if len(name) > _MAX_NAME_LENGTH:
            raise ValueError(
                f"El nombre del workspace es demasiado largo "
                f"({len(name)} caracteres). Máximo: {_MAX_NAME_LENGTH}."
            )

        # Caracteres inválidos
        invalid_match = _INVALID_CHARS_PATTERN.search(name)
        if invalid_match:
            raise ValueError(
                f"El nombre del workspace contiene el carácter inválido "
                f"'{invalid_match.group()}'. "
                f"Los nombres no pueden contener: / \\ < > : \" | ? * "
                f"ni caracteres de control."
            )

        # Path traversal — nombres como ".." o que empiecen con "."
        if name.startswith("."):
            raise ValueError(
                f"El nombre del workspace no puede empezar con punto: '{name}'."
            )

        # Nombres reservados — los usamos internamente
        if name.lower() in _RESERVED_NAMES:
            raise ValueError(
                f"'{name}' es un nombre reservado por el sistema. "
                f"Elige un nombre diferente."
            )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _is_first_workspace(self, new_workspace_id: str) -> bool:
        """
        Verifica si este es el único workspace en el sistema.

        Se usa para decidir si activarlo automáticamente.
        Carga el índice y verifica si el único ID registrado
        es el del workspace recién creado.
        """
        index = self._store.workspaces.load_index()
        active_workspaces = [
            wid for wid in index.workspace_ids
            if wid != new_workspace_id
        ]
        return len(active_workspaces) == 0

    def _initialize_workspace_preferences(self, workspace_id: str) -> None:
        """
        Crea las preferencias iniciales del workspace con los defaults.

        No lanza excepciones — si falla la inicialización de preferencias,
        el workspace ya existe y es funcional. El sistema usará los defaults
        globales hasta que el usuario configure preferencias propias.
        """
        try:
            default_prefs = UserPrefsSchema()
            self._store.preferences.save_for_workspace(
                workspace_id=workspace_id,
                prefs=default_prefs,
            )
            logger.debug(
                "preferencias del workspace inicializadas",
                workspace_id=workspace_id,
            )
        except Exception as exc:
            # No fatal — el workspace sigue siendo funcional sin preferencias propias
            logger.warning(
                "no se pudieron inicializar las preferencias del workspace",
                workspace_id=workspace_id,
                error=str(exc),
            )