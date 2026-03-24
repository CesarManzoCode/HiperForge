"""
SwitchWorkspaceUseCase — Cambia el workspace activo del sistema.

El workspace activo es el contexto global de trabajo. Todas las operaciones
de HiperForge (crear tasks, proyectos, leer preferencias) operan sobre
el workspace activo por defecto — a menos que se especifique uno explícito.

════════════════════════════════════════════════════════════
RESOLUCIÓN DE WORKSPACE
════════════════════════════════════════════════════════════

  El use case acepta el workspace por ID o por nombre.
  Esto es deliberado — en la CLI es más natural escribir:

    hiperforge workspace switch trabajo

  que recordar y escribir:

    hiperforge workspace switch 01HX4K2J8QNVR0SBPZ1Y3W9D6E

  PRIORIDAD DE RESOLUCIÓN:
    1. Si workspace_id está presente → resolución directa por ID.
       Más eficiente — sin búsqueda en todos los workspaces.
    2. Si workspace_name está presente → búsqueda case-insensitive.
       Itera todos los workspaces hasta encontrar el nombre.
    3. Si ambos están presentes → workspace_id tiene precedencia.
       (El DTO impide que ambos estén presentes simultáneamente
       solo si se pasa exactamente uno — ver validación del DTO.)

════════════════════════════════════════════════════════════
PROTECCIONES
════════════════════════════════════════════════════════════

  No se puede activar un workspace:
    - ARCHIVED: está en modo solo-lectura, no tiene sentido
                como contexto de trabajo activo.
    - DELETED:  eliminación lógica — no debe ser accesible.

  Si el workspace YA es el activo, el use case lo detecta
  y devuelve el summary sin modificar nada — operación idempotente.

════════════════════════════════════════════════════════════
FLUJO
════════════════════════════════════════════════════════════

  1. Resolver el workspace por ID o nombre.
  2. Verificar que el workspace existe y está ACTIVE.
  3. Verificar que no es ya el workspace activo.
  4. Actualizar index.json con el nuevo workspace activo.
  5. Devolver WorkspaceSummary marcado como is_active=True.
"""

from __future__ import annotations

from hiperforge.application.dto import SwitchWorkspaceInput, WorkspaceSummary
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.workspace import Workspace, WorkspaceStatus
from hiperforge.domain.exceptions import EntityNotFound
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)


class SwitchWorkspaceUseCase:
    """
    Cambia el workspace activo del sistema.

    Soporta resolución tanto por ID como por nombre para mayor comodidad
    desde la CLI.

    Parámetros:
        store: MemoryStore para resolver, verificar y activar el workspace.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    def execute(self, input_data: SwitchWorkspaceInput) -> WorkspaceSummary:
        """
        Cambia el workspace activo.

        Parámetros:
            input_data: workspace_id o workspace_name del workspace destino.

        Returns:
            WorkspaceSummary con is_active=True del workspace activado.

        Raises:
            EntityNotFound:  Si el workspace no existe.
            PermissionError: Si el workspace está archivado o eliminado.
        """
        # ── Paso 1: resolver el workspace destino ────────────────────
        workspace = self._resolve_workspace(input_data)

        # ── Paso 2: verificar que puede activarse ────────────────────
        self._assert_can_be_activated(workspace)

        # ── Paso 3: verificar si ya es el activo ─────────────────────
        current_active_id = self._store.get_active_workspace_id()

        if current_active_id == workspace.id:
            logger.info(
                "workspace ya es el activo — sin cambios",
                workspace_id=workspace.id,
                name=workspace.name,
            )
            return self._build_summary(workspace, is_active=True)

        # ── Paso 4: activar el workspace destino ─────────────────────
        self._store.workspaces.set_active_workspace(workspace.id)

        logger.info(
            "workspace activo cambiado",
            previous_id=current_active_id,
            new_id=workspace.id,
            new_name=workspace.name,
        )

        return self._build_summary(workspace, is_active=True)

    # ------------------------------------------------------------------
    # Resolución del workspace
    # ------------------------------------------------------------------

    def _resolve_workspace(self, input_data: SwitchWorkspaceInput) -> Workspace:
        """
        Resuelve el workspace por ID o por nombre según lo que se especificó.

        PRIORIDAD: workspace_id > workspace_name.
        Si se especificaron ambos, workspace_id tiene precedencia.

        Returns:
            Workspace en modo meta (sin proyectos cargados).

        Raises:
            EntityNotFound: Si no se encuentra el workspace.
        """
        if input_data.workspace_id:
            return self._resolve_by_id(input_data.workspace_id)

        # workspace_name está garantizado no-None por la validación del DTO
        return self._resolve_by_name(input_data.workspace_name)  # type: ignore[arg-type]

    def _resolve_by_id(self, workspace_id: str) -> Workspace:
        """
        Resuelve el workspace directamente por su ID.

        Carga en modo meta (sin proyectos) — suficiente para verificar
        estado y construir el summary.

        Raises:
            EntityNotFound: Si el workspace_id no existe.
        """
        try:
            return self._store.workspaces.find_by_id_meta(workspace_id)
        except EntityNotFound:
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

    def _resolve_by_name(self, name: str) -> Workspace:
        """
        Busca un workspace por nombre (case-insensitive).

        Itera todos los workspaces activos y archivados buscando
        el que tenga el nombre especificado.

        Raises:
            EntityNotFound: Si ningún workspace tiene ese nombre.
                            El mensaje incluye los nombres disponibles
                            para ayudar al usuario a corregir el typo.
        """
        name_lower = name.strip().lower()
        all_workspaces = self._store.workspaces.find_all()

        # Búsqueda exacta primero (case-insensitive)
        for ws in all_workspaces:
            if ws.name.lower() == name_lower:
                return ws

        # Búsqueda parcial como fallback — si el usuario escribió un prefix
        # Ejemplo: "trab" cuando el workspace se llama "Trabajo"
        partial_matches = [
            ws for ws in all_workspaces
            if name_lower in ws.name.lower()
        ]

        if len(partial_matches) == 1:
            # Una sola coincidencia parcial — asumimos que es lo que quería
            logger.info(
                "workspace resuelto por coincidencia parcial",
                query=name,
                matched_name=partial_matches[0].name,
                matched_id=partial_matches[0].id,
            )
            return partial_matches[0]

        if len(partial_matches) > 1:
            # Múltiples coincidencias — ambiguo, pedimos más especificidad
            names = ", ".join(f"'{ws.name}'" for ws in partial_matches)
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=name,
            )

        # Sin coincidencias — mostramos los disponibles para ayudar
        available_names = ", ".join(
            f"'{ws.name}'" for ws in all_workspaces
        ) or "(ninguno)"

        raise EntityNotFound(
            entity_type="Workspace",
            entity_id=name,
        )

    # ------------------------------------------------------------------
    # Validación de activación
    # ------------------------------------------------------------------

    def _assert_can_be_activated(self, workspace: Workspace) -> None:
        """
        Verifica que el workspace puede ser activado como contexto de trabajo.

        Los workspaces ARCHIVED y DELETED no pueden ser el workspace activo:
          - ARCHIVED: está en modo solo-lectura. Activarlo permitiría
            crear tasks y proyectos en un workspace que se marcó como
            "pausado" — inconsistente con la intención del usuario.
          - DELETED: eliminación lógica. El workspace no debería
            ser accesible como contexto de trabajo.

        Raises:
            PermissionError: Si el workspace no está ACTIVE.
        """
        if workspace.status == WorkspaceStatus.ARCHIVED:
            raise PermissionError(
                f"No se puede activar el workspace '{workspace.name}' "
                f"porque está archivado. "
                f"Reactívalo primero con: hiperforge workspace reactivate {workspace.id}"
            )

        if workspace.status == WorkspaceStatus.DELETED:
            raise PermissionError(
                f"El workspace '{workspace.name}' fue eliminado "
                f"y no puede activarse."
            )

    # ------------------------------------------------------------------
    # Construcción del output
    # ------------------------------------------------------------------

    def _build_summary(
        self,
        workspace: Workspace,
        is_active: bool,
    ) -> WorkspaceSummary:
        """Construye el WorkspaceSummary para el output de la CLI."""
        return WorkspaceSummary(
            id=workspace.id,
            name=workspace.name,
            description=workspace.description,
            status=workspace.status.value,
            project_count=workspace.project_count,
            is_active=is_active,
            created_at=workspace.created_at,
        )