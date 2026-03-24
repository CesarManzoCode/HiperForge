"""
ManagePrefsUseCase — Lee y actualiza preferencias del sistema.

Las preferencias de HiperForge operan en dos niveles con cascada:

  NIVEL GLOBAL   (~/.hiperforge/preferences.json)
    Se aplican a todos los workspaces que no las sobreescriban.
    Ejemplo: cambiar el proveedor LLM globalmente.

  NIVEL WORKSPACE (~/.hiperforge/workspaces/{id}/preferences.json)
    Sobreescriben las globales para ese workspace específico.
    Ejemplo: usar Groq para el workspace "Trabajo" y Anthropic para el resto.

════════════════════════════════════════════════════════════
OPERACIONES DISPONIBLES
════════════════════════════════════════════════════════════

  get_effective(workspace_id)
    → Devuelve las preferencias efectivas con cascada aplicada.
      Si workspace_id es None, devuelve solo las globales.
      Si workspace_id está presente, combina global + workspace.

  get_global()
    → Solo las preferencias globales sin cascada.

  get_for_workspace(workspace_id)
    → Solo las preferencias del workspace específico (sin globals).
      None si el workspace no tiene preferencias propias.

  update(input_data)
    → Actualiza campos específicos usando notación de punto.
      Ejemplo: {"llm.provider": "groq", "ui.verbose": True}
      Solo los campos indicados cambian — el resto permanece igual.

  reset(workspace_id)
    → Elimina las preferencias del workspace (vuelve a usar las globales).
      Para globales: resetea todo a los defaults del código.

════════════════════════════════════════════════════════════
NOTACIÓN DE PUNTO PARA ACTUALIZACIONES
════════════════════════════════════════════════════════════

  La notación de punto permite actualizar campos anidados sin
  tener que especificar el objeto completo:

    {"llm.provider": "groq"}
    → prefs.llm.provider = "groq"

    {"agent.max_react_iterations": 20, "ui.verbose": True}
    → prefs.agent.max_react_iterations = 20
    → prefs.ui.verbose = True

  Campos soportados:
    llm.provider          → proveedor del LLM
    llm.model             → modelo específico (None = default del proveedor)
    llm.temperature       → temperatura de generación (0.0 - 2.0)
    llm.max_tokens        → máximo de tokens en respuesta
    agent.max_react_iterations → iteraciones máximas del loop ReAct
    agent.max_subtasks    → subtasks máximas en un plan
    agent.tool_timeout_seconds → timeout por defecto para tools
    agent.auto_confirm_plan → ejecutar sin confirmación del usuario
    agent.show_reasoning  → mostrar razonamiento del agente
    ui.show_token_usage   → mostrar tokens y costo al terminar
    ui.show_timestamps    → mostrar timestamps en los logs
    ui.verbose            → modo detallado en la terminal

════════════════════════════════════════════════════════════
VALIDACIÓN DE ACTUALIZACIONES
════════════════════════════════════════════════════════════

  Antes de persistir, validamos que:
    - Las claves existen en el schema (no se permiten campos desconocidos)
    - Los valores son del tipo correcto
    - Los valores están en los rangos permitidos

  Si cualquier validación falla, NO se persiste nada — la operación
  es atómica: o todos los campos se actualizan o ninguno.
"""

from __future__ import annotations

from typing import Any

from hiperforge.application.dto import UpdatePreferencesInput
from hiperforge.core.logging import get_logger
from hiperforge.domain.exceptions import EntityNotFound
from hiperforge.memory.schemas.preferences import UserPrefsSchema
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Mapa de campos válidos en notación de punto
#
# Estructura: "ruta.de.punto" → (tipo_esperado, descripción)
# Usamos esto para validar las claves antes de aplicar updates.
# ---------------------------------------------------------------------------
_VALID_PREF_FIELDS: dict[str, tuple[type, str]] = {
    # ── LLM ────────────────────────────────────────────────────────────
    "llm.provider":      (str,   "Proveedor del LLM: anthropic, openai, groq, ollama"),
    "llm.model":         (str,   "Modelo específico del LLM (o null para usar el default)"),
    "llm.temperature":   (float, "Temperatura de generación entre 0.0 y 2.0"),
    "llm.max_tokens":    (int,   "Máximo de tokens en la respuesta del LLM"),

    # ── Agent ───────────────────────────────────────────────────────────
    "agent.max_react_iterations": (int,   "Iteraciones máximas del loop ReAct por subtask (1-50)"),
    "agent.max_subtasks":         (int,   "Subtasks máximas en un plan (1-50)"),
    "agent.tool_timeout_seconds": (float, "Timeout por defecto para tools en segundos"),
    "agent.auto_confirm_plan":    (bool,  "Ejecutar el plan sin confirmación del usuario"),
    "agent.show_reasoning":       (bool,  "Mostrar el razonamiento del agente en la terminal"),

    # ── UI ──────────────────────────────────────────────────────────────
    "ui.show_token_usage":  (bool, "Mostrar uso de tokens y costo al terminar"),
    "ui.show_timestamps":   (bool, "Mostrar timestamps en los logs de la terminal"),
    "ui.verbose":           (bool, "Modo verbose: más detalle en la terminal"),
}

# Proveedores LLM válidos
_VALID_LLM_PROVIDERS: frozenset[str] = frozenset({
    "anthropic", "openai", "groq", "ollama"
})


class ManagePrefsUseCase:
    """
    Lee y actualiza preferencias del sistema a nivel global o de workspace.

    Parámetros:
        store: MemoryStore para acceder al repositorio de preferencias.
    """

    def __init__(self, store: MemoryStore) -> None:
        self._store = store

    # ------------------------------------------------------------------
    # Lectura
    # ------------------------------------------------------------------

    def get_effective(
        self,
        workspace_id: str | None = None,
    ) -> UserPrefsSchema:
        """
        Devuelve las preferencias efectivas con la cascada aplicada.

        Cascada: defaults del código → globales → workspace.
        El workspace sobreescribe las globales.

        Parámetros:
            workspace_id: Workspace del que obtener las efectivas.
                          None = solo las globales (sin cascada).

        Returns:
            UserPrefsSchema con la configuración efectiva final.
        """
        return self._store.get_effective_preferences(workspace_id)

    def get_global(self) -> UserPrefsSchema:
        """
        Devuelve solo las preferencias globales sin aplicar cascada.

        Útil para mostrar qué hay configurado a nivel global
        independientemente de los workspaces.
        """
        return self._store.preferences.load_global()

    def get_for_workspace(
        self,
        workspace_id: str,
    ) -> UserPrefsSchema | None:
        """
        Devuelve las preferencias específicas de un workspace.

        NO aplica cascada — devuelve solo lo que está en el
        preferences.json del workspace, sin combinar con globales.

        Returns:
            UserPrefsSchema si el workspace tiene preferencias propias.
            None si el workspace usa solo los defaults globales.
        """
        self._verify_workspace_exists(workspace_id)
        return self._store.preferences.load_for_workspace(workspace_id)

    # ------------------------------------------------------------------
    # Actualización
    # ------------------------------------------------------------------

    def update(self, input_data: UpdatePreferencesInput) -> UserPrefsSchema:
        """
        Actualiza campos de preferencias usando notación de punto.

        ATOMICIDAD:
          Si cualquier campo tiene un valor inválido, no se persiste
          nada — la operación falla completamente antes de tocar disco.
          Esto evita estados parcialmente actualizados.

        Parámetros:
            input_data: Contiene los updates en notación de punto y
                        el workspace_id opcional (None = globales).

        Returns:
            UserPrefsSchema con los valores actualizados y aplicados.

        Raises:
            ValueError:     Si alguna clave es desconocida o el valor
                            es del tipo incorrecto o fuera de rango.
            EntityNotFound: Si workspace_id no existe.
        """
        # ── Verificar workspace si aplica ─────────────────────────────
        if input_data.workspace_id:
            self._verify_workspace_exists(input_data.workspace_id)

        # ── Validar TODOS los updates antes de aplicar ninguno ────────
        # Atómico: falla todo o nada
        validation_errors = self._validate_updates(input_data.updates)
        if validation_errors:
            error_list = "\n".join(f"  • {e}" for e in validation_errors)
            raise ValueError(
                f"Los siguientes campos tienen valores inválidos:\n{error_list}"
            )

        # ── Cargar las preferencias actuales ──────────────────────────
        if input_data.workspace_id:
            current = (
                self._store.preferences.load_for_workspace(input_data.workspace_id)
                or UserPrefsSchema()
            )
        else:
            current = self._store.preferences.load_global()

        # ── Aplicar los updates sobre el dict actual ──────────────────
        current_dict = current.model_dump()
        self._apply_dot_notation_updates(current_dict, input_data.updates)

        # ── Validar el resultado final con Pydantic ───────────────────
        # La validación de tipos y rangos de Pydantic actúa como
        # red de seguridad final antes de persistir
        try:
            updated_prefs = UserPrefsSchema.model_validate(current_dict)
        except Exception as exc:
            raise ValueError(
                f"La configuración resultante es inválida: {exc}"
            ) from exc

        # ── Persistir ─────────────────────────────────────────────────
        if input_data.workspace_id:
            self._store.preferences.save_for_workspace(
                workspace_id=input_data.workspace_id,
                prefs=updated_prefs,
            )
        else:
            self._store.preferences.save_global(updated_prefs)

        logger.info(
            "preferencias actualizadas",
            level="workspace" if input_data.workspace_id else "global",
            workspace_id=input_data.workspace_id,
            fields_updated=list(input_data.updates.keys()),
        )

        return updated_prefs

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(
        self,
        workspace_id: str | None = None,
    ) -> UserPrefsSchema:
        """
        Resetea las preferencias a los valores por defecto del código.

        Si workspace_id está presente:
          Elimina las preferencias del workspace — vuelve a heredar
          solo de las globales. El workspace queda sin sobreescrituras.

        Si workspace_id es None (globales):
          Reemplaza las preferencias globales con los defaults
          definidos en el código (UserPrefsSchema() sin argumentos).

        Returns:
            UserPrefsSchema con los defaults después del reset.
        """
        default_prefs = UserPrefsSchema()

        if workspace_id:
            self._verify_workspace_exists(workspace_id)
            self._store.preferences.save_for_workspace(
                workspace_id=workspace_id,
                prefs=default_prefs,
            )
            logger.info(
                "preferencias del workspace reseteadas a defaults",
                workspace_id=workspace_id,
            )
        else:
            self._store.preferences.save_global(default_prefs)
            logger.info("preferencias globales reseteadas a defaults")

        return default_prefs

    # ------------------------------------------------------------------
    # Introspección del schema
    # ------------------------------------------------------------------

    def list_available_fields(self) -> dict[str, dict[str, str]]:
        """
        Devuelve todos los campos configurables con sus tipos y descripciones.

        Útil para que la CLI pueda mostrar ayuda sobre qué campos
        se pueden configurar con `hiperforge config set`.

        Returns:
            Dict de campo → {"type": tipo_str, "description": descripción}
        """
        result: dict[str, dict[str, str]] = {}

        for field_path, (field_type, description) in _VALID_PREF_FIELDS.items():
            result[field_path] = {
                "type": field_type.__name__,
                "description": description,
            }

        return result

    # ------------------------------------------------------------------
    # Validación de updates
    # ------------------------------------------------------------------

    def _validate_updates(self, updates: dict[str, Any]) -> list[str]:
        """
        Valida todas las claves y valores del dict de updates.

        Returns:
            Lista de strings de error. Vacía = todo válido.
        """
        errors: list[str] = []

        for key, value in updates.items():
            # ── Verificar que el campo existe ─────────────────────────
            if key not in _VALID_PREF_FIELDS:
                similar = self._suggest_similar_field(key)
                suggestion = f" ¿Quisiste decir '{similar}'?" if similar else ""
                errors.append(
                    f"Campo desconocido: '{key}'.{suggestion} "
                    f"Usa 'hiperforge config fields' para ver los campos disponibles."
                )
                continue

            expected_type, description = _VALID_PREF_FIELDS[key]

            # ── Verificar el tipo ─────────────────────────────────────
            # Para None (limpiar un campo opcional como llm.model), siempre válido
            if value is None:
                continue

            # Coerción numérica: int puede recibir float si es entero
            if expected_type is int and isinstance(value, float):
                if value != int(value):
                    errors.append(
                        f"'{key}' requiere un entero, recibido float: {value}"
                    )
                    continue
            elif expected_type is float and isinstance(value, int):
                pass  # int es válido donde se espera float
            elif not isinstance(value, expected_type):
                errors.append(
                    f"'{key}' requiere {expected_type.__name__}, "
                    f"recibido {type(value).__name__}: {value!r}"
                )
                continue

            # ── Validaciones de rango y dominio ───────────────────────
            field_errors = self._validate_field_value(key, value)
            errors.extend(field_errors)

        return errors

    def _validate_field_value(self, key: str, value: Any) -> list[str]:
        """
        Valida que el valor de un campo está dentro de los rangos permitidos.

        Returns:
            Lista de errores específicos para este campo.
        """
        errors: list[str] = []

        if key == "llm.provider":
            if value not in _VALID_LLM_PROVIDERS:
                valid_str = ", ".join(sorted(_VALID_LLM_PROVIDERS))
                errors.append(
                    f"Proveedor LLM inválido: '{value}'. "
                    f"Opciones válidas: {valid_str}"
                )

        elif key == "llm.temperature":
            if not (0.0 <= float(value) <= 2.0):
                errors.append(
                    f"'llm.temperature' debe estar entre 0.0 y 2.0, recibido: {value}"
                )

        elif key == "llm.max_tokens":
            int_val = int(value)
            if int_val <= 0:
                errors.append(
                    f"'llm.max_tokens' debe ser mayor que 0, recibido: {int_val}"
                )
            elif int_val > 200_000:
                errors.append(
                    f"'llm.max_tokens' excede el límite razonable de 200,000: {int_val}"
                )

        elif key == "agent.max_react_iterations":
            int_val = int(value)
            if not (1 <= int_val <= 50):
                errors.append(
                    f"'agent.max_react_iterations' debe estar entre 1 y 50, recibido: {int_val}"
                )

        elif key == "agent.max_subtasks":
            int_val = int(value)
            if not (1 <= int_val <= 50):
                errors.append(
                    f"'agent.max_subtasks' debe estar entre 1 y 50, recibido: {int_val}"
                )

        elif key == "agent.tool_timeout_seconds":
            float_val = float(value)
            if float_val <= 0:
                errors.append(
                    f"'agent.tool_timeout_seconds' debe ser mayor que 0, recibido: {float_val}"
                )
            elif float_val > 600:
                errors.append(
                    f"'agent.tool_timeout_seconds' excede el máximo de 600 segundos: {float_val}"
                )

        return errors

    # ------------------------------------------------------------------
    # Aplicación de updates con notación de punto
    # ------------------------------------------------------------------

    @staticmethod
    def _apply_dot_notation_updates(
        data: dict[str, Any],
        updates: dict[str, Any],
    ) -> None:
        """
        Aplica updates con notación de punto sobre un dict anidado in-place.

        Ejemplo:
          data = {"llm": {"provider": "anthropic", "temperature": 0.2}}
          updates = {"llm.provider": "groq"}
          → data["llm"]["provider"] = "groq"
          → data["llm"]["temperature"] permanece 0.2

        La notación de punto permite actualizar un campo específico
        dentro de un objeto anidado sin tener que especificar todos
        los demás campos del mismo objeto.
        """
        for key, value in updates.items():
            parts = key.split(".")
            target = data

            # Navegar hasta el penúltimo nivel del dict
            for part in parts[:-1]:
                if part not in target:
                    target[part] = {}
                elif not isinstance(target[part], dict):
                    # El campo intermedio no es un dict — lo reemplazamos
                    target[part] = {}
                target = target[part]

            # Coerción de tipos para campos numéricos
            final_key = parts[-1]
            field_path = ".".join(parts)

            if field_path in _VALID_PREF_FIELDS:
                expected_type, _ = _VALID_PREF_FIELDS[field_path]
                if value is not None:
                    if expected_type is int and isinstance(value, float):
                        value = int(value)
                    elif expected_type is float and isinstance(value, int):
                        value = float(value)

            target[final_key] = value

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _verify_workspace_exists(self, workspace_id: str) -> None:
        """
        Verifica que el workspace existe en disco.

        Raises:
            EntityNotFound: Si no existe.
        """
        from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator

        if not self._store.workspaces.load_index().workspace_ids.__contains__(workspace_id):
            raise EntityNotFound(
                entity_type="Workspace",
                entity_id=workspace_id,
            )

    @staticmethod
    def _suggest_similar_field(key: str) -> str | None:
        """
        Sugiere un campo similar cuando se escribe uno inválido.

        Usa coincidencia de substring para sugerir el campo más parecido.
        Ejemplo: "llm.privder" → sugiere "llm.provider"

        Returns:
            Nombre del campo más similar, o None si no hay sugerencia clara.
        """
        key_lower = key.lower()
        candidates: list[tuple[float, str]] = []

        for valid_field in _VALID_PREF_FIELDS:
            # Calculamos similitud simple como proporción de chars en común
            common = sum(
                1 for c in key_lower
                if c in valid_field.lower()
            )
            if common > 0:
                similarity = common / max(len(key_lower), len(valid_field))
                candidates.append((similarity, valid_field))

        if not candidates:
            return None

        # El campo con mayor similitud
        best_similarity, best_field = max(candidates, key=lambda x: x[0])

        # Solo sugerimos si hay una similitud razonable
        return best_field if best_similarity > 0.5 else None