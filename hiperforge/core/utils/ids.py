"""
Utilidad: Generación de IDs únicos

Todas las entidades del sistema (Workspace, Project, Task, Subtask, ToolCall)
usan IDs generados por este módulo. Centralizar aquí significa que si alguna
vez necesitamos cambiar el formato de IDs, lo cambiamos en un solo lugar.

¿Por qué ULID y no UUID4?
  UUID4:  "f47ac10b-58cc-4372-a567-0e02b2c3d479"
  ULID:   "01HX4KM2Q3R5T6V7W8X9Y0Z1A2"

  ULID tiene dos ventajas críticas para este proyecto:

  1. ORDENABLE CRONOLÓGICAMENTE:
     Los primeros 10 caracteres del ULID son un timestamp.
     Esto significa que si ordenas IDs alfabéticamente,
     también los ordenas por fecha de creación — sin necesidad
     de un campo `created_at` extra para ordenar en disco.

  2. MÁS LEGIBLE EN LOGS Y TERMINAL:
     Un ULID es más corto y no tiene guiones — más fácil de
     copiar/pegar cuando un dev está debuggeando en la terminal.

  La desventaja: necesita la librería `python-ulid`.
  El fallback a UUID4 existe para cuando el dev no la tiene instalada.

USO:
  from hiperforge.core.utils.ids import generate_id, is_valid_id

  task_id = generate_id()           # "01HX4KM2Q3R5T6V7W8X9Y0Z1A2"
  is_valid = is_valid_id(task_id)   # True
"""

from __future__ import annotations

import re
import uuid


# ---------------------------------------------------------------------------
# Intentamos usar ULID. Si no está disponible, usamos UUID4 como fallback.
# Esto evita que el proyecto explote en entornos sin python-ulid instalado.
# ---------------------------------------------------------------------------

try:
    from ulid import ULID
    _ULID_AVAILABLE = True
except ImportError:
    _ULID_AVAILABLE = False


# Patrón para validar ULIDs (26 caracteres en base32 Crockford)
_ULID_PATTERN = re.compile(r"^[0-9A-Z]{26}$")

# Patrón para validar UUID4 (fallback)
_UUID4_PATTERN = re.compile(
    r"^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$"
)


def generate_id() -> str:
    """
    Genera un ID único para una entidad del sistema.

    Usa ULID si está disponible, UUID4 como fallback.
    El formato devuelto depende de qué librería esté instalada,
    pero siempre es un string no vacío y globalmente único.

    Returns:
        String con el ID generado. Ejemplo: "01HX4KM2Q3R5T6V7W8X9Y0Z1A2"
    """
    if _ULID_AVAILABLE:
        return str(ULID())
    # Fallback a UUID4 — menos legible pero igual de único
    return str(uuid.uuid4())


def is_valid_id(value: str) -> bool:
    """
    Verifica si un string tiene el formato válido de un ID del sistema.

    Acepta tanto ULID como UUID4 para manejar sistemas que migraron
    de un formato al otro sin romper IDs existentes.

    Parámetros:
        value: String a validar.

    Returns:
        True si el formato es válido (ULID o UUID4).
        False si es vacío, None, o tiene formato incorrecto.
    """
    if not value or not isinstance(value, str):
        return False

    return bool(_ULID_PATTERN.match(value)) or bool(_UUID4_PATTERN.match(value))


def generate_session_id() -> str:
    """
    Genera un ID específico para sesiones.

    Funcionalmente idéntico a generate_id() pero semánticamente
    distinto — deja claro en el código que este ID es para una sesión.

    Útil si en el futuro queremos prefijos por tipo de entidad
    (ej: "ses_01HX4K..." vs "tsk_01HX4K...") sin cambiar los callers.
    """
    return generate_id()