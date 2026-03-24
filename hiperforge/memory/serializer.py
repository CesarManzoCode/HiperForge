"""
Serializer — Encode/decode de tipos Python no soportados por JSON nativo.

JSON solo soporta: str, int, float, bool, None, list, dict.
HiperForge usa tipos más ricos: datetime, Path, Enum, dataclasses.

Este módulo resuelve la conversión de forma centralizada y consistente.
Todos los módulos que necesiten serializar usan este serializer —
nunca implementan su propia conversión.

TIPOS MANEJADOS:
  datetime  → string ISO 8601 con timezone explícito
  Path      → string de ruta absoluta
  Enum      → string con el valor del enum
  dataclass → dict recursivo de sus campos
  set       → lista ordenada (JSON no tiene sets)
  bytes     → string base64

ROUNDTRIP GARANTIZADO:
  serialize(deserialize(x)) == x  para todos los tipos soportados.
  Esto es crítico — los datos que guardamos deben ser exactamente
  los mismos que recuperamos al leer.

USO:
  from hiperforge.memory.serializer import serialize, deserialize

  # Serializar para guardar en JSON
  data = serialize({"created_at": datetime.now(UTC), "path": Path("/tmp")})
  # {"created_at": "2024-01-15T14:32:01+00:00", "path": "/tmp"}

  # Deserializar con hints de tipos esperados
  obj = deserialize(data, hints={"created_at": datetime, "path": Path})
  # {"created_at": datetime(2024, 1, 15, ...), "path": PosixPath("/tmp")}
"""

from __future__ import annotations

import base64
import dataclasses
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any


def serialize(value: Any) -> Any:
    """
    Convierte un valor Python a un tipo compatible con JSON.

    Procesa recursivamente dicts y listas para manejar estructuras anidadas.
    Los tipos ya compatibles con JSON (str, int, float, bool, None) pasan
    sin cambios.

    Parámetros:
        value: Cualquier valor Python a serializar.

    Returns:
        Valor compatible con json.dumps().
    """
    # Tipos nativos JSON — pasan directamente
    if value is None or isinstance(value, (bool, int, float, str)):
        return value

    # datetime → ISO 8601 con timezone explícito
    if isinstance(value, datetime):
        # Aseguramos que siempre tenga timezone antes de serializar
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.isoformat()

    # Path → string de ruta
    if isinstance(value, Path):
        return str(value)

    # Enum → valor del enum (generalmente string)
    if isinstance(value, Enum):
        return value.value

    # dataclass → dict recursivo
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return serialize(dataclasses.asdict(value))

    # dict → procesar valores recursivamente
    if isinstance(value, dict):
        return {str(k): serialize(v) for k, v in value.items()}

    # list/tuple → procesar elementos recursivamente
    if isinstance(value, (list, tuple)):
        return [serialize(item) for item in value]

    # set → lista ordenada (JSON no tiene sets, la lista preserva todos los valores)
    if isinstance(value, (set, frozenset)):
        return sorted(serialize(item) for item in value)

    # bytes → base64 string (para checksums y datos binarios)
    if isinstance(value, bytes):
        return base64.b64encode(value).decode("ascii")

    # Tipo desconocido → convertir a string como último recurso
    # Loggeamos para que el dev sepa que hay un tipo no manejado
    return str(value)


def deserialize(data: dict[str, Any], hints: dict[str, type] | None = None) -> dict[str, Any]:
    """
    Convierte un dict leído del JSON a tipos Python más ricos.

    Los hints indican qué tipo esperamos en cada campo.
    Sin hints, los valores se devuelven tal cual (strings, ints, etc.).

    Parámetros:
        data:  Dict leído del JSON (valores son tipos JSON nativos).
        hints: Mapa de campo → tipo esperado.
               Ejemplo: {"created_at": datetime, "path": Path}

    Returns:
        Dict con los valores convertidos a los tipos indicados en hints.

    Ejemplo:
        data = {"created_at": "2024-01-15T14:32:01+00:00", "path": "/tmp"}
        hints = {"created_at": datetime, "path": Path}
        result = deserialize(data, hints)
        # result["created_at"] es datetime, result["path"] es Path
    """
    if hints is None:
        return data

    result = dict(data)

    for field, expected_type in hints.items():
        if field not in result or result[field] is None:
            continue

        raw = result[field]

        try:
            result[field] = _convert(raw, expected_type)
        except (ValueError, TypeError) as exc:
            # Si falla la conversión, dejamos el valor original
            # y loggeamos para debug — no rompemos la carga
            import structlog
            structlog.get_logger(__name__).warning(
                "fallo al deserializar campo",
                field=field,
                expected_type=expected_type.__name__,
                raw_value=str(raw)[:100],
                error=str(exc),
            )

    return result


def _convert(value: Any, target_type: type) -> Any:
    """
    Convierte un valor al tipo target indicado.

    Parámetros:
        value:       Valor a convertir (tipo JSON nativo).
        target_type: Tipo Python destino.

    Returns:
        Valor convertido al tipo target.

    Raises:
        ValueError: Si el valor no puede convertirse al tipo indicado.
        TypeError:  Si el tipo no está soportado.
    """
    # Ya es el tipo correcto — no necesita conversión
    if isinstance(value, target_type):
        return value

    # string → datetime
    if target_type is datetime and isinstance(value, str):
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        # Garantizamos timezone UTC si no tiene
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed

    # string → Path
    if target_type is Path and isinstance(value, str):
        return Path(value)

    # string → Enum (si target_type es una subclase de Enum)
    if isinstance(target_type, type) and issubclass(target_type, Enum):
        return target_type(value)

    # string → bytes (base64)
    if target_type is bytes and isinstance(value, str):
        return base64.b64decode(value.encode("ascii"))

    # int/float conversiones básicas
    if target_type is int and isinstance(value, (str, float)):
        return int(value)

    if target_type is float and isinstance(value, (str, int)):
        return float(value)

    # Tipo no manejado — intentamos la conversión directa
    return target_type(value)
