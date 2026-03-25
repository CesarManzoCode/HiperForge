"""
Utilidad: Manejo de fechas y tiempos

Centraliza todas las operaciones con fechas para garantizar consistencia
en todo el sistema. La regla es simple:

  REGLA DE ORO:
    Todo timestamp interno se maneja en UTC.
    Solo se convierte a hora local en el momento de mostrarlo al usuario en la CLI.
    Nunca guardar timestamps en hora local en el JSON — causa bugs al cambiar
    de zona horaria o en equipos distribuidos.

  REGLA DE SERIALIZACIÓN:
    Todo timestamp se serializa a ISO 8601 con timezone explícito.
    Ejemplo: "2024-01-15T14:32:01.123456+00:00"
    Nunca timestamps sin timezone — son ambiguos y difíciles de debuggear.

¿Por qué un módulo dedicado para esto?
  Sin centralización, cada archivo haría `datetime.now()` (sin UTC),
  o `datetime.utcnow()` (deprecated en Python 3.12), o formatos distintos
  de string. Cuando hay un bug de timezone en producción, es muy difícil
  rastrear de dónde vino.

USO:
  from hiperforge.core.utils.datetime import utcnow, format_duration, parse_iso

  created_at = utcnow()                    # datetime con UTC garantizado
  label = format_duration(125.4)           # "2m 5s"
  dt = parse_iso("2024-01-15T14:32:01Z")  # datetime con UTC
"""

from __future__ import annotations

import time as _time
from datetime import datetime, timedelta, timezone


def utcnow() -> datetime:
    """
    Devuelve el momento actual en UTC con timezone explícito.

    Usar siempre esta función en vez de datetime.now() o datetime.utcnow().

      datetime.now()     → hora local sin timezone. NUNCA usar.
      datetime.utcnow()  → UTC pero sin timezone (naive). Deprecated en 3.12.
      utcnow()           → UTC con timezone. Siempre correcto.

    Returns:
        datetime timezone-aware en UTC.
    """
    return datetime.now(timezone.utc)


def parse_iso(value: str) -> datetime:
    """
    Parsea un string ISO 8601 a datetime con timezone.

    Maneja los formatos más comunes que aparecen en los JSONs:
      "2024-01-15T14:32:01.123456+00:00"  → formato completo con offset
      "2024-01-15T14:32:01Z"              → formato con Z (UTC shorthand)
      "2024-01-15T14:32:01"               → sin timezone (asume UTC)

    Parámetros:
        value: String con fecha en formato ISO 8601.

    Returns:
        datetime timezone-aware en UTC.

    Raises:
        ValueError: Si el string no tiene un formato ISO 8601 reconocible.
    """
    if not value:
        raise ValueError("No se puede parsear un string vacío como fecha")

    # Normalizamos la Z de UTC al offset explícito que fromisoformat entiende
    normalized = value.replace("Z", "+00:00")

    parsed = datetime.fromisoformat(normalized)

    # Si no tiene timezone, asumimos UTC — es nuestra convención
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed


def format_duration(seconds: float) -> str:
    """
    Convierte segundos a una string legible para humanos.

    Usado en la CLI para mostrar cuánto tardó una task, subtask o tool call.

    Ejemplos:
        0.4   → "0.4s"
        5.0   → "5s"
        65.3  → "1m 5s"
        3725  → "1h 2m 5s"
        90061 → "1d 1h 1m 1s"

    Parámetros:
        seconds: Duración en segundos (puede tener decimales).

    Returns:
        String legible. Nunca devuelve string vacío.
    """
    if seconds < 0:
        return "0s"

    # Menos de 60 segundos → mostrar con decimal si es menor a 10s
    if seconds < 10:
        return f"{seconds:.1f}s"

    if seconds < 60:
        return f"{int(seconds)}s"

    # Descomponemos en unidades
    total_seconds = int(seconds)
    days, remainder = divmod(total_seconds, 86_400)
    hours, remainder = divmod(remainder, 3_600)
    minutes, secs = divmod(remainder, 60)

    parts = []
    if days:
        parts.append(f"{days}d")
    if hours:
        parts.append(f"{hours}h")
    if minutes:
        parts.append(f"{minutes}m")
    if secs or not parts:   # siempre mostramos segundos si no hay nada más
        parts.append(f"{secs}s")

    return " ".join(parts)


def format_relative(dt: datetime) -> str:
    """
    Describe cuánto tiempo pasó desde un momento hasta ahora.

    Usado en la CLI para mostrar "hace 5 minutos" en vez de timestamps
    completos que son difíciles de interpretar rápidamente.

    Ejemplos:
        hace 30 segundos  → "hace 30s"
        hace 5 minutos    → "hace 5m"
        hace 2 horas      → "hace 2h"
        hace 3 días       → "hace 3d"
        en el futuro      → "en el futuro"

    Parámetros:
        dt: datetime del evento (debe tener timezone).

    Returns:
        String descriptivo de tiempo relativo.
    """
    now = utcnow()

    # Aseguramos timezone para comparar
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = now - dt

    if delta.total_seconds() < 0:
        return "en el futuro"

    total_seconds = int(delta.total_seconds())

    if total_seconds < 60:
        return f"hace {total_seconds}s"

    minutes = total_seconds // 60
    if minutes < 60:
        return f"hace {minutes}m"

    hours = minutes // 60
    if hours < 24:
        return f"hace {hours}h"

    days = hours // 24
    return f"hace {days}d"


def format_timestamp(dt: datetime, *, include_date: bool = True) -> str:
    """
    Formatea un datetime para mostrarlo en la CLI de forma legible.

    Convierte de UTC a hora local del sistema para que el usuario
    vea su hora, no UTC.

    Ejemplos con include_date=True:  "2024-01-15 14:32:01"
    Ejemplos con include_date=False: "14:32:01"

    Parámetros:
        dt:           datetime a formatear (con o sin timezone).
        include_date: Si True incluye la fecha, si False solo la hora.

    Returns:
        String formateado en hora local del sistema.
    """
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    # Convertimos a hora local para mostrar al usuario
    local_dt = dt.astimezone()

    if include_date:
        return local_dt.strftime("%Y-%m-%d %H:%M:%S")
    return local_dt.strftime("%H:%M:%S")


def seconds_since(dt: datetime | float) -> float:
    """
    Calcula cuántos segundos pasaron desde un momento hasta ahora.

    Útil para medir duración de operaciones en curso.

    Parámetros:
        dt: datetime del inicio (debe tener timezone), o float de time.monotonic().

    Returns:
        Segundos transcurridos. Negativo si dt está en el futuro.
    """
    if isinstance(dt, float):
        return round(_time.monotonic() - dt, 3)

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    delta = utcnow() - dt
    return round(delta.total_seconds(), 3)
