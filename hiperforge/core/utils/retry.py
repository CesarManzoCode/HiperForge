"""
Utilidad: Reintentos con backoff exponencial

El agente ReAct interactúa constantemente con servicios externos:
  - APIs de LLMs (Anthropic, OpenAI, Ollama)
  - Sistema de archivos
  - Comandos del sistema operativo

Estos servicios fallan de forma transitoria con frecuencia:
  - Rate limits del LLM: "demasiadas requests, espera X segundos"
  - Timeouts de red: conexión cortada momentáneamente
  - File locks: otro proceso tiene el archivo bloqueado

La solución no es explotar con error — es reintentar inteligentemente.

¿QUÉ ES BACKOFF EXPONENCIAL?
  En vez de reintentar inmediatamente (lo que empeoraría un rate limit),
  esperamos cada vez más tiempo entre intentos:

  Intento 1 falla → esperar 1s  → Intento 2
  Intento 2 falla → esperar 2s  → Intento 3
  Intento 3 falla → esperar 4s  → Intento 4
  Intento 4 falla → esperar 8s  → lanzar excepción final

  El tiempo se duplica en cada intento (exponencial).
  Esto da tiempo al servicio de recuperarse sin bombardearlo.

  Con jitter (ruido aleatorio pequeño) evitamos que múltiples
  instancias del agente reintenten exactamente al mismo tiempo.

USO:
  # Como decorador en métodos de un adapter
  @retry(max_attempts=3, retryable_exceptions=(LLMRateLimitError,))
  def complete(self, messages):
      ...

  # Como función directa para casos más específicos
  result = retry_call(
      fn=lambda: self._client.complete(messages),
      max_attempts=3,
      retryable_exceptions=(LLMRateLimitError, LLMConnectionError),
      on_retry=lambda attempt, error, wait: logger.warning(f"Reintento {attempt}"),
  )
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from functools import wraps
from typing import Any, TypeVar

from hiperforge.core.constants import REACT_MAX_TOOL_RETRIES, REACT_RETRY_DELAY_SECONDS

# TypeVar para preservar el tipo de retorno en el decorador
F = TypeVar("F", bound=Callable[..., Any])


def retry(
    *,
    max_attempts: int = REACT_MAX_TOOL_RETRIES,
    base_delay: float = REACT_RETRY_DELAY_SECONDS,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
) -> Callable[[F], F]:
    """
    Decorador que agrega reintentos con backoff exponencial a una función.

    Parámetros:
        max_attempts:          Número máximo de intentos (incluyendo el primero).
                               Default: REACT_MAX_TOOL_RETRIES (3).
        base_delay:            Segundos de espera después del primer fallo.
                               Default: REACT_RETRY_DELAY_SECONDS (2.0s).
        max_delay:             Techo del tiempo de espera. El backoff no
                               crecerá más allá de este valor. Default: 30s.
        backoff_factor:        Multiplicador del delay en cada intento.
                               Default: 2.0 (duplica cada vez).
        jitter:                Si True, agrega ruido aleatorio de ±20% al delay.
                               Evita que múltiples agentes reintenten simultáneamente.
        retryable_exceptions:  Solo reintenta si el error es de estos tipos.
                               Errores de otros tipos se propagan inmediatamente.
                               Default: (Exception,) — reintenta cualquier error.

    Ejemplo:
        @retry(
            max_attempts=4,
            retryable_exceptions=(LLMRateLimitError, LLMConnectionError),
        )
        def complete(self, messages: list[Message]) -> LLMResponse:
            return self._client.messages.create(...)
    """
    def decorator(fn: F) -> F:
        @wraps(fn)  # preserva __name__, __doc__, etc. de la función original
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            return retry_call(
                fn=lambda: fn(*args, **kwargs),
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                backoff_factor=backoff_factor,
                jitter=jitter,
                retryable_exceptions=retryable_exceptions,
            )
        return wrapper  # type: ignore[return-value]
    return decorator


def retry_call(
    fn: Callable[[], Any],
    *,
    max_attempts: int = REACT_MAX_TOOL_RETRIES,
    base_delay: float = REACT_RETRY_DELAY_SECONDS,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
    retryable_exceptions: tuple[type[Exception], ...] = (Exception,),
    on_retry: Callable[[int, Exception, float], None] | None = None,
) -> Any:
    """
    Ejecuta una función con reintentos y backoff exponencial.

    Versión funcional del decorador @retry — útil cuando no podemos
    decorar la función directamente (lambdas, funciones de terceros, etc.).

    Parámetros:
        fn:                    Función sin argumentos a ejecutar.
                               Usar lambda si necesitas pasar argumentos:
                               retry_call(fn=lambda: do_thing(arg1, arg2))
        max_attempts:          Número máximo de intentos.
        base_delay:            Delay inicial en segundos.
        max_delay:             Delay máximo en segundos.
        backoff_factor:        Multiplicador del delay por intento.
        jitter:                Si True agrega ruido aleatorio al delay.
        retryable_exceptions:  Tipos de excepción que activan el reintento.
        on_retry:              Callback opcional llamado antes de cada reintento.
                               Recibe (número_intento, error, segundos_de_espera).
                               Útil para logging: logger.warning(f"Reintento {n}...")

    Returns:
        El valor de retorno de fn() en el primer intento exitoso.

    Raises:
        La última excepción recibida si se agotan todos los intentos.
        Cualquier excepción no retryable se propaga inmediatamente.
    """
    last_exception: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        try:
            return fn()

        except retryable_exceptions as error:
            last_exception = error
            is_last_attempt = attempt == max_attempts

            if is_last_attempt:
                # Agotamos todos los intentos — propagamos el último error
                raise

            # Calculamos el delay para este intento con backoff exponencial
            # Fórmula: base_delay * (backoff_factor ^ (attempt - 1))
            delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)

            # Jitter: ±20% del delay para evitar thundering herd
            if jitter:
                delay = delay * random.uniform(0.8, 1.2)

            # Si el error trae información de cuánto esperar (rate limit),
            # respetamos ese valor si es mayor que nuestro backoff calculado
            retry_after = getattr(error, "retry_after_seconds", None)
            if retry_after is not None and retry_after > delay:
                delay = retry_after

            # Notificamos al caller antes de esperar (para logging)
            if on_retry is not None:
                on_retry(attempt, error, delay)

            time.sleep(delay)

        except Exception:
            # Error no retryable — propagamos inmediatamente sin reintentar
            raise

    # Este punto es inalcanzable (el raise dentro del loop siempre se ejecuta
    # en el último intento), pero satisface al type checker
    raise last_exception  # type: ignore[misc]


def calculate_backoff_delay(
    attempt: int,
    *,
    base_delay: float = REACT_RETRY_DELAY_SECONDS,
    max_delay: float = 30.0,
    backoff_factor: float = 2.0,
    jitter: bool = True,
) -> float:
    """
    Calcula el delay para un intento específico sin ejecutar ningún sleep.

    Útil para:
      - Tests: verificar que los delays se calculan correctamente.
      - Logging: mostrar al usuario cuánto va a esperar antes de dormir.
      - UI: mostrar una barra de progreso durante la espera.

    Parámetros:
        attempt: Número de intento (1-indexed). Intento 1 = base_delay.

    Returns:
        Segundos de espera calculados para ese intento.
    """
    delay = min(base_delay * (backoff_factor ** (attempt - 1)), max_delay)

    if jitter:
        delay = delay * random.uniform(0.8, 1.2)

    return round(delay, 3)