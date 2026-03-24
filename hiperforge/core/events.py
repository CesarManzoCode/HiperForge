"""
EventBus — Sistema de eventos del agente ReAct.

El EventBus es el tejido conectivo entre el loop ReAct y el resto del sistema.
Cuando el agente ejecuta una acción, dispara un evento. Los listeners
reaccionan a ese evento de forma desacoplada — el agente no sabe ni le importa
quién escucha.

DISEÑO: SÍNCRONO CON AISLAMIENTO DE ERRORES
  El loop ReAct es síncrono por naturaleza — razonar → actuar → observar
  es una secuencia estricta donde cada paso depende del anterior.

  El EventBus es síncrono para mantener esa consistencia: cuando el agente
  dispara "subtask_completada", tiene la garantía de que la CLI ya actualizó
  el progreso y el logger ya registró el evento ANTES de continuar al siguiente
  paso. Sin esto, podría haber una ventana donde el usuario ve un estado
  desactualizado.

  CONDICIÓN CRÍTICA: los listeners nunca hacen I/O pesado.
  Si un listener es lento, bloquea el loop completo.
  Regla: los listeners solo actualizan estado en RAM.
         El flush a disco ocurre fuera del loop.

  AISLAMIENTO DE ERRORES: si un listener explota, el EventBus
  captura el error, lo loggea y continúa con los demás listeners.
  Un bug en la CLI nunca puede tumbar el loop ReAct.

ARQUITECTURA:
  EventBus (singleton)
  │
  ├── Registro de listeners por EventType
  │   {EventType.TASK_STARTED: [listener_a, listener_b], ...}
  │
  ├── emit(event) → itera listeners del tipo → llama cada uno
  │                → aísla excepciones de cada listener
  │
  └── Listeners registrados:
      ├── CLI listener     → actualiza estado de la UI en RAM
      ├── Logger listener  → structlog (bufferizado, no bloquea)
      └── Session listener → append a lista en RAM

PATRÓN DE USO EN EL EXECUTOR:
  # El executor solo emite eventos — no sabe quién escucha
  bus = get_event_bus()
  bus.emit(AgentEvent.task_started(task))

  # La CLI, el logger y la sesión reaccionan automáticamente
  # sin que el executor los conozca directamente

PATRÓN DE REGISTRO DE LISTENERS:
  bus = get_event_bus()

  @bus.on(EventType.TASK_STARTED)
  def handle_task_started(event: AgentEvent) -> None:
      print(f"Task iniciada: {event.task_id}")

  # O de forma imperativa:
  bus.subscribe(EventType.TOOL_CALLED, mi_listener)
"""

from __future__ import annotations

import traceback
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.core.constants import APP_NAME


# ---------------------------------------------------------------------------
# Tipos de eventos — catálogo completo de lo que puede ocurrir en el agente
# ---------------------------------------------------------------------------

class EventType(str, Enum):
    """
    Catálogo de todos los eventos que el agente puede emitir.

    Organizados por fase del ciclo de vida para facilitar
    el filtrado en listeners y en logs.

    Hereda de str para que los valores sean directamente
    serializables a JSON sin conversión extra.
    """

    # ── Ciclo de vida de la Task ──────────────────────────────────────
    TASK_STARTED        = "task.started"
    TASK_PLANNING       = "task.planning"
    TASK_EXECUTING      = "task.executing"
    TASK_COMPLETED      = "task.completed"
    TASK_FAILED         = "task.failed"
    TASK_CANCELLED      = "task.cancelled"

    # ── Ciclo de vida de Subtasks ─────────────────────────────────────
    SUBTASK_STARTED     = "subtask.started"
    SUBTASK_COMPLETED   = "subtask.completed"
    SUBTASK_FAILED      = "subtask.failed"
    SUBTASK_SKIPPED     = "subtask.skipped"

    # ── Loop ReAct ───────────────────────────────────────────────────
    # Cada iteración del loop produce estos eventos en orden:
    #   REACT_ITERATION_STARTED
    #   → LLM_REQUEST_SENT
    #   → LLM_RESPONSE_RECEIVED
    #   → TOOL_CALLED (si el LLM solicitó una tool)
    #   → TOOL_RESULT_RECEIVED
    #   → REACT_ITERATION_COMPLETED
    REACT_ITERATION_STARTED   = "react.iteration.started"
    REACT_ITERATION_COMPLETED = "react.iteration.completed"
    LLM_REQUEST_SENT          = "react.llm.request_sent"
    LLM_RESPONSE_RECEIVED     = "react.llm.response_received"
    LLM_STREAMING_CHUNK       = "react.llm.streaming_chunk"  # token por token
    TOOL_CALLED               = "react.tool.called"
    TOOL_RESULT_RECEIVED      = "react.tool.result_received"

    # ── Errores y recuperación ────────────────────────────────────────
    ERROR_OCCURRED      = "error.occurred"
    RETRY_SCHEDULED     = "error.retry_scheduled"

    # ── Sistema ───────────────────────────────────────────────────────
    SESSION_STARTED     = "system.session_started"
    SESSION_ENDED       = "system.session_ended"
    WORKSPACE_SWITCHED  = "system.workspace_switched"


# ---------------------------------------------------------------------------
# AgentEvent — el objeto que viaja por el bus
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AgentEvent:
    """
    Evento inmutable emitido por el agente durante la ejecución.

    Diseñado para ser liviano — vive en RAM y se crea en microsegundos.
    Los listeners reciben este objeto y extraen lo que necesitan.

    Atributos:
        event_type:  Qué ocurrió.
        occurred_at: Cuándo ocurrió (UTC, con precisión de microsegundos).
        data:        Contexto del evento. El contenido depende del event_type.
                     Ver los factory methods para los campos de cada tipo.

    INMUTABILIDAD:
        frozen=True garantiza que ningún listener pueda modificar el evento
        después de ser emitido. Esto es crítico cuando múltiples listeners
        procesan el mismo objeto simultáneamente.
    """

    event_type: EventType
    occurred_at: datetime
    data: dict[str, Any]

    # ------------------------------------------------------------------
    # Factory methods — un constructor semántico por tipo de evento.
    # Garantizan que cada evento lleve los campos correctos.
    # Mucho más claro que construir dicts a mano en el executor.
    # ------------------------------------------------------------------

    @classmethod
    def _make(cls, event_type: EventType, **data: Any) -> AgentEvent:
        """Constructor interno. Usar los factory methods públicos."""
        return cls(
            event_type=event_type,
            occurred_at=datetime.now(timezone.utc),
            data=data,
        )

    # ── Task ──────────────────────────────────────────────────────────

    @classmethod
    def task_started(cls, task_id: str, prompt: str, project_id: str | None = None) -> AgentEvent:
        return cls._make(
            EventType.TASK_STARTED,
            task_id=task_id,
            prompt_preview=prompt[:100],
            project_id=project_id,
        )

    @classmethod
    def task_planning(cls, task_id: str) -> AgentEvent:
        return cls._make(EventType.TASK_PLANNING, task_id=task_id)

    @classmethod
    def task_executing(cls, task_id: str, subtask_count: int) -> AgentEvent:
        return cls._make(
            EventType.TASK_EXECUTING,
            task_id=task_id,
            subtask_count=subtask_count,
        )

    @classmethod
    def task_completed(
        cls,
        task_id: str,
        duration_seconds: float,
        total_tokens: int,
        estimated_cost_usd: float,
    ) -> AgentEvent:
        return cls._make(
            EventType.TASK_COMPLETED,
            task_id=task_id,
            duration_seconds=duration_seconds,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost_usd,
        )

    @classmethod
    def task_failed(cls, task_id: str, reason: str) -> AgentEvent:
        return cls._make(
            EventType.TASK_FAILED,
            task_id=task_id,
            reason=reason,
        )

    @classmethod
    def task_cancelled(cls, task_id: str) -> AgentEvent:
        return cls._make(EventType.TASK_CANCELLED, task_id=task_id)

    # ── Subtask ───────────────────────────────────────────────────────

    @classmethod
    def subtask_started(cls, task_id: str, subtask_id: str, order: int, description: str) -> AgentEvent:
        return cls._make(
            EventType.SUBTASK_STARTED,
            task_id=task_id,
            subtask_id=subtask_id,
            order=order,
            description=description,
        )

    @classmethod
    def subtask_completed(
        cls,
        task_id: str,
        subtask_id: str,
        duration_seconds: float,
        react_iterations: int,
    ) -> AgentEvent:
        return cls._make(
            EventType.SUBTASK_COMPLETED,
            task_id=task_id,
            subtask_id=subtask_id,
            duration_seconds=duration_seconds,
            react_iterations=react_iterations,
        )

    @classmethod
    def subtask_failed(cls, task_id: str, subtask_id: str, reason: str) -> AgentEvent:
        return cls._make(
            EventType.SUBTASK_FAILED,
            task_id=task_id,
            subtask_id=subtask_id,
            reason=reason,
        )

    # ── Loop ReAct ────────────────────────────────────────────────────

    @classmethod
    def react_iteration_started(
        cls,
        task_id: str,
        subtask_id: str,
        iteration: int,
    ) -> AgentEvent:
        return cls._make(
            EventType.REACT_ITERATION_STARTED,
            task_id=task_id,
            subtask_id=subtask_id,
            iteration=iteration,
        )

    @classmethod
    def react_iteration_completed(
        cls,
        task_id: str,
        subtask_id: str,
        iteration: int,
    ) -> AgentEvent:
        return cls._make(
            EventType.REACT_ITERATION_COMPLETED,
            task_id=task_id,
            subtask_id=subtask_id,
            iteration=iteration,
        )

    @classmethod
    def llm_request_sent(
        cls,
        task_id: str,
        subtask_id: str,
        provider: str,
        model: str,
        message_count: int,
    ) -> AgentEvent:
        return cls._make(
            EventType.LLM_REQUEST_SENT,
            task_id=task_id,
            subtask_id=subtask_id,
            provider=provider,
            model=model,
            message_count=message_count,
        )

    @classmethod
    def llm_response_received(
        cls,
        task_id: str,
        subtask_id: str,
        provider: str,
        model: str,
        input_tokens: int,
        output_tokens: int,
        finish_reason: str,
        duration_seconds: float,
    ) -> AgentEvent:
        return cls._make(
            EventType.LLM_RESPONSE_RECEIVED,
            task_id=task_id,
            subtask_id=subtask_id,
            provider=provider,
            model=model,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            finish_reason=finish_reason,
            duration_seconds=duration_seconds,
        )

    @classmethod
    def llm_streaming_chunk(cls, task_id: str, subtask_id: str, chunk: str) -> AgentEvent:
        """
        Emitido por cada fragmento de texto en modo streaming.

        ATENCIÓN: Este evento se emite potencialmente cientos de veces
        por segundo durante el streaming. Los listeners de este evento
        deben ser extremadamente rápidos — solo actualizar un buffer
        en RAM, nunca I/O.
        """
        return cls._make(
            EventType.LLM_STREAMING_CHUNK,
            task_id=task_id,
            subtask_id=subtask_id,
            chunk=chunk,
        )

    @classmethod
    def tool_called(
        cls,
        task_id: str,
        subtask_id: str,
        tool_call_id: str,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> AgentEvent:
        return cls._make(
            EventType.TOOL_CALLED,
            task_id=task_id,
            subtask_id=subtask_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            # Solo preview de los argumentos — pueden ser muy grandes
            arguments_preview=str(arguments)[:200],
        )

    @classmethod
    def tool_result_received(
        cls,
        task_id: str,
        subtask_id: str,
        tool_call_id: str,
        tool_name: str,
        success: bool,
        duration_seconds: float,
        output_preview: str,
    ) -> AgentEvent:
        return cls._make(
            EventType.TOOL_RESULT_RECEIVED,
            task_id=task_id,
            subtask_id=subtask_id,
            tool_call_id=tool_call_id,
            tool_name=tool_name,
            success=success,
            duration_seconds=duration_seconds,
            output_preview=output_preview[:200],
        )

    # ── Errores ───────────────────────────────────────────────────────

    @classmethod
    def error_occurred(
        cls,
        task_id: str | None,
        subtask_id: str | None,
        error_type: str,
        error_message: str,
    ) -> AgentEvent:
        return cls._make(
            EventType.ERROR_OCCURRED,
            task_id=task_id,
            subtask_id=subtask_id,
            error_type=error_type,
            error_message=error_message,
        )

    @classmethod
    def retry_scheduled(
        cls,
        task_id: str,
        subtask_id: str,
        attempt: int,
        max_attempts: int,
        wait_seconds: float,
        reason: str,
    ) -> AgentEvent:
        return cls._make(
            EventType.RETRY_SCHEDULED,
            task_id=task_id,
            subtask_id=subtask_id,
            attempt=attempt,
            max_attempts=max_attempts,
            wait_seconds=wait_seconds,
            reason=reason,
        )

    # ── Sistema ───────────────────────────────────────────────────────

    @classmethod
    def session_started(cls, session_id: str, workspace_id: str) -> AgentEvent:
        return cls._make(
            EventType.SESSION_STARTED,
            session_id=session_id,
            workspace_id=workspace_id,
        )

    @classmethod
    def session_ended(cls, session_id: str, duration_seconds: float) -> AgentEvent:
        return cls._make(
            EventType.SESSION_ENDED,
            session_id=session_id,
            duration_seconds=duration_seconds,
        )

    def __str__(self) -> str:
        """
        Ejemplo: [14:32:01] react.tool.called  tool_name=shell duration=0.42s
        """
        time_label = self.occurred_at.strftime("%H:%M:%S")
        data_preview = " ".join(f"{k}={v!r}" for k, v in list(self.data.items())[:3])
        return f"[{time_label}] {self.event_type.value}  {data_preview}"


# ---------------------------------------------------------------------------
# Tipo del listener — función que recibe un AgentEvent y no devuelve nada
# ---------------------------------------------------------------------------

ListenerFn = Callable[[AgentEvent], None]


# ---------------------------------------------------------------------------
# EventBus — el motor central
# ---------------------------------------------------------------------------

class EventBus:
    """
    Bus de eventos síncrono para el agente ReAct.

    GARANTÍAS:
      1. Orden de ejecución: los listeners se ejecutan en el orden
         en que fueron registrados.
      2. Aislamiento: un error en un listener no afecta a los demás
         ni detiene el loop ReAct.
      3. Sin estado mutable compartido: el EventBus solo almacena
         referencias a funciones — no datos de eventos pasados.
      4. Thread-safety: el registro de listeners y el emit son seguros
         para uso desde un único thread (el loop ReAct es single-thread).

    LIMITACIÓN INTENCIONADA:
      No hay async/await. El loop ReAct de HiperForge es síncrono.
      Agregar async aquí requeriría un event loop separado y complicaría
      el modelo mental sin beneficio real para este caso de uso.
      Si en el futuro se necesita async (webhooks, notificaciones remotas),
      se puede agregar un AsyncEventBus separado sin modificar este.
    """

    def __init__(self) -> None:
        # defaultdict evita KeyError al acceder a tipos sin listeners registrados
        # La clave es EventType, el valor es lista ordenada de listeners
        self._listeners: defaultdict[EventType, list[ListenerFn]] = defaultdict(list)

        # Logger interno del bus — usamos el nombre del módulo
        # para que los logs del bus sean identificables en el archivo
        self._logger = _get_internal_logger()

        # Contador de eventos emitidos — útil para métricas y debugging
        self._emit_count: int = 0

        # Flag para silenciar el bus durante tests
        self._silent: bool = False

    # ------------------------------------------------------------------
    # Registro de listeners
    # ------------------------------------------------------------------

    def subscribe(self, event_type: EventType, listener: ListenerFn) -> None:
        """
        Registra un listener para un tipo de evento específico.

        El listener se llama sincrónicamente cada vez que se emite
        un evento de ese tipo.

        Parámetros:
            event_type: Tipo de evento a escuchar.
            listener:   Función a llamar. Debe ser rápida y nunca
                        hacer I/O pesado (ver docstring del módulo).

        Ejemplo:
            bus.subscribe(EventType.TOOL_CALLED, mi_listener)
        """
        self._listeners[event_type].append(listener)

    def subscribe_many(
        self,
        event_types: list[EventType],
        listener: ListenerFn,
    ) -> None:
        """
        Registra un listener para múltiples tipos de evento a la vez.

        Útil para listeners que manejan varios eventos relacionados,
        como el logger que loggea todo, o la CLI que muestra progreso
        de múltiples fases.

        Ejemplo:
            bus.subscribe_many(
                [EventType.SUBTASK_STARTED, EventType.SUBTASK_COMPLETED],
                actualizar_progreso_cli,
            )
        """
        for event_type in event_types:
            self.subscribe(event_type, listener)

    def on(self, event_type: EventType) -> Callable[[ListenerFn], ListenerFn]:
        """
        Decorador para registrar listeners de forma declarativa.

        Más expresivo que subscribe() cuando el listener está definido
        cerca del punto de registro.

        Ejemplo:
            @bus.on(EventType.TASK_COMPLETED)
            def notificar_completado(event: AgentEvent) -> None:
                print(f"Task completada en {event.data['duration_seconds']}s")
        """
        def decorator(fn: ListenerFn) -> ListenerFn:
            self.subscribe(event_type, fn)
            return fn
        return decorator

    def unsubscribe(self, event_type: EventType, listener: ListenerFn) -> bool:
        """
        Elimina un listener previamente registrado.

        Útil en tests para limpiar listeners entre casos de prueba.

        Returns:
            True si el listener existía y fue eliminado.
            False si el listener no estaba registrado.
        """
        listeners = self._listeners[event_type]
        try:
            listeners.remove(listener)
            return True
        except ValueError:
            return False

    def clear(self, event_type: EventType | None = None) -> None:
        """
        Elimina todos los listeners.

        Parámetros:
            event_type: Si se especifica, solo limpia ese tipo.
                        Si None, limpia todos los listeners de todos los tipos.

        Usado principalmente en tests para resetear el estado entre casos.
        """
        if event_type is not None:
            self._listeners[event_type].clear()
        else:
            self._listeners.clear()

    # ------------------------------------------------------------------
    # Emisión de eventos — el corazón del bus
    # ------------------------------------------------------------------

    def emit(self, event: AgentEvent) -> EmitResult:
        """
        Emite un evento a todos los listeners registrados para su tipo.

        COMPORTAMIENTO CRÍTICO:
          - Los listeners se ejecutan en orden de registro.
          - Si un listener lanza una excepción, el bus la captura,
            la registra en el log, y CONTINÚA con los siguientes listeners.
          - El loop ReAct NUNCA se interrumpe por un error de listener.
          - El método devuelve EmitResult con el detalle de qué pasó.

        Parámetros:
            event: El evento a emitir. Debe ser un AgentEvent inmutable.

        Returns:
            EmitResult con conteo de éxitos y fallos por listener.
        """
        if self._silent:
            return EmitResult(event_type=event.event_type, success_count=0, failure_count=0)

        self._emit_count += 1
        listeners = self._listeners.get(event.event_type, [])

        success_count = 0
        failures: list[ListenerFailure] = []

        for listener in listeners:
            try:
                listener(event)
                success_count += 1

            except Exception as exc:
                # AISLAMIENTO: capturamos el error del listener sin propagar
                failure = ListenerFailure(
                    listener_name=_get_listener_name(listener),
                    error_type=type(exc).__name__,
                    error_message=str(exc),
                    traceback=traceback.format_exc(),
                )
                failures.append(failure)

                # Loggeamos el error del listener pero NO lo propagamos
                # Usamos el logger interno para evitar recursión si el
                # listener que falló era el logger mismo
                self._logger(
                    f"[EventBus] Listener '{failure.listener_name}' "
                    f"falló en evento '{event.event_type.value}': {exc}"
                )

        return EmitResult(
            event_type=event.event_type,
            success_count=success_count,
            failure_count=len(failures),
            failures=failures,
        )

    # ------------------------------------------------------------------
    # Modo silencioso para tests
    # ------------------------------------------------------------------

    def silence(self) -> _SilencedBus:
        """
        Context manager que silencia el bus temporalmente.

        Úsalo en tests unitarios que no necesitan los efectos
        secundarios de los listeners:

            with bus.silence():
                executor.run(task)   # no dispara eventos

        Returns:
            Context manager que restaura el estado al salir.
        """
        return _SilencedBus(self)

    # ------------------------------------------------------------------
    # Introspección — útil para debugging y tests
    # ------------------------------------------------------------------

    @property
    def emit_count(self) -> int:
        """Total de eventos emitidos desde que se creó el bus."""
        return self._emit_count

    def listener_count(self, event_type: EventType | None = None) -> int:
        """
        Número de listeners registrados.

        Parámetros:
            event_type: Si se especifica, cuenta solo para ese tipo.
                        Si None, cuenta el total de todos los tipos.
        """
        if event_type is not None:
            return len(self._listeners.get(event_type, []))
        return sum(len(ls) for ls in self._listeners.values())

    def registered_event_types(self) -> list[EventType]:
        """Devuelve los EventTypes que tienen al menos un listener."""
        return [et for et, ls in self._listeners.items() if ls]

    def __repr__(self) -> str:
        return (
            f"EventBus("
            f"listeners={self.listener_count()}, "
            f"emitted={self._emit_count})"
        )


# ---------------------------------------------------------------------------
# Tipos de soporte para el resultado de emit()
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ListenerFailure:
    """Detalle de un listener que falló durante emit()."""

    listener_name: str
    error_type: str
    error_message: str
    traceback: str


@dataclass(frozen=True)
class EmitResult:
    """
    Resultado de una llamada a emit().

    Permite al caller saber si algún listener falló sin que
    el error haya interrumpido el flujo principal.

    En producción normalmente se ignora. En tests es útil para
    verificar que todos los listeners procesaron el evento correctamente.
    """

    event_type: EventType
    success_count: int
    failure_count: int
    failures: list[ListenerFailure] = field(default_factory=list)

    @property
    def all_succeeded(self) -> bool:
        """True si todos los listeners procesaron el evento sin errores."""
        return self.failure_count == 0

    def __str__(self) -> str:
        status = "OK" if self.all_succeeded else f"{self.failure_count} fallos"
        return f"EmitResult({self.event_type.value}, {status})"


# ---------------------------------------------------------------------------
# Context manager para modo silencioso
# ---------------------------------------------------------------------------

class _SilencedBus:
    """
    Context manager que silencia el EventBus temporalmente.

    No es parte de la API pública — acceder solo a través de bus.silence().
    """

    def __init__(self, bus: EventBus) -> None:
        self._bus = bus

    def __enter__(self) -> EventBus:
        self._bus._silent = True
        return self._bus

    def __exit__(self, *_: Any) -> None:
        self._bus._silent = False


# ---------------------------------------------------------------------------
# Singleton del EventBus — una instancia por proceso
# ---------------------------------------------------------------------------

_bus_instance: EventBus | None = None


def get_event_bus() -> EventBus:
    """
    Devuelve la instancia única del EventBus (singleton).

    El bus es único por proceso — todos los módulos que llamen
    get_event_bus() reciben la misma instancia y comparten los mismos
    listeners registrados.

    Thread-safety: el bus asume single-thread (el loop ReAct).
    Si en el futuro hay múltiples threads, agregar un Lock alrededor
    de emit() y de la modificación de _listeners.

    Returns:
        Instancia única del EventBus.
    """
    global _bus_instance
    if _bus_instance is None:
        _bus_instance = EventBus()
    return _bus_instance


def reset_event_bus() -> None:
    """
    Destruye el singleton y crea uno nuevo.

    SOLO para tests — permite que cada test empiece con un bus limpio
    sin listeners del test anterior.

    Ejemplo en conftest.py:
        @pytest.fixture(autouse=True)
        def clean_event_bus():
            reset_event_bus()
            yield
            reset_event_bus()
    """
    global _bus_instance
    _bus_instance = None


# ---------------------------------------------------------------------------
# Helpers internos
# ---------------------------------------------------------------------------

def _get_listener_name(listener: ListenerFn) -> str:
    """
    Extrae un nombre legible de una función listener.

    Usado en los logs de error del bus para identificar qué listener falló
    sin necesidad de un stack trace completo.
    """
    # Intentamos obtener el nombre más descriptivo posible
    module = getattr(listener, "__module__", "unknown")
    name = getattr(listener, "__name__", repr(listener))
    return f"{module}.{name}"


def _get_internal_logger() -> Callable[[str], None]:
    """
    Devuelve una función de logging segura para uso interno del EventBus.

    Usamos print como fallback en vez de structlog para evitar
    recursión infinita si el listener que falla es el logger mismo.
    El logger real del sistema puede no estar inicializado todavía
    cuando el EventBus se crea.
    """
    def log(message: str) -> None:
        # Intentamos usar structlog si está disponible
        try:
            import structlog
            structlog.get_logger(APP_NAME).warning(message)
        except Exception:
            # Fallback a print — nunca debe fallar
            print(f"[{APP_NAME}:event_bus] {message}", flush=True)
    return log