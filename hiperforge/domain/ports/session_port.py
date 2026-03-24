"""
Port: SessionPort

Define el contrato para gestionar el estado de una sesión activa
del agente en memoria RAM durante la ejecución del loop ReAct.

¿Qué es una sesión?
  Una sesión es el estado temporal que existe desde que el usuario
  ejecuta `hiperforge run "..."` hasta que el agente termina.

  Durante ese tiempo el agente necesita:
    - Mantener el historial de mensajes con el LLM (contexto de conversación)
    - Registrar eventos de lo que va ocurriendo (subtask iniciada, tool ejecutada)
    - Acceder a la task activa y sus subtasks en progreso
    - Acumular el uso de tokens en tiempo real

  TODO esto vive en RAM — es rápido, sin I/O en cada operación.
  Al terminar la sesión, session_flusher.py persiste lo necesario a JSON.

¿Por qué no escribir directo a disco en cada paso?
  El loop ReAct puede iterar decenas de veces por subtask.
  Escribir a disco en cada iteración sería lento y generaría
  contención de I/O innecesaria. La estrategia es:

    RAM (rápido, volátil)        Disco (lento, persistente)
    ─────────────────────        ──────────────────────────
    SessionPort                  StoragePort
    estado durante ejecución  →  flush al terminar la sesión
    mensajes, eventos, task      task.json, session.json

IMPLEMENTACIÓN ESPERADA:
  InMemorySession  →  todo en dicts y listas Python en RAM

USO TÍPICO (desde executor.py):
  class ExecutorService:
      def __init__(self, session: SessionPort, ...) -> None:
          self._session = session

      def run_react_loop(self, subtask: Subtask) -> Subtask:
          # El agente razona
          self._session.push_message(Message.user(subtask.description))

          while not subtask.is_terminal:
              response = self._llm.complete(self._session.get_messages())
              self._session.push_message(Message.assistant(response.content))
              self._session.record_event("llm_response", {"tokens": ...})

              # Actuar, observar, repetir...
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.domain.entities.task import Task
from hiperforge.domain.value_objects.message import Message
from hiperforge.domain.value_objects.token_usage import TokenUsage


class EventType(str, Enum):
    """
    Tipos de eventos que ocurren durante una sesión.

    Estos eventos forman el log de actividad de la sesión —
    permiten reconstruir exactamente qué hizo el agente y en qué orden.
    Son la base para debugging y para la vista de progreso en la CLI.
    """

    # Ciclo de vida de la task
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    TASK_CANCELLED = "task_cancelled"

    # Ciclo de vida del plan
    PLANNING_STARTED = "planning_started"
    PLANNING_COMPLETED = "planning_completed"

    # Ciclo de vida de subtasks
    SUBTASK_STARTED = "subtask_started"
    SUBTASK_COMPLETED = "subtask_completed"
    SUBTASK_FAILED = "subtask_failed"

    # Loop ReAct
    REACT_ITERATION = "react_iteration"     # cada vuelta del loop
    LLM_CALLED = "llm_called"              # se hizo una llamada al LLM
    TOOL_CALLED = "tool_called"            # el agente ejecutó una tool
    TOOL_RESULT_RECEIVED = "tool_result_received"  # se recibió el resultado

    # Errores y advertencias
    ERROR_OCCURRED = "error_occurred"
    RETRY_ATTEMPTED = "retry_attempted"


@dataclass(frozen=True)
class SessionEvent:
    """
    Evento inmutable registrado durante la ejecución de la sesión.

    Cada acción significativa del agente produce un evento.
    La secuencia de eventos es el historial completo de la sesión.

    Atributos:
        event_type:  Tipo de evento (qué ocurrió).
        data:        Datos extra del evento (IDs, mensajes, métricas).
        occurred_at: Momento exacto del evento en UTC.
    """

    event_type: EventType
    data: dict[str, Any]
    occurred_at: datetime

    @classmethod
    def create(cls, event_type: EventType, data: dict[str, Any] | None = None) -> SessionEvent:
        """Crea un evento con el timestamp actual."""
        return cls(
            event_type=event_type,
            data=data or {},
            occurred_at=datetime.now(timezone.utc),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializa para persistir en session.json al hacer flush."""
        return {
            "event_type": self.event_type.value,
            "data": self.data,
            "occurred_at": self.occurred_at.isoformat(),
        }

    def __str__(self) -> str:
        """
        Ejemplo: [14:32:01] TOOL_CALLED → {"tool": "shell", "command": "pytest..."}
        """
        time_label = self.occurred_at.strftime("%H:%M:%S")
        data_preview = str(self.data)[:80]
        return f"[{time_label}] {self.event_type.value} → {data_preview}"


class SessionPort(ABC):
    """
    Contrato abstracto para la gestión del estado de sesión en memoria.

    El executor del loop ReAct usa este port para mantener el contexto
    de la conversación con el LLM y el registro de eventos de la sesión.
    """

    # ------------------------------------------------------------------
    # Identidad de la sesión
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def session_id(self) -> str:
        """ID único de esta sesión (ULID). Generado al crear la sesión."""
        ...

    @property
    @abstractmethod
    def task(self) -> Task:
        """
        La task que se está ejecutando en esta sesión.

        Es una referencia al estado más reciente de la task.
        Se actualiza con update_task() en cada iteración del loop ReAct.
        """
        ...

    # ------------------------------------------------------------------
    # Gestión del historial de mensajes (contexto del LLM)
    # ------------------------------------------------------------------

    @abstractmethod
    def push_message(self, message: Message) -> None:
        """
        Agrega un mensaje al historial de la conversación con el LLM.

        El historial se pasa completo en cada llamada al LLM para que
        el modelo tenga contexto de todo lo que ocurrió antes.

        Parámetros:
            message: Mensaje a agregar (user, assistant o system).
        """
        ...

    @abstractmethod
    def get_messages(self) -> list[Message]:
        """
        Devuelve el historial completo de mensajes en orden cronológico.

        Este es el contexto que se envía al LLM en cada iteración.
        El orden importa — el LLM interpreta los mensajes secuencialmente.

        Returns:
            Lista de mensajes desde el más antiguo al más reciente.
        """
        ...

    @abstractmethod
    def get_messages_for_subtask(self, subtask_id: str) -> list[Message]:
        """
        Devuelve solo los mensajes relacionados con una subtask específica.

        Útil para:
          - Mostrar el detalle de una subtask en la CLI.
          - Construir el contexto reducido al cambiar de subtask.
          - Debug: ver exactamente qué vio el LLM en una subtask.

        Parámetros:
            subtask_id: ID de la subtask a filtrar.

        Returns:
            Mensajes del historial que corresponden a esa subtask.
        """
        ...

    @abstractmethod
    def clear_messages(self) -> None:
        """
        Vacía el historial de mensajes.

        Usado al cambiar de subtask para evitar que el contexto
        de la subtask anterior contamine la siguiente.

        ATENCIÓN: Los mensajes eliminados de memoria se habrán
        persistido previamente via flush si son importantes.
        """
        ...

    # ------------------------------------------------------------------
    # Gestión de la task activa
    # ------------------------------------------------------------------

    @abstractmethod
    def update_task(self, task: Task) -> None:
        """
        Actualiza el estado de la task en la sesión.

        Se llama en cada paso del loop ReAct que modifica la task —
        al completar una subtask, al agregar tool calls, etc.

        La sesión mantiene siempre la versión más reciente de la task
        en RAM para no leer de disco en cada iteración.
        """
        ...

    # ------------------------------------------------------------------
    # Registro de eventos
    # ------------------------------------------------------------------

    @abstractmethod
    def record_event(self, event_type: EventType, data: dict[str, Any] | None = None) -> None:
        """
        Registra un evento de la sesión.

        Cada acción significativa del agente debe registrarse:
          - Cuando empieza una subtask
          - Cuando el LLM responde
          - Cuando se ejecuta una tool
          - Cuando ocurre un error

        Los eventos son la fuente de verdad del log de actividad.
        El EventBus de core/ puede reaccionar a estos eventos
        para actualizar la CLI en tiempo real.

        Parámetros:
            event_type: Tipo de evento (qué ocurrió).
            data:       Datos extra del evento. None si no hay datos adicionales.
        """
        ...

    @abstractmethod
    def get_events(self) -> list[SessionEvent]:
        """
        Devuelve todos los eventos registrados en orden cronológico.

        Usado por session_flusher.py al persistir la sesión a JSON.
        También usado por la CLI para mostrar el log de actividad.
        """
        ...

    # ------------------------------------------------------------------
    # Tracking de tokens
    # ------------------------------------------------------------------

    @abstractmethod
    def accumulate_tokens(self, usage: TokenUsage) -> None:
        """
        Acumula el uso de tokens de una llamada al LLM.

        Se llama después de cada respuesta del LLM durante el loop ReAct.
        El total acumulado se usa para mostrar el costo de la sesión
        al terminar y para persistirlo en el JSON de la task.
        """
        ...

    @abstractmethod
    def get_total_token_usage(self) -> TokenUsage:
        """
        Devuelve el uso total acumulado de tokens en esta sesión.

        Incluye todas las llamadas al LLM desde que empezó la sesión.
        """
        ...

    # ------------------------------------------------------------------
    # Serialización para flush a disco
    # ------------------------------------------------------------------

    @abstractmethod
    def to_dict(self) -> dict[str, Any]:
        """
        Serializa el estado completo de la sesión para persistir a JSON.

        Llamado por session_flusher.py al terminar la sesión
        (ya sea por completarse, fallar o recibir señal de interrupción).

        El resultado se guarda en:
          workspaces/{workspace_id}/sessions/{session_id}.json
        """
        ...
