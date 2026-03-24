"""
InMemorySession — Implementación de SessionPort en memoria RAM.

El estado de una sesión activa del agente vive completamente en RAM
durante la ejecución. Al terminar, session_flusher.py lo persiste a disco.

¿POR QUÉ TODO EN RAM?
  El loop ReAct puede iterar decenas de veces por subtask.
  En cada iteración:
    - Se agregan mensajes al historial
    - Se registran eventos
    - Se acumula token usage
    - Se actualiza la task activa

  Si cada una de esas operaciones escribiera a disco, el loop
  tendría latencia de I/O en cada iteración. En RAM todas estas
  operaciones son nanosegundos — sin fricción.

  Al terminar la sesión (éxito, fallo, Ctrl+C via session_flusher),
  todo se persiste de una vez en una sola operación atómica.

THREAD SAFETY:
  El loop ReAct es single-thread. InMemorySession NO es thread-safe
  intencionalmente — agregar locks añadiría complejidad innecesaria
  para un caso de uso que no lo requiere.

CONTENIDO DE LA SESIÓN EN RAM:
  - Historial completo de mensajes con el LLM (contexto de conversación)
  - Registro cronológico de eventos (log de actividad)
  - Task activa en su estado más reciente
  - Token usage acumulado de todas las llamadas al LLM
  - Metadata de la sesión (ID, timestamps, workspace)
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from hiperforge.core.logging import get_logger
from hiperforge.core.utils.ids import generate_session_id
from hiperforge.domain.entities.task import Task
from hiperforge.domain.ports.session_port import EventType, SessionEvent, SessionPort
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.domain.value_objects.token_usage import TokenUsage

logger = get_logger(__name__)


class InMemorySession(SessionPort):
    """
    Sesión activa del agente almacenada completamente en RAM.

    Se crea al inicio de cada ejecución de `hiperforge run "..."`.
    Se destruye (después de hacer flush a disco) al terminar.

    Parámetros:
        task:         Task que se va a ejecutar en esta sesión.
        workspace_id: ID del workspace activo.
    """

    def __init__(self, task: Task, workspace_id: str) -> None:
        self._session_id = generate_session_id()
        self._task = task
        self._workspace_id = workspace_id
        self._started_at = datetime.now(timezone.utc)

        # Historial de mensajes con el LLM — el contexto de la conversación
        # Lista ordenada cronológicamente: más antiguo primero
        self._messages: list[Message] = []

        # Índice de mensajes por subtask_id para consultas rápidas
        # { subtask_id: [índices en self._messages] }
        self._messages_by_subtask: dict[str, list[int]] = {}

        # ID de la subtask activa — para indexar mensajes correctamente
        self._active_subtask_id: str | None = None

        # Registro cronológico de eventos de la sesión
        self._events: list[SessionEvent] = []

        # Acumulador de tokens de todas las llamadas al LLM en esta sesión
        self._total_token_usage = TokenUsage.zero()

        # Registramos el evento de inicio de sesión
        self.record_event(
            EventType.SESSION_STARTED,
            {
                "session_id": self._session_id,
                "workspace_id": workspace_id,
                "task_id": task.id,
            },
        )

        logger.info(
            "sesión iniciada",
            session_id=self._session_id,
            task_id=task.id,
            workspace_id=workspace_id,
        )

    # ------------------------------------------------------------------
    # Identidad de la sesión
    # ------------------------------------------------------------------

    @property
    def session_id(self) -> str:
        return self._session_id

    @property
    def task(self) -> Task:
        return self._task

    @property
    def workspace_id(self) -> str:
        return self._workspace_id

    # ------------------------------------------------------------------
    # Gestión del historial de mensajes
    # ------------------------------------------------------------------

    def push_message(self, message: Message) -> None:
        """
        Agrega un mensaje al historial y lo indexa por subtask activa.

        El índice por subtask permite recuperar solo los mensajes
        relevantes a una subtask específica sin iterar todo el historial.
        """
        index = len(self._messages)
        self._messages.append(message)

        # Indexamos el mensaje bajo la subtask activa
        if self._active_subtask_id is not None:
            if self._active_subtask_id not in self._messages_by_subtask:
                self._messages_by_subtask[self._active_subtask_id] = []
            self._messages_by_subtask[self._active_subtask_id].append(index)

    def get_messages(self) -> list[Message]:
        """
        Devuelve el historial completo en orden cronológico.

        Devuelve una copia para evitar modificaciones externas
        que rompan la consistencia del historial.
        """
        return list(self._messages)

    def get_messages_for_subtask(self, subtask_id: str) -> list[Message]:
        """
        Devuelve solo los mensajes de una subtask específica.

        Usa el índice interno para O(k) donde k es el número de
        mensajes de esa subtask — no itera todo el historial.
        """
        indices = self._messages_by_subtask.get(subtask_id, [])
        return [self._messages[i] for i in indices]

    def clear_messages(self) -> None:
        """
        Vacía el historial de mensajes manteniendo el mensaje de sistema.

        El mensaje de sistema contiene las instrucciones del agente —
        siempre debe estar presente. Al cambiar de subtask, limpiamos
        el historial conversacional pero preservamos el sistema.

        Después de clear_messages(), el historial contiene SOLO
        el mensaje de sistema si existe — listo para la siguiente subtask.
        """
        # Preservamos el mensaje de sistema si existe
        system_messages = [m for m in self._messages if m.role == Role.SYSTEM]

        self._messages = system_messages
        self._messages_by_subtask = {}

        logger.debug(
            "historial de mensajes limpiado",
            session_id=self._session_id,
            preserved_system_messages=len(system_messages),
        )

    def set_active_subtask(self, subtask_id: str | None) -> None:
        """
        Establece la subtask activa para indexar mensajes correctamente.

        Llamado por el executor al inicio de cada subtask.
        Los mensajes agregados después de esta llamada se indexan
        bajo el nuevo subtask_id.

        Parámetros:
            subtask_id: ID de la subtask que empieza. None al terminar.
        """
        self._active_subtask_id = subtask_id

        if subtask_id is not None:
            logger.debug(
                "subtask activa actualizada en sesión",
                session_id=self._session_id,
                subtask_id=subtask_id,
            )

    # ------------------------------------------------------------------
    # Gestión de la task activa
    # ------------------------------------------------------------------

    def update_task(self, task: Task) -> None:
        """
        Actualiza el estado de la task en la sesión.

        Reemplaza la referencia interna con la versión más reciente.
        Como Task es inmutable, cada modificación produce una nueva
        instancia — guardamos siempre la más reciente.
        """
        self._task = task

    # ------------------------------------------------------------------
    # Registro de eventos
    # ------------------------------------------------------------------

    def record_event(
        self,
        event_type: EventType,
        data: dict[str, Any] | None = None,
    ) -> None:
        """
        Registra un evento en el historial de la sesión.

        Los eventos forman el log de actividad completo de la sesión.
        Se persisten a disco al hacer flush al terminar.
        """
        event = SessionEvent.create(event_type=event_type, data=data)
        self._events.append(event)

    def get_events(self) -> list[SessionEvent]:
        """Devuelve todos los eventos en orden cronológico."""
        return list(self._events)

    def get_events_for_subtask(self, subtask_id: str) -> list[SessionEvent]:
        """
        Filtra eventos relacionados con una subtask específica.

        Útil para mostrar el detalle de una subtask en la CLI
        o para generar el resumen de lo que hizo el agente en ella.
        """
        return [
            event
            for event in self._events
            if event.data.get("subtask_id") == subtask_id
        ]

    # ------------------------------------------------------------------
    # Tracking de tokens
    # ------------------------------------------------------------------

    def accumulate_tokens(self, usage: TokenUsage) -> None:
        """
        Acumula el uso de tokens de una llamada al LLM.

        Usa el operador + de TokenUsage que produce un nuevo objeto —
        consistente con la inmutabilidad de los value objects del dominio.
        """
        self._total_token_usage = self._total_token_usage + usage

    def get_total_token_usage(self) -> TokenUsage:
        return self._total_token_usage

    # ------------------------------------------------------------------
    # Propiedades de conveniencia para el executor y la CLI
    # ------------------------------------------------------------------

    @property
    def message_count(self) -> int:
        """Total de mensajes en el historial actual."""
        return len(self._messages)

    @property
    def event_count(self) -> int:
        """Total de eventos registrados en esta sesión."""
        return len(self._events)

    @property
    def duration_seconds(self) -> float:
        """Segundos transcurridos desde el inicio de la sesión."""
        delta = datetime.now(timezone.utc) - self._started_at
        return round(delta.total_seconds(), 3)

    @property
    def started_at(self) -> datetime:
        """Momento en que inició la sesión (UTC)."""
        return self._started_at

    # ------------------------------------------------------------------
    # Serialización para flush a disco
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa el estado completo de la sesión para persistir a JSON.

        Llamado por session_flusher.py al terminar la sesión.
        Guarda todo lo necesario para reconstruir el historial
        de actividad de la sesión — útil para auditoría y debug.

        NOTA: No guardamos los mensajes completos del LLM para no
        duplicar datos que ya están en task.json. Solo guardamos
        los eventos y los metadatos de la sesión.
        """
        return {
            "session_id": self._session_id,
            "workspace_id": self._workspace_id,
            "task_id": self._task.id,
            "started_at": self._started_at.isoformat(),
            "duration_seconds": self.duration_seconds,
            "message_count": self.message_count,
            "event_count": self.event_count,
            "total_token_usage": self._total_token_usage.to_dict(),
            "events": [event.to_dict() for event in self._events],
        }

    def __repr__(self) -> str:
        return (
            f"InMemorySession("
            f"id={self._session_id[:8]}, "
            f"task={self._task.id[:8]}, "
            f"messages={self.message_count}, "
            f"events={self.event_count}, "
            f"duration={self.duration_seconds}s)"
        )
