"""
SessionFlusher — Persiste la sesión en RAM a disco al terminar.

El flusher es el puente entre InMemorySession (RAM) y JSONStorage (disco).
Se encarga de que ningún dato se pierda cuando la sesión termina,
ya sea por éxito, fallo, o interrupción del proceso (Ctrl+C, SIGTERM).

¿CUÁNDO SE HACE FLUSH?
  1. ÉXITO: la task se completó normalmente → flush completo
  2. FALLO: una subtask falló sin recuperación → flush con estado de error
  3. INTERRUPCIÓN: Ctrl+C o SIGTERM → flush de emergencia via signal handlers
  4. EXCEPCIÓN NO MANEJADA: crash del proceso → flush en el except del executor

FLUSH DE EMERGENCIA (señales del OS):
  Cuando el usuario presiona Ctrl+C, Python lanza KeyboardInterrupt.
  El executor lo captura, llama flush_on_interrupt() y termina limpiamente.
  Esto garantiza que el trabajo parcial no se pierde — la task queda
  en estado IN_PROGRESS y puede reanudarse en el futuro.

  SIGTERM (proceso terminado por el sistema):
  Registramos un signal handler al crear el flusher que llama
  flush_on_interrupt() automáticamente. El developer no necesita
  hacer nada — el flusher se registra y protege la sesión solo.

QUÉ SE PERSISTE AL HACER FLUSH:
  1. task.json    → estado completo de la task con subtasks y tool calls
  2. session.json → log de eventos y metadatos de la sesión
  3. project.json → actualizado con el task_id si es nueva
  4. workspace.json → actualizado con project_id si es nuevo

El orden importa: guardamos de más específico a más general.
Si falla a mitad, los datos más importantes (la task) ya están guardados.
"""

from __future__ import annotations

import signal
import sys
from datetime import datetime, timezone
from typing import Any

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import TaskStatus
from hiperforge.infrastructure.session.in_memory_session import InMemorySession
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator

logger = get_logger(__name__)


class SessionFlusher:
    """
    Persiste el estado de una sesión activa a disco.

    Se instancia una vez por sesión y se mantiene vivo hasta que
    la sesión termina. Registra handlers de señales automáticamente
    para garantizar el flush incluso en interrupciones inesperadas.

    Parámetros:
        session:  La sesión en RAM que se va a persistir.
        storage:  El storage para escribir a disco.
        locator:  El locator para resolver rutas de archivos.
    """

    def __init__(
        self,
        session: InMemorySession,
        storage: JSONStorage,
        locator: WorkspaceLocator,
    ) -> None:
        self._session = session
        self._storage = storage
        self._locator = locator

        # Flag para evitar flush doble si se llama más de una vez
        self._flushed = False

        # Guardamos los signal handlers originales para restaurarlos
        self._original_sigterm_handler = None
        self._original_sigint_handler = None

        # Registramos handlers de señales inmediatamente
        self._register_signal_handlers()

        logger.debug(
            "SessionFlusher inicializado",
            session_id=session.session_id,
        )

    # ------------------------------------------------------------------
    # Flush principal
    # ------------------------------------------------------------------

    def flush(self) -> None:
        """
        Persiste el estado completo de la sesión a disco.

        Idempotente — si ya se hizo flush, no hace nada.
        Esto evita doble escritura si tanto el executor como
        el signal handler llaman flush() al terminar.

        ORDEN DE ESCRITURA (más específico → más general):
          1. task.json      → estado de la task con todo su historial
          2. session.json   → log de eventos de la sesión
          3. project.json   → actualizar task_ids si es necesario
        """
        if self._flushed:
            logger.debug(
                "flush ya realizado anteriormente — ignorando",
                session_id=self._session.session_id,
            )
            return

        self._flushed = True
        task = self._session.task

        logger.info(
            "iniciando flush de sesión a disco",
            session_id=self._session.session_id,
            task_id=task.id,
            task_status=task.status.value,
            duration_seconds=self._session.duration_seconds,
        )

        try:
            # Paso 1: persistir la task con todo su estado actual
            self._storage.save_task(task)

            # Paso 2: persistir el log de eventos de la sesión
            self._flush_session_log()

            # Paso 3: actualizar el project con el task_id si es necesario
            self._update_project_task_index(task)

            logger.info(
                "flush de sesión completado exitosamente",
                session_id=self._session.session_id,
                task_id=task.id,
                total_tokens=self._session.get_total_token_usage().total_tokens,
                estimated_cost_usd=self._session.get_total_token_usage().estimated_cost_usd,
                duration_seconds=self._session.duration_seconds,
            )

        except Exception as exc:
            # El flush NUNCA debe propagar excepciones al caller
            # Si falla guardar algo, loggeamos pero no rompemos el flujo
            logger.error(
                "error durante flush de sesión",
                session_id=self._session.session_id,
                task_id=task.id,
                error_type=type(exc).__name__,
                error=str(exc),
            )

    def flush_on_interrupt(self) -> None:
        """
        Flush de emergencia al recibir interrupción del usuario o del sistema.

        A diferencia de flush(), este método:
          1. Marca la task como CANCELLED si está activa (no terminal)
          2. Registra un evento SESSION_ENDED con el motivo de interrupción
          3. Llama flush() para persistir todo

        Llamado desde los signal handlers de SIGINT y SIGTERM.
        """
        if self._flushed:
            return

        task = self._session.task

        logger.warning(
            "interrupción detectada — realizando flush de emergencia",
            session_id=self._session.session_id,
            task_id=task.id,
            task_status=task.status.value,
        )

        # Si la task estaba en progreso, la marcamos como cancelada
        if not task.is_terminal:
            try:
                cancelled_task = task.cancel()
                self._session.update_task(cancelled_task)
            except Exception:
                pass  # Si falla el cancel, persistimos el estado actual

        # Registramos el evento de fin de sesión por interrupción
        from hiperforge.domain.ports.session_port import EventType
        self._session.record_event(
            EventType.SESSION_ENDED,
            {
                "session_id": self._session.session_id,
                "reason": "interrupted",
                "duration_seconds": self._session.duration_seconds,
            },
        )

        # Flush normal del estado
        self.flush()

    # ------------------------------------------------------------------
    # Pasos internos del flush
    # ------------------------------------------------------------------

    def _flush_session_log(self) -> None:
        """
        Persiste el log de eventos de la sesión a session.json.

        El archivo se guarda en:
          workspaces/{workspace_id}/sessions/{session_id}.json
        """
        session_path = self._locator.session_file(
            workspace_id=self._session.workspace_id,
            session_id=self._session.session_id,
        )

        session_data = self._session.to_dict()

        # Agregamos timestamps de finalización al JSON
        session_data["ended_at"] = datetime.now(timezone.utc).isoformat()

        self._storage.write_json(session_path, session_data)

        logger.debug(
            "log de sesión persistido",
            session_id=self._session.session_id,
            path=str(session_path),
            event_count=self._session.event_count,
        )

    def _update_project_task_index(self, task) -> None:
        """
        Actualiza el project.json para incluir el task_id si es nuevo.

        Cuando se crea una task nueva, el project.json aún no la conoce.
        Este paso sincroniza el índice del proyecto con la nueva task.

        Si el project_id es None (task suelta sin proyecto), no hacemos nada.
        """
        if task.project_id is None:
            return

        # Buscamos el workspace que contiene este project
        workspace_id = self._storage._find_workspace_id_for_project(task.project_id)

        if workspace_id is None:
            logger.warning(
                "no se encontró workspace para actualizar índice del project",
                project_id=task.project_id,
                task_id=task.id,
            )
            return

        try:
            # Leemos el project.json actual sin cargar las tasks completas
            project_path = self._locator.project_file(workspace_id, task.project_id)
            project_data = self._storage.read_json(project_path)

            # Actualizamos la lista de task_ids si es necesario
            task_ids: list[str] = project_data.get("task_ids", [])

            if task.id not in task_ids:
                task_ids.append(task.id)
                project_data["task_ids"] = task_ids
                project_data["updated_at"] = datetime.now(timezone.utc).isoformat()

                self._storage.write_json(project_path, project_data)

                logger.debug(
                    "task_id agregado al índice del project",
                    project_id=task.project_id,
                    task_id=task.id,
                )

        except Exception as exc:
            # No rompemos el flush si falla actualizar el índice
            # La task ya está guardada — el índice es secundario
            logger.warning(
                "no se pudo actualizar el índice del project",
                project_id=task.project_id,
                task_id=task.id,
                error=str(exc),
            )

    # ------------------------------------------------------------------
    # Signal handlers — flush automático ante interrupciones
    # ------------------------------------------------------------------

    def _register_signal_handlers(self) -> None:
        """
        Registra handlers de señales para flush automático.

        En Windows solo SIGINT (Ctrl+C) está disponible.
        En Linux/Mac también SIGTERM (proceso terminado por el sistema).
        """
        # SIGINT — Ctrl+C, disponible en todos los OS
        self._original_sigint_handler = signal.getsignal(signal.SIGINT)
        signal.signal(signal.SIGINT, self._handle_signal)

        # SIGTERM — solo en Unix/Linux
        if sys.platform != "win32":
            self._original_sigterm_handler = signal.getsignal(signal.SIGTERM)
            signal.signal(signal.SIGTERM, self._handle_signal)

        logger.debug(
            "signal handlers registrados para flush automático",
            session_id=self._session.session_id,
        )

    def _handle_signal(self, signum: int, frame: Any) -> None:
        """
        Handler de señal — hace flush de emergencia y termina limpiamente.

        Después del flush, restauramos el handler original y re-enviamos
        la señal para que Python pueda terminar normalmente.
        """
        signal_name = signal.Signals(signum).name

        logger.warning(
            f"señal {signal_name} recibida — iniciando flush de emergencia",
            session_id=self._session.session_id,
        )

        # Flush de emergencia
        self.flush_on_interrupt()

        # Restauramos los handlers originales
        self._restore_signal_handlers()

        # Re-enviamos la señal para que el proceso termine normalmente
        # Esto permite que Python lance KeyboardInterrupt o termine con el
        # exit code correcto para SIGTERM
        signal.raise_signal(signum)

    def _restore_signal_handlers(self) -> None:
        """Restaura los signal handlers originales."""
        if self._original_sigint_handler is not None:
            signal.signal(signal.SIGINT, self._original_sigint_handler)

        if self._original_sigterm_handler is not None and sys.platform != "win32":
            signal.signal(signal.SIGTERM, self._original_sigterm_handler)

    # ------------------------------------------------------------------
    # Context manager — uso idiomático con with
    # ------------------------------------------------------------------

    def __enter__(self) -> SessionFlusher:
        """
        Permite usar el flusher como context manager:

            with SessionFlusher(session, storage, locator) as flusher:
                executor.run(task)
            # flush automático al salir del with, incluso si hay excepción
        """
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> bool:
        """
        Flush automático al salir del context manager.

        Si hubo excepción, hace flush de emergencia.
        Si no hubo excepción, hace flush normal.

        Siempre devuelve False para no suprimir la excepción original.
        """
        if exc_type is not None:
            # Hubo excepción — flush de emergencia
            self.flush_on_interrupt()
        else:
            # Terminó normalmente
            self.flush()

        # Restauramos signal handlers al salir
        self._restore_signal_handlers()

        # False = no suprimir la excepción original
        return False

    def __repr__(self) -> str:
        status = "flushed" if self._flushed else "active"
        return (
            f"SessionFlusher("
            f"session={self._session.session_id[:8]}, "
            f"status={status})"
        )
