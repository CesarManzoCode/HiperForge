"""
Logging estructurado de HiperForge.

Este módulo configura structlog — una librería que produce logs
en formato JSON en producción y logs coloreados y legibles en desarrollo.

¿POR QUÉ STRUCTLOG Y NO EL LOGGING ESTÁNDAR DE PYTHON?
  El logging estándar produce strings como:
    "2024-01-15 14:32:01 INFO task ejecutada en 5.3s"

  Structlog produce dicts estructurados:
    {"timestamp": "2024-01-15T14:32:01Z", "level": "info",
     "event": "task ejecutada", "duration_seconds": 5.3,
     "task_id": "01HX4K...", "provider": "anthropic"}

  La diferencia crítica: los campos estructurados son filtrables y
  buscables. En producción puedes hacer:
    grep '"task_id": "01HX4K"' hiperforge.log
  y obtener TODOS los logs de esa task específica, en orden.

  Con strings planos tendrías que parsear texto con regex — frágil y lento.

DOS MODOS DE OUTPUT:
  Desarrollo (debug=True):
    14:32:01 [INFO] task ejecutada  task_id=01HX4K duration=5.3s
    Coloreado, una línea, fácil de leer en terminal.

  Producción (debug=False):
    {"timestamp":"2024-01-15T14:32:01Z","level":"info","event":"task ejecutada",...}
    JSON por línea, fácil de procesar con herramientas externas.

USO EN CUALQUIER MÓDULO:
  from hiperforge.core.logging import get_logger

  logger = get_logger(__name__)

  # Logging con contexto estructurado — los kwargs son campos del log
  logger.info("task completada", task_id=task.id, duration=5.3)
  logger.warning("rate limit alcanzado", provider="anthropic", retry_after=8.3)
  logger.error("tool falló", tool_name="shell", exit_code=1, command="pytest")

  # Binding de contexto persistente para un scope específico
  # Todos los logs del executor llevarán task_id automáticamente
  bound_logger = logger.bind(task_id=task.id, subtask_id=subtask.id)
  bound_logger.info("iniciando loop ReAct")   # incluye task_id y subtask_id
  bound_logger.info("tool ejecutada")         # también incluye task_id y subtask_id
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

import structlog

from hiperforge.core.constants import (
    APP_NAME,
    DIR_LOGS,
    LOG_FILENAME,
    LOG_LEVEL_DEFAULT,
    LOG_RETENTION,
    LOG_ROTATION,
)


def setup_logging(
    *,
    debug: bool = False,
    log_dir: Path | None = None,
) -> None:
    """
    Inicializa el sistema de logging para toda la aplicación.

    Debe llamarse UNA SOLA VEZ al arrancar HiperForge, antes de
    cualquier otra operación. El entrypoint CLI lo llama automáticamente.

    Configura dos destinos simultáneos:
      1. Terminal: formato legible (dev) o JSON compacto (prod).
      2. Archivo:  siempre JSON, con rotación diaria automática.

    Parámetros:
        debug:   Si True, usa formato colorido en terminal y nivel DEBUG.
        log_dir: Directorio para el archivo de log.
                 Default: DIR_LOGS (~/.hiperforge/logs/).
    """
    effective_log_dir = log_dir or DIR_LOGS
    log_level = "DEBUG" if debug else LOG_LEVEL_DEFAULT

    # ------------------------------------------------------------------
    # Configuramos el logging estándar de Python para capturar logs
    # de librerías externas (httpx, anthropic-sdk, etc.) y procesarlos
    # con structlog también
    # ------------------------------------------------------------------
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, log_level),
    )

    # ------------------------------------------------------------------
    # Procesadores compartidos — se aplican a todos los logs
    # Un procesador recibe el evento y lo transforma antes de renderizar
    # ------------------------------------------------------------------
    shared_processors: list[Any] = [
        # Agrega timestamp UTC a cada log
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        # Agrega el nivel del log como campo
        structlog.stdlib.add_log_level,
        # Agrega el nombre del logger (módulo que lo llamó)
        structlog.stdlib.add_logger_name,
        # Formatea excepciones si las hay
        structlog.processors.format_exc_info,
        # Convierte el stack trace a string limpio
        structlog.processors.StackInfoRenderer(),
    ]

    # ------------------------------------------------------------------
    # Renderer según el modo — colorido para dev, JSON para prod/archivo
    # ------------------------------------------------------------------
    if debug:
        # Formato colorido y legible para desarrollo en terminal
        renderer = structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        )
    else:
        # JSON por línea para producción — fácil de grep y parsear
        renderer = structlog.processors.JSONRenderer()

    # ------------------------------------------------------------------
    # Configuración global de structlog
    # ------------------------------------------------------------------
    structlog.configure(
        processors=[
            *shared_processors,
            # Convierte el evento a dict antes del renderer final
            structlog.processors.UnicodeDecoder(),
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, log_level)
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # ------------------------------------------------------------------
    # Configuramos el archivo de log con rotación usando loguru
    # (structlog escribe a stdout, loguru captura y rota el archivo)
    # ------------------------------------------------------------------
    _setup_file_logging(
        log_dir=effective_log_dir,
        log_level=log_level,
        shared_processors=shared_processors,
    )


def _setup_file_logging(
    *,
    log_dir: Path,
    log_level: str,
    shared_processors: list[Any],
) -> None:
    """
    Configura el handler de archivo con rotación diaria.

    Los logs en archivo siempre son JSON, sin importar el modo debug.
    Esto permite procesar los archivos con herramientas externas
    independientemente de si el dev usa debug o no.

    Parámetros:
        log_dir:           Directorio donde se escriben los logs.
        log_level:         Nivel mínimo de log a escribir en archivo.
        shared_processors: Procesadores compartidos ya configurados.
    """
    # Creamos el directorio si no existe
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / LOG_FILENAME

    # Handler de archivo con rotación — usando el logging estándar
    # para compatibilidad con structlog
    file_handler = logging.handlers.TimedRotatingFileHandler(
        filename=log_path,
        when="midnight",    # rotar a medianoche
        backupCount=30,     # guardar 30 días de logs históricos
        encoding="utf-8",
    )
    file_handler.setLevel(getattr(logging, log_level))

    # El archivo siempre escribe JSON
    file_formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            *shared_processors,
            structlog.processors.JSONRenderer(),
        ],
    )
    file_handler.setFormatter(file_formatter)

    # Agregamos el handler al logger raíz para capturar todos los logs
    logging.getLogger().addHandler(file_handler)


def get_logger(name: str) -> structlog.BoundLogger:
    """
    Devuelve un logger estructurado para un módulo específico.

    Usar siempre esta función para obtener loggers — nunca instanciar
    structlog directamente en los módulos.

    Parámetros:
        name: Nombre del módulo. Convención: usar __name__ siempre.
              Ejemplo: get_logger(__name__) en executor.py produce
              logs con logger_name="hiperforge.application.services.executor"

    Returns:
        Logger estructurado listo para usar con campos adicionales.

    Ejemplo:
        logger = get_logger(__name__)
        logger.info("subtask completada",
                    subtask_id="01HX4K...",
                    duration_seconds=3.2,
                    tool_calls=5)
    """
    return structlog.get_logger(name)


def get_agent_logger(
    task_id: str,
    *,
    subtask_id: str | None = None,
    provider: str | None = None,
) -> structlog.BoundLogger:
    """
    Devuelve un logger pre-configurado con el contexto del agente ReAct.

    Todos los logs emitidos con este logger incluirán automáticamente
    task_id, subtask_id y provider — sin necesidad de pasarlos en
    cada llamada individual.

    Usado principalmente por el executor del loop ReAct para que
    todos los logs de una ejecución sean fácilmente filtrables por task.

    Parámetros:
        task_id:    ID de la task que se está ejecutando.
        subtask_id: ID de la subtask activa. None si aún no hay subtask.
        provider:   Nombre del proveedor LLM activo.

    Returns:
        Logger con contexto del agente pre-bindeado.

    Ejemplo de log producido:
        {
          "timestamp": "2024-01-15T14:32:01Z",
          "level": "info",
          "event": "tool ejecutada",
          "task_id": "01HX4K...",
          "subtask_id": "01HX4M...",
          "provider": "anthropic",
          "tool_name": "shell",
          "duration_seconds": 0.42
        }
    """
    context: dict[str, Any] = {"task_id": task_id}

    if subtask_id is not None:
        context["subtask_id"] = subtask_id

    if provider is not None:
        context["provider"] = provider

    return structlog.get_logger(APP_NAME).bind(**context)


# ---------------------------------------------------------------------------
# Import necesario para el handler de archivo — agregado al final para
# no contaminar el namespace del módulo con imports de stdlib
# ---------------------------------------------------------------------------
import logging.handlers  # noqa: E402
