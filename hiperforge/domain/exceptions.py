"""
Jerarquía de excepciones del dominio de HiperForge.

REGLAS DE USO:
  - Nunca lanzar Exception() genérica. Siempre usar el tipo más específico.
  - Cada excepción debe llevar suficiente contexto para debuggear sin abrir los logs.
  - El mensaje debe ser legible por un humano, el context por una máquina.

ÁRBOL DE HERENCIA:
  HiperForgeError
  ├── DomainError
  │   ├── InvalidStatusTransition
  │   ├── EntityNotFound
  │   └── DuplicateEntity
  ├── PlanError
  │   ├── EmptyPlanError
  │   └── InvalidPlanError
  ├── ToolError
  │   ├── ToolNotFound
  │   ├── ToolExecutionError
  │   └── ToolTimeoutError
  ├── LLMError
  │   ├── LLMConnectionError
  │   ├── LLMRateLimitError
  │   └── LLMResponseError
  └── StorageError
      ├── StorageReadError
      ├── StorageWriteError
      └── StorageCorruptedError
"""

from __future__ import annotations


# ---------------------------------------------------------------------------
# Base — raíz de todo el árbol de errores propio
# ---------------------------------------------------------------------------

class HiperForgeError(Exception):
    """
    Clase base de todos los errores de HiperForge.

    Permite hacer `except HiperForgeError` para atrapar cualquier error
    del proyecto sin atrapar errores de Python o librerías externas.

    Parámetros:
        message: Descripción legible del error (para logs y UI).
        context: Diccionario con datos extra para debug (IDs, rutas, valores).
    """

    def __init__(self, message: str, *, context: dict | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.context: dict = context or {}

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(message={self.message!r}, context={self.context})"


# ---------------------------------------------------------------------------
# DomainError — violaciones de reglas de negocio
# ---------------------------------------------------------------------------

class DomainError(HiperForgeError):
    """Error en las reglas de negocio del dominio."""


class InvalidStatusTransition(DomainError):
    """
    Se intentó mover una entidad a un estado inválido.

    Ejemplo: pasar una Task de COMPLETED → PENDING no está permitido.
    """

    def __init__(self, entity: str, from_status: str, to_status: str) -> None:
        super().__init__(
            f"Transición inválida en '{entity}': '{from_status}' → '{to_status}'",
            context={
                "entity": entity,
                "from_status": from_status,
                "to_status": to_status,
            },
        )


class EntityNotFound(DomainError):
    """
    Se buscó una entidad por ID y no existe.

    Ejemplo: buscar Project con id='01HX...' que fue eliminado.
    """

    def __init__(self, entity_type: str, entity_id: str) -> None:
        super().__init__(
            f"{entity_type} con id '{entity_id}' no encontrado",
            context={
                "entity_type": entity_type,
                "entity_id": entity_id,
            },
        )


class DuplicateEntity(DomainError):
    """
    Se intentó crear una entidad que ya existe.

    Ejemplo: crear un Workspace con el mismo nombre que uno existente.
    """

    def __init__(self, entity_type: str, identifier: str) -> None:
        super().__init__(
            f"{entity_type} '{identifier}' ya existe",
            context={
                "entity_type": entity_type,
                "identifier": identifier,
            },
        )


# ---------------------------------------------------------------------------
# PlanError — errores al generar o validar el plan de ejecución
# ---------------------------------------------------------------------------

class PlanError(HiperForgeError):
    """Error al generar o validar un ExecutionPlan."""


class EmptyPlanError(PlanError):
    """
    El LLM devolvió un plan sin ninguna subtask.

    Guardamos solo los primeros 200 chars del prompt para no saturar los logs,
    pero con suficiente contexto para reproducir el problema.
    """

    def __init__(self, task_prompt: str) -> None:
        super().__init__(
            "El LLM devolvió un plan vacío (sin subtasks)",
            context={"task_prompt_preview": task_prompt[:200]},
        )


class InvalidPlanError(PlanError):
    """
    El plan tiene estructura inválida.

    Puede ser: campos faltantes, tipos incorrectos, JSON malformado, etc.
    """

    def __init__(self, reason: str, *, raw_response: str | None = None) -> None:
        super().__init__(
            f"Plan inválido: {reason}",
            context={
                "reason": reason,
                # Solo primeros 500 chars para debug sin saturar memoria
                "raw_response_preview": (raw_response or "")[:500],
            },
        )


# ---------------------------------------------------------------------------
# ToolError — errores durante la ejecución de herramientas
# ---------------------------------------------------------------------------

class ToolError(HiperForgeError):
    """
    Error base para todos los errores de tools.

    Siempre incluye el nombre de la tool para saber cuál falló
    sin tener que rastrear el stack completo.
    """

    def __init__(self, tool_name: str, message: str, *, context: dict | None = None) -> None:
        super().__init__(
            message,
            context={"tool_name": tool_name, **(context or {})},
        )
        self.tool_name = tool_name


class ToolNotFound(ToolError):
    """El agente intentó usar una tool que no está registrada en el ToolRegistry."""

    def __init__(self, tool_name: str) -> None:
        super().__init__(
            tool_name,
            f"La tool '{tool_name}' no está registrada",
        )


class ToolExecutionError(ToolError):
    """
    La tool se ejecutó pero falló durante su operación.

    Ejemplo: ShellTool ejecutó un comando que devolvió exit code != 0.
    """

    def __init__(self, tool_name: str, reason: str, *, exit_code: int | None = None) -> None:
        super().__init__(
            tool_name,
            f"Error ejecutando '{tool_name}': {reason}",
            context={"reason": reason, "exit_code": exit_code},
        )


class ToolTimeoutError(ToolError):
    """
    La tool excedió el tiempo máximo de ejecución permitido.

    Ejemplo: un comando de shell que nunca termina.
    """

    def __init__(self, tool_name: str, timeout_seconds: float) -> None:
        super().__init__(
            tool_name,
            f"La tool '{tool_name}' excedió el límite de {timeout_seconds}s",
            context={"timeout_seconds": timeout_seconds},
        )


# ---------------------------------------------------------------------------
# LLMError — errores de comunicación con el modelo de lenguaje
# ---------------------------------------------------------------------------

class LLMError(HiperForgeError):
    """
    Error base para todos los errores relacionados con el LLM.

    Siempre incluye el nombre del proveedor (anthropic, openai, ollama)
    para identificar cuál adapter falló sin revisar el stack.
    """

    def __init__(self, provider: str, message: str, *, context: dict | None = None) -> None:
        super().__init__(
            message,
            context={"provider": provider, **(context or {})},
        )
        self.provider = provider


class LLMConnectionError(LLMError):
    """No se pudo conectar al proveedor del LLM (red, URL incorrecta, servidor caído)."""

    def __init__(self, provider: str, reason: str) -> None:
        super().__init__(
            provider,
            f"No se pudo conectar a '{provider}': {reason}",
            context={"reason": reason},
        )


class LLMRateLimitError(LLMError):
    """
    El proveedor rechazó la petición por límite de requests o tokens.

    retry_after_seconds: cuánto esperar antes de reintentar.
    None significa que el proveedor no informó el tiempo de espera.
    """

    def __init__(self, provider: str, *, retry_after_seconds: float | None = None) -> None:
        super().__init__(
            provider,
            f"Rate limit alcanzado en '{provider}'",
            context={"retry_after_seconds": retry_after_seconds},
        )
        self.retry_after_seconds = retry_after_seconds


class LLMResponseError(LLMError):
    """
    El LLM respondió pero con contenido inesperado o inparseable.

    Ejemplo: se esperaba JSON y devolvió texto plano con disculpas.
    """

    def __init__(self, provider: str, reason: str, *, raw_response: str | None = None) -> None:
        super().__init__(
            provider,
            f"Respuesta inválida de '{provider}': {reason}",
            context={
                "reason": reason,
                "raw_response_preview": (raw_response or "")[:500],
            },
        )


# ---------------------------------------------------------------------------
# StorageError — errores de lectura/escritura del sistema de archivos JSON
# ---------------------------------------------------------------------------

class StorageError(HiperForgeError):
    """
    Error base para todos los errores de almacenamiento.

    Siempre incluye la ruta del archivo afectado para debug inmediato.
    """

    def __init__(self, path: str, message: str, *, context: dict | None = None) -> None:
        super().__init__(
            message,
            context={"path": path, **(context or {})},
        )
        self.path = path


class StorageReadError(StorageError):
    """No se pudo leer el archivo JSON (no existe, sin permisos, etc.)."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            path,
            f"Error leyendo '{path}': {reason}",
            context={"reason": reason},
        )


class StorageWriteError(StorageError):
    """No se pudo escribir el archivo JSON (disco lleno, sin permisos, etc.)."""

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            path,
            f"Error escribiendo '{path}': {reason}",
            context={"reason": reason},
        )


class StorageCorruptedError(StorageError):
    """
    El archivo JSON existe pero su contenido no pasa la validación de schema.

    Este es uno de los errores más críticos — indica corrupción de datos
    o un bug en el serializer. Requiere atención inmediata.
    """

    def __init__(self, path: str, reason: str) -> None:
        super().__init__(
            path,
            f"Datos corruptos en '{path}': {reason}",
            context={"reason": reason},
        )
