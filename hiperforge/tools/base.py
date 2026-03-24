"""
Base de todas las tools de HiperForge.

Este módulo define tres cosas que trabajan juntas:

  1. BaseTool — clase base que todas las tools concretas heredan.
     Implementa ToolPort y agrega comportamiento transversal:
     validación de argumentos, truncado de output, logging y
     emisión de eventos al EventBus — automático en todas las tools.

  2. ToolRegistry — registro centralizado de tools disponibles.
     Las tools se registran automáticamente al importarse.
     El executor y el planner consultan el registry para saber
     qué tools existen y cuáles están habilitadas.

  3. @register_tool — decorador que registra una tool en el registry
     al momento de definir la clase. Agregar una tool nueva al sistema
     es tan simple como decorar su clase con @register_tool.

FLUJO COMPLETO DE UNA TOOL CALL EN EL LOOP REACT:

  LLM decide usar "shell" con {"command": "pytest tests/"}
         │
         ▼
  ToolRegistry.get("shell")          → resuelve ShellTool
         │
         ▼
  BaseTool.execute_safe(arguments)   → punto de entrada siempre seguro
    ├── validate_arguments()          → verifica campos requeridos
    ├── is_safe_to_run()              → verifica seguridad
    ├── _emit_tool_called_event()     → EventBus notifica a CLI y logger
    ├── execute(arguments)            → implementación concreta de ShellTool
    ├── _truncate_output()            → limita chars para no saturar tokens
    ├── _emit_tool_result_event()     → EventBus notifica resultado
    └── return ToolResult             → nunca lanza excepciones al caller

REGLA FUNDAMENTAL:
  execute_safe() NUNCA lanza excepciones.
  Cualquier error se convierte en ToolResult.failure().
  Esto es lo que hace al loop ReAct resistente a fallos de tools.
"""

from __future__ import annotations

import time
from abc import abstractmethod
from typing import Any

from hiperforge.core.constants import (
    TOOL_DEFAULT_TIMEOUT_SECONDS,
    TOOL_MAX_OUTPUT_CHARS,
)
from hiperforge.core.events import AgentEvent, get_event_bus
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolCall, ToolResult
from hiperforge.domain.exceptions import ToolNotFound, ToolTimeoutError
from hiperforge.domain.ports.tool_port import ToolPort, ToolSchema

logger = get_logger(__name__)


class BaseTool(ToolPort):
    """
    Clase base para todas las tools concretas de HiperForge.

    Implementa ToolPort y agrega comportamiento transversal que
    todas las tools necesitan sin tener que repetirlo:

      - Validación de argumentos antes de ejecutar
      - Truncado automático de output largo
      - Logging estructurado de cada ejecución
      - Emisión de eventos al EventBus
      - Captura de excepciones para convertirlas en ToolResult.failure()

    LAS TOOLS CONCRETAS DEBEN IMPLEMENTAR:
      - name       (property) → identificador único, snake_case
      - description (property) → descripción corta para logs
      - execute()              → lógica real de la tool
      - schema()               → descripción para el LLM

    LAS TOOLS CONCRETAS PUEDEN SOBREESCRIBIR:
      - validate_arguments()   → validaciones adicionales específicas
      - is_safe_to_run()       → checks de seguridad para ops destructivas
      - _max_output_chars      → límite de output (default: TOOL_MAX_OUTPUT_CHARS)

    NUNCA SOBREESCRIBIR:
      - execute_safe()         → punto de entrada del executor, maneja todo
    """

    # Límite de caracteres del output enviado al LLM.
    # Las subclases pueden sobreescribir para tools con output más largo.
    _max_output_chars: int = TOOL_MAX_OUTPUT_CHARS

    # Contexto de la sesión activa — seteado por el executor al inicio de cada subtask
    _task_id: str | None = None
    _subtask_id: str | None = None

    # ------------------------------------------------------------------
    # Punto de entrada principal — NUNCA sobreescribir
    # ------------------------------------------------------------------

    def execute_safe(
        self,
        arguments: dict[str, Any],
        *,
        tool_call_id: str | None = None,
    ) -> ToolResult:
        """
        Ejecuta la tool de forma completamente segura.

        Este es el método que el executor SIEMPRE debe llamar —
        nunca llamar execute() directamente desde fuera de la tool.

        Garantías:
          1. Nunca lanza excepciones al caller.
          2. Valida argumentos antes de ejecutar.
          3. Trunca el output si es muy largo.
          4. Emite eventos al EventBus antes y después.
          5. Loggea cada ejecución con métricas de duración.

        Parámetros:
            arguments:    Argumentos del LLM para esta tool.
            tool_call_id: ID del ToolCall asociado (para eventos y logs).
                          None si se llama fuera del loop ReAct (tests, etc.).

        Returns:
            ToolResult.success() o ToolResult.failure() — nunca lanza.
        """
        call_id = tool_call_id or "direct"
        start_time = time.monotonic()

        # Paso 1: validar argumentos antes de intentar ejecutar
        validation_errors = self.validate_arguments(arguments)
        if validation_errors:
            error_msg = f"Argumentos inválidos: {'; '.join(validation_errors)}"
            logger.warning(
                "tool rechazada por argumentos inválidos",
                tool=self.name,
                errors=validation_errors,
                task_id=self._task_id,
            )
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=error_msg,
                output="",
            )

        # Paso 2: verificar seguridad antes de ejecutar
        if not self.is_safe_to_run(arguments):
            error_msg = (
                f"La tool '{self.name}' requiere confirmación del usuario "
                f"para estos argumentos."
            )
            logger.warning(
                "tool bloqueada por verificación de seguridad",
                tool=self.name,
                task_id=self._task_id,
            )
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=error_msg,
                output="",
            )

        # Paso 3: emitir evento de inicio al EventBus
        self._emit_tool_called_event(call_id, arguments)

        # Paso 4: ejecutar la tool capturando cualquier excepción
        try:
            result = self.execute(arguments)

        except ToolTimeoutError:
            # ToolTimeoutError es la única excepción que se propaga
            # El executor la captura para manejar el timeout del loop ReAct
            duration = round(time.monotonic() - start_time, 3)
            logger.error(
                "tool timeout",
                tool=self.name,
                duration_seconds=duration,
                task_id=self._task_id,
            )
            raise

        except Exception as exc:
            # Cualquier otra excepción se convierte en ToolResult.failure()
            # Nunca debe romper el loop ReAct
            duration = round(time.monotonic() - start_time, 3)
            error_msg = f"{type(exc).__name__}: {exc}"

            logger.error(
                "excepción inesperada en tool",
                tool=self.name,
                error_type=type(exc).__name__,
                error=str(exc),
                duration_seconds=duration,
                task_id=self._task_id,
            )

            result = ToolResult.failure(
                tool_call_id=call_id,
                error_message=error_msg,
                output="",
            )

        duration = round(time.monotonic() - start_time, 3)

        # Paso 5: truncar output si excede el límite
        result = self._truncate_output_if_needed(result)

        # Paso 6: logging estructurado de la ejecución completada
        self._log_execution(result=result, duration=duration)

        # Paso 7: emitir evento de resultado al EventBus
        self._emit_tool_result_event(result=result, duration=duration)

        return result

    # ------------------------------------------------------------------
    # Métodos abstractos — implementar en cada tool concreta
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """Identificador único de la tool en snake_case."""
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """Descripción corta para logs internos."""
        ...

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Lógica real de la tool.

        REGLAS PARA IMPLEMENTADORES:
          - Capturar todas las excepciones propias y devolverlas como
            ToolResult.failure() con un mensaje claro.
          - Solo dejar propagar ToolTimeoutError.
          - El output debe ser texto plano legible — el LLM lo leerá.
          - No emitir eventos ni hacer logging aquí — execute_safe() lo hace.
        """
        ...

    @abstractmethod
    def schema(self) -> ToolSchema:
        """Schema de la tool para el LLM."""
        ...

    # ------------------------------------------------------------------
    # Context binding — el executor setea esto al inicio de cada subtask
    # ------------------------------------------------------------------

    def bind_context(self, task_id: str, subtask_id: str | None = None) -> None:
        """
        Vincula la tool al contexto de la task/subtask activa.

        El executor lo llama al inicio de cada subtask para que los
        eventos y logs emitidos por la tool lleven los IDs correctos.
        """
        self._task_id = task_id
        self._subtask_id = subtask_id

    # ------------------------------------------------------------------
    # Helpers privados del base
    # ------------------------------------------------------------------

    def _truncate_output_if_needed(self, result: ToolResult) -> ToolResult:
        """
        Trunca el output si excede _max_output_chars.

        El LLM tiene un context window limitado. Un output de 100KB
        de un comando verbose desperdiciaría miles de tokens.
        Truncamos y agregamos un aviso al final para que el LLM sepa
        que el output fue cortado y pueda pedir más si lo necesita.
        """
        if len(result.output) <= self._max_output_chars:
            return result

        truncated_output = (
            result.output[: self._max_output_chars]
            + f"\n\n[Output truncado — {len(result.output)} chars totales, "
            f"mostrando primeros {self._max_output_chars}]"
        )

        logger.debug(
            "output de tool truncado",
            tool=self.name,
            original_chars=len(result.output),
            truncated_chars=self._max_output_chars,
        )

        # Reconstruimos el ToolResult con el output truncado
        if result.success:
            return ToolResult.success(
                tool_call_id=result.tool_call_id,
                output=truncated_output,
            )
        return ToolResult.failure(
            tool_call_id=result.tool_call_id,
            error_message=result.error_message,
            output=truncated_output,
        )

    def _emit_tool_called_event(
        self,
        tool_call_id: str,
        arguments: dict[str, Any],
    ) -> None:
        """Emite TOOL_CALLED al EventBus antes de ejecutar."""
        get_event_bus().emit(
            AgentEvent.tool_called(
                task_id=self._task_id or "",
                subtask_id=self._subtask_id or "",
                tool_call_id=tool_call_id,
                tool_name=self.name,
                arguments=arguments,
            )
        )

    def _emit_tool_result_event(
        self,
        result: ToolResult,
        duration: float,
    ) -> None:
        """Emite TOOL_RESULT_RECEIVED al EventBus después de ejecutar."""
        get_event_bus().emit(
            AgentEvent.tool_result_received(
                task_id=self._task_id or "",
                subtask_id=self._subtask_id or "",
                tool_call_id=result.tool_call_id,
                tool_name=self.name,
                success=result.success,
                duration_seconds=duration,
                output_preview=result.output[:200],
            )
        )

    def _log_execution(self, result: ToolResult, duration: float) -> None:
        """Logging estructurado de cada ejecución."""
        log_fn = logger.info if result.success else logger.warning
        log_fn(
            "tool ejecutada",
            tool=self.name,
            success=result.success,
            duration_seconds=duration,
            output_chars=len(result.output),
            error=result.error_message if not result.success else None,
            task_id=self._task_id,
            subtask_id=self._subtask_id,
        )


# ---------------------------------------------------------------------------
# ToolRegistry — registro centralizado de todas las tools disponibles
# ---------------------------------------------------------------------------

class ToolRegistry:
    """
    Registro centralizado de tools del agente.

    Mantiene un mapa de nombre → instancia de tool.
    Las tools se registran automáticamente via @register_tool
    al momento de definir la clase.

    El registry es un singleton — hay una sola instancia por proceso.
    Acceder a él via get_tool_registry().

    DISEÑO INTENCIONAL:
      El registry almacena INSTANCIAS, no clases. Esto significa que
      cada tool es un singleton también — se crea una vez y se reutiliza.
      Las tools no tienen estado mutable entre llamadas (excepto el
      context binding de task_id/subtask_id que se setea antes de cada uso).
    """

    def __init__(self) -> None:
        # Mapa nombre_tool → instancia_tool
        self._tools: dict[str, BaseTool] = {}

    def register(self, tool: BaseTool) -> None:
        """
        Registra una instancia de tool en el registry.

        Si ya existe una tool con el mismo nombre, la sobreescribe.
        Esto permite reemplazar tools en tests sin cambiar el código.

        Parámetros:
            tool: Instancia de la tool a registrar.
        """
        if tool.name in self._tools:
            logger.warning(
                "sobreescribiendo tool ya registrada",
                tool_name=tool.name,
                existing_type=type(self._tools[tool.name]).__name__,
                new_type=type(tool).__name__,
            )

        self._tools[tool.name] = tool

        logger.debug(
            "tool registrada",
            tool_name=tool.name,
            tool_type=type(tool).__name__,
        )

    def get(self, name: str) -> BaseTool:
        """
        Obtiene una tool por su nombre.

        Raises:
            ToolNotFound: Si no hay ninguna tool registrada con ese nombre.
                          El mensaje incluye la lista de tools disponibles
                          para facilitar el debug cuando el LLM usa un nombre incorrecto.
        """
        tool = self._tools.get(name)

        if tool is None:
            available = sorted(self._tools.keys())
            raise ToolNotFound(
                tool_name=name,
            )

        return tool

    def get_all(self) -> list[BaseTool]:
        """
        Devuelve todas las tools registradas.

        Usado por el planner para incluir los schemas de todas las
        tools disponibles en el prompt de sistema del LLM.
        """
        return list(self._tools.values())

    def get_schemas(self) -> list[ToolSchema]:
        """
        Devuelve los schemas de todas las tools registradas.

        Usado por context_builder.py para construir el prompt
        de sistema que describe las tools disponibles al LLM.
        """
        return [tool.schema() for tool in self._tools.values()]

    def is_registered(self, name: str) -> bool:
        """True si existe una tool registrada con ese nombre."""
        return name in self._tools

    def unregister(self, name: str) -> bool:
        """
        Elimina una tool del registry.

        Devuelve True si existía y fue eliminada.
        Devuelve False si no existía.

        Usado principalmente en tests para limpiar el estado.
        """
        if name in self._tools:
            del self._tools[name]
            return True
        return False

    def bind_context_to_all(self, task_id: str, subtask_id: str | None = None) -> None:
        """
        Propaga el contexto de task/subtask a todas las tools registradas.

        El executor llama esto al inicio de cada subtask.
        Todas las tools emitirán eventos con los IDs correctos
        sin que el executor tenga que iterar y bindear una por una.
        """
        for tool in self._tools.values():
            tool.bind_context(task_id=task_id, subtask_id=subtask_id)

    @property
    def tool_count(self) -> int:
        """Número de tools registradas."""
        return len(self._tools)

    @property
    def tool_names(self) -> list[str]:
        """Nombres de todas las tools registradas, ordenados."""
        return sorted(self._tools.keys())

    def __repr__(self) -> str:
        return f"ToolRegistry(tools={self.tool_names})"


# ---------------------------------------------------------------------------
# Singleton del ToolRegistry
# ---------------------------------------------------------------------------

_registry_instance: ToolRegistry | None = None


def get_tool_registry() -> ToolRegistry:
    """
    Devuelve la instancia única del ToolRegistry (singleton).

    Todas las tools registradas via @register_tool van al mismo registry.
    El executor y el planner acceden a él via esta función.
    """
    global _registry_instance
    if _registry_instance is None:
        _registry_instance = ToolRegistry()
    return _registry_instance


def reset_tool_registry() -> None:
    """
    Destruye el singleton y crea uno nuevo vacío.

    SOLO para tests — permite que cada test empiece con un registry
    limpio sin tools del test anterior.

    Ejemplo en conftest.py:
        @pytest.fixture(autouse=True)
        def clean_registry():
            reset_tool_registry()
            yield
            reset_tool_registry()
    """
    global _registry_instance
    _registry_instance = None


# ---------------------------------------------------------------------------
# Decorador @register_tool
# ---------------------------------------------------------------------------

def register_tool(cls: type[BaseTool]) -> type[BaseTool]:
    """
    Decorador que registra automáticamente una tool al definir la clase.

    Al aplicar este decorador, la tool se instancia y registra en el
    ToolRegistry global. No se necesita hacer nada más — solo importar
    el módulo donde está la tool es suficiente para que esté disponible.

    IMPORTANTE: El módulo de la tool debe importarse en algún punto
    antes de que el executor intente usarla. La convención es importar
    todas las tools en tools/__init__.py.

    Uso:
        @register_tool
        class ShellTool(BaseTool):
            ...

    Equivalente a:
        class ShellTool(BaseTool):
            ...
        get_tool_registry().register(ShellTool())

    Raises:
        TypeError: Si la clase decorada no hereda de BaseTool.
    """
    if not issubclass(cls, BaseTool):
        raise TypeError(
            f"@register_tool solo puede aplicarse a subclases de BaseTool. "
            f"'{cls.__name__}' no hereda de BaseTool."
        )

    # Instanciamos la tool y la registramos en el registry global
    instance = cls()
    get_tool_registry().register(instance)

    logger.debug(
        "tool auto-registrada via decorador",
        tool_name=instance.name,
        tool_class=cls.__name__,
    )

    return cls