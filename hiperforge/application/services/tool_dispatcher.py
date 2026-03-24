"""
ToolDispatcher — Intermediario entre el LLM y las tools del agente.

El dispatcher es la capa que separa el loop ReAct (executor) de las
implementaciones concretas de las tools. Cuando el LLM solicita ejecutar
"shell" con ciertos argumentos, el dispatcher:

  1. Verifica que la tool existe en el ToolRegistry.
  2. Gestiona el ciclo de vida del ToolCall (PENDING → RUNNING → COMPLETED/FAILED).
  3. Ejecuta la tool via execute_safe() — nunca lanza excepciones excepto Timeout.
  4. Devuelve el par (ToolCall completado, ToolResult) al executor.

¿POR QUÉ UN DISPATCHER Y NO LLAMAR AL REGISTRY DIRECTAMENTE?
  El executor tiene una responsabilidad enorme — gestionar el loop ReAct
  completo, el historial de mensajes, los eventos y el estado de las subtasks.
  Si además tuviera que gestionar el ciclo de vida de cada ToolCall,
  se convertiría en un God Object.

  El dispatcher extrae esa responsabilidad:
    Executor  → "necesito ejecutar esta tool con estos argumentos"
    Dispatcher → "yo me encargo de todo lo demás"

  Esta separación hace que ambas clases sean más simples, testeables
  y fáciles de mantener.

GESTIÓN DEL CICLO DE VIDA DEL TOOLCALL:
  El dispatcher es el único lugar donde los ToolCalls transicionan
  a través de sus estados:

    ToolCall.create()      → status = PENDING
    tool_call.mark_running() → status = RUNNING
    running_call.with_result() → status = COMPLETED o FAILED

  El executor recibe el ToolCall ya en estado terminal — nunca lo modifica.

MANEJO DE ERRORES:
  ToolNotFound    → tool desconocida, devuelve ToolResult.failure() inmediato.
  ToolTimeoutError → timeout de la tool, se propaga al executor para
                     que decida si reintenta o falla la subtask.
  Cualquier otro error → capturado por execute_safe() y devuelto como
                         ToolResult.failure().

  En ningún caso el dispatcher deja que un error rompa el loop ReAct —
  excepto ToolTimeoutError, que es la señal para que el executor actúe.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolCall, ToolResult
from hiperforge.domain.exceptions import ToolNotFound, ToolTimeoutError
from hiperforge.infrastructure.llm.base import ToolCallRequest
from hiperforge.tools.base import ToolRegistry

logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Resultado del dispatch — contiene todo lo que el executor necesita
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class DispatchResult:
    """
    Resultado completo de ejecutar una tool via el dispatcher.

    Encapsula el ToolCall (con su ciclo de vida completo) y el ToolResult
    en un solo objeto que el executor puede manejar atomicamente.

    Atributos:
        tool_call:      ToolCall en estado terminal (COMPLETED o FAILED).
                        Incluye el resultado incrustado via with_result().
        result:         ToolResult con el output y status de la ejecución.
        duration_seconds: Tiempo total desde PENDING hasta terminal.
        was_timeout:    True si la tool falló por timeout.
                        El executor usa esto para decidir si reintentar.
    """

    tool_call: ToolCall
    result: ToolResult
    duration_seconds: float
    was_timeout: bool = False

    @property
    def succeeded(self) -> bool:
        """True si la tool se ejecutó exitosamente."""
        return self.result.success

    def __str__(self) -> str:
        status = "OK" if self.succeeded else "FAIL"
        timeout_label = " [TIMEOUT]" if self.was_timeout else ""
        return (
            f"DispatchResult({self.tool_call.tool_name}, "
            f"{status}{timeout_label}, "
            f"{self.duration_seconds}s)"
        )


# ---------------------------------------------------------------------------
# ToolDispatcher
# ---------------------------------------------------------------------------

class ToolDispatcher:
    """
    Resuelve y ejecuta tools durante el loop ReAct.

    Gestiona el ciclo de vida completo de cada ToolCall:
    creación → running → resultado.

    Parámetros:
        registry: ToolRegistry con todas las tools disponibles.
    """

    def __init__(self, registry: ToolRegistry) -> None:
        self._registry = registry

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def dispatch(
        self,
        request: ToolCallRequest,
        *,
        task_id: str,
        subtask_id: str,
    ) -> DispatchResult:
        """
        Ejecuta una tool solicitada por el LLM durante el loop ReAct.

        Proceso completo:
          1. Crear ToolCall en estado PENDING
          2. Verificar que la tool existe
          3. Bindear contexto (task_id, subtask_id) a la tool
          4. Transicionar a RUNNING
          5. Ejecutar via execute_safe()
          6. Transicionar a COMPLETED o FAILED según el resultado
          7. Devolver DispatchResult

        Parámetros:
            request:    Solicitud del LLM con nombre de tool y argumentos.
            task_id:    ID de la task activa — para logs, eventos y binding.
            subtask_id: ID de la subtask activa — ídem.

        Returns:
            DispatchResult con el ToolCall terminado y el ToolResult.

        Raises:
            ToolTimeoutError: Si la tool excedió su timeout.
                              El executor decide qué hacer ante un timeout —
                              puede reintentar, saltar la subtask o fallar.
                              Por eso no lo convertimos en failure silencioso.
        """
        dispatch_start = time.monotonic()

        # Paso 1: crear el ToolCall en estado PENDING
        tool_call = ToolCall.create(
            tool_name=request.tool_name,
            arguments=request.arguments,
        )

        logger.debug(
            "dispatching tool",
            tool_name=request.tool_name,
            tool_call_id=tool_call.id,
            task_id=task_id,
            subtask_id=subtask_id,
            arguments_preview=str(request.arguments)[:150],
        )

        # Paso 2: verificar que la tool está registrada
        # Hacemos esto ANTES de transicionar a RUNNING para que el ToolCall
        # quede en FAILED limpiamente si la tool no existe
        try:
            tool = self._registry.get(request.tool_name)
        except ToolNotFound:
            result = self._build_tool_not_found_result(
                tool_call_id=tool_call.id,
                tool_name=request.tool_name,
            )
            # Transicionamos PENDING → RUNNING → FAILED en una secuencia limpia
            failed_call = tool_call.mark_running().with_result(result)
            duration = round(time.monotonic() - dispatch_start, 3)

            logger.warning(
                "tool no encontrada en el registry",
                tool_name=request.tool_name,
                available_tools=self._registry.tool_names,
                task_id=task_id,
                subtask_id=subtask_id,
            )

            return DispatchResult(
                tool_call=failed_call,
                result=result,
                duration_seconds=duration,
                was_timeout=False,
            )

        # Paso 3: bindear contexto a la tool para eventos y logs correctos
        tool.bind_context(task_id=task_id, subtask_id=subtask_id)

        # Paso 4: transicionar a RUNNING
        running_call = tool_call.mark_running()

        # Paso 5: ejecutar — ToolTimeoutError puede propagarse al executor
        try:
            result = tool.execute_safe(
                arguments=request.arguments,
                tool_call_id=tool_call.id,
            )

            # Paso 6: transicionar a COMPLETED o FAILED según el resultado
            completed_call = running_call.with_result(result)
            duration = round(time.monotonic() - dispatch_start, 3)

            log_fn = logger.info if result.success else logger.warning
            log_fn(
                "tool dispatched",
                tool_name=request.tool_name,
                tool_call_id=tool_call.id,
                success=result.success,
                duration_seconds=duration,
                output_chars=len(result.output),
                error=result.error_message if not result.success else None,
                task_id=task_id,
                subtask_id=subtask_id,
            )

            return DispatchResult(
                tool_call=completed_call,
                result=result,
                duration_seconds=duration,
                was_timeout=False,
            )

        except ToolTimeoutError as exc:
            # Timeout — construimos el resultado de fallo y PROPAGAMOS
            # El executor necesita saber que fue un timeout para decidir
            # si reintenta con extended_timeout=true u otra estrategia
            result = ToolResult.failure(
                tool_call_id=tool_call.id,
                error_message=(
                    f"La tool '{request.tool_name}' excedió el timeout. "
                    f"Si necesitas más tiempo, incluye extended_timeout=true "
                    f"en los argumentos (para shell) o usa un timeout mayor."
                ),
            )
            failed_call = running_call.with_result(result)
            duration = round(time.monotonic() - dispatch_start, 3)

            logger.error(
                "tool timeout en dispatcher",
                tool_name=request.tool_name,
                tool_call_id=tool_call.id,
                duration_seconds=duration,
                task_id=task_id,
                subtask_id=subtask_id,
            )

            # Devolvemos el DispatchResult con was_timeout=True
            # Y TAMBIÉN relanzamos para que el executor pueda reaccionar
            # El executor captura ToolTimeoutError y lee was_timeout del resultado
            raise ToolTimeoutError(
                tool_name=exc.tool_name,
                timeout_seconds=exc.context.get("timeout_seconds", 0),
            ) from exc

    def dispatch_and_format_for_llm(
        self,
        request: ToolCallRequest,
        *,
        task_id: str,
        subtask_id: str,
        format_result_fn: callable,
    ) -> tuple[DispatchResult, str]:
        """
        Ejecuta una tool y formatea el resultado para devolverlo al LLM.

        Variante de dispatch() que además aplica format_result_fn al resultado
        para producir el mensaje que se agrega al historial del LLM.

        El executor usa este método cuando quiere ejecutar la tool Y obtener
        el mensaje formateado en una sola operación.

        Parámetros:
            request:          Solicitud del LLM.
            task_id:          ID de la task activa.
            subtask_id:       ID de la subtask activa.
            format_result_fn: Función del adapter LLM que formatea el resultado.
                              Signature: (tool_call_id, tool_name, output, success) -> Message

        Returns:
            Tupla (DispatchResult, mensaje_formateado_para_llm).

        Raises:
            ToolTimeoutError: Si la tool excedió su timeout.
        """
        dispatch_result = self.dispatch(
            request=request,
            task_id=task_id,
            subtask_id=subtask_id,
        )

        # Formateamos el resultado usando el método del adapter LLM
        # Esto produce el mensaje correcto para cada proveedor
        formatted_message = format_result_fn(
            tool_call_id=dispatch_result.tool_call.id,
            tool_name=request.tool_name,
            output=dispatch_result.result.output,
            success=dispatch_result.result.success,
        )

        return dispatch_result, formatted_message

    # ------------------------------------------------------------------
    # Introspección — útil para el executor y para tests
    # ------------------------------------------------------------------

    def is_available(self, tool_name: str) -> bool:
        """
        Verifica si una tool está disponible en el registry.

        El executor puede usar esto para validar los nombres de tools
        del plan antes de empezar a ejecutar subtasks — dando feedback
        temprano si el LLM solicitó una tool que no existe.
        """
        return self._registry.is_registered(tool_name)

    def available_tool_names(self) -> list[str]:
        """
        Devuelve los nombres de todas las tools disponibles, ordenados.

        Usado en los mensajes de error para decirle al LLM exactamente
        qué tools puede usar.
        """
        return self._registry.tool_names

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _build_tool_not_found_result(
        self,
        tool_call_id: str,
        tool_name: str,
    ) -> ToolResult:
        """
        Construye un ToolResult descriptivo cuando la tool no existe.

        El mensaje incluye la lista de tools disponibles para que el LLM
        pueda corregir su siguiente solicitud usando un nombre válido.
        """
        available = self._registry.tool_names
        available_str = ", ".join(f'"{t}"' for t in available)

        return ToolResult.failure(
            tool_call_id=tool_call_id,
            error_message=(
                f"Tool '{tool_name}' no encontrada en el registry. "
                f"Tools disponibles: {available_str}. "
                f"Verifica el nombre exacto e intenta de nuevo."
            ),
        )