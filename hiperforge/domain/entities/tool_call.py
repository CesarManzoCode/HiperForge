"""
Entidades: ToolCall y ToolResult

Representan una llamada a una herramienta y su resultado.
Son el registro histórico de cada acción que el agente ejecutó.

¿Cuál es la diferencia con un value object?
  ToolCall tiene identidad propia — dos llamadas a la misma tool con los
  mismos argumentos son entidades DISTINTAS si ocurrieron en momentos
  diferentes. El `id` es lo que las diferencia.

¿Cómo fluye la información?
  1. El LLM decide usar una tool → se crea un ToolCall con status=PENDING.
  2. El agente ejecuta la tool → se crea un ToolResult con el output.
  3. El ToolCall se actualiza a COMPLETED o FAILED con el resultado.

  ToolCall              ToolResult
  ──────────────────    ─────────────────────
  id                    tool_call_id  ←── referencia al ToolCall
  tool_name             output
  arguments             success
  status                error_message
  created_at            executed_at
  result ──────────────→ (el mismo ToolResult)

USO TÍPICO:
  # El LLM solicita ejecutar una tool
  call = ToolCall.create(
      tool_name="shell",
      arguments={"command": "pytest tests/", "timeout": 30},
  )

  # La tool se ejecuta y devuelve un resultado
  result = ToolResult.success(
      tool_call_id=call.id,
      output="5 passed in 0.42s",
  )

  # Registramos el resultado en el ToolCall
  completed_call = call.with_result(result)
  print(completed_call.status)  # ToolCallStatus.COMPLETED
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.core.utils.ids import generate_id


class ToolCallStatus(str, Enum):
    """
    Estado del ciclo de vida de una llamada a una tool.

    Transiciones válidas:
      PENDING → RUNNING → COMPLETED
      PENDING → RUNNING → FAILED
      PENDING → SKIPPED  (si el plan se cancela antes de ejecutar)
    """

    PENDING = "pending"       # Creada, esperando ejecución
    RUNNING = "running"       # Ejecutándose en este momento
    COMPLETED = "completed"   # Terminó exitosamente
    FAILED = "failed"         # Terminó con error
    SKIPPED = "skipped"       # No se ejecutó (plan cancelado)


# Mapa explícito de transiciones válidas.
# Si una transición no está aquí, está prohibida.
_VALID_TRANSITIONS: dict[ToolCallStatus, frozenset[ToolCallStatus]] = {
    ToolCallStatus.PENDING:    frozenset({ToolCallStatus.RUNNING, ToolCallStatus.SKIPPED}),
    ToolCallStatus.RUNNING:    frozenset({ToolCallStatus.COMPLETED, ToolCallStatus.FAILED}),
    ToolCallStatus.COMPLETED:  frozenset(),  # estado terminal
    ToolCallStatus.FAILED:     frozenset(),  # estado terminal
    ToolCallStatus.SKIPPED:    frozenset(),  # estado terminal
}


@dataclass(frozen=True)
class ToolResult:
    """
    Resultado inmutable de ejecutar una tool.

    Es un value object dentro del contexto de ToolCall — no tiene
    ciclo de vida propio, existe como parte del ToolCall.

    Atributos:
        tool_call_id:  ID del ToolCall al que pertenece este resultado.
        output:        Salida de la tool (stdout, contenido leído, respuesta HTTP, etc.).
        success:       True si la tool terminó sin errores.
        error_message: Descripción del error si success=False. None si fue exitoso.
        executed_at:   Momento exacto en que terminó la ejecución (UTC).
    """

    tool_call_id: str
    output: str
    success: bool
    error_message: str | None = None
    executed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    # ------------------------------------------------------------------
    # Constructores semánticos — más expresivos que el constructor directo
    # ------------------------------------------------------------------

    @classmethod
    def success(cls, tool_call_id: str, output: str) -> ToolResult:
        """Crea un resultado exitoso."""
        return cls(
            tool_call_id=tool_call_id,
            output=output,
            success=True,
        )

    @classmethod
    def failure(cls, tool_call_id: str, error_message: str, *, output: str = "") -> ToolResult:
        """
        Crea un resultado fallido.

        output puede contener el stderr o lo que se alcanzó a ejecutar
        antes del error — útil para debug.
        """
        return cls(
            tool_call_id=tool_call_id,
            output=output,
            success=False,
            error_message=error_message,
        )

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardar en JSON."""
        return {
            "tool_call_id": self.tool_call_id,
            "output": self.output,
            "success": self.success,
            "error_message": self.error_message,
            "executed_at": self.executed_at.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolResult:
        """Reconstruye desde un diccionario leído del JSON."""
        return cls(
            tool_call_id=data["tool_call_id"],
            output=data["output"],
            success=data["success"],
            error_message=data.get("error_message"),
            executed_at=datetime.fromisoformat(data["executed_at"]),
        )

    def __str__(self) -> str:
        """
        Ejemplo exitoso: [OK] shell → "5 passed in 0.42s" (truncado a 60 chars)
        Ejemplo fallido: [FAIL] shell → "Command not found: pytest"
        """
        status_label = "OK" if self.success else "FAIL"
        preview = self.output[:60] + "..." if len(self.output) > 60 else self.output
        return f"[{status_label}] → {preview!r}"


@dataclass(frozen=True)
class ToolCall:
    """
    Registro de una llamada a una herramienta solicitada por el LLM.

    Inmutable — cada cambio de estado produce un nuevo ToolCall.
    Esto garantiza que el historial completo sea reconstruible.

    Atributos:
        id:         Identificador único (ULID).
        tool_name:  Nombre de la tool en el ToolRegistry (ej: "shell", "file").
        arguments:  Argumentos que el LLM decidió pasar a la tool.
        status:     Estado actual del ciclo de vida.
        created_at: Momento en que el LLM solicitó la tool (UTC).
        result:     Resultado de la ejecución. None hasta que termine.
    """

    id: str
    tool_name: str
    arguments: dict[str, Any]
    status: ToolCallStatus
    created_at: datetime
    result: ToolResult | None = None

    # ------------------------------------------------------------------
    # Constructor principal — siempre crear con .create(), nunca directo
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, tool_name: str, arguments: dict[str, Any]) -> ToolCall:
        """
        Crea un nuevo ToolCall en estado PENDING.

        Este es el único punto de entrada para crear ToolCalls nuevos.
        Genera el ID automáticamente para evitar IDs duplicados o vacíos.
        """
        return cls(
            id=generate_id(),
            tool_name=tool_name,
            arguments=arguments,
            status=ToolCallStatus.PENDING,
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Métodos de transición de estado — producen nuevas instancias
    # ------------------------------------------------------------------

    def _transition_to(self, new_status: ToolCallStatus) -> ToolCall:
        """
        Método interno que valida y ejecuta una transición de estado.

        Centraliza la lógica de validación para que los métodos públicos
        (mark_running, with_result, mark_failed, mark_skipped) sean simples.

        Raises:
            InvalidStatusTransition: Si la transición no está permitida.
        """
        # Importación local para evitar circular imports entre domain modules
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _VALID_TRANSITIONS[self.status]

        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"ToolCall({self.id})",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        # dataclass frozen=True → usamos object.__setattr__ para crear una copia
        # En realidad creamos una nueva instancia con replace-like pattern
        return ToolCall(
            id=self.id,
            tool_name=self.tool_name,
            arguments=self.arguments,
            status=new_status,
            created_at=self.created_at,
            result=self.result,
        )

    def mark_running(self) -> ToolCall:
        """
        Marca la tool como en ejecución.
        Devuelve un nuevo ToolCall con status=RUNNING.
        """
        return self._transition_to(ToolCallStatus.RUNNING)

    def with_result(self, result: ToolResult) -> ToolCall:
        """
        Registra el resultado y marca la tool como completada o fallida
        según el campo `success` del resultado.

        Devuelve un nuevo ToolCall con el resultado adjunto.
        """
        new_status = (
            ToolCallStatus.COMPLETED if result.success else ToolCallStatus.FAILED
        )

        # Validamos la transición manualmente aquí porque necesitamos
        # adjuntar el result además de cambiar el status
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"ToolCall({self.id})",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        return ToolCall(
            id=self.id,
            tool_name=self.tool_name,
            arguments=self.arguments,
            status=new_status,
            created_at=self.created_at,
            result=result,
        )

    def mark_skipped(self) -> ToolCall:
        """
        Marca la tool como omitida (plan cancelado antes de ejecutarse).
        Devuelve un nuevo ToolCall con status=SKIPPED.
        """
        return self._transition_to(ToolCallStatus.SKIPPED)

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True si el ToolCall ya no puede cambiar de estado."""
        return not _VALID_TRANSITIONS[self.status]

    @property
    def duration_seconds(self) -> float | None:
        """
        Duración de la ejecución en segundos.
        None si la tool no ha terminado todavía.
        """
        if self.result is None:
            return None
        delta = self.result.executed_at - self.created_at
        return round(delta.total_seconds(), 3)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardar en JSON."""
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "result": self.result.to_dict() if self.result else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ToolCall:
        """Reconstruye desde un diccionario leído del JSON."""
        result = (
            ToolResult.from_dict(data["result"])
            if data.get("result") is not None
            else None
        )

        return cls(
            id=data["id"],
            tool_name=data["tool_name"],
            arguments=data["arguments"],
            status=ToolCallStatus(data["status"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            result=result,
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo pendiente: ToolCall(shell, PENDING) args={"command": "pytest..."}
        Ejemplo completo:  ToolCall(shell, COMPLETED, 0.42s)
        """
        duration = f", {self.duration_seconds}s" if self.duration_seconds is not None else ""
        args_preview = str(self.arguments)[:60]
        return (
            f"ToolCall({self.tool_name}, {self.status.value}{duration})"
            f" args={args_preview}"
        )