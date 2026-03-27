"""
Entidad: Task y Subtask

Task es la unidad central de trabajo en HiperForge.
Representa una instrucción del usuario que el agente debe completar
usando el ciclo ReAct: Razonar → Actuar → Observar → repetir.

¿Qué es una Subtask?
  Cuando el agente recibe una Task, primero la descompone en Subtasks —
  pasos más pequeños y concretos. Cada Subtask puede requerir varias
  iteraciones del ciclo ReAct antes de completarse.

  Task: "Agrega autenticación JWT al servidor"
    ├── Subtask 1: "Instalar dependencias necesarias"
    ├── Subtask 2: "Crear middleware de autenticación"
    ├── Subtask 3: "Proteger las rutas existentes"
    └── Subtask 4: "Escribir tests de integración"

  Cada Subtask acumula ToolCalls — cada vuelta del ciclo ReAct
  puede generar uno o más ToolCalls.

CICLO DE VIDA DE UNA TASK:
  PENDING → PLANNING → IN_PROGRESS → COMPLETED
  PENDING → PLANNING → IN_PROGRESS → FAILED
  PENDING → PLANNING → CANCELLED
  (cualquier estado no terminal → CANCELLED)

CICLO DE VIDA DE UNA SUBTASK:
  PENDING → IN_PROGRESS → COMPLETED
  PENDING → IN_PROGRESS → FAILED
  PENDING → SKIPPED

USO TÍPICO:
  # Crear una task nueva
  task = Task.create(
      prompt="Agrega autenticación JWT al servidor",
      project_id="01HX4K...",
  )

  # El agente genera el plan y lo registra
  task = task.start_planning()
  subtasks = [
      Subtask.create(task_id=task.id, description="Instalar dependencias"),
      Subtask.create(task_id=task.id, description="Crear middleware"),
  ]
  task = task.start_execution(subtasks)

  # El agente ejecuta cada subtask con su loop ReAct
  first_subtask = subtasks[0].mark_running()
  first_subtask = first_subtask.add_tool_call(tool_call)
  first_subtask = first_subtask.complete()

  # Al terminar todas las subtasks, completamos la task
  task = task.complete(summary="JWT implementado correctamente en 4 pasos")
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any

from hiperforge.core.utils.ids import generate_id
from hiperforge.domain.entities.tool_call import ToolCall
from hiperforge.domain.value_objects.token_usage import TokenUsage


class TaskStatus(str, Enum):
    """
    Estado del ciclo de vida de una Task.

    Transiciones válidas → ver _TASK_VALID_TRANSITIONS abajo.
    """

    PENDING = "pending"           # Creada, esperando procesamiento
    PLANNING = "planning"         # El agente está generando el plan de subtasks
    IN_PROGRESS = "in_progress"   # Ejecutando subtasks en loop ReAct
    COMPLETED = "completed"       # Todas las subtasks completadas exitosamente
    FAILED = "failed"             # Una subtask crítica falló sin recuperación
    CANCELLED = "cancelled"       # Cancelada por el usuario o por el sistema


class SubtaskStatus(str, Enum):
    """
    Estado del ciclo de vida de una Subtask.

    Transiciones válidas → ver _SUBTASK_VALID_TRANSITIONS abajo.
    """

    PENDING = "pending"           # Esperando su turno de ejecución
    IN_PROGRESS = "in_progress"   # Ejecutándose (loop ReAct activo)
    COMPLETED = "completed"       # Completada exitosamente
    FAILED = "failed"             # Falló tras agotar reintentos
    SKIPPED = "skipped"           # Omitida (task cancelada antes de llegar aquí)


# ---------------------------------------------------------------------------
# Transiciones de estado válidas — fuente de verdad del ciclo de vida
# ---------------------------------------------------------------------------

_TASK_VALID_TRANSITIONS: dict[TaskStatus, frozenset[TaskStatus]] = {
    TaskStatus.PENDING:      frozenset({TaskStatus.PLANNING, TaskStatus.CANCELLED}),
    TaskStatus.PLANNING:     frozenset({TaskStatus.IN_PROGRESS, TaskStatus.CANCELLED}),
    TaskStatus.IN_PROGRESS:  frozenset({TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED}),
    TaskStatus.COMPLETED:    frozenset(),   # terminal
    TaskStatus.FAILED:       frozenset(),   # terminal
    TaskStatus.CANCELLED:    frozenset(),   # terminal
}

_SUBTASK_VALID_TRANSITIONS: dict[SubtaskStatus, frozenset[SubtaskStatus]] = {
    SubtaskStatus.PENDING:      frozenset({SubtaskStatus.IN_PROGRESS, SubtaskStatus.SKIPPED}),
    SubtaskStatus.IN_PROGRESS:  frozenset({SubtaskStatus.COMPLETED, SubtaskStatus.FAILED}),
    SubtaskStatus.COMPLETED:    frozenset(),   # terminal
    SubtaskStatus.FAILED:       frozenset(),   # terminal
    SubtaskStatus.SKIPPED:      frozenset(),   # terminal
}


# ---------------------------------------------------------------------------
# Subtask
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Subtask:
    """
    Paso concreto dentro de una Task.

    Cada Subtask es ejecutada por el agente con su loop ReAct:
    el agente razona, ejecuta tools, observa el resultado y repite
    hasta que considera la subtask completada.

    Atributos:
        id:           Identificador único (ULID).
        task_id:      ID de la Task padre a la que pertenece.
        description:  Qué debe hacer el agente en este paso.
        order:        Posición en el plan (0-indexed). Determina el orden de ejecución.
        status:       Estado actual del ciclo de vida.
        tool_calls:   Historial de todas las ToolCalls del loop ReAct.
                      Guardada como tuple para mantener inmutabilidad.
        reasoning:    Último razonamiento del agente sobre esta subtask.
                      Se actualiza en cada iteración del loop ReAct.
        created_at:   Cuándo fue creada esta subtask (UTC).
        completed_at: Cuándo terminó (exitosa o con error). None si sigue activa.
    """

    id: str
    task_id: str
    description: str
    order: int
    status: SubtaskStatus
    tool_calls: tuple[ToolCall, ...]
    reasoning: str | None
    created_at: datetime
    completed_at: datetime | None = None

    # ------------------------------------------------------------------
    # Constructor principal
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, task_id: str, description: str, order: int) -> Subtask:
        """
        Crea una nueva Subtask en estado PENDING.

        Parámetros:
            task_id:     ID de la Task a la que pertenece.
            description: Instrucción concreta para el agente.
            order:       Posición en el plan de ejecución (empieza en 0).
        """
        return cls(
            id=generate_id(),
            task_id=task_id,
            description=description,
            order=order,
            status=SubtaskStatus.PENDING,
            tool_calls=(),      # sin tool calls al inicio
            reasoning=None,     # sin razonamiento hasta que el agente empiece
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Métodos de transición — cada uno devuelve una nueva instancia
    # ------------------------------------------------------------------

    def _transition_to(self, new_status: SubtaskStatus) -> Subtask:
        """Valida y ejecuta una transición de estado."""
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _SUBTASK_VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"Subtask({self.id})",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        # Marcamos completed_at solo al llegar a un estado terminal
        completed_at = (
            datetime.now(timezone.utc)
            if not _SUBTASK_VALID_TRANSITIONS[new_status]  # si el nuevo estado es terminal
            else None
        )

        return Subtask(
            id=self.id,
            task_id=self.task_id,
            description=self.description,
            order=self.order,
            status=new_status,
            tool_calls=self.tool_calls,
            reasoning=self.reasoning,
            created_at=self.created_at,
            completed_at=completed_at,
        )

    def mark_running(self) -> Subtask:
        """El agente empieza a trabajar en esta subtask."""
        return self._transition_to(SubtaskStatus.IN_PROGRESS)

    def complete(self) -> Subtask:
        """El agente considera esta subtask terminada exitosamente."""
        return self._transition_to(SubtaskStatus.COMPLETED)

    def fail(self) -> Subtask:
        """La subtask falló tras agotar los reintentos del loop ReAct."""
        return self._transition_to(SubtaskStatus.FAILED)

    def skip(self) -> Subtask:
        """La subtask fue omitida porque la task se canceló antes de llegar aquí."""
        return self._transition_to(SubtaskStatus.SKIPPED)

    def reset_to_pending(self) -> Subtask:
        """
        Crea una nueva instancia de Subtask en estado PENDING conservando la
        descripción y el orden, pero limpiando todo el estado de ejecución.

        Usado cuando el usuario elige "Retry" tras agotar las iteraciones.
        No es una transición de estado estándar — es un reset completo que
        genera una Subtask fresca como si nunca se hubiera ejecutado.

        NOTA: Se conserva el mismo ID para que el tracking de progreso
        en la task funcione correctamente (update_subtask busca por ID).
        """
        return Subtask(
            id=self.id,
            task_id=self.task_id,
            description=self.description,
            order=self.order,
            status=SubtaskStatus.PENDING,
            tool_calls=(),
            reasoning=None,
            created_at=self.created_at,
            completed_at=None,
        )

    # ------------------------------------------------------------------
    # Métodos de actualización ReAct — producen nuevas instancias
    # ------------------------------------------------------------------

    def add_tool_call(self, tool_call: ToolCall) -> Subtask:
        """
        Agrega un ToolCall al historial de esta subtask.

        Se llama en cada iteración del loop ReAct cuando el agente
        decide ejecutar una acción.
        """
        return Subtask(
            id=self.id,
            task_id=self.task_id,
            description=self.description,
            order=self.order,
            status=self.status,
            tool_calls=(*self.tool_calls, tool_call),   # nueva tupla con el call agregado
            reasoning=self.reasoning,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )

    def update_reasoning(self, reasoning: str) -> Subtask:
        """
        Actualiza el razonamiento del agente sobre esta subtask.

        Se llama en cada iteración del loop ReAct cuando el agente
        produce un nuevo pensamiento antes de actuar.
        """
        return Subtask(
            id=self.id,
            task_id=self.task_id,
            description=self.description,
            order=self.order,
            status=self.status,
            tool_calls=self.tool_calls,
            reasoning=reasoning,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True si la subtask ya no puede cambiar de estado."""
        return not _SUBTASK_VALID_TRANSITIONS[self.status]

    @property
    def total_token_usage(self) -> TokenUsage:
        """
        Suma el uso de tokens de todos los ToolCalls de esta subtask.

        Útil para medir el costo del loop ReAct de una subtask específica.
        """
        # Importación local para evitar circular imports
        total = TokenUsage.zero()
        for call in self.tool_calls:
            if call.result and hasattr(call.result, "token_usage"):
                total = total + call.result.token_usage
        return total

    @property
    def react_iterations(self) -> int:
        """
        Número de iteraciones completadas del loop ReAct en esta subtask.

        Cada ToolCall representa una acción en el loop, así que
        el número de tool calls es un proxy del número de iteraciones.
        """
        return len(self.tool_calls)

    @property
    def duration_seconds(self) -> float | None:
        """Duración total de la subtask. None si todavía está activa."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.created_at
        return round(delta.total_seconds(), 3)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardar en JSON."""
        return {
            "id": self.id,
            "task_id": self.task_id,
            "description": self.description,
            "order": self.order,
            "status": self.status.value,
            "tool_calls": [tc.to_dict() for tc in self.tool_calls],
            "reasoning": self.reasoning,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Subtask:
        """Reconstruye desde un diccionario leído del JSON."""
        return cls(
            id=data["id"],
            task_id=data["task_id"],
            description=data["description"],
            order=data["order"],
            status=SubtaskStatus(data["status"]),
            tool_calls=tuple(ToolCall.from_dict(tc) for tc in data.get("tool_calls", [])),
            reasoning=data.get("reasoning"),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
        )

    def __str__(self) -> str:
        """
        Ejemplo: [1] Instalar dependencias (IN_PROGRESS, 3 tool calls)
        """
        return (
            f"[{self.order}] {self.description}"
            f" ({self.status.value}, {self.react_iterations} tool calls)"
        )


# ---------------------------------------------------------------------------
# Task
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Task:
    """
    Unidad central de trabajo: una instrucción del usuario al agente.

    El agente procesa una Task así:
      1. PLANNING:     Descompone el prompt en Subtasks.
      2. IN_PROGRESS:  Ejecuta cada Subtask con su loop ReAct.
      3. COMPLETED:    Todas las Subtasks terminaron exitosamente.

    Atributos:
        id:           Identificador único (ULID).
        project_id:   ID del proyecto al que pertenece. None si es una task suelta.
        prompt:       Instrucción original del usuario, sin modificar.
        status:       Estado actual del ciclo de vida.
        subtasks:     Plan de ejecución generado durante PLANNING.
                      Tuple vacía hasta que el agente genere el plan.
        summary:      Resumen del resultado al completar la task.
                      None hasta que se complete.
        token_usage:  Consumo total de tokens de toda la task.
        created_at:   Cuándo fue creada (UTC).
        completed_at: Cuándo terminó. None si sigue activa.
    """

    id: str
    project_id: str | None
    prompt: str
    status: TaskStatus
    subtasks: tuple[Subtask, ...]
    summary: str | None
    token_usage: TokenUsage
    created_at: datetime
    completed_at: datetime | None = None

    # ------------------------------------------------------------------
    # Constructor principal
    # ------------------------------------------------------------------

    @classmethod
    def create(cls, prompt: str, *, project_id: str | None = None) -> Task:
        """
        Crea una nueva Task en estado PENDING.

        Parámetros:
            prompt:     Instrucción del usuario al agente.
            project_id: Proyecto al que pertenece. None para tasks sueltas.
        """
        return cls(
            id=generate_id(),
            project_id=project_id,
            prompt=prompt,
            status=TaskStatus.PENDING,
            subtasks=(),
            summary=None,
            token_usage=TokenUsage.zero(),
            created_at=datetime.now(timezone.utc),
        )

    # ------------------------------------------------------------------
    # Métodos de transición de estado
    # ------------------------------------------------------------------

    def _transition_to(self, new_status: TaskStatus) -> Task:
        """Valida y ejecuta una transición de estado."""
        from hiperforge.domain.exceptions import InvalidStatusTransition

        allowed = _TASK_VALID_TRANSITIONS[self.status]
        if new_status not in allowed:
            raise InvalidStatusTransition(
                entity=f"Task({self.id})",
                from_status=self.status.value,
                to_status=new_status.value,
            )

        completed_at = (
            datetime.now(timezone.utc)
            if not _TASK_VALID_TRANSITIONS[new_status]
            else None
        )

        return Task(
            id=self.id,
            project_id=self.project_id,
            prompt=self.prompt,
            status=new_status,
            subtasks=self.subtasks,
            summary=self.summary,
            token_usage=self.token_usage,
            created_at=self.created_at,
            completed_at=completed_at,
        )

    def start_planning(self) -> Task:
        """El agente empieza a descomponer el prompt en subtasks."""
        return self._transition_to(TaskStatus.PLANNING)

    def start_execution(self, subtasks: list[Subtask]) -> Task:
        """
        El agente terminó de planear y empieza a ejecutar.

        Recibe la lista de subtasks generadas durante PLANNING
        y la convierte en el plan inmutable de ejecución.
        """
        from hiperforge.domain.exceptions import InvalidStatusTransition

        if self.status != TaskStatus.PLANNING:
            raise InvalidStatusTransition(
                entity=f"Task({self.id})",
                from_status=self.status.value,
                to_status=TaskStatus.IN_PROGRESS.value,
            )

        return Task(
            id=self.id,
            project_id=self.project_id,
            prompt=self.prompt,
            status=TaskStatus.IN_PROGRESS,
            subtasks=tuple(subtasks),
            summary=self.summary,
            token_usage=self.token_usage,
            created_at=self.created_at,
            completed_at=None,
        )

    def complete(self, summary: str) -> Task:
        """
        Marca la task como completada con un resumen del resultado.

        El summary lo genera el agente al finalizar todas las subtasks.
        """
        if self.status != TaskStatus.IN_PROGRESS:
            from hiperforge.domain.exceptions import InvalidStatusTransition
            raise InvalidStatusTransition(
                entity=f"Task({self.id})",
                from_status=self.status.value,
                to_status=TaskStatus.COMPLETED.value,
            )

        return Task(
            id=self.id,
            project_id=self.project_id,
            prompt=self.prompt,
            status=TaskStatus.COMPLETED,
            subtasks=self.subtasks,
            summary=summary,
            token_usage=self.token_usage,
            created_at=self.created_at,
            completed_at=datetime.now(timezone.utc),
        )

    def fail(self) -> Task:
        """Una subtask crítica falló sin posibilidad de recuperación."""
        return self._transition_to(TaskStatus.FAILED)

    def cancel(self) -> Task:
        """Cancelada por el usuario o por el sistema."""
        return self._transition_to(TaskStatus.CANCELLED)

    # ------------------------------------------------------------------
    # Métodos de actualización
    # ------------------------------------------------------------------

    def update_subtask(self, updated_subtask: Subtask) -> Task:
        """
        Reemplaza una subtask por su versión actualizada.

        Se llama en cada iteración del loop ReAct para reflejar
        el progreso de la subtask activa en el estado de la Task.

        Busca la subtask por ID y la reemplaza. Si no existe, lanza KeyError.
        """
        updated_subtasks = tuple(
            updated_subtask if st.id == updated_subtask.id else st
            for st in self.subtasks
        )

        # Verificamos que realmente se encontró y reemplazó la subtask
        if updated_subtasks == self.subtasks and updated_subtask not in self.subtasks:
            raise KeyError(f"Subtask con id '{updated_subtask.id}' no encontrada en Task({self.id})")

        return Task(
            id=self.id,
            project_id=self.project_id,
            prompt=self.prompt,
            status=self.status,
            subtasks=updated_subtasks,
            summary=self.summary,
            token_usage=self.token_usage,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )

    def add_token_usage(self, usage: TokenUsage) -> Task:
        """
        Acumula el uso de tokens de una llamada al LLM.

        Se llama después de cada interacción con el LLM durante el loop ReAct
        para mantener el total actualizado.
        """
        return Task(
            id=self.id,
            project_id=self.project_id,
            prompt=self.prompt,
            status=self.status,
            subtasks=self.subtasks,
            summary=self.summary,
            token_usage=self.token_usage + usage,
            created_at=self.created_at,
            completed_at=self.completed_at,
        )

    # ------------------------------------------------------------------
    # Propiedades de conveniencia
    # ------------------------------------------------------------------

    @property
    def is_terminal(self) -> bool:
        """True si la task ya no puede cambiar de estado."""
        return not _TASK_VALID_TRANSITIONS[self.status]

    @property
    def pending_subtasks(self) -> tuple[Subtask, ...]:
        """Subtasks que todavía no han empezado a ejecutarse."""
        return tuple(st for st in self.subtasks if st.status == SubtaskStatus.PENDING)

    @property
    def active_subtask(self) -> Subtask | None:
        """
        La subtask que el agente está ejecutando en este momento.
        None si ninguna está activa (esperando, completada o fallida).
        """
        for st in self.subtasks:
            if st.status == SubtaskStatus.IN_PROGRESS:
                return st
        return None

    @property
    def completed_subtasks(self) -> tuple[Subtask, ...]:
        """Subtasks que terminaron exitosamente."""
        return tuple(st for st in self.subtasks if st.status == SubtaskStatus.COMPLETED)

    @property
    def progress_percentage(self) -> float:
        """
        Porcentaje de avance basado en subtasks completadas.
        0.0 si no hay subtasks. 100.0 si todas están completadas.
        """
        if not self.subtasks:
            return 0.0
        completed = len(self.completed_subtasks)
        return round((completed / len(self.subtasks)) * 100, 1)

    @property
    def duration_seconds(self) -> float | None:
        """Duración total de la task. None si todavía está activa."""
        if self.completed_at is None:
            return None
        delta = self.completed_at - self.created_at
        return round(delta.total_seconds(), 3)

    # ------------------------------------------------------------------
    # Serialización
    # ------------------------------------------------------------------

    def to_dict(self) -> dict[str, Any]:
        """Serializa para guardar en JSON."""
        return {
            "id": self.id,
            "project_id": self.project_id,
            "prompt": self.prompt,
            "status": self.status.value,
            "subtasks": [st.to_dict() for st in self.subtasks],
            "summary": self.summary,
            "token_usage": self.token_usage.to_dict(),
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        """Reconstruye desde un diccionario leído del JSON."""
        return cls(
            id=data["id"],
            project_id=data.get("project_id"),
            prompt=data["prompt"],
            status=TaskStatus(data["status"]),
            subtasks=tuple(Subtask.from_dict(st) for st in data.get("subtasks", [])),
            summary=data.get("summary"),
            token_usage=TokenUsage.from_dict(data["token_usage"]),
            created_at=datetime.fromisoformat(data["created_at"]),
            completed_at=(
                datetime.fromisoformat(data["completed_at"])
                if data.get("completed_at")
                else None
            ),
        )

    # ------------------------------------------------------------------
    # Representación legible para debugging
    # ------------------------------------------------------------------

    def __str__(self) -> str:
        """
        Ejemplo: Task(IN_PROGRESS, 2/4 subtasks, 42.3%) — "Agrega autenticación JWT..."
        """
        prompt_preview = self.prompt[:60] + "..." if len(self.prompt) > 60 else self.prompt
        completed = len(self.completed_subtasks)
        total = len(self.subtasks)
        return (
            f"Task({self.status.value}, {completed}/{total} subtasks,"
            f" {self.progress_percentage}%) — {prompt_preview!r}"
        )
