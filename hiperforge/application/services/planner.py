"""
PlannerService — El cerebro que convierte instrucciones en planes ejecutables.

El planner es la primera fase del agente ReAct y una de las más críticas.
Un plan mal generado lleva a un agente que falla en la ejecución o que
toma un camino ineficiente. Un plan bien generado hace que el executor
complete la task con el mínimo de iteraciones posible.

RESPONSABILIDADES:
  1. Recibir el prompt del usuario y generar un plan estructurado de subtasks.
  2. Validar que el plan sea ejecutable (no vago, no redundante, bien ordenado).
  3. Manejar fallos del LLM con reintentos inteligentes.
  4. Adaptar la estrategia de planificación a la complejidad de la task.

FLUJO COMPLETO:
  prompt del usuario
       │
       ▼
  _classify_complexity()          → detecta si es simple, media o compleja
       │
       ▼
  _build_planning_messages()      → construye el prompt de planificación
       │                            (adaptado a la complejidad)
       ▼
  LLM.complete()                  → llama al LLM con temperature=0.1
       │                            (queremos determinismo, no creatividad)
       ▼
  _parse_plan_response()          → extrae el JSON del contenido
       │                            (maneja markdown, texto extra, etc.)
       ▼
  _validate_plan()                → verifica calidad de las subtasks
       │                            (no vagas, no redundantes, ejecutables)
       ▼
  [si falla] _retry_with_feedback() → reintenta con el error como contexto
       │
       ▼
  _build_subtasks()               → construye las entidades Subtask
       │
       ▼
  lista de Subtask[]

ESTRATEGIA DE REINTENTOS:
  El planner tiene su propio sistema de reintentos independiente del retry
  de red del base adapter. La diferencia es crucial:

    Retry de red (base adapter): para errores transitorios (rate limit, timeout).
    Retry del planner: para planes semánticamente inválidos que necesitan
                       un prompt diferente, no solo esperar y reintentar.

  Si el LLM devuelve un plan con subtasks vagas, el retry del planner
  incluye el feedback exacto de qué estuvo mal para que el LLM mejore.

CLASIFICACIÓN DE COMPLEJIDAD:
  SIMPLE  → 1-2 subtasks, prompt corto, tarea clara y acotada.
             Ejemplos: "crea un archivo config.py", "agrega un test"
  MEDIA   → 3-5 subtasks, requiere varios pasos coordinados.
             Ejemplos: "implementa un endpoint REST", "refactoriza esta clase"
  COMPLEJA → 6+ subtasks, múltiples componentes, riesgo de impacto amplio.
             Ejemplos: "migra la base de datos", "implementa autenticación JWT"

  La complejidad afecta:
    - Cuántas subtasks se permiten en el plan
    - Qué tan detallado es el prompt de planificación
    - Cuántos reintentos se permiten antes de fallar
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any

from hiperforge.core.constants import (
    LLM_DEFAULT_MAX_TOKENS_PLANNING,
    REACT_MAX_SUBTASKS,
)
from hiperforge.core.events import AgentEvent, get_event_bus
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.task import Subtask, Task
from hiperforge.domain.exceptions import EmptyPlanError, InvalidPlanError, PlanError
from hiperforge.infrastructure.llm.base import BaseLLMAdapter
from hiperforge.domain.value_objects.message import Message
from hiperforge.memory.store import MemoryStore

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración interna del planner
# ---------------------------------------------------------------------------

# Temperatura muy baja para planificación — queremos determinismo
_PLANNING_TEMPERATURE = 0.1

# Máximo de reintentos del planner ante planes inválidos
_MAX_PLANNING_RETRIES = 3

# Mínimo de caracteres para que una descripción de subtask sea válida
_MIN_SUBTASK_DESCRIPTION_LENGTH = 10

# Máximo de caracteres para una descripción de subtask
# Descripciones muy largas confunden al executor
_MAX_SUBTASK_DESCRIPTION_LENGTH = 240

# Palabras que indican subtasks vagas que el agente no puede ejecutar
_VAGUE_INDICATORS: frozenset[str] = frozenset({
    "asegurarse", "verificar que todo", "confirmar que funciona",
    "revisar todo", "hacer lo necesario", "completar", "finalizar",
    "ensure", "make sure", "verify everything", "check everything",
})

# ---------------------------------------------------------------------------
# Clasificación de complejidad
# ---------------------------------------------------------------------------

class TaskComplexity(str, Enum):
    """
    Nivel de complejidad estimado de la task.

    Afecta directamente el prompt de planificación y los límites del plan.
    """
    SIMPLE   = "simple"    # 1-2 subtasks esperadas
    MEDIUM   = "medium"    # 3-5 subtasks esperadas
    COMPLEX  = "complex"   # 6+ subtasks esperadas


# Límites de subtasks por nivel de complejidad
_MAX_SUBTASKS_BY_COMPLEXITY: dict[TaskComplexity, int] = {
    TaskComplexity.SIMPLE:  2,
    TaskComplexity.MEDIUM:  5,
    TaskComplexity.COMPLEX: REACT_MAX_SUBTASKS,
}

# ---------------------------------------------------------------------------
# Tipos internos del planner
# ---------------------------------------------------------------------------

@dataclass
class PlanningResult:
    """
    Resultado interno del proceso de planificación.

    Contiene las subtasks generadas y metadatos del proceso
    para logging y debugging.
    """
    subtasks: list[Subtask]
    complexity: TaskComplexity
    attempts: int                   # cuántos intentos tomó generar el plan
    planning_duration_seconds: float
    plan_summary: str               # el summary que generó el LLM


@dataclass
class RawPlan:
    """
    Plan extraído del JSON de la respuesta del LLM, antes de validar.

    Separa el parsing del JSON de la validación semántica para
    poder dar mensajes de error más precisos en cada etapa.
    """
    subtask_descriptions: list[str]   # descripciones en orden
    summary: str                       # resumen del plan
    raw_json: dict[str, Any]           # JSON completo para debugging


# ---------------------------------------------------------------------------
# Prompts de planificación
# ---------------------------------------------------------------------------

_PLANNING_SYSTEM_PROMPT = """\
Eres un planificador de tareas de desarrollo. Descompón la instrucción en los MÍNIMOS pasos necesarios.

FORMATO — responde SOLO con este JSON:
{{
  "subtasks": [
    {{"description": "<paso concreto y verificable>", "order": 0}}
  ],
  "summary": "<resumen en una línea>"
}}

REGLA PRINCIPAL: USA EL MENOR NÚMERO DE SUBTASKS POSIBLE.
- Si la tarea es crear/modificar un solo archivo → USA 1 SUBTASK que incluya creación Y verificación.
- Si la tarea requiere varios archivos independientes → 1 subtask por archivo.
- Máximo {max_subtasks} subtasks. Prefiere MENOS.
- Cada subtask debe indicar QUÉ hacer y CÓMO verificarlo.
- No agregues pasos de documentación, README, limpieza o tests salvo que se pidan.
- No incluyas pasos vagos como "asegurarse de que funciona".

EJEMPLO:
  Tarea: "crea un script Python que lea un CSV y genere un reporte en HTML"
  Plan correcto (1 subtask):
  {{"subtasks": [{{"description": "Crear script report.py que lea CSV con csv.DictReader, genere HTML con tabla escapada (html.escape), acepte args --csv y --html. Verificar con py_compile.", "order": 0}}], "summary": "Script CSV→HTML"}}
"""

_PLANNING_USER_TEMPLATE = """\
Tarea: {prompt}
Complejidad: {complexity}. Máximo {max_subtasks} subtasks (prefiere menos).
"""

_RETRY_FEEDBACK_TEMPLATE = """\
Tu plan anterior tenía los siguientes problemas:
{feedback}

Por favor genera un nuevo plan corrigiendo estos problemas.
La tarea sigue siendo:

{prompt}
"""


# ---------------------------------------------------------------------------
# PlannerService
# ---------------------------------------------------------------------------

class PlannerService:
    """
    Genera y valida el plan de subtasks para una Task.

    Parámetros:
        llm:   Adapter del LLM para generar el plan.
        store: MemoryStore para leer preferencias de planificación
               (temperatura, max_subtasks, etc.).
    """

    def __init__(self, llm: BaseLLMAdapter, store: MemoryStore) -> None:
        self._llm = llm
        self._store = store

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def generate_plan(self, task: Task) -> list[Subtask]:
        """
        Genera el plan de subtasks para una Task.

        Punto de entrada principal. Orquesta todo el proceso:
        clasificación → planificación → validación → reintentos si falla.

        Parámetros:
            task: La Task para la que se genera el plan.
                  Debe estar en estado PLANNING.

        Returns:
            Lista de Subtasks en orden de ejecución, listas para el executor.

        Raises:
            EmptyPlanError:   Si tras todos los reintentos el plan sigue vacío.
            InvalidPlanError: Si el plan tiene estructura o calidad inválida
                              y no puede corregirse.
            LLMConnectionError: Si el LLM no responde.
        """
        start_time = time.monotonic()

        logger.info(
            "iniciando planificación",
            task_id=task.id,
            prompt_preview=task.prompt[:100],
        )

        # Bindeamos contexto al LLM para que sus eventos lleven el task_id
        self._llm.bind_context(task_id=task.id)

        # Emitimos evento de inicio de planificación
        get_event_bus().emit(AgentEvent.task_planning(task_id=task.id))

        # Paso 1: clasificar complejidad para adaptar el proceso
        complexity = self._classify_complexity(task.prompt)

        logger.debug(
            "complejidad de la task clasificada",
            task_id=task.id,
            complexity=complexity.value,
        )

        # Paso 2: generar el plan con reintentos ante planes inválidos
        planning_result = self._generate_with_retries(
            task=task,
            complexity=complexity,
            start_time=start_time,
        )

        logger.info(
            "plan generado exitosamente",
            task_id=task.id,
            subtask_count=len(planning_result.subtasks),
            complexity=complexity.value,
            attempts=planning_result.attempts,
            duration_seconds=round(planning_result.planning_duration_seconds, 2),
        )

        return planning_result.subtasks

    # ------------------------------------------------------------------
    # Generación con reintentos
    # ------------------------------------------------------------------

    def _generate_with_retries(
        self,
        task: Task,
        complexity: TaskComplexity,
        start_time: float,
    ) -> PlanningResult:
        """
        Intenta generar un plan válido hasta _MAX_PLANNING_RETRIES veces.

        Cada reintento incluye el feedback del intento anterior para que
        el LLM pueda corregir exactamente lo que estuvo mal.

        La diferencia con el retry de red del base adapter:
          - El retry de red repite exactamente el mismo prompt.
          - Este retry MODIFICA el prompt con feedback del error.
        """
        last_error_feedback: str | None = None
        max_subtasks = _MAX_SUBTASKS_BY_COMPLEXITY[complexity]

        for attempt in range(1, _MAX_PLANNING_RETRIES + 1):
            logger.debug(
                "intento de planificación",
                task_id=task.id,
                attempt=attempt,
                max_attempts=_MAX_PLANNING_RETRIES,
            )

            try:
                # Construir mensajes según si es primer intento o reintento
                if attempt == 1:
                    messages = self._build_planning_messages(
                        prompt=task.prompt,
                        complexity=complexity,
                        max_subtasks=max_subtasks,
                    )
                else:
                    # Reintento con feedback del error anterior
                    messages = self._build_retry_messages(
                        prompt=task.prompt,
                        complexity=complexity,
                        max_subtasks=max_subtasks,
                        feedback=last_error_feedback or "El plan anterior era inválido.",
                    )

                # Llamar al LLM
                response = self._llm.complete(
                    messages=messages,
                    max_tokens=LLM_DEFAULT_MAX_TOKENS_PLANNING,
                    temperature=_PLANNING_TEMPERATURE,
                    # Stop sequence para evitar texto después del JSON
                    stop_sequences=["\n\nTarea:", "\n\nPor favor"],
                )

                # Parsear el JSON de la respuesta
                raw_plan = self._parse_plan_response(
                    content=response.content,
                    task_id=task.id,
                )

                # Validar la calidad del plan
                validation_errors = self._validate_plan(raw_plan, max_subtasks)

                if validation_errors:
                    # El plan tiene problemas — preparamos feedback para el reintento
                    last_error_feedback = self._format_validation_feedback(validation_errors)

                    logger.warning(
                        "plan inválido — reintentando",
                        task_id=task.id,
                        attempt=attempt,
                        errors=validation_errors,
                    )

                    # Si este fue el último intento, lanzamos el error
                    if attempt == _MAX_PLANNING_RETRIES:
                        raise InvalidPlanError(
                            reason=(
                                f"Plan inválido tras {_MAX_PLANNING_RETRIES} intentos. "
                                f"Último error: {'; '.join(validation_errors)}"
                            ),
                            raw_response=response.content,
                        )

                    continue  # reintentamos

                # Plan válido — construimos las entidades Subtask
                subtasks = self._build_subtasks(
                    raw_plan=raw_plan,
                    task_id=task.id,
                )

                duration = time.monotonic() - start_time

                return PlanningResult(
                    subtasks=subtasks,
                    complexity=complexity,
                    attempts=attempt,
                    planning_duration_seconds=duration,
                    plan_summary=raw_plan.summary,
                )

            except (EmptyPlanError, InvalidPlanError):
                # Errores semánticos — propagamos si es el último intento
                if attempt == _MAX_PLANNING_RETRIES:
                    raise
                # Si no es el último, preparamos feedback genérico y continuamos
                last_error_feedback = "El plan anterior no tenía el formato correcto."
                continue

        # Este punto es inalcanzable por la lógica del loop,
        # pero satisface al type checker
        raise PlanError(
            f"No se pudo generar un plan válido tras {_MAX_PLANNING_RETRIES} intentos."
        )

    # ------------------------------------------------------------------
    # Construcción de mensajes para el LLM
    # ------------------------------------------------------------------

    def _build_planning_messages(
        self,
        prompt: str,
        complexity: TaskComplexity,
        max_subtasks: int,
    ) -> list[Message]:
        """
        Construye los mensajes para la primera llamada de planificación.

        Usa un sistema prompt dedicado a planificación, diferente
        al prompt de sistema del executor. El planner no necesita
        saber sobre las tools — solo necesita generar un buen plan.
        """
        system_msg = Message.system(
            _PLANNING_SYSTEM_PROMPT.format(max_subtasks=max_subtasks)
        )

        user_msg = Message.user(
            _PLANNING_USER_TEMPLATE.format(
                prompt=prompt,
                complexity=self._complexity_label(complexity),
                max_subtasks=max_subtasks,
            )
        )

        return [system_msg, user_msg]

    def _build_retry_messages(
        self,
        prompt: str,
        complexity: TaskComplexity,
        max_subtasks: int,
        feedback: str,
    ) -> list[Message]:
        """
        Construye los mensajes para un reintento de planificación.

        Incluye el feedback del intento anterior para que el LLM
        corrija exactamente los problemas identificados.

        La estrategia de incluir el error en el prompt es más efectiva
        que simplemente reintentar con el mismo prompt, porque le da
        al LLM información concreta de qué mejorar.
        """
        system_msg = Message.system(
            _PLANNING_SYSTEM_PROMPT.format(max_subtasks=max_subtasks)
        )

        retry_user_msg = Message.user(
            _RETRY_FEEDBACK_TEMPLATE.format(
                feedback=feedback,
                prompt=prompt,
            )
        )

        return [system_msg, retry_user_msg]

    # ------------------------------------------------------------------
    # Parsing de la respuesta del LLM
    # ------------------------------------------------------------------

    def _parse_plan_response(self, content: str, task_id: str) -> RawPlan:
        """
        Extrae el JSON del plan de la respuesta del LLM.

        Los LLMs a veces devuelven el JSON de formas inesperadas.
        Este método maneja todos los casos conocidos:

          Caso 1: JSON limpio (lo esperado)
            {"subtasks": [...], "summary": "..."}

          Caso 2: JSON envuelto en markdown
            ```json
            {"subtasks": [...], "summary": "..."}
            ```

          Caso 3: Texto explicativo antes del JSON
            "Aquí está el plan para tu tarea:\n\n{...}"

          Caso 4: JSON con comentarios inválidos (algunos modelos)
            {"subtasks": [...] // estos son los pasos}

          Caso 5: Múltiples objetos JSON (solo tomamos el primero válido)

        Raises:
            InvalidPlanError: Si no se puede extraer ningún JSON válido.
        """
        if not content or not content.strip():
            raise EmptyPlanError(task_prompt="")

        # Estrategia 1: parsear el content directamente (caso más común)
        parsed = self._try_parse_json(content.strip())
        if parsed is not None:
            return self._extract_raw_plan(parsed, content)

        # Estrategia 2: extraer de bloque markdown ```json ... ```
        markdown_match = re.search(
            r"```(?:json)?\s*\n?(.*?)\n?```",
            content,
            re.DOTALL | re.IGNORECASE,
        )
        if markdown_match:
            parsed = self._try_parse_json(markdown_match.group(1).strip())
            if parsed is not None:
                return self._extract_raw_plan(parsed, content)

        # Estrategia 3: encontrar el primer objeto JSON en el texto
        # Busca { ... } más externo que sea válido
        json_match = re.search(r"\{.*\}", content, re.DOTALL)
        if json_match:
            parsed = self._try_parse_json(json_match.group(0))
            if parsed is not None:
                return self._extract_raw_plan(parsed, content)

        # Estrategia 4: eliminar comentarios tipo // y reintentar
        cleaned = re.sub(r"//[^\n]*", "", content)
        parsed = self._try_parse_json(cleaned.strip())
        if parsed is not None:
            return self._extract_raw_plan(parsed, content)

        # Ninguna estrategia funcionó
        logger.warning(
            "no se pudo extraer JSON del plan",
            task_id=task_id,
            content_preview=content[:300],
        )

        raise InvalidPlanError(
            reason=(
                "No se pudo extraer JSON válido de la respuesta del LLM. "
                "El LLM no siguió el formato de respuesta requerido."
            ),
            raw_response=content,
        )

    def _extract_raw_plan(
        self,
        parsed: dict[str, Any],
        raw_content: str,
    ) -> RawPlan:
        """
        Extrae los campos del JSON parseado y construye un RawPlan.

        Valida que los campos mínimos existan antes de construir.

        Raises:
            InvalidPlanError: Si el JSON no tiene la estructura esperada.
        """
        if not isinstance(parsed, dict):
            raise InvalidPlanError(
                reason="La respuesta del LLM no es un objeto JSON",
                raw_response=raw_content,
            )

        raw_subtasks = parsed.get("subtasks")
        summary = parsed.get("summary", "")

        if raw_subtasks is None:
            raise InvalidPlanError(
                reason=(
                    "El JSON del plan no tiene el campo 'subtasks'. "
                    "Campos presentes: " + ", ".join(str(k) for k in parsed.keys())
                ),
                raw_response=raw_content,
            )

        if not isinstance(raw_subtasks, list):
            raise InvalidPlanError(
                reason=f"El campo 'subtasks' debe ser una lista, recibido: {type(raw_subtasks).__name__}",
                raw_response=raw_content,
            )

        if len(raw_subtasks) == 0:
            raise EmptyPlanError(task_prompt="")

        # Extraer las descripciones en orden
        descriptions: list[str] = []
        for i, raw in enumerate(raw_subtasks):
            if not isinstance(raw, dict):
                raise InvalidPlanError(
                    reason=f"La subtask en posición {i} no es un objeto JSON",
                    raw_response=raw_content,
                )

            description = raw.get("description", "").strip()
            if not description:
                raise InvalidPlanError(
                    reason=f"La subtask en posición {i} no tiene campo 'description'",
                    raw_response=raw_content,
                )

            descriptions.append(description)

        return RawPlan(
            subtask_descriptions=descriptions,
            summary=str(summary).strip(),
            raw_json=parsed,
        )

    # ------------------------------------------------------------------
    # Validación semántica del plan
    # ------------------------------------------------------------------

    def _validate_plan(
        self,
        raw_plan: RawPlan,
        max_subtasks: int,
    ) -> list[str]:
        """
        Valida la calidad semántica del plan.

        Va más allá de la validación estructural — verifica que las
        subtasks sean ejecutables y tengan sentido como plan coherente.

        Returns:
            Lista de errores encontrados. Lista vacía = plan válido.
        """
        errors: list[str] = []
        descriptions = raw_plan.subtask_descriptions

        # Verificación de cantidad
        if len(descriptions) > max_subtasks:
            errors.append(
                f"El plan tiene {len(descriptions)} subtasks pero el máximo es {max_subtasks}. "
                f"Agrupa los pasos más pequeños."
            )

        for i, desc in enumerate(descriptions, 1):
            # Longitud mínima
            if len(desc) < _MIN_SUBTASK_DESCRIPTION_LENGTH:
                errors.append(
                    f"Subtask {i}: descripción demasiado corta ({len(desc)} chars). "
                    f"Mínimo {_MIN_SUBTASK_DESCRIPTION_LENGTH} caracteres."
                )
                continue

            # Longitud máxima
            if len(desc) > _MAX_SUBTASK_DESCRIPTION_LENGTH:
                errors.append(
                    f"Subtask {i}: descripción demasiado larga ({len(desc)} chars). "
                    f"Máximo {_MAX_SUBTASK_DESCRIPTION_LENGTH} caracteres."
                )

            # Detectar descripciones vagas
            desc_lower = desc.lower()
            for vague_word in _VAGUE_INDICATORS:
                if vague_word in desc_lower:
                    errors.append(
                        f"Subtask {i}: descripción vaga — contiene '{vague_word}'. "
                        f"Describe exactamente qué archivo crear, qué comando ejecutar, "
                        f"qué función implementar."
                    )
                    break

        # Detectar subtasks duplicadas o muy similares
        duplicates = self._find_duplicate_descriptions(descriptions)
        for dup_pair in duplicates:
            errors.append(
                f"Subtasks {dup_pair[0]} y {dup_pair[1]} parecen duplicadas. "
                f"Combínalas o diferéncialas claramente."
            )

        return errors

    def _find_duplicate_descriptions(
        self,
        descriptions: list[str],
    ) -> list[tuple[int, int]]:
        """
        Detecta descripciones de subtasks que son muy similares entre sí.

        Usa comparación de palabras clave en vez de distancia de edición
        para ser eficiente sin dependencias externas.

        Returns:
            Lista de pares (índice_1, índice_2) de subtasks similares.
            Índices son 1-based para mensajes de error legibles.
        """
        duplicates: list[tuple[int, int]] = []

        # Tokenizamos cada descripción en palabras significativas
        def key_words(text: str) -> frozenset[str]:
            # Palabras de más de 3 letras, en minúsculas, sin stopwords básicas
            stopwords = {"para", "con", "los", "las", "del", "que", "una", "un",
                         "the", "for", "and", "with", "from", "into", "this"}
            return frozenset(
                w for w in re.findall(r"\b\w{4,}\b", text.lower())
                if w not in stopwords
            )

        tokenized = [key_words(d) for d in descriptions]

        for i in range(len(tokenized)):
            for j in range(i + 1, len(tokenized)):
                set_a = tokenized[i]
                set_b = tokenized[j]

                if not set_a or not set_b:
                    continue

                # Similitud de Jaccard: |A ∩ B| / |A ∪ B|
                intersection = len(set_a & set_b)
                union = len(set_a | set_b)

                if union > 0 and (intersection / union) > 0.7:
                    duplicates.append((i + 1, j + 1))

        return duplicates

    # ------------------------------------------------------------------
    # Construcción de entidades Subtask
    # ------------------------------------------------------------------

    def _build_subtasks(self, raw_plan: RawPlan, task_id: str) -> list[Subtask]:
        """
        Construye las entidades Subtask a partir del plan validado.

        Preserva el orden del plan respetando el campo 'order' si existe,
        o usando el orden de aparición si no existe.
        """
        # El plan puede tener el campo order o no — normalizamos
        raw_subtasks = raw_plan.raw_json.get("subtasks", [])

        # Intentamos usar el orden declarado en el plan
        ordered_items: list[tuple[int, str]] = []
        for i, (raw, desc) in enumerate(
            zip(raw_subtasks, raw_plan.subtask_descriptions)
        ):
            order = raw.get("order", i) if isinstance(raw, dict) else i
            # Nos aseguramos de que order sea un entero válido
            try:
                order = int(order)
            except (TypeError, ValueError):
                order = i
            ordered_items.append((order, desc))

        # Ordenamos por el campo order del plan
        ordered_items.sort(key=lambda x: x[0])

        subtasks: list[Subtask] = []
        for final_order, description in enumerate(
            desc for _, desc in ordered_items
        ):
            subtask = Subtask.create(
                task_id=task_id,
                description=description,
                order=final_order,
            )
            subtasks.append(subtask)

        return subtasks

    # ------------------------------------------------------------------
    # Clasificación de complejidad
    # ------------------------------------------------------------------

    def _classify_complexity(self, prompt: str) -> TaskComplexity:
        """
        Clasifica la complejidad estimada de la task basándose en el prompt.

        Usa heurísticas basadas en:
          - Longitud del prompt (prompts más largos → más complejos)
          - Palabras clave que indican alcance amplio
          - Número de verbos de acción (cada verbo puede ser una subtask)
          - Detección de tareas de archivo único (crea un script, un archivo, etc.)

        Esta clasificación es una ESTIMACIÓN — el planner puede generar
        más o menos subtasks de lo esperado. La clasificación afecta los
        límites y el nivel de detalle del prompt, no el resultado final.
        """
        prompt_lower = prompt.lower()
        word_count = len(prompt.split())

        # Indicadores de alta complejidad
        complex_keywords: frozenset[str] = frozenset({
            "migra", "refactoriza", "implementa", "diseña", "arquitectura",
            "sistema", "módulo completo", "todo el", "toda la",
            "migrate", "refactor", "implement", "design", "architecture",
            "entire", "complete", "full", "system",
        })

        # Indicadores de baja complejidad — tareas de un solo artefacto
        simple_keywords: frozenset[str] = frozenset({
            "agrega", "crea", "añade", "cambia", "arregla", "corrige",
            "add", "create", "fix", "change", "update", "rename",
        })

        # Contar indicadores presentes (usando límites de palabra para evitar
        # falsos positivos como "implement" dentro de "implementa")
        _boundary = r"(?<![a-záéíóúñü]){kw}(?![a-záéíóúñü])"

        def _matches(kw: str) -> bool:
            return bool(re.search(_boundary.format(kw=re.escape(kw)), prompt_lower))

        complex_count = sum(1 for kw in complex_keywords if _matches(kw))
        simple_count = sum(1 for kw in simple_keywords if _matches(kw))

        # Detección de tarea de archivo único: "crea un script/archivo/clase/módulo"
        # Estas tareas SIEMPRE deben ser SIMPLE (max 2 subtasks, idealmente 1)
        single_file_patterns: tuple[str, ...] = (
            "crea un script", "crea un archivo", "crea una clase",
            "crea un módulo", "crea un programa", "crea un fichero",
            "escribe un script", "escribe un archivo", "escribe un programa",
            "create a script", "create a file", "create a class",
            "create a module", "create a program", "write a script",
            "write a file", "write a program",
        )
        is_single_file = any(pattern in prompt_lower for pattern in single_file_patterns)

        if is_single_file and word_count <= 30:
            return TaskComplexity.SIMPLE

        # Clasificar por combinación de factores
        if complex_count >= 2 or word_count > 50:
            return TaskComplexity.COMPLEX

        if complex_count >= 1 or word_count > 20 or simple_count >= 2:
            return TaskComplexity.MEDIUM

        return TaskComplexity.SIMPLE

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _try_parse_json(text: str) -> dict[str, Any] | None:
        """
        Intenta parsear un string como JSON.

        Devuelve el dict parseado si tiene éxito, None si falla.
        No lanza excepciones — es una operación de "intento seguro".
        """
        try:
            result = json.loads(text)
            return result if isinstance(result, dict) else None
        except (json.JSONDecodeError, ValueError):
            return None

    @staticmethod
    def _format_validation_feedback(errors: list[str]) -> str:
        """
        Formatea los errores de validación como feedback legible para el LLM.

        El feedback se incluye en el prompt de reintento para que el LLM
        sepa exactamente qué corregir.
        """
        lines = ["Los problemas con el plan anterior son:"]
        for i, error in enumerate(errors, 1):
            lines.append(f"  {i}. {error}")
        return "\n".join(lines)

    @staticmethod
    def _complexity_label(complexity: TaskComplexity) -> str:
        """Devuelve una etiqueta legible de la complejidad para el prompt."""
        labels = {
            TaskComplexity.SIMPLE:  "simple (1-2 pasos)",
            TaskComplexity.MEDIUM:  "media (3-5 pasos)",
            TaskComplexity.COMPLEX: "compleja (puede requerir muchos pasos)",
        }
        return labels[complexity]
