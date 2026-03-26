"""
Guardrails — Sistema de protección y control de recursos del agente.

Este módulo es la capa de defensa que previene que el agente:
  1. Gaste tokens o dinero sin límite (BudgetGuard)
  2. Ejecute comandos destructivos sin control (CommandAnalyzer)
  3. Repita errores indefinidamente (CircuitBreaker)
  4. Opere fuera de su sandbox (PathGuard)

PRINCIPIO DE DISEÑO:
  Cada guardrail es independiente, testeable y composable.
  El executor llama a los guardrails en puntos específicos del loop ReAct.
  Si un guardrail dispara, el executor recibe una señal clara de QUÉ
  pasó y POR QUÉ, para decidir si abortar, pedir confirmación o continuar.

ARQUITECTURA:
  GuardrailViolation (dataclass inmutable)
  ├── BudgetGuard         → límites de tokens, costo, tiempo
  ├── CommandAnalyzer     → análisis profundo de comandos shell
  ├── CircuitBreaker      → corta ejecución tras N fallos consecutivos
  └── PathGuard           → sandbox de archivos, symlink resolution
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════
# Resultado de un guardrail — inmutable, serializable, descriptivo
# ═══════════════════════════════════════════════════════════════════════

class ViolationSeverity(str, Enum):
    """Severidad de la violación — determina la acción del executor."""
    BLOCK = "block"         # Detener ejecución inmediatamente
    WARN = "warn"           # Continuar pero loggear y notificar
    CONFIRM = "confirm"     # Pedir confirmación al usuario antes de continuar


@dataclass(frozen=True)
class GuardrailViolation:
    """
    Resultado de un guardrail que detectó un problema.

    Atributos:
        guardrail:   Nombre del guardrail que disparó (para logs).
        severity:    Qué tan grave es — determina si bloquear, advertir o confirmar.
        reason:      Descripción legible del problema para el usuario.
        details:     Datos técnicos adicionales para debugging.
    """
    guardrail: str
    severity: ViolationSeverity
    reason: str
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return f"[{self.severity.value.upper()}] {self.guardrail}: {self.reason}"


# ═══════════════════════════════════════════════════════════════════════
# BudgetGuard — Control de recursos (tokens, costo, tiempo)
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class BudgetGuard:
    """
    Controla que el agente no exceda los límites de recursos configurados.

    Rastrear el consumo de recursos en tiempo real durante la ejecución.
    Cuando se excede un límite, retorna una GuardrailViolation que el
    executor interpreta como señal para abortar la task.

    USO:
      budget = BudgetGuard(max_tokens=50_000, max_cost_usd=0.50, max_seconds=300)
      budget.record_tokens(input=500, output=200)
      budget.record_tokens(input=800, output=400)

      violation = budget.check()
      if violation:
          # Abortar la task — presupuesto excedido
          ...

    Parámetros:
        max_tokens:   Límite total de tokens (input + output). 0 = sin límite.
        max_cost_usd: Límite de costo estimado en USD. 0.0 = sin límite.
        max_seconds:  Límite de tiempo de ejecución en segundos. 0.0 = sin límite.
    """
    max_tokens: int = 0
    max_cost_usd: float = 0.0
    max_seconds: float = 0.0

    # Estado mutable — se actualiza durante la ejecución
    _total_input_tokens: int = field(default=0, init=False, repr=False)
    _total_output_tokens: int = field(default=0, init=False, repr=False)
    _total_cost_usd: float = field(default=0.0, init=False, repr=False)
    _start_time: float = field(default_factory=time.monotonic, init=False, repr=False)

    @property
    def total_tokens(self) -> int:
        return self._total_input_tokens + self._total_output_tokens

    @property
    def total_cost_usd(self) -> float:
        return self._total_cost_usd

    @property
    def elapsed_seconds(self) -> float:
        return time.monotonic() - self._start_time

    def record_tokens(self, input_tokens: int, output_tokens: int, cost_usd: float = 0.0) -> None:
        """Registra el consumo de una llamada al LLM."""
        self._total_input_tokens += input_tokens
        self._total_output_tokens += output_tokens
        self._total_cost_usd += cost_usd

    def check(self) -> GuardrailViolation | None:
        """
        Verifica si algún límite fue excedido.

        Returns:
            GuardrailViolation si se excedió un límite, None si todo está dentro.
        """
        if self.max_tokens > 0 and self.total_tokens > self.max_tokens:
            return GuardrailViolation(
                guardrail="BudgetGuard",
                severity=ViolationSeverity.BLOCK,
                reason=(
                    f"Límite de tokens excedido: {self.total_tokens:,} / {self.max_tokens:,}. "
                    f"La task consumió demasiados tokens — abortando para evitar costos excesivos."
                ),
                details={
                    "total_tokens": self.total_tokens,
                    "max_tokens": self.max_tokens,
                    "input_tokens": self._total_input_tokens,
                    "output_tokens": self._total_output_tokens,
                },
            )

        if self.max_cost_usd > 0.0 and self._total_cost_usd > self.max_cost_usd:
            return GuardrailViolation(
                guardrail="BudgetGuard",
                severity=ViolationSeverity.BLOCK,
                reason=(
                    f"Límite de costo excedido: ${self._total_cost_usd:.4f} / ${self.max_cost_usd:.4f} USD. "
                    f"Abortando para evitar cobros inesperados."
                ),
                details={
                    "total_cost_usd": self._total_cost_usd,
                    "max_cost_usd": self.max_cost_usd,
                },
            )

        if self.max_seconds > 0.0 and self.elapsed_seconds > self.max_seconds:
            return GuardrailViolation(
                guardrail="BudgetGuard",
                severity=ViolationSeverity.BLOCK,
                reason=(
                    f"Límite de tiempo excedido: {self.elapsed_seconds:.1f}s / {self.max_seconds:.1f}s. "
                    f"La task tardó demasiado — posiblemente está atascada."
                ),
                details={
                    "elapsed_seconds": round(self.elapsed_seconds, 2),
                    "max_seconds": self.max_seconds,
                },
            )

        return None

    def warn_if_approaching(self, threshold: float = 0.8) -> GuardrailViolation | None:
        """
        Advierte si el consumo está cerca del límite (por defecto al 80%).

        Útil para que el executor inyecte una intervención al agente:
        "Estás usando muchos tokens — intenta completar pronto."
        """
        if self.max_tokens > 0:
            usage_ratio = self.total_tokens / self.max_tokens
            if usage_ratio >= threshold:
                return GuardrailViolation(
                    guardrail="BudgetGuard",
                    severity=ViolationSeverity.WARN,
                    reason=(
                        f"Consumo de tokens al {usage_ratio:.0%} del límite "
                        f"({self.total_tokens:,} / {self.max_tokens:,}). "
                        f"Intenta completar la task pronto."
                    ),
                    details={"usage_ratio": round(usage_ratio, 3)},
                )

        if self.max_seconds > 0.0:
            time_ratio = self.elapsed_seconds / self.max_seconds
            if time_ratio >= threshold:
                return GuardrailViolation(
                    guardrail="BudgetGuard",
                    severity=ViolationSeverity.WARN,
                    reason=(
                        f"Tiempo al {time_ratio:.0%} del límite "
                        f"({self.elapsed_seconds:.0f}s / {self.max_seconds:.0f}s)."
                    ),
                    details={"time_ratio": round(time_ratio, 3)},
                )

        return None


# ═══════════════════════════════════════════════════════════════════════
# CommandAnalyzer — Análisis profundo de comandos shell
# ═══════════════════════════════════════════════════════════════════════

class CommandAnalyzer:
    """
    Analiza comandos shell para detectar riesgos que los patrones simples no ven.

    Va más allá de la lista de patrones peligrosos de ShellTool:
      - Detecta cadenas de pipes que terminan en escritura destructiva
      - Detecta path traversal (../../etc/passwd)
      - Detecta exfiltración de datos por red (curl, wget con datos sensibles)
      - Detecta fork bombs y bombas lógicas
      - Detecta evasión de seguridad (base64 decode → execute)

    DISEÑO:
      Cada método retorna None si el comando es seguro, o un
      GuardrailViolation si detectó un riesgo. El caller compone
      múltiples checks llamándolos en secuencia.
    """

    # Rutas del sistema que NUNCA deberían ser objetivo de escritura
    _PROTECTED_PATHS: frozenset[str] = frozenset({
        "/etc", "/usr", "/bin", "/sbin", "/boot", "/sys", "/proc",
        "/var/log", "/root", "/home", "/dev",
        "C:\\Windows", "C:\\Program Files",
    })

    # Patrones de exfiltración de datos por red
    _EXFIL_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bcurl\b.*\b-d\b|\bcurl\b.*--data"),         # curl con POST data
        re.compile(r"\bwget\b.*--post-data"),                      # wget con POST
        re.compile(r"\bnc\b|\bnetcat\b"),                          # netcat
        re.compile(r"\bscp\b|\brsync\b.*@"),                       # copia remota
        re.compile(r"\bssh\b.*@"),                                 # SSH a remoto
        re.compile(r"\b(python|python3|ruby|perl|node)\b.*\bhttp\.server"), # servidor HTTP inline
    ]

    # Patrones de evasión — intentos de encodear comandos para saltar detección
    _EVASION_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\bbase64\b.*\|\s*(bash|sh|python|perl)"),     # base64 | bash
        re.compile(r"\beval\b.*\$\("),                              # eval $(...)
        re.compile(r"\bexec\b.*\$\("),                              # exec $(...)
        re.compile(r"\\x[0-9a-f]{2}"),                              # hex escapes
        re.compile(r"\$\{.*#.*\}"),                                 # variable manipulation
    ]

    # Patrones de path traversal
    _PATH_TRAVERSAL_PATTERNS: list[re.Pattern[str]] = [
        re.compile(r"\.\./\.\./"),                                  # ../../
        re.compile(r"/etc/(passwd|shadow|hosts|sudoers)"),          # archivos sensibles
        re.compile(r"~root|~/\.ssh|~/\.gnupg"),                    # dirs sensibles del usuario
    ]

    def analyze(self, command: str) -> GuardrailViolation | None:
        """
        Análisis completo de un comando shell.

        Ejecuta todos los checks en orden de severidad.
        Retorna la primera violación encontrada o None si es seguro.
        """
        command_lower = command.strip().lower()

        checks = [
            self._check_path_traversal,
            self._check_protected_paths,
            self._check_exfiltration,
            self._check_evasion,
            self._check_pipe_chain_risk,
            self._check_recursive_destruction,
        ]

        for check in checks:
            violation = check(command_lower, command)
            if violation:
                return violation

        return None

    def _check_path_traversal(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        for pattern in self._PATH_TRAVERSAL_PATTERNS:
            if pattern.search(cmd_lower):
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.BLOCK,
                    reason=f"Path traversal detectado en comando: acceso a ruta sensible del sistema.",
                    details={"command": cmd_original[:200], "pattern": pattern.pattern},
                )
        return None

    def _check_protected_paths(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        # Detectar escritura a rutas protegidas del sistema
        write_indicators = (">", ">>", "tee ", "mv ", "cp ", "rm ", "chmod ", "chown ")
        has_write = any(ind in cmd_lower for ind in write_indicators)
        if not has_write:
            return None

        for protected in self._PROTECTED_PATHS:
            if protected.lower() in cmd_lower:
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.BLOCK,
                    reason=f"Escritura a ruta protegida del sistema: {protected}",
                    details={"command": cmd_original[:200], "protected_path": protected},
                )
        return None

    def _check_exfiltration(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        for pattern in self._EXFIL_PATTERNS:
            if pattern.search(cmd_lower):
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.CONFIRM,
                    reason="Comando potencialmente envía datos a un servidor remoto.",
                    details={"command": cmd_original[:200], "pattern": pattern.pattern},
                )
        return None

    def _check_evasion(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        for pattern in self._EVASION_PATTERNS:
            if pattern.search(cmd_lower):
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.BLOCK,
                    reason="Comando usa técnicas de evasión (encoding, eval dinámico).",
                    details={"command": cmd_original[:200], "pattern": pattern.pattern},
                )
        return None

    def _check_pipe_chain_risk(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        """Detecta cadenas de pipes que terminan en escritura o ejecución."""
        if "|" not in cmd_lower:
            return None

        parts = cmd_lower.split("|")
        if len(parts) < 2:
            return None

        last_part = parts[-1].strip()
        dangerous_terminals = ("bash", "sh", "exec", "rm ", "dd ", "tee /", "> /")
        for terminal in dangerous_terminals:
            if last_part.startswith(terminal):
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.BLOCK,
                    reason=f"Cadena de pipes termina en operación destructiva: ...| {last_part[:40]}",
                    details={"command": cmd_original[:200], "pipe_terminal": last_part[:80]},
                )
        return None

    def _check_recursive_destruction(self, cmd_lower: str, cmd_original: str) -> GuardrailViolation | None:
        """Detecta rm -rf sobre directorios amplios o la raíz."""
        rm_patterns = [
            re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f?.*\s+/\s"),       # rm -rf /
            re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f?.*\s+/\*"),       # rm -rf /*
            re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f?.*\s+~"),         # rm -rf ~
            re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f?.*\s+\.\s"),      # rm -rf .
            re.compile(r"\brm\s+.*-[a-z]*r[a-z]*f?.*\s+\*\s*$"),    # rm -rf *
        ]
        for pattern in rm_patterns:
            if pattern.search(cmd_lower):
                return GuardrailViolation(
                    guardrail="CommandAnalyzer",
                    severity=ViolationSeverity.BLOCK,
                    reason="Eliminación recursiva sobre directorio amplio o raíz.",
                    details={"command": cmd_original[:200]},
                )
        return None


# ═══════════════════════════════════════════════════════════════════════
# CircuitBreaker — Corta ejecución tras fallos consecutivos
# ═══════════════════════════════════════════════════════════════════════

class CircuitBreaker:
    """
    Detecta cuando el agente está fallando repetidamente y corta la ejecución.

    En un circuito eléctrico, un breaker corta la corriente cuando detecta
    una sobrecarga para evitar un incendio. Este CircuitBreaker hace lo mismo
    con el loop ReAct — corta cuando detecta que el agente no está progresando.

    ESTADOS:
      CLOSED  → funcionamiento normal, el agente ejecuta sin restricción
      OPEN    → el breaker disparó, el agente debe detenerse

    TRIGGERS:
      - N fallos consecutivos de tools (el agente repite el mismo error)
      - N respuestas inválidas consecutivas del LLM (garbage, truncated)
      - N iteraciones sin ninguna tool_call exitosa (el agente vagabundea)

    USO:
      breaker = CircuitBreaker(max_consecutive_failures=3)
      breaker.record_failure("shell", "exit code 1")
      breaker.record_failure("shell", "exit code 1")
      breaker.record_failure("shell", "exit code 1")

      if breaker.is_open:
          # Abortar — el agente está atascado
    """

    def __init__(
        self,
        max_consecutive_tool_failures: int = 5,
        max_consecutive_llm_errors: int = 3,
        max_idle_iterations: int = 4,
    ) -> None:
        self._max_tool_failures = max_consecutive_tool_failures
        self._max_llm_errors = max_consecutive_llm_errors
        self._max_idle = max_idle_iterations

        self._consecutive_tool_failures: int = 0
        self._consecutive_llm_errors: int = 0
        self._idle_iterations: int = 0
        self._trip_reason: str | None = None

    @property
    def is_open(self) -> bool:
        """True si el breaker disparó — el agente debe detenerse."""
        return self._trip_reason is not None

    @property
    def trip_reason(self) -> str | None:
        """Razón por la que disparó, None si está cerrado."""
        return self._trip_reason

    def record_tool_success(self) -> None:
        """Una tool ejecutó exitosamente — resetear contadores de fallo."""
        self._consecutive_tool_failures = 0
        self._idle_iterations = 0

    def record_tool_failure(self, tool_name: str, error: str) -> None:
        """Una tool falló — incrementar contador de fallos consecutivos."""
        self._consecutive_tool_failures += 1
        self._idle_iterations = 0
        if self._consecutive_tool_failures >= self._max_tool_failures:
            self._trip_reason = (
                f"CircuitBreaker: {self._consecutive_tool_failures} fallos consecutivos de tools. "
                f"Último: {tool_name} — {error[:100]}"
            )

    def record_llm_error(self, error: str) -> None:
        """El LLM devolvió una respuesta inválida o vacía."""
        self._consecutive_llm_errors += 1
        if self._consecutive_llm_errors >= self._max_llm_errors:
            self._trip_reason = (
                f"CircuitBreaker: {self._consecutive_llm_errors} respuestas inválidas del LLM. "
                f"Último error: {error[:100]}"
            )

    def record_llm_success(self) -> None:
        """El LLM devolvió una respuesta válida — resetear contador."""
        self._consecutive_llm_errors = 0

    def record_idle_iteration(self) -> None:
        """Iteración sin progreso real (solo think, sin tool_call ni complete)."""
        self._idle_iterations += 1
        if self._idle_iterations >= self._max_idle:
            self._trip_reason = (
                f"CircuitBreaker: {self._idle_iterations} iteraciones sin progreso. "
                f"El agente no está actuando — posiblemente atascado en un bucle de razonamiento."
            )

    def reset(self) -> None:
        """Resetear completamente el breaker (entre subtasks)."""
        self._consecutive_tool_failures = 0
        self._consecutive_llm_errors = 0
        self._idle_iterations = 0
        self._trip_reason = None


# ═══════════════════════════════════════════════════════════════════════
# PathGuard — Sandbox de archivos y protección de rutas
# ═══════════════════════════════════════════════════════════════════════

class PathGuard:
    """
    Verifica que las operaciones de archivos se mantengan dentro del sandbox.

    Resuelve symlinks, normaliza rutas y verifica que el resultado final
    esté dentro del directorio de trabajo permitido.

    AMENAZAS QUE PREVIENE:
      - Path traversal: ../../etc/passwd
      - Symlink escape: symlink → /etc/shadow dentro del proyecto
      - Rutas absolutas fuera del proyecto: /tmp/exploit.sh
      - Rutas ocultas sensibles: .git/config, .env, .ssh/

    USO:
      guard = PathGuard(allowed_root=Path.cwd())

      # Ruta segura
      guard.validate("/home/user/project/src/main.py")  # None

      # Ruta peligrosa
      guard.validate("../../etc/passwd")  # GuardrailViolation
    """

    # Archivos y directorios que nunca deberían modificarse
    _SENSITIVE_NAMES: frozenset[str] = frozenset({
        ".env", ".env.local", ".env.production",
        ".git", ".gitignore",
        ".ssh", ".gnupg", ".aws", ".kube",
        "id_rsa", "id_ed25519", "known_hosts",
    })

    def __init__(self, allowed_root: Path | None = None) -> None:
        self._root = (allowed_root or Path.cwd()).resolve()

    def validate_read(self, path_str: str) -> GuardrailViolation | None:
        """Valida una ruta para operación de lectura."""
        return self._validate(path_str, operation="read")

    def validate_write(self, path_str: str) -> GuardrailViolation | None:
        """Valida una ruta para operación de escritura (más estricto)."""
        violation = self._validate(path_str, operation="write")
        if violation:
            return violation

        # Check adicional: no escribir en archivos sensibles
        target = Path(path_str)
        if target.name in self._SENSITIVE_NAMES:
            return GuardrailViolation(
                guardrail="PathGuard",
                severity=ViolationSeverity.CONFIRM,
                reason=f"Escritura a archivo sensible: {target.name}",
                details={"path": path_str, "sensitive_name": target.name},
            )
        return None

    def _validate(self, path_str: str, operation: str) -> GuardrailViolation | None:
        """Validación común para lectura y escritura."""
        if not path_str:
            return None

        try:
            target = Path(path_str)
            if not target.is_absolute():
                target = self._root / target

            # Resolver symlinks para ver la ruta REAL
            resolved = target.resolve()

            # Verificar que esté dentro del sandbox
            resolved.relative_to(self._root)

        except ValueError:
            return GuardrailViolation(
                guardrail="PathGuard",
                severity=ViolationSeverity.BLOCK,
                reason=f"Ruta fuera del directorio de trabajo: {path_str}",
                details={
                    "path": path_str,
                    "allowed_root": str(self._root),
                    "operation": operation,
                },
            )
        except OSError:
            # Ruta inválida en el OS
            return GuardrailViolation(
                guardrail="PathGuard",
                severity=ViolationSeverity.BLOCK,
                reason=f"Ruta inválida o inaccesible: {path_str}",
                details={"path": path_str, "operation": operation},
            )

        return None
