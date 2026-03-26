"""
ShellTool — Ejecuta comandos en el sistema operativo.

Es la tool más crítica del agente. Con ella el agente puede:
  - Instalar dependencias (pip install, npm install)
  - Correr tests (pytest, jest, cargo test)
  - Compilar código (cargo build, go build, tsc)
  - Ejecutar scripts (python script.py, ./setup.sh)
  - Cualquier operación que requiera la terminal

SEGURIDAD:
  ShellTool es potencialmente destructiva — un comando como
  `rm -rf /` causaría daño irreversible. Por eso implementa
  is_safe_to_run() con una lista de patrones peligrosos que
  requieren confirmación explícita del usuario.

  Los comandos que pasan is_safe_to_run() se ejecutan directamente.
  Los que no pasan retornan ToolResult.failure() con un mensaje
  descriptivo — el agente puede pedir confirmación al usuario.

COMPATIBILIDAD OS:
  Compatible con Linux, Mac y Windows.
  En Windows usa subprocess con shell=True para manejar comandos nativos.
  En Linux/Mac usa shell=True con /bin/sh.

WORKING DIRECTORY:
  Por defecto ejecuta en el directorio de trabajo actual del proceso.
  El LLM puede especificar un working_dir diferente para comandos
  que necesitan ejecutarse en un directorio específico del proyecto.

STREAMS:
  Captura tanto stdout como stderr.
  Los combina en el output para que el LLM tenga contexto completo
  de lo que ocurrió — incluidos los mensajes de error.
"""

from __future__ import annotations

import re
import subprocess
import sys
from typing import Any

from hiperforge.core.constants import (
    TOOL_DEFAULT_TIMEOUT_SECONDS,
    TOOL_EXTENDED_TIMEOUT_SECONDS,
)
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.exceptions import ToolTimeoutError
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.tools.base import BaseTool, register_tool
from hiperforge.core.guardrails import CommandAnalyzer, ViolationSeverity

logger = get_logger(__name__)

# Instancia singleton del analizador de comandos — reutilizada en cada llamada
_command_analyzer = CommandAnalyzer()

# ---------------------------------------------------------------------------
# Patrones de comandos peligrosos que requieren confirmación del usuario.
#
# Esta lista NO es exhaustiva — es una red de seguridad para los casos
# más obvios. La responsabilidad final de qué ejecutar es del usuario.
#
# Formato: lista de patrones regex compilados.
# Se aplican contra el comando completo en lowercase.
# ---------------------------------------------------------------------------
_DANGEROUS_PATTERNS: list[re.Pattern[str]] = [
    re.compile(r"\brm\s+-rf?\b"),           # rm -rf, rm -r
    re.compile(r"\brm\s+--recursive\b"),     # rm --recursive
    re.compile(r"\bmkfs\b"),                 # formatear filesystem
    re.compile(r"\bdd\b.+\bof=/dev/"),       # dd escribiendo a dispositivo
    re.compile(r"\bchmod\s+777\b"),          # permisos abiertos totales
    re.compile(r"\bsudo\b"),                 # escalada de privilegios
    re.compile(r"\bsu\s+-?\b"),              # cambio de usuario
    re.compile(r">\s*/dev/sd"),              # sobreescribir disco
    re.compile(r"\bshutdown\b"),             # apagar el sistema
    re.compile(r"\breboot\b"),               # reiniciar el sistema
    re.compile(r"\bkill\s+-9\b"),            # matar procesos forzado
    re.compile(r"\bpkill\b"),                # matar procesos por nombre
    re.compile(r":\(\)\{.*\|.*&\s*\}"),     # fork bomb
]

# Comandos de solo lectura que nunca son peligrosos.
# Si el comando empieza con alguno de estos, saltamos la verificación de seguridad.
_SAFE_PREFIXES: frozenset[str] = frozenset({
    "cat", "ls", "pwd", "echo", "which", "whereis",
    "git status", "git log", "git diff", "git branch",
    "python --version", "python3 --version",
    "node --version", "npm --version",
    "cargo --version", "go version",
    "pytest", "python -m pytest",
    "grep", "find", "head", "tail", "wc",
    "env", "printenv", "uname",
})


@register_tool
class ShellTool(BaseTool):
    """
    Tool para ejecutar comandos en la terminal del sistema.

    Es la tool de mayor impacto del agente — puede hacer casi
    cualquier cosa, por eso tiene la verificación de seguridad
    más estricta de todas las tools.
    """

    @property
    def name(self) -> str:
        return "shell"

    @property
    def description(self) -> str:
        return "Ejecuta comandos en la terminal del sistema operativo"

    # ------------------------------------------------------------------
    # Schema para el LLM
    # ------------------------------------------------------------------

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Ejecuta un comando en la terminal del sistema. "
                "Usa esta tool para: instalar dependencias, correr tests, "
                "compilar código, ejecutar scripts, o cualquier operación "
                "que requiera la terminal. "
                "Captura tanto stdout como stderr. "
                "Si el comando falla, el exit code y el error estarán en el output. "
                "Evita usarla para explorar el repo con ls/find repetidos si ya conoces la ruta objetivo. "
                "Después de una verificación exitosa no repitas el mismo comando salvo que haya cambiado algo."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": (
                            "El comando completo a ejecutar. "
                            "Ejemplos: 'pytest tests/ -v', "
                            "'pip install requests', "
                            "'git status', "
                            "'python main.py --debug'"
                        ),
                    },
                    "working_dir": {
                        "type": "string",
                        "description": (
                            "Directorio desde donde ejecutar el comando. "
                            "Default: directorio de trabajo actual del proceso. "
                            "Usar cuando el comando necesita ejecutarse dentro "
                            "de un subdirectorio específico del proyecto."
                        ),
                    },
                    "timeout": {
                        "type": "number",
                        "description": (
                            f"Tiempo máximo de ejecución en segundos. "
                            f"Default: {TOOL_DEFAULT_TIMEOUT_SECONDS}s. "
                            f"Para operaciones lentas (npm install, cargo build) "
                            f"usar hasta {TOOL_EXTENDED_TIMEOUT_SECONDS}s."
                        ),
                    },
                    "extended_timeout": {
                        "type": "boolean",
                        "description": (
                            "Si true, usa el timeout extendido "
                            f"({TOOL_EXTENDED_TIMEOUT_SECONDS}s) automáticamente. "
                            "Útil para instalaciones de dependencias o compilaciones."
                        ),
                    },
                },
                "required": ["command"],
            },
        )

    # ------------------------------------------------------------------
    # Verificación de seguridad
    # ------------------------------------------------------------------

    def is_safe_to_run(self, arguments: dict[str, Any]) -> bool:
        """
        Verifica si el comando es seguro para ejecutar sin confirmación.

        Dos niveles de verificación:
          1. Patrones rápidos (regex) — detectan los casos más obvios
          2. CommandAnalyzer (análisis profundo) — detecta path traversal,
             exfiltración, evasión, pipe chains y destrucción recursiva

        Retorna False si el comando coincide con algún patrón peligroso
        o si el CommandAnalyzer detecta un riesgo de severidad BLOCK.
        """
        raw_command = arguments.get("command", "").strip()
        command = raw_command.lower()

        if not command:
            return True  # comando vacío — fallará en execute(), no es peligroso

        # Bloquear operadores de control/redirección para evitar chains mutantes
        # incluso en comandos con prefijo normalmente seguro (ej: "ls && rm ...")
        if any(op in command for op in ("&&", "||", "|", ">", "<", ";")):
            logger.warning(
                "comando bloqueado por operador de control",
                command_preview=command[:100],
                task_id=self._task_id,
            )
            return False

        # Comandos de solo lectura conocidos — siempre seguros
        for safe_prefix in _SAFE_PREFIXES:
            if command.startswith(safe_prefix.lower()):
                return True

        # Nivel 1: Verificar contra patrones peligrosos rápidos
        for pattern in _DANGEROUS_PATTERNS:
            if pattern.search(command):
                logger.warning(
                    "comando bloqueado por patrón peligroso",
                    command_preview=command[:100],
                    pattern=pattern.pattern,
                    task_id=self._task_id,
                )
                return False

        # Nivel 2: Análisis profundo con CommandAnalyzer
        # Detecta amenazas que los patrones simples no ven:
        # path traversal, exfiltración, evasión, pipe chains
        violation = _command_analyzer.analyze(arguments.get("command", "").strip())
        if violation is not None:
            if violation.severity == ViolationSeverity.BLOCK:
                logger.warning(
                    "comando bloqueado por CommandAnalyzer",
                    command_preview=command[:100],
                    guardrail=violation.guardrail,
                    reason=violation.reason,
                    task_id=self._task_id,
                )
                return False
            elif violation.severity == ViolationSeverity.CONFIRM:
                logger.warning(
                    "comando requiere confirmación por CommandAnalyzer",
                    command_preview=command[:100],
                    reason=violation.reason,
                    task_id=self._task_id,
                )
                return False  # En modo agente, CONFIRM también bloquea

        return True

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        Validaciones adicionales específicas de ShellTool.

        Extiende la validación base con checks específicos:
          - Comando no puede estar vacío
          - Timeout debe ser positivo si se especifica
          - working_dir debe existir si se especifica
        """
        # Primero las validaciones base (campos requeridos)
        errors = super().validate_arguments(arguments)

        command = arguments.get("command", "").strip()
        if not command:
            errors.append("El comando no puede estar vacío")

        timeout = arguments.get("timeout")
        if timeout is not None:
            if not isinstance(timeout, (int, float)):
                errors.append(f"timeout debe ser un número, recibido: {type(timeout).__name__}")
            elif timeout <= 0:
                errors.append(f"timeout debe ser mayor que 0, recibido: {timeout}")
            elif timeout > TOOL_EXTENDED_TIMEOUT_SECONDS:
                errors.append(
                    f"timeout {timeout}s excede el máximo permitido "
                    f"({TOOL_EXTENDED_TIMEOUT_SECONDS}s)"
                )

        working_dir = arguments.get("working_dir")
        if working_dir is not None:
            from pathlib import Path
            dir_path = Path(working_dir)
            if not dir_path.exists():
                errors.append(f"working_dir '{working_dir}' no existe")
            elif not dir_path.is_dir():
                errors.append(f"working_dir '{working_dir}' no es un directorio")

        return errors

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Ejecuta el comando en la terminal y devuelve el resultado.

        Captura stdout y stderr combinados para que el LLM tenga
        contexto completo del resultado — incluyendo mensajes de error.

        El exit code del proceso se incluye en el output cuando es != 0
        para que el LLM pueda diagnosticar el fallo correctamente.
        """
        command = arguments["command"].strip()
        working_dir = arguments.get("working_dir")
        extended = arguments.get("extended_timeout", False)

        # Resolver el timeout efectivo
        timeout = self._resolve_timeout(arguments, extended)

        logger.debug(
            "ejecutando comando",
            command=command,
            working_dir=working_dir,
            timeout=timeout,
            task_id=self._task_id,
        )

        call_id = self._get_active_tool_call_id()
        try:
            result = subprocess.run(
                command,
                shell=True,
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=working_dir,
                # Evitar que el proceso hijo herede señales del proceso padre
                # En Unix esto evita que Ctrl+C mate al proceso hijo antes
                # de que podamos capturar su output
                start_new_session=True if sys.platform != "win32" else False,
            )

            # Combinamos stdout y stderr para dar al LLM contexto completo
            output = self._build_output(
                stdout=result.stdout,
                stderr=result.stderr,
                exit_code=result.returncode,
            )

            if result.returncode == 0:
                return ToolResult.success(
                    tool_call_id=call_id,
                    output=output,
                )
            else:
                # Exit code != 0 es un fallo del comando, no de la tool
                # Lo devolvemos como failure para que el LLM lo observe
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message=f"Comando terminó con exit code {result.returncode}",
                    output=output,
                )

        except subprocess.TimeoutExpired as exc:
            # Timeout del subprocess — lo convertimos a ToolTimeoutError
            # para que execute_safe() lo propague al executor
            raise ToolTimeoutError(
                tool_name=self.name,
                timeout_seconds=timeout,
            ) from exc

        except FileNotFoundError as exc:
            # El comando no existe en el sistema
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Comando no encontrado: {command.split()[0]}",
                output=str(exc),
            )

        except PermissionError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Sin permisos para ejecutar: {command}",
                output=str(exc),
            )

        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Error del sistema operativo: {exc}",
                output=str(exc),
            )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _resolve_timeout(
        self,
        arguments: dict[str, Any],
        extended: bool,
    ) -> float:
        """
        Resuelve el timeout efectivo para este comando.

        Prioridad:
          1. Timeout explícito en los argumentos
          2. extended_timeout=True → TOOL_EXTENDED_TIMEOUT_SECONDS
          3. Default → TOOL_DEFAULT_TIMEOUT_SECONDS
        """
        explicit = arguments.get("timeout")
        if explicit is not None:
            return float(explicit)

        if extended:
            return TOOL_EXTENDED_TIMEOUT_SECONDS

        return TOOL_DEFAULT_TIMEOUT_SECONDS

    def _build_output(
        self,
        stdout: str,
        stderr: str,
        exit_code: int,
    ) -> str:
        """
        Construye el output combinado de stdout y stderr.

        FORMATO:
          Si hay stdout y stderr:
            [stdout]
            --- stderr ---
            [stderr]
            --- exit code: N ---

          Si solo hay stdout:
            [stdout]

          Si solo hay stderr (comando fallido sin output):
            [stderr]
            --- exit code: N ---

        Este formato permite al LLM entender claramente qué salió
        en stdout vs stderr, y cuál fue el resultado del comando.
        """
        parts = []

        stdout = stdout.strip()
        stderr = stderr.strip()

        if stdout:
            parts.append(stdout)

        if stderr:
            if parts:
                parts.append("--- stderr ---")
            parts.append(stderr)

        if exit_code != 0:
            parts.append(f"--- exit code: {exit_code} ---")

        # Si no hubo output de ningún tipo, lo indicamos explícitamente
        if not parts:
            parts.append("(sin output)")

        return "\n".join(parts)
