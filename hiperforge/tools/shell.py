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

from pathlib import Path
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
from hiperforge.core.guardrails import CommandAnalyzer, PathGuard, ViolationSeverity

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
                "Después de una verificación exitosa no repitas el mismo comando salvo que haya cambiado algo. "
                "Para crear archivos prefiere FileTool/write en vez de redirecciones > o heredoc."
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
                            f"Tiempo máximo en segundos. RANGO VÁLIDO: 1 a "
                            f"{int(TOOL_EXTENDED_TIMEOUT_SECONDS)}. "
                            f"Default: {int(TOOL_DEFAULT_TIMEOUT_SECONDS)}s (se usa si omites este campo). "
                            f"PROHIBIDO enviar valores mayores a {int(TOOL_EXTENDED_TIMEOUT_SECONDS)}. "
                            f"Para la mayoría de comandos, OMITE este campo y usa el default."
                        ),
                    },
                    "extended_timeout": {
                        "type": "boolean",
                        "description": (
                            "Si true, usa el timeout extendido "
                            f"({int(TOOL_EXTENDED_TIMEOUT_SECONDS)}s) automáticamente. "
                            "Útil para instalaciones de dependencias o compilaciones. "
                            "Preferir este flag en vez de especificar timeout manualmente."
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

        Tres niveles de verificación:
          1. Auto-transformación de `cd <path> && <cmd>` a working_dir (sin bloquear)
          2. Si hay operadores de control (&&, ||, ;, |), verificar CADA segmento
             individualmente contra patrones peligrosos
          3. CommandAnalyzer (análisis profundo) — detecta path traversal,
             exfiltración, evasión, pipe chains y destrucción recursiva

        CAMBIO RESPECTO A LA VERSIÓN ANTERIOR:
          Antes se bloqueaban TODOS los comandos con &&, ||, etc. Esto era
          demasiado agresivo — el LLM usa `cd /path && python script.py`
          constantemente, que es un patrón perfectamente seguro. Ahora
          verificamos cada segmento del chain individualmente.
        """
        raw_command = arguments.get("command", "").strip()
        command = raw_command.lower()

        if not command:
            return True  # comando vacío — fallará en execute(), no es peligroso

        # ── AUTO-TRANSFORMACIÓN: cd <path> && <resto> ─────────────────
        # El LLM usa este patrón constantemente en vez de working_dir.
        # En vez de bloquearlo, lo transformamos transparentemente.
        self._auto_transform_cd_chain(arguments)
        raw_command = arguments.get("command", "").strip()
        command = raw_command.lower()

        if self._is_safe_heredoc_write(arguments):
            return True

        # ── BLOQUEO DE REDIRECTS ──────────────────────────────────────
        # Los operadores de redirección (>, <, >>) pueden escribir a
        # archivos arbitrarios incluso desde comandos aparentemente seguros.
        # Se bloquean siempre, a diferencia de && y || que se verifican
        # por segmentos.
        if any(op in command for op in (">", "<")):
            logger.warning(
                "comando bloqueado por operador de redirección",
                command_preview=command[:100],
                task_id=self._task_id,
            )
            return False

        # ── VERIFICACIÓN POR SEGMENTOS ────────────────────────────────
        # Si el comando contiene operadores de control, verificamos cada
        # segmento individualmente contra patrones peligrosos.
        # Un chain es seguro solo si TODOS sus segmentos son seguros.
        if any(op in command for op in ("&&", "||", ";", "|")):
            segments = re.split(r'\s*(?:&&|\|\|?|;)\s*', command)
            for segment in segments:
                segment = segment.strip()
                if not segment:
                    continue

                # Verificar contra patrones peligrosos
                for pattern in _DANGEROUS_PATTERNS:
                    if pattern.search(segment):
                        logger.warning(
                            "segmento de comando bloqueado por patrón peligroso",
                            command_preview=command[:100],
                            dangerous_segment=segment[:60],
                            pattern=pattern.pattern,
                            task_id=self._task_id,
                        )
                        return False

            # Verificar el comando completo con CommandAnalyzer
            violation = _command_analyzer.analyze(raw_command)
            if violation is not None and violation.severity in (
                ViolationSeverity.BLOCK,
                ViolationSeverity.CONFIRM,
            ):
                logger.warning(
                    "comando con operadores bloqueado por CommandAnalyzer",
                    command_preview=command[:100],
                    reason=violation.reason,
                    task_id=self._task_id,
                )
                return False

            # Todos los segmentos son seguros y CommandAnalyzer aprueba
            return True

        # ── CAMINO ESTÁNDAR (sin operadores) ──────────────────────────
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
        violation = _command_analyzer.analyze(raw_command)
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

    @staticmethod
    def _auto_transform_cd_chain(arguments: dict[str, Any]) -> None:
        """
        Transforma `cd <path> && <cmd>` en working_dir=<path>, command=<cmd>.

        Este es el patrón más frecuente que los LLMs generan para ejecutar
        comandos en un directorio específico. En vez de bloquearlo por el
        operador &&, lo convertimos transparentemente al formato correcto
        usando el parámetro working_dir que ShellTool ya soporta.

        PATRON DETECTADO:
          command: "cd /home/user/project && python script.py"
          →
          command: "python script.py"
          working_dir: "/home/user/project"

        SOLO se transforma si:
          - El comando empieza con "cd " seguido de una ruta
          - La ruta no contiene operadores peligrosos
          - No hay working_dir ya especificado en los argumentos
          - El comando después del && no contiene más operadores de cadena
        """
        command = arguments.get("command", "").strip()
        if not command.startswith("cd "):
            return

        # Solo transformar si no hay working_dir ya especificado
        if arguments.get("working_dir"):
            return

        # Buscar el patrón "cd <path> && <rest>"
        match = re.match(
            r'^cd\s+([^\s&|;><]+(?:\s+[^\s&|;><]+)*?)\s+&&\s+(.+)$',
            command,
            re.DOTALL,
        )
        if not match:
            return

        cd_path = match.group(1).strip()
        rest_command = match.group(2).strip()

        # Verificación de seguridad: la ruta no debe contener operadores
        if any(op in cd_path for op in ("&&", "||", ";", "|", ">", "<")):
            return

        # Aplicar la transformación
        arguments["working_dir"] = cd_path
        arguments["command"] = rest_command

        logger.debug(
            "auto-transformado cd && chain a working_dir",
            original_command=command[:100],
            new_command=rest_command[:80],
            working_dir=cd_path,
        )

    @staticmethod
    def _resolve_case_insensitive_path(path_str: str) -> str | None:
        """
        Corrige rutas con casing incorrecto cuando existe una coincidencia única.
        """
        path = Path(path_str)
        if path.exists():
            return str(path)
        if not path.is_absolute():
            return None

        current = Path(path.anchor or "/")
        for part in path.parts[1:]:
            if not current.exists() or not current.is_dir():
                return None

            try:
                entries = {child.name.lower(): child.name for child in current.iterdir()}
            except OSError:
                return None

            matched_name = entries.get(part.lower())
            if matched_name is None:
                return None
            current = current / matched_name

        return str(current)

    @staticmethod
    def _extract_heredoc_write(command: str) -> tuple[str, str, str] | None:
        """
        Detecta un heredoc mínimo del tipo `cat > file <<TAG ... TAG`.
        """
        match = re.match(
            r"^cat\s+(>>?)\s+([^\s><|;&]+)\s+<<['\"]?([A-Za-z0-9_-]+)['\"]?\n([\s\S]*)\n\3\s*$",
            command,
        )
        if not match:
            return None

        redirect_op, path_str, _, body = match.groups()
        mode = "append" if redirect_op == ">>" else "write"
        return mode, path_str, body

    def _is_safe_heredoc_write(self, arguments: dict[str, Any]) -> bool:
        """
        Permite un heredoc estricto solo si escribe dentro del proyecto.
        """
        command = arguments.get("command", "").strip()
        extracted = self._extract_heredoc_write(command)
        if extracted is None:
            return False

        _, path_str, _ = extracted
        target = Path(path_str)
        working_dir = arguments.get("working_dir")
        if not target.is_absolute():
            base_dir = Path(working_dir) if working_dir else Path.cwd()
            target = base_dir / target

        violation = PathGuard(allowed_root=Path.cwd()).validate_write(str(target))
        if violation is not None and violation.severity in (
            ViolationSeverity.BLOCK,
            ViolationSeverity.CONFIRM,
        ):
            logger.warning(
                "heredoc bloqueado por PathGuard",
                path=str(target),
                reason=violation.reason,
                task_id=self._task_id,
            )
            return False

        return True

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        Validaciones adicionales específicas de ShellTool.

        Extiende la validación base con checks específicos:
          - Comando no puede estar vacío.
          - Timeout debe ser positivo si se especifica.
          - Timeout fuera de rango se AUTO-CLAMPEA al máximo válido en vez de
            rechazar. Esto evita que el LLM entre en un bucle infinito enviando
            el mismo timeout inválido — un patrón observado frecuentemente con
            modelos que ignoran los mensajes de error de validación.
          - working_dir debe existir si se especifica.
        """
        # Primero las validaciones base (campos requeridos)
        self._auto_transform_cd_chain(arguments)
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
                # ══════════════════════════════════════════════════════════════
                # AUTO-CLAMP: en vez de rechazar el comando completo, clampeamos
                # el timeout al máximo válido y continuamos con la ejecución.
                #
                # RAZÓN: Muchos modelos LLM (especialmente los más pequeños)
                # ignoran los mensajes de error de validación y repiten los
                # mismos argumentos inválidos indefinidamente. El rechazo duro
                # causaba bucles de 5+ iteraciones donde el agente enviaba
                # timeout=100000 → error → timeout=100000 → error → ...
                # hasta agotar el límite de iteraciones sin haber ejecutado
                # ni un solo comando.
                #
                # El clampeo permite que el comando se ejecute con un timeout
                # razonable, y el agente puede avanzar con su tarea.
                # ══════════════════════════════════════════════════════════════
                logger.warning(
                    "timeout auto-clampeado al máximo permitido",
                    original_timeout=timeout,
                    clamped_timeout=TOOL_EXTENDED_TIMEOUT_SECONDS,
                    task_id=self._task_id,
                )
                arguments["timeout"] = TOOL_EXTENDED_TIMEOUT_SECONDS

        working_dir = arguments.get("working_dir")
        if working_dir is not None:
            dir_path = Path(working_dir)
            if not dir_path.exists():
                corrected = self._resolve_case_insensitive_path(working_dir)
                if corrected is not None:
                    arguments["working_dir"] = corrected
                    dir_path = Path(corrected)

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
        heredoc_result = self._execute_safe_heredoc_write(arguments, call_id)
        if heredoc_result is not None:
            return heredoc_result

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

    def _execute_safe_heredoc_write(
        self,
        arguments: dict[str, Any],
        call_id: str,
    ) -> ToolResult | None:
        """
        Ejecuta localmente un heredoc seguro sin invocar el shell real.
        """
        command = arguments.get("command", "").strip()
        extracted = self._extract_heredoc_write(command)
        if extracted is None:
            return None

        mode, path_str, body = extracted
        target = Path(path_str)
        working_dir = arguments.get("working_dir")
        if not target.is_absolute():
            base_dir = Path(working_dir) if working_dir else Path.cwd()
            target = base_dir / target

        target.parent.mkdir(parents=True, exist_ok=True)

        payload = body + "\n"

        try:
            if mode == "append":
                with target.open("a", encoding="utf-8") as handle:
                    handle.write(payload)
                action = "actualizado"
            else:
                target.write_text(payload, encoding="utf-8")
                action = "creado"
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo escribir '{target}': {exc}",
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output=(
                f"Archivo {action} via heredoc seguro: {target}\n"
                f"{payload.count(chr(10))} líneas, {len(payload)} caracteres"
            ),
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
