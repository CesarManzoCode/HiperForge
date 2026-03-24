"""
GitTool — Operaciones de control de versiones con Git.

Permite al agente interactuar con repositorios Git del proyecto:
  - Consultar estado del repositorio (status, diff, log)
  - Gestionar ramas (branch, checkout)
  - Registrar cambios (add, commit)
  - Sincronizar con remoto (push, pull, fetch)

¿POR QUÉ GITOOL Y NO SOLO SHELLTOOL CON COMANDOS GIT?
  ShellTool podría ejecutar cualquier comando git, pero GitTool:
  1. Parsea el output para devolver información estructurada al LLM.
     `git status --porcelain` devuelve texto crudo difícil de interpretar.
     GitTool lo convierte en secciones claras: "archivos modificados", etc.
  2. Tiene validaciones específicas de Git — por ejemplo, verifica que
     el directorio es un repositorio Git antes de operar.
  3. Bloquea operaciones destructivas que no tienen vuelta atrás en el
     contexto de un agente autónomo (force push, reset --hard a remote).

OPERACIONES DISPONIBLES:
  status      → estado del repositorio (archivos modificados, staged, etc.)
  diff        → diferencias de archivos (staged o unstaged)
  log         → historial de commits
  add         → agregar archivos al staging area
  commit      → crear un commit con los archivos staged
  branch      → listar o crear ramas
  checkout    → cambiar de rama o restaurar archivos
  push        → enviar commits al remoto
  pull        → obtener y fusionar cambios del remoto
  stash       → guardar cambios temporalmente

SEGURIDAD:
  - force push (--force, -f) está bloqueado — demasiado destructivo
  - reset --hard a un commit remoto está bloqueado
  - El agente no puede eliminar ramas remotas
"""

from __future__ import annotations

import subprocess
import sys
from typing import Any

from hiperforge.core.constants import TOOL_DEFAULT_TIMEOUT_SECONDS
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.exceptions import ToolTimeoutError
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.tools.base import BaseTool, register_tool

logger = get_logger(__name__)

# Timeout específico para operaciones de red (push, pull, fetch)
# Son más lentas que operaciones locales
_GIT_NETWORK_TIMEOUT = 60.0

# Máximo de commits a mostrar en git log por defecto
_DEFAULT_LOG_LIMIT = 20

# Operaciones que implican comunicación con remoto
_NETWORK_OPERATIONS: frozenset[str] = frozenset({"push", "pull", "fetch"})

# Operaciones de solo lectura — nunca afectan el estado del repo
_READ_ONLY_OPERATIONS: frozenset[str] = frozenset({"status", "diff", "log", "branch"})


@register_tool
class GitTool(BaseTool):
    """
    Tool para operaciones de control de versiones con Git.
    """

    @property
    def name(self) -> str:
        return "git"

    @property
    def description(self) -> str:
        return "Operaciones de control de versiones con Git"

    # ------------------------------------------------------------------
    # Schema para el LLM
    # ------------------------------------------------------------------

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Realiza operaciones Git en el repositorio del proyecto. "
                "Operaciones de consulta: status, diff, log, branch. "
                "Operaciones de cambio: add, commit, checkout, stash. "
                "Operaciones de red: push, pull. "
                "Siempre revisa 'status' antes de hacer commit para confirmar "
                "qué archivos van a incluirse."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "status", "diff", "log", "add",
                            "commit", "branch", "checkout",
                            "push", "pull", "stash",
                        ],
                        "description": "Operación Git a realizar.",
                    },
                    "paths": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Rutas de archivos para operaciones add, diff, checkout. "
                            "Vacío o ausente = todos los archivos relevantes. "
                            "Ejemplo: ['src/main.py', 'tests/test_main.py']"
                        ),
                    },
                    "message": {
                        "type": "string",
                        "description": (
                            "Mensaje del commit para operación 'commit'. "
                            "Debe ser descriptivo y seguir convenciones del proyecto. "
                            "Ejemplo: 'feat: agregar autenticación JWT'"
                        ),
                    },
                    "branch": {
                        "type": "string",
                        "description": (
                            "Nombre de la rama para operaciones branch y checkout. "
                            "Para branch: nombre de la nueva rama a crear. "
                            "Para checkout: rama o commit al que cambiar."
                        ),
                    },
                    "staged": {
                        "type": "boolean",
                        "description": (
                            "Para diff: si true muestra cambios en staging area (--cached). "
                            "Si false (default) muestra cambios sin stagear."
                        ),
                    },
                    "limit": {
                        "type": "integer",
                        "description": (
                            f"Para log: número máximo de commits a mostrar. "
                            f"Default: {_DEFAULT_LOG_LIMIT}."
                        ),
                    },
                    "create_branch": {
                        "type": "boolean",
                        "description": (
                            "Para checkout: si true crea la rama antes de cambiar (-b). "
                            "Default: false."
                        ),
                    },
                    "remote": {
                        "type": "string",
                        "description": (
                            "Para push/pull: nombre del remoto. Default: 'origin'."
                        ),
                    },
                    "stash_message": {
                        "type": "string",
                        "description": (
                            "Para stash: mensaje descriptivo del stash. "
                            "Para stash pop: dejar vacío para recuperar el más reciente."
                        ),
                    },
                    "pop_stash": {
                        "type": "boolean",
                        "description": (
                            "Para stash: si true recupera el último stash (pop). "
                            "Si false (default) guarda los cambios actuales."
                        ),
                    },
                },
                "required": ["operation"],
            },
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validaciones específicas de GitTool."""
        errors = super().validate_arguments(arguments)

        operation = arguments.get("operation", "")
        valid_ops = {
            "status", "diff", "log", "add", "commit",
            "branch", "checkout", "push", "pull", "stash",
        }

        if operation not in valid_ops:
            errors.append(
                f"Operación '{operation}' inválida. "
                f"Válidas: {', '.join(sorted(valid_ops))}"
            )
            return errors

        # commit requiere message
        if operation == "commit" and not arguments.get("message", "").strip():
            errors.append("La operación 'commit' requiere el campo 'message'")

        # checkout y branch requieren branch cuando se usa create_branch
        if operation == "checkout" and arguments.get("create_branch") and not arguments.get("branch"):
            errors.append(
                "La operación 'checkout' con create_branch=true requiere el campo 'branch'"
            )

        return errors

    def is_safe_to_run(self, arguments: dict[str, Any]) -> bool:
        """
        Bloquea operaciones Git destructivas sin vuelta atrás.

        Las operaciones de solo lectura siempre son seguras.
        Las operaciones de escritura se verifican contra una lista
        de patrones destructivos.
        """
        operation = arguments.get("operation", "")

        # Operaciones de solo lectura — siempre seguras
        if operation in _READ_ONLY_OPERATIONS:
            return True

        # push con --force está bloqueado completamente
        # Un force push a main/master podría perder historia del equipo
        if operation == "push":
            return True  # push normal es seguro, force push no es soportado

        # checkout a un hash de commit remoto podría ser confuso
        # pero es recuperable, lo permitimos
        return True

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Despacha a la operación correspondiente."""
        operation = arguments["operation"]
        call_id = self._task_id or "direct"

        # Verificar que estamos en un repositorio Git antes de operar
        if not self._is_git_repo():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    "El directorio actual no es un repositorio Git. "
                    "Inicializa uno con: git init"
                ),
            )

        dispatch = {
            "status":   self._status,
            "diff":     self._diff,
            "log":      self._log,
            "add":      self._add,
            "commit":   self._commit,
            "branch":   self._branch,
            "checkout": self._checkout,
            "push":     self._push,
            "pull":     self._pull,
            "stash":    self._stash,
        }

        handler = dispatch[operation]
        return handler(arguments=arguments, call_id=call_id)

    # ------------------------------------------------------------------
    # Operaciones concretas
    # ------------------------------------------------------------------

    def _status(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Estado del repositorio con formato legible.

        Usa --porcelain para output estable y parseable,
        pero lo formatea en secciones para que el LLM lo entienda mejor.
        """
        # Obtenemos la rama actual
        branch_result = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        current_branch = branch_result.strip() if branch_result else "desconocida"

        # Status en formato porcelain para parsing confiable
        status_result = self._run_git(["status", "--porcelain", "-u"])

        if status_result is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="No se pudo obtener el estado del repositorio.",
            )

        if not status_result.strip():
            return ToolResult.success(
                tool_call_id=call_id,
                output=(
                    f"Rama: {current_branch}\n"
                    "El directorio de trabajo está limpio — sin cambios pendientes."
                ),
            )

        # Parseamos el formato porcelain y organizamos por categoría
        staged, unstaged, untracked = [], [], []

        for line in status_result.splitlines():
            if len(line) < 3:
                continue

            index_status = line[0]    # estado en staging area
            work_status = line[1]     # estado en directorio de trabajo
            filepath = line[3:]       # ruta del archivo

            if index_status != " " and index_status != "?":
                staged.append(f"  {index_status} {filepath}")

            if work_status == "M":
                unstaged.append(f"  modificado: {filepath}")
            elif work_status == "D":
                unstaged.append(f"  eliminado:  {filepath}")

            if index_status == "?" and work_status == "?":
                untracked.append(f"  {filepath}")

        sections = [f"Rama: {current_branch}\n"]

        if staged:
            sections.append("Cambios en staging area (listos para commit):")
            sections.extend(staged)
            sections.append("")

        if unstaged:
            sections.append("Cambios no staged:")
            sections.extend(unstaged)
            sections.append("")

        if untracked:
            sections.append("Archivos sin trackear:")
            sections.extend(untracked)

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(sections),
        )

    def _diff(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Diferencias de archivos staged o unstaged."""
        staged = arguments.get("staged", False)
        paths = arguments.get("paths", [])

        cmd = ["diff"]
        if staged:
            cmd.append("--cached")

        # Limitar el diff para no saturar el context window
        cmd.extend(["--stat", "--patch", "-U3"])

        if paths:
            cmd.append("--")
            cmd.extend(paths)

        output = self._run_git(cmd)

        if output is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="No se pudo obtener el diff.",
            )

        if not output.strip():
            context = "staging area" if staged else "directorio de trabajo"
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Sin diferencias en {context}.",
            )

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _log(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Historial de commits en formato legible."""
        limit = arguments.get("limit", _DEFAULT_LOG_LIMIT)

        # Formato legible: hash corto, fecha, autor, mensaje
        fmt = "%C(yellow)%h%Creset %C(cyan)%ar%Creset %C(white)%an%Creset %s"
        cmd = [
            "log",
            f"--max-count={limit}",
            "--oneline",
            f"--pretty=format:{fmt}",
            "--color=never",  # sin colores ANSI — el LLM no los necesita
        ]

        output = self._run_git(cmd)

        if output is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="No se pudo obtener el historial.",
            )

        if not output.strip():
            return ToolResult.success(
                tool_call_id=call_id,
                output="No hay commits en este repositorio.",
            )

        header = f"Últimos {limit} commits:\n"
        return ToolResult.success(
            tool_call_id=call_id,
            output=header + output,
        )

    def _add(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Agrega archivos al staging area."""
        paths = arguments.get("paths", [])

        if not paths:
            # Sin paths específicos, agrega todos los cambios
            cmd = ["add", "--all"]
            description = "todos los archivos"
        else:
            cmd = ["add", "--"] + paths
            description = ", ".join(paths)

        result = self._run_git(cmd)

        if result is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo agregar al staging: {description}",
            )

        # Mostramos el status después de add para confirmar qué quedó staged
        status_result = self._run_git(["diff", "--cached", "--stat"])
        status_info = f"\n\nStaging area después de add:\n{status_result}" if status_result else ""

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"Archivos agregados al staging: {description}{status_info}",
        )

    def _commit(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Crea un commit con los archivos en staging area."""
        message = arguments["message"].strip()

        # Verificar que hay algo staged antes de intentar el commit
        staged_output = self._run_git(["diff", "--cached", "--name-only"])
        if not staged_output or not staged_output.strip():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    "No hay archivos en el staging area. "
                    "Usa la operación 'add' primero para agregar archivos al staging."
                ),
            )

        output = self._run_git(["commit", "-m", message])

        if output is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="El commit falló. Revisa la configuración de Git (user.email, user.name).",
            )

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _branch(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Lista ramas existentes o crea una nueva."""
        branch_name = arguments.get("branch")

        if branch_name:
            # Crear rama nueva
            output = self._run_git(["branch", branch_name])
            if output is None:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message=f"No se pudo crear la rama '{branch_name}'.",
                )
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Rama '{branch_name}' creada exitosamente.",
            )
        else:
            # Listar ramas
            output = self._run_git(["branch", "-a", "--color=never"])
            if output is None:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message="No se pudo listar las ramas.",
                )
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Ramas disponibles:\n{output}",
            )

    def _checkout(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Cambia de rama o restaura archivos."""
        branch_name = arguments.get("branch")
        create = arguments.get("create_branch", False)
        paths = arguments.get("paths", [])

        if paths and not branch_name:
            # Restaurar archivos específicos a HEAD
            cmd = ["checkout", "--"] + paths
            output = self._run_git(cmd)
            if output is None:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message=f"No se pudieron restaurar los archivos: {', '.join(paths)}",
                )
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Archivos restaurados a HEAD: {', '.join(paths)}",
            )

        if not branch_name:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="Especifica 'branch' para checkout.",
            )

        cmd = ["checkout"]
        if create:
            cmd.append("-b")
        cmd.append(branch_name)

        output = self._run_git(cmd)

        if output is None:
            action = "crear y cambiar a" if create else "cambiar a"
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"No se pudo {action} la rama '{branch_name}'. "
                    "Verifica que no hay cambios sin guardar."
                ),
            )

        action = "creada y" if create else ""
        return ToolResult.success(
            tool_call_id=call_id,
            output=f"Rama '{branch_name}' {action} activa.\n{output}".strip(),
        )

    def _push(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Envía commits al repositorio remoto."""
        remote = arguments.get("remote", "origin")

        # Obtenemos la rama actual para el push
        current_branch = self._run_git(["rev-parse", "--abbrev-ref", "HEAD"])
        if not current_branch:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="No se pudo determinar la rama actual.",
            )
        current_branch = current_branch.strip()

        output = self._run_git(
            ["push", remote, current_branch],
            timeout=_GIT_NETWORK_TIMEOUT,
        )

        if output is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"No se pudo hacer push a '{remote}/{current_branch}'. "
                    "Verifica la conexión y los permisos del remoto."
                ),
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"Push exitoso a {remote}/{current_branch}:\n{output}",
        )

    def _pull(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Obtiene y fusiona cambios del repositorio remoto."""
        remote = arguments.get("remote", "origin")

        output = self._run_git(
            ["pull", remote],
            timeout=_GIT_NETWORK_TIMEOUT,
        )

        if output is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"No se pudo hacer pull desde '{remote}'. "
                    "Verifica la conexión y que no hay conflictos sin resolver."
                ),
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"Pull desde {remote}:\n{output}",
        )

    def _stash(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Guarda o recupera cambios temporales en el stash."""
        pop = arguments.get("pop_stash", False)

        if pop:
            output = self._run_git(["stash", "pop"])
            if output is None:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message="No hay stash para recuperar o hubo un conflicto.",
                )
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Stash recuperado:\n{output}",
            )
        else:
            msg = arguments.get("stash_message", "")
            cmd = ["stash", "push"]
            if msg:
                cmd.extend(["-m", msg])

            output = self._run_git(cmd)
            if output is None:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message="No se pudo guardar el stash. ¿Hay cambios sin guardar?",
                )
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Cambios guardados en stash:\n{output}",
            )

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    def _run_git(
        self,
        args: list[str],
        timeout: float = TOOL_DEFAULT_TIMEOUT_SECONDS,
    ) -> str | None:
        """
        Ejecuta un comando git y devuelve el output combinado.

        Devuelve None si el comando falla (exit code != 0).
        Devuelve string vacío si el comando tuvo éxito sin output.

        No lanza excepciones — las encapsula en None para que los
        callers puedan manejarlas uniformemente.
        """
        try:
            result = subprocess.run(
                ["git"] + args,
                capture_output=True,
                text=True,
                timeout=timeout,
                start_new_session=True if sys.platform != "win32" else False,
            )

            if result.returncode != 0:
                # Loggeamos stderr de git para debug sin exponerlo al LLM directamente
                if result.stderr.strip():
                    logger.debug(
                        "comando git falló",
                        args=args,
                        stderr=result.stderr.strip()[:300],
                        exit_code=result.returncode,
                        task_id=self._task_id,
                    )
                # Devolvemos None para indicar fallo al caller
                # El caller construye un mensaje de error más amigable
                return None

            # Combinamos stdout y stderr para operaciones como push/pull
            # que escriben info en stderr aunque sean exitosas
            output = result.stdout
            if result.stderr.strip():
                output = output + result.stderr if output else result.stderr

            return output

        except subprocess.TimeoutExpired as exc:
            raise ToolTimeoutError(
                tool_name=self.name,
                timeout_seconds=timeout,
            ) from exc

        except FileNotFoundError:
            # Git no está instalado
            logger.error(
                "git no encontrado en el sistema",
                task_id=self._task_id,
            )
            return None

        except OSError as exc:
            logger.error(
                "error del sistema al ejecutar git",
                error=str(exc),
                task_id=self._task_id,
            )
            return None

    def _is_git_repo(self) -> bool:
        """
        Verifica que el directorio actual es un repositorio Git.

        Más eficiente que correr `git status` completo solo para verificar.
        `git rev-parse --git-dir` retorna exit 0 si es un repo, 128 si no.
        """
        try:
            result = subprocess.run(
                ["git", "rev-parse", "--git-dir"],
                capture_output=True,
                timeout=5.0,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
            return False