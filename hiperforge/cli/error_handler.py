"""
ErrorHandler — Manejo centralizado de errores para la CLI de HiperForge.

La CLI es la frontera entre el dominio técnico del sistema y el usuario.
Cuando algo falla, el error_handler es el responsable de traducir las
excepciones técnicas del dominio en mensajes claros, accionables y
apropiados para el contexto en que ocurrieron.

════════════════════════════════════════════════════════════
FILOSOFÍA DE MANEJO DE ERRORES EN CLIs DE PRODUCCIÓN
════════════════════════════════════════════════════════════

  Un error bien manejado tiene tres partes:
    1. QUÉ ocurrió — descripción clara del problema.
    2. POR QUÉ ocurrió — contexto técnico si es útil para el usuario.
    3. QUÉ HACER — instrucciones accionables para resolverlo.

  Un error mal manejado muestra el stack trace completo o,
  peor aún, falla silenciosamente.

  Principios aplicados en este módulo:
    - Cada tipo de excepción del dominio tiene su handler específico.
    - Los errores se formatean con Rich para máxima legibilidad.
    - Los exit codes siguen la convención POSIX (0=éxito, 1=error general,
      2=uso incorrecto, 127=comando no encontrado, 130=interrupción).
    - Los mensajes distinguen entre errores del usuario (uso incorrecto)
      y errores del sistema (bugs o problemas de infraestructura).
    - En modo debug se muestra el traceback completo.

════════════════════════════════════════════════════════════
JERARQUÍA DE ERRORES Y SU MAPEO A MENSAJES CLI
════════════════════════════════════════════════════════════

  HiperForgeError
  ├── DomainError
  │   ├── EntityNotFound    → "No se encontró X con id Y"
  │   ├── DuplicateEntity   → "Ya existe un X con ese nombre"
  │   └── InvalidStatus...  → "Operación no permitida en el estado actual"
  ├── PlanError
  │   ├── EmptyPlanError    → "El agente no pudo generar un plan válido"
  │   └── InvalidPlanError  → "El plan generado tiene formato inválido"
  ├── LLMError
  │   ├── LLMConnectionError → "No se pudo conectar al proveedor LLM"
  │   ├── LLMRateLimitError  → "Límite de rate del proveedor alcanzado"
  │   └── LLMResponseError   → "Respuesta inesperada del LLM"
  ├── ToolError
  │   ├── ToolNotFound      → "Tool desconocida solicitada por el agente"
  │   └── ToolTimeoutError  → "Una tool excedió su timeout"
  └── StorageError
      ├── StorageReadError  → "No se pudo leer archivo de datos"
      ├── StorageWriteError → "No se pudo escribir archivo de datos"
      └── StorageCorrupted  → "Datos corruptos detectados"

  ValueError / PermissionError → errores de validación/permisos del usuario
  KeyboardInterrupt             → salida limpia sin error
  Exception genérica            → error inesperado con contexto para reporte

════════════════════════════════════════════════════════════
EXIT CODES
════════════════════════════════════════════════════════════

  Los exit codes son parte del contrato público de la CLI.
  Scripts y pipelines CI/CD los usan para detectar fallos.

  EXIT_SUCCESS     = 0   Operación completada exitosamente
  EXIT_ERROR       = 1   Error general durante la ejecución
  EXIT_USAGE       = 2   Uso incorrecto del comando (argumentos inválidos)
  EXIT_NOT_FOUND   = 3   Entidad no encontrada (EntityNotFound)
  EXIT_DUPLICATE   = 4   Entidad duplicada (DuplicateEntity)
  EXIT_LLM_ERROR   = 5   Error de comunicación con el LLM
  EXIT_STORAGE     = 6   Error de lectura/escritura de datos
  EXIT_INTERRUPTED = 130 Proceso interrumpido por el usuario (Ctrl+C)

USO:
  # En cualquier comando de la CLI:
  with ErrorHandler.context("run"):
      output = container.run_task.execute(input_data)

  # O directamente:
  handler = ErrorHandler(debug=settings.debug)
  handler.handle(exception)
"""

from __future__ import annotations

import sys
import traceback
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Generator, NoReturn

from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from rich.traceback import Traceback

from hiperforge.core.logging import get_logger

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Importaciones tardías para evitar circular imports con el dominio
# ---------------------------------------------------------------------------
# Las excepciones del dominio se importan aquí para que error_handler.py
# no dependa del dominio en tiempo de importación del módulo.
# Esto permite que los tests carguen error_handler.py sin instanciar el dominio completo.


# ---------------------------------------------------------------------------
# Exit codes — contrato público de la CLI
# ---------------------------------------------------------------------------

EXIT_SUCCESS     = 0    # Operación completada exitosamente
EXIT_ERROR       = 1    # Error general durante la ejecución
EXIT_USAGE       = 2    # Uso incorrecto (argumentos inválidos, validación)
EXIT_NOT_FOUND   = 3    # Entidad no encontrada (EntityNotFound)
EXIT_DUPLICATE   = 4    # Entidad duplicada (DuplicateEntity)
EXIT_LLM_ERROR   = 5    # Error de comunicación con el LLM
EXIT_STORAGE     = 6    # Error de lectura/escritura de datos
EXIT_INTERRUPTED = 130  # Proceso interrumpido por el usuario (Ctrl+C / SIGTERM)


# ---------------------------------------------------------------------------
# Resultado del manejo de error
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ErrorResult:
    """
    Resultado de manejar una excepción.

    Contiene el exit code apropiado y si el error fue manejado
    de forma esperada o fue inesperado.

    Atributos:
        exit_code:   Código de salida POSIX para el proceso.
        was_handled: True si la excepción tenía un handler específico.
                     False si fue manejada por el handler genérico de fallback.
        user_facing: True si el mensaje fue mostrado al usuario.
    """
    exit_code: int
    was_handled: bool
    user_facing: bool


# ---------------------------------------------------------------------------
# ErrorHandler
# ---------------------------------------------------------------------------

class ErrorHandler:
    """
    Manejador centralizado de excepciones para la CLI.

    Traduce excepciones técnicas del dominio en mensajes comprensibles
    para el usuario y determina el exit code apropiado del proceso.

    Parámetros:
        debug:   Si True, muestra el traceback completo de Rich para debugging.
                 Activado con --debug o la variable HIPERFORGE_DEBUG=true.
        console: Consola Rich para output. None = usar stderr para errores.
    """

    def __init__(
        self,
        debug: bool = False,
        console: Console | None = None,
    ) -> None:
        self._debug = debug
        # Los errores van a stderr para no contaminar el output de la CLI
        # que puede estar siendo capturado por scripts o pipelines
        self._console = console or Console(stderr=True, highlight=True)

    # ------------------------------------------------------------------
    # Context manager — captura y maneja excepciones automáticamente
    # ------------------------------------------------------------------

    @contextmanager
    def context(self, operation: str) -> Generator[None, None, None]:
        """
        Context manager que captura y maneja excepciones de una operación.

        En caso de error, muestra el mensaje apropiado y termina el proceso
        con el exit code correcto.

        Parámetros:
            operation: Nombre de la operación para mensajes de error contextualizados.
                       Ejemplo: "run", "workspace create", "config set"

        Uso:
            with error_handler.context("workspace create"):
                output = container.create_workspace.execute(input_data)

        Raises:
            SystemExit: Siempre que ocurra una excepción — con el exit code apropiado.
        """
        try:
            yield
        except KeyboardInterrupt:
            self._handle_keyboard_interrupt()
        except Exception as exc:
            result = self.handle(exc, operation=operation)
            sys.exit(result.exit_code)

    # ------------------------------------------------------------------
    # Manejo directo de excepciones
    # ------------------------------------------------------------------

    def handle(
        self,
        exc: Exception,
        *,
        operation: str = "operación",
    ) -> ErrorResult:
        """
        Maneja una excepción mostrando el mensaje apropiado al usuario.

        Determina el handler específico según el tipo de excepción y
        devuelve el ErrorResult con el exit code correcto.

        Parámetros:
            exc:       La excepción a manejar.
            operation: Nombre de la operación para contextualizar el mensaje.

        Returns:
            ErrorResult con el exit code y metadatos del manejo.
        """
        # Importamos las excepciones del dominio aquí para evitar circular imports
        from hiperforge.domain.exceptions import (
            DuplicateEntity,
            EmptyPlanError,
            EntityNotFound,
            HiperForgeError,
            InvalidPlanError,
            InvalidStatusTransition,
            LLMConnectionError,
            LLMRateLimitError,
            LLMResponseError,
            PlanError,
            StorageCorruptedError,
            StorageError,
            StorageReadError,
            StorageWriteError,
            ToolTimeoutError,
        )

        # Logging siempre — independientemente de si se muestra al usuario
        logger.error(
            "error en CLI",
            operation=operation,
            error_type=type(exc).__name__,
            error=str(exc),
        )

        # Despachar al handler específico según el tipo
        if isinstance(exc, EntityNotFound):
            return self._handle_entity_not_found(exc)

        if isinstance(exc, DuplicateEntity):
            return self._handle_duplicate_entity(exc)

        if isinstance(exc, InvalidStatusTransition):
            return self._handle_invalid_transition(exc)

        if isinstance(exc, EmptyPlanError):
            return self._handle_empty_plan(exc)

        if isinstance(exc, InvalidPlanError):
            return self._handle_invalid_plan(exc)

        if isinstance(exc, PlanError):
            return self._handle_plan_error(exc)

        if isinstance(exc, LLMConnectionError):
            return self._handle_llm_connection_error(exc)

        if isinstance(exc, LLMRateLimitError):
            return self._handle_llm_rate_limit(exc)

        if isinstance(exc, LLMResponseError):
            return self._handle_llm_response_error(exc)

        if isinstance(exc, ToolTimeoutError):
            return self._handle_tool_timeout(exc)

        if isinstance(exc, StorageCorruptedError):
            return self._handle_storage_corrupted(exc)

        if isinstance(exc, StorageReadError):
            return self._handle_storage_read_error(exc)

        if isinstance(exc, StorageWriteError):
            return self._handle_storage_write_error(exc)

        if isinstance(exc, StorageError):
            return self._handle_storage_error(exc)

        if isinstance(exc, HiperForgeError):
            return self._handle_hiperforge_error(exc, operation)

        if isinstance(exc, ValueError):
            return self._handle_value_error(exc, operation)

        if isinstance(exc, PermissionError):
            return self._handle_permission_error(exc, operation)

        # Excepción no reconocida — handler de fallback
        return self._handle_unexpected_error(exc, operation)

    # ------------------------------------------------------------------
    # Handlers específicos por tipo de excepción
    # ------------------------------------------------------------------

    def _handle_entity_not_found(self, exc) -> ErrorResult:
        """Entidad no encontrada — EntityNotFound."""
        entity_type = exc.context.get("entity_type", "Entidad")
        entity_id = exc.context.get("entity_id", "desconocido")

        self._print_error(
            title=f"{entity_type} no encontrado",
            message=str(exc),
            hint=(
                f"Verifica que el ID '{entity_id}' es correcto.\n"
                f"Usa [bold]hiperforge {entity_type.lower()} list[/bold] "
                f"para ver los {entity_type.lower()}s disponibles."
            ),
        )
        return ErrorResult(exit_code=EXIT_NOT_FOUND, was_handled=True, user_facing=True)

    def _handle_duplicate_entity(self, exc) -> ErrorResult:
        """Entidad duplicada — DuplicateEntity."""
        entity_type = exc.context.get("entity_type", "Entidad")
        identifier = exc.context.get("identifier", "desconocido")

        self._print_error(
            title=f"{entity_type} ya existe",
            message=str(exc),
            hint=(
                f"Ya existe un {entity_type.lower()} con el nombre '{identifier}'.\n"
                f"Elige un nombre diferente o usa el existente."
            ),
        )
        return ErrorResult(exit_code=EXIT_DUPLICATE, was_handled=True, user_facing=True)

    def _handle_invalid_transition(self, exc) -> ErrorResult:
        """Transición de estado inválida."""
        entity = exc.context.get("entity", "entidad")
        from_status = exc.context.get("from_status", "?")
        to_status = exc.context.get("to_status", "?")

        self._print_error(
            title="Operación no permitida",
            message=str(exc),
            hint=(
                f"La entidad '{entity}' está en estado '{from_status}' "
                f"y no puede transicionar a '{to_status}'.\n"
                f"Verifica el estado actual antes de ejecutar esta operación."
            ),
        )
        return ErrorResult(exit_code=EXIT_USAGE, was_handled=True, user_facing=True)

    def _handle_empty_plan(self, exc) -> ErrorResult:
        """El LLM devolvió un plan sin subtasks."""
        self._print_error(
            title="El agente no pudo generar un plan",
            message="El LLM devolvió un plan vacío sin ningún paso de ejecución.",
            hint=(
                "Esto puede ocurrir cuando:\n"
                "  • El prompt es demasiado vago o ambiguo.\n"
                "  • El contexto del proyecto no es claro.\n\n"
                "Sugerencias:\n"
                "  • Sé más específico en tu instrucción.\n"
                "  • Divide la tarea en pasos más pequeños.\n"
                "  • Verifica que el modelo LLM configurado soporta esta complejidad."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_invalid_plan(self, exc) -> ErrorResult:
        """El LLM devolvió un plan con formato inválido."""
        reason = exc.context.get("reason", "formato inesperado")

        self._print_error(
            title="Plan con formato inválido",
            message=f"El agente generó un plan que no pudo ser interpretado: {reason}",
            hint=(
                "El LLM no siguió el formato de respuesta esperado.\n\n"
                "Sugerencias:\n"
                "  • Intenta de nuevo — puede ser un fallo transitorio.\n"
                "  • Verifica que el modelo configurado soporta JSON estructurado.\n"
                "  • Considera usar un modelo más capaz para tareas complejas."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_plan_error(self, exc) -> ErrorResult:
        """Error genérico de planificación."""
        self._print_error(
            title="Error durante la planificación",
            message=str(exc),
            hint=(
                "El agente no pudo completar la fase de planificación.\n"
                "Intenta de nuevo o usa una instrucción más simple."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_llm_connection_error(self, exc) -> ErrorResult:
        """No se pudo conectar al proveedor LLM."""
        provider = exc.context.get("provider", "desconocido")
        reason = exc.context.get("reason", "error de conexión")

        hints_by_provider = {
            "anthropic": (
                "Verifica que ANTHROPIC_API_KEY está configurada correctamente.\n"
                "Obtén una key en: https://console.anthropic.com/settings/keys"
            ),
            "openai": (
                "Verifica que OPENAI_API_KEY está configurada correctamente.\n"
                "Obtén una key en: https://platform.openai.com/api-keys"
            ),
            "groq": (
                "Verifica que GROQ_API_KEY está configurada correctamente.\n"
                "Obtén una key gratuita en: https://console.groq.com/keys"
            ),
            "ollama": (
                "Verifica que Ollama está corriendo: ollama serve\n"
                "Verifica la URL: HIPERFORGE_OLLAMA_BASE_URL (default: http://localhost:11434)\n"
                "Descarga Ollama en: https://ollama.ai"
            ),
        }

        provider_hint = hints_by_provider.get(
            provider.lower(),
            f"Verifica la configuración del proveedor '{provider}'."
        )

        self._print_error(
            title=f"No se pudo conectar a {provider}",
            message=reason,
            hint=(
                f"{provider_hint}\n\n"
                f"Comprueba tu conexión a internet y las variables de entorno."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_llm_rate_limit(self, exc) -> ErrorResult:
        """Rate limit del proveedor LLM alcanzado."""
        provider = exc.context.get("provider", "desconocido")
        retry_after = exc.context.get("retry_after_seconds")

        wait_hint = (
            f"Espera aproximadamente {retry_after:.0f} segundos antes de reintentar."
            if retry_after
            else "Espera unos minutos antes de reintentar."
        )

        self._print_error(
            title=f"Rate limit alcanzado en {provider}",
            message="El proveedor rechazó la petición por exceso de requests.",
            hint=(
                f"{wait_hint}\n\n"
                "Si el problema persiste:\n"
                "  • Verifica tu plan de suscripción con el proveedor.\n"
                "  • Considera usar un proveedor alternativo con --provider."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_llm_response_error(self, exc) -> ErrorResult:
        """El LLM respondió con contenido inesperado."""
        provider = exc.context.get("provider", "desconocido")
        reason = exc.context.get("reason", "respuesta inesperada")

        self._print_error(
            title=f"Respuesta inesperada de {provider}",
            message=reason,
            hint=(
                "El LLM devolvió una respuesta que no pudo ser interpretada.\n\n"
                "Sugerencias:\n"
                "  • Intenta de nuevo — puede ser un fallo transitorio.\n"
                "  • Verifica el estado del proveedor en su página de status.\n"
                "  • Si tienes --debug activo, revisa el log para más detalles."
            ),
        )
        return ErrorResult(exit_code=EXIT_LLM_ERROR, was_handled=True, user_facing=True)

    def _handle_tool_timeout(self, exc) -> ErrorResult:
        """Una tool excedió su timeout."""
        tool_name = exc.context.get("tool_name", "desconocida")
        timeout = exc.context.get("timeout_seconds", 0)

        self._print_error(
            title=f"Timeout en tool '{tool_name}'",
            message=f"La tool '{tool_name}' excedió el límite de {timeout}s.",
            hint=(
                "Para operaciones lentas (instalación de dependencias, compilación):\n"
                f"  • El agente reintentará automáticamente con timeout extendido.\n"
                f"  • Si el problema persiste, incrementa el timeout en la configuración:\n"
                f"    [bold]hiperforge config set agent.tool_timeout_seconds 120[/bold]"
            ),
        )
        return ErrorResult(exit_code=EXIT_ERROR, was_handled=True, user_facing=True)

    def _handle_storage_corrupted(self, exc) -> ErrorResult:
        """Datos corruptos detectados en disco."""
        path = exc.context.get("path", "desconocido")
        reason = exc.context.get("reason", "corrupción detectada")

        self._print_error(
            title="Datos corruptos detectados",
            message=f"El archivo '{path}' está dañado: {reason}",
            hint=(
                "Este es un error crítico que requiere atención manual.\n\n"
                "Opciones:\n"
                f"  1. Revisa el archivo manualmente: [bold]{path}[/bold]\n"
                f"  2. Si tienes backup, restáuralo desde ahí.\n"
                f"  3. Si no puedes recuperarlo, elimínalo y HiperForge\n"
                f"     lo recreará con valores por defecto.\n\n"
                f"Reporta este error en: https://github.com/hiperforge/hiperforge/issues"
            ),
            severity="critical",
        )
        return ErrorResult(exit_code=EXIT_STORAGE, was_handled=True, user_facing=True)

    def _handle_storage_read_error(self, exc) -> ErrorResult:
        """Error leyendo un archivo de datos."""
        path = exc.context.get("path", "desconocido")
        reason = exc.context.get("reason", "error de lectura")

        self._print_error(
            title="Error leyendo datos",
            message=f"No se pudo leer '{path}': {reason}",
            hint=(
                "Posibles causas:\n"
                "  • El archivo no existe o fue movido.\n"
                "  • Permisos insuficientes para leer el archivo.\n"
                "  • El disco tiene problemas de I/O.\n\n"
                f"Verifica los permisos: [bold]ls -la {path}[/bold]"
            ),
        )
        return ErrorResult(exit_code=EXIT_STORAGE, was_handled=True, user_facing=True)

    def _handle_storage_write_error(self, exc) -> ErrorResult:
        """Error escribiendo un archivo de datos."""
        path = exc.context.get("path", "desconocido")
        reason = exc.context.get("reason", "error de escritura")

        self._print_error(
            title="Error guardando datos",
            message=f"No se pudo escribir '{path}': {reason}",
            hint=(
                "Posibles causas:\n"
                "  • Disco lleno — verifica el espacio disponible.\n"
                "  • Permisos insuficientes para escribir en el directorio.\n"
                "  • El directorio padre no existe.\n\n"
                "Verifica espacio en disco: [bold]df -h ~/.hiperforge[/bold]"
            ),
        )
        return ErrorResult(exit_code=EXIT_STORAGE, was_handled=True, user_facing=True)

    def _handle_storage_error(self, exc) -> ErrorResult:
        """Error genérico de almacenamiento."""
        path = exc.context.get("path", "desconocido")

        self._print_error(
            title="Error de almacenamiento",
            message=str(exc),
            hint=(
                f"Problema con el archivo: '{path}'\n"
                "Verifica que el directorio ~/.hiperforge/ es accesible y tiene espacio."
            ),
        )
        return ErrorResult(exit_code=EXIT_STORAGE, was_handled=True, user_facing=True)

    def _handle_hiperforge_error(self, exc, operation: str) -> ErrorResult:
        """Error propio de HiperForge sin handler específico."""
        self._print_error(
            title=f"Error en {operation}",
            message=str(exc),
            hint=(
                "Si este error persiste, reporta el problema en:\n"
                "https://github.com/hiperforge/hiperforge/issues"
            ),
        )
        return ErrorResult(exit_code=EXIT_ERROR, was_handled=True, user_facing=True)

    def _handle_value_error(self, exc: ValueError, operation: str) -> ErrorResult:
        """Error de validación — uso incorrecto por el usuario."""
        self._print_error(
            title="Argumento inválido",
            message=str(exc),
            hint="Revisa los argumentos del comando. Usa --help para ver los disponibles.",
        )
        return ErrorResult(exit_code=EXIT_USAGE, was_handled=True, user_facing=True)

    def _handle_permission_error(self, exc: PermissionError, operation: str) -> ErrorResult:
        """Error de permisos — operación no permitida en el estado actual."""
        self._print_error(
            title="Operación no permitida",
            message=str(exc),
            hint="Verifica el estado actual de la entidad antes de ejecutar esta operación.",
        )
        return ErrorResult(exit_code=EXIT_USAGE, was_handled=True, user_facing=True)

    def _handle_unexpected_error(self, exc: Exception, operation: str) -> ErrorResult:
        """
        Handler de fallback para excepciones no reconocidas.

        Muestra un mensaje genérico y, en modo debug, el traceback completo.
        Estos errores indican potencialmente un bug en HiperForge.
        """
        self._print_error(
            title="Error inesperado",
            message=(
                f"Ocurrió un error inesperado durante '{operation}':\n"
                f"{type(exc).__name__}: {exc}"
            ),
            hint=(
                "Este error no debería ocurrir — puede ser un bug en HiperForge.\n\n"
                "Para obtener más información:\n"
                "  • Ejecuta el comando con [bold]--debug[/bold] para ver el traceback completo.\n"
                "  • Reporta el error en: https://github.com/hiperforge/hiperforge/issues\n"
                "    Incluye el comando que ejecutaste y el mensaje de error."
            ),
            severity="critical",
        )

        # En modo debug, mostramos el traceback completo de Rich
        if self._debug:
            self._console.print(Traceback(show_locals=True))

        return ErrorResult(exit_code=EXIT_ERROR, was_handled=False, user_facing=True)

    # ------------------------------------------------------------------
    # Manejo de interrupciones del usuario
    # ------------------------------------------------------------------

    def _handle_keyboard_interrupt(self) -> NoReturn:
        """
        Maneja Ctrl+C del usuario.

        No muestra error — es una salida limpia intencional.
        El SessionFlusher ya habrá persistido el estado antes de llegar aquí.
        """
        # Salto de línea para no dejar el cursor en medio de la línea del spinner
        self._console.print()
        self._console.print("[yellow]⊘[/yellow] Operación cancelada por el usuario.")
        sys.exit(EXIT_INTERRUPTED)

    # ------------------------------------------------------------------
    # Renderizado de mensajes de error
    # ------------------------------------------------------------------

    def _print_error(
        self,
        title: str,
        message: str,
        hint: str | None = None,
        severity: str = "error",
    ) -> None:
        """
        Imprime un mensaje de error formateado con Rich.

        Parámetros:
            title:    Título del error — qué ocurrió.
            message:  Descripción técnica del problema.
            hint:     Instrucciones accionables para resolverlo.
            severity: "error" (rojo) o "critical" (fondo rojo).
        """
        # Color del panel según severidad
        border_style = "red" if severity == "error" else "bold red"

        # Construimos el contenido del panel
        content_parts: list[str] = []

        # Mensaje principal
        content_parts.append(f"[white]{message}[/white]")

        # Hint con instrucciones accionables
        if hint:
            content_parts.append("")
            content_parts.append(f"[dim]{hint}[/dim]")

        # En modo debug, agregamos el contexto técnico del error
        if self._debug:
            content_parts.append("")
            content_parts.append(
                "[dim]─── Modo debug activo: usa --debug para ver el traceback completo ───[/dim]"
            )

        content = "\n".join(content_parts)

        # Icono según severidad
        icon = "✗" if severity == "error" else "⚠"

        self._console.print(
            Panel(
                content,
                title=f"[bold red]{icon} {title}[/bold red]",
                border_style=border_style,
                padding=(1, 2),
                expand=False,
            )
        )

    # ------------------------------------------------------------------
    # Factory method — construye desde el container
    # ------------------------------------------------------------------

    @classmethod
    def from_settings(cls, debug: bool = False) -> ErrorHandler:
        """
        Construye un ErrorHandler con la configuración del sistema.

        Factory method para uso en los comandos de la CLI:
            handler = ErrorHandler.from_settings(debug=settings.debug)
        """
        return cls(debug=debug)