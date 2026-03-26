"""
FileTool — Operaciones sobre el sistema de archivos.

Permite al agente interactuar con archivos del proyecto:
  - Leer archivos completos o por rangos de líneas
  - Escribir o sobreescribir archivos
  - Aplicar patches precisos (reemplazar líneas específicas)
  - Listar directorios con filtros
  - Verificar si un archivo existe y sus metadatos

¿POR QUÉ UNA SOLA TOOL Y NO VARIAS (file_read, file_write, etc.)?
  El LLM necesita aprender a usar las tools. Menos tools = menos
  contexto en el prompt = más tokens disponibles para razonar.
  Una tool con sub-operaciones es más eficiente que cinco tools separadas.
  El parámetro `operation` actúa como discriminador.

OPERACIONES DISPONIBLES:
  read         → lee el contenido completo o por rango de líneas
  write        → escribe contenido (sobreescribe si existe)
  append       → agrega contenido al final del archivo
  patch        → reemplaza líneas específicas (más preciso que write completo)
  delete       → elimina un archivo
  exists       → verifica si existe y devuelve metadatos
  list         → lista archivos de un directorio con filtros

SEGURIDAD:
  Operaciones destructivas (write, delete) tienen verificaciones adicionales:
  - No permite escribir fuera del directorio de trabajo actual
  - No permite eliminar directorios completos (solo archivos)
  - No permite sobreescribir archivos binarios accidentalmente

INTEGRACIÓN CON FileRef:
  Las operaciones de lectura devuelven información compatible con FileRef
  para que el agente pueda trackear cambios en archivos entre iteraciones.
"""

from __future__ import annotations

import fnmatch
import os
from pathlib import Path
from typing import Any

from hiperforge.core.constants import TOOL_MAX_FILE_SIZE_BYTES, TOOL_MAX_OUTPUT_CHARS
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.domain.value_objects.file_ref import FileRef
from hiperforge.tools.base import BaseTool, register_tool

logger = get_logger(__name__)

# Extensiones binarias conocidas — nunca intentar leer como texto
_BINARY_EXTENSIONS: frozenset[str] = frozenset({
    ".pyc", ".pyo", ".so", ".dylib", ".dll", ".exe", ".bin",
    ".jpg", ".jpeg", ".png", ".gif", ".webp", ".ico", ".bmp",
    ".mp3", ".mp4", ".wav", ".ogg", ".avi", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".ttf", ".otf", ".woff", ".woff2",
    ".db", ".sqlite", ".sqlite3",
})

# Encoding por defecto para lectura/escritura de archivos
_DEFAULT_ENCODING = "utf-8"

# Número máximo de archivos a listar en una operación list
_MAX_LIST_RESULTS = 200


@register_tool
class FileTool(BaseTool):
    """
    Tool para operaciones completas sobre el sistema de archivos.
    """

    @property
    def name(self) -> str:
        return "file"

    @property
    def description(self) -> str:
        return "Lee, escribe, modifica y lista archivos del proyecto"

    # ------------------------------------------------------------------
    # Schema para el LLM
    # ------------------------------------------------------------------

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Realiza operaciones sobre archivos del proyecto. "
                "Operaciones disponibles: read, write, append, patch, delete, exists, list. "
                "Usa 'read' solo para archivos, no para directorios. "
                "Usa 'list' para directorios o para descubrir rutas cuando no conoces la ubicación exacta. "
                "Si ya conoces el archivo objetivo, evita listar el proyecto completo. "
                "Prefiere read parcial con start_line/end_line para archivos largos. "
                "Usa 'write' para crear o sobreescribir un archivo completo. "
                "Usa 'patch' para modificar líneas específicas sin reescribir todo. "
                "No releas completo un archivo que acabas de escribir salvo que sea imprescindible."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["read", "write", "append", "patch", "delete", "exists", "list"],
                        "description": "Operación a realizar sobre el archivo.",
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Ruta al archivo o directorio. "
                            "Puede ser absoluta o relativa al directorio de trabajo actual."
                        ),
                    },
                    "content": {
                        "type": "string",
                        "description": (
                            "Contenido para operaciones write, append y patch. "
                            "Para patch: el texto NUEVO que reemplazará al texto antiguo."
                        ),
                    },
                    "start_line": {
                        "type": "integer",
                        "description": (
                            "Línea de inicio para read parcial o patch (1-indexed). "
                            "Para read: primera línea a incluir en el output. "
                            "Para patch: primera línea a reemplazar."
                        ),
                    },
                    "end_line": {
                        "type": "integer",
                        "description": (
                            "Línea de fin para read parcial o patch (1-indexed, inclusivo). "
                            "Para read: última línea a incluir. "
                            "Para patch: última línea a reemplazar."
                        ),
                    },
                    "pattern": {
                        "type": "string",
                        "description": (
                            "Patrón glob para filtrar archivos en operación list. "
                            "Ejemplos: '*.py', '*.ts', 'test_*.py', '**/*.json'"
                        ),
                    },
                    "encoding": {
                        "type": "string",
                        "description": (
                            f"Encoding del archivo. Default: {_DEFAULT_ENCODING}. "
                            "Otros comunes: latin-1, cp1252."
                        ),
                    },
                },
                "required": ["operation", "path"],
            },
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validaciones específicas de FileTool por operación."""
        errors = super().validate_arguments(arguments)

        operation = arguments.get("operation", "")
        valid_ops = {"read", "write", "append", "patch", "delete", "exists", "list"}

        if operation not in valid_ops:
            errors.append(
                f"Operación '{operation}' inválida. "
                f"Válidas: {', '.join(sorted(valid_ops))}"
            )
            return errors  # Sin operación válida, el resto de validaciones no aplica

        # Operaciones que requieren content
        if operation in {"write", "append"} and not arguments.get("content"):
            errors.append(f"La operación '{operation}' requiere el campo 'content'")

        # patch requiere content + rango de líneas
        if operation == "patch":
            if not arguments.get("content"):
                errors.append("La operación 'patch' requiere el campo 'content'")
            if arguments.get("start_line") is None:
                errors.append("La operación 'patch' requiere 'start_line'")
            if arguments.get("end_line") is None:
                errors.append("La operación 'patch' requiere 'end_line'")

        # Validar rango de líneas si se especifica
        start = arguments.get("start_line")
        end = arguments.get("end_line")
        if start is not None and end is not None:
            if not isinstance(start, int) or start < 1:
                errors.append("start_line debe ser un entero >= 1")
            if not isinstance(end, int) or end < 1:
                errors.append("end_line debe ser un entero >= 1")
            if isinstance(start, int) and isinstance(end, int) and start > end:
                errors.append(f"start_line ({start}) no puede ser mayor que end_line ({end})")

        return errors

    def is_safe_to_run(self, arguments: dict[str, Any]) -> bool:
        """
        Verifica que las operaciones destructivas no salgan del proyecto.

        Para write, append, patch y delete: verifica que la ruta
        no intente escapar del directorio de trabajo actual.
        Esto previene que el agente modifique archivos del sistema.
        """
        operation = arguments.get("operation", "")
        path_str = arguments.get("path", "")

        if operation not in {"write", "append", "patch", "delete"}:
            return True  # read, exists, list son siempre seguras

        if not path_str:
            return True  # path vacío fallará en validate_arguments

        try:
            target = Path(path_str).resolve()
            cwd = Path.cwd().resolve()

            # La ruta debe estar dentro del directorio de trabajo
            target.relative_to(cwd)
            return True

        except ValueError:
            # relative_to lanza ValueError si target no está bajo cwd
            logger.warning(
                "operación de escritura bloqueada — ruta fuera del directorio de trabajo",
                operation=operation,
                path=path_str,
                cwd=str(Path.cwd()),
                task_id=self._task_id,
            )
            return False

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Despacha a la operación correspondiente."""
        operation = arguments["operation"]
        path_str = arguments["path"]
        call_id = self._get_active_tool_call_id()

        # Resolvemos la ruta (absoluta o relativa al cwd)
        path = Path(path_str)
        if not path.is_absolute():
            path = Path.cwd() / path

        dispatch = {
            "read":   self._read,
            "write":  self._write,
            "append": self._append,
            "patch":  self._patch,
            "delete": self._delete,
            "exists": self._exists,
            "list":   self._list,
        }

        handler = dispatch[operation]
        return handler(path=path, arguments=arguments, call_id=call_id)

    # ------------------------------------------------------------------
    # Operaciones concretas
    # ------------------------------------------------------------------

    def _read(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Lee el contenido de un archivo.

        Soporta lectura completa o por rango de líneas.
        Si el archivo es muy grande, lee solo hasta TOOL_MAX_FILE_SIZE_BYTES.
        """
        if not path.exists():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Archivo no encontrado: {path}",
            )

        if not path.is_file():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"La ruta no es un archivo: {path}",
            )

        # Verificar si es binario
        if path.suffix.lower() in _BINARY_EXTENSIONS:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"'{path.name}' es un archivo binario y no puede leerse como texto. "
                    f"Si necesitas su contenido, usa una herramienta específica para "
                    f"ese tipo de archivo."
                ),
            )

        # Verificar tamaño antes de leer
        size_bytes = path.stat().st_size
        if size_bytes > TOOL_MAX_FILE_SIZE_BYTES:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"'{path.name}' es demasiado grande ({size_bytes / 1024:.1f} KB). "
                    f"Máximo: {TOOL_MAX_FILE_SIZE_BYTES / 1024:.0f} KB. "
                    f"Usa start_line y end_line para leer secciones específicas."
                ),
            )

        encoding = arguments.get("encoding", _DEFAULT_ENCODING)

        try:
            content = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"No se pudo leer '{path.name}' con encoding '{encoding}'. "
                    f"Intenta con encoding='latin-1' o encoding='cp1252'."
                ),
            )

        # Leer por rango de líneas si se especificó
        start_line = arguments.get("start_line")
        end_line = arguments.get("end_line")

        if start_line is not None or end_line is not None:
            lines = content.splitlines(keepends=True)
            total_lines = len(lines)

            start = (start_line or 1) - 1  # convertimos a 0-indexed
            end = end_line or total_lines    # end es inclusivo

            # Clampear al rango válido
            start = max(0, min(start, total_lines))
            end = max(start, min(end, total_lines))

            selected_lines = lines[start:end]
            content = "".join(selected_lines)

            # Incluimos información del rango en el output para contexto del LLM
            header = (
                f"--- {path.name} "
                f"(líneas {start + 1}-{end} de {total_lines}) ---\n"
            )
            output = header + content
        else:
            total_lines = content.count("\n") + 1
            header = f"--- {path.name} ({total_lines} líneas) ---\n"
            output = header + content

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _write(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Escribe contenido en un archivo, creando directorios intermedios si es necesario.

        Si el archivo ya existe, lo sobreescribe completamente.
        """
        content = arguments["content"]
        encoding = arguments.get("encoding", _DEFAULT_ENCODING)

        # Crear directorios intermedios automáticamente
        path.parent.mkdir(parents=True, exist_ok=True)

        existed = path.exists()

        try:
            path.write_text(content, encoding=encoding)
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo escribir '{path}': {exc}",
            )

        lines = content.count("\n") + 1
        action = "actualizado" if existed else "creado"
        output = (
            f"Archivo {action}: {path}\n"
            f"{lines} líneas, {len(content)} caracteres"
        )

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _append(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """Agrega contenido al final del archivo sin sobreescribir."""
        content = arguments["content"]
        encoding = arguments.get("encoding", _DEFAULT_ENCODING)

        path.parent.mkdir(parents=True, exist_ok=True)

        try:
            with path.open("a", encoding=encoding) as f:
                f.write(content)
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo agregar a '{path}': {exc}",
            )

        output = (
            f"Contenido agregado a: {path}\n"
            f"{len(content)} caracteres añadidos"
        )

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _patch(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Reemplaza un rango de líneas específico en un archivo.

        Más preciso que write completo — solo modifica las líneas indicadas,
        dejando el resto del archivo intacto.

        Ejemplo: reemplazar líneas 10-15 con nuevo contenido.
        """
        if not path.exists():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Archivo no encontrado para patch: {path}",
            )

        new_content = arguments["content"]
        start_line = arguments["start_line"]  # 1-indexed
        end_line = arguments["end_line"]       # 1-indexed, inclusivo
        encoding = arguments.get("encoding", _DEFAULT_ENCODING)

        try:
            original = path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo leer '{path.name}' para aplicar patch.",
            )

        lines = original.splitlines(keepends=True)
        total_lines = len(lines)

        # Validar rango contra el archivo real
        if start_line > total_lines:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"start_line {start_line} excede el total de líneas "
                    f"del archivo ({total_lines})."
                ),
            )

        # Convertimos a 0-indexed
        start_idx = start_line - 1
        end_idx = min(end_line, total_lines)  # clampeamos al final del archivo

        # Reemplazamos las líneas del rango con el nuevo contenido
        # Aseguramos que el nuevo contenido termine en newline si el original lo hacía
        new_lines = new_content.splitlines(keepends=True)
        if new_lines and not new_lines[-1].endswith("\n"):
            new_lines[-1] += "\n"

        patched_lines = lines[:start_idx] + new_lines + lines[end_idx:]
        patched_content = "".join(patched_lines)

        try:
            path.write_text(patched_content, encoding=encoding)
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo escribir patch en '{path}': {exc}",
            )

        replaced_count = end_idx - start_idx
        new_count = len(new_lines)
        output = (
            f"Patch aplicado en: {path}\n"
            f"Líneas {start_line}-{end_line} reemplazadas "
            f"({replaced_count} → {new_count} líneas)"
        )

        return ToolResult.success(tool_call_id=call_id, output=output)

    def _delete(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Elimina un archivo.

        No permite eliminar directorios — solo archivos individuales.
        Para directorios usar ShellTool con rm.
        """
        if not path.exists():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Archivo no encontrado: {path}",
            )

        if path.is_dir():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"'{path}' es un directorio. FileTool solo elimina archivos. "
                    f"Para eliminar un directorio usa ShellTool con: rm -rf {path}"
                ),
            )

        try:
            path.unlink()
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo eliminar '{path}': {exc}",
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"Archivo eliminado: {path}",
        )

    def _exists(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Verifica si existe un archivo y devuelve sus metadatos.

        Devuelve información útil incluso cuando NO existe,
        para que el LLM pueda tomar decisiones informadas.
        """
        if not path.exists():
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"No existe: {path}",
            )

        stat = path.stat()
        kind = "directorio" if path.is_dir() else "archivo"
        size_kb = stat.st_size / 1024

        output_lines = [
            f"Existe: {path}",
            f"Tipo: {kind}",
            f"Tamaño: {size_kb:.1f} KB",
        ]

        if path.is_file():
            # Intentamos contar líneas para archivos de texto
            if path.suffix.lower() not in _BINARY_EXTENSIONS:
                try:
                    line_count = sum(1 for _ in path.open(encoding=_DEFAULT_ENCODING))
                    output_lines.append(f"Líneas: {line_count}")
                except (OSError, UnicodeDecodeError):
                    pass

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(output_lines),
        )

    def _list(self, path: Path, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Lista archivos de un directorio con filtros opcionales.

        Soporta patrones glob para filtrar por extensión o nombre.
        Muestra la estructura en formato árbol para facilitar la navegación.
        """
        if not path.exists():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Directorio no encontrado: {path}",
            )

        if not path.is_dir():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"'{path}' no es un directorio",
            )

        pattern = arguments.get("pattern", "*")

        # Recolectamos archivos aplicando el patrón
        try:
            if "**" in pattern:
                # Glob recursivo
                matches = sorted(path.glob(pattern))
            else:
                # Glob no recursivo (solo el directorio especificado)
                matches = sorted(path.glob(pattern))
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Error listando directorio: {exc}",
            )

        # Limitamos resultados para no saturar el context window del LLM
        truncated = len(matches) > _MAX_LIST_RESULTS
        matches = matches[:_MAX_LIST_RESULTS]

        if not matches:
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"No se encontraron archivos en '{path}' con patrón '{pattern}'",
            )

        # Formateamos la lista de forma legible
        lines = [f"Directorio: {path}  (patrón: {pattern})\n"]

        for match in matches:
            # Ruta relativa al directorio listado para mayor claridad
            try:
                relative = match.relative_to(path)
            except ValueError:
                relative = match

            if match.is_dir():
                lines.append(f"  {relative}/")
            else:
                size_kb = match.stat().st_size / 1024
                lines.append(f"  {relative}  ({size_kb:.1f} KB)")

        if truncated:
            lines.append(
                f"\n[Mostrando primeros {_MAX_LIST_RESULTS} resultados. "
                f"Usa un patrón más específico para filtrar.]"
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(lines),
        )
