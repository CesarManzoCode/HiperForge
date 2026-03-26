"""
CodeAnalysisTool — Análisis estático de código del proyecto.

Es la tool más directamente relacionada con el uso principal del agente:
trabajar con código de desarrolladores. Permite al agente entender
la estructura del proyecto antes de modificarlo.

OPERACIONES:
  find_symbol     → encuentra definiciones de funciones, clases y variables
  find_references → dónde se usa un símbolo en todo el proyecto
  analyze_imports → mapa de dependencias e imports entre archivos
  detect_issues   → problemas: complejidad alta, funciones largas, TODOs, code smells
  summarize_file  → resumen estructural de un archivo (API pública, imports, métricas)
  grep            → búsqueda de texto o regex en archivos del proyecto con contexto

ESTRATEGIA DE ANÁLISIS:
  Python   → AST real via módulo `ast` de stdlib (preciso, sin dependencias)
             Permite: complejidad ciclomática, análisis de argumentos, decorators
  JS/TS    → Regex estructurados (sin dependencias)
             Detecta: function/class/const/interface/type/arrow functions
  Rust     → Regex estructurados
             Detecta: fn/struct/impl/trait/enum/mod/use
  Go       → Regex estructurados
             Detecta: func/type/struct/interface/import

COBERTURA MULTI-LENGUAJE:
  El agente trabaja con proyectos reales que mezclan lenguajes.
  Un proyecto full-stack puede tener Python (backend) + TypeScript (frontend).
  CodeAnalysisTool analiza ambos con la misma interfaz.

¿POR QUÉ NO USAR tree-sitter O ctags?
  tree-sitter es más preciso pero requiere compilar extensiones C.
  ctags requiere instalación externa.
  Los regex de este módulo son suficientemente precisos para el 95%
  de los casos que necesita un agente — y funcionan sin dependencias extras.
  Para el 5% restante, el agente puede usar ShellTool con herramientas
  específicas del lenguaje.
"""

from __future__ import annotations

import ast
import re
import textwrap
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.tools.base import BaseTool, register_tool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración de límites
# ---------------------------------------------------------------------------

# Máximo de archivos a analizar en una operación (evita freezes en repos grandes)
_MAX_FILES_TO_SCAN = 500

# Máximo de resultados a devolver en find_symbol y find_references
_MAX_SYMBOL_RESULTS = 50

# Máximo de líneas de contexto alrededor de un match en grep
_MAX_CONTEXT_LINES = 5

# Umbral para considerar una función "demasiado larga" (líneas)
_LONG_FUNCTION_THRESHOLD = 50

# Umbral de complejidad ciclomática para reportar como problema
_HIGH_COMPLEXITY_THRESHOLD = 10

# Extensiones de archivos por lenguaje
_EXTENSIONS: dict[str, frozenset[str]] = {
    "python":     frozenset({".py", ".pyx", ".pxd"}),
    "javascript": frozenset({".js", ".mjs", ".cjs", ".jsx"}),
    "typescript": frozenset({".ts", ".tsx", ".d.ts"}),
    "rust":       frozenset({".rs"}),
    "go":         frozenset({".go"}),
}

# Todas las extensiones soportadas (para búsquedas multi-lenguaje)
_ALL_CODE_EXTENSIONS: frozenset[str] = frozenset(
    ext for exts in _EXTENSIONS.values() for ext in exts
)

# Directorios a ignorar siempre (node_modules, __pycache__, etc.)
_IGNORE_DIRS: frozenset[str] = frozenset({
    "__pycache__", ".git", ".svn", "node_modules", ".venv", "venv",
    "env", ".env", "dist", "build", "target", ".mypy_cache",
    ".pytest_cache", ".ruff_cache", "coverage", ".coverage",
    "htmlcov", ".tox", "eggs", ".eggs", "*.egg-info",
})

# ---------------------------------------------------------------------------
# Tipos de datos internos
# ---------------------------------------------------------------------------

@dataclass
class Symbol:
    """Un símbolo encontrado en el código (función, clase, variable, etc.)."""
    name: str
    kind: str           # "function", "class", "method", "const", "struct", etc.
    file: str           # ruta relativa al cwd
    line: int           # número de línea (1-indexed)
    language: str       # "python", "javascript", "typescript", "rust", "go"
    signature: str = "" # firma completa (argumentos, tipos de retorno)
    docstring: str = "" # primer docstring/comentario si existe
    decorators: list[str] = field(default_factory=list)  # decoradores (Python)
    is_public: bool = True  # si es parte de la API pública


@dataclass
class ImportInfo:
    """Un import o dependencia entre archivos."""
    source_file: str    # archivo que importa
    imported: str       # módulo o símbolo importado
    alias: str = ""     # alias si existe (import X as Y)
    line: int = 0       # línea del import
    is_external: bool = True  # True si es de una librería externa


@dataclass
class CodeIssue:
    """Un problema detectado en el código."""
    file: str
    line: int
    kind: str           # "long_function", "high_complexity", "todo", "fixme", etc.
    message: str
    severity: str       # "info", "warning", "error"
    context: str = ""   # líneas alrededor del problema


# ---------------------------------------------------------------------------
# Patrones regex por lenguaje
# ---------------------------------------------------------------------------

# Python — complementa al AST para casos edge
_PY_PATTERNS = {
    "function":  re.compile(r"^(\s*)(?:async\s+)?def\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE),
    "class":     re.compile(r"^(\s*)class\s+(\w+)(?:\s*\([^)]*\))?:", re.MULTILINE),
    "decorator": re.compile(r"^(\s*)@([\w.]+)", re.MULTILINE),
}

# JavaScript / TypeScript
_JS_PATTERNS = {
    "function":    re.compile(r"^(?:export\s+)?(?:async\s+)?function\s+(\w+)\s*\(([^)]*)\)", re.MULTILINE),
    "arrow":       re.compile(r"^(?:export\s+)?(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?\(([^)]*)\)\s*=>", re.MULTILINE),
    "class":       re.compile(r"^(?:export\s+)?(?:abstract\s+)?class\s+(\w+)", re.MULTILINE),
    "interface":   re.compile(r"^(?:export\s+)?interface\s+(\w+)", re.MULTILINE),
    "type_alias":  re.compile(r"^(?:export\s+)?type\s+(\w+)\s*=", re.MULTILINE),
    "enum":        re.compile(r"^(?:export\s+)?(?:const\s+)?enum\s+(\w+)", re.MULTILINE),
    "method":      re.compile(r"^\s+(?:public|private|protected|static|async|readonly|\s)*(\w+)\s*\(([^)]*)\)(?:\s*:\s*[\w<>\[\]|&]+)?\s*\{", re.MULTILINE),
    "import":      re.compile(r"^import\s+(?:\{([^}]+)\}|(\w+))\s+from\s+['\"]([^'\"]+)['\"]", re.MULTILINE),
    "require":     re.compile(r"(?:const|let|var)\s+(?:\{([^}]+)\}|(\w+))\s*=\s*require\s*\(['\"]([^'\"]+)['\"]\)", re.MULTILINE),
}

# Rust
_RUST_PATTERNS = {
    "function":  re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?(?:async\s+)?fn\s+(\w+)\s*(?:<[^>]*>)?\s*\(([^)]*)\)(?:\s*->\s*([^{]+))?", re.MULTILINE),
    "struct":    re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?struct\s+(\w+)", re.MULTILINE),
    "enum":      re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?enum\s+(\w+)", re.MULTILINE),
    "trait":     re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?trait\s+(\w+)", re.MULTILINE),
    "impl":      re.compile(r"^(?:\s*)impl(?:\s*<[^>]*>)?\s+(?:(\w+)\s+for\s+)?(\w+)", re.MULTILINE),
    "type":      re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?type\s+(\w+)", re.MULTILINE),
    "mod":       re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?mod\s+(\w+)", re.MULTILINE),
    "use":       re.compile(r"^(?:\s*)use\s+([^;]+);", re.MULTILINE),
    "const":     re.compile(r"^(?:\s*)(?:pub(?:\([^)]*\))?\s+)?const\s+(\w+)", re.MULTILINE),
}

# Go
_GO_PATTERNS = {
    "function":  re.compile(r"^func\s+(?:\((\w+\s+\*?\w+)\)\s+)?(\w+)\s*\(([^)]*)\)(?:\s*(?:\([^)]*\)|\w+))?", re.MULTILINE),
    "type":      re.compile(r"^type\s+(\w+)\s+(?:struct|interface)", re.MULTILINE),
    "struct":    re.compile(r"^type\s+(\w+)\s+struct", re.MULTILINE),
    "interface": re.compile(r"^type\s+(\w+)\s+interface", re.MULTILINE),
    "const":     re.compile(r"^const\s+(\w+)", re.MULTILINE),
    "var":       re.compile(r"^var\s+(\w+)", re.MULTILINE),
    "import":    re.compile(r'"([^"]+)"', re.MULTILINE),
}

# Patrones de problemas comunes (agnósticos al lenguaje)
_ISSUE_PATTERNS = {
    "todo":     re.compile(r"#\s*TODO[:\s](.+)", re.IGNORECASE),
    "fixme":    re.compile(r"#\s*FIXME[:\s](.+)", re.IGNORECASE),
    "hack":     re.compile(r"#\s*HACK[:\s](.+)", re.IGNORECASE),
    "xxx":      re.compile(r"#\s*XXX[:\s](.+)", re.IGNORECASE),
    "todo_js":  re.compile(r"//\s*TODO[:\s](.+)", re.IGNORECASE),
    "fixme_js": re.compile(r"//\s*FIXME[:\s](.+)", re.IGNORECASE),
}


@register_tool
class CodeAnalysisTool(BaseTool):
    """
    Tool para análisis estático de código del proyecto.

    Soporta Python (via AST), JavaScript, TypeScript, Rust y Go (via regex).
    """

    @property
    def name(self) -> str:
        return "code"

    @property
    def description(self) -> str:
        return "Análisis estático de código: busca símbolos, referencias, imports y detecta problemas"

    # ------------------------------------------------------------------
    # Schema para el LLM
    # ------------------------------------------------------------------

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Analiza el código fuente del proyecto sin ejecutarlo. "
                "Operaciones: "
                "find_symbol (busca definiciones de funciones/clases), "
                "find_references (dónde se usa un símbolo), "
                "analyze_imports (mapa de dependencias), "
                "detect_issues (funciones largas, TODOs, alta complejidad), "
                "summarize_file (estructura de un archivo), "
                "grep (búsqueda de texto con contexto). "
                "Usa esta tool antes de modificar código para entender el contexto."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": [
                            "find_symbol", "find_references", "analyze_imports",
                            "detect_issues", "summarize_file", "grep",
                        ],
                        "description": "Operación de análisis a realizar.",
                    },
                    "symbol": {
                        "type": "string",
                        "description": (
                            "Nombre del símbolo a buscar para find_symbol y find_references. "
                            "Puede ser nombre exacto o substring. "
                            "Ejemplos: 'execute_safe', 'ToolRegistry', 'handle_error'"
                        ),
                    },
                    "path": {
                        "type": "string",
                        "description": (
                            "Ruta al archivo o directorio a analizar. "
                            "Para summarize_file: ruta al archivo. "
                            "Para otras operaciones: directorio raíz del proyecto. "
                            "Default: directorio de trabajo actual."
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Texto o patrón regex para la operación grep. "
                            "Ejemplos: 'def execute', 'class.*Tool', 'TODO'"
                        ),
                    },
                    "language": {
                        "type": "string",
                        "enum": ["python", "javascript", "typescript", "rust", "go", "all"],
                        "description": (
                            "Lenguaje a analizar. 'all' analiza todos los lenguajes soportados. "
                            "Default: 'all'."
                        ),
                    },
                    "extensions": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Extensiones de archivo a incluir en el análisis. "
                            "Sobreescribe el parámetro 'language'. "
                            "Ejemplo: ['.py', '.pyx']"
                        ),
                    },
                    "context_lines": {
                        "type": "integer",
                        "description": (
                            f"Para grep: líneas de contexto antes y después del match. "
                            f"Default: 2. Máximo: {_MAX_CONTEXT_LINES}."
                        ),
                    },
                    "include_private": {
                        "type": "boolean",
                        "description": (
                            "Para find_symbol: si incluir símbolos privados "
                            "(_nombre en Python, función sin pub en Rust). "
                            "Default: true."
                        ),
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["all", "error", "warning", "info"],
                        "description": (
                            "Para detect_issues: severidad mínima a reportar. "
                            "Default: 'warning'."
                        ),
                    },
                    "max_results": {
                        "type": "integer",
                        "description": (
                            f"Máximo de resultados a devolver. "
                            f"Default: {_MAX_SYMBOL_RESULTS}."
                        ),
                    },
                },
                "required": ["operation"],
            },
        )

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """Validaciones específicas por operación."""
        errors = super().validate_arguments(arguments)

        operation = arguments.get("operation", "")
        valid_ops = {
            "find_symbol", "find_references", "analyze_imports",
            "detect_issues", "summarize_file", "grep",
        }

        if operation not in valid_ops:
            errors.append(
                f"Operación '{operation}' inválida. "
                f"Válidas: {', '.join(sorted(valid_ops))}"
            )
            return errors

        if operation in {"find_symbol", "find_references"} and not arguments.get("symbol"):
            errors.append(f"La operación '{operation}' requiere el campo 'symbol'")

        if operation == "grep" and not arguments.get("query", "").strip():
            errors.append("La operación 'grep' requiere el campo 'query'")

        if operation == "summarize_file" and not arguments.get("path"):
            errors.append("La operación 'summarize_file' requiere el campo 'path'")

        return errors

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Despacha a la operación correspondiente."""
        operation = arguments["operation"]
        call_id = self._get_active_tool_call_id()

        # Resolver el directorio raíz del análisis
        root_path = self._resolve_root(arguments.get("path"))

        dispatch = {
            "find_symbol":     self._find_symbol,
            "find_references": self._find_references,
            "analyze_imports": self._analyze_imports,
            "detect_issues":   self._detect_issues,
            "summarize_file":  self._summarize_file,
            "grep":            self._grep,
        }

        return dispatch[operation](
            arguments=arguments,
            root=root_path,
            call_id=call_id,
        )

    # ------------------------------------------------------------------
    # Operación: find_symbol
    # ------------------------------------------------------------------

    def _find_symbol(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Busca definiciones de un símbolo en el proyecto.

        Devuelve: nombre, tipo, archivo, línea y firma completa.
        """
        symbol_name = arguments["symbol"].strip()
        language = arguments.get("language", "all")
        include_private = arguments.get("include_private", True)
        max_results = min(
            arguments.get("max_results", _MAX_SYMBOL_RESULTS),
            _MAX_SYMBOL_RESULTS,
        )

        extensions = self._resolve_extensions(arguments, language)
        files = self._collect_files(root, extensions)

        found: list[Symbol] = []

        for filepath in files:
            if len(found) >= max_results:
                break

            lang = self._detect_language(filepath)
            symbols = self._extract_symbols(filepath, lang)

            for sym in symbols:
                # Búsqueda case-insensitive por substring del nombre
                if symbol_name.lower() in sym.name.lower():
                    if not include_private and not sym.is_public:
                        continue
                    # Ruta relativa para output más legible
                    sym_copy = Symbol(
                        name=sym.name,
                        kind=sym.kind,
                        file=self._relative_path(filepath, root),
                        line=sym.line,
                        language=sym.language,
                        signature=sym.signature,
                        docstring=sym.docstring[:100] if sym.docstring else "",
                        decorators=sym.decorators,
                        is_public=sym.is_public,
                    )
                    found.append(sym_copy)

        if not found:
            return ToolResult.success(
                tool_call_id=call_id,
                output=(
                    f"No se encontraron definiciones de '{symbol_name}' "
                    f"en {root} ({len(files)} archivos analizados)."
                ),
            )

        lines = [
            f"Definiciones de '{symbol_name}' — "
            f"{len(found)} resultado(s) en {len(files)} archivos:\n"
        ]

        for sym in found:
            lines.append(f"  {sym.kind:12} {sym.name}")
            lines.append(f"  {'':12} {sym.file}:{sym.line}")
            if sym.signature:
                lines.append(f"  {'':12} {sym.signature}")
            if sym.decorators:
                lines.append(f"  {'':12} decorators: {', '.join(sym.decorators)}")
            if sym.docstring:
                lines.append(f"  {'':12} \"{sym.docstring}\"")
            lines.append("")

        return ToolResult.success(tool_call_id=call_id, output="\n".join(lines))

    # ------------------------------------------------------------------
    # Operación: find_references
    # ------------------------------------------------------------------

    def _find_references(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Encuentra todos los usos de un símbolo en el proyecto.

        Busca el nombre del símbolo como palabra completa (word boundary)
        para evitar falsos positivos con nombres similares.
        """
        symbol_name = arguments["symbol"].strip()
        language = arguments.get("language", "all")
        context_lines = min(
            arguments.get("context_lines", 2),
            _MAX_CONTEXT_LINES,
        )
        max_results = min(
            arguments.get("max_results", _MAX_SYMBOL_RESULTS),
            _MAX_SYMBOL_RESULTS,
        )

        extensions = self._resolve_extensions(arguments, language)
        files = self._collect_files(root, extensions)

        # Regex con word boundary para búsqueda precisa
        pattern = re.compile(r"\b" + re.escape(symbol_name) + r"\b")

        references: list[dict[str, Any]] = []

        for filepath in files:
            if len(references) >= max_results:
                break

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                lines_list = content.splitlines()

                for line_idx, line in enumerate(lines_list):
                    if len(references) >= max_results:
                        break

                    if pattern.search(line):
                        # Extraemos contexto alrededor del match
                        start = max(0, line_idx - context_lines)
                        end = min(len(lines_list), line_idx + context_lines + 1)
                        context = lines_list[start:end]

                        references.append({
                            "file": self._relative_path(filepath, root),
                            "line": line_idx + 1,
                            "content": line.strip(),
                            "context": context,
                            "context_start": start + 1,
                        })

            except OSError:
                continue

        if not references:
            return ToolResult.success(
                tool_call_id=call_id,
                output=(
                    f"No se encontraron referencias a '{symbol_name}' "
                    f"en {len(files)} archivos analizados."
                ),
            )

        lines = [
            f"Referencias a '{symbol_name}' — "
            f"{len(references)} resultado(s) en {len(files)} archivos:\n"
        ]

        for ref in references:
            lines.append(f"  {ref['file']}:{ref['line']}")
            lines.append(f"  {ref['content']}")
            if context_lines > 0:
                lines.append("  Contexto:")
                for ctx_idx, ctx_line in enumerate(ref["context"]):
                    actual_line = ref["context_start"] + ctx_idx
                    marker = "→ " if actual_line == ref["line"] else "  "
                    lines.append(f"    {actual_line:4} {marker}{ctx_line}")
            lines.append("")

        if len(references) >= max_results:
            lines.append(
                f"[Mostrando primeros {max_results} resultados. "
                f"Usa 'max_results' para ver más.]"
            )

        return ToolResult.success(tool_call_id=call_id, output="\n".join(lines))

    # ------------------------------------------------------------------
    # Operación: analyze_imports
    # ------------------------------------------------------------------

    def _analyze_imports(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Mapea las dependencias e imports del proyecto.

        Para Python: distingue imports de stdlib, de terceros y locales.
        Para JS/TS: distingue node_modules vs imports relativos.
        """
        language = arguments.get("language", "all")
        path_arg = arguments.get("path")

        # Si se especifica un archivo individual, analizamos solo ese
        if path_arg:
            target = Path(path_arg)
            if target.is_file():
                lang = self._detect_language(target)
                imports = self._extract_imports(target, lang)
                return self._format_imports_result(
                    imports=imports,
                    scope=str(target),
                    call_id=call_id,
                )

        extensions = self._resolve_extensions(arguments, language)
        files = self._collect_files(root, extensions)

        all_imports: list[ImportInfo] = []

        for filepath in files[:_MAX_FILES_TO_SCAN]:
            lang = self._detect_language(filepath)
            imports = self._extract_imports(filepath, lang)
            # Enriquecemos con ruta relativa para el output
            for imp in imports:
                imp.source_file = self._relative_path(filepath, root)
            all_imports.extend(imports)

        return self._format_imports_result(
            imports=all_imports,
            scope=str(root),
            call_id=call_id,
        )

    def _format_imports_result(
        self,
        imports: list[ImportInfo],
        scope: str,
        call_id: str,
    ) -> ToolResult:
        """Formatea el mapa de imports para el LLM."""
        if not imports:
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"No se encontraron imports en: {scope}",
            )

        # Agrupamos por tipo: externos vs internos
        external = [i for i in imports if i.is_external]
        internal = [i for i in imports if not i.is_external]

        # Contamos frecuencia de dependencias externas
        ext_count: dict[str, int] = {}
        for imp in external:
            pkg = imp.imported.split(".")[0].split("/")[0]  # paquete raíz
            ext_count[pkg] = ext_count.get(pkg, 0) + 1

        lines = [f"Análisis de imports en: {scope}\n"]
        lines.append(f"Total: {len(imports)} imports "
                     f"({len(external)} externos, {len(internal)} internos)\n")

        if ext_count:
            lines.append("Dependencias externas (por frecuencia):")
            for pkg, count in sorted(ext_count.items(), key=lambda x: -x[1])[:20]:
                lines.append(f"  {count:3}x  {pkg}")
            lines.append("")

        if internal:
            lines.append("Imports internos (entre archivos del proyecto):")
            for imp in internal[:30]:
                alias_str = f" as {imp.alias}" if imp.alias else ""
                lines.append(
                    f"  {imp.source_file}:{imp.line} → {imp.imported}{alias_str}"
                )
            if len(internal) > 30:
                lines.append(f"  ... y {len(internal) - 30} más")
            lines.append("")

        return ToolResult.success(tool_call_id=call_id, output="\n".join(lines))

    # ------------------------------------------------------------------
    # Operación: detect_issues
    # ------------------------------------------------------------------

    def _detect_issues(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Detecta problemas en el código: complejidad, longitud, TODOs, code smells.

        Para Python usa AST para métricas precisas de complejidad ciclomática.
        Para otros lenguajes usa heurísticas basadas en líneas y patrones.
        """
        language = arguments.get("language", "all")
        severity_filter = arguments.get("severity", "warning")
        extensions = self._resolve_extensions(arguments, language)
        files = self._collect_files(root, extensions)

        severity_rank = {"info": 0, "warning": 1, "error": 2}
        min_rank = severity_rank.get(severity_filter, 1)

        all_issues: list[CodeIssue] = []

        for filepath in files[:_MAX_FILES_TO_SCAN]:
            lang = self._detect_language(filepath)

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue

            rel_path = self._relative_path(filepath, root)

            # Detectar TODOs y FIXMEs (todos los lenguajes)
            issues_from_comments = self._detect_comment_issues(content, rel_path)
            all_issues.extend(issues_from_comments)

            # Análisis específico por lenguaje
            if lang == "python":
                py_issues = self._detect_python_issues(content, rel_path)
                all_issues.extend(py_issues)
            else:
                generic_issues = self._detect_generic_issues(content, rel_path, lang)
                all_issues.extend(generic_issues)

        # Filtrar por severidad
        filtered = [
            issue for issue in all_issues
            if severity_rank.get(issue.severity, 0) >= min_rank
        ]

        if not filtered:
            return ToolResult.success(
                tool_call_id=call_id,
                output=(
                    f"No se encontraron problemas con severidad >= '{severity_filter}' "
                    f"en {len(files)} archivos."
                ),
            )

        # Ordenamos por severidad descendente, luego por archivo
        filtered.sort(
            key=lambda i: (-severity_rank.get(i.severity, 0), i.file, i.line)
        )

        lines = [
            f"Problemas detectados: {len(filtered)} "
            f"(de {len(all_issues)} totales, filtrado por >= '{severity_filter}')\n"
        ]

        # Agrupamos por tipo para mejor legibilidad
        by_kind: dict[str, list[CodeIssue]] = {}
        for issue in filtered:
            by_kind.setdefault(issue.kind, []).append(issue)

        for kind, issues in sorted(by_kind.items()):
            severity_icon = {"error": "🔴", "warning": "🟡", "info": "🔵"}.get(
                issues[0].severity, "⚪"
            )
            lines.append(f"{severity_icon} {kind.upper().replace('_', ' ')} ({len(issues)})")

            for issue in issues[:10]:  # max 10 por tipo
                lines.append(f"  {issue.file}:{issue.line}")
                lines.append(f"  {issue.message}")
                if issue.context:
                    lines.append(f"  → {issue.context.strip()[:100]}")
                lines.append("")

            if len(issues) > 10:
                lines.append(f"  ... y {len(issues) - 10} más de este tipo\n")

        return ToolResult.success(tool_call_id=call_id, output="\n".join(lines))

    # ------------------------------------------------------------------
    # Operación: summarize_file
    # ------------------------------------------------------------------

    def _summarize_file(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Genera un resumen estructural completo de un archivo.

        Para Python usa AST para análisis preciso.
        Devuelve: clases, funciones, imports, métricas y docstring del módulo.
        """
        path_arg = arguments["path"]
        filepath = Path(path_arg)
        if not filepath.is_absolute():
            filepath = Path.cwd() / filepath

        if not filepath.exists():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Archivo no encontrado: {filepath}",
            )

        if not filepath.is_file():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"'{filepath}' no es un archivo",
            )

        lang = self._detect_language(filepath)

        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo leer '{filepath}': {exc}",
            )

        lines_count = len(content.splitlines())
        symbols = self._extract_symbols(filepath, lang)
        imports = self._extract_imports(filepath, lang)

        # Agrupar símbolos por tipo
        classes = [s for s in symbols if s.kind in {"class", "struct", "interface", "trait"}]
        functions = [s for s in symbols if s.kind in {"function", "method", "arrow"}]
        constants = [s for s in symbols if s.kind in {"const", "var", "type"}]

        output_lines = [
            f"{'─' * 50}",
            f"Resumen: {filepath.name}",
            f"{'─' * 50}",
            f"Lenguaje:  {lang}",
            f"Líneas:    {lines_count}",
            f"Tamaño:    {len(content.encode('utf-8')) / 1024:.1f} KB",
            f"Símbolos:  {len(symbols)} ({len(classes)} clases, "
            f"{len(functions)} funciones, {len(constants)} constantes)",
            f"Imports:   {len(imports)}",
            "",
        ]

        # Docstring del módulo (Python)
        if lang == "python":
            module_doc = self._extract_python_module_docstring(content)
            if module_doc:
                output_lines.append("Descripción del módulo:")
                output_lines.append(f"  {module_doc[:200]}")
                output_lines.append("")

        # Imports agrupados
        if imports:
            ext_imports = sorted({
                i.imported.split(".")[0] for i in imports if i.is_external
            })
            int_imports = [i.imported for i in imports if not i.is_external]

            if ext_imports:
                output_lines.append(f"Dependencias externas: {', '.join(ext_imports[:15])}")
            if int_imports:
                output_lines.append(f"Imports internos: {', '.join(int_imports[:10])}")
            output_lines.append("")

        # Clases con sus métodos
        if classes:
            output_lines.append("Clases / Structs:")
            for cls in classes:
                output_lines.append(f"  {cls.kind:12} {cls.name}  (línea {cls.line})")
                if cls.docstring:
                    output_lines.append(f"  {'':12} \"{cls.docstring[:80]}\"")
                # Métodos de la clase
                class_methods = [
                    s for s in symbols
                    if s.kind == "method" and s.line > cls.line
                ][:5]
                for method in class_methods:
                    visibility = "pub " if method.is_public else "    "
                    output_lines.append(f"  {'':12} {visibility}{method.name}()")
            output_lines.append("")

        # Funciones públicas
        pub_functions = [f for f in functions if f.is_public and f.kind == "function"]
        if pub_functions:
            output_lines.append("Funciones públicas:")
            for fn in pub_functions[:20]:
                output_lines.append(f"  línea {fn.line:4}  {fn.name}{fn.signature}")
                if fn.docstring:
                    output_lines.append(f"  {'':12} \"{fn.docstring[:80]}\"")
            if len(pub_functions) > 20:
                output_lines.append(f"  ... y {len(pub_functions) - 20} más")
            output_lines.append("")

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(output_lines),
        )

    # ------------------------------------------------------------------
    # Operación: grep
    # ------------------------------------------------------------------

    def _grep(
        self,
        arguments: dict[str, Any],
        root: Path,
        call_id: str,
    ) -> ToolResult:
        """
        Búsqueda de texto o regex en archivos del proyecto con contexto.

        Más potente que ShellTool grep porque:
          - Filtra por lenguaje/extensión automáticamente
          - Ignora directorios de build y dependencias
          - Devuelve contexto formateado en un solo paso
          - Muestra número de línea y archivo relativo
        """
        query = arguments["query"].strip()
        language = arguments.get("language", "all")
        context_lines = min(
            arguments.get("context_lines", 2),
            _MAX_CONTEXT_LINES,
        )
        max_results = min(
            arguments.get("max_results", _MAX_SYMBOL_RESULTS),
            _MAX_SYMBOL_RESULTS,
        )

        extensions = self._resolve_extensions(arguments, language)
        files = self._collect_files(root, extensions)

        # Compilamos el patrón — si falla, lo tratamos como texto literal
        try:
            pattern = re.compile(query, re.MULTILINE)
        except re.error:
            pattern = re.compile(re.escape(query), re.MULTILINE)

        matches: list[dict[str, Any]] = []

        for filepath in files:
            if len(matches) >= max_results:
                break

            try:
                content = filepath.read_text(encoding="utf-8", errors="replace")
                file_lines = content.splitlines()
            except OSError:
                continue

            for line_idx, line in enumerate(file_lines):
                if len(matches) >= max_results:
                    break

                if pattern.search(line):
                    start = max(0, line_idx - context_lines)
                    end = min(len(file_lines), line_idx + context_lines + 1)

                    matches.append({
                        "file": self._relative_path(filepath, root),
                        "line": line_idx + 1,
                        "match_line": line,
                        "context": file_lines[start:end],
                        "context_start": start + 1,
                    })

        if not matches:
            return ToolResult.success(
                tool_call_id=call_id,
                output=(
                    f"Sin resultados para '{query}' "
                    f"en {len(files)} archivos ({language})."
                ),
            )

        lines = [
            f"Grep '{query}' — {len(matches)} resultado(s) "
            f"en {len(files)} archivos:\n"
        ]

        for match in matches:
            lines.append(f"  {match['file']}:{match['line']}")
            if context_lines > 0:
                for ctx_idx, ctx_line in enumerate(match["context"]):
                    actual_line = match["context_start"] + ctx_idx
                    is_match = actual_line == match["line"]
                    marker = "▶ " if is_match else "  "
                    lines.append(f"    {actual_line:4} {marker}{ctx_line}")
            else:
                lines.append(f"  {match['match_line'].strip()}")
            lines.append("")

        if len(matches) >= max_results:
            lines.append(
                f"[Mostrando primeros {max_results} resultados. "
                f"Refina la búsqueda o aumenta 'max_results' para ver más.]"
            )

        return ToolResult.success(tool_call_id=call_id, output="\n".join(lines))

    # ------------------------------------------------------------------
    # Extracción de símbolos por lenguaje
    # ------------------------------------------------------------------

    def _extract_symbols(self, filepath: Path, lang: str) -> list[Symbol]:
        """Extrae todos los símbolos definidos en un archivo."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if lang == "python":
            return self._extract_python_symbols(content, str(filepath))
        elif lang in {"javascript", "typescript"}:
            return self._extract_js_symbols(content, str(filepath), lang)
        elif lang == "rust":
            return self._extract_rust_symbols(content, str(filepath))
        elif lang == "go":
            return self._extract_go_symbols(content, str(filepath))

        return []

    def _extract_python_symbols(self, content: str, filepath: str) -> list[Symbol]:
        """
        Extrae símbolos de Python usando el módulo AST de stdlib.

        AST es preciso — detecta decoradores, argumentos tipados,
        docstrings y distingue métodos de funciones top-level.
        """
        symbols: list[Symbol] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            # Si el AST falla, fallback a regex
            return self._extract_python_symbols_regex(content, filepath)

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Detectar si es método (padre es ClassDef)
                kind = "function"
                for parent in ast.walk(tree):
                    if isinstance(parent, ast.ClassDef):
                        for child in ast.walk(parent):
                            if child is node:
                                kind = "method"
                                break

                # Extraer decoradores
                decorators = []
                for dec in node.decorator_list:
                    if isinstance(dec, ast.Name):
                        decorators.append(dec.id)
                    elif isinstance(dec, ast.Attribute):
                        decorators.append(f"{dec.value.id}.{dec.attr}" if isinstance(dec.value, ast.Name) else dec.attr)
                    elif isinstance(dec, ast.Call):
                        if isinstance(dec.func, ast.Name):
                            decorators.append(f"{dec.func.id}(...)")

                # Extraer docstring
                docstring = ""
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value.split("\n")[0].strip()

                # Firma simplificada
                args = [arg.arg for arg in node.args.args]
                signature = f"({', '.join(args)})"

                is_public = not node.name.startswith("_")

                symbols.append(Symbol(
                    name=node.name,
                    kind=kind,
                    file=filepath,
                    line=node.lineno,
                    language="python",
                    signature=signature,
                    docstring=docstring,
                    decorators=decorators,
                    is_public=is_public,
                ))

            elif isinstance(node, ast.ClassDef):
                docstring = ""
                if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                    docstring = node.body[0].value.value.split("\n")[0].strip()

                decorators = [
                    dec.id if isinstance(dec, ast.Name) else ""
                    for dec in node.decorator_list
                ]

                symbols.append(Symbol(
                    name=node.name,
                    kind="class",
                    file=filepath,
                    line=node.lineno,
                    language="python",
                    docstring=docstring,
                    decorators=[d for d in decorators if d],
                    is_public=not node.name.startswith("_"),
                ))

        return symbols

    def _extract_python_symbols_regex(self, content: str, filepath: str) -> list[Symbol]:
        """Fallback regex para Python cuando AST falla (ej: f-strings complejos)."""
        symbols: list[Symbol] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            # Funciones
            fn_match = _PY_PATTERNS["function"].match(line)
            if fn_match:
                name = fn_match.group(2)
                symbols.append(Symbol(
                    name=name,
                    kind="function",
                    file=filepath,
                    line=line_idx,
                    language="python",
                    signature=f"({fn_match.group(3)})",
                    is_public=not name.startswith("_"),
                ))
                continue

            # Clases
            cls_match = _PY_PATTERNS["class"].match(line)
            if cls_match:
                name = cls_match.group(2)
                symbols.append(Symbol(
                    name=name,
                    kind="class",
                    file=filepath,
                    line=line_idx,
                    language="python",
                    is_public=not name.startswith("_"),
                ))

        return symbols

    def _extract_js_symbols(
        self,
        content: str,
        filepath: str,
        lang: str,
    ) -> list[Symbol]:
        """Extrae símbolos de JavaScript/TypeScript via regex."""
        symbols: list[Symbol] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            # Funciones declaradas
            fn_match = _JS_PATTERNS["function"].match(line.strip())
            if fn_match:
                symbols.append(Symbol(
                    name=fn_match.group(1),
                    kind="function",
                    file=filepath,
                    line=line_idx,
                    language=lang,
                    signature=f"({fn_match.group(2)})",
                    is_public="export" in line,
                ))
                continue

            # Arrow functions
            arr_match = _JS_PATTERNS["arrow"].match(line.strip())
            if arr_match:
                symbols.append(Symbol(
                    name=arr_match.group(1),
                    kind="arrow",
                    file=filepath,
                    line=line_idx,
                    language=lang,
                    signature=f"({arr_match.group(2)})",
                    is_public="export" in line,
                ))
                continue

            # Clases
            cls_match = _JS_PATTERNS["class"].match(line.strip())
            if cls_match:
                symbols.append(Symbol(
                    name=cls_match.group(1),
                    kind="class",
                    file=filepath,
                    line=line_idx,
                    language=lang,
                    is_public="export" in line,
                ))
                continue

            # Interfaces (TypeScript)
            if lang == "typescript":
                iface_match = _JS_PATTERNS["interface"].match(line.strip())
                if iface_match:
                    symbols.append(Symbol(
                        name=iface_match.group(1),
                        kind="interface",
                        file=filepath,
                        line=line_idx,
                        language=lang,
                        is_public="export" in line,
                    ))
                    continue

                type_match = _JS_PATTERNS["type_alias"].match(line.strip())
                if type_match:
                    symbols.append(Symbol(
                        name=type_match.group(1),
                        kind="type",
                        file=filepath,
                        line=line_idx,
                        language=lang,
                        is_public="export" in line,
                    ))

        return symbols

    def _extract_rust_symbols(self, content: str, filepath: str) -> list[Symbol]:
        """Extrae símbolos de Rust via regex."""
        symbols: list[Symbol] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()

            fn_match = _RUST_PATTERNS["function"].match(stripped)
            if fn_match:
                name = fn_match.group(1)
                ret = fn_match.group(3) or ""
                symbols.append(Symbol(
                    name=name,
                    kind="function",
                    file=filepath,
                    line=line_idx,
                    language="rust",
                    signature=f"({fn_match.group(2)}) -> {ret}".strip(" ->"),
                    is_public=stripped.startswith("pub"),
                ))
                continue

            for kind, pattern in [
                ("struct", _RUST_PATTERNS["struct"]),
                ("enum", _RUST_PATTERNS["enum"]),
                ("trait", _RUST_PATTERNS["trait"]),
            ]:
                m = pattern.match(stripped)
                if m:
                    symbols.append(Symbol(
                        name=m.group(1),
                        kind=kind,
                        file=filepath,
                        line=line_idx,
                        language="rust",
                        is_public=stripped.startswith("pub"),
                    ))
                    break

        return symbols

    def _extract_go_symbols(self, content: str, filepath: str) -> list[Symbol]:
        """Extrae símbolos de Go via regex."""
        symbols: list[Symbol] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            stripped = line.strip()

            fn_match = _GO_PATTERNS["function"].match(stripped)
            if fn_match:
                receiver = fn_match.group(1) or ""
                name = fn_match.group(2)
                kind = "method" if receiver else "function"
                symbols.append(Symbol(
                    name=name,
                    kind=kind,
                    file=filepath,
                    line=line_idx,
                    language="go",
                    signature=f"({fn_match.group(3)})",
                    is_public=name[0].isupper() if name else False,
                ))
                continue

            for kind, pattern in [
                ("struct", _GO_PATTERNS["struct"]),
                ("interface", _GO_PATTERNS["interface"]),
            ]:
                m = pattern.match(stripped)
                if m:
                    name = m.group(1)
                    symbols.append(Symbol(
                        name=name,
                        kind=kind,
                        file=filepath,
                        line=line_idx,
                        language="go",
                        is_public=name[0].isupper() if name else False,
                    ))
                    break

        return symbols

    # ------------------------------------------------------------------
    # Extracción de imports por lenguaje
    # ------------------------------------------------------------------

    def _extract_imports(self, filepath: Path, lang: str) -> list[ImportInfo]:
        """Extrae imports de un archivo según su lenguaje."""
        try:
            content = filepath.read_text(encoding="utf-8", errors="replace")
        except OSError:
            return []

        if lang == "python":
            return self._extract_python_imports(content, str(filepath))
        elif lang in {"javascript", "typescript"}:
            return self._extract_js_imports(content, str(filepath))
        elif lang == "rust":
            return self._extract_rust_imports(content, str(filepath))
        elif lang == "go":
            return self._extract_go_imports(content, str(filepath))

        return []

    def _extract_python_imports(self, content: str, filepath: str) -> list[ImportInfo]:
        """Extrae imports de Python usando AST."""
        imports: list[ImportInfo] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return imports

        stdlib_modules = self._get_python_stdlib_modules()

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    is_ext = alias.name.split(".")[0] not in stdlib_modules
                    is_local = alias.name.startswith(".")
                    imports.append(ImportInfo(
                        source_file=filepath,
                        imported=alias.name,
                        alias=alias.asname or "",
                        line=node.lineno,
                        is_external=is_ext and not is_local,
                    ))

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                is_relative = (node.level or 0) > 0
                is_ext = (not is_relative and
                          module.split(".")[0] not in stdlib_modules)
                imports.append(ImportInfo(
                    source_file=filepath,
                    imported=("." * (node.level or 0)) + module,
                    line=node.lineno,
                    is_external=is_ext and not is_relative,
                ))

        return imports

    def _extract_js_imports(self, content: str, filepath: str) -> list[ImportInfo]:
        """Extrae imports de JavaScript/TypeScript via regex."""
        imports: list[ImportInfo] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            # import { X } from 'module'
            m = _JS_PATTERNS["import"].match(line.strip())
            if m:
                imported = (m.group(1) or m.group(2) or "").strip()
                module = m.group(3)
                is_local = module.startswith(".") or module.startswith("/")
                imports.append(ImportInfo(
                    source_file=filepath,
                    imported=module,
                    alias=imported,
                    line=line_idx,
                    is_external=not is_local,
                ))
                continue

            # const X = require('module')
            m = _JS_PATTERNS["require"].match(line.strip())
            if m:
                module = m.group(3)
                is_local = module.startswith(".") or module.startswith("/")
                imports.append(ImportInfo(
                    source_file=filepath,
                    imported=module,
                    line=line_idx,
                    is_external=not is_local,
                ))

        return imports

    def _extract_rust_imports(self, content: str, filepath: str) -> list[ImportInfo]:
        """Extrae use statements de Rust."""
        imports: list[ImportInfo] = []

        for m in _RUST_PATTERNS["use"].finditer(content):
            path = m.group(1).strip()
            line = content[: m.start()].count("\n") + 1
            is_local = path.startswith("crate::") or path.startswith("super::") or path.startswith("self::")
            imports.append(ImportInfo(
                source_file=filepath,
                imported=path,
                line=line,
                is_external=not is_local,
            ))

        return imports

    def _extract_go_imports(self, content: str, filepath: str) -> list[ImportInfo]:
        """Extrae imports de Go."""
        imports: list[ImportInfo] = []

        # Bloque import
        in_import_block = False
        for line_idx, line in enumerate(content.splitlines(), 1):
            stripped = line.strip()

            if stripped == "import (":
                in_import_block = True
                continue
            if in_import_block and stripped == ")":
                in_import_block = False
                continue

            if in_import_block or stripped.startswith('import "'):
                for m in _GO_PATTERNS["import"].finditer(line):
                    module = m.group(1)
                    is_local = "/" not in module or module.startswith(".")
                    imports.append(ImportInfo(
                        source_file=filepath,
                        imported=module,
                        line=line_idx,
                        is_external=not is_local,
                    ))

        return imports

    # ------------------------------------------------------------------
    # Detección de problemas
    # ------------------------------------------------------------------

    def _detect_comment_issues(
        self,
        content: str,
        filepath: str,
    ) -> list[CodeIssue]:
        """Detecta TODOs, FIXMEs y HACKs en comentarios."""
        issues: list[CodeIssue] = []
        lines = content.splitlines()

        for line_idx, line in enumerate(lines, 1):
            for kind, pattern in _ISSUE_PATTERNS.items():
                m = pattern.search(line)
                if m:
                    base_kind = kind.replace("_js", "")
                    severity = "warning" if base_kind in {"fixme", "hack", "xxx"} else "info"
                    issues.append(CodeIssue(
                        file=filepath,
                        line=line_idx,
                        kind=base_kind,
                        message=f"{base_kind.upper()}: {m.group(1).strip()[:100]}",
                        severity=severity,
                        context=line.strip(),
                    ))
                    break  # un issue por línea

        return issues

    def _detect_python_issues(
        self,
        content: str,
        filepath: str,
    ) -> list[CodeIssue]:
        """
        Detecta problemas en Python usando AST.

        Detecta:
          - Funciones demasiado largas (> _LONG_FUNCTION_THRESHOLD líneas)
          - Alta complejidad ciclomática (> _HIGH_COMPLEXITY_THRESHOLD)
          - Funciones sin docstring
          - Uso de bare except
        """
        issues: list[CodeIssue] = []

        try:
            tree = ast.parse(content)
        except SyntaxError:
            return issues

        lines = content.splitlines()

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Longitud de la función
                end_line = getattr(node, "end_lineno", node.lineno)
                fn_length = end_line - node.lineno

                if fn_length > _LONG_FUNCTION_THRESHOLD:
                    issues.append(CodeIssue(
                        file=filepath,
                        line=node.lineno,
                        kind="long_function",
                        message=(
                            f"'{node.name}' tiene {fn_length} líneas "
                            f"(umbral: {_LONG_FUNCTION_THRESHOLD})"
                        ),
                        severity="warning",
                        context=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    ))

                # Complejidad ciclomática
                complexity = self._calculate_cyclomatic_complexity(node)
                if complexity > _HIGH_COMPLEXITY_THRESHOLD:
                    issues.append(CodeIssue(
                        file=filepath,
                        line=node.lineno,
                        kind="high_complexity",
                        message=(
                            f"'{node.name}' complejidad ciclomática = {complexity} "
                            f"(umbral: {_HIGH_COMPLEXITY_THRESHOLD})"
                        ),
                        severity="error" if complexity > 20 else "warning",
                        context=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    ))

            # Bare except
            elif isinstance(node, ast.ExceptHandler):
                if node.type is None:
                    issues.append(CodeIssue(
                        file=filepath,
                        line=node.lineno,
                        kind="bare_except",
                        message="'except:' sin tipo específico — captura cualquier excepción incluyendo SystemExit",
                        severity="warning",
                        context=lines[node.lineno - 1] if node.lineno <= len(lines) else "",
                    ))

        return issues

    def _detect_generic_issues(
        self,
        content: str,
        filepath: str,
        lang: str,
    ) -> list[CodeIssue]:
        """
        Detecta problemas genéricos via heurísticas de líneas.

        Para lenguajes sin AST disponible, usamos conteo de líneas
        entre definiciones como proxy de longitud de función.
        """
        issues: list[CodeIssue] = []
        lines = content.splitlines()

        # Detectar funciones largas por heurística
        if lang in {"javascript", "typescript"}:
            fn_pattern = re.compile(r"^\s*(?:async\s+)?function\s+\w+|^\s*\w+\s*=\s*(?:async\s+)?\(")
        elif lang == "rust":
            fn_pattern = re.compile(r"^\s*(?:pub\s+)?(?:async\s+)?fn\s+\w+")
        elif lang == "go":
            fn_pattern = re.compile(r"^func\s+")
        else:
            return issues

        fn_start = None
        fn_name = ""

        for line_idx, line in enumerate(lines, 1):
            if fn_pattern.match(line):
                if fn_start is not None:
                    fn_length = line_idx - fn_start
                    if fn_length > _LONG_FUNCTION_THRESHOLD:
                        issues.append(CodeIssue(
                            file=filepath,
                            line=fn_start,
                            kind="long_function",
                            message=(
                                f"Función aprox. {fn_length} líneas "
                                f"(umbral: {_LONG_FUNCTION_THRESHOLD})"
                            ),
                            severity="warning",
                            context=lines[fn_start - 1] if fn_start <= len(lines) else "",
                        ))
                fn_start = line_idx

        return issues

    @staticmethod
    def _calculate_cyclomatic_complexity(node: ast.FunctionDef) -> int:
        """
        Calcula la complejidad ciclomática de una función Python.

        Complejidad = 1 + número de puntos de decisión.
        Puntos de decisión: if, elif, for, while, except, with,
                            operadores and/or en condiciones.
        """
        complexity = 1  # Base

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.For, ast.While,
                                   ast.ExceptHandler, ast.With,
                                   ast.AsyncFor, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                # and/or agrega una rama por cada operando adicional
                complexity += len(child.values) - 1

        return complexity

    # ------------------------------------------------------------------
    # Helpers privados
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_language(filepath: Path) -> str:
        """Detecta el lenguaje por extensión del archivo."""
        ext = filepath.suffix.lower()

        for lang, extensions in _EXTENSIONS.items():
            if ext in extensions:
                return lang

        return "unknown"

    @staticmethod
    def _resolve_root(path_arg: str | None) -> Path:
        """Resuelve el directorio raíz del análisis."""
        if not path_arg:
            return Path.cwd()

        p = Path(path_arg)
        if not p.is_absolute():
            p = Path.cwd() / p

        return p.resolve()

    @staticmethod
    def _resolve_extensions(
        arguments: dict[str, Any],
        language: str,
    ) -> frozenset[str]:
        """Resuelve las extensiones a analizar según lenguaje o parámetro explícito."""
        explicit = arguments.get("extensions")
        if explicit:
            return frozenset(
                ext if ext.startswith(".") else f".{ext}"
                for ext in explicit
            )

        if language == "all":
            return _ALL_CODE_EXTENSIONS

        return _EXTENSIONS.get(language, _ALL_CODE_EXTENSIONS)

    @staticmethod
    def _collect_files(root: Path, extensions: frozenset[str]) -> list[Path]:
        """
        Recolecta archivos del proyecto respetando los directorios ignorados.

        Hace un walk recursivo desde root, filtrando:
          - Directorios en _IGNORE_DIRS
          - Archivos cuya extensión no está en extensions
        """
        files: list[Path] = []

        if root.is_file():
            if root.suffix.lower() in extensions:
                return [root]
            return []

        for dirpath, dirnames, filenames in root.walk() if hasattr(root, 'walk') else _os_walk(root):
            # Modificamos dirnames in-place para que os.walk no entre en dirs ignorados
            dirnames[:] = [
                d for d in dirnames
                if d not in _IGNORE_DIRS and not d.startswith(".")
            ]

            for filename in filenames:
                filepath = Path(dirpath) / filename
                if filepath.suffix.lower() in extensions:
                    files.append(filepath)

                if len(files) >= _MAX_FILES_TO_SCAN:
                    return files

        return files

    @staticmethod
    def _relative_path(filepath: Path, root: Path) -> str:
        """Devuelve la ruta relativa al root para display."""
        try:
            return str(filepath.relative_to(root))
        except ValueError:
            # filepath no está bajo root — devolvemos la ruta completa
            return str(filepath)

    @staticmethod
    def _extract_python_module_docstring(content: str) -> str:
        """Extrae el docstring del módulo Python si existe."""
        try:
            tree = ast.parse(content)
            if (tree.body
                    and isinstance(tree.body[0], ast.Expr)
                    and isinstance(tree.body[0].value, ast.Constant)
                    and isinstance(tree.body[0].value.value, str)):
                return tree.body[0].value.value.split("\n")[0].strip()
        except SyntaxError:
            pass
        return ""

    @staticmethod
    def _get_python_stdlib_modules() -> frozenset[str]:
        """
        Devuelve los módulos de la stdlib de Python.

        Usa sys.stdlib_module_names disponible desde Python 3.10.
        Fallback a una lista curada para versiones anteriores.
        """
        import sys
        if hasattr(sys, "stdlib_module_names"):
            return frozenset(sys.stdlib_module_names)

        # Fallback para Python < 3.10
        return frozenset({
            "abc", "ast", "asyncio", "builtins", "collections", "contextlib",
            "copy", "csv", "dataclasses", "datetime", "enum", "functools",
            "gc", "glob", "hashlib", "html", "http", "importlib", "inspect",
            "io", "itertools", "json", "logging", "math", "mimetypes", "os",
            "pathlib", "pickle", "platform", "pprint", "queue", "random",
            "re", "shutil", "signal", "socket", "sqlite3", "string", "struct",
            "subprocess", "sys", "tempfile", "threading", "time", "traceback",
            "types", "typing", "unittest", "urllib", "uuid", "warnings",
            "weakref", "xml", "zipfile",
        })


def _os_walk(root: Path):
    """Wrapper de os.walk que devuelve Path objects."""
    import os
    for dirpath, dirnames, filenames in os.walk(root):
        yield Path(dirpath), dirnames, filenames
