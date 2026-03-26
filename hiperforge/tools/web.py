"""
WebTool — Acceso a recursos web para el agente.

Permite al agente acceder a información externa durante el loop ReAct:

  OPERACIONES:
    fetch     → obtiene el contenido de cualquier URL
    search    → búsqueda web via DuckDuckGo (sin API key requerida)
    docs      → encuentra y extrae documentación técnica de librerías
    github    → busca en GitHub (repos, issues, código, READMEs)
    download  → descarga archivos raw (JSON, CSV, texto plano)
    ping      → verifica si una URL está disponible

  LA OPERACIÓN docs ES LA MÁS PODEROSA:
    El agente puede preguntar "dame la documentación de fastapi HTTPException"
    y docs:
      1. Identifica que fastapi es Python y conoce su URL de docs
      2. Construye la URL específica para HTTPException
      3. Si falla, hace búsqueda web con DuckDuckGo
      4. Extrae el contenido relevante eliminando nav, headers, footers

  EXTRACCIÓN DE CONTENIDO:
    HTML → texto limpio usando html.parser de stdlib (sin dependencias extra).
    Elimina: <script>, <style>, <nav>, <header>, <footer>, <aside>, cookies banners.
    Preserva: código fuente, ejemplos, tablas de parámetros, descripciones.

  RATE LIMITING:
    Para no abusar de servicios externos, hay un delay mínimo entre requests
    del mismo dominio. Configurable via _MIN_REQUEST_INTERVAL_SECONDS.

  DEPENDENCIAS:
    Solo usa stdlib + requests (ya en requirements por el LLM client).
    Sin beautifulsoup, scrapy ni dependencias pesadas.
"""

from __future__ import annotations

import json
import re
import time
import urllib.parse
from html.parser import HTMLParser
from typing import Any

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from hiperforge.core.constants import TOOL_DEFAULT_TIMEOUT_SECONDS
from hiperforge.core.logging import get_logger
from hiperforge.domain.entities.tool_call import ToolResult
from hiperforge.domain.ports.tool_port import ToolSchema
from hiperforge.tools.base import BaseTool, register_tool

logger = get_logger(__name__)

# ---------------------------------------------------------------------------
# Configuración de HTTP
# ---------------------------------------------------------------------------

# Timeout para requests web (más generoso que tools locales)
_HTTP_TIMEOUT_SECONDS = 15.0

# Delay mínimo entre requests al mismo dominio (segundos)
# Evita ser bloqueado por rate limiting de los servidores
_MIN_REQUEST_INTERVAL_SECONDS = 0.5

# Tamaño máximo de respuesta que procesamos (bytes)
# Documentación muy larga se trunca a esto antes de extraer texto
_MAX_RESPONSE_BYTES = 500_000  # 500 KB

# User-Agent que se identifica apropiadamente
_USER_AGENT = (
    "HiperForge-Agent/1.0 (AI development assistant; "
    "https://github.com/hiperforge) "
    "Python-requests/2.x"
)

# Número máximo de resultados de búsqueda a devolver
_MAX_SEARCH_RESULTS = 8

# ---------------------------------------------------------------------------
# Resolver de documentación oficial por ecosistema
#
# Estructura:
#   "nombre_paquete": {
#       "base_url": URL base de la documentación,
#       "search_pattern": patrón para búsqueda dentro del sitio,
#       "ecosystem": para búsquedas fallback,
#   }
#
# Para paquetes NO listados aquí, se usa búsqueda web como fallback.
# ---------------------------------------------------------------------------

_DOCS_REGISTRY: dict[str, dict[str, str]] = {
    # ── Python — stdlib ──────────────────────────────────────────────────
    "python":       {"base_url": "https://docs.python.org/3/", "ecosystem": "python"},
    "asyncio":      {"base_url": "https://docs.python.org/3/library/asyncio.html", "ecosystem": "python"},
    "pathlib":      {"base_url": "https://docs.python.org/3/library/pathlib.html", "ecosystem": "python"},
    "typing":       {"base_url": "https://docs.python.org/3/library/typing.html", "ecosystem": "python"},
    "dataclasses":  {"base_url": "https://docs.python.org/3/library/dataclasses.html", "ecosystem": "python"},
    "subprocess":   {"base_url": "https://docs.python.org/3/library/subprocess.html", "ecosystem": "python"},
    "re":           {"base_url": "https://docs.python.org/3/library/re.html", "ecosystem": "python"},
    "json":         {"base_url": "https://docs.python.org/3/library/json.html", "ecosystem": "python"},

    # ── Python — web frameworks ───────────────────────────────────────────
    "fastapi":      {"base_url": "https://fastapi.tiangolo.com/", "ecosystem": "python"},
    "flask":        {"base_url": "https://flask.palletsprojects.com/en/stable/", "ecosystem": "python"},
    "django":       {"base_url": "https://docs.djangoproject.com/en/stable/", "ecosystem": "python"},
    "starlette":    {"base_url": "https://www.starlette.io/", "ecosystem": "python"},
    "aiohttp":      {"base_url": "https://docs.aiohttp.org/en/stable/", "ecosystem": "python"},
    "httpx":        {"base_url": "https://www.python-httpx.org/", "ecosystem": "python"},
    "requests":     {"base_url": "https://requests.readthedocs.io/en/latest/", "ecosystem": "python"},

    # ── Python — data / AI ────────────────────────────────────────────────
    "pydantic":     {"base_url": "https://docs.pydantic.dev/latest/", "ecosystem": "python"},
    "sqlalchemy":   {"base_url": "https://docs.sqlalchemy.org/en/20/", "ecosystem": "python"},
    "pandas":       {"base_url": "https://pandas.pydata.org/docs/reference/", "ecosystem": "python"},
    "numpy":        {"base_url": "https://numpy.org/doc/stable/reference/", "ecosystem": "python"},
    "pytest":       {"base_url": "https://docs.pytest.org/en/stable/", "ecosystem": "python"},
    "anthropic":    {"base_url": "https://docs.anthropic.com/en/api/", "ecosystem": "python"},
    "openai":       {"base_url": "https://platform.openai.com/docs/api-reference/", "ecosystem": "python"},
    "langchain":    {"base_url": "https://python.langchain.com/docs/", "ecosystem": "python"},
    "structlog":    {"base_url": "https://www.structlog.org/en/stable/", "ecosystem": "python"},
    "typer":        {"base_url": "https://typer.tiangolo.com/", "ecosystem": "python"},
    "rich":         {"base_url": "https://rich.readthedocs.io/en/stable/", "ecosystem": "python"},
    "click":        {"base_url": "https://click.palletsprojects.com/en/8.x/", "ecosystem": "python"},

    # ── JavaScript / TypeScript ───────────────────────────────────────────
    "react":        {"base_url": "https://react.dev/reference/react", "ecosystem": "npm"},
    "nextjs":       {"base_url": "https://nextjs.org/docs", "ecosystem": "npm"},
    "next":         {"base_url": "https://nextjs.org/docs", "ecosystem": "npm"},
    "vue":          {"base_url": "https://vuejs.org/api/", "ecosystem": "npm"},
    "nuxt":         {"base_url": "https://nuxt.com/docs", "ecosystem": "npm"},
    "express":      {"base_url": "https://expressjs.com/en/api.html", "ecosystem": "npm"},
    "nestjs":       {"base_url": "https://docs.nestjs.com/", "ecosystem": "npm"},
    "typescript":   {"base_url": "https://www.typescriptlang.org/docs/", "ecosystem": "npm"},
    "lodash":       {"base_url": "https://lodash.com/docs/", "ecosystem": "npm"},
    "axios":        {"base_url": "https://axios-http.com/docs/", "ecosystem": "npm"},
    "zod":          {"base_url": "https://zod.dev/", "ecosystem": "npm"},
    "vitest":       {"base_url": "https://vitest.dev/api/", "ecosystem": "npm"},
    "jest":         {"base_url": "https://jestjs.io/docs/", "ecosystem": "npm"},
    "prisma":       {"base_url": "https://www.prisma.io/docs/", "ecosystem": "npm"},
    "drizzle":      {"base_url": "https://orm.drizzle.team/docs/", "ecosystem": "npm"},
    "tailwind":     {"base_url": "https://tailwindcss.com/docs/", "ecosystem": "npm"},
    "tailwindcss":  {"base_url": "https://tailwindcss.com/docs/", "ecosystem": "npm"},
    "trpc":         {"base_url": "https://trpc.io/docs/", "ecosystem": "npm"},
    "hono":         {"base_url": "https://hono.dev/docs/", "ecosystem": "npm"},

    # ── Rust ──────────────────────────────────────────────────────────────
    "rust":         {"base_url": "https://doc.rust-lang.org/std/", "ecosystem": "rust"},
    "tokio":        {"base_url": "https://docs.rs/tokio/latest/tokio/", "ecosystem": "rust"},
    "serde":        {"base_url": "https://docs.rs/serde/latest/serde/", "ecosystem": "rust"},
    "axum":         {"base_url": "https://docs.rs/axum/latest/axum/", "ecosystem": "rust"},
    "actix":        {"base_url": "https://actix.rs/docs/", "ecosystem": "rust"},
    "actix-web":    {"base_url": "https://actix.rs/docs/", "ecosystem": "rust"},
    "clap":         {"base_url": "https://docs.rs/clap/latest/clap/", "ecosystem": "rust"},
    "anyhow":       {"base_url": "https://docs.rs/anyhow/latest/anyhow/", "ecosystem": "rust"},
    "sqlx":         {"base_url": "https://docs.rs/sqlx/latest/sqlx/", "ecosystem": "rust"},

    # ── Go ────────────────────────────────────────────────────────────────
    "go":           {"base_url": "https://pkg.go.dev/std", "ecosystem": "go"},
    "gin":          {"base_url": "https://gin-gonic.com/docs/", "ecosystem": "go"},
    "echo":         {"base_url": "https://echo.labstack.com/docs/", "ecosystem": "go"},
    "fiber":        {"base_url": "https://docs.gofiber.io/", "ecosystem": "go"},
    "gorm":         {"base_url": "https://gorm.io/docs/", "ecosystem": "go"},

    # ── Web APIs (MDN) ────────────────────────────────────────────────────
    "fetch":        {"base_url": "https://developer.mozilla.org/en-US/docs/Web/API/Fetch_API", "ecosystem": "mdn"},
    "websocket":    {"base_url": "https://developer.mozilla.org/en-US/docs/Web/API/WebSocket", "ecosystem": "mdn"},
    "webworker":    {"base_url": "https://developer.mozilla.org/en-US/docs/Web/API/Web_Workers_API", "ecosystem": "mdn"},
    "css":          {"base_url": "https://developer.mozilla.org/en-US/docs/Web/CSS", "ecosystem": "mdn"},
    "html":         {"base_url": "https://developer.mozilla.org/en-US/docs/Web/HTML", "ecosystem": "mdn"},
    "javascript":   {"base_url": "https://developer.mozilla.org/en-US/docs/Web/JavaScript", "ecosystem": "mdn"},

    # ── Bases de datos ────────────────────────────────────────────────────
    "postgresql":   {"base_url": "https://www.postgresql.org/docs/current/", "ecosystem": "db"},
    "postgres":     {"base_url": "https://www.postgresql.org/docs/current/", "ecosystem": "db"},
    "mysql":        {"base_url": "https://dev.mysql.com/doc/refman/8.0/en/", "ecosystem": "db"},
    "redis":        {"base_url": "https://redis.io/docs/", "ecosystem": "db"},
    "mongodb":      {"base_url": "https://www.mongodb.com/docs/manual/", "ecosystem": "db"},
    "sqlite":       {"base_url": "https://www.sqlite.org/docs.html", "ecosystem": "db"},

    # ── DevOps / Cloud ────────────────────────────────────────────────────
    "docker":       {"base_url": "https://docs.docker.com/reference/", "ecosystem": "devops"},
    "kubernetes":   {"base_url": "https://kubernetes.io/docs/reference/", "ecosystem": "devops"},
    "k8s":          {"base_url": "https://kubernetes.io/docs/reference/", "ecosystem": "devops"},
    "git":          {"base_url": "https://git-scm.com/docs", "ecosystem": "devops"},
    "terraform":    {"base_url": "https://developer.hashicorp.com/terraform/docs", "ecosystem": "devops"},
    "github-actions": {"base_url": "https://docs.github.com/en/actions", "ecosystem": "devops"},
}

# Patrones para construir URLs de docs.rs (Rust) dinámicamente
_DOCS_RS_PATTERN = "https://docs.rs/{package}/latest/{package}/"

# Patrones para construir URLs de PyPI dinámicamente para paquetes no listados
_PYPI_PATTERN = "https://pypi.org/project/{package}/"


@register_tool
class WebTool(BaseTool):
    """
    Tool para acceder a recursos web, documentación técnica y GitHub.
    """

    def __init__(self) -> None:
        super().__init__()
        # Session HTTP con retry automático y connection pooling
        self._session = self._build_http_session()
        # Tracking de último request por dominio para rate limiting
        self._last_request_time: dict[str, float] = {}

    @property
    def name(self) -> str:
        return "web"

    @property
    def description(self) -> str:
        return "Accede a URLs, busca en la web y obtiene documentación técnica"

    # ------------------------------------------------------------------
    # Schema para el LLM
    # ------------------------------------------------------------------

    def schema(self) -> ToolSchema:
        return ToolSchema(
            name=self.name,
            description=(
                "Accede a recursos web para obtener información externa. "
                "Operaciones: fetch (URL arbitraria), search (búsqueda web), "
                "docs (documentación técnica de librerías), "
                "github (repos, issues, código), "
                "download (archivos raw), ping (verificar disponibilidad). "
                "Para documentación técnica usa 'docs' con el nombre de la librería — "
                "es más preciso que fetch o search."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "operation": {
                        "type": "string",
                        "enum": ["fetch", "search", "docs", "github", "download", "ping"],
                        "description": "Operación web a realizar.",
                    },
                    "url": {
                        "type": "string",
                        "description": (
                            "URL completa para fetch, download y ping. "
                            "Debe incluir el protocolo: https://ejemplo.com"
                        ),
                    },
                    "query": {
                        "type": "string",
                        "description": (
                            "Consulta de búsqueda para search y github. "
                            "Para search: texto libre, ej: 'python asyncio gather example'. "
                            "Para github: texto libre, ej: 'fastapi dependency injection'. "
                            "Para docs: nombre de la librería + tema, "
                            "ej: 'requests timeout', 'pydantic field validator', "
                            "'react useEffect cleanup'."
                        ),
                    },
                    "package": {
                        "type": "string",
                        "description": (
                            "Nombre del paquete/librería para operación docs. "
                            "Ejemplos: 'fastapi', 'react', 'tokio', 'pandas'. "
                            "Si no se especifica, se extrae del campo query."
                        ),
                    },
                    "ecosystem": {
                        "type": "string",
                        "enum": ["python", "npm", "rust", "go", "mdn", "auto"],
                        "description": (
                            "Ecosistema de la librería para docs. "
                            "'auto' (default) lo detecta automáticamente. "
                            "Útil cuando el nombre es ambiguo."
                        ),
                    },
                    "github_type": {
                        "type": "string",
                        "enum": ["repos", "issues", "code", "readme"],
                        "description": (
                            "Tipo de búsqueda en GitHub. "
                            "'repos': busca repositorios. "
                            "'issues': busca issues y PRs. "
                            "'code': busca en el código fuente. "
                            "'readme': obtiene el README de un repo (requiere url o query='owner/repo')."
                        ),
                    },
                    "extract_code": {
                        "type": "boolean",
                        "description": (
                            "Para fetch y docs: si true extrae solo los bloques de código. "
                            "Útil para obtener ejemplos de uso sin el texto descriptivo."
                        ),
                    },
                    "max_length": {
                        "type": "integer",
                        "description": (
                            "Máximo de caracteres a devolver en el output. "
                            f"Default: usa el límite global de la tool. "
                            "Útil para obtener solo un preview."
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
        valid_ops = {"fetch", "search", "docs", "github", "download", "ping"}

        if operation not in valid_ops:
            errors.append(
                f"Operación '{operation}' inválida. "
                f"Válidas: {', '.join(sorted(valid_ops))}"
            )
            return errors

        # fetch, download, ping requieren url
        if operation in {"fetch", "download", "ping"} and not arguments.get("url"):
            errors.append(f"La operación '{operation}' requiere el campo 'url'")

        # search requiere query
        if operation == "search" and not arguments.get("query", "").strip():
            errors.append("La operación 'search' requiere el campo 'query'")

        # docs requiere package o query
        if operation == "docs":
            if not arguments.get("package") and not arguments.get("query", "").strip():
                errors.append(
                    "La operación 'docs' requiere 'package' o 'query'. "
                    "Ejemplo: package='fastapi' o query='fastapi dependency injection'"
                )

        # github requiere query o url
        if operation == "github":
            if not arguments.get("query") and not arguments.get("url"):
                errors.append("La operación 'github' requiere 'query' o 'url'")

        # Validar formato de URL si se proporciona
        url = arguments.get("url", "")
        if url and not (url.startswith("http://") or url.startswith("https://")):
            errors.append(
                f"URL inválida: '{url}'. Debe empezar con http:// o https://"
            )

        return errors

    # ------------------------------------------------------------------
    # Ejecución
    # ------------------------------------------------------------------

    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """Despacha a la operación correspondiente."""
        operation = arguments["operation"]
        call_id = self._get_active_tool_call_id()

        dispatch = {
            "fetch":    self._fetch,
            "search":   self._search,
            "docs":     self._docs,
            "github":   self._github,
            "download": self._download,
            "ping":     self._ping,
        }

        return dispatch[operation](arguments=arguments, call_id=call_id)

    # ------------------------------------------------------------------
    # Operación: fetch
    # ------------------------------------------------------------------

    def _fetch(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Obtiene el contenido de una URL y lo convierte a texto limpio.

        Para HTML: extrae el texto principal eliminando nav, scripts, etc.
        Para JSON: lo formatea con indentación para mayor legibilidad.
        Para texto plano: lo devuelve directamente.
        """
        url = arguments["url"]
        extract_code = arguments.get("extract_code", False)
        max_length = arguments.get("max_length")

        response = self._http_get(url, call_id)
        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo obtener el contenido de: {url}",
            )

        content_type = response.headers.get("content-type", "").lower()
        raw_content = response.content[:_MAX_RESPONSE_BYTES]

        if "application/json" in content_type:
            output = self._format_json(raw_content)
        elif "text/html" in content_type:
            if extract_code:
                output = self._extract_code_blocks(raw_content.decode("utf-8", errors="replace"))
            else:
                output = self._extract_text_from_html(raw_content.decode("utf-8", errors="replace"))
        else:
            output = raw_content.decode("utf-8", errors="replace")

        if max_length and len(output) > max_length:
            output = output[:max_length] + f"\n\n[Truncado a {max_length} caracteres]"

        if not output.strip():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"La URL devolvió contenido vacío: {url}",
            )

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"URL: {url}\n\n{output}",
        )

    # ------------------------------------------------------------------
    # Operación: search
    # ------------------------------------------------------------------

    def _search(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Búsqueda web via DuckDuckGo HTML (sin API key).

        DuckDuckGo permite scraping razonable de su interfaz HTML.
        Extraemos los resultados y devolvemos título + snippet + URL.
        """
        query = arguments["query"].strip()

        # DuckDuckGo HTML endpoint — más estable que la API oficial
        search_url = f"https://html.duckduckgo.com/html/?q={urllib.parse.quote(query)}"

        response = self._http_get(search_url, call_id)
        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo conectar al motor de búsqueda para: {query}",
            )

        results = self._parse_duckduckgo_results(
            response.content.decode("utf-8", errors="replace")
        )

        if not results:
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Sin resultados para: {query}",
            )

        lines = [f"Resultados de búsqueda para: {query!r}\n"]
        for i, result in enumerate(results[:_MAX_SEARCH_RESULTS], 1):
            lines.append(f"{i}. {result['title']}")
            lines.append(f"   {result['url']}")
            if result.get("snippet"):
                lines.append(f"   {result['snippet']}")
            lines.append("")

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(lines),
        )

    # ------------------------------------------------------------------
    # Operación: docs
    # ------------------------------------------------------------------

    def _docs(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Obtiene documentación técnica de una librería.

        FLUJO:
          1. Extraer nombre del paquete de package o query
          2. Buscar en _DOCS_REGISTRY la URL oficial
          3. Si no está en el registry, construir URL de docs.rs (Rust) o PyPI
          4. Si falla el fetch directo, hacer búsqueda web con DuckDuckGo
          5. Extraer el contenido relevante de la documentación
        """
        package = arguments.get("package", "").strip().lower()
        query = arguments.get("query", "").strip()
        extract_code = arguments.get("extract_code", False)

        # Si no hay package explícito, extraemos el primero del query
        if not package and query:
            package = query.split()[0].lower()

        # Construimos el query de búsqueda completo para fallback
        full_query = query if query else package
        if package and query and package not in query.lower():
            full_query = f"{package} {query}"

        # Paso 1: buscar en el registry de docs conocidas
        doc_info = _DOCS_REGISTRY.get(package)

        if doc_info:
            base_url = doc_info["base_url"]
            logger.debug(
                "URL de docs encontrada en registry",
                package=package,
                url=base_url,
                task_id=self._task_id,
            )

            # Intentamos fetch directo de la URL base
            result = self._fetch_docs_page(
                url=base_url,
                query=full_query,
                package=package,
                extract_code=extract_code,
                call_id=call_id,
            )

            if result.success:
                return result

            logger.debug(
                "fetch directo falló, intentando búsqueda web",
                package=package,
                task_id=self._task_id,
            )

        # Paso 2: para paquetes Rust no listados, intentar docs.rs
        ecosystem = arguments.get("ecosystem", "auto")
        if ecosystem == "rust" or (ecosystem == "auto" and self._looks_like_rust_crate(package)):
            docs_rs_url = _DOCS_RS_PATTERN.format(package=package)
            result = self._fetch_docs_page(
                url=docs_rs_url,
                query=full_query,
                package=package,
                extract_code=extract_code,
                call_id=call_id,
            )
            if result.success:
                return result

        # Paso 3: fallback a búsqueda web con DuckDuckGo
        search_query = self._build_docs_search_query(
            package=package,
            query=full_query,
            ecosystem=ecosystem,
        )

        logger.debug(
            "usando búsqueda web como fallback para docs",
            package=package,
            search_query=search_query,
            task_id=self._task_id,
        )

        # Buscamos y tomamos el primer resultado relevante
        search_args = {"query": search_query}
        search_result = self._search(search_args, call_id)

        if not search_result.success:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    f"No se pudo encontrar documentación para '{package}'. "
                    f"Intenta con una búsqueda más específica o proporciona la URL directamente."
                ),
            )

        # Extraemos la primera URL de los resultados de búsqueda
        first_url = self._extract_first_url_from_search(search_result.output)

        if first_url:
            result = self._fetch_docs_page(
                url=first_url,
                query=full_query,
                package=package,
                extract_code=extract_code,
                call_id=call_id,
            )
            if result.success:
                return result

        # Si todo falló, devolvemos los resultados de búsqueda tal cual
        return ToolResult.success(
            tool_call_id=call_id,
            output=(
                f"Documentación directa no disponible para '{package}'. "
                f"Resultados de búsqueda:\n\n{search_result.output}"
            ),
        )

    def _fetch_docs_page(
        self,
        url: str,
        query: str,
        package: str,
        extract_code: bool,
        call_id: str,
    ) -> ToolResult:
        """
        Obtiene y procesa una página de documentación.

        A diferencia de _fetch genérico, optimiza el output para
        documentación técnica: preserva ejemplos de código y
        estructura de API.
        """
        response = self._http_get(url, call_id)
        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo acceder a: {url}",
            )

        raw_html = response.content[:_MAX_RESPONSE_BYTES].decode("utf-8", errors="replace")

        if extract_code:
            content = self._extract_code_blocks(raw_html)
        else:
            content = self._extract_docs_content(raw_html, query=query)

        if not content.strip():
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo extraer contenido útil de: {url}",
            )

        header = f"Documentación de '{package}'\nFuente: {url}\n{'─' * 50}\n\n"
        return ToolResult.success(
            tool_call_id=call_id,
            output=header + content,
        )

    # ------------------------------------------------------------------
    # Operación: github
    # ------------------------------------------------------------------

    def _github(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Busca en GitHub o accede a recursos específicos de un repositorio.

        Usa la API pública de GitHub (sin autenticación para operaciones básicas).
        Sin auth: 60 requests/hora. Con GITHUB_TOKEN en env: 5000/hora.
        """
        github_type = arguments.get("github_type", "repos")
        query = arguments.get("query", "").strip()
        url = arguments.get("url", "").strip()

        if github_type == "readme" and (url or query):
            return self._github_readme(url=url, query=query, call_id=call_id)

        if not query:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="La búsqueda en GitHub requiere el campo 'query'.",
            )

        # Mapeamos el tipo a la API de búsqueda de GitHub
        search_type_map = {
            "repos":  "repositories",
            "issues": "issues",
            "code":   "code",
        }

        api_type = search_type_map.get(github_type, "repositories")
        api_url = (
            f"https://api.github.com/search/{api_type}"
            f"?q={urllib.parse.quote(query)}&per_page=10"
        )

        response = self._http_get(
            api_url,
            call_id,
            headers={"Accept": "application/vnd.github+json"},
        )

        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo conectar a la API de GitHub.",
            )

        try:
            data = response.json()
        except Exception:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message="Respuesta inválida de la API de GitHub.",
            )

        items = data.get("items", [])
        if not items:
            return ToolResult.success(
                tool_call_id=call_id,
                output=f"Sin resultados en GitHub para: {query!r}",
            )

        lines = [f"GitHub {github_type} — búsqueda: {query!r}\n"]

        for item in items[:8]:
            if github_type == "repos":
                lines.append(f"📦 {item.get('full_name', '')}")
                lines.append(f"   {item.get('description', 'Sin descripción')}")
                lines.append(f"   ⭐ {item.get('stargazers_count', 0)} | {item.get('html_url', '')}")
            elif github_type == "issues":
                state = item.get("state", "")
                state_icon = "✅" if state == "closed" else "🔴"
                lines.append(f"{state_icon} [{state}] {item.get('title', '')}")
                lines.append(f"   {item.get('html_url', '')}")
            elif github_type == "code":
                lines.append(f"📄 {item.get('path', '')} en {item.get('repository', {}).get('full_name', '')}")
                lines.append(f"   {item.get('html_url', '')}")
            lines.append("")

        total = data.get("total_count", 0)
        if total > 8:
            lines.append(f"[{total} resultados totales — mostrando los primeros 8]")

        return ToolResult.success(
            tool_call_id=call_id,
            output="\n".join(lines),
        )

    def _github_readme(self, url: str, query: str, call_id: str) -> ToolResult:
        """Obtiene el README de un repositorio GitHub."""
        # Resolvemos el repo de la URL o query
        repo_path = ""
        if url:
            # Extraemos owner/repo de URLs como https://github.com/owner/repo
            match = re.search(r"github\.com/([^/]+/[^/]+)", url)
            if match:
                repo_path = match.group(1).rstrip("/")
        elif "/" in query:
            repo_path = query.strip()

        if not repo_path:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=(
                    "Para obtener el README especifica la URL del repo "
                    "o query='owner/repo'. Ejemplo: query='fastapi/fastapi'"
                ),
            )

        api_url = f"https://api.github.com/repos/{repo_path}/readme"
        response = self._http_get(
            api_url,
            call_id,
            headers={"Accept": "application/vnd.github.raw"},
        )

        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo obtener el README de: {repo_path}",
            )

        content = response.text[:_MAX_RESPONSE_BYTES]

        return ToolResult.success(
            tool_call_id=call_id,
            output=f"README de {repo_path}:\n\n{content}",
        )

    # ------------------------------------------------------------------
    # Operación: download
    # ------------------------------------------------------------------

    def _download(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Descarga el contenido raw de una URL (JSON, CSV, texto plano).

        A diferencia de fetch, no procesa HTML — devuelve el contenido tal cual.
        Ideal para: archivos de configuración, datasets pequeños, APIs REST.
        """
        url = arguments["url"]
        max_length = arguments.get("max_length")

        response = self._http_get(url, call_id)
        if response is None:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo descargar: {url}",
            )

        content_bytes = response.content[:_MAX_RESPONSE_BYTES]
        content_type = response.headers.get("content-type", "").lower()

        # Para JSON, formateamos con indentación
        if "application/json" in content_type or url.endswith(".json"):
            output = self._format_json(content_bytes)
        else:
            output = content_bytes.decode("utf-8", errors="replace")

        if max_length and len(output) > max_length:
            output = output[:max_length] + f"\n\n[Truncado a {max_length} caracteres]"

        size_kb = len(content_bytes) / 1024
        header = f"Descargado: {url} ({size_kb:.1f} KB)\n\n"

        return ToolResult.success(
            tool_call_id=call_id,
            output=header + output,
        )

    # ------------------------------------------------------------------
    # Operación: ping
    # ------------------------------------------------------------------

    def _ping(self, arguments: dict[str, Any], call_id: str) -> ToolResult:
        """
        Verifica si una URL está disponible y devuelve su status HTTP.

        Útil para verificar que una API o servicio está corriendo
        antes de intentar usarlo.
        """
        url = arguments["url"]

        try:
            self._respect_rate_limit(url)
            response = self._session.head(
                url,
                timeout=5.0,
                allow_redirects=True,
                headers={"User-Agent": _USER_AGENT},
            )

            status = response.status_code
            content_type = response.headers.get("content-type", "desconocido")
            server = response.headers.get("server", "desconocido")

            if status < 400:
                output = (
                    f"✓ {url}\n"
                    f"  Status: {status}\n"
                    f"  Content-Type: {content_type}\n"
                    f"  Server: {server}"
                )
                return ToolResult.success(tool_call_id=call_id, output=output)
            else:
                return ToolResult.failure(
                    tool_call_id=call_id,
                    error_message=f"URL respondió con error HTTP {status}: {url}",
                    output=f"Status: {status}",
                )

        except requests.exceptions.ConnectionError:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"No se pudo conectar a: {url}",
            )
        except requests.exceptions.Timeout:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Timeout al intentar conectar a: {url}",
            )
        except Exception as exc:
            return ToolResult.failure(
                tool_call_id=call_id,
                error_message=f"Error verificando {url}: {exc}",
            )

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _http_get(
        self,
        url: str,
        call_id: str,
        headers: dict[str, str] | None = None,
    ) -> requests.Response | None:
        """
        Hace un GET HTTP con manejo robusto de errores y rate limiting.

        Devuelve None en cualquier error — el caller decide el mensaje.
        """
        self._respect_rate_limit(url)

        request_headers = {"User-Agent": _USER_AGENT}
        if headers:
            request_headers.update(headers)

        try:
            response = self._session.get(
                url,
                timeout=_HTTP_TIMEOUT_SECONDS,
                headers=request_headers,
                stream=False,
            )
            response.raise_for_status()
            return response

        except requests.exceptions.HTTPError as exc:
            status = exc.response.status_code if exc.response else "desconocido"
            logger.warning(
                "HTTP error en WebTool",
                url=url,
                status=status,
                task_id=self._task_id,
            )
            return None

        except requests.exceptions.ConnectionError:
            logger.warning("error de conexión en WebTool", url=url, task_id=self._task_id)
            return None

        except requests.exceptions.Timeout:
            logger.warning("timeout en WebTool", url=url, task_id=self._task_id)
            return None

        except Exception as exc:
            logger.warning(
                "error inesperado en WebTool",
                url=url,
                error=str(exc),
                task_id=self._task_id,
            )
            return None

    def _respect_rate_limit(self, url: str) -> None:
        """
        Espera si hicimos un request al mismo dominio recientemente.

        Extrae el dominio de la URL y verifica cuánto tiempo pasó
        desde el último request a ese dominio.
        """
        try:
            domain = urllib.parse.urlparse(url).netloc
        except Exception:
            return

        if not domain:
            return

        last_time = self._last_request_time.get(domain, 0.0)
        elapsed = time.monotonic() - last_time

        if elapsed < _MIN_REQUEST_INTERVAL_SECONDS:
            sleep_time = _MIN_REQUEST_INTERVAL_SECONDS - elapsed
            time.sleep(sleep_time)

        self._last_request_time[domain] = time.monotonic()

    @staticmethod
    def _build_http_session() -> requests.Session:
        """
        Construye una sesión HTTP con retry automático y connection pooling.

        Retry en:
          - Errores de conexión (servidor no disponible momentáneamente)
          - Timeout de lectura
          - Status 429 (rate limit) y 5xx (errores del servidor)

        Connection pooling permite reutilizar conexiones TCP al mismo host,
        reduciendo latencia en requests consecutivos a la misma URL base.
        """
        session = requests.Session()

        retry_strategy = Retry(
            total=3,
            backoff_factor=0.5,           # espera: 0.5s, 1s, 2s
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET", "HEAD"],
            raise_on_status=False,
        )

        adapter = HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=5,
            pool_maxsize=10,
        )

        session.mount("https://", adapter)
        session.mount("http://", adapter)

        return session

    # ------------------------------------------------------------------
    # Extracción de contenido HTML
    # ------------------------------------------------------------------

    def _extract_text_from_html(self, html: str) -> str:
        """
        Extrae texto limpio de HTML eliminando elementos no deseados.

        Usa html.parser de stdlib — sin dependencias extra.
        Elimina: scripts, estilos, nav, header, footer, aside, banners.
        Preserva: párrafos, encabezados, listas, bloques de código, tablas.
        """
        extractor = _HtmlTextExtractor()
        extractor.feed(html)
        return extractor.get_text()

    def _extract_docs_content(self, html: str, query: str = "") -> str:
        """
        Extrae contenido de documentación técnica de forma inteligente.

        A diferencia de extract_text_from_html, optimizado para docs:
          - Prioriza secciones que contienen el término de búsqueda
          - Preserva bloques de código con mayor prioridad
          - Limita a las secciones más relevantes si el doc es muy largo
        """
        text = self._extract_text_from_html(html)

        if not query or not text:
            return text

        # Si el texto completo cabe en el límite, lo devolvemos todo
        if len(text) <= self._max_output_chars:
            return text

        # Dividimos en secciones y priorizamos las que contienen el query
        query_lower = query.lower()
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

        # Separamos párrafos relevantes de los demás
        relevant = []
        context = []

        for para in paragraphs:
            if any(term in para.lower() for term in query_lower.split()):
                relevant.append(para)
            else:
                context.append(para)

        # Construimos el output priorizando relevantes
        if relevant:
            selected = relevant[:20]  # máximo 20 párrafos relevantes
            if len("\n\n".join(selected)) < self._max_output_chars // 2:
                # Hay espacio — agregamos contexto adicional
                for para in context[:10]:
                    selected.append(para)
                    if len("\n\n".join(selected)) > self._max_output_chars * 0.8:
                        break

            return "\n\n".join(selected)

        # Sin párrafos relevantes — devolvemos el inicio del documento
        result = ""
        for para in paragraphs:
            if len(result) + len(para) > self._max_output_chars:
                break
            result += para + "\n\n"

        return result.strip()

    def _extract_code_blocks(self, html: str) -> str:
        """
        Extrae solo los bloques de código del HTML.

        Busca contenido dentro de <code>, <pre> y bloques markdown.
        Ideal cuando el agente solo necesita ejemplos de uso.
        """
        extractor = _CodeBlockExtractor()
        extractor.feed(html)
        blocks = extractor.get_blocks()

        if not blocks:
            return "No se encontraron bloques de código en la página."

        lines = [f"Bloques de código ({len(blocks)} encontrados):\n"]
        for i, block in enumerate(blocks[:20], 1):
            if len(block.strip()) > 10:  # ignoramos snippets muy cortos
                lines.append(f"── Bloque {i} ──")
                lines.append(block.strip())
                lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _format_json(content: bytes) -> str:
        """Formatea JSON con indentación para mayor legibilidad."""
        try:
            data = json.loads(content.decode("utf-8", errors="replace"))
            return json.dumps(data, indent=2, ensure_ascii=False)
        except (json.JSONDecodeError, UnicodeDecodeError):
            return content.decode("utf-8", errors="replace")

    @staticmethod
    def _parse_duckduckgo_results(html: str) -> list[dict[str, str]]:
        """
        Parsea los resultados HTML de DuckDuckGo.

        DuckDuckGo HTML tiene una estructura relativamente estable:
        los resultados están en elementos con clase 'result__body'.
        """
        parser = _DuckDuckGoParser()
        parser.feed(html)
        return parser.results

    @staticmethod
    def _extract_first_url_from_search(search_output: str) -> str | None:
        """Extrae la primera URL de los resultados de búsqueda."""
        lines = search_output.split("\n")
        for line in lines:
            line = line.strip()
            if line.startswith("http://") or line.startswith("https://"):
                return line
        return None

    @staticmethod
    def _build_docs_search_query(package: str, query: str, ecosystem: str) -> str:
        """Construye una query de búsqueda optimizada para documentación."""
        ecosystem_hints = {
            "python": f"{package} python documentation",
            "npm":    f"{package} javascript npm documentation",
            "rust":   f"{package} rust crate documentation",
            "go":     f"{package} golang documentation",
            "mdn":    f"{query} mdn developer mozilla",
            "auto":   f"{query} documentation",
        }
        return ecosystem_hints.get(ecosystem, f"{query} documentation")

    @staticmethod
    def _looks_like_rust_crate(package: str) -> bool:
        """Heurística para detectar si un paquete es probablemente un crate de Rust."""
        rust_indicators = {
            "tokio", "serde", "axum", "actix", "clap", "anyhow",
            "thiserror", "tracing", "reqwest", "hyper", "tonic",
        }
        return package in rust_indicators or package.endswith("-rs")


# ---------------------------------------------------------------------------
# Parsers HTML internos — usando html.parser de stdlib
# ---------------------------------------------------------------------------

class _HtmlTextExtractor(HTMLParser):
    """
    Extrae texto limpio de HTML.

    Ignora completamente el contenido de tags que no aportan
    información legible: script, style, nav, header, footer, aside.
    """

    # Tags cuyo contenido completo se ignora (incluyendo hijos)
    _SKIP_TAGS: frozenset[str] = frozenset({
        "script", "style", "nav", "header", "footer", "aside",
        "noscript", "iframe", "svg", "canvas", "figure",
        "meta", "link", "input", "button", "select", "textarea",
        "cookie-consent", "notification", "banner",
    })

    # Tags que agregan salto de línea antes/después
    _BLOCK_TAGS: frozenset[str] = frozenset({
        "p", "div", "section", "article", "main",
        "h1", "h2", "h3", "h4", "h5", "h6",
        "li", "tr", "br", "hr",
        "pre", "blockquote",
    })

    def __init__(self) -> None:
        super().__init__()
        self._text_parts: list[str] = []
        self._skip_depth: int = 0  # profundidad de tags ignorados
        self._in_pre: bool = False  # estamos dentro de <pre>?

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth += 1
        if tag == "pre":
            self._in_pre = True
        if tag in self._BLOCK_TAGS and not self._skip_depth:
            self._text_parts.append("\n")

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP_TAGS:
            self._skip_depth = max(0, self._skip_depth - 1)
        if tag == "pre":
            self._in_pre = False
        if tag in self._BLOCK_TAGS and not self._skip_depth:
            self._text_parts.append("\n")

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return

        text = data if self._in_pre else data.strip()
        if text:
            self._text_parts.append(text)

    def get_text(self) -> str:
        """Devuelve el texto extraído, limpiando líneas vacías múltiples."""
        raw = " ".join(
            part if part == "\n" else part
            for part in self._text_parts
        )
        # Colapsamos múltiples líneas en blanco a máximo dos
        result = re.sub(r"\n{3,}", "\n\n", raw)
        return result.strip()


class _CodeBlockExtractor(HTMLParser):
    """Extrae bloques de código de etiquetas <pre> y <code>."""

    def __init__(self) -> None:
        super().__init__()
        self._blocks: list[str] = []
        self._current_block: list[str] = []
        self._depth: int = 0  # profundidad dentro de pre/code

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"pre", "code"}:
            self._depth += 1
            self._current_block = []

    def handle_endtag(self, tag: str) -> None:
        if tag in {"pre", "code"} and self._depth > 0:
            self._depth -= 1
            if self._current_block:
                self._blocks.append("".join(self._current_block))
                self._current_block = []

    def handle_data(self, data: str) -> None:
        if self._depth > 0:
            self._current_block.append(data)

    def get_blocks(self) -> list[str]:
        return self._blocks


class _DuckDuckGoParser(HTMLParser):
    """
    Parsea los resultados de búsqueda HTML de DuckDuckGo.

    La estructura HTML de DuckDuckGo es relativamente estable.
    Extrae: título, URL y snippet de cada resultado.
    """

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict[str, str]] = []
        self._current: dict[str, str] = {}
        self._in_title: bool = False
        self._in_snippet: bool = False
        self._in_url: bool = False
        self._current_tag_classes: list[str] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        attr_dict = dict(attrs)
        classes = attr_dict.get("class", "").split()

        if "result__a" in classes:
            self._in_title = True
            self._current = {"title": "", "url": attr_dict.get("href", ""), "snippet": ""}
        elif "result__snippet" in classes:
            self._in_snippet = True
        elif "result__url" in classes:
            self._in_url = True

    def handle_endtag(self, tag: str) -> None:
        if self._in_title and tag == "a":
            self._in_title = False
            if self._current.get("title"):
                self.results.append(self._current)
                self._current = {}
        elif self._in_snippet:
            self._in_snippet = False
        elif self._in_url:
            self._in_url = False

    def handle_data(self, data: str) -> None:
        if self._in_title:
            self._current["title"] = self._current.get("title", "") + data
        elif self._in_snippet:
            self._current["snippet"] = self._current.get("snippet", "") + data.strip()
