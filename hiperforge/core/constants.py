"""
Constantes globales de HiperForge.

REGLA FUNDAMENTAL:
  Ningún valor literal ("hiperforge", 30, 0.2, etc.) debe aparecer
  hardcodeado en el código fuente. Todo valor que se use en más de
  un lugar, o que pueda necesitar cambiar, vive aquí.

  Si en algún módulo ves `timeout = 30` en vez de `timeout = DEFAULT_TOOL_TIMEOUT`,
  es un bug de mantenibilidad — ese valor se convierte en una bomba de tiempo
  cuando hay que cambiar el timeout en producción.

ORGANIZACIÓN:
  Las constantes están agrupadas por dominio funcional.
  Cada grupo tiene un comentario que explica su propósito.

CONVENCIÓN DE NOMBRES:
  - SCREAMING_SNAKE_CASE para todas las constantes.
  - Prefijo del grupo para evitar colisiones: APP_*, LLM_*, TOOL_*, etc.
"""

from __future__ import annotations

from pathlib import Path


# ---------------------------------------------------------------------------
# Identidad de la aplicación
# ---------------------------------------------------------------------------

APP_NAME = "hiperforge"
APP_VERSION = "0.1.0"
APP_DISPLAY_NAME = "HiperForge"
APP_DESCRIPTION = "Terminal AI agent for software developers"

# Directorio raíz de datos en el sistema del usuario
# TODA ruta en disco se construye a partir de este valor — nunca hardcodeada
APP_DIR = Path.home() / f".{APP_NAME}"


# ---------------------------------------------------------------------------
# Estructura de directorios dentro de APP_DIR
# Definidas como constantes para que workspace_locator.py las use
# y cualquier cambio de estructura sea automático en todo el código
# ---------------------------------------------------------------------------

# ~/.hiperforge/
DIR_WORKSPACES = APP_DIR / "workspaces"
DIR_LOGS = APP_DIR / "logs"
DIR_LOCKS = APP_DIR / "locks"

# Archivos del índice global
FILE_INDEX = APP_DIR / "index.json"
FILE_GLOBAL_PREFERENCES = APP_DIR / "preferences.json"

# Nombres de archivos dentro de cada entidad (no rutas completas)
FILENAME_WORKSPACE = "workspace.json"
FILENAME_PROJECT = "project.json"
FILENAME_TASK = "task.json"
FILENAME_PREFERENCES = "preferences.json"
FILENAME_SESSION = "session.json"       # plantilla: {session_id}.json


# ---------------------------------------------------------------------------
# Versiones de schema de datos JSON
# Incrementar cuando cambie la estructura de un archivo JSON en disco.
# migrations.py usa estos valores para saber qué migrar.
# ---------------------------------------------------------------------------

SCHEMA_VERSION_WORKSPACE = 1
SCHEMA_VERSION_PROJECT = 1
SCHEMA_VERSION_TASK = 1
SCHEMA_VERSION_PREFERENCES = 1


# ---------------------------------------------------------------------------
# Configuración del LLM
# Valores por defecto que se usan cuando el usuario no configura nada
# ---------------------------------------------------------------------------

# Proveedor por defecto al crear un workspace nuevo
LLM_DEFAULT_PROVIDER = "anthropic"

# Modelo por defecto de cada proveedor
LLM_DEFAULT_MODEL_ANTHROPIC = "claude-sonnet-4-6"
LLM_DEFAULT_MODEL_OPENAI = "gpt-4o"
LLM_DEFAULT_MODEL_OLLAMA = "llama3"

# Parámetros de generación — bajos para el agente (queremos precisión, no creatividad)
LLM_DEFAULT_TEMPERATURE = 0.2
LLM_DEFAULT_MAX_TOKENS = 4096
LLM_DEFAULT_MAX_TOKENS_PLANNING = 2048   # el plan no necesita respuestas largas
LLM_DEFAULT_MAX_TOKENS_SUMMARY = 1024    # el resumen final tampoco

# Tamaño del context window por defecto (para modelos desconocidos)
LLM_DEFAULT_CONTEXT_WINDOW = 8_192

# Cuántos tokens del context window reservar para la respuesta del LLM.
# El resto se usa para el historial de mensajes.
# Si el historial excede (context_window - LLM_CONTEXT_RESPONSE_RESERVE),
# context_builder.py trunca los mensajes más antiguos.
LLM_CONTEXT_RESPONSE_RESERVE = 1_024


# ---------------------------------------------------------------------------
# Configuración del loop ReAct
# ---------------------------------------------------------------------------

# Máximo de iteraciones del loop ReAct por subtask.
# Evita loops infinitos cuando el agente no puede avanzar.
# Un agente que itera más de esto probablemente está atascado.
REACT_MAX_ITERATIONS_PER_SUBTASK = 15

# Máximo de subtasks que puede tener un plan.
# Planes muy largos suelen ser síntoma de un prompt muy vago.
REACT_MAX_SUBTASKS = 20

# Máximo de reintentos cuando una tool falla antes de marcar la subtask como FAILED
REACT_MAX_TOOL_RETRIES = 3

# Tiempo de espera entre reintentos de tool (segundos)
REACT_RETRY_DELAY_SECONDS = 2.0


# ---------------------------------------------------------------------------
# Configuración de tools
# ---------------------------------------------------------------------------

# Timeout por defecto para ejecución de ShellTool (segundos)
TOOL_DEFAULT_TIMEOUT_SECONDS = 30.0

# Timeout extendido para comandos que sabemos que son lentos
# (ej: instalar dependencias, compilar proyectos grandes)
TOOL_EXTENDED_TIMEOUT_SECONDS = 120.0

# Tamaño máximo de output de una tool que se envía al LLM (caracteres).
# Outputs más largos se truncan para no desperdiciar tokens.
TOOL_MAX_OUTPUT_CHARS = 8_000

# Tamaño máximo de un archivo que FileTool puede leer completo (bytes).
# Archivos más grandes se leen en chunks o se resume su contenido.
TOOL_MAX_FILE_SIZE_BYTES = 500_000   # 500 KB


# ---------------------------------------------------------------------------
# Configuración de storage y persistencia
# ---------------------------------------------------------------------------

# Tiempo máximo para adquirir un file lock antes de abandonar (segundos)
STORAGE_LOCK_TIMEOUT_SECONDS = 5.0

# Extensión de los archivos de lock
STORAGE_LOCK_EXTENSION = ".lock"

# Encoding para todos los archivos JSON — nunca latin-1 ni cp1252
STORAGE_FILE_ENCODING = "utf-8"

# Indentación del JSON en disco (para que sea legible por humanos)
STORAGE_JSON_INDENT = 2


# ---------------------------------------------------------------------------
# Configuración de logging
# ---------------------------------------------------------------------------

# Nombre del archivo de log principal
LOG_FILENAME = f"{APP_NAME}.log"

# Rotación diaria — el log se archiva cada 24 horas
LOG_ROTATION = "1 day"

# Retención de logs históricos
LOG_RETENTION = "30 days"

# Nivel de log por defecto en producción
LOG_LEVEL_DEFAULT = "INFO"

# Nivel de log en modo debug (activado con --debug o env var)
LOG_LEVEL_DEBUG = "DEBUG"


# ---------------------------------------------------------------------------
# Configuración de la CLI y UI
# ---------------------------------------------------------------------------

# Ancho máximo del panel de Rich en la terminal
CLI_PANEL_WIDTH = 100

# Prefijo del prompt interactivo del agente
CLI_PROMPT_PREFIX = f"[{APP_DISPLAY_NAME}]"

# Caracteres de separación visual en la terminal
CLI_SEPARATOR = "─" * 60


# ---------------------------------------------------------------------------
# Variables de entorno que HiperForge reconoce
# Documentadas aquí para que sean fáciles de descubrir
# ---------------------------------------------------------------------------

ENV_ANTHROPIC_API_KEY = "ANTHROPIC_API_KEY"
ENV_OPENAI_API_KEY = "OPENAI_API_KEY"
ENV_OLLAMA_BASE_URL = "OLLAMA_BASE_URL"
ENV_LLM_PROVIDER = "HIPERFORGE_LLM_PROVIDER"
ENV_LLM_MODEL = "HIPERFORGE_LLM_MODEL"
ENV_DEBUG = "HIPERFORGE_DEBUG"
ENV_APP_DIR = "HIPERFORGE_APP_DIR"   # permite sobreescribir APP_DIR (útil en tests)
