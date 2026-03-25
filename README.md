# HiperForge

> **Terminal AI agent for software developers** — planifica, ejecuta y hace ship de código desde tu terminal.

HiperForge es un agente ReAct que entiende tu instrucción en lenguaje natural, la descompone en pasos concretos y los ejecuta usando herramientas reales: shell, archivos, git, web y análisis de código. Todo desde la línea de comandos, en tu entorno local.

```bash
hiperforge run "agrega tests unitarios al módulo auth y asegúrate de que pasan"
```

```
Plan de ejecución (4 pasos):
  01. Analizar la estructura actual del módulo auth
  02. Crear tests unitarios para cada función pública
  03. Ejecutar pytest y corregir los fallos
  04. Verificar que la cobertura supera el 80%

✓ [1/4] Análisis completado  (2 iter, 3.1s)
✓ [2/4] Tests creados        (5 iter, 12.4s)
✓ [3/4] Tests pasando        (3 iter, 8.7s)
✓ [4/4] Cobertura: 87%       (1 iter, 2.2s)

✓ Task completada en 26.4s · 4,231 tokens · ~$0.0127
```

---

## Índice

- [Quick Start](#quick-start)
- [Instalación](#instalación)
  - [Linux](#linux)
  - [macOS](#macos)
  - [Windows](#windows)
- [Configuración de API Keys](#configuración-de-api-keys)
  - [Anthropic (Claude)](#anthropic-claude)
  - [OpenAI (GPT-4)](#openai-gpt-4)
  - [Groq (Llama / Mixtral)](#groq-llama--mixtral)
  - [Ollama (local, sin key)](#ollama-local-sin-key)
- [Primeros pasos](#primeros-pasos)
- [Referencia de comandos](#referencia-de-comandos)
- [Configuración avanzada](#configuración-avanzada)
- [Desarrollo local](#desarrollo-local)

---

## Quick Start

**3 pasos para tener HiperForge funcionando:**

```bash
# 1. Instalar
pip install hiperforge

# 2. Configurar tu API key (ejemplo con Anthropic)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# 3. Ejecutar
hiperforge run "crea un script Python que lea un CSV y genere un reporte en HTML"
```

> Si prefieres no usar variables de entorno, crea un archivo `.env` en tu directorio de trabajo. Ver [Configuración de API Keys](#configuración-de-api-keys).

---

## Instalación

### Requisitos previos

| Requisito | Versión mínima | Verificar |
|-----------|----------------|-----------|
| Python | 3.10+ | `python3 --version` |
| pip | 21.0+ | `pip --version` |

### Linux

```bash
# Ubuntu / Debian
sudo apt update && sudo apt install python3 python3-pip -y

# Arch Linux
sudo pacman -S python python-pip

# Instalar HiperForge
pip install hiperforge

# Verificar que el ejecutable está en el PATH
# Si no funciona, agrega ~/.local/bin a tu PATH:
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# Verificar instalación
hiperforge --version
```

### macOS

```bash
# Con Homebrew (recomendado)
brew install python3
pip3 install hiperforge

# El ejecutable puede quedar en una ruta fuera del PATH.
# Agrégala con:
echo 'export PATH="$(python3 -m site --user-base)/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# Verificar instalación
hiperforge --version
```

> **Apple Silicon (M1/M2/M3):** HiperForge funciona nativamente en ARM. No necesitas Rosetta.

### Windows

**Opción A — Instalación directa (recomendada):**

```powershell
# En PowerShell (como administrador o con Python en el PATH)
pip install hiperforge

# Agregar Scripts\ al PATH si no está:
# Panel de control → Variables de entorno → Path → Agregar:
# C:\Users\<tu_usuario>\AppData\Roaming\Python\Python3XX\Scripts
```

**Opción B — WSL2 (experiencia más fluida):**

```bash
# En WSL2 (Ubuntu)
pip install hiperforge
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc
hiperforge --version
```

> **Nota Windows:** HiperForge usa Rich para colores en la terminal. Para mejor experiencia visual usa [Windows Terminal](https://aka.ms/terminal) en vez de `cmd.exe` o PowerShell clásico.

### Instalación en modo desarrollo

Si quieres contribuir al proyecto o modificar el código:

```bash
git clone https://github.com/hiperforge/hiperforge.git
cd hiperforge
pip install -e ".[dev]"

# Verificar
hiperforge --version
make help   # ver todos los comandos de desarrollo
```

---

## Configuración de API Keys

HiperForge soporta cuatro proveedores de LLM. **Solo necesitas configurar uno.**

Puedes configurar las keys de tres formas (en orden de prioridad):

1. **Variable de entorno del sistema** — recomendada para uso diario
2. **Archivo `.env`** en el directorio donde ejecutas HiperForge
3. **`hiperforge config set`** — guarda en `~/.hiperforge/preferences.json`

### Anthropic (Claude)

El proveedor por defecto. Claude es el más capaz para tareas de desarrollo complejas.

```bash
# Obtén tu key en: https://console.anthropic.com/settings/keys

# Opción 1: variable de entorno (sesión actual)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# Opción 2: permanente en tu shell
echo 'export ANTHROPIC_API_KEY="sk-ant-api03-..."' >> ~/.bashrc  # o ~/.zshrc

# Opción 3: archivo .env en el proyecto
echo 'ANTHROPIC_API_KEY=sk-ant-api03-...' >> .env
```

### OpenAI (GPT-4)

```bash
# Obtén tu key en: https://platform.openai.com/api-keys

export OPENAI_API_KEY="sk-proj-..."
hiperforge config set llm.provider openai
```

### Groq (Llama / Mixtral)

Inferencia ultra-rápida. Ideal para el loop ReAct donde la latencia importa.
Tier gratuito disponible con límites generosos.

```bash
# Obtén tu key gratuita en: https://console.groq.com/keys

export GROQ_API_KEY="gsk_..."
hiperforge config set llm.provider groq
hiperforge config set llm.model llama-3.3-70b-versatile
```

### Ollama (local, sin key)

Corre modelos de lenguaje completamente en tu máquina. Sin costos, sin datos en la nube.

```bash
# 1. Instalar Ollama: https://ollama.ai
curl -fsSL https://ollama.ai/install.sh | sh   # Linux/macOS
# Windows: descargar el instalador desde https://ollama.ai

# 2. Descargar un modelo
ollama pull llama3          # 4.7 GB — buena calidad general
ollama pull codellama       # 3.8 GB — especializado en código
ollama pull phi3            # 2.2 GB — ligero, para máquinas con poca RAM

# 3. Iniciar el servidor (en segundo plano)
ollama serve &

# 4. Configurar HiperForge
hiperforge config set llm.provider ollama
hiperforge config set llm.model llama3
```

> **Requisitos para Ollama:** mínimo 8 GB de RAM. Para modelos grandes (70B) se recomienda 32 GB o una GPU.

---

## Primeros pasos

### Configurar tu primer workspace

Los workspaces son contextos de trabajo aislados. HiperForge crea uno automáticamente en la primera ejecución, pero puedes organizarlos manualmente:

```bash
# Crear workspaces por contexto
hiperforge workspace create trabajo --desc "Proyectos profesionales"
hiperforge workspace create personal

# Ver todos los workspaces
hiperforge workspace list

# Cambiar el activo
hiperforge workspace switch trabajo
```

### Crear un proyecto

```bash
hiperforge project create "API de pagos" --desc "Backend con FastAPI y Stripe" --tags backend,python
hiperforge project list
```

### Ejecutar tu primera task

```bash
# Flujo rápido: crea, planifica y ejecuta en un solo comando
hiperforge run "inicializa un proyecto FastAPI con estructura básica y requirements.txt"

# Flujo en tres pasos (más control):
hiperforge task create "agrega autenticación JWT al endpoint /users"
hiperforge task plan <task_id>    # muestra el plan antes de ejecutar
hiperforge task run  <task_id>    # ejecuta el plan
```

### Ejemplos de instrucciones

```bash
# Desarrollo
hiperforge run "refactoriza la clase UserService para seguir los principios SOLID"
hiperforge run "agrega manejo de errores apropiado a todas las funciones async"
hiperforge run "crea un Dockerfile para la aplicación con multi-stage build"

# Testing
hiperforge run "escribe tests de integración para el endpoint /api/orders"
hiperforge run "aumenta la cobertura de tests del módulo payments al 90%"

# Git y DevOps
hiperforge run "revisa los últimos 10 commits y genera un CHANGELOG.md"
hiperforge run "configura GitHub Actions para CI con pytest y ruff"

# Documentación
hiperforge run "documenta todas las funciones públicas del módulo core con docstrings"
```

---

## Referencia de comandos

### `hiperforge run`

```bash
hiperforge run "<instrucción>" [opciones]

Opciones:
  -p, --project   ID o nombre del proyecto al que vincular la task
  -w, --workspace ID o nombre del workspace (default: activo)
  -y, --yes       Ejecutar el plan sin pedir confirmación previa
  -v, --verbose   Mostrar IDs, timestamps y detalles adicionales
  -d, --debug     Logs detallados y traceback completo en errores
      --provider  Sobreescribir el proveedor LLM para esta ejecución
      --model     Sobreescribir el modelo LLM para esta ejecución
```

### `hiperforge task`

```bash
hiperforge task create "<instrucción>"   # Crear task en PENDING
hiperforge task plan   <task_id>         # Generar plan sin ejecutar
hiperforge task run    <task_id>         # Ejecutar plan existente
hiperforge task list   [--status] [--project]
```

### `hiperforge workspace`

```bash
hiperforge workspace create <nombre>    [--desc] [--activate]
hiperforge workspace list
hiperforge workspace switch <id_o_nombre>
hiperforge workspace show   [id_o_nombre]
hiperforge workspace rename <id_o_nombre> <nuevo_nombre>
hiperforge workspace archive    <id_o_nombre>
hiperforge workspace reactivate <id_o_nombre>
hiperforge workspace delete     <id_o_nombre>
```

### `hiperforge project`

```bash
hiperforge project create <nombre>   [--desc] [--tags]
hiperforge project list              [--status] [--tag]
hiperforge project show  <id_o_nombre>
hiperforge project rename <id_o_nombre> <nuevo_nombre>
hiperforge project tag    <id_o_nombre> <tag1> [tag2...]
hiperforge project untag  <id_o_nombre> <tag1> [tag2...]
hiperforge project archive    <id_o_nombre>
hiperforge project reactivate <id_o_nombre>
hiperforge project delete     <id_o_nombre>
```

### `hiperforge config`

```bash
hiperforge config get              [--workspace] [--global]
hiperforge config set <campo> <valor> [--workspace]
hiperforge config unset <campo>    [--workspace]
hiperforge config reset            [--workspace] [--yes]
hiperforge config fields           [--section llm|agent|ui]
```

**Campos configurables:**

| Campo | Tipo | Default | Descripción |
|-------|------|---------|-------------|
| `llm.provider` | str | `anthropic` | Proveedor del LLM |
| `llm.model` | str | *(default del proveedor)* | Modelo específico |
| `llm.temperature` | float | `0.2` | Temperatura de generación (0.0–2.0) |
| `llm.max_tokens` | int | `4096` | Máximo de tokens por respuesta |
| `agent.max_react_iterations` | int | `15` | Iteraciones máximas del loop ReAct |
| `agent.max_subtasks` | int | `20` | Subtasks máximas por plan |
| `agent.tool_timeout_seconds` | float | `30.0` | Timeout de tools en segundos |
| `agent.auto_confirm_plan` | bool | `false` | Ejecutar sin confirmar el plan |
| `agent.show_reasoning` | bool | `true` | Mostrar razonamiento del agente |
| `ui.show_token_usage` | bool | `true` | Mostrar tokens y costo al terminar |
| `ui.verbose` | bool | `false` | Modo detallado en la terminal |

---

## Configuración avanzada

### Cambiar proveedor por ejecución

```bash
# Sin cambiar la configuración global
hiperforge run "optimiza las queries SQL" --provider groq --model llama-3.3-70b-versatile
hiperforge run "refactoriza el módulo" --provider anthropic --model claude-opus-4-6
```

### Configuración por workspace

Cada workspace puede tener su propia configuración de LLM:

```bash
# El workspace "trabajo" usa Anthropic (más capaz, para proyectos críticos)
hiperforge config set llm.provider anthropic --workspace trabajo

# El workspace "experimentos" usa Groq (más rápido, para iteraciones rápidas)
hiperforge config set llm.provider groq --workspace experimentos
hiperforge config set llm.model llama-3.3-70b-versatile --workspace experimentos
```

### Usar `.env` por proyecto

Crea un `.env` en la raíz de cada proyecto para configuración específica:

```bash
# ~/proyectos/mi-api/.env
ANTHROPIC_API_KEY=sk-ant-api03-...
HIPERFORGE_LLM_PROVIDER=anthropic
HIPERFORGE_LLM_MODEL=claude-opus-4-6
```

HiperForge carga automáticamente el `.env` del directorio donde se ejecuta.

### Flujo de tres fases para mayor control

```bash
# 1. Crear la task (queda en PENDING)
hiperforge task create "implementa rate limiting en la API" --project mi-api
# → Task ID: 01HX4K2J8QNVR0SBPZ1Y3W9D6E

# 2. Generar el plan y revisarlo antes de ejecutar
hiperforge task plan 01HX4K2J8QNVR0SBPZ1Y3W9D6E
# Muestra los pasos que el agente ejecutará. Si no te convence,
# ajusta la instrucción y crea una nueva task.

# 3. Ejecutar el plan aprobado
hiperforge task run 01HX4K2J8QNVR0SBPZ1Y3W9D6E
```

### Variables de entorno disponibles

| Variable | Default | Descripción |
|----------|---------|-------------|
| `ANTHROPIC_API_KEY` | — | API key de Anthropic |
| `OPENAI_API_KEY` | — | API key de OpenAI |
| `GROQ_API_KEY` | — | API key de Groq |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | URL del servidor Ollama |
| `HIPERFORGE_LLM_PROVIDER` | `anthropic` | Proveedor del LLM |
| `HIPERFORGE_LLM_MODEL` | *(por proveedor)* | Modelo específico |
| `HIPERFORGE_DEBUG` | `false` | Activar modo debug |
| `HIPERFORGE_APP_DIR` | `~/.hiperforge` | Directorio de datos |

---

## Desarrollo local

### Setup del entorno

```bash
git clone https://github.com/hiperforge/hiperforge.git
cd hiperforge

# Instalar con dependencias de desarrollo
pip install -e ".[dev]"

# Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key
```

### Comandos de desarrollo

```bash
make help          # Ver todos los comandos disponibles

make test          # Tests unitarios (rápidos)
make test-int      # Tests de integración
make cov           # Tests con reporte de cobertura

make lint          # Verificar estilo con ruff
make format        # Formatear código automáticamente
make types         # Type checking con mypy
make check         # Pipeline completo: format + lint + types + tests
```

### Estructura del proyecto

```
hiperforge/
├── cli/              # Interfaz de línea de comandos (Typer + Rich)
│   ├── commands/     # Grupos de comandos: run, task, workspace, project, config
│   └── ui/           # Componentes visuales: renderer, spinner, plan_view, confirm
├── application/      # Capa de aplicación: use cases, services, DTOs
│   ├── use_cases/    # run_task, create_task, plan_task, create_workspace, ...
│   └── services/     # executor (loop ReAct), planner, tool_dispatcher
├── domain/           # Entidades y reglas de negocio (sin dependencias externas)
│   ├── entities/     # Task, Subtask, Project, Workspace, ToolCall
│   └── ports/        # Interfaces: LLMPort, StoragePort, ToolPort
├── infrastructure/   # Adaptadores concretos
│   ├── llm/          # Anthropic, OpenAI, Groq, Ollama
│   └── storage/      # JSON storage, workspace locator
├── memory/           # Persistencia: repositorios, schemas, migraciones
├── tools/            # Tools del agente: shell, file_ops, git, web, code_analysis
└── core/             # Configuración, constantes, logging, eventos
```

### Agregar un nuevo proveedor LLM

1. Crea `hiperforge/infrastructure/llm/mi_proveedor.py` heredando de `BaseLLMAdapter`.
2. Implementa los métodos abstractos: `complete()`, `format_tool_result()`, etc.
3. Regístralo en `hiperforge/infrastructure/llm/registry.py`.
4. Agrégalo a `_VALID_LLM_PROVIDERS` en `manage_prefs.py` y al `.env.example`.

### Agregar una nueva tool

1. Crea `hiperforge/tools/mi_tool.py` heredando de `BaseTool`.
2. Usa el decorador `@register_tool` para registrarla automáticamente.
3. Impleméntala en `hiperforge/tools/__init__.py`.

---

## Solución de problemas

**`hiperforge: command not found`**
```bash
# Linux/macOS
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc && source ~/.bashrc

# macOS con Homebrew Python
echo 'export PATH="$(python3 -m site --user-base)/bin:$PATH"' >> ~/.zshrc && source ~/.zshrc
```

**`AuthenticationError` o `401 Unauthorized`**
```bash
# Verificar que la key está configurada
echo $ANTHROPIC_API_KEY

# Si está vacía, configurarla
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

**`ConnectionError` con Ollama**
```bash
# Verificar que el servidor está corriendo
ollama serve

# Verificar que el modelo está descargado
ollama list

# Si no está, descargarlo
ollama pull llama3
```

**El agente se queda atascado en una subtask**

Cuando el agente agota sus iteraciones, la CLI pregunta si quieres reintentar, omitir o cancelar. Si el problema persiste:
```bash
# Aumentar el límite de iteraciones
hiperforge config set agent.max_react_iterations 25

# O usar un modelo más capaz
hiperforge run "..." --model claude-opus-4-6
```

---

## Licencia

MIT © [César Manzo](https://github.com/cesarmanzo)