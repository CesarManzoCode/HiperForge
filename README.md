# HiperForge

> **Terminal AI agent for software developers** — planifica, ejecuta y hace ship de código desde tu terminal.

HiperForge es un agente de IA orientado a desarrollo de software que corre en tu terminal, entiende instrucciones en lenguaje natural, genera un plan de ejecución y lo lleva a cabo usando herramientas reales: shell, archivos, git, web y análisis de código.

Está diseñado para developers que quieren algo más que un chatbot con esteroides: un agente práctico, extensible y operable, capaz de convertir intención en acciones concretas dentro de un entorno local controlado.

```bash
hiperforge run "agrega tests unitarios al módulo auth y asegúrate de que pasan"
```

```text
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

## ¿Qué es HiperForge?

HiperForge es un **framework de agente autónomo para workflows de desarrollo**.

No se limita a responder texto: interpreta una instrucción, la descompone, decide qué herramienta usar, ejecuta acciones, observa resultados, corrige su curso si hace falta y devuelve una salida útil para avanzar trabajo real.

En otras palabras:

- **piensa en pasos**
- **usa herramientas reales**
- **itera sobre resultados**
- **mantiene contexto**
- **está hecho para developers**

---

## ¿Por qué existe?

La mayoría de proyectos de agentes caen en uno de estos dos extremos:

- **Demasiado superficiales**: bonitas demos, poca ejecución real
- **Demasiado abstractos**: mucha arquitectura, poca utilidad operativa

HiperForge busca el punto medio que sí importa en la práctica:

- ejecución concreta
- herramientas útiles
- guardrails razonables
- memoria y contexto
- ergonomía de terminal
- extensibilidad para evolucionar con el proyecto

---

## Principios de diseño

### 1. Execution-first
La prioridad es que el agente pueda **resolver tareas reales**, no solo producir texto convincente.

### 2. Developer-native
Todo gira alrededor del flujo natural de un developer: terminal, código, git, archivos, tests, debugging, scripts y automatización.

### 3. Guardrails over chaos
Un agente útil necesita poder actuar, pero un agente confiable necesita límites. HiperForge incorpora validación, restricciones, timeouts y control de herramientas.

### 4. Extensible by default
No está pensado como una caja cerrada, sino como una base para agregar nuevas tools, proveedores LLM, estrategias de ejecución y flujos propios.

### 5. Local-first mindset
HiperForge vive en tu máquina, en tu terminal, sobre tus archivos y tu entorno de trabajo.

---

## Índice

- [Quick Start](#quick-start)
- [¿Por qué HiperForge?](#por-qué-hiperforge)
- [Cómo funciona](#cómo-funciona)
- [Capacidades clave](#capacidades-clave)
- [Arquitectura](#arquitectura)
- [Seguridad y guardrails](#seguridad-y-guardrails)
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
- [Ejemplos de uso](#ejemplos-de-uso)
- [Referencia de comandos](#referencia-de-comandos)
- [Configuración avanzada](#configuración-avanzada)
- [Desarrollo local](#desarrollo-local)
- [Solución de problemas](#solución-de-problemas)
- [Roadmap](#roadmap)
- [Licencia](#licencia)

---

## Quick Start

**3 pasos para tener HiperForge funcionando:**

```bash
# 1. Clonar e instalar desde el repositorio
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge
python3 -m venv .venv
source .venv/bin/activate
pip install .

# 2. Configurar tu API key (ejemplo con Anthropic)
export ANTHROPIC_API_KEY="sk-ant-api03-..."

# 3. Ejecutar
hiperforge run "crea un script Python que lea un CSV y genere un reporte en HTML"
```

> Si prefieres no usar variables de entorno, crea un archivo `.env` en tu directorio de trabajo. Ver [Configuración de API Keys](#configuración-de-api-keys).

---

## ¿Por qué HiperForge?

Porque un agente para developers debe hacer más que “sugerir”.

HiperForge está hecho para:

- analizar estructuras de proyecto
- modificar código
- crear archivos
- ejecutar tests
- revisar resultados
- iterar hasta completar una task
- trabajar desde CLI sin fricción innecesaria

### Diferenciadores

- **Terminal-native**: no depende de una interfaz visual compleja
- **ReAct loop real**: planifica, ejecuta, observa y corrige
- **Tooling útil**: shell, archivos, git, web y code analysis
- **Workspace/project aware**: organiza el trabajo por contextos
- **Multi-provider**: Anthropic, OpenAI, Groq y Ollama
- **Extensible**: puedes agregar providers, tools y flujos nuevos
- **Operable**: pensado para tareas reales de ingeniería, no solo demos

---

## Cómo funciona

A nivel conceptual, HiperForge sigue un flujo como este:

```text
Instrucción del usuario
        ↓
Interpretación de la task
        ↓
Generación del plan
        ↓
Descomposición en subtasks
        ↓
Loop ReAct por subtask
  ├─ razona
  ├─ selecciona tool
  ├─ ejecuta
  ├─ observa resultado
  └─ decide siguiente paso
        ↓
Consolidación del resultado
        ↓
Salida final en CLI
```

### Ciclo operativo

1. El usuario describe una tarea en lenguaje natural
2. El planner la convierte en un plan ejecutable
3. El executor recorre subtasks con un loop iterativo
4. El dispatcher selecciona y ejecuta herramientas
5. El agente evalúa el output y corrige si es necesario
6. La CLI presenta progreso, costo, duración y estado final

---

## Capacidades clave

### Planificación de tareas
Transforma una instrucción abierta en una secuencia de pasos accionables.

### Ejecución iterativa
No depende de una sola respuesta. HiperForge puede reintentar, ajustar y continuar.

### Orquestación de herramientas
Integra varias herramientas dentro del flujo del agente, en lugar de tratarlas como plugins aislados.

### Workspaces y proyectos
Permite organizar tareas por contexto de trabajo y mantener trazabilidad.

### Configuración multi-LLM
Puedes elegir proveedor y modelo según latencia, costo o capacidad.

### Observabilidad en terminal
Muestra progreso, iteraciones, tiempos, tokens y costo estimado.

---

## Arquitectura

HiperForge sigue una estructura modular orientada a separar responsabilidades:

```text
hiperforge/
├── cli/              # Interfaz de línea de comandos (Typer + Rich)
│   ├── commands/     # run, task, workspace, project, config
│   └── ui/           # renderer, spinner, plan_view, confirm
├── application/      # Use cases, services y orquestación
│   ├── use_cases/    # run_task, create_task, plan_task, etc.
│   └── services/     # executor, planner, tool_dispatcher
├── domain/           # Entidades y puertos
│   ├── entities/     # Task, Subtask, Project, Workspace, ToolCall
│   └── ports/        # LLMPort, StoragePort, ToolPort
├── infrastructure/   # Adaptadores concretos
│   ├── llm/          # Anthropic, OpenAI, Groq, Ollama
│   └── storage/      # JSON storage, workspace locator
├── memory/           # Persistencia, repositorios, schemas, migraciones
├── tools/            # shell, file_ops, git, web, code_analysis
└── core/             # config, logging, constantes, eventos
```

### Componentes principales

#### CLI
La puerta de entrada del usuario. Se encarga de comandos, flags, renderizado y experiencia de terminal.

#### Planner
Genera el plan inicial a partir de la instrucción del usuario.

#### Executor
Coordina la ejecución iterativa de cada subtask.

#### Tool Dispatcher
Decide qué tool llamar, valida la invocación y enruta la ejecución.

#### Memory / Storage
Guarda estado, configuración, tareas, proyectos y workspaces.

#### LLM Adapters
Abstraen la integración con distintos proveedores de modelos.

---

## Seguridad y guardrails

HiperForge está diseñado para ser útil, pero no irresponsable.

### Mecanismos de control

- validación de comandos
- límites de iteraciones
- timeout por tool
- control de directorio de trabajo
- manejo de errores
- confirmación previa opcional del plan
- posibilidad de revisar antes de ejecutar
- separación entre plan y run

### Filosofía de seguridad

HiperForge no intenta ser un “auto-runner sin frenos”.  
El objetivo es combinar autonomía práctica con suficiente control para hacerlo usable en entornos reales de desarrollo.

---

## Instalación

### Requisitos previos

| Requisito | Versión mínima | Verificar |
|-----------|----------------|-----------|
| Python | 3.10+ | `python3 --version` |
| pip | 21.0+ | `pip --version` |

### Instalación desde git clone (recomendada)

```bash
# 1) Clona el repositorio oficial
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge

# 2) Crea y activa un entorno virtual
python3 -m venv .venv
source .venv/bin/activate

# 3) Instala HiperForge desde el código fuente local
pip install .

# Verificar instalación
hiperforge --version
```

### Linux

```bash
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge
python3 -m venv .venv
source .venv/bin/activate
pip install .
hiperforge --version
```

### macOS

```bash
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge
python3 -m venv .venv
source .venv/bin/activate
pip install .
hiperforge --version
```

### Windows (PowerShell)

```powershell
# 1) Clona el repositorio oficial
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge

# 2) Crea y activa un entorno virtual
py -m venv .venv
.venv\Scripts\Activate.ps1

# 3) Instala HiperForge desde el código fuente local
pip install .

# Verificar instalación
hiperforge --version
```

> **Nota:** En macOS y Linux usa `source .venv/bin/activate`. En Windows usa `.venv\Scripts\Activate.ps1`.

### Instalación en modo desarrollo

Si quieres contribuir al proyecto o modificar el código:

```bash
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge
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
ollama pull llama3
ollama pull codellama
ollama pull phi3

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
hiperforge task plan <task_id>
hiperforge task run  <task_id>
```

---

## Ejemplos de uso

### Desarrollo

```bash
hiperforge run "refactoriza la clase UserService para seguir los principios SOLID"
hiperforge run "agrega manejo de errores apropiado a todas las funciones async"
hiperforge run "crea un Dockerfile para la aplicación con multi-stage build"
```

### Testing

```bash
hiperforge run "escribe tests de integración para el endpoint /api/orders"
hiperforge run "aumenta la cobertura de tests del módulo payments al 90%"
```

### Git y DevOps

```bash
hiperforge run "revisa los últimos 10 commits y genera un CHANGELOG.md"
hiperforge run "configura GitHub Actions para CI con pytest y ruff"
```

### Documentación

```bash
hiperforge run "documenta todas las funciones públicas del módulo core con docstrings"
```

### Modo controlado por fases

```bash
hiperforge task create "implementa rate limiting en la API" --project mi-api
hiperforge task plan <task_id>
hiperforge task run <task_id>
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
hiperforge run "optimiza las queries SQL" --provider groq --model llama-3.3-70b-versatile
hiperforge run "refactoriza el módulo" --provider anthropic --model claude-opus-4-6
```

### Configuración por workspace

Cada workspace puede tener su propia configuración de LLM:

```bash
hiperforge config set llm.provider anthropic --workspace trabajo
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

# 2. Generar el plan y revisarlo antes de ejecutar
hiperforge task plan 01HX4K2J8QNVR0SBPZ1Y3W9D6E

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
git clone https://github.com/CesarManzoCode/HiperForge.git
cd HiperForge

# Instalar con dependencias de desarrollo
pip install -e ".[dev]"

# Copiar y configurar variables de entorno
cp .env.example .env
# Editar .env con tu API key
```

### Comandos de desarrollo

```bash
make help
make test
make test-int
make cov
make lint
make format
make types
make check
```

### Estructura del proyecto

```text
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
echo $ANTHROPIC_API_KEY
export ANTHROPIC_API_KEY="sk-ant-api03-..."
```

**`ConnectionError` con Ollama**
```bash
ollama serve
ollama list
ollama pull llama3
```

**El agente se queda atascado en una subtask**

Cuando el agente agota sus iteraciones, la CLI pregunta si quieres reintentar, omitir o cancelar. Si el problema persiste:

```bash
hiperforge config set agent.max_react_iterations 25
hiperforge run "..." --model claude-opus-4-6
```

---

## Roadmap

Direcciones naturales para la evolución del proyecto:

- más tools especializadas para desarrollo
- mejor memoria contextual y recuperación
- políticas de ejecución más refinadas
- observabilidad más profunda
- mejores flujos para repos grandes
- soporte para estrategias más avanzadas de planificación
- más ergonomía para equipos y entornos colaborativos

---

## Licencia

MIT © [César Manzo](https://github.com/cesarmanzo)