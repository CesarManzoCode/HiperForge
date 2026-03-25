# ===========================================================================
# Makefile — Automatización del ciclo de desarrollo de HiperForge
#
# PROPÓSITO:
#   Interfaz unificada para todas las operaciones del ciclo de desarrollo:
#   instalación, testing, linting, formateo, build y release.
#   Un solo `make <target>` reemplaza secuencias largas de comandos.
#
# COMPATIBILIDAD MULTIPLATAFORMA:
#   Este Makefile está diseñado para funcionar en:
#     - Linux (Ubuntu, Arch, Debian, RHEL)
#     - macOS (Intel y Apple Silicon)
#     - Windows — vía Git Bash, WSL2 o MSYS2
#       (PowerShell nativo NO soporta sintaxis POSIX de Makefile)
#
#   Restricciones de compatibilidad aplicadas:
#     - Usamos $$(command) en vez de $(shell command) donde el shell
#       puede diferir entre plataformas.
#     - Evitamos comandos Unix-only sin alternativa (ej: usamos python3
#       directamente en vez de scripts sh complejos).
#     - Los path separadores usan / (POSIX) — Git Bash los traduce en Windows.
#
# USO BÁSICO:
#   make install     → instalar en modo desarrollo
#   make test        → correr tests unitarios
#   make lint        → verificar estilo de código
#   make format      → formatear código automáticamente
#   make check       → lint + types + tests (pre-commit completo)
#   make run         → ejecutar hiperforge directamente
#   make help        → listar todos los targets disponibles
#
# VARIABLES CONFIGURABLES:
#   Puedes sobreescribir cualquier variable desde la línea de comandos:
#     make test PYTEST_ARGS="-k test_executor --pdb"
#     make install PYTHON=python3.12
#     make run RUN_ARGS="workspace list"
# ===========================================================================

# ---------------------------------------------------------------------------
# Detección de plataforma
# ---------------------------------------------------------------------------
# Detectamos el OS para usar los comandos correctos en operaciones
# que difieren entre plataformas (ej: limpiar archivos .pyc).

ifeq ($(OS),Windows_NT)
    DETECTED_OS := Windows
    # En Windows (Git Bash / WSL2), rm -rf funciona via Git Bash
    RM_RF        := rm -rf
    # Python en Windows puede ser 'python' o 'python3'
    PYTHON_CMD   := python
else
    UNAME_S := $(shell uname -s)
    ifeq ($(UNAME_S),Darwin)
        DETECTED_OS := macOS
    else
        DETECTED_OS := Linux
    endif
    RM_RF        := rm -rf
    PYTHON_CMD   := python3
endif

# ---------------------------------------------------------------------------
# Variables principales — sobreescribibles desde la línea de comandos
# ---------------------------------------------------------------------------

# Intérprete de Python a usar.
# Se puede sobreescribir: make install PYTHON=python3.12
PYTHON       ?= $(PYTHON_CMD)

# Directorio del paquete fuente
SRC_DIR      := hiperforge

# Directorio de tests
TEST_DIR     := tests

# Directorio de datos de HiperForge (~/.hiperforge)
APP_DATA_DIR := $(HOME)/.hiperforge

# Argumentos extra para pytest — se pasan directamente a pytest
# Uso: make test PYTEST_ARGS="-k test_executor -x --pdb"
PYTEST_ARGS  ?=

# Argumentos extra para el comando hiperforge
# Uso: make run RUN_ARGS="workspace list --verbose"
RUN_ARGS     ?=

# Umbral mínimo de cobertura (debe coincidir con pyproject.toml)
COV_THRESHOLD := 80

# Colores para output del Makefile — desactivados si NO_COLOR está definido
# Respeta el estándar https://no-color.org/
ifndef NO_COLOR
    COLOR_RESET  := \033[0m
    COLOR_BOLD   := \033[1m
    COLOR_GREEN  := \033[32m
    COLOR_YELLOW := \033[33m
    COLOR_CYAN   := \033[36m
    COLOR_RED    := \033[31m
    COLOR_DIM    := \033[2m
else
    COLOR_RESET  :=
    COLOR_BOLD   :=
    COLOR_GREEN  :=
    COLOR_YELLOW :=
    COLOR_CYAN   :=
    COLOR_RED    :=
    COLOR_DIM    :=
endif

# ---------------------------------------------------------------------------
# Configuración de Make
# ---------------------------------------------------------------------------

# .DEFAULT_GOAL: el target que se ejecuta con solo `make` (sin argumentos)
.DEFAULT_GOAL := help

# .PHONY declara los targets que no son archivos.
# Make no verificará si existe un archivo con ese nombre — siempre ejecuta.
.PHONY: help \
        install install-dev install-test install-docs \
        run \
        test test-unit test-integration test-e2e test-all \
        cov cov-html \
        lint format format-check check types \
        clean clean-build clean-pyc clean-test clean-data \
        build build-check \
        release-check \
        docs docs-serve \
        info

# ===========================================================================
# TARGET: help
# ===========================================================================

help: ## Muestra este mensaje de ayuda
	@echo ""
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)HiperForge$(COLOR_RESET) — Comandos de desarrollo"
	@echo ""
	@echo "$(COLOR_BOLD)Instalación:$(COLOR_RESET)"
	@grep -E '^install[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Ejecución:$(COLOR_RESET)"
	@grep -E '^run[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Testing:$(COLOR_RESET)"
	@grep -E '^(test|cov)[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Calidad de código:$(COLOR_RESET)"
	@grep -E '^(lint|format|check|types)[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Build y release:$(COLOR_RESET)"
	@grep -E '^(build|release)[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Limpieza:$(COLOR_RESET)"
	@grep -E '^clean[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_BOLD)Utilidades:$(COLOR_RESET)"
	@grep -E '^(docs|info)[a-zA-Z_-]*:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  $(COLOR_GREEN)%-20s$(COLOR_RESET) %s\n", $$1, $$2}'
	@echo ""
	@echo "$(COLOR_DIM)Variables configurables:$(COLOR_RESET)"
	@echo "  $(COLOR_DIM)PYTHON=$(PYTHON)    PYTEST_ARGS=\"-k test_name\"    RUN_ARGS=\"workspace list\"$(COLOR_RESET)"
	@echo ""


# ===========================================================================
# TARGETS DE INSTALACIÓN
# ===========================================================================

install: ## Instalar HiperForge en modo desarrollo (editable) con dependencias de dev
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Instalando HiperForge en modo desarrollo...$(COLOR_RESET)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -e ".[dev]"
	@echo "$(COLOR_GREEN)✓ Instalación completada$(COLOR_RESET)"
	@echo "$(COLOR_DIM)  Ejecuta: hiperforge --version$(COLOR_RESET)"

install-dev: install ## Alias de install (instalación de desarrollo completa)

install-test: ## Instalar solo con dependencias de testing (más ligero, para CI)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Instalando con dependencias de test...$(COLOR_RESET)"
	@$(PYTHON) -m pip install --upgrade pip
	@$(PYTHON) -m pip install -e ".[test]"
	@echo "$(COLOR_GREEN)✓ Listo para testing$(COLOR_RESET)"

install-docs: ## Instalar con dependencias para generar documentación
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Instalando con dependencias de docs...$(COLOR_RESET)"
	@$(PYTHON) -m pip install -e ".[docs]"
	@echo "$(COLOR_GREEN)✓ Listo para generar docs$(COLOR_RESET)"


# ===========================================================================
# TARGET DE EJECUCIÓN
# ===========================================================================

run: ## Ejecutar hiperforge (usa RUN_ARGS para pasar argumentos: make run RUN_ARGS="run 'hola'")
	@$(PYTHON) -m hiperforge.cli.main $(RUN_ARGS)


# ===========================================================================
# TARGETS DE TESTING
# ===========================================================================

test: test-unit ## Alias de test-unit (el más común en desarrollo diario)

test-unit: ## Ejecutar solo los tests unitarios (rápidos, sin I/O ni LLM)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Tests unitarios...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/unit \
		--tb=short \
		--no-header \
		-q \
		$(PYTEST_ARGS)

test-integration: ## Ejecutar tests de integración (con disco, sin LLM real)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Tests de integración...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/integration \
		--tb=short \
		$(PYTEST_ARGS)

test-e2e: ## Ejecutar tests e2e (requiere API keys configuradas en .env)
	@echo "$(COLOR_BOLD)$(COLOR_YELLOW)→ Tests e2e (requiere API keys)...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/e2e \
		--tb=short \
		-v \
		$(PYTEST_ARGS)

test-all: ## Ejecutar toda la suite de tests
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Suite completa de tests...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR) \
		--tb=short \
		$(PYTEST_ARGS)

cov: ## Ejecutar tests unitarios con reporte de cobertura en terminal
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Tests con cobertura...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/unit \
		--cov=$(SRC_DIR) \
		--cov-report=term-missing \
		--cov-fail-under=$(COV_THRESHOLD) \
		-q \
		$(PYTEST_ARGS)

cov-html: ## Ejecutar tests con reporte de cobertura HTML (abre en navegador)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Tests con cobertura HTML...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/unit \
		--cov=$(SRC_DIR) \
		--cov-report=html \
		--cov-report=term-missing \
		--cov-fail-under=$(COV_THRESHOLD) \
		-q \
		$(PYTEST_ARGS)
	@echo "$(COLOR_GREEN)✓ Reporte generado en htmlcov/index.html$(COLOR_RESET)"
	@$(PYTHON) -m webbrowser htmlcov/index.html 2>/dev/null || \
		echo "$(COLOR_DIM)  Abre htmlcov/index.html en tu navegador$(COLOR_RESET)"


# ===========================================================================
# TARGETS DE CALIDAD DE CÓDIGO
# ===========================================================================

lint: ## Verificar estilo y errores con ruff (sin modificar archivos)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Linting con ruff...$(COLOR_RESET)"
	@$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(COLOR_GREEN)✓ Sin errores de estilo$(COLOR_RESET)"

format: ## Formatear código automáticamente con ruff
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Formateando código...$(COLOR_RESET)"
	@$(PYTHON) -m ruff format $(SRC_DIR)/ $(TEST_DIR)/
	@$(PYTHON) -m ruff check --fix $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(COLOR_GREEN)✓ Código formateado$(COLOR_RESET)"

format-check: ## Verificar formato sin modificar (falla si hay diferencias)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Verificando formato...$(COLOR_RESET)"
	@$(PYTHON) -m ruff format --check $(SRC_DIR)/ $(TEST_DIR)/
	@$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/
	@echo "$(COLOR_GREEN)✓ Formato correcto$(COLOR_RESET)"

types: ## Verificar tipos estáticos con mypy
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Type checking con mypy...$(COLOR_RESET)"
	@$(PYTHON) -m mypy $(SRC_DIR)/
	@echo "$(COLOR_GREEN)✓ Sin errores de tipos$(COLOR_RESET)"

check: ## Pipeline completo de calidad: format-check + lint + types + test-unit
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)━━━ Pipeline de calidad de código ━━━$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)[1/4] Verificando formato...$(COLOR_RESET)"
	@$(PYTHON) -m ruff format --check $(SRC_DIR)/ $(TEST_DIR)/ || \
		(echo "$(COLOR_RED)✗ Ejecuta 'make format' para corregir$(COLOR_RESET)" && exit 1)
	@echo "$(COLOR_GREEN)✓ Formato OK$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)[2/4] Linting...$(COLOR_RESET)"
	@$(PYTHON) -m ruff check $(SRC_DIR)/ $(TEST_DIR)/ || \
		(echo "$(COLOR_RED)✗ Ejecuta 'make lint' para ver los errores$(COLOR_RESET)" && exit 1)
	@echo "$(COLOR_GREEN)✓ Lint OK$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)[3/4] Type checking...$(COLOR_RESET)"
	@$(PYTHON) -m mypy $(SRC_DIR)/ || \
		(echo "$(COLOR_RED)✗ Corrige los errores de tipos antes de commitear$(COLOR_RESET)" && exit 1)
	@echo "$(COLOR_GREEN)✓ Types OK$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)[4/4] Tests unitarios...$(COLOR_RESET)"
	@$(PYTHON) -m pytest $(TEST_DIR)/unit \
		--tb=short \
		-q \
		$(PYTEST_ARGS) || \
		(echo "$(COLOR_RED)✗ Tests fallando$(COLOR_RESET)" && exit 1)
	@echo ""
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✓ Pipeline completo sin errores$(COLOR_RESET)"


# ===========================================================================
# TARGETS DE BUILD Y RELEASE
# ===========================================================================

build: clean-build ## Construir el paquete (wheel + sdist) para distribución
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Construyendo paquete...$(COLOR_RESET)"
	@$(PYTHON) -m pip install --upgrade build
	@$(PYTHON) -m build
	@echo "$(COLOR_GREEN)✓ Paquete construido en dist/$(COLOR_RESET)"
	@ls -lh dist/

build-check: build ## Construir y verificar el paquete con twine
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Verificando paquete con twine...$(COLOR_RESET)"
	@$(PYTHON) -m pip install --upgrade twine
	@$(PYTHON) -m twine check dist/*
	@echo "$(COLOR_GREEN)✓ Paquete válido para PyPI$(COLOR_RESET)"

release-check: check build-check ## Verificación completa pre-release (calidad + build + twine)
	@echo ""
	@echo "$(COLOR_GREEN)$(COLOR_BOLD)✓ Todo listo para publicar en PyPI$(COLOR_RESET)"
	@echo "$(COLOR_DIM)  Ejecuta: python -m twine upload dist/*$(COLOR_RESET)"


# ===========================================================================
# TARGETS DE DOCUMENTACIÓN
# ===========================================================================

docs: ## Generar documentación con MkDocs
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Generando documentación...$(COLOR_RESET)"
	@$(PYTHON) -m mkdocs build --strict
	@echo "$(COLOR_GREEN)✓ Documentación generada en site/$(COLOR_RESET)"

docs-serve: ## Servir documentación en modo live-reload (http://localhost:8000)
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)→ Servidor de docs en http://localhost:8000$(COLOR_RESET)"
	@$(PYTHON) -m mkdocs serve


# ===========================================================================
# TARGETS DE LIMPIEZA
# ===========================================================================

clean: clean-build clean-pyc clean-test ## Limpiar todos los artefactos generados

clean-build: ## Limpiar artefactos de build (dist/, build/, *.egg-info/)
	@echo "$(COLOR_DIM)→ Limpiando artefactos de build...$(COLOR_RESET)"
	@$(RM_RF) dist/
	@$(RM_RF) build/
	@$(RM_RF) *.egg-info/
	@$(RM_RF) $(SRC_DIR)/*.egg-info/
	@$(RM_RF) .eggs/
	@echo "$(COLOR_GREEN)✓ Build limpio$(COLOR_RESET)"

clean-pyc: ## Limpiar archivos Python compilados (__pycache__, *.pyc, *.pyo)
	@echo "$(COLOR_DIM)→ Limpiando archivos .pyc y __pycache__...$(COLOR_RESET)"
	@find . -type f -name "*.pyc" -delete
	@find . -type f -name "*.pyo" -delete
	@find . -type f -name "*~" -delete
	@find . -type d -name "__pycache__" -exec $(RM_RF) {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec $(RM_RF) {} + 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Caché Python limpio$(COLOR_RESET)"

clean-test: ## Limpiar artefactos de testing (.coverage, htmlcov/, .pytest_cache/)
	@echo "$(COLOR_DIM)→ Limpiando artefactos de tests...$(COLOR_RESET)"
	@$(RM_RF) .coverage
	@$(RM_RF) .coverage.*
	@$(RM_RF) htmlcov/
	@$(RM_RF) .pytest_cache/
	@$(RM_RF) coverage.xml
	@find . -type d -name ".mypy_cache" -exec $(RM_RF) {} + 2>/dev/null || true
	@find . -type d -name ".ruff_cache" -exec $(RM_RF) {} + 2>/dev/null || true
	@echo "$(COLOR_GREEN)✓ Artefactos de test limpios$(COLOR_RESET)"

clean-data: ## ⚠ PELIGROSO: Eliminar todos los datos de HiperForge (~/.hiperforge/)
	@echo "$(COLOR_RED)$(COLOR_BOLD)⚠ ADVERTENCIA: Esto eliminará TODOS los datos de HiperForge$(COLOR_RESET)"
	@echo "$(COLOR_RED)  Directorio: $(APP_DATA_DIR)$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_YELLOW)¿Continuar? [s/N]$(COLOR_RESET)" && read ans && \
		[ "$$ans" = "s" ] || [ "$$ans" = "S" ] || \
		(echo "$(COLOR_DIM)Cancelado$(COLOR_RESET)" && exit 1)
	@$(RM_RF) $(APP_DATA_DIR)
	@echo "$(COLOR_GREEN)✓ Datos eliminados$(COLOR_RESET)"


# ===========================================================================
# TARGET DE INFORMACIÓN
# ===========================================================================

info: ## Mostrar información del entorno de desarrollo
	@echo ""
	@echo "$(COLOR_BOLD)$(COLOR_CYAN)HiperForge — Información del entorno$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Sistema:$(COLOR_RESET)"
	@echo "  OS:       $(DETECTED_OS)"
	@echo "  Python:   $$($(PYTHON) --version 2>&1)"
	@echo "  pip:      $$($(PYTHON) -m pip --version 2>&1 | cut -d' ' -f1-2)"
	@echo ""
	@echo "$(COLOR_BOLD)Herramientas de desarrollo:$(COLOR_RESET)"
	@$(PYTHON) -m ruff --version 2>/dev/null && \
		echo "  ruff:     $$($(PYTHON) -m ruff --version)" || \
		echo "  ruff:     $(COLOR_RED)no instalado$(COLOR_RESET)"
	@$(PYTHON) -m mypy --version 2>/dev/null && \
		echo "  mypy:     $$($(PYTHON) -m mypy --version)" || \
		echo "  mypy:     $(COLOR_RED)no instalado$(COLOR_RESET)"
	@$(PYTHON) -m pytest --version 2>/dev/null && \
		echo "  pytest:   $$($(PYTHON) -m pytest --version)" || \
		echo "  pytest:   $(COLOR_RED)no instalado$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)HiperForge:$(COLOR_RESET)"
	@$(PYTHON) -m hiperforge.cli.main --version 2>/dev/null || \
		echo "  hiperforge: $(COLOR_YELLOW)no instalado — ejecuta 'make install'$(COLOR_RESET)"
	@echo ""
	@echo "$(COLOR_BOLD)Datos:$(COLOR_RESET)"
	@[ -d "$(APP_DATA_DIR)" ] && \
		echo "  ~/.hiperforge: existe ($$(du -sh $(APP_DATA_DIR) 2>/dev/null | cut -f1))" || \
		echo "  ~/.hiperforge: $(COLOR_DIM)no creado aún (se crea al ejecutar hiperforge)$(COLOR_RESET)"
	@echo ""