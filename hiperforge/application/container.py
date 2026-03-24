"""
Container — Inyección de dependencias manual de HiperForge.

El container es el único lugar donde se instancian los objetos concretos
del sistema. Todo el resto del código trabaja con abstracciones (ports,
interfaces, clases base) y recibe sus dependencias ya construidas.

¿POR QUÉ INYECCIÓN DE DEPENDENCIAS MANUAL Y NO UN FRAMEWORK?
  Frameworks como dependency-injector o injector son poderosos pero
  añaden una capa de indirección que hace el código difícil de trazar:
    - ¿De dónde viene este objeto? No lo sé, lo resolvió el framework.
    - ¿Qué implementación concreta se usa? Depende de la configuración.
    - ¿Por qué falla este test? El framework hizo algo inesperado.

  Con DI manual, el flujo es 100% explícito y debuggeable:
    - Abres container.py y ves exactamente qué implementación se usa.
    - Si quieres cambiar algo, cambias una línea aquí.
    - Los tests construyen un Container con mocks en el constructor.

PRINCIPIO DE CONSTRUCCIÓN:
  Las dependencias se construyen en orden de menor a mayor acoplamiento.
  Cada objeto solo depende de objetos ya construidos antes que él.

  ORDEN:
    1. Settings             → sin dependencias (lee env/archivo)
    2. Locator              → depende de Settings (app_dir)
    3. Storage              → depende de Locator
    4. Store                → depende de Storage + Locator
    5. ToolRegistry         → se auto-poblado al importar tools/
    6. LLM Adapter          → depende de Settings (lazy)
    7. Services             → dependen de LLM + Tools + Store (lazy)
    8. Use Cases            → dependen de Services + Store (lazy)

FACTORIES LAZY:
  Los servicios y use cases se construyen SOLO cuando se necesitan.
  Esto tiene dos ventajas:
    1. Si el usuario hace `hiperforge workspace list`, el LLM adapter
       nunca se instancia — no hay llamada a la API ni validación de key.
    2. Si la construcción de un servicio falla, el error se produce
       en el momento de uso, con un stack trace claro y contextualizado.

USO EN LA CLI:
  # Construir el container al arrancar
  container = Container.build()

  # Obtener un use case y ejecutarlo
  output = container.run_task.execute(RunTaskInput(prompt="..."))
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from hiperforge.core.config import Settings, get_settings
from hiperforge.core.logging import get_logger, setup_logging
from hiperforge.infrastructure.llm.base import BaseLLMAdapter
from hiperforge.infrastructure.llm.registry import LLMRegistry
from hiperforge.infrastructure.storage.json_storage import JSONStorage
from hiperforge.infrastructure.storage.workspace_locator import WorkspaceLocator
from hiperforge.memory.store import MemoryStore
from hiperforge.tools.base import ToolRegistry, get_tool_registry

# Importar el paquete tools dispara @register_tool en cada tool.
# Debe ocurrir antes de que cualquier service intente usar el registry.
import hiperforge.tools  # noqa: F401  — efecto secundario intencional

if TYPE_CHECKING:
    # Importaciones solo para type hints — evitan circular imports en runtime
    from hiperforge.application.services.context_builder import ContextBuilder
    from hiperforge.application.services.executor import ExecutorService
    from hiperforge.application.services.planner import PlannerService
    from hiperforge.application.services.tool_dispatcher import ToolDispatcher
    from hiperforge.application.use_cases.create_project import CreateProjectUseCase
    from hiperforge.application.use_cases.create_workspace import CreateWorkspaceUseCase
    from hiperforge.application.use_cases.manage_prefs import ManagePrefsUseCase
    from hiperforge.application.use_cases.run_task import RunTaskUseCase
    from hiperforge.application.use_cases.switch_workspace import SwitchWorkspaceUseCase

logger = get_logger(__name__)


class Container:
    """
    Contenedor de todas las dependencias del sistema HiperForge.

    Construir con Container.build() — nunca instanciar directamente.

    Las dependencias de infraestructura (settings, storage, store) se
    construyen en build() y viven toda la vida del proceso.

    Los servicios y use cases se construyen lazy mediante properties
    para evitar instanciar código que no se va a usar en esta invocación.

    THREAD SAFETY:
      El container no es thread-safe intencionalmente.
      HiperForge es una CLI single-process — un comando a la vez.
      Si en el futuro se necesita concurrencia, cada thread debe
      construir su propio Container o agregar locks aquí.
    """

    def __init__(
        self,
        settings: Settings,
        locator: WorkspaceLocator,
        storage: JSONStorage,
        store: MemoryStore,
        tool_registry: ToolRegistry,
    ) -> None:
        # Dependencias de infraestructura — viven toda la vida del proceso
        self.settings = settings
        self.locator = locator
        self.storage = storage
        self.store = store
        self.tool_registry = tool_registry

        # Cache de instancias lazy — se llenan la primera vez que se acceden
        # None significa "aún no construido", no "no existe"
        self._llm_adapter: BaseLLMAdapter | None = None
        self._context_builder: ContextBuilder | None = None
        self._tool_dispatcher: ToolDispatcher | None = None
        self._planner: PlannerService | None = None
        self._executor: ExecutorService | None = None

    # ------------------------------------------------------------------
    # Constructor principal — único punto de entrada correcto
    # ------------------------------------------------------------------

    @classmethod
    def build(cls, app_dir: Path | None = None) -> Container:
        """
        Construye el container completo con todas las dependencias de infraestructura.

        Este es el único método que debe llamarse para crear un Container.
        Instanciar Container() directamente está prohibido porque el orden
        de construcción importa.

        Parámetros:
            app_dir: Sobreescribe el directorio de datos (~/.hiperforge/).
                     Útil en tests para aislarlos del entorno real del dev.
                     Ejemplo: Container.build(app_dir=tmp_path)

        Returns:
            Container completamente inicializado y listo para usar.

        Raises:
            ValidationError:    Si la configuración tiene valores inválidos.
            StorageWriteError:  Si no se pueden crear los directorios base.
        """
        # 1. Settings — primer objeto, sin dependencias
        settings = get_settings()

        # 2. Logging — lo antes posible para capturar todo desde aquí
        setup_logging(debug=settings.debug)

        logger.info(
            "construyendo container de HiperForge",
            provider=settings.llm_provider,
            model=settings.effective_llm_model,
            app_dir=str(app_dir or settings.app_dir),
            debug=settings.debug,
        )

        # 3. Locator — resuelve todas las rutas del sistema de archivos
        effective_app_dir = app_dir or settings.app_dir
        locator = WorkspaceLocator(app_dir=effective_app_dir)

        # 4. Storage — operaciones atómicas en disco
        storage = JSONStorage(locator=locator)

        # 5. Store — fachada unificada de acceso a datos
        store = MemoryStore(storage=storage, locator=locator)

        # 6. ToolRegistry — ya está poblado por el import de hiperforge.tools
        tool_registry = get_tool_registry()

        logger.info(
            "tools disponibles",
            tools=tool_registry.tool_names,
            count=tool_registry.tool_count,
        )

        container = cls(
            settings=settings,
            locator=locator,
            storage=storage,
            store=store,
            tool_registry=tool_registry,
        )

        logger.info("container construido exitosamente")

        return container

    # ------------------------------------------------------------------
    # LLM Adapter — lazy, se construye la primera vez que se necesita
    # ------------------------------------------------------------------

    @property
    def llm(self) -> BaseLLMAdapter:
        """
        Adapter del LLM configurado en settings.

        Se construye la primera vez que se accede y se reutiliza.
        Si la API key no está configurada, lanza LLMConnectionError
        con un mensaje descriptivo de cómo configurarla.

        Raises:
            LLMConnectionError: Si falta la API key o el proveedor no responde.
        """
        if self._llm_adapter is None:
            self._llm_adapter = LLMRegistry.get_adapter(
                provider=self.settings.llm_provider
            )
            logger.debug(
                "LLM adapter instanciado",
                provider=self._llm_adapter.get_provider_name(),
                model=self._llm_adapter.get_model_id(),
            )
        return self._llm_adapter

    # ------------------------------------------------------------------
    # Services — lazy, construidos bajo demanda
    # ------------------------------------------------------------------

    @property
    def context_builder(self) -> ContextBuilder:
        """
        ContextBuilder — construye el prompt de sistema para el LLM.

        Depende del ToolRegistry para incluir los schemas de las tools.
        """
        if self._context_builder is None:
            from hiperforge.application.services.context_builder import ContextBuilder
            self._context_builder = ContextBuilder(registry=self.tool_registry)
        return self._context_builder

    @property
    def tool_dispatcher(self) -> ToolDispatcher:
        """
        ToolDispatcher — resuelve y ejecuta las tools solicitadas por el LLM.

        Depende del ToolRegistry para buscar tools por nombre.
        """
        if self._tool_dispatcher is None:
            from hiperforge.application.services.tool_dispatcher import ToolDispatcher
            self._tool_dispatcher = ToolDispatcher(registry=self.tool_registry)
        return self._tool_dispatcher

    @property
    def planner(self) -> PlannerService:
        """
        PlannerService — genera el plan de subtasks con el LLM.

        Depende del LLM adapter y del store para leer preferencias.
        Acceder a esta propiedad instancia el LLM adapter si no está listo.
        """
        if self._planner is None:
            from hiperforge.application.services.planner import PlannerService
            self._planner = PlannerService(
                llm=self.llm,
                store=self.store,
            )
        return self._planner

    @property
    def executor(self) -> ExecutorService:
        """
        ExecutorService — el loop ReAct completo.

        Depende del LLM adapter, el tool dispatcher, el store y settings.
        Es el servicio más central del sistema — orquesta toda la ejecución.
        """
        if self._executor is None:
            from hiperforge.application.services.executor import ExecutorService
            self._executor = ExecutorService(
                llm=self.llm,
                tool_dispatcher=self.tool_dispatcher,
                context_builder=self.context_builder,
                store=self.store,
                settings=self.settings,
            )
        return self._executor

    # ------------------------------------------------------------------
    # Use Cases — siempre nuevos, sin cache
    #
    # Los use cases no se cachean porque son ligeros (no tienen estado
    # entre ejecuciones) y cada invocación de la CLI es un proceso nuevo.
    # Si en el futuro se necesita reutilizarlos en el mismo proceso,
    # agregar cache aquí con el mismo patrón que los services.
    # ------------------------------------------------------------------

    @property
    def run_task(self) -> RunTaskUseCase:
        """
        RunTaskUseCase — orquesta el flujo completo de una task.

        Es el use case principal de HiperForge.
        Acceder a esta propiedad instancia el LLM adapter, el planner
        y el executor si no están listos todavía.
        """
        from hiperforge.application.use_cases.run_task import RunTaskUseCase
        return RunTaskUseCase(
            planner=self.planner,
            executor=self.executor,
            store=self.store,
            locator=self.locator,
            settings=self.settings,
        )

    @property
    def create_project(self) -> CreateProjectUseCase:
        """CreateProjectUseCase — crea un nuevo proyecto en el workspace activo."""
        from hiperforge.application.use_cases.create_project import CreateProjectUseCase
        return CreateProjectUseCase(store=self.store)

    @property
    def create_workspace(self) -> CreateWorkspaceUseCase:
        """CreateWorkspaceUseCase — crea un nuevo workspace."""
        from hiperforge.application.use_cases.create_workspace import CreateWorkspaceUseCase
        return CreateWorkspaceUseCase(store=self.store)

    @property
    def switch_workspace(self) -> SwitchWorkspaceUseCase:
        """SwitchWorkspaceUseCase — cambia el workspace activo."""
        from hiperforge.application.use_cases.switch_workspace import SwitchWorkspaceUseCase
        return SwitchWorkspaceUseCase(store=self.store)

    @property
    def manage_prefs(self) -> ManagePrefsUseCase:
        """ManagePrefsUseCase — lee y actualiza preferencias del sistema."""
        from hiperforge.application.use_cases.manage_prefs import ManagePrefsUseCase
        return ManagePrefsUseCase(store=self.store)

    # ------------------------------------------------------------------
    # Utilidades de diagnóstico
    # ------------------------------------------------------------------

    def check_llm_availability(self) -> bool:
        """
        Verifica que el LLM configurado está disponible y responde.

        Hace una llamada mínima al proveedor para confirmar que:
          - La API key es válida
          - El modelo existe y está disponible
          - La conexión de red funciona

        Devuelve False ante cualquier error — nunca lanza excepción.
        Usado por la CLI para mostrar un warning antes de ejecutar
        si el LLM no está disponible.

        Returns:
            True si el LLM responde correctamente.
            False si hay cualquier problema de conexión o configuración.
        """
        try:
            return self.llm.is_available()
        except Exception:
            return False

    def __repr__(self) -> str:
        llm_status = "instanciado" if self._llm_adapter else "lazy"
        return (
            f"Container("
            f"provider={self.settings.llm_provider}, "
            f"model={self.settings.effective_llm_model}, "
            f"tools={self.tool_registry.tool_count}, "
            f"llm={llm_status})"
        )