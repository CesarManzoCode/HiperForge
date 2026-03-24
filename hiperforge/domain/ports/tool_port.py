"""
Port: ToolPort

Define el contrato que cada herramienta del agente debe cumplir
para poder ser usada en el loop ReAct de HiperForge.

¿Cómo encajan las tools en el loop ReAct?
  En cada iteración del loop, el LLM decide si necesita ejecutar
  una acción para avanzar. Esa acción es una tool call.

  Razonar  →  "Necesito instalar dependencias, usaré ShellTool"
  Actuar   →  tool_dispatcher llama shell.execute({"command": "pip install ..."})
  Observar →  el resultado vuelve al LLM como contexto para la siguiente iteración

  ToolPort define cómo cada tool expone:
    - execute(): cómo se ejecuta
    - schema():  cómo el LLM sabe que existe y qué argumentos acepta

¿Qué es el schema?
  El schema es la descripción de la tool en un formato que el LLM entiende.
  Le dice al modelo: "existe esta tool, sirve para esto, y acepta estos argumentos".
  Sin el schema, el LLM no puede decidir cuándo usarla.

  El formato del schema es compatible con el function calling de OpenAI
  y el tool use de Anthropic — es el estándar de facto de la industria.

IMPLEMENTACIONES ESPERADAS:
  ShellTool      →  ejecuta comandos en el sistema operativo
  FileTool       →  lee, escribe y lista archivos
  GitTool        →  operaciones de git (status, diff, commit)
  WebTool        →  hace requests HTTP, busca en la web
  CodeTool       →  análisis estático, búsqueda de símbolos en el código

USO TÍPICO (desde tool_dispatcher.py):
  class ToolDispatcher:
      def __init__(self, tools: list[ToolPort]) -> None:
          self._registry = {tool.name: tool for tool in tools}

      def dispatch(self, tool_name: str, arguments: dict) -> ToolResult:
          tool = self._registry[tool_name]
          return tool.execute(arguments)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from hiperforge.domain.entities.tool_call import ToolResult


@dataclass(frozen=True)
class ToolSchema:
    """
    Descripción de una tool en formato entendible por el LLM.

    Este es el "menú" que el agente le muestra al LLM al inicio
    de cada iteración del loop ReAct. El LLM lee los schemas
    disponibles y decide cuál usar (o si necesita usar alguna).

    Atributos:
        name:        Identificador único de la tool. Debe ser snake_case.
                     Ejemplo: "shell", "file_read", "git_status".
        description: Explicación clara de qué hace la tool y cuándo usarla.
                     El LLM toma decisiones basándose en esta descripción —
                     una descripción mala lleva a un agente que usa las tools
                     incorrectamente.
        parameters:  JSON Schema de los argumentos que acepta la tool.
                     El LLM usa esto para saber exactamente qué argumentos
                     pasar al llamar la tool.

    Ejemplo de schema completo para ShellTool:
        ToolSchema(
            name="shell",
            description="Ejecuta un comando en la terminal del sistema. "
                        "Usa esta tool para instalar dependencias, correr tests, "
                        "compilar código o cualquier operación que requiera la terminal.",
            parameters={
                "type": "object",
                "properties": {
                    "command": {
                        "type": "string",
                        "description": "El comando a ejecutar. Ejemplo: 'pytest tests/ -v'"
                    },
                    "timeout": {
                        "type": "number",
                        "description": "Tiempo máximo en segundos. Default 30.",
                    }
                },
                "required": ["command"]
            }
        )
    """

    name: str
    description: str
    parameters: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        """
        Serializa al formato estándar de tool/function calling.

        Compatible con:
          - Anthropic tool use:  {"name": ..., "description": ..., "input_schema": ...}
          - OpenAI function calling: {"name": ..., "description": ..., "parameters": ...}

        Los adapters de LLM ajustan este dict a su formato específico.
        """
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }


class ToolPort(ABC):
    """
    Contrato abstracto para todas las herramientas del agente.

    Cada tool implementa dos responsabilidades:
      1. Describirse al LLM (schema)
      2. Ejecutarse cuando el LLM la solicita (execute)

    La propiedad `name` es el identificador que conecta ambas —
    el LLM solicita la tool por nombre, y el ToolRegistry la busca por nombre.
    """

    # ------------------------------------------------------------------
    # Identidad de la tool — debe ser consistente entre schema() y el registry
    # ------------------------------------------------------------------

    @property
    @abstractmethod
    def name(self) -> str:
        """
        Nombre único de la tool en snake_case.

        IMPORTANTE: Este nombre debe coincidir exactamente con el `name`
        devuelto por schema(). El LLM solicita tools por este nombre —
        si hay discrepancia, el tool_dispatcher no podrá resolver la tool.

        Ejemplos válidos: "shell", "file_read", "git_commit", "web_fetch"
        """
        ...

    @property
    @abstractmethod
    def description(self) -> str:
        """
        Descripción corta de la tool (1-2 líneas).

        Usada internamente para logs y para construir el schema.
        La descripción detallada para el LLM va dentro de schema().
        """
        ...

    # ------------------------------------------------------------------
    # Contrato principal
    # ------------------------------------------------------------------

    @abstractmethod
    def execute(self, arguments: dict[str, Any]) -> ToolResult:
        """
        Ejecuta la tool con los argumentos que el LLM proporcionó.

        Este método es el "Actuar" del loop ReAct — aquí ocurre
        la acción real en el mundo (escribir un archivo, correr un
        comando, hacer un request HTTP, etc.).

        Parámetros:
            arguments: Diccionario con los argumentos del LLM.
                       El formato debe coincidir con el schema() de la tool.
                       Ejemplo para ShellTool: {"command": "pytest tests/"}

        Returns:
            ToolResult.success() si la tool terminó sin errores.
            ToolResult.failure() si ocurrió algún error durante la ejecución.

            IMPORTANTE: Este método NUNCA debe lanzar excepciones al caller.
            Cualquier error interno debe capturarse y devolverse como
            ToolResult.failure(). Esto permite al agente ReAct observar
            el error y decidir cómo recuperarse, en vez de que el error
            rompa el loop.

        Raises:
            ToolTimeoutError: Única excepción permitida — cuando la tool
                              supera el tiempo máximo configurado. El executor
                              la maneja para interrumpir el loop ReAct.
        """
        ...

    @abstractmethod
    def schema(self) -> ToolSchema:
        """
        Devuelve la descripción de esta tool para el LLM.

        Este schema se incluye en cada llamada al LLM durante el loop ReAct
        para que el modelo sepa qué tools tiene disponibles.

        La calidad de la descripción y los parámetros impacta directamente
        en qué tan bien el agente usa la tool — invertir tiempo aquí vale la pena.

        Returns:
            ToolSchema con nombre, descripción detallada y JSON Schema
            de los parámetros aceptados.
        """
        ...

    # ------------------------------------------------------------------
    # Métodos con implementación por defecto
    # ------------------------------------------------------------------

    def validate_arguments(self, arguments: dict[str, Any]) -> list[str]:
        """
        Valida los argumentos antes de ejecutar la tool.

        Devuelve lista de errores encontrados. Lista vacía = argumentos válidos.

        Implementación por defecto: verifica que los campos `required`
        del schema estén presentes en los argumentos.

        Los adapters pueden sobreescribir este método para validaciones
        más específicas (tipos, rangos, formatos, etc.).

        Usado por el executor antes de llamar execute() para dar feedback
        claro al LLM cuando envía argumentos incorrectos.
        """
        errors = []
        schema = self.schema()
        required_fields = schema.parameters.get("required", [])

        for field in required_fields:
            if field not in arguments:
                errors.append(f"Campo requerido '{field}' no encontrado en argumentos")

        return errors

    def is_safe_to_run(self, arguments: dict[str, Any]) -> bool:
        """
        Verifica si es seguro ejecutar la tool con estos argumentos.

        Implementación por defecto: siempre True.

        Las tools que realizan operaciones destructivas (eliminar archivos,
        ejecutar comandos con privilegios elevados, etc.) deben sobreescribir
        este método con sus propias verificaciones de seguridad.

        Usado por el executor para pedir confirmación al usuario antes
        de ejecutar operaciones potencialmente peligrosas.
        """
        return True

    def __repr__(self) -> str:
        """
        Ejemplo: ToolPort(name='shell', description='Ejecuta comandos...')
        """
        return f"ToolPort(name={self.name!r}, description={self.description[:50]!r})"
