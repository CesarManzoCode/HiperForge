"""
Paquete de tools de HiperForge.

La importación de este paquete registra automáticamente todas las tools
en el ToolRegistry via el decorador @register_tool.

El orden de importación no importa — cada tool se registra independientemente.
"""

# Importamos todas las tools para que @register_tool las registre automáticamente.
# Basta con importar el módulo — el decorador hace el registro al definir la clase.
from hiperforge.tools import shell        # noqa: F401  ShellTool
from hiperforge.tools import file_ops     # noqa: F401  FileTool
from hiperforge.tools import git          # noqa: F401  GitTool
from hiperforge.tools import web          # noqa: F401  WebTool
from hiperforge.tools import code_analysis  # noqa: F401  CodeAnalysisTool

__all__ = ["shell", "file_ops", "git", "web", "code_analysis"]