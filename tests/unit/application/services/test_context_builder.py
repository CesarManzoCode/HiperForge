"""
Tests unitarios para ContextBuilder — prompt de sistema, tools, truncado.

Cobertura:
  - Construcción del prompt de sistema con todas las secciones
  - Sección de tools correctamente formateada
  - Truncado de historial cuando excede context window
  - Preservación del mensaje de sistema durante truncado
  - Integración de contexto de subtask anterior
"""

from __future__ import annotations

import pytest
from unittest.mock import MagicMock

from hiperforge.application.services.context_builder import ContextBuilder
from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.tools.base import ToolRegistry


@pytest.fixture
def mock_registry():
    """ToolRegistry con schemas mockeados."""
    registry = MagicMock(spec=ToolRegistry)
    registry.get_schemas.return_value = [
        MagicMock(
            name="shell",
            description="Ejecuta comandos del sistema operativo.",
            parameters={
                "properties": {
                    "command": {"type": "string", "description": "Comando a ejecutar"},
                    "timeout": {"type": "number", "description": "Timeout en segundos"},
                },
                "required": ["command"],
            },
        ),
        MagicMock(
            name="file",
            description="Operaciones sobre archivos del proyecto.",
            parameters={
                "properties": {
                    "operation": {"type": "string", "description": "read, write, list, etc."},
                    "path": {"type": "string", "description": "Ruta del archivo"},
                    "content": {"type": "string", "description": "Contenido a escribir"},
                },
                "required": ["operation", "path"],
            },
        ),
    ]
    return registry


@pytest.fixture
def builder(mock_registry) -> ContextBuilder:
    return ContextBuilder(registry=mock_registry)


# ═══════════════════════════════════════════════════════════════════
# CONSTRUCCIÓN DEL PROMPT DE SISTEMA
# ═══════════════════════════════════════════════════════════════════

class TestBuildSystemMessage:

    def test_contiene_protocolo_json(self, builder):
        msg = builder.build_system_message(
            subtask_description="Crear report.py",
            task_prompt="crea un script",
        )
        assert "tool_call" in msg.content
        assert "complete" in msg.content

    def test_contiene_nombre_herramientas(self, builder):
        msg = builder.build_system_message(
            subtask_description="Crear report.py",
            task_prompt="crea un script",
        )
        assert "shell" in msg.content
        assert "file" in msg.content

    def test_contiene_contexto_subtask(self, builder):
        msg = builder.build_system_message(
            subtask_description="Crear report.py con csv.DictReader",
            task_prompt="crea un script Python",
        )
        assert "Crear report.py" in msg.content
        assert "crea un script Python" in msg.content

    def test_role_es_system(self, builder):
        msg = builder.build_system_message(
            subtask_description="test",
            task_prompt="test",
        )
        assert msg.role == Role.SYSTEM

    def test_no_contiene_think_en_protocolo(self, builder):
        """El protocolo optimizado NO debe incluir action=think."""
        msg = builder.build_system_message(
            subtask_description="test",
            task_prompt="test",
        )
        # Verificar que think NO está en la sección PROTOCOLO
        lines = msg.content.split("\n")
        protocol_section = False
        for line in lines:
            if "PROTOCOLO" in line:
                protocol_section = True
            if protocol_section and "REGLAS" in line:
                protocol_section = False
            if protocol_section and "think" in line.lower():
                pytest.fail("'think' encontrado en sección PROTOCOLO del system prompt")

    def test_contiene_regla_anti_relectura(self, builder):
        msg = builder.build_system_message(
            subtask_description="test",
            task_prompt="test",
        )
        assert "NUNCA releas" in msg.content or "releas" in msg.content.lower()

    def test_working_dir_incluido(self, builder):
        msg = builder.build_system_message(
            subtask_description="test",
            task_prompt="test",
            working_dir="/home/user/proyecto",
        )
        assert "/home/user/proyecto" in msg.content


class TestBuildSystemMessageWithPreviousContext:
    """Verificar que previous_subtask_summary se integra correctamente."""

    def test_sin_contexto_anterior(self, builder):
        msg = builder.build_system_message(
            subtask_description="Paso 2",
            task_prompt="tarea compleja",
            previous_subtask_summary=None,
        )
        assert "Paso anterior" not in msg.content

    def test_con_contexto_anterior(self, builder):
        msg = builder.build_system_message(
            subtask_description="Paso 2: escribir tests",
            task_prompt="tarea compleja",
            previous_subtask_summary="Creé el módulo auth.py con JWT middleware",
        )
        assert "auth.py" in msg.content or "Paso anterior" in msg.content


# ═══════════════════════════════════════════════════════════════════
# SECCIÓN DE TOOLS
# ═══════════════════════════════════════════════════════════════════

class TestToolsSection:

    def test_muestra_params_requeridos(self, builder):
        msg = builder.build_system_message(
            subtask_description="test",
            task_prompt="test",
        )
        assert "command" in msg.content
        assert "operation" in msg.content
        assert "path" in msg.content

    def test_sin_tools_no_crashea(self):
        empty_registry = MagicMock(spec=ToolRegistry)
        empty_registry.get_schemas.return_value = []
        b = ContextBuilder(registry=empty_registry)
        msg = b.build_system_message(subtask_description="test", task_prompt="test")
        assert "PROTOCOLO" in msg.content


# ═══════════════════════════════════════════════════════════════════
# TRUNCADO DE HISTORIAL
# ═══════════════════════════════════════════════════════════════════

class TestTruncateMessages:

    def test_historial_corto_no_se_trunca(self, builder):
        messages = [
            Message.system("instrucciones cortas"),
            Message.user("hola"),
            Message.assistant("respuesta"),
        ]
        result = builder.truncate_messages_for_context_window(
            messages=messages,
            context_window_size=128_000,
            max_tokens_response=1024,
        )
        assert len(result) == 3

    def test_sistema_siempre_se_preserva(self, builder):
        messages = [
            Message.system("instrucciones críticas"),
        ] + [
            Message.user("x" * 10000) for _ in range(50)
        ]
        result = builder.truncate_messages_for_context_window(
            messages=messages,
            context_window_size=1000,
            max_tokens_response=200,
        )
        assert result[0].role == Role.SYSTEM
        assert result[0].content == "instrucciones críticas"

    def test_mensajes_recientes_se_preservan(self, builder):
        messages = [
            Message.system("system"),
            Message.user("antiguo 1" + "x" * 5000),
            Message.assistant("antiguo 2" + "x" * 5000),
            Message.user("antiguo 3" + "x" * 5000),
            Message.assistant("antiguo 4" + "x" * 5000),
            Message.user("reciente 1"),
            Message.assistant("reciente 2"),
            Message.user("reciente 3"),
            Message.assistant("reciente 4"),
        ]
        result = builder.truncate_messages_for_context_window(
            messages=messages,
            context_window_size=2000,
            max_tokens_response=500,
        )
        # Los últimos 4 deben estar presentes
        contents = [m.content for m in result]
        assert "reciente 4" in contents[-1]

    def test_context_window_extremo_preserva_sistema(self, builder):
        messages = [
            Message.system("system prompt"),
            Message.user("query"),
        ]
        result = builder.truncate_messages_for_context_window(
            messages=messages,
            context_window_size=100,
            max_tokens_response=90,
        )
        assert any(m.role == Role.SYSTEM for m in result)
