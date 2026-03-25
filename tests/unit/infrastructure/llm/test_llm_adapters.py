"""
Tests unitarios para los LLM adapters — parsing de JSON, multi-block, format_tool_result.

Cobertura:
  - Parsing de acción tool_call desde JSON limpio
  - Parsing de acción complete
  - Parsing de múltiples bloques JSON (OPTIMIZACIÓN CLAVE)
  - Parsing de texto libre (no JSON)
  - Parsing de JSON malformado
  - format_tool_result con output comprimido
  - Normalización de finish_reason
  - _extract_json_object_blocks con trailing content
"""

from __future__ import annotations

import json
import pytest
from unittest.mock import MagicMock, patch

from hiperforge.infrastructure.llm.openai import OpenAIAdapter
from hiperforge.infrastructure.llm.base import BaseLLMAdapter, RichLLMResponse, ToolCallRequest
from hiperforge.domain.value_objects.message import Message
from hiperforge.domain.value_objects.token_usage import TokenUsage


# ═══════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════

@pytest.fixture
def adapter():
    """OpenAIAdapter con API key dummy (no llamamos la API real)."""
    with patch("openai.OpenAI"):
        a = OpenAIAdapter(api_key="sk-test", model_id="gpt-4o")
        a._task_id = "test-task"
        a._subtask_id = "test-subtask"
        return a


# ═══════════════════════════════════════════════════════════════════
# PARSING DE ACCIONES JSON
# ═══════════════════════════════════════════════════════════════════

class TestParseActionFromContent:
    """Verificar _parse_action_from_content con todos los formatos."""

    def test_tool_call_simple(self, adapter):
        content = json.dumps({
            "action": "tool_call",
            "tool": "shell",
            "arguments": {"command": "ls"},
        })
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 1
        assert tool_calls[0].tool_name == "shell"
        assert tool_calls[0].arguments == {"command": "ls"}
        assert reason == "tool_use"

    def test_complete_action(self, adapter):
        content = json.dumps({
            "action": "complete",
            "summary": "Script creado exitosamente.",
        })
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0
        assert parsed == "Script creado exitosamente."
        assert reason == "complete"

    def test_think_action(self, adapter):
        content = json.dumps({
            "action": "think",
            "content": "Necesito crear el archivo primero.",
        })
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0
        assert parsed == "Necesito crear el archivo primero."
        assert reason == "stop"

    def test_texto_libre_no_json(self, adapter):
        content = "Esto es texto libre sin JSON."
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0
        assert parsed == content
        assert reason == "stop"

    def test_json_malformado(self, adapter):
        content = '{action: tool_call, broken json'
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0
        assert reason == "stop"


class TestMultiBlockParsing:
    """OPTIMIZACIÓN CLAVE: verificar que múltiples tool_calls se parsean todas."""

    def test_dos_tool_calls(self, adapter):
        """Antes: solo se usaba la primera. Ahora: ambas se retornan."""
        content = (
            json.dumps({"action": "tool_call", "tool": "file", "arguments": {"operation": "write", "path": "a.py", "content": "pass"}})
            + "\n"
            + json.dumps({"action": "tool_call", "tool": "shell", "arguments": {"command": "python -m py_compile a.py"}})
        )
        tool_calls, _, reason, _ = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 2
        assert tool_calls[0].tool_name == "file"
        assert tool_calls[1].tool_name == "shell"
        assert reason == "tool_use"

    def test_tres_tool_calls(self, adapter):
        blocks = [
            {"action": "tool_call", "tool": "file", "arguments": {"operation": "write", "path": "a.py", "content": "x"}},
            {"action": "tool_call", "tool": "file", "arguments": {"operation": "write", "path": "b.py", "content": "y"}},
            {"action": "tool_call", "tool": "shell", "arguments": {"command": "pytest"}},
        ]
        content = "\n".join(json.dumps(b) for b in blocks)
        tool_calls, _, reason, _ = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 3

    def test_tool_call_seguido_de_complete(self, adapter):
        """tool_call + complete → ejecuta tool y guarda deferred_summary."""
        content = (
            json.dumps({"action": "tool_call", "tool": "shell", "arguments": {"command": "pytest"}})
            + "\n"
            + json.dumps({"action": "complete", "summary": "Tests pasan correctamente."})
        )
        tool_calls, _, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 1
        assert reason == "tool_use"
        assert deferred == "Tests pasan correctamente."

    def test_complete_solo(self, adapter):
        content = json.dumps({"action": "complete", "summary": "Listo"})
        tool_calls, parsed, reason, deferred = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0
        assert reason == "complete"
        assert deferred is None

    def test_tool_call_sin_nombre_se_ignora(self, adapter):
        content = json.dumps({"action": "tool_call", "tool": "", "arguments": {}})
        tool_calls, _, reason, _ = adapter._parse_action_from_content(content)
        assert len(tool_calls) == 0


# ═══════════════════════════════════════════════════════════════════
# EXTRACT JSON OBJECT BLOCKS
# ═══════════════════════════════════════════════════════════════════

class TestExtractJsonBlocks:

    def test_un_bloque(self, adapter):
        content = '{"key": "value"}'
        blocks, trailing = adapter._extract_json_object_blocks(content)
        assert len(blocks) == 1
        assert blocks[0] == {"key": "value"}
        assert trailing == ""

    def test_dos_bloques(self, adapter):
        content = '{"a": 1} {"b": 2}'
        blocks, trailing = adapter._extract_json_object_blocks(content)
        assert len(blocks) == 2

    def test_trailing_content(self, adapter):
        content = '{"a": 1} extra text'
        blocks, trailing = adapter._extract_json_object_blocks(content)
        assert len(blocks) == 1
        assert "extra text" in trailing

    def test_no_json(self, adapter):
        content = "plain text"
        blocks, trailing = adapter._extract_json_object_blocks(content)
        assert len(blocks) == 0

    def test_json_malformado(self, adapter):
        content = '{"broken'
        blocks, trailing = adapter._extract_json_object_blocks(content)
        assert len(blocks) == 0


# ═══════════════════════════════════════════════════════════════════
# FORMAT TOOL RESULT
# ═══════════════════════════════════════════════════════════════════

class TestFormatToolResult:

    def test_resultado_exitoso(self, adapter):
        msg = adapter.format_tool_result(
            tool_call_id="tc-1",
            tool_name="shell",
            output="5 passed in 0.4s",
            success=True,
        )
        assert msg.role.value == "user"
        payload = json.loads(msg.content)
        assert payload["success"] is True
        assert payload["tool_name"] == "shell"

    def test_resultado_fallido(self, adapter):
        msg = adapter.format_tool_result(
            tool_call_id="tc-1",
            tool_name="shell",
            output="",
            success=False,
        )
        payload = json.loads(msg.content)
        assert payload["success"] is False

    def test_output_largo_se_comprime(self, adapter):
        long_output = "x" * 5000
        msg = adapter.format_tool_result(
            tool_call_id="tc-1",
            tool_name="file",
            output=long_output,
            success=True,
        )
        payload = json.loads(msg.content)
        assert len(payload["output"]) < 5000
        assert "truncated" in payload["output"]


# ═══════════════════════════════════════════════════════════════════
# NORMALIZE FINISH REASON
# ═══════════════════════════════════════════════════════════════════

class TestNormalizeFinishReason:

    def test_stop(self, adapter):
        assert adapter._normalize_finish_reason("stop") == "stop"

    def test_length_a_max_tokens(self, adapter):
        assert adapter._normalize_finish_reason("length") == "max_tokens"

    def test_content_filter(self, adapter):
        assert adapter._normalize_finish_reason("content_filter") == "stop"

    def test_tool_calls(self, adapter):
        assert adapter._normalize_finish_reason("tool_calls") == "tool_use"

    def test_desconocido_es_stop(self, adapter):
        assert adapter._normalize_finish_reason("unknown_value") == "stop"


# ═══════════════════════════════════════════════════════════════════
# RICH LLM RESPONSE
# ═══════════════════════════════════════════════════════════════════

class TestRichLLMResponse:

    def test_has_tool_calls(self):
        r = RichLLMResponse(
            content="",
            tool_calls=[ToolCallRequest("t1", "shell", {"command": "ls"})],
            token_usage=TokenUsage.zero(),
            model="test",
        )
        assert r.has_tool_calls
        assert not r.has_content

    def test_has_content(self):
        r = RichLLMResponse(
            content="pensando...",
            tool_calls=[],
            token_usage=TokenUsage.zero(),
            model="test",
        )
        assert r.has_content
        assert not r.has_tool_calls

    def test_was_truncated(self):
        r = RichLLMResponse(
            content="", tool_calls=[], token_usage=TokenUsage.zero(),
            model="test", finish_reason="max_tokens",
        )
        assert r.was_truncated

    def test_not_truncated(self):
        r = RichLLMResponse(
            content="", tool_calls=[], token_usage=TokenUsage.zero(),
            model="test", finish_reason="stop",
        )
        assert not r.was_truncated
