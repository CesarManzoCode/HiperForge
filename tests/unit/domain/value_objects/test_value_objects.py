"""
Tests unitarios para Value Objects — Message, TokenUsage, FileRef.

Cobertura:
  - Constructores semánticos (Message.user(), Message.system(), etc.)
  - Inmutabilidad (frozen dataclass)
  - Propiedades calculadas (total_tokens, estimated_cost_usd)
  - Operaciones aritméticas (TokenUsage.__add__)
  - Serialización/deserialización (to_dict, from_dict)
  - FileRef.from_path con checksum real
"""

from __future__ import annotations

import pytest
from pathlib import Path

from hiperforge.domain.value_objects.message import Message, Role
from hiperforge.domain.value_objects.token_usage import TokenUsage
from hiperforge.domain.value_objects.file_ref import FileRef


# ═══════════════════════════════════════════════════════════════════
# MESSAGE
# ═══════════════════════════════════════════════════════════════════

class TestMessageCreation:

    def test_system_message(self):
        msg = Message.system("Eres un agente.")
        assert msg.role == Role.SYSTEM
        assert msg.content == "Eres un agente."

    def test_user_message(self):
        msg = Message.user("Crea un script")
        assert msg.role == Role.USER
        assert msg.content == "Crea un script"

    def test_assistant_message(self):
        msg = Message.assistant("Aquí está el código")
        assert msg.role == Role.ASSISTANT

    def test_message_tiene_timestamp(self):
        msg = Message.user("test")
        assert msg.created_at is not None
        assert msg.created_at.tzinfo is not None  # timezone-aware

    def test_message_con_metadata(self):
        msg = Message.user("test", tool_call_id="tc-1", tokens=100)
        assert msg.meta["tool_call_id"] == "tc-1"
        assert msg.meta["tokens"] == 100

    def test_message_is_empty(self):
        assert Message.user("").is_empty
        assert Message.user("   ").is_empty
        assert not Message.user("hola").is_empty


class TestMessageSerialization:

    def test_to_dict(self):
        msg = Message.user("hola")
        d = msg.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hola"

    def test_from_dict_roundtrip(self):
        original = Message.assistant("respuesta")
        d = original.to_dict()
        restored = Message.from_dict(d)
        assert restored.role == original.role
        assert restored.content == original.content


class TestMessageImmutability:

    def test_frozen_no_se_puede_mutar(self):
        msg = Message.user("test")
        with pytest.raises(AttributeError):
            msg.content = "otro"  # type: ignore


# ═══════════════════════════════════════════════════════════════════
# TOKEN USAGE
# ═══════════════════════════════════════════════════════════════════

class TestTokenUsage:

    def test_total_tokens(self):
        usage = TokenUsage(input_tokens=500, output_tokens=200)
        assert usage.total_tokens == 700

    def test_zero(self):
        zero = TokenUsage.zero()
        assert zero.total_tokens == 0
        assert zero.input_tokens == 0
        assert zero.output_tokens == 0

    def test_addition(self):
        a = TokenUsage(input_tokens=100, output_tokens=50, model="gpt-4o")
        b = TokenUsage(input_tokens=200, output_tokens=100, model="gpt-4o")
        total = a + b
        assert total.input_tokens == 300
        assert total.output_tokens == 150

    def test_add_zero(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50)
        total = usage + TokenUsage.zero()
        assert total.total_tokens == 150

    def test_estimated_cost_modelo_conocido(self):
        usage = TokenUsage(input_tokens=1_000_000, output_tokens=0, model="gpt-4o")
        assert usage.estimated_cost_usd > 0

    def test_estimated_cost_modelo_desconocido(self):
        usage = TokenUsage(input_tokens=100, output_tokens=50, model="modelo-inventado")
        assert usage.estimated_cost_usd == 0.0

    def test_estimated_cost_ollama_gratis(self):
        usage = TokenUsage(input_tokens=5000, output_tokens=2000, model="llama3")
        assert usage.estimated_cost_usd == 0.0

    def test_serialization_roundtrip(self):
        original = TokenUsage(input_tokens=500, output_tokens=200, model="gpt-4o")
        d = original.to_dict()
        restored = TokenUsage.from_dict(d)
        assert restored.input_tokens == original.input_tokens
        assert restored.output_tokens == original.output_tokens
        assert restored.model == original.model


# ═══════════════════════════════════════════════════════════════════
# FILE REF
# ═══════════════════════════════════════════════════════════════════

class TestFileRef:

    def test_from_path_archivo_existente(self, tmp_dir):
        f = tmp_dir / "test.py"
        f.write_text("print('hello')")
        ref = FileRef.from_path(f)
        assert ref.path == f.resolve()
        assert ref.checksum is not None
        assert ref.size_bytes == len("print('hello')")

    def test_from_path_archivo_inexistente(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            FileRef.from_path(tmp_dir / "no_existe.py")

    def test_from_path_detecta_mime(self, tmp_dir):
        f = tmp_dir / "script.py"
        f.write_text("pass")
        ref = FileRef.from_path(f)
        assert ref.mime_type is not None
        assert "python" in ref.mime_type

    def test_from_path_sin_checksum(self, tmp_dir):
        f = tmp_dir / "big.txt"
        f.write_text("data" * 1000)
        ref = FileRef.from_path(f, compute_checksum=False)
        assert ref.checksum is None
        assert ref.size_bytes is not None

    def test_checksum_determinista(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("contenido fijo")
        ref1 = FileRef.from_path(f)
        ref2 = FileRef.from_path(f)
        assert ref1.checksum == ref2.checksum

    def test_checksum_cambia_con_contenido(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("version 1")
        ref1 = FileRef.from_path(f)
        f.write_text("version 2")
        ref2 = FileRef.from_path(f)
        assert ref1.checksum != ref2.checksum
