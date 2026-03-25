"""
Tests unitarios para utilidades del core — ids, hashing, datetime, retry.
"""

from __future__ import annotations

import time
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from hiperforge.core.utils.ids import generate_id, is_valid_id
from hiperforge.core.utils.hashing import checksum_file, hash_str, checksum_bytes
from hiperforge.core.utils.datetime import utcnow, parse_iso, format_duration, seconds_since
from hiperforge.core.utils.retry import retry_call


# ═══════════════════════════════════════════════════════════════════
# IDS
# ═══════════════════════════════════════════════════════════════════

class TestGenerateId:

    def test_genera_string_no_vacio(self):
        assert len(generate_id()) > 0

    def test_ids_son_unicos(self):
        ids = {generate_id() for _ in range(100)}
        assert len(ids) == 100

    def test_id_generado_es_valido(self):
        assert is_valid_id(generate_id())

    def test_string_vacio_no_es_valido(self):
        assert not is_valid_id("")

    def test_string_aleatorio_no_es_valido(self):
        assert not is_valid_id("esto-no-es-un-id-real")


# ═══════════════════════════════════════════════════════════════════
# HASHING
# ═══════════════════════════════════════════════════════════════════

class TestHashing:

    def test_checksum_file_determinista(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("contenido fijo")
        h1 = checksum_file(f)
        h2 = checksum_file(f)
        assert h1 == h2

    def test_checksum_file_hex_64_chars(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("data")
        h = checksum_file(f)
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)

    def test_checksum_file_cambia_con_contenido(self, tmp_dir):
        f = tmp_dir / "test.txt"
        f.write_text("v1")
        h1 = checksum_file(f)
        f.write_text("v2")
        h2 = checksum_file(f)
        assert h1 != h2

    def test_checksum_file_no_existente(self, tmp_dir):
        with pytest.raises(FileNotFoundError):
            checksum_file(tmp_dir / "noexiste.txt")

    def test_hash_str_determinista(self):
        assert hash_str("hola") == hash_str("hola")

    def test_hash_str_diferente_input(self):
        assert hash_str("a") != hash_str("b")

    def test_checksum_bytes(self):
        h = checksum_bytes(b"hello world")
        assert len(h) == 64


# ═══════════════════════════════════════════════════════════════════
# DATETIME
# ═══════════════════════════════════════════════════════════════════

class TestDatetime:

    def test_utcnow_tiene_timezone(self):
        now = utcnow()
        assert now.tzinfo is not None

    def test_parse_iso_con_z(self):
        dt = parse_iso("2024-01-15T14:32:01Z")
        assert dt.year == 2024
        assert dt.month == 1
        assert dt.tzinfo is not None

    def test_parse_iso_con_offset(self):
        dt = parse_iso("2024-01-15T14:32:01+00:00")
        assert dt.year == 2024

    def test_parse_iso_sin_timezone_asume_utc(self):
        dt = parse_iso("2024-01-15T14:32:01")
        assert dt.tzinfo is not None

    def test_parse_iso_string_vacio_error(self):
        with pytest.raises(ValueError):
            parse_iso("")

    def test_format_duration_segundos(self):
        result = format_duration(45.3)
        assert "45" in result
        assert "s" in result

    def test_format_duration_minutos(self):
        result = format_duration(125.0)
        assert "2m" in result

    def test_seconds_since(self):
        start = time.monotonic()
        time.sleep(0.01)
        elapsed = seconds_since(start)
        assert elapsed >= 0.01


# ═══════════════════════════════════════════════════════════════════
# RETRY
# ═══════════════════════════════════════════════════════════════════

class TestRetryCall:

    def test_exito_al_primer_intento(self):
        fn = MagicMock(return_value="ok")
        result = retry_call(fn=fn, max_attempts=3, retryable_exceptions=(ValueError,))
        assert result == "ok"
        fn.assert_called_once()

    def test_reintenta_ante_error_retryable(self):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        result = retry_call(
            fn=fn,
            max_attempts=3,
            retryable_exceptions=(ValueError,),
            base_delay=0.01,
        )
        assert result == "ok"
        assert fn.call_count == 2

    def test_error_no_retryable_se_propaga(self):
        fn = MagicMock(side_effect=TypeError("wrong type"))
        with pytest.raises(TypeError):
            retry_call(fn=fn, max_attempts=3, retryable_exceptions=(ValueError,))
        fn.assert_called_once()

    def test_agota_intentos_propaga_ultimo_error(self):
        fn = MagicMock(side_effect=ValueError("fail"))
        with pytest.raises(ValueError, match="fail"):
            retry_call(
                fn=fn,
                max_attempts=3,
                retryable_exceptions=(ValueError,),
                base_delay=0.01,
            )
        assert fn.call_count == 3

    def test_callback_on_retry(self):
        fn = MagicMock(side_effect=[ValueError("fail"), "ok"])
        on_retry = MagicMock()
        retry_call(
            fn=fn,
            max_attempts=3,
            retryable_exceptions=(ValueError,),
            on_retry=on_retry,
            base_delay=0.01,
        )
        on_retry.assert_called_once()
