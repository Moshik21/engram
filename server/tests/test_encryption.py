"""Tests for AES-256-GCM field encryption."""

import os

import pytest

from engram.security.encryption import FieldEncryptor


def _hex_key(n: int = 32) -> str:
    return os.urandom(n).hex()


class TestFieldEncryptor:
    def test_roundtrip(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        plaintext = "Hello, World!"
        ciphertext = enc.encrypt("tenant_a", plaintext)
        assert ciphertext != plaintext
        assert enc.is_encrypted(ciphertext)
        result = enc.decrypt("tenant_a", ciphertext)
        assert result == plaintext

    def test_different_tenants_different_ciphertext(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        ct_a = enc.encrypt("tenant_a", "same text")
        ct_b = enc.encrypt("tenant_b", "same text")
        assert ct_a != ct_b

    def test_wrong_tenant_cannot_decrypt(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        ct = enc.encrypt("tenant_a", "secret")
        with pytest.raises(Exception):
            enc.decrypt("tenant_b", ct)

    def test_plaintext_passthrough(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        # Non-encrypted data passes through
        assert enc.decrypt("any", "just plain text") == "just plain text"

    def test_empty_string(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        # Empty string is not encrypted
        assert enc.encrypt("t", "") == ""
        assert enc.decrypt("t", "") == ""

    def test_none_passthrough(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        assert enc.encrypt("t", None) is None
        assert enc.decrypt("t", None) is None

    def test_prefix_check(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        ct = enc.encrypt("t", "data")
        assert ct.startswith("enc::")
        assert not FieldEncryptor.is_encrypted("plain text")
        assert FieldEncryptor.is_encrypted("enc::abc123")

    def test_invalid_key_length(self):
        with pytest.raises(ValueError, match="32 bytes"):
            FieldEncryptor("abcd")  # Too short

    def test_unicode_roundtrip(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        text = "こんにちは世界 🌍 café"
        ct = enc.encrypt("t", text)
        assert enc.decrypt("t", ct) == text

    def test_long_text_roundtrip(self):
        key = _hex_key()
        enc = FieldEncryptor(key)
        text = "x" * 10000
        ct = enc.encrypt("t", text)
        assert enc.decrypt("t", ct) == text
