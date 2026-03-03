"""AES-256-GCM field-level encryption with per-tenant key derivation."""

from __future__ import annotations

import base64
import os

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.hkdf import HKDF

ENCRYPTED_PREFIX = "enc::"


class FieldEncryptor:
    """Encrypts/decrypts individual fields using AES-256-GCM.

    Each tenant gets a derived key via HKDF-SHA256 so that
    ciphertext from one tenant cannot be decrypted by another.
    """

    def __init__(self, master_key_hex: str) -> None:
        key_bytes = bytes.fromhex(master_key_hex)
        if len(key_bytes) != 32:
            raise ValueError(
                f"Master key must be 32 bytes (64 hex chars), got {len(key_bytes)} bytes"
            )
        self._master_key = key_bytes

    def _derive_tenant_key(self, group_id: str) -> bytes:
        """Derive a per-tenant 256-bit key using HKDF-SHA256."""
        hkdf = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=group_id.encode("utf-8"),
        )
        return hkdf.derive(self._master_key)

    def encrypt(self, group_id: str, plaintext: str) -> str:
        """Encrypt plaintext → 'enc::<base64(nonce || ciphertext)>'."""
        if not plaintext:
            return plaintext
        key = self._derive_tenant_key(group_id)
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)  # 96-bit nonce for GCM
        ciphertext = aesgcm.encrypt(nonce, plaintext.encode("utf-8"), None)
        combined = nonce + ciphertext
        encoded = base64.b64encode(combined).decode("ascii")
        return f"{ENCRYPTED_PREFIX}{encoded}"

    def decrypt(self, group_id: str, data: str) -> str:
        """Decrypt 'enc::...' → plaintext. Passes through non-encrypted data."""
        if not data or not self.is_encrypted(data):
            return data
        encoded = data[len(ENCRYPTED_PREFIX) :]
        combined = base64.b64decode(encoded)
        nonce = combined[:12]
        ciphertext = combined[12:]
        key = self._derive_tenant_key(group_id)
        aesgcm = AESGCM(key)
        plaintext = aesgcm.decrypt(nonce, ciphertext, None)
        return plaintext.decode("utf-8")

    @staticmethod
    def is_encrypted(data: str) -> bool:
        """Check if a string is encrypted (has the enc:: prefix)."""
        return isinstance(data, str) and data.startswith(ENCRYPTED_PREFIX)
