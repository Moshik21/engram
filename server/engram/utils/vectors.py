"""Vector utility functions (pack/unpack/cosine)."""

from __future__ import annotations

import struct

import numpy as np


def pack_vector(vec: list[float]) -> bytes:
    """Pack a float vector into a compact binary BLOB (4 bytes per dimension)."""
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes, dim: int) -> list[float]:
    """Unpack a binary BLOB back into a float vector."""
    return list(struct.unpack(f"{dim}f", blob))


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Compute cosine similarity between two vectors using numpy."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))
