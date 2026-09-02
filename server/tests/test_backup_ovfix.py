"""repair_overflow_headers: stale LMDB overflow-head headers are found and fixed."""

from __future__ import annotations

import struct

from engram.backup_cli import _LMDB_PAGE_SIZE, repair_overflow_headers

P_LEAF, P_OVERFLOW = 0x02, 0x04


def _page(
    index: int, *, pgno: int, stamp: int, flags: int, lower: int = 1, upper: int = 0
) -> bytes:
    header = struct.pack("<QQHHHH", pgno, stamp, 0, flags, lower, upper)
    return header + bytes(_LMDB_PAGE_SIZE - len(header))


def _store(tmp_path):
    pages = [
        _page(0, pgno=0, stamp=5, flags=P_LEAF),
        _page(1, pgno=1, stamp=5, flags=P_OVERFLOW, lower=2),  # healthy head
        _page(2, pgno=608527, stamp=1565016, flags=P_OVERFLOW, lower=2),  # stale head
        _page(3, pgno=99, stamp=7, flags=P_LEAF),  # mis-numbered leaf: not ours to touch
        _page(4, pgno=0, stamp=0, flags=0),  # continuation / raw data
    ]
    path = tmp_path / "data.mdb"
    path.write_bytes(b"".join(pages))
    return path


def test_report_lists_only_stale_overflow_heads(tmp_path) -> None:
    path = _store(tmp_path)
    before = path.read_bytes()
    stale = repair_overflow_headers(path, apply=False)
    assert stale == [(2, 608527, 1565016)]
    assert path.read_bytes() == before, "report mode must not write"


def test_apply_rewrites_pgno_and_stamp_and_nothing_else(tmp_path) -> None:
    path = _store(tmp_path)
    before = path.read_bytes()
    assert repair_overflow_headers(path, apply=True) == [(2, 608527, 1565016)]
    after = path.read_bytes()
    pgno, stamp, _pad, flags, lower, upper = struct.unpack_from(
        "<QQHHHH", after, 2 * _LMDB_PAGE_SIZE
    )
    assert (pgno, stamp, flags, lower, upper) == (2, 1, P_OVERFLOW, 2, 0)
    # every other byte is untouched
    off = 2 * _LMDB_PAGE_SIZE
    assert after[:off] == before[:off] and after[off + 16 :] == before[off + 16 :]
    assert repair_overflow_headers(path, apply=False) == [], "idempotent: nothing left to fix"
