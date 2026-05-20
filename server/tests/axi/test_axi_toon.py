from __future__ import annotations

from engram.axi.toon import render_toon


def test_render_toon_uses_compact_table_for_uniform_objects() -> None:
    output = render_toon(
        {
            "status": "healthy",
            "next": [
                {"cmd": "engram axi context", "reason": "Load context"},
                {"cmd": "engram axi recall query", "reason": "Search memory"},
            ],
        }
    )

    assert "status: healthy" in output
    assert "next[2]{cmd,reason}:" in output
    assert "engram axi context,Load context" in output


def test_render_toon_quotes_special_scalars() -> None:
    output = render_toon({"cmd": 'engram axi recall "query"', "text": "a:b"})

    assert 'cmd: "engram axi recall \\"query\\""' in output
    assert 'text: "a:b"' in output

