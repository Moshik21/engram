"""HelixDB storage backend for Engram — graph, vector, atlas, conversations, consolidation."""

from __future__ import annotations


def unwrap_helix_results(raw: list) -> list[dict]:
    """Unwrap Helix v2 response envelopes.

    Helix v2 wraps results in ``{"node": {...}}`` or ``{"edge": {...}}``
    envelopes, or returns named variables like ``{"entities": [...]}``.
    This flattens them into a list of plain dicts with field access.
    """
    unwrapped: list[dict] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        # Single-value envelope: {"node": {...}} or {"edge": {...}}
        if len(item) == 1:
            key = next(iter(item))
            val = item[key]
            if isinstance(val, dict):
                unwrapped.append(val)
                continue
            if isinstance(val, list):
                for v in val:
                    if isinstance(v, dict):
                        unwrapped.append(v)
                continue
        # Named return vars: {"entities": [...], "edges": [...]}
        has_lists = any(isinstance(v, list) for v in item.values())
        if has_lists:
            for v in item.values():
                if isinstance(v, list):
                    for elem in v:
                        if isinstance(elem, dict):
                            unwrapped.append(elem)
                elif isinstance(v, dict):
                    unwrapped.append(v)
            continue
        # Already flat
        unwrapped.append(item)
    return unwrapped
