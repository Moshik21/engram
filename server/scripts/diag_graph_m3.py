#!/usr/bin/env python3
"""Diagnose why graph-ON failed m3 (Japan->Priya->Director). Inspects the actual
graph state in the /tmp/cd-on native store for group longmemeval_m3."""

from __future__ import annotations

import asyncio


async def main() -> None:
    from engram.config import EngramConfig
    from engram.storage.factory import create_stores
    from engram.storage.resolver import resolve_mode

    cfg = EngramConfig()
    cfg.mode = "helix"
    cfg.helix.transport = "native"
    cfg.helix.data_dir = "/tmp/cd-on"
    cfg.embedding.provider = "local"

    mode = await resolve_mode(cfg.mode)
    graph, activation, search = create_stores(mode, cfg)
    await graph.initialize()
    await search.initialize()

    gid = "longmemeval_m3"
    ents = await graph.find_entities(group_id=gid, limit=300)
    print(f"=== group {gid}: {len(ents)} entities ===")
    for e in sorted(ents, key=lambda x: x.name.lower()):
        print(f"  {e.name!r:42} type={e.entity_type:12} id={e.id[:12]}")

    # Look for Priya / Director / Japan entities
    def match(sub):
        return [e for e in ents if sub.lower() in e.name.lower()]

    for key in ("priya", "director", "japan", "nimbus", "helix"):
        hits = match(key)
        print(f"\n-- {key!r}: {len(hits)} matching entit(y/ies): {[h.name for h in hits]}")
        for h in hits:
            rels = await graph.get_relationships(h.id, direction="both", group_id=gid)
            for r in rels:
                print(
                    f"     {r.subject if hasattr(r, 'subject') else '?'} "
                    f"-[{getattr(r, 'predicate', '?')}]-> "
                    f"{getattr(r, 'object', getattr(r, 'target', None))}"
                )

    # Episodes: is the Director (s7) content present + what entities link to it?
    eps = await graph.get_episodes(group_id=gid, limit=50)
    print(f"\n=== {len(eps)} episodes; ones mentioning 'Director' or 'promoted' ===")
    for ep in eps:
        c = ep.content or ""
        if "Director" in c or "promoted" in c:
            print(f"  ep {ep.id[:12]}: {c[:90]!r}")

    close = getattr(graph, "close", None)
    if close:
        await close()


if __name__ == "__main__":
    asyncio.run(main())
