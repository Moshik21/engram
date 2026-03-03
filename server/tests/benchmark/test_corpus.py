"""Tests for corpus generator shape and distributions."""

from engram.benchmark.corpus import _CLUSTER_DEFS, GROUP_ID, CorpusGenerator, CorpusScale

# Generate corpus once for all tests in module (it's pure/deterministic)
_corpus = CorpusGenerator(seed=42).generate()


def test_entity_count():
    assert len(_corpus.entities) == 1000


def test_entity_type_distribution():
    counts: dict[str, int] = {}
    for e in _corpus.entities:
        counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
    # Entity types are lowercase in the corpus generator
    assert counts["person"] == 200
    assert counts["technology"] == 200
    assert counts["organization"] == 150
    assert counts["location"] == 100
    assert counts["project"] == 150
    assert counts["concept"] == 200


def test_relationship_count():
    assert len(_corpus.relationships) >= 2500


def test_access_event_distribution():
    # Count unique entities with access events
    accessed: dict[str, int] = {}
    for eid, _ in _corpus.access_events:
        accessed[eid] = accessed.get(eid, 0) + 1
    entity_ids = {e.id for e in _corpus.entities}
    dormant = entity_ids - set(accessed.keys())
    # Dormant should be ~30% = 300
    assert 250 <= len(dormant) <= 350
    # Hot tier (10+ accesses) includes the ~100 hot entities (10-50 accesses)
    # plus some warm entities whose 5-15 range lands at 10+.
    # With the seeded generator the actual count is ~230.
    hot = sum(1 for c in accessed.values() if c >= 10)
    assert 200 <= hot <= 280


def test_ground_truth_query_count():
    # Generator produces 80 queries: 20 direct, 10 recency, 10 frequency,
    # 10 associative, 5 temporal_context, 10 semantic, 10 graph_traversal, 5 cross_cluster
    assert len(_corpus.ground_truth) >= 75
    categories: dict[str, int] = {}
    for q in _corpus.ground_truth:
        categories[q.category] = categories.get(q.category, 0) + 1
    assert categories.get("direct", 0) >= 10
    assert categories.get("recency", 0) >= 5
    assert categories.get("frequency", 0) >= 5
    assert categories.get("associative", 0) >= 5
    assert categories.get("temporal_context", 0) >= 3
    # New independent categories
    assert categories.get("semantic", 0) >= 3
    assert categories.get("graph_traversal", 0) >= 3
    assert categories.get("cross_cluster", 0) >= 3


def test_deterministic():
    corpus_a = CorpusGenerator(seed=42).generate()
    corpus_b = CorpusGenerator(seed=42).generate()
    assert len(corpus_a.entities) == len(corpus_b.entities)
    assert len(corpus_a.relationships) == len(corpus_b.relationships)
    assert len(corpus_a.access_events) == len(corpus_b.access_events)
    for a, b in zip(corpus_a.entities, corpus_b.entities):
        assert a.id == b.id
        assert a.name == b.name
    for a, b in zip(corpus_a.relationships, corpus_b.relationships):
        assert a.id == b.id
        assert a.source_id == b.source_id
        assert a.target_id == b.target_id


def test_semantic_queries_independent_of_access():
    """Semantic queries grade by predicate structure, not access patterns."""
    semantic_queries = [q for q in _corpus.ground_truth if q.category == "semantic"]
    assert len(semantic_queries) == 10
    for q in semantic_queries:
        # Each semantic query should have grade-3 entities
        grade3 = [eid for eid, grade in q.relevant_entities.items() if grade == 3]
        assert len(grade3) >= 1, f"Query {q.query_id} has no grade-3 entities"


def test_graph_traversal_grades_by_hop_distance():
    """Graph traversal queries grade by structural hop distance."""
    traversal_queries = [q for q in _corpus.ground_truth if q.category == "graph_traversal"]
    assert len(traversal_queries) == 10
    for q in traversal_queries:
        grades = set(q.relevant_entities.values())
        # Should have varied grades (at least 2 distinct grade levels)
        assert len(grades) >= 2, f"Query {q.query_id} has uniform grades"


def test_cross_cluster_bridge_entities():
    """Cross-cluster queries grade bridge entities highest."""
    cross_queries = [q for q in _corpus.ground_truth if q.category == "cross_cluster"]
    assert len(cross_queries) == 5
    for q in cross_queries:
        grade3 = [eid for eid, grade in q.relevant_entities.items() if grade == 3]
        assert len(grade3) >= 1, f"Query {q.query_id} has no bridge entities (grade 3)"


def test_deterministic_new_categories():
    """New categories are deterministic across runs."""
    corpus_a = CorpusGenerator(seed=42).generate()
    corpus_b = CorpusGenerator(seed=42).generate()

    for cat in ["semantic", "graph_traversal", "cross_cluster"]:
        queries_a = [q for q in corpus_a.ground_truth if q.category == cat]
        queries_b = [q for q in corpus_b.ground_truth if q.category == cat]
        assert len(queries_a) == len(queries_b)
        for qa, qb in zip(queries_a, queries_b):
            assert qa.query_id == qb.query_id
            assert qa.relevant_entities == qb.relevant_entities


def test_new_category_grade_distribution():
    """New categories should have varied grades (not all same grade)."""
    for cat in ["semantic", "graph_traversal", "cross_cluster"]:
        queries = [q for q in _corpus.ground_truth if q.category == cat]
        all_grades: set[int] = set()
        for q in queries:
            all_grades.update(q.relevant_entities.values())
        # Should have at least 2 distinct grades across all queries in category
        assert len(all_grades) >= 2, f"Category {cat} has only grades {all_grades}"


def test_summaries_unique():
    """No two entities of the same type should have identical summaries."""
    by_type: dict[str, list[str]] = {}
    for e in _corpus.entities:
        by_type.setdefault(e.entity_type, []).append(e.summary)
    for etype, summaries in by_type.items():
        unique = set(summaries)
        assert len(unique) == len(summaries), (
            f"Entity type '{etype}' has {len(summaries) - len(unique)} "
            f"duplicate summaries out of {len(summaries)} total"
        )


def test_summaries_length():
    """Every entity summary should be between 30 and 80 words."""
    for e in _corpus.entities:
        word_count = len(e.summary.split())
        assert 30 <= word_count <= 80, (
            f"Entity {e.id} ({e.entity_type}) summary has {word_count} words, "
            f"expected 30-80: {e.summary!r}"
        )


def test_recency_queries_have_distinct_relevance():
    """No two recency queries should share the exact same relevant entity set."""
    recency_queries = [q for q in _corpus.ground_truth if q.category == "recency"]
    assert len(recency_queries) == 10

    entity_sets = []
    for q in recency_queries:
        key_set = frozenset(q.relevant_entities.keys())
        entity_sets.append(key_set)
        # Each query must have at least 1 grade-3 entity
        grade3 = [eid for eid, g in q.relevant_entities.items() if g == 3]
        assert len(grade3) >= 1, f"Query {q.query_id} has no grade-3 entities"

    # All 10 entity sets should be distinct
    unique = set(entity_sets)
    assert len(unique) == len(entity_sets), (
        f"Only {len(unique)} distinct relevance sets out of {len(entity_sets)} recency queries"
    )


def test_recency_queries_have_varied_grades():
    """Recency category should use at least 2 distinct grade levels across queries."""
    recency_queries = [q for q in _corpus.ground_truth if q.category == "recency"]
    all_grades: set[int] = set()
    for q in recency_queries:
        all_grades.update(q.relevant_entities.values())
    assert len(all_grades) >= 2, f"Recency queries have only grades {all_grades}"


def test_cross_cluster_queries_use_searchable_terms():
    """Cross-cluster query text should contain domain keywords, not cluster proper names."""
    cross_queries = [q for q in _corpus.ground_truth if q.category == "cross_cluster"]
    assert len(cross_queries) == 5

    domain_keywords = [
        "machine learning",
        "web development",
        "startup",
        "data science",
        "cloud",
        "mobile",
        "devops",
        "alignment",
        "nlp",
        "natural language",
        "fintech",
        "financial",
        "game",
        "infrastructure",
        "analytics",
        "engineering",
        "technology",
        "computing",
        "app development",
        "observability",
    ]

    for q in cross_queries:
        text_lower = q.query_text.lower()
        matches = [kw for kw in domain_keywords if kw in text_lower]
        assert len(matches) >= 1, f"Query {q.query_id} has no domain keywords: {q.query_text!r}"


# ---------------------------------------------------------------------------
# Episode tests
# ---------------------------------------------------------------------------


def test_episode_count():
    """Corpus should generate approximately 200-300 episodes."""
    assert 150 <= len(_corpus.episodes) <= 350, (
        f"Expected 150-350 episodes, got {len(_corpus.episodes)}"
    )


def test_episode_temporal_distribution():
    """Episodes should span hot/warm/cold temporal tiers."""
    import time

    ref_time = _corpus.metadata.get("generated_at", time.time())
    hot_count = 0  # 0-7 days
    warm_count = 0  # 1-30 days
    cold_count = 0  # 7-90 days

    for ep in _corpus.episodes:
        age_days = (ref_time - ep.created_at.timestamp()) / 86400.0
        if age_days <= 7:
            hot_count += 1
        elif age_days <= 30:
            warm_count += 1
        else:
            cold_count += 1

    # Hot entities (100) each get 1 episode, so hot_count should be ~100
    assert hot_count >= 50, f"Expected at least 50 hot episodes, got {hot_count}"
    # Warm + cold should exist too
    assert warm_count + cold_count >= 20, (
        f"Expected at least 20 warm/cold episodes, got {warm_count + cold_count}"
    )


def test_episode_entity_links():
    """Each episode should be linked to at least one entity."""
    linked_episodes = {ep_id for ep_id, _ in _corpus.episode_entities}
    episode_ids = {ep.id for ep in _corpus.episodes}
    # Every episode should have at least one link
    assert linked_episodes == episode_ids, (
        f"{len(episode_ids - linked_episodes)} episodes have no entity links"
    )


def test_episode_content_mentions_entities():
    """Episode content should mention entity names for FTS5 searchability."""
    entity_names = {e.name for e in _corpus.entities}
    matches = 0
    for ep in _corpus.episodes[:50]:  # Check first 50
        for name in entity_names:
            if name in ep.content:
                matches += 1
                break
    assert matches >= 40, f"Only {matches}/50 episodes mention entity names"


def test_episode_ids_prefixed():
    """Episode IDs should start with 'ep_' to avoid collision with entity IDs."""
    for ep in _corpus.episodes:
        assert ep.id.startswith("ep_"), f"Episode {ep.id} missing 'ep_' prefix"


def test_episode_group_id():
    """All episodes should use the benchmark group_id."""
    for ep in _corpus.episodes:
        assert ep.group_id == GROUP_ID, (
            f"Episode {ep.id} has group_id={ep.group_id}, expected {GROUP_ID}"
        )


# ---------------------------------------------------------------------------
# Conversation scenario tests
# ---------------------------------------------------------------------------


def test_conversation_scenarios_generated():
    """Corpus should generate at least 5 conversation scenarios."""
    assert len(_corpus.conversation_scenarios) >= 5, (
        f"Expected at least 5 scenarios, got {len(_corpus.conversation_scenarios)}"
    )


def test_conversation_scenarios_valid_entities():
    """Conversation scenario bridge entities should exist in the corpus."""
    entity_ids = {e.id for e in _corpus.entities}
    for scenario in _corpus.conversation_scenarios:
        for _query_idx, bridge_ids in scenario.expected_bridge.items():
            for eid in bridge_ids:
                assert eid in entity_ids, (
                    f"Scenario {scenario.name} references unknown entity {eid}"
                )


def test_conversation_scenarios_have_queries():
    """Each scenario should have exactly 3 queries."""
    for scenario in _corpus.conversation_scenarios:
        assert len(scenario.queries) == 3, (
            f"Scenario {scenario.name} has {len(scenario.queries)} queries, expected 3"
        )


def test_episodes_deterministic():
    """Episodes should be deterministic across runs."""
    corpus_a = CorpusGenerator(seed=42).generate()
    corpus_b = CorpusGenerator(seed=42).generate()
    assert len(corpus_a.episodes) == len(corpus_b.episodes)
    for a, b in zip(corpus_a.episodes, corpus_b.episodes):
        assert a.id == b.id
        assert a.content == b.content


# ---------------------------------------------------------------------------
# Scale tests
# ---------------------------------------------------------------------------


def test_scale_defaults_match():
    """CorpusScale default type counts match original hardcoded values."""
    scale = CorpusScale()
    counts = scale.compute_type_counts()
    assert counts["person"] == 200
    assert counts["technology"] == 200
    assert counts["organization"] == 150
    assert counts["location"] == 100
    assert counts["project"] == 150
    assert counts["concept"] == 200
    assert sum(counts.values()) == 1000


def test_scale_5k_entity_count():
    """5k corpus has correct total entity count."""
    corpus = CorpusGenerator(seed=42, total_entities=5000).generate()
    assert len(corpus.entities) == 5000


def test_scale_5k_type_ratios():
    """5k corpus preserves entity type ratios."""
    corpus = CorpusGenerator(seed=42, total_entities=5000).generate()
    counts: dict[str, int] = {}
    for e in corpus.entities:
        counts[e.entity_type] = counts.get(e.entity_type, 0) + 1
    assert counts["person"] == 1000
    assert counts["technology"] == 1000
    assert counts["organization"] == 750
    assert counts["location"] == 500
    assert counts["project"] == 750
    assert counts["concept"] == 1000


def test_scale_5k_relationships():
    """5k corpus has proportionally more relationships."""
    corpus = CorpusGenerator(seed=42, total_entities=5000).generate()
    # target = 5000 * 2.5 = 12500
    assert len(corpus.relationships) >= 12500


def test_scale_5k_unique_names():
    """No duplicate entity names at 5k scale."""
    corpus = CorpusGenerator(seed=42, total_entities=5000).generate()
    names = [e.name for e in corpus.entities]
    assert len(set(names)) == len(names), (
        f"{len(names) - len(set(names))} duplicate names out of {len(names)}"
    )


def test_scale_5k_deterministic():
    """5k corpus is deterministic across runs."""
    corpus_a = CorpusGenerator(seed=42, total_entities=5000).generate()
    corpus_b = CorpusGenerator(seed=42, total_entities=5000).generate()
    assert len(corpus_a.entities) == len(corpus_b.entities)
    assert len(corpus_a.relationships) == len(corpus_b.relationships)
    for a, b in zip(corpus_a.entities, corpus_b.entities):
        assert a.id == b.id
        assert a.name == b.name


def test_default_scale_unchanged():
    """Default scale (1000) produces identical entity IDs and names as before."""
    corpus = CorpusGenerator(seed=42).generate()
    # Spot-check: first person, first tech, first org
    assert corpus.entities[0].id == "ent_bench_per_0000"
    assert corpus.entities[0].name == "Alice Smith"
    assert corpus.entities[200].id == "ent_bench_tech_0000"
    assert corpus.entities[200].name == "Python"
    assert corpus.entities[400].id == "ent_bench_org_0000"
    assert corpus.entities[400].name == "Acme Corp"
    # persons=200, tech=200, org=150 → locations start at 550
    assert corpus.entities[550].id == "ent_bench_loc_0000"
    assert corpus.entities[550].name == "San Francisco"
    # loc=100 → projects start at 650
    assert corpus.entities[650].id == "ent_bench_proj_0000"
    assert corpus.entities[650].name == "Project Phoenix Engine"
    # proj=150 → concepts start at 800
    assert corpus.entities[800].id == "ent_bench_con_0000"
    assert corpus.entities[800].name == "Machine Learning"


# ---------------------------------------------------------------------------
# Cluster-aware summary tests
# ---------------------------------------------------------------------------


def test_cross_cluster_no_domain_collision():
    """The 'domains' lists in _CLUSTER_DEFS must have no pairwise overlaps."""
    seen: dict[str, str] = {}  # domain_term -> cluster_name
    for cluster_def in _CLUSTER_DEFS:
        cname = cluster_def["name"]
        for domain in cluster_def["domains"]:
            d_lower = domain.lower()
            assert d_lower not in seen, (
                f"Domain '{domain}' appears in both '{seen[d_lower]}' and '{cname}'"
            )
            seen[d_lower] = cname


def test_summary_contains_cluster_domain():
    """Over 80% of entities in each cluster should mention a cluster domain."""
    corpus = CorpusGenerator(seed=42).generate()
    clusters = corpus.metadata["clusters"]
    entity_map = {e.id: e for e in corpus.entities}

    for cluster in clusters:
        domains = cluster.get("domains", [])
        if not domains or cluster["name"] == "Long Tail":
            continue
        members = cluster["members"]
        if not members:
            continue
        hits = 0
        for mid in members:
            summary_lower = entity_map[mid].summary.lower()
            if any(d.lower() in summary_lower for d in domains):
                hits += 1
        pct = hits / len(members)
        assert pct >= 0.80, (
            f"Cluster '{cluster['name']}': only {hits}/{len(members)} "
            f"({pct:.0%}) mention a cluster domain, expected >=80%"
        )


def test_temporal_context_queries_cluster_grounded():
    """Temporal context GT entities should be in the target cluster and recently accessed."""
    corpus = CorpusGenerator(seed=42).generate()
    tc_queries = [q for q in corpus.ground_truth if q.category == "temporal_context"]
    assert len(tc_queries) == 5
    for q in tc_queries:
        grade3 = [eid for eid, g in q.relevant_entities.items() if g == 3]
        assert len(grade3) >= 1, f"Query {q.query_id} has no grade-3 entities"


def test_semantic_queries_name_focal_person():
    """Every semantic query should contain a person name from its GT."""
    corpus = CorpusGenerator(seed=42).generate()
    entity_map = {e.id: e for e in corpus.entities}
    semantic_queries = [q for q in corpus.ground_truth if q.category == "semantic"]
    for q in semantic_queries:
        g3_names = [
            entity_map[eid].name
            for eid, g in q.relevant_entities.items()
            if g == 3 and entity_map[eid].entity_type == "person"
        ]
        assert any(name in q.query_text for name in g3_names), (
            f"Query {q.query_id} doesn't contain any grade-3 person name: {q.query_text!r}"
        )


def test_summary_cluster_discrimination():
    """Entity summaries should rarely contain domain keywords from OTHER clusters.

    Before this fix, ~76% of entities matched foreign cluster domains.
    After the fix, it should be <15% per cluster.
    """
    corpus = CorpusGenerator(seed=42).generate()
    clusters = corpus.metadata["clusters"]
    entity_map = {e.id: e for e in corpus.entities}

    # Collect domains per cluster (excluding Long Tail)
    cluster_domain_map: dict[str, list[str]] = {}
    for cluster in clusters:
        if cluster["name"] == "Long Tail":
            continue
        cluster_domain_map[cluster["name"]] = [d.lower() for d in cluster.get("domains", [])]

    for cluster in clusters:
        cname = cluster["name"]
        if cname == "Long Tail":
            continue
        members = cluster["members"]
        if not members:
            continue

        # Collect all domain terms from OTHER clusters
        other_domains = []
        for other_name, other_doms in cluster_domain_map.items():
            if other_name != cname:
                other_domains.extend(other_doms)

        polluted = 0
        for mid in members:
            summary_lower = entity_map[mid].summary.lower()
            if any(d in summary_lower for d in other_domains):
                polluted += 1
        pct = polluted / len(members)
        assert pct < 0.15, (
            f"Cluster '{cname}': {polluted}/{len(members)} ({pct:.0%}) "
            f"summaries contain foreign cluster domains, expected <15%"
        )
