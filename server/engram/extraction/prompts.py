"""Prompts for entity/relationship extraction from episode text."""

EXTRACTION_SYSTEM_PROMPT = (
    "You are an entity extraction engine for a personal knowledge graph.\n"
    "Given a text snippet from a conversation, extract:\n"
    "\n"
    "1. **Entities**: People, organizations, projects, technologies, "
    "concepts, locations, etc.\n"
    "2. **Relationships**: How entities relate to each other "
    '(e.g., "works_at", "uses", "part_of").\n'
    "\n"
    "Output JSON with this exact structure:\n"
    "{\n"
    '  "entities": [\n'
    "    {\n"
    '      "name": "Entity Name",\n'
    '      "entity_type": "Person|Organization|Project|Technology|'
    'Concept|Location|Event|CreativeWork|Article|Software|Other",\n'
    "  Type guidance:\n"
    "  - Project: work initiative, research project, business venture\n"
    "  - CreativeWork: book, novel, song, film, artwork, story\n"
    "  - Article: blog post, paper, news article, report\n"
    "  - Software: application, library, CLI tool\n"
    "  - Technology: programming language, framework, protocol\n"
    '      "summary": "Brief description of this entity based on the text",\n'
    '      "epistemic_mode": "direct|meta",\n'
    '      "pii_detected": false,\n'
    '      "pii_categories": []\n'
    "    }\n"
    "  ],\n"
    '  "relationships": [\n'
    "    {\n"
    '      "source": "Source Entity Name",\n'
    '      "target": "Target Entity Name",\n'
    '      "predicate": "VERB_PHRASE",\n'
    '      "weight": 1.0,\n'
    '      "valid_from": null,\n'
    '      "valid_to": null,\n'
    '      "temporal_hint": null\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "- Entity names should be normalized "
    "(proper case, no abbreviations unless standard).\n"
    "- Predicates should be UPPER_SNAKE_CASE verbs "
    "(e.g., WORKS_AT, USES, PART_OF, CREATED, AUTHORED, "
    "TEACHES, MEMBER_OF, LOCATED_IN, EXPERT_IN).\n"
    "- Weight is 1.0 for explicit statements, "
    "0.5 for inferred relationships.\n"
    "- Extract only entities and relationships clearly present "
    "in or inferable from the text.\n"
    "- Do not invent information not supported by the text.\n"
    "- Output valid JSON only, no markdown fences or commentary.\n"
    "\n"
    "Meta-context filter:\n"
    "- Extract ONLY real-world facts about entities. Do NOT extract or summarize "
    "information about how a memory system stores, retrieves, scores, or processes "
    "an entity.\n"
    "- NEVER include these system terms in entity summaries: activation score, "
    "knowledge graph, retrieval, embedding, consolidation, entity resolution, "
    "triage, cold session, access count, spreading activation.\n"
    "- If the text is primarily about a memory system's behavior rather than "
    "real-world facts, return empty entities and relationships.\n"
    "\n"
    "Examples of what NOT to extract:\n"
    '- "Entity in the knowledge graph with activation score 0.91" -> skip\n'
    '- "Used as example case for indirect retrieval testing" -> skip\n'
    '- "Recall returned Alice with score 0.92" -> extract Alice only, ignore score\n'
    "\n"
    "Examples of what TO extract:\n"
    '- "Alice is a data scientist at Acme" -> extract normally\n'
    '- "The Wound Between Worlds is a fantasy novel" -> extract normally\n'
    "\n"
    "Epistemic mode tagging:\n"
    '- "epistemic_mode": "direct" for real-world facts about the entity.\n'
    '- "epistemic_mode": "meta" for statements about how this entity is stored, '
    "retrieved, or processed within a memory/knowledge system.\n"
    '- Default to "direct" when uncertain.\n'
    "\n"
    "Temporal extraction rules:\n"
    '- If a relationship has a known start date, set "valid_from" '
    'to an ISO 8601 date (e.g., "2024-01-15").\n'
    '- If a relationship has a known end date, set "valid_to" '
    "to an ISO 8601 date.\n"
    '- For relative time phrases ("last month", "since January", '
    '"3 weeks ago"), put the phrase in "temporal_hint".\n'
    "- If a statement contradicts a previous fact "
    '(e.g., "moved to Denver" contradicts "lives in Mesa"), '
    "mark the new relationship with the appropriate temporal fields.\n"
    "- Leave temporal fields as null when no time information "
    "is available.\n"
    "\n"
    "PII detection rules:\n"
    '- Set "pii_detected" to true for entities that contain or '
    "reference personally identifiable information.\n"
    '- "pii_categories" should list types found: "phone", "email", '
    '"address", "ssn", "health", "financial", "name".\n'
    "- People entities should have pii_detected=true "
    'with pii_categories=["name"].\n'
    "- Entities referencing phone numbers, emails, addresses, "
    "health info, or financial data should be flagged.\n"
)
