"""Prompts for entity/relationship extraction from episode text."""

# --- Cached system block for extraction (prompt caching) ---
# Used by EntityExtractor to reduce input costs via Anthropic ephemeral cache.
# Defined after EXTRACTION_SYSTEM_PROMPT below.

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
    'Concept|Location|Event|CreativeWork|Article|Software|'
    'HealthCondition|BodyPart|Emotion|Goal|Preference|Habit|Other",\n'
    "  Type guidance:\n"
    "  - Project: work initiative, research project, business venture\n"
    "  - CreativeWork: book, novel, song, film, artwork, story\n"
    "  - Article: blog post, paper, news article, report\n"
    "  - Software: application, library, CLI tool\n"
    "  - Technology: programming language, framework, protocol\n"
    "  - HealthCondition: injury, illness, diagnosis, symptom\n"
    "  - BodyPart: anatomical body part, organ\n"
    "  - Emotion: feeling, mood, psychological state\n"
    "  - Goal: aspiration, objective, plan, intention\n"
    "  - Preference: like, dislike, taste, opinion\n"
    "  - Habit: routine, practice, recurring behavior\n"
    '      "summary": "Brief description of this entity based on the text",\n'
    '      "attributes": {},\n'
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
    '      "polarity": "positive",\n'
    '      "weight": 1.0,\n'
    '      "valid_from": null,\n'
    '      "valid_to": null,\n'
    '      "temporal_hint": null\n'
    "    }\n"
    "  ]\n"
    "}\n"
    "\n"
    "Rules:\n"
    "Polarity rules:\n"
    "- Set polarity to 'negative' when the text explicitly negates, denies, "
    "or ends a relationship (e.g., 'stopped using', 'no longer works at', "
    "'doesn't like', 'quit', 'left', 'divorced').\n"
    "- Set polarity to 'uncertain' for hedged statements ('might', "
    "'considering', 'not sure if', 'thinking about', 'maybe').\n"
    "- Default polarity is 'positive' for affirmative statements.\n"
    "\n"
    "Attributes extraction rules:\n"
    "- Extract key-value attributes for structured facts about the entity. "
    "Use attributes for state, status, quantities, and properties that may "
    "change over time.\n"
    "- Only include attributes when specific properties are mentioned "
    '(e.g., {"status": "recovering", "duration": "3 weeks", '
    '"severity": "mild"}).\n'
    "- Omit the attributes field or use an empty dict when no specific "
    "properties are mentioned.\n"
    "\n"
    "- Entity names should be normalized "
    "(proper case, no abbreviations unless standard).\n"
    "- Predicates should be UPPER_SNAKE_CASE verbs "
    "(e.g., WORKS_AT, USES, PART_OF, CREATED, AUTHORED, "
    "TEACHES, MEMBER_OF, LOCATED_IN, EXPERT_IN, LIKES, DISLIKES, "
    "PREFERS, AIMS_FOR, RECOVERING_FROM, HAS_CONDITION, TREATS, "
    "REQUIRES, LED_TO, CAUSED_BY, HAS_PART, PARENT_OF, CHILD_OF, "
    "STUDYING).\n"
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

# Cached block wrapping the extraction prompt for Anthropic prompt caching.
EXTRACTION_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": EXTRACTION_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]


# ---------------------------------------------------------------------------
# Triage LLM Judge Prompt
# ---------------------------------------------------------------------------

TRIAGE_JUDGE_SYSTEM_PROMPT = (
    "You are a triage judge for a personal knowledge graph memory system.\n"
    "Given a text snippet, decide whether it contains extractable real-world "
    "information worth storing in a knowledge graph.\n"
    "\n"
    "IMPORTANT: Personal and emotional content should score as high as "
    "technical content. Memories about family, relationships, health, feelings, "
    "life events, hobbies, and personal milestones are highly valuable.\n"
    "\n"
    "Respond with JSON only:\n"
    "{\n"
    '  "extract": true/false,\n'
    '  "score": 0.0-1.0,\n'
    '  "reason": "brief explanation",\n'
    '  "tags": ["personal", "technical", "factual", "emotional", "creative"]\n'
    "}\n"
    "\n"
    "Scoring guidelines:\n"
    "- 0.8-1.0: Rich factual content — names, relationships, projects, events, "
    "preferences, personal milestones\n"
    "- 0.5-0.8: Moderate content — some extractable facts mixed with filler\n"
    "- 0.2-0.5: Low content — mostly generic, few specific facts\n"
    "- 0.0-0.2: No content — greetings, system commands, meta-commentary "
    "about the memory system itself\n"
    "\n"
    "Examples:\n"
    '- "My mom was diagnosed with cancer last month" → '
    '{"extract": true, "score": 0.9, "reason": "personal health event", '
    '"tags": ["personal", "emotional"]}\n'
    '- "I use Python and FastAPI at work" → '
    '{"extract": true, "score": 0.85, "reason": "technical preferences", '
    '"tags": ["technical", "factual"]}\n'
    '- "hi" → '
    '{"extract": false, "score": 0.05, "reason": "greeting only", "tags": []}\n'
)

TRIAGE_JUDGE_SYSTEM_CACHED = [
    {
        "type": "text",
        "text": TRIAGE_JUDGE_SYSTEM_PROMPT,
        "cache_control": {"type": "ephemeral"},
    }
]
