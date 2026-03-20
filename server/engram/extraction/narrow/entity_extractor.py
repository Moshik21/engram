"""Narrow extractor for entity evidence from proper names, tech tokens, identity patterns."""

from __future__ import annotations

import re

from engram.config import ActivationConfig
from engram.extraction.evidence import EvidenceCandidate
from engram.models.episode_cue import EpisodeCue

# Strip protocol markers, XML/HTML tags, code blocks, inline code, and URLs before extraction
_NOISE_STRIP = re.compile(
    r"\[(?:user|assistant|system)[^\]]*\]|"  # protocol markers
    r"<[^>]{1,80}>|"  # XML/HTML tags
    r"```[\s\S]*?```|"  # fenced code blocks
    r"`[^`]+`|"  # inline code
    r"https?://\S+",  # URLs
    re.MULTILINE,
)

# Reuse patterns from cues.py
_PROPER_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s+[A-Z][a-z]+)*\b")
_TECHNICAL_TOKENS = re.compile(
    r"\b(?:[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+|"
    r"[A-Za-z][A-Za-z0-9_-]*\.(?:py|ts|tsx|js|jsx|json|toml|md|yaml|yml)|"
    r"(?:API|SDK|CLI|MCP|LLM|SQL|FTS5|Redis|SQLite|FalkorDB|FastAPI|React"
    r"|Next\.js|TypeScript))\b"
)
_IDENTITY_PATTERNS = re.compile(
    r"\b(?:my name is|i am|i'm|my wife|my husband|my partner|my mom|my dad|"
    r"i work at|i live in|we live in)\b",
    re.IGNORECASE,
)

# More specific identity captures that extract the *value*
_IDENTITY_CAPTURES: list[tuple[re.Pattern[str], str, str]] = [
    # (pattern, entity_type, signal_name)
    (
        re.compile(
            r"\bmy name is\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "name_declaration",
    ),
    (
        re.compile(
            r"\bi(?:'m| am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b",
        ),
        "Person",
        "self_introduction",
    ),
    (
        re.compile(
            r"\bi work(?:s)? (?:at|for)\s+([A-Z][a-zA-Z0-9 ]+?)"
            r"(?:\.|,|\band\b|$)",
            re.I,
        ),
        "Organization",
        "workplace_declaration",
    ),
    (
        re.compile(
            r"\bi live in\s+([A-Z][a-zA-Z ]+?)(?:\.|,|\band\b|$)",
            re.I,
        ),
        "Location",
        "residence_declaration",
    ),
    (
        re.compile(
            r"\bwe live in\s+([A-Z][a-zA-Z ]+?)(?:\.|,|\band\b|$)",
            re.I,
        ),
        "Location",
        "residence_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:wife|husband|partner)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:mom|dad|mother|father)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
    (
        re.compile(
            r"\bmy (?:son|daughter|brother|sister)\s+(?:is\s+)?"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.I,
        ),
        "Person",
        "family_declaration",
    ),
]

# Type inference heuristics
_TECH_KEYWORDS = frozenset(
    {
        # Original
        "api", "sdk", "cli", "mcp", "llm", "sql", "fts5",
        "redis", "sqlite", "falkordb", "fastapi", "react",
        "next.js", "typescript", "python", "javascript", "node",
        "docker", "kubernetes", "aws", "gcp", "azure",
        # Languages / runtimes
        "rust", "golang", "go", "java", "swift", "kotlin",
        "ruby", "php", "dart", "elixir", "haskell", "scala",
        "deno", "bun",
        # Frontend frameworks
        "vue", "angular", "svelte", "tailwind", "remix", "astro",
        # Build tools / package managers
        "webpack", "vite", "esbuild", "rollup", "turbopack",
        "npm", "pnpm", "yarn", "pip", "cargo", "maven", "gradle",
        # Cloud / infra
        "vercel", "heroku", "netlify", "cloudflare", "digitalocean",
        "terraform", "ansible", "pulumi", "nginx", "caddy",
        # Databases / search
        "postgres", "postgresql", "mongodb", "mysql", "mariadb",
        "elasticsearch", "opensearch", "kafka", "rabbitmq",
        "dynamodb", "cassandra", "neo4j", "dgraph",
        # AI / ML
        "pytorch", "tensorflow", "numpy", "pandas", "scikit-learn",
        "langchain", "openai", "anthropic", "gemini", "ollama",
        "huggingface", "transformers",
        # Vector DBs / tools
        "chromadb", "pinecone", "weaviate", "qdrant", "milvus",
        # ORMs / data
        "prisma", "drizzle", "sqlalchemy", "sequelize", "typeorm",
        "graphql", "grpc", "protobuf",
        # DevOps / CI
        "github", "gitlab", "bitbucket", "jenkins", "circleci",
        # Other common tech
        "supabase", "firebase", "stripe", "twilio", "auth0",
        "storybook", "cypress", "playwright", "jest", "vitest",
        "recharts", "threejs", "three.js", "zustand", "redux",
        "electron", "tauri", "flutter",
    }
)

_COMPANY_SUFFIXES = frozenset(
    {"inc", "llc", "corp", "ltd", "co", "gmbh", "labs", "ai", "io"}
)

_PRODUCT_SUFFIXES = frozenset(
    {"app", "pro", "studio", "cloud", "hub", "kit", "os"}
)

_LOCATION_SUFFIXES = frozenset(
    {
        "city",
        "town",
        "village",
        "state",
        "county",
        "island",
        "creek",
        "valley",
        "heights",
        "hills",
        "beach",
        "springs",
    }
)

# Common non-entity proper nouns to skip
_STOPWORDS = frozenset(
    {
        # Determiners / demonstratives
        "The",
        "This",
        "That",
        "These",
        "Those",
        "Here",
        "There",
        # Question words
        "What",
        "Which",
        "Where",
        "When",
        "Who",
        "How",
        "Why",
        # Conjunctions / adverbs
        "But",
        "And",
        "However",
        "Also",
        "Just",
        "Very",
        # Days / months
        "Monday",
        "Tuesday",
        "Wednesday",
        "Thursday",
        "Friday",
        "Saturday",
        "Sunday",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
        # Sentence connectives / discourse markers
        "First",
        "Second",
        "Third",
        "Finally",
        "Next",
        "Then",
        "Now",
        "Note",
        "Please",
        "Sure",
        "Okay",
        "Right",
        "Well",
        # Determiners / quantifiers
        "Some",
        "Any",
        "All",
        "Each",
        "Every",
        "Both",
        "Many",
        "Most",
        "Few",
        "More",
        "Less",
        "Other",
        "Another",
        "Such",
        # Protocol / code fragments
        "True",
        "False",
        "None",
        "Null",
        "Error",
        "Result",
        "Input",
        "Output",
        "Value",
        "Type",
        "Name",
        "Text",
        "Status",
        "Check",
        "Return",
        # Common AI assistant sentence openers / verbs
        "Let",
        "Lets",
        "Make",
        "Use",
        "Get",
        "Set",
        "Run",
        "Try",
        "Add",
        "Keep",
        "Look",
        "Feel",
        "Think",
        "Know",
        "See",
        "Call",
        "Send",
        "Either",
        "Subtle",
    }
)


def _infer_entity_type(name: str, signal: str | None = None) -> str:
    """Infer entity type from name and context."""
    lower = name.lower()
    if signal in (
        "name_declaration",
        "self_introduction",
        "family_declaration",
    ):
        return "Person"
    if signal == "workplace_declaration":
        return "Organization"
    if signal == "residence_declaration":
        return "Location"
    if lower in _TECH_KEYWORDS or "." in name or "/" in name:
        return "Technology"
    tokens = lower.split()
    if tokens and tokens[-1] in _LOCATION_SUFFIXES:
        return "Location"
    if tokens and tokens[-1] in _COMPANY_SUFFIXES:
        return "Organization"
    if tokens and tokens[-1] in _PRODUCT_SUFFIXES:
        return "Product"
    # Default to Concept for bare proper names — identity captures handle real Person entities
    if name[0].isupper() and name.isalpha():
        return "Concept"
    return "Concept"


def _get_source_span(text: str, name: str) -> str | None:
    """Extract the sentence containing the entity name."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    for sent in sentences:
        if name in sent:
            span = sent[:200]
            if _is_noisy_span(span):
                return None
            return span
    return None


def _is_noisy_span(span: str) -> bool:
    """Reject source_span that is >50% non-alphanumeric (excluding spaces)."""
    if not span:
        return True
    alnum = sum(1 for c in span if c.isalnum() or c == " ")
    return alnum / len(span) < 0.50


def _is_sentence_initial(m: re.Match, text: str) -> bool:  # type: ignore[type-arg]
    """Check if a regex match occurs at sentence-initial position."""
    start = m.start()
    if start == 0:
        return True
    preceding = text[:start].rstrip()
    return len(preceding) > 0 and preceding[-1] in ".!?\n"


class IdentityEntityExtractor:
    """Extracts entity evidence from proper names, tech tokens, and identity patterns."""

    name = "identity_entity"

    def extract(
        self,
        text: str,
        episode_id: str,
        group_id: str,
        cue: EpisodeCue | None = None,
        cfg: ActivationConfig | None = None,
    ) -> list[EvidenceCandidate]:
        # Sanitize input: strip protocol markers, tags, code blocks, URLs
        text = _NOISE_STRIP.sub(" ", text)

        candidates: list[EvidenceCandidate] = []
        seen_names: set[str] = set()

        # 1. Identity captures (highest confidence -- explicit declarations)
        for pattern, entity_type, signal in _IDENTITY_CAPTURES:
            for match in pattern.finditer(text):
                name = match.group(1).strip()
                if not name or name.lower() in seen_names:
                    continue
                seen_names.add(name.lower())
                candidates.append(
                    EvidenceCandidate(
                        episode_id=episode_id,
                        group_id=group_id,
                        fact_class="entity",
                        confidence=0.90,
                        source_type="narrow_extractor",
                        extractor_name=self.name,
                        payload={
                            "name": name,
                            "entity_type": entity_type,
                        },
                        source_span=_get_source_span(text, name),
                        corroborating_signals=[
                            signal,
                            "identity_pattern",
                        ],
                    )
                )

        # 2. Proper names (medium confidence)
        for match in _PROPER_NAMES.finditer(text):
            name = match.group()
            if not name or name in _STOPWORDS or name.lower() in seen_names:
                continue
            # Skip single-word sentence-initial candidates unless they also appear mid-sentence
            if " " not in name and _is_sentence_initial(match, text):
                mid_pattern = re.compile(
                    r"(?<![.!?\n]\s)" + re.escape(name) + r"\b"
                )
                if not mid_pattern.search(text[match.end():]):
                    continue
            seen_names.add(name.lower())
            entity_type = _infer_entity_type(name)
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="entity",
                    confidence=0.55,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload={
                        "name": name,
                        "entity_type": entity_type,
                    },
                    source_span=_get_source_span(text, name),
                    corroborating_signals=["proper_name"],
                )
            )

        # 3. Technical tokens (medium confidence)
        for match in _TECHNICAL_TOKENS.finditer(text):
            token = match.group()
            if not token or token.lower() in seen_names:
                continue
            seen_names.add(token.lower())
            candidates.append(
                EvidenceCandidate(
                    episode_id=episode_id,
                    group_id=group_id,
                    fact_class="entity",
                    confidence=0.70,
                    source_type="narrow_extractor",
                    extractor_name=self.name,
                    payload={
                        "name": token,
                        "entity_type": "Technology",
                    },
                    source_span=_get_source_span(text, token),
                    corroborating_signals=["technical_token"],
                )
            )

        return candidates
