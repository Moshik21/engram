"""Bridge corpus for the graph deciding experiment (GRAPH_THESIS.md §5).

The bridge construction is M3.1's, unchanged: ``ep1`` co-mentions person **A**
and topic **B**; ``ep2`` (gold) states a concrete fact about **B** and never
mentions A; the query names only A. Reaching gold therefore requires
``query -> A -> (edge A~B) -> B -> (membership B~ep2) -> ep2``.

Two things are deliberately different from M3.1, and both come from
hard-won lessons rather than taste:

* **The text is harvested from this repository, not imagined.** A fixture
  assembled from imagination inherits the author's blind spots — the live
  Correction predicate that killed 100% of real entities while its 47-case
  suite passed was every-entry-Title-Cased. So topics are real module stems
  under ``server/engram/``, and the sentences around them are real commit
  subjects that actually touched those files. That also puts the corpus in
  Engram's actual regime: GRAPH_THESIS.md §6 records that the live corpus is
  "88% filesystem/document scaffolding and coding-session transcripts", a
  regime no external agent-memory benchmark covers.

* **The graph is not planted by default.** ``--producer`` selects who builds
  the graph. ``proposals`` reproduces M3.1's planted control (harness as
  extractor); ``narrow``/``auto``/``ollama``/``anthropic`` build it ORGANICALLY
  from the same text. Whichever runs, ``verify_bridges`` re-checks every bridge
  against the real store before a single question is scored, and questions
  whose bridge did not materialise are dropped from the scored set rather than
  silently counted as misses.

Personas are synthetic on purpose: A must be guaranteed absent from the gold
episode, and the only way to guarantee that against harvested text is to own
the name. ``build_corpus`` asserts the invariant rather than assuming it.
"""

from __future__ import annotations

import json
import random
import re
import subprocess
from dataclasses import asdict, dataclass, field
from pathlib import Path

# Synthetic personas. Verified absent from every harvested string at build
# time; a collision raises rather than silently weakening the bridge.
_GIVEN = [
    "Marisol",
    "Tobias",
    "Ilse",
    "Rune",
    "Anouk",
    "Kwame",
    "Sanne",
    "Emeka",
    "Freya",
    "Oskar",
    "Nadia",
    "Bastien",
    "Yuki",
    "Lorcan",
    "Petra",
    "Isandro",
    "Hana",
    "Merrick",
    "Solveig",
    "Zephyr",
]
_FAMILY = [
    "Okonkwo",
    "Vandermolen",
    "Bergqvist",
    "Achterberg",
    "Nakamura",
    "Delacroix",
    "Farkas",
    "Lindqvist",
    "Osei",
    "Brandsma",
    "Kowalczyk",
    "Ravensworth",
    "Quintero",
    "Halvorsen",
    "Marchetti",
]

_LINK_TEMPLATES = [
    "Standup note: {person} picked up the {topic} work this cycle. "
    "We agreed to keep the change behind a flag until the numbers land.",
    "Handoff recorded: {person} now owns {topic}. Ping them before anyone touches that path again.",
    "{person} walked me through {topic} during review — they wrote most of it "
    "and will keep reviewing changes there.",
    "Ownership update: {topic} moves to {person} for the rest of the quarter. "
    "Everything else on the rota is unchanged.",
]

_GOLD_TEMPLATES = [
    "{subject}\n\nChange landed in {path}. The {topic} path is the one that moved.",
    "Change log for {topic} ({path}):\n{subject}",
    "{subject}\n\nScope: {topic} only. No other module in {path} was touched.",
]

_DISTRACTOR_TEMPLATES = [
    "{person} is out on Thursday; the review queue moves to the following morning.",
    "Scheduling: {person} asked to swap the pairing slot to the afternoon block.",
    "{person} filed the weekly summary late again — no action needed, just noting it.",
    "Access request approved for {person}. Nothing else changed in the rota.",
]

_QUERY_TEMPLATES = [
    "What is {person} working on right now?",
    "What did {person} take ownership of?",
    "Where is {person} spending their time this cycle?",
]

_IDENT = re.compile(r"^[a-z][a-z0-9_]{5,}$")

# Stems that are too generic to act as a discriminating topic: they appear in
# dozens of unrelated episodes, so "the query names only A" stops isolating one
# bridge. Excluded by name rather than by a similarity heuristic so the
# exclusion is auditable.
_TOO_GENERIC = frozenset(
    {
        "__init__",
        "config",
        "models",
        "utils",
        "helpers",
        "common",
        "base",
        "types",
        "constants",
        "protocols",
        "service",
        "services",
        "manager",
        "server",
        "client",
        "main",
    }
)


@dataclass(frozen=True)
class Topic:
    """A real module in this repo plus a real commit subject that touched it."""

    stem: str
    path: str
    subject: str


@dataclass(frozen=True)
class CorpusEpisode:
    tag: str
    content: str
    role: str  # link | gold | distractor | filler
    proposed_entities: list[dict] = field(default_factory=list)
    proposed_relationships: list[dict] = field(default_factory=list)


@dataclass(frozen=True)
class BridgeQuestion:
    qid: str
    person: str
    topic: str
    query: str
    link_tag: str
    gold_tag: str
    # Token groups for lane 1's multi-source scorer. Each group deliberately
    # spans TWO episodes — the topic name lives in the link episode, the marker
    # only in the gold — so the group is a MISS by construction under the
    # battery's one-row containment rule and a HIT only under a union rule.
    # That is the exact phenomenon GRAPH_THESIS.md M16 says the battery cannot
    # see, made into a fixture.
    expected_tokens: list[list[str]] = field(default_factory=list)


@dataclass(frozen=True)
class Corpus:
    episodes: list[CorpusEpisode]
    questions: list[BridgeQuestion]
    provenance: dict

    def to_json(self) -> str:
        return json.dumps(
            {
                "episodes": [asdict(e) for e in self.episodes],
                "questions": [asdict(q) for q in self.questions],
                "provenance": self.provenance,
            },
            indent=1,
            sort_keys=True,
        )

    @classmethod
    def from_json(cls, text: str) -> Corpus:
        raw = json.loads(text)
        return cls(
            episodes=[CorpusEpisode(**e) for e in raw["episodes"]],
            questions=[BridgeQuestion(**q) for q in raw["questions"]],
            provenance=raw["provenance"],
        )


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(  # noqa: S603
        ["git", "-C", str(repo_root), *args],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    ).stdout


def harvest_topics(repo_root: Path, *, limit: int = 400) -> list[Topic]:
    """Pair real ``server/engram`` modules with a real commit subject.

    One ``git log --name-only`` call, parsed into (subject -> files) and
    inverted. Deterministic for a given HEAD, which is what makes a re-run with
    a different HNSW seed comparable to the first run.
    """
    raw = _git(repo_root, "log", f"-n{limit}", "--name-only", "--format=@@%s")
    subject_for: dict[str, tuple[str, str]] = {}
    current = ""
    for line in raw.splitlines():
        if line.startswith("@@"):
            current = line[2:].strip()
            continue
        path = line.strip()
        if not path or not current:
            continue
        if not path.startswith("server/engram/") or not path.endswith(".py"):
            continue
        stem = Path(path).stem
        if stem in _TOO_GENERIC or not _IDENT.match(stem):
            continue
        # Newest commit wins (git log is newest-first, so keep the first sighting).
        subject_for.setdefault(stem, (current, path))

    return [
        Topic(stem=stem, path=path, subject=subject)
        for stem, (subject, path) in sorted(subject_for.items())
    ]


def _gold_marker(gold_text: str, link_text: str, topic: str) -> str:
    """A word that appears in the gold episode and NOWHERE in the link episode.

    Guarantees the token group cannot be covered by one row: the topic name is
    in both episodes, the marker only in the gold. Longest candidate wins so the
    marker is distinctive rather than a stopword that happens to be missing.
    """
    link_words = set(re.findall(r"[a-z0-9_]{3,}", link_text.casefold()))
    candidates = [
        word
        for word in re.findall(r"[A-Za-z0-9_]{6,}", gold_text)
        if word.casefold() not in link_words and word.casefold() != topic.casefold()
    ]
    if not candidates:
        raise ValueError(f"no distinctive gold marker for topic {topic!r}")
    return max(candidates, key=len)


def _entity(name: str, entity_type: str, span: str) -> dict:
    return {"name": name, "entity_type": entity_type, "source_span": span}


def build_corpus(
    *,
    repo_root: Path,
    n: int = 60,
    seed: int = 17,
    distractors_per_person: int = 1,
    filler: int = 30,
) -> Corpus:
    """Build ``n`` bridge questions from harvested repo material.

    Raises when the invariants that make a bridge a bridge do not hold, rather
    than emitting a corpus that quietly measures something else.
    """
    topics = harvest_topics(repo_root)
    if len(topics) < n:
        raise ValueError(
            f"harvested only {len(topics)} usable module topics from git history but "
            f"n={n} bridges were requested; lower --n or widen the harvest"
        )

    rng = random.Random(seed)
    chosen = rng.sample(topics, n)
    chosen.sort(key=lambda t: t.stem)

    people = [f"{g} {f}" for g in _GIVEN for f in _FAMILY]
    rng.shuffle(people)
    if len(people) < n:
        raise ValueError(f"only {len(people)} personas available for n={n}")
    people = people[:n]

    # Lesson: a persona that collides with harvested text destroys the "gold
    # never mentions A" invariant silently. Check, do not assume.
    harvested_blob = "\n".join(f"{t.stem} {t.path} {t.subject}" for t in topics).casefold()
    harvested_words = set(re.findall(r"[a-z0-9_]+", harvested_blob))
    for person in people:
        for part in person.split():
            if part.casefold() in harvested_words:
                raise ValueError(
                    f"persona token {part!r} collides with harvested repo text; "
                    "pick a different synthetic name"
                )

    episodes: list[CorpusEpisode] = []
    questions: list[BridgeQuestion] = []

    for idx, (topic, person) in enumerate(zip(chosen, people, strict=True)):
        link_text = _LINK_TEMPLATES[idx % len(_LINK_TEMPLATES)].format(
            person=person, topic=topic.stem
        )
        gold_text = _GOLD_TEMPLATES[idx % len(_GOLD_TEMPLATES)].format(
            subject=topic.subject, path=topic.path, topic=topic.stem
        )
        query = _QUERY_TEMPLATES[idx % len(_QUERY_TEMPLATES)].format(person=person)

        link_tag = f"link{idx:03d}"
        gold_tag = f"gold{idx:03d}"

        episodes.append(
            CorpusEpisode(
                tag=link_tag,
                content=link_text,
                role="link",
                proposed_entities=[
                    _entity(person, "Person", person),
                    _entity(topic.stem, "Technology", topic.stem),
                ],
                proposed_relationships=[
                    {
                        "subject": person,
                        "predicate": "WORKS_ON",
                        "object": topic.stem,
                        "source_span": link_text[:120],
                    }
                ],
            )
        )
        episodes.append(
            CorpusEpisode(
                tag=gold_tag,
                content=gold_text,
                role="gold",
                proposed_entities=[_entity(topic.stem, "Technology", topic.stem)],
            )
        )
        for d in range(distractors_per_person):
            text = _DISTRACTOR_TEMPLATES[(idx + d) % len(_DISTRACTOR_TEMPLATES)].format(
                person=person
            )
            episodes.append(
                CorpusEpisode(
                    tag=f"dist{idx:03d}_{d}",
                    content=text,
                    role="distractor",
                    proposed_entities=[_entity(person, "Person", person)],
                )
            )

        questions.append(
            BridgeQuestion(
                qid=f"q{idx:03d}",
                person=person,
                topic=topic.stem,
                query=query,
                link_tag=link_tag,
                gold_tag=gold_tag,
                expected_tokens=[[topic.stem, _gold_marker(gold_text, link_text, topic.stem)]],
            )
        )

    unused = [t for t in topics if t not in set(chosen)]
    for i, topic in enumerate(unused[:filler]):
        episodes.append(
            CorpusEpisode(
                tag=f"fill{i:03d}",
                content=f"{topic.subject}\n\nTouched {topic.path}.",
                role="filler",
                proposed_entities=[_entity(topic.stem, "Technology", topic.stem)],
            )
        )

    _assert_bridge_invariants(episodes, questions)

    head = _git(repo_root, "rev-parse", "HEAD").strip()
    provenance = {
        "generator": "engram.evaluation.graph_kill_rig.corpus",
        "repo_head": head,
        "seed": seed,
        "n_bridges": n,
        "distractors_per_person": distractors_per_person,
        "filler": len([e for e in episodes if e.role == "filler"]),
        "episodes": len(episodes),
        "topics_harvested": len(topics),
        "topic_source": "git log --name-only over server/engram/**/*.py",
    }
    return Corpus(episodes=episodes, questions=questions, provenance=provenance)


def _assert_bridge_invariants(
    episodes: list[CorpusEpisode], questions: list[BridgeQuestion]
) -> None:
    """Fail loudly when the corpus is not actually a bridge corpus."""
    by_tag = {e.tag: e for e in episodes}
    for q in questions:
        link = by_tag[q.link_tag].content
        gold = by_tag[q.gold_tag].content
        if q.person not in link:
            raise ValueError(f"{q.qid}: link episode does not name the person")
        if q.topic not in link:
            raise ValueError(f"{q.qid}: link episode does not name the topic")
        if q.topic not in gold:
            raise ValueError(f"{q.qid}: gold episode does not name the topic")
        if q.person in gold:
            raise ValueError(f"{q.qid}: gold episode LEAKS the person — not a bridge")
        if q.person not in q.query:
            raise ValueError(f"{q.qid}: query does not name the person")
        if q.topic in q.query:
            raise ValueError(f"{q.qid}: query LEAKS the topic — not a bridge")

    # The person must appear nowhere except their own link + distractors, or
    # a query naming A stops isolating one bridge.
    owned: dict[str, set[str]] = {}
    for q in questions:
        owned.setdefault(q.person, set()).update({q.link_tag})
    for q in questions:
        for ep in episodes:
            if ep.tag in owned[q.person] or ep.role == "distractor":
                continue
            if q.person in ep.content:
                raise ValueError(
                    f"{q.qid}: person {q.person!r} leaks into unrelated episode {ep.tag}"
                )
