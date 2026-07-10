"""Fixed seed episodes and scripted demo beats."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ShowcaseSeedEpisode:
    label: str
    content: str
    source: str = "showcase:seed"


SHOWCASE_SEED_EPISODES: tuple[ShowcaseSeedEpisode, ...] = (
    ShowcaseSeedEpisode(
        label="liam_soccer",
        content=(
            "My son Liam plays soccer every Tuesday after school. "
            "He is 9 years old and we track his games in the family calendar."
        ),
        source="showcase:liam_soccer",
    ),
    ShowcaseSeedEpisode(
        label="liam_correction",
        content=(
            "Correction: Liam switched from soccer to baseball this spring. "
            "Tuesday practices are now baseball, not soccer."
        ),
        source="showcase:liam_correction",
    ),
    ShowcaseSeedEpisode(
        label="family_calendar",
        content=(
            "After Liam's practice, ask how the game went. "
            "We keep his schedule in the family calendar."
        ),
        source="showcase:family_calendar",
    ),
)


@dataclass(frozen=True)
class ShowcaseBeat:
    id: str
    title: str
    user_message: str
    action: Literal["recall", "get_context"]
    query: str
    expect_tokens: tuple[str, ...]
    narrative: str
    answer_hint: str


SHOWCASE_BEATS: tuple[ShowcaseBeat, ...] = (
    ShowcaseBeat(
        id="continuity",
        title="Continuity",
        user_message="He had a great game today.",
        action="recall",
        query="He had a great game today",
        expect_tokens=("liam",),
        narrative="Harness captured prior turns. Agent recalls Liam before replying.",
        answer_hint="Liam's soccer or baseball practice, depending on the latest correction.",
    ),
    ShowcaseBeat(
        id="correction",
        title="Correction",
        user_message="What sport does Liam play now?",
        action="recall",
        query="What sport does Liam play now",
        expect_tokens=("liam", "baseball"),
        narrative="A correction episode updated durable knowledge in the graph.",
        answer_hint="Baseball since the spring switch from soccer.",
    ),
    ShowcaseBeat(
        id="cross_session",
        title="Cross-session briefing",
        user_message="(new session) What do we know about Liam?",
        action="get_context",
        query="Liam family practice",
        expect_tokens=("liam",),
        narrative="get_context loads a briefing without re-explaining the whole family history.",
        answer_hint="Briefing surfaces Liam, practice day, and family calendar habits.",
    ),
)
