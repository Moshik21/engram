#!/usr/bin/env python3
"""Build a 'connect-the-dots' thesis-validation dataset in LongMemEval-oracle format.

Tests the graph's UNIQUE claim: surfacing an answer that lives in a session which is
NOT semantically similar to the query but IS connected to it via a shared entity.
Episode-vector retrieval structurally misses such sessions (the answer-bearing session
shares no query terms); only graph traversal (query-entity -> shared node -> answer
session) can reach them. Single-hop controls (answer IS in a query-similar session)
check the graph stays neutral where episodes already win — guarding against a self-
flattering eval.

Runs through the validated benchmark_longmemeval harness (graph on/off), so the metric
(recall@k of answer_session_ids + embedding-containment) is the same proven one.
"""

from __future__ import annotations

import json
from pathlib import Path

# One coherent persona ("Jordan", a developer) over dated sessions. Facts about each
# entity are deliberately SPREAD across sessions so multi-hop queries require chaining.
SESSIONS = [
    (
        "s1",
        "2024/01/08 (Mon) 09:15",
        [
            (
                "user",
                (
                    "Started my new job today at Nimbus Labs. "
                    "I'm on the Helix project with a teammate named Priya."
                ),
            ),
            (
                "assistant",
                (
                    "Congrats on the new role at Nimbus Labs! "
                    "Working with Priya on Helix sounds exciting."
                ),
            ),
        ],
    ),
    (
        "s2",
        "2024/01/20 (Sat) 18:40",
        [
            ("user", "I really prefer tea over coffee, and I usually work late on Thursdays."),
            ("assistant", "Tea it is — and a Thursday night owl. Noted."),
        ],
    ),
    (
        "s3",
        "2024/02/03 (Sat) 11:05",
        [
            (
                "user",
                (
                    "Spent the day deep in the Helix codebase. "
                    "It's written in Rust and backed by a Postgres database."
                ),
            ),
            ("assistant", "Rust plus Postgres is a solid stack for Helix."),
        ],
    ),
    (
        "s4",
        "2024/02/18 (Sun) 14:20",
        [
            ("user", "I adopted a dog! Her name is Biscuit and she's a beagle."),
            ("assistant", "Welcome to the family, Biscuit the beagle!"),
        ],
    ),
    (
        "s5",
        "2024/03/05 (Tue) 10:00",
        [
            (
                "user",
                "Priya is our team lead. She actually used to work at Acme before joining Nimbus.",
            ),
            ("assistant", "Good context on Priya — team lead, ex-Acme."),
        ],
    ),
    (
        "s6",
        "2024/03/22 (Fri) 19:30",
        [
            ("user", "I started learning to play the guitar. My instructor is a guy named Marcus."),
            ("assistant", "Nice — guitar lessons with Marcus. Have fun!"),
        ],
    ),
    (
        "s7",
        "2024/04/10 (Wed) 16:45",
        [
            (
                "user",
                (
                    "Huge week — we shipped the launch. "
                    "Right after it, Priya was promoted to Director."
                ),
            ),
            ("assistant", "Massive milestone, and a well-earned promotion to Director."),
        ],
    ),
    (
        "s8",
        "2024/04/25 (Thu) 08:50",
        [
            (
                "user",
                "Rough day: Biscuit had surgery on her leg. The vet says she'll recover fully.",
            ),
            (
                "assistant",
                "Sorry to hear about the leg surgery — glad Biscuit's expected to fully recover.",
            ),
        ],
    ),
    (
        "s9",
        "2024/05/12 (Sun) 13:15",
        [
            (
                "user",
                "Turns out Marcus also teaches piano, not just guitar. I'm tempted to switch.",
            ),
            ("assistant", "Marcus teaching piano too gives you a nice option to switch."),
        ],
    ),
    (
        "s10",
        "2024/05/30 (Thu) 21:10",
        [
            ("user", "Planning a trip to Japan this fall with Priya and another coworker."),
            ("assistant", "A fall trip to Japan with Priya sounds wonderful!"),
        ],
    ),
]

# (qid, type, question, answer, answer_session_ids, has_answer_session_for_marking)
# multi_hop: the answer session shares NO query terms — reachable only via a shared entity.
# single_hop_control: the answer IS in a query-similar session (episodes should already win).
# Genuinely disjoint multi-hops: the question anchors on an entity via ONE session,
# but the answer lives in a DIFFERENT session whose wording shares no terms with the
# question — so semantic episode search ranks the answer-session low and only graph
# traversal (question-entity -> shared node -> answer-session) reaches it.
QUERIES = [
    (
        "m1",
        "multi_hop",
        # anchor: Priya = Helix teammate (s1); answer 'Director' is in s7
        "What is the current job title of my Helix teammate?",
        "Director",
        ["s7"],
    ),
    (
        "m2",
        "multi_hop",
        # anchor: pet with leg surgery = Biscuit (s8); answer 'beagle' is in s4
        "What breed is the pet of mine that later needed leg surgery?",
        "beagle",
        ["s4"],
    ),
    (
        "m3",
        "multi_hop",
        # anchor: Japan travel companion = Priya (s10); answer 'Director' is in s7
        "What is the job title of the person I'm planning to travel to Japan with?",
        "Director",
        ["s7"],
    ),
    (
        "m4",
        "multi_hop",
        # anchor: piano teacher = Marcus (s9); answer 'guitar' is in s6
        (
            "Which instrument did I originally sign up to learn from the "
            "teacher who also offers piano?"
        ),
        "guitar",
        ["s6"],
    ),
    (
        "m5",
        "multi_hop",
        # anchor: the Director on my team = Priya (s7); answer 'Acme' (prior employer) is in s5
        "Which company did the Director on my team work at before Nimbus Labs?",
        "Acme",
        ["s5"],
    ),
    ("c1", "single_hop_control", "Do I prefer tea or coffee?", "Tea", ["s2"]),
    ("c2", "single_hop_control", "What is my dog's name?", "Biscuit", ["s4"]),
    ("c3", "single_hop_control", "Who is my guitar instructor?", "Marcus", ["s6"]),
]


def _session_turns(session_id: str, answer_session_ids: list[str]) -> list[dict]:
    for sid, _date, turns in SESSIONS:
        if sid == session_id:
            has = session_id in answer_session_ids
            return [
                {"role": role, "content": text, "has_answer": bool(has and role == "user")}
                for role, text in turns
            ]
    return []


def build_instances() -> list[dict]:
    haystack_ids = [s[0] for s in SESSIONS]
    haystack_dates = [s[1] for s in SESSIONS]
    instances = []
    for qid, qtype, question, answer, ans_ids in QUERIES:
        haystack_sessions = [_session_turns(sid, ans_ids) for sid in haystack_ids]
        instances.append(
            {
                "question_id": qid,
                "question_type": qtype,
                "question": question,
                "answer": answer,
                "question_date": "2024/06/05 (Wed) 10:00",
                "haystack_dates": haystack_dates,
                "haystack_session_ids": haystack_ids,
                "haystack_sessions": haystack_sessions,
                "answer_session_ids": ans_ids,
            }
        )
    return instances


def main() -> None:
    out = Path("data/connectdots/connectdots.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    instances = build_instances()
    out.write_text(json.dumps(instances, indent=2))
    n_multi = sum(1 for i in instances if i["question_type"] == "multi_hop")
    n_ctrl = sum(1 for i in instances if i["question_type"] == "single_hop_control")
    print(f"wrote {len(instances)} instances ({n_multi} multi_hop, {n_ctrl} control) -> {out}")
    print(f"sessions per haystack: {len(SESSIONS)}")


if __name__ == "__main__":
    main()
