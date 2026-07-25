"""Pre-registered SUCCESS / KILL thresholds for the graph deciding experiment.

Transcribed verbatim from ``docs/product/GRAPH_THESIS.md`` §5 ("Pre-registered
thresholds — written before any result is seen") at commit ``d7c764e``, BEFORE
any arm of this rig had ever been run.

**Why this module exists at all.** This project has a documented history of
re-reading its own nulls into wins (GRAPH_THESIS.md §2.4, §6: "a
*reinterpretation of someone else's null*, precisely the move this project has
made five times and been wrong about"). A threshold that lives in prose gets
renegotiated once a number is on the table. A threshold that lives in a pure
function with a test pinning its truth table does not. Nothing in this module
reads a result file, a config, or an environment variable: the verdict is a
function of the measured arms and nothing else.

Three deliberate encodings, each a judgment call made in advance and recorded
here rather than discovered later:

1. **Counts, not rates.** "``B − A ≥ +6/N``" is read as *six questions out of
   N*, not as the rate 6/N. The noise floor it is calibrated against
   (``969b00d``'s ``depth_flips=2`` per slice) is an absolute flip count, so an
   absolute delta is the comparable quantity. Because absolute counts only mean
   the same thing at a comparable N, ``MIN_SCORED_QUESTIONS`` refuses any run
   whose scored set shrank below the anchor experiment's N=36 — the obvious way
   to game an absolute threshold is to quietly shrink N.

2. **KILL outranks SUCCESS.** K4 (the market test) is the one kill criterion
   that can hold at the same time as every success criterion: the graph really
   did beat both A and C, on a question set where the linking episode almost
   always ranked anyway. The thesis is explicit that this makes the win
   commercially irrelevant, so the verdict is KILL and the fact that the
   success bar was cleared is recorded alongside it rather than allowed to
   overturn it.

3. **The kill arm is scored at its strongest.** Arm C is run in two merge
   variants (see ``arms.py``) and ``select_kill_arm`` takes the better one.
   That biases every comparison AGAINST the graph, which is the correct
   direction of error for this experiment: "A rigged kill arm is worse than no
   kill arm, because it would license a false positive for the graph."
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

PRE_REGISTRATION_SOURCE = "docs/product/GRAPH_THESIS.md §5 @ d7c764e"

# The scored question set may never fall below the anchor experiment's N.
# M3.1 ran N=36 (2/36 -> 22/36); the deciding experiment targets N ~ 60.
MIN_SCORED_QUESTIONS = 36

# --- SUCCESS (all four must hold) -----------------------------------------
SUCCESS_B_MINUS_A_MIN = 6  # questions, reachability@5
SUCCESS_B_MINUS_C_MIN = 3  # questions, reachability@5
SUCCESS_P50_ADDED_MS_MAX = 10.0  # arm B p50 minus arm A p50
SUCCESS_MEAN_ROWS_MAX = 20.0  # rows handed to the answerer

# --- KILL (any one is sufficient) -----------------------------------------
KILL_K2_B_MINUS_A_MAX = 3  # inside HNSW jitter
KILL_K3_MEAN_ROWS_MAX = 20.0  # AtomMem degradation threshold
KILL_K4_RESIDUAL_MIN = 0.10  # the graph's addressable market

KILL_DESCRIPTIONS: dict[str, str] = {
    "K1": (
        "the round-trip test — arm C (the agent's own second query) reaches the "
        "gold episode at least as often as arm B. The graph is buying a "
        "round-trip, not a capability."
    ),
    "K2": (
        "the noise test — B - A is inside the +/-3 HNSW jitter implied by "
        "969b00d's depth_flips=2 per slice."
    ),
    "K3": (
        "the context test — arm B hands the answerer more than 20 rows without "
        "a reachability gain that clears the success bar."
    ),
    "K4": (
        "the market test — the residual (questions where arm A fails to rank the "
        "LINKING episode in its top-10) is under 10%. The linking episode almost "
        "always ranks on its own, so the graph is a latency optimisation dressed "
        "as a capability."
    ),
}


@dataclass(frozen=True)
class ArmResult:
    """One arm's measured outcome. Every field is counted, never estimated."""

    arm: str
    n: int
    reach_at_5: int
    reach_at_10: int
    mean_rows: float
    mean_chars: float
    p50_ms: float
    # Token counts are deliberately absent unless a real tokenizer produced
    # them: an estimate here would be exactly the "plausible-but-wrong" metric
    # docs/product/INSTRUMENT_AUDIT.md forbids. mean_chars is exact.
    mean_tokens: float | None = None


@dataclass(frozen=True)
class Verdict:
    """The pre-registered decision, computed from arms and nothing else."""

    verdict: Literal["SUCCESS", "KILL", "AMBIGUOUS"]
    kill_reasons: list[str] = field(default_factory=list)
    success_criteria: dict[str, bool] = field(default_factory=dict)
    deltas: dict[str, float] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "verdict": self.verdict,
            "kill_reasons": list(self.kill_reasons),
            "success_criteria": dict(self.success_criteria),
            "deltas": dict(self.deltas),
            "notes": list(self.notes),
            "pre_registration_source": PRE_REGISTRATION_SOURCE,
        }


def select_kill_arm(variants: list[ArmResult]) -> ArmResult:
    """Return the STRONGEST arm-C variant by reachability@5.

    Deliberately adversarial to the graph (see module docstring). Ties break on
    arm name so the selection is deterministic and reproducible.
    """
    if not variants:
        raise ValueError("select_kill_arm requires at least one arm-C variant")
    return sorted(variants, key=lambda r: (-r.reach_at_5, r.arm))[0]


def evaluate(
    arm_a: ArmResult,
    arm_b: ArmResult,
    arm_c: ArmResult,
    *,
    residual_rate: float,
) -> Verdict:
    """Apply the pre-registered thresholds. Pure; no I/O, no config.

    ``residual_rate`` is pre-flight measurement #4: the fraction of scored
    questions where arm A fails to rank the LINKING episode in its top-10.
    """
    if not (arm_a.n == arm_b.n == arm_c.n):
        raise ValueError(
            f"arms scored different question sets (A={arm_a.n} B={arm_b.n} C={arm_c.n}); "
            "a delta across unequal N is not a measurement"
        )
    if arm_a.n < MIN_SCORED_QUESTIONS:
        raise ValueError(
            f"scored N={arm_a.n} is below the pre-registered floor of "
            f"{MIN_SCORED_QUESTIONS}; absolute thresholds are not comparable at a "
            "smaller N and shrinking N is the obvious way to game them"
        )

    b_minus_a = arm_b.reach_at_5 - arm_a.reach_at_5
    b_minus_c = arm_b.reach_at_5 - arm_c.reach_at_5
    p50_added = round(arm_b.p50_ms - arm_a.p50_ms, 4)

    deltas = {
        "b_minus_a_reach5": float(b_minus_a),
        "b_minus_c_reach5": float(b_minus_c),
        "p50_added_ms": p50_added,
        "b_mean_rows": arm_b.mean_rows,
        "residual_rate": round(residual_rate, 4),
    }

    success_criteria = {
        "b_minus_a_ge_6": b_minus_a >= SUCCESS_B_MINUS_A_MIN,
        "b_minus_c_ge_3": b_minus_c >= SUCCESS_B_MINUS_C_MIN,
        "p50_added_le_10ms": p50_added <= SUCCESS_P50_ADDED_MS_MAX,
        "mean_rows_le_20": arm_b.mean_rows <= SUCCESS_MEAN_ROWS_MAX,
    }
    reach_gain_qualifies = success_criteria["b_minus_a_ge_6"] and success_criteria["b_minus_c_ge_3"]

    kill_reasons: list[str] = []
    if arm_c.reach_at_5 >= arm_b.reach_at_5:
        kill_reasons.append(
            f"K1: {KILL_DESCRIPTIONS['K1']} (C={arm_c.reach_at_5} >= B={arm_b.reach_at_5})"
        )
    if b_minus_a <= KILL_K2_B_MINUS_A_MAX:
        kill_reasons.append(f"K2: {KILL_DESCRIPTIONS['K2']} (B-A={b_minus_a})")
    if arm_b.mean_rows > KILL_K3_MEAN_ROWS_MAX and not reach_gain_qualifies:
        # "with no reachability gain that survives the token cost" is
        # operationalised, in advance, as "the gain does not clear the success
        # bar against BOTH A and C".
        kill_reasons.append(
            f"K3: {KILL_DESCRIPTIONS['K3']} (mean_rows={arm_b.mean_rows}, "
            f"B-A={b_minus_a}, B-C={b_minus_c})"
        )
    if residual_rate < KILL_K4_RESIDUAL_MIN:
        kill_reasons.append(f"K4: {KILL_DESCRIPTIONS['K4']} (residual={residual_rate:.3f})")

    notes: list[str] = []
    all_success = all(success_criteria.values())

    if kill_reasons:
        if all_success:
            notes.append(
                "SUCCESS criteria were all met but a KILL criterion also fired. "
                "KILL takes precedence by pre-registration — see thresholds.py "
                "docstring, encoding (2)."
            )
        return Verdict(
            verdict="KILL",
            kill_reasons=kill_reasons,
            success_criteria=success_criteria,
            deltas=deltas,
            notes=notes,
        )

    if all_success:
        notes.append(
            "This experiment scores a RETRIEVAL LIST. No rig in this repo scores an "
            "agent's task outcome (GRAPH_THESIS.md §5, P5). Do not flip a shipped "
            "default on this result alone without an agent-task arm."
        )
        return Verdict(
            verdict="SUCCESS",
            success_criteria=success_criteria,
            deltas=deltas,
            notes=notes,
        )

    notes.append(
        "AMBIGUOUS: re-run once with a different HNSW seed. If still ambiguous, "
        "that IS the answer — the effect is smaller than this project's "
        "measurement noise, and the correct move is FREEZE (GRAPH_THESIS.md §4-B), "
        "not build."
    )
    return Verdict(
        verdict="AMBIGUOUS",
        success_criteria=success_criteria,
        deltas=deltas,
        notes=notes,
    )
