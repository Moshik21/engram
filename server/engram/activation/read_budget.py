"""The read budget a bounded graph traversal spends, and why it stopped.

Three strategies (BFS, PPR, ACT-R) each read one node's adjacency per step and
each has to answer the same two questions: *may I start another read?* and
*what do I tell the caller about how far I got?* They carried three identical
copies of the answer, and the copies were identically wrong.

## Why the estimator is an EWMA and not a running max

The bail exists because overshooting a deadline is not free for a bounded
caller — the overshoot comes out of the recall stages that run after this one.
So before starting a read the traversal predicts whether that read fits in the
remaining budget.

The first version predicted with a **running max that never decayed**
(``slowest_read = max(slowest_read, elapsed)``). One slow read therefore
poisoned the estimate for the rest of the traversal. Measured on a 0.5 ms/read
store with a 50 ms budget and ``max_reads=64``:

    no outlier                 64 reads / 498 reached / 43.4 ms
    ONE 25 ms read injected     6 reads /  34 reached / 30.5 ms  <- 93% of reach gone
    cold 30 ms FIRST read       1 read  /   0 reached / 32.1 ms  <- 18 ms of 50 unspent

The estimator's error is **asymmetric**, and that asymmetry decides the design:

* Predicting **too low** costs at most ONE read of overshoot, and the stage's
  outer wall clock (``retrieval_spread_timeout_ms``, 75 ms) is deliberately set
  above the traversal budget (``retrieval_spread_traversal_budget_ms``, 50 ms)
  with exactly that headroom in mind.
* Predicting **too high** costs the entire remaining traversal — measured, 93%
  of the reach, with 40% of the budget left unspent.

Live per-read latency spans 0.1–150 ms (a ~1500x spread), and per ticket #31 the
native executor is ``max_workers=4`` and contended, so outlier reads are the
normal case in the degraded regime rather than an edge case. Against a spread
that wide, an estimator that tracks the WORST read is tracking the wrong
statistic: it converges on the tail and then refuses to work.

Rejected alternatives, and why:

* **Percentile (p90) of observed reads** — needs the sample kept and sorted, and
  with ``max_reads=64`` a p90 is still dragged up by one read in ten. It also
  reacts to a genuine regime change (store goes cold and STAYS cold) only after
  ~half the sample, where an EWMA reacts in ~3 reads.
* **Arm the bail only after N reads** — fixes a cold first read but does nothing
  about an outlier at read 30, and buys its warm-up by allowing N unpredicted
  reads at the exact moment the store is slowest.

An EWMA seeded at **0.0** gets both properties from one line. It converges on
the *typical* read (the statistic the prediction is actually about), it forgets
a spike geometrically, and because it starts optimistic it has a built-in
warm-up: after a single cold 30 ms first read the estimate is 9 ms, not 30 ms,
so the traversal keeps going instead of stopping dead with a third of its budget
unspent. That is the "require evidence before the bail arms" behaviour without a
counter or a second knob.

What is knowingly given up: the running max guaranteed "never overshoot the
deadline after the first read". The EWMA guarantees "never START a read after
the deadline, and overshoot by at most one read's cost". On a store where every
read costs a large fraction of the budget the two estimators do the same work
(verified for 15 ms and 20 ms reads against a 50 ms budget: both do 3 and 2
reads respectively); they diverge only where the running max was destroying
reach.

``READ_COST_EWMA_ALPHA`` is a module constant rather than a config field on
purpose. It is not an operator-tunable quantity — it encodes the cost asymmetry
above, not a policy — and every config field this project has added without a
consumer has turned into a frozen question with zero read sites.
"""

from __future__ import annotations

import time

# Weight on the newest read when updating the cost estimate. At 0.3 a spike
# decays to ~a third of its peak in 3 reads and to noise in ~8, while a
# sustained slowdown is tracked within ~3 reads.
READ_COST_EWMA_ALPHA = 0.3

# Stop reasons. The first two are the traversal finishing its own work; the last
# three are a bound firing, and ``PREDICTED_COST`` is the only one that can end a
# traversal while budget remains.
STOP_COMPLETE = "complete"
STOP_MAX_READS = "max_reads"
STOP_DEADLINE = "deadline"
STOP_PREDICTED_COST = "predicted_cost"


class ReadBudget:
    """Decides whether the next adjacency read may start, and records why not.

    Usage mirrors the shape every strategy already had::

        budget = ReadBudget(max_reads=max_reads, deadline=deadline, stats=traversal_stats)
        while queue:
            if not budget.start_read():
                break
            neighbors = await provider.get_active_neighbors_with_weights(...)
            budget.finish_read()
        budget.close()

    Both bounds default to UNBOUNDED. The bound belongs to the caller that owes
    a user a latency budget (recall); the offline (dream) and write-path
    (prospective memory) callers pass neither and are never truncated.

    ``stats`` is a caller-owned dict, updated **after every read** rather than
    once at the end. That is deliberate: the recall stage wraps the traversal in
    an ``asyncio.wait_for`` that CANCELS it, and a cancelled coroutine never
    reaches its own return statement — so a sink written at the end would be
    empty on exactly the failure it exists to describe. Written per read, the
    caller can still see how many reads were discarded and why.
    """

    __slots__ = (
        "_est_s",
        "_max_s",
        "_read_started",
        "_stats",
        "_stopped_at",
        "deadline",
        "max_reads",
        "reads",
        "stop_reason",
    )

    def __init__(
        self,
        *,
        max_reads: int | None = None,
        deadline: float | None = None,
        stats: dict[str, float | str] | None = None,
    ) -> None:
        self.max_reads = max_reads or None
        self.deadline = deadline
        self.reads = 0
        self.stop_reason = STOP_COMPLETE
        self._est_s = 0.0
        self._max_s = 0.0
        self._read_started = 0.0
        self._stopped_at: float | None = None
        self._stats = stats
        # Publish before the first read so a traversal that is cancelled during
        # read #1, or that never enters its loop at all, still reports itself.
        self._publish()

    def start_read(self) -> bool:
        """True if a read may start. False stops the traversal and records why."""
        if self.max_reads is not None and self.reads >= self.max_reads:
            return self._stop(STOP_MAX_READS, None)

        now = time.monotonic()
        if self.deadline is not None:
            # Hard: the budget is already gone. Never START a read past the
            # deadline, whatever the estimator believes.
            if now >= self.deadline:
                return self._stop(STOP_DEADLINE, now)
            # Predictive: this read is not expected to finish in what is left.
            if now + self._est_s >= self.deadline:
                return self._stop(STOP_PREDICTED_COST, now)

        self._read_started = now
        self.reads += 1
        # Published on START, so a traversal cancelled mid-read still reports
        # the read that was thrown away. ``reads`` counts reads STARTED.
        self._publish()
        return True

    def finish_read(self) -> None:
        """Fold the read just completed into the cost estimate."""
        elapsed = time.monotonic() - self._read_started
        if elapsed > self._max_s:
            self._max_s = elapsed
        self._est_s += READ_COST_EWMA_ALPHA * (elapsed - self._est_s)
        self._publish()

    def close(self) -> None:
        """Mark the traversal stopped on its own terms (its loop ran out of work)."""
        if self._stopped_at is None:
            self._stopped_at = time.monotonic()
        self._publish()

    def _stop(self, reason: str, now: float | None) -> bool:
        self.stop_reason = reason
        self._stopped_at = time.monotonic() if now is None else now
        self._publish()
        return False

    def _publish(self) -> None:
        """Write this traversal's provenance into the caller-owned sink.

        ``budget_unspent_ms`` is OMITTED in two cases, both deliberate
        (``INSTRUMENT_AUDIT.md``, pattern 1 — absence beats a plausible number):

        * **No deadline.** An unbounded traversal has no unspent budget, and
          ``0.0`` there would read as "it used every millisecond it had".
        * **The traversal has not stopped yet.** Mid-traversal the only value
          available is "budget remaining right now", which is a different
          quantity from "budget left unspent when it stopped" and is wrong by
          however long the traversal ran after this publish. A traversal killed
          by the caller's outer cancel therefore reports its reads but reports
          NO unspent figure — the stop instant is genuinely unknown.
        """
        sink = self._stats
        if sink is None:
            return
        sink["reads"] = self.reads
        sink["stop_reason"] = self.stop_reason
        sink["read_ms_max"] = round(self._max_s * 1000, 4)
        sink["read_ms_est"] = round(self._est_s * 1000, 4)
        if self.deadline is not None and self._stopped_at is not None:
            unspent = max(0.0, self.deadline - self._stopped_at)
            sink["budget_unspent_ms"] = round(unspent * 1000, 4)
