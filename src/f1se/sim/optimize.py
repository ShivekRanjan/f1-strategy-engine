"""Phase 4 — strategy optimiser. STUB.

Search the space of strategies (number of stops, pit laps, compound sequence)
and return the recommendation with quantified uncertainty — e.g. the strategy
that minimises expected race time, plus the spread and the probability it beats
the alternatives.

Stretch: in-race re-optimisation as a sequential decision problem — re-solve as
the race state evolves (safety car out, undercut threat) rather than committing
to a plan at lights-out.
"""

from __future__ import annotations

from f1se.sim.simulate import Strategy


def enumerate_strategies(total_laps: int, compounds: tuple[str, ...], **kwargs) -> list[Strategy]:
    """Generate the candidate strategy space to evaluate. TODO Phase 4."""
    raise NotImplementedError("Phase 4: strategy enumeration")


def recommend_strategy(*, n_runs: int = 1000, **kwargs):
    """Return the best strategy + uncertainty over the candidate space. TODO Phase 4."""
    raise NotImplementedError("Phase 4: strategy optimiser")
