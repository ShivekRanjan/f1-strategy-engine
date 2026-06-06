"""Phase 3 — Monte Carlo race simulator + safety-car hazard model. STUB.

Given a starting situation and a candidate strategy (stint lengths + compounds),
roll the race forward many times, sampling per-lap pace from the models and
safety-car deployments from a hazard model, to produce a *distribution* of
finishing outcomes (race time / position) — not a point estimate.

This is the heart of the "decision under uncertainty" framing and the project's
real differentiator. Protect it over the fancier DL work if time gets tight.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Strategy:
    """A candidate plan: pit laps and the compound fitted at each stint."""

    pit_laps: tuple[int, ...]
    compounds: tuple[str, ...]


def sample_safety_car(total_laps: int, **kwargs) -> list[int]:
    """Sample safety-car deployment laps from the hazard model. TODO Phase 3."""
    raise NotImplementedError("Phase 3: safety-car hazard model")


def simulate_race(
    strategy: Strategy,
    *,
    n_runs: int = 1000,
    **kwargs,
):
    """Monte Carlo a strategy; return a distribution of outcomes. TODO Phase 3."""
    raise NotImplementedError("Phase 3: Monte Carlo race simulator")
