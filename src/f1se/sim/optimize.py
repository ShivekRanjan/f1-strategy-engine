"""Phase 4 — strategy optimiser.

Searches the strategy space (number of stops, pit laps, compound sequence) and
returns the recommendation **with quantified uncertainty** — not just the lowest
expected time, but how reliably it beats the alternatives.

Every candidate is scored against the *same* sampled races (common random
numbers, drawn once via :func:`f1se.sim.simulate.draw_scenarios`), so the ranking
is paired and low-variance and "probability A beats B" is meaningful. The
objective is pluggable: minimise expected time, the median, or a high quantile
(risk-averse — protect against the bad-luck tail).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from f1se.sim.safety_car import SafetyCarModel
from f1se.sim.simulate import (
    PaceFn,
    Strategy,
    draw_scenarios,
    green_and_pit,
    race_totals,
)

DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# Objective: map a strategy's sampled total times -> a scalar to MINIMISE.
OBJECTIVES: dict[str, Callable[[np.ndarray], float]] = {
    "mean": lambda s: float(np.mean(s)),
    "median": lambda s: float(np.median(s)),
    "p85": lambda s: float(np.quantile(s, 0.85)),  # risk-averse
}


@dataclass(frozen=True)
class OptimizationResult:
    """The recommended strategy plus a ranked, uncertainty-aware shortlist."""

    best: Strategy
    best_samples: np.ndarray
    shortlist: list[dict]
    n_evaluated: int
    objective: str

    @property
    def best_summary(self) -> dict:
        s = self.best_samples
        return {
            "compounds": self.best.compounds,
            "pit_laps": self.best.pit_laps,
            "mean_s": float(np.mean(s)),
            "p50_s": float(np.quantile(s, 0.5)),
            "p90_s": float(np.quantile(s, 0.9)),
        }


def enumerate_strategies(
    total_laps: int,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
    *,
    max_stops: int = 2,
    pit_grid_step: int = 3,
    min_stint: int = 8,
) -> list[Strategy]:
    """Generate candidate strategies, honouring the realistic constraints.

    Constraints:
      * 1..``max_stops`` stops;
      * every stint at least ``min_stint`` laps;
      * pit laps on a ``pit_grid_step`` grid;
      * at least two distinct compounds used (the dry-race rule).
    """
    grid = list(range(min_stint, total_laps - min_stint + 1, pit_grid_step))
    out: list[Strategy] = []

    for n_stops in range(1, max_stops + 1):
        seqs = [c for c in product(compounds, repeat=n_stops + 1) if len(set(c)) >= 2]
        for pit_laps in combinations(grid, n_stops):
            bounds = [0, *pit_laps, total_laps]
            if any(bounds[i + 1] - bounds[i] < min_stint for i in range(len(bounds) - 1)):
                continue
            for seq in seqs:
                out.append(Strategy(compounds=seq, pit_laps=tuple(pit_laps)))
    return out


def recommend_strategy(
    total_laps: int,
    pace_fn: PaceFn,
    *,
    sc_model: SafetyCarModel | None = None,
    candidates: list[Strategy] | None = None,
    objective: str = "mean",
    n_runs: int = 2000,
    seed: int = 0,
    top_k: int = 5,
    pace_noise_s: float = 0.3,
    pit_loss_s: float = 21.0,
    pit_loss_sc_s: float = 11.0,
    sc_lap_factor: float = 1.4,
    **enum_kwargs,
) -> OptimizationResult:
    """Search the strategy space and return the best plan with uncertainty.

    ``objective`` is one of :data:`OBJECTIVES` ("mean", "median", "p85").
    """
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {list(OBJECTIVES)}")
    score_fn = OBJECTIVES[objective]

    if candidates is None:
        candidates = enumerate_strategies(total_laps, **enum_kwargs)
    if not candidates:
        raise ValueError("no candidate strategies to evaluate")

    # One shared set of sampled races for every candidate (common random numbers).
    sc_mask, noise = draw_scenarios(total_laps, n_runs, sc_model=sc_model,
                                    pace_noise_s=pace_noise_s, seed=seed)

    totals = np.empty((len(candidates), n_runs), dtype=float)
    for i, strat in enumerate(candidates):
        green_det, pit_mask = green_and_pit(strat, total_laps, pace_fn)
        totals[i] = race_totals(green_det, pit_mask, sc_mask, noise, pit_loss_s=pit_loss_s,
                                pit_loss_sc_s=pit_loss_sc_s, sc_lap_factor=sc_lap_factor)

    scores = np.array([score_fn(totals[i]) for i in range(len(candidates))])
    order = np.argsort(scores)
    best_idx = int(order[0])
    best_samples = totals[best_idx]

    shortlist: list[dict] = []
    for rank, i in enumerate(order[:top_k]):
        s = totals[i]
        shortlist.append({
            "rank": rank + 1,
            "compounds": candidates[i].compounds,
            "pit_laps": candidates[i].pit_laps,
            "n_stops": candidates[i].n_stops,
            "score": float(scores[i]),
            "mean_s": float(np.mean(s)),
            "p50_s": float(np.quantile(s, 0.5)),
            "p90_s": float(np.quantile(s, 0.9)),
            # Paired probability THIS candidate beats the chosen best.
            "win_prob_vs_best": float(np.mean(s < best_samples)),
        })

    return OptimizationResult(
        best=candidates[best_idx],
        best_samples=best_samples,
        shortlist=shortlist,
        n_evaluated=len(candidates),
        objective=objective,
    )
