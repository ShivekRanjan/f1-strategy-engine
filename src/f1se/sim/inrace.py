"""In-race re-optimisation — "given where we are NOW, what should we do?"

The Phase 4 optimiser plans from lights-out. This re-plans from the *current*
race state (lap now, current tyre and its age, compounds already used, optional
safety car), recommending the best strategy for the laps that remain. It's the
brain behind the live/replay predictor: feed it the current state each lap and it
re-decides. Reuses the same vectorised simulator core and common-random-numbers
comparison as the full-race optimiser.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

import numpy as np

from f1se.sim.optimize import DRY_COMPOUNDS, OBJECTIVES
from f1se.sim.safety_car import SafetyCarModel
from f1se.sim.simulate import PaceFn, draw_scenarios, race_totals


@dataclass(frozen=True)
class RaceState:
    """Where the car is right now."""

    total_laps: int
    current_lap: int                       # laps completed
    current_compound: str
    tyre_age: int                          # age of the tyres currently fitted
    compounds_used: tuple[str, ...] = ()   # distinct compounds used so far (incl. current)

    @property
    def laps_remaining(self) -> int:
        return self.total_laps - self.current_lap


@dataclass(frozen=True)
class RemainingPlan:
    """A plan for the rest of the race: future pit laps + the compound at each."""

    future_pits: tuple[int, ...]
    future_compounds: tuple[str, ...]      # one per future pit (the tyre fitted)

    def __post_init__(self):
        if len(self.future_pits) != len(self.future_compounds):
            raise ValueError("future_pits and future_compounds must align")


def remaining_arrays(state: RaceState, plan: RemainingPlan, pace_fn: PaceFn):
    """Per-remaining-lap green lap time + pit mask for laps current_lap+1..total."""
    bounds = [state.current_lap, *plan.future_pits, state.total_laps]
    comps = [state.current_compound, *plan.future_compounds]
    pit_set = set(plan.future_pits)
    green, pit = [], []
    for k, comp in enumerate(comps):
        for lap in range(bounds[k] + 1, bounds[k + 1] + 1):
            # The ongoing stint (k==0) continues ageing; later stints start fresh.
            age = state.tyre_age + (lap - state.current_lap) if k == 0 else lap - bounds[k]
            green.append(pace_fn(comp, age, lap))
            pit.append(lap in pit_set)
    return np.array(green, dtype=float), np.array(pit, dtype=bool)


def enumerate_remaining(
    state: RaceState,
    *,
    compounds: tuple[str, ...] = DRY_COMPOUNDS,
    max_future_stops: int = 2,
    pit_grid_step: int = 2,
    min_stint: int = 6,
    max_stint: dict[str, int] | None = None,
) -> list[RemainingPlan]:
    """Generate legal remaining plans from the current state.

    Enforces: future stints within [min_stint, max_stint[compound]]; the ongoing
    stint's total length capped too; and the dry-race rule that >=2 distinct
    compounds are used across the *whole* race (counting what's already been run).
    """
    used = set(state.compounds_used) | {state.current_compound}
    cur, total = state.current_lap, state.total_laps
    grid = list(range(cur + min_stint, total - min_stint + 1, pit_grid_step))
    plans: list[RemainingPlan] = []

    def ongoing_ok(first_pit: int) -> bool:
        length = state.tyre_age + (first_pit - cur)
        return max_stint is None or length <= max_stint.get(state.current_compound, 10**9)

    def stints_ok(pits, seq) -> bool:
        bounds = [cur, *pits, total]
        for i in range(1, len(bounds) - 1):           # future stints (after a pit)
            length = bounds[i + 1] - bounds[i]
            if length < min_stint:
                return False
            if max_stint and length > max_stint.get(seq[i - 1], 10**9):
                return False
        return True

    # Option: no more stops (only legal if the 2-compound rule is already met).
    if len(used) >= 2:
        plans.append(RemainingPlan((), ()))

    for n_stops in range(1, max_future_stops + 1):
        seqs = list(product(compounds, repeat=n_stops))
        for pits in combinations(grid, n_stops):
            if not ongoing_ok(pits[0]):
                continue
            for seq in seqs:
                if len(used | set(seq)) < 2:           # whole-race 2-compound rule
                    continue
                if stints_ok(pits, seq):
                    plans.append(RemainingPlan(tuple(pits), tuple(seq)))
    return plans


@dataclass(frozen=True)
class InRaceRecommendation:
    best: RemainingPlan
    best_samples: np.ndarray
    shortlist: list[dict]
    n_evaluated: int


def recommend_remaining(
    state: RaceState,
    pace_fn: PaceFn,
    *,
    sc_model: SafetyCarModel | None = None,
    objective: str = "mean",
    n_runs: int = 2000,
    seed: int = 0,
    top_k: int = 5,
    pit_loss_s: float = 21.0,
    pit_loss_sc_s: float = 11.0,
    sc_lap_factor: float = 1.4,
    pace_noise_s: float = 0.3,
    **enum_kwargs,
) -> InRaceRecommendation:
    """Recommend the best plan for the remaining laps, with uncertainty."""
    score_fn = OBJECTIVES[objective]
    plans = enumerate_remaining(state, **enum_kwargs)
    if not plans:
        raise ValueError("no legal remaining plans from this state")

    rem = state.laps_remaining
    sc_mask, noise = draw_scenarios(rem, n_runs, sc_model=sc_model,
                                    pace_noise_s=pace_noise_s, seed=seed)

    totals = np.empty((len(plans), n_runs))
    for i, plan in enumerate(plans):
        green, pit = remaining_arrays(state, plan, pace_fn)
        totals[i] = race_totals(green, pit, sc_mask, noise, pit_loss_s=pit_loss_s,
                                pit_loss_sc_s=pit_loss_sc_s, sc_lap_factor=sc_lap_factor)

    scores = np.array([score_fn(totals[i]) for i in range(len(plans))])
    order = np.argsort(scores)
    best_i = int(order[0])

    def _label(p: RemainingPlan) -> str:
        if not p.future_pits:
            return "stay out (no more stops)"
        return " then ".join(f"lap {lp}->{c}" for lp, c in zip(p.future_pits, p.future_compounds))

    shortlist = [{
        "rank": r + 1,
        "plan": _label(plans[i]),
        "future_pits": list(plans[i].future_pits),
        "future_compounds": list(plans[i].future_compounds),
        "mean_remaining_s": float(np.mean(totals[i])),
        "win_prob_vs_best": float(np.mean(totals[i] < totals[best_i])),
    } for r, i in enumerate(order[:top_k])]

    return InRaceRecommendation(
        best=plans[best_i], best_samples=totals[best_i],
        shortlist=shortlist, n_evaluated=len(plans),
    )
