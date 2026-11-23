"""Phase 3 — Monte Carlo race simulator.

Given a candidate strategy (stint compounds + pit laps) and a per-lap pace
function, roll the race forward many times — sampling safety cars and per-lap
pace noise — to produce a **distribution** of total race times, not a point
estimate. This is the object the optimiser (Phase 4) searches and the engine
reports with quantified uncertainty.

Decoupled by design: the simulator takes a ``pace_fn(compound, tyre_age, lap) ->
seconds`` callable and a :class:`~f1se.sim.safety_car.SafetyCarModel`; it knows
nothing about how pace is modelled. :func:`pace_fn_from_model` wires a fitted
degradation model in, but lives here only as a convenience.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from f1se.sim.safety_car import (
    RED_LAP_FACTOR,
    RED_PIT_LOSS_S,
    VSC_LAP_FACTOR,
    VSC_PIT_LOSS_FRACTION,
    SafetyCarModel,
)

PaceFn = Callable[[str, int, int], float]


@dataclass(frozen=True)
class Strategy:
    """A candidate plan.

    ``compounds`` is the tyre for each stint; ``pit_laps`` are the in-laps where
    a stop happens (one fewer than the number of stints). Stint *k* runs from
    ``pit_laps[k-1]+1`` to ``pit_laps[k]`` (with virtual bounds 0 and total_laps).
    """

    compounds: tuple[str, ...]
    pit_laps: tuple[int, ...]

    def __post_init__(self):
        if len(self.compounds) != len(self.pit_laps) + 1:
            raise ValueError("compounds must have exactly one more entry than pit_laps")
        if list(self.pit_laps) != sorted(self.pit_laps):
            raise ValueError("pit_laps must be ascending")

    @property
    def n_stops(self) -> int:
        return len(self.pit_laps)


@dataclass(frozen=True)
class SimResult:
    """Outcome distribution of a simulated strategy (total race time, seconds)."""

    samples: np.ndarray
    strategy: Strategy
    p_safety_car: float
    # P(at least one lap neutralised by anything — VSC, full SC or red flag).
    # Kept separate from p_safety_car rather than folded into it: the two answer
    # different questions, and quietly widening p_safety_car would have changed
    # a number already on screen without telling anyone.
    p_neutralised: float = 0.0

    @property
    def mean(self) -> float:
        return float(np.mean(self.samples))

    @property
    def std(self) -> float:
        return float(np.std(self.samples))

    def quantile(self, q: float) -> float:
        return float(np.quantile(self.samples, q))

    def summary(self) -> dict:
        return {
            "mean_s": self.mean,
            "std_s": self.std,
            "p10_s": self.quantile(0.10),
            "p50_s": self.quantile(0.50),
            "p90_s": self.quantile(0.90),
            "p_safety_car": self.p_safety_car,
            "p_neutralised": self.p_neutralised,
            "n_stops": self.strategy.n_stops,
        }


def stint_plan(strategy: Strategy, total_laps: int,
               start_age: dict | None = None) -> list[tuple[str, float, bool]]:
    """Per-lap ``(compound, tyre_age, is_pit_lap)`` for laps 1..total_laps.

    ``start_age`` maps ``(compound, is_repeat)`` to the laps a set typically
    already has when a stint after a stop begins: teams race on what qualifying
    left them, and a driver's second stint on a compound is the likeliest to be
    on a used set. Missing keys (and the opening stint) start at zero.
    """
    bounds = [0, *strategy.pit_laps, total_laps]
    if strategy.pit_laps and strategy.pit_laps[-1] >= total_laps:
        raise ValueError("a pit lap is beyond the race distance")
    plan: list[tuple[str, float, bool]] = []
    pit_set = set(strategy.pit_laps)
    seen: set[str] = set()
    for k, comp in enumerate(strategy.compounds):
        extra = (start_age or {}).get((comp, comp in seen), 0.0) if k > 0 else 0.0
        seen.add(comp)
        for lap in range(bounds[k] + 1, bounds[k + 1] + 1):
            plan.append((comp, lap - bounds[k] + extra, lap in pit_set))
    return plan


def draw_scenarios(
    total_laps: int,
    n_runs: int,
    *,
    sc_model: SafetyCarModel | None = None,
    pace_noise_s: float = 0.3,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the shared random draws (sc_mask, noise) for a set of races.

    Drawing these once and reusing them across strategies gives true common
    random numbers — every strategy faces the *same* sampled races, so the
    comparison is paired and low-variance.
    """
    rng = np.random.default_rng(seed)
    if sc_model is None:
        sc_mask = np.zeros((n_runs, total_laps), dtype=np.int8)
    else:
        # Three-state: 0 green, 1 VSC, 2 full SC. Falls back to SC-only when the
        # model carries no VSC hazard, so an uncalibrated model behaves as before.
        sc_mask = sc_model.sample_states(total_laps, n_runs, rng)
    noise = rng.normal(0.0, pace_noise_s, size=(n_runs, total_laps))
    return sc_mask, noise


def green_and_pit(strategy: Strategy, total_laps: int, pace_fn: PaceFn,
                  start_age: dict | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Deterministic per-lap green lap times and the pit-lap mask for a strategy."""
    plan = stint_plan(strategy, total_laps, start_age)
    green_det = np.array([pace_fn(c, age, i + 1) for i, (c, age, _) in enumerate(plan)])
    pit_mask = np.array([is_pit for _, _, is_pit in plan], dtype=bool)
    return green_det, pit_mask


def race_totals(
    green_det: np.ndarray,
    pit_mask: np.ndarray,
    sc_mask: np.ndarray,
    noise: np.ndarray,
    *,
    pit_loss_s: float = 21.0,
    pit_loss_sc_s: float = 11.0,
    sc_lap_factor: float = 1.4,
    pit_loss_vsc_s: float | None = None,
    vsc_lap_factor: float = VSC_LAP_FACTOR,
) -> np.ndarray:
    """Total race time per sampled race, given deterministic pace + shared draws.

    ``sc_mask`` may be a boolean SC mask (legacy two-state) or an int8 state
    array from :meth:`SafetyCarModel.sample_states` — 0 green, 1 VSC, 2 full SC.
    A VSC lap is slower than green but quicker than trailing the safety car, and
    a stop taken under one is cheaper than green but dearer than under a full SC,
    because the field never bunches up.
    """
    states = sc_mask.astype(np.int8) * 2 if sc_mask.dtype == bool else sc_mask
    neutralised = states > 0
    # Derive VSC pit loss from *this track's* green loss so the discount scales
    # the way the SC one does, instead of being a constant that means something
    # different at Spa (19.5 s) than at Imola (28.5 s).
    if pit_loss_vsc_s is None:
        pit_loss_vsc_s = pit_loss_s * VSC_PIT_LOSS_FRACTION

    factor = np.select([states == 3, states == 2, states == 1],
                       [RED_LAP_FACTOR, sc_lap_factor, vsc_lap_factor], 1.0)
    pit_price = np.select([states == 3, states == 2, states == 1],
                          [RED_PIT_LOSS_S, pit_loss_sc_s, pit_loss_vsc_s], pit_loss_s)

    # Pace noise is a green-running effect; under any neutralisation the field is
    # speed-limited and lap-to-lap scatter largely disappears.
    lap_times = green_det * factor + np.where(neutralised, 0.0, noise)
    return (lap_times + pit_price * pit_mask).sum(axis=1)


def simulate_race(
    strategy: Strategy,
    total_laps: int,
    pace_fn: PaceFn,
    *,
    sc_model: SafetyCarModel | None = None,
    pit_loss_s: float = 21.0,
    pit_loss_sc_s: float = 11.0,
    sc_lap_factor: float = 1.4,
    pace_noise_s: float = 0.3,
    n_runs: int = 2000,
    seed: int = 0,
) -> SimResult:
    """Monte Carlo a strategy; return the total-race-time distribution.

    Parameters
    ----------
    pace_fn
        ``pace_fn(compound, tyre_age, lap) -> green lap time (s)`` including fuel.
    sc_model
        Safety-car hazard. ``None`` disables SC (deterministic green race).
    pit_loss_s / pit_loss_sc_s
        Time lost for a stop under green / under safety car (SC stops are cheaper).
    sc_lap_factor
        Multiplier applied to a lap's green time when run under SC.
    pace_noise_s
        Std of per-lap Gaussian pace noise.
    """
    green_det, pit_mask = green_and_pit(strategy, total_laps, pace_fn)
    sc_mask, noise = draw_scenarios(total_laps, n_runs, sc_model=sc_model,
                                    pace_noise_s=pace_noise_s, seed=seed)
    totals = race_totals(green_det, pit_mask, sc_mask, noise, pit_loss_s=pit_loss_s,
                         pit_loss_sc_s=pit_loss_sc_s, sc_lap_factor=sc_lap_factor)
    # Keep this meaning "a full safety car or red flag" (states >= 2) rather than
    # any neutralisation — with int8 states a bare .any() would silently start
    # counting VSC laps too and inflate a number the UI displays.
    p_sc = float(np.mean((sc_mask >= 2).any(axis=1))) if sc_model is not None else 0.0
    p_any = float(np.mean((sc_mask > 0).any(axis=1))) if sc_model is not None else 0.0
    return SimResult(samples=totals, strategy=strategy, p_safety_car=p_sc,
                     p_neutralised=p_any)


def pace_fn_from_model(deg_model, track: str, total_laps: int, *, sec_per_kg=0.03,
                       start_fuel_kg=110.0, cliff=None, deg_slope_delta: float = 0.0) -> PaceFn:
    """Build a ``pace_fn`` from a fitted degradation model + a fuel model.

    Green lap time = fuel-corrected pace (intercept + data-fitted degradation)
    + fuel penalty + optional :class:`~f1se.models.cliff.CliffPrior` extra. The
    cliff term is a domain prior (not data); pass ``cliff=None`` for the pure
    data model.

    ``deg_slope_delta`` shifts the degradation *slope* (s/lap) — the hook the
    :class:`~f1se.models.thermal.ThermalPrior` uses to make wear track temperature
    dependent. The effective slope is clamped ≥ 0 (tyres never gain pace with age).
    """
    from f1se.models.degradation import predict_corrected_laptime

    def pace_fn(compound: str, tyre_age: int, lap: int) -> float:
        if deg_slope_delta:
            base = predict_corrected_laptime(deg_model, compound, 0, track=track)  # intercept
            eff_slope = max(deg_model.slope(compound, track) + deg_slope_delta, 0.0)
            corrected = base + eff_slope * tyre_age
        else:
            corrected = predict_corrected_laptime(deg_model, compound, tyre_age, track=track)
        fuel_mass = start_fuel_kg * max(total_laps - lap, 0) / total_laps
        extra = cliff.extra_loss(compound, tyre_age) if cliff is not None else 0.0
        return corrected + sec_per_kg * fuel_mass + extra

    return pace_fn


def compare_strategies(
    strategies: list[Strategy],
    total_laps: int,
    pace_fn: PaceFn,
    **kwargs,
) -> list[SimResult]:
    """Simulate several strategies under the same conditions; sorted by mean time."""
    results = [simulate_race(s, total_laps, pace_fn, **kwargs) for s in strategies]
    return sorted(results, key=lambda r: r.mean)
