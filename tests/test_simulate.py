"""No-network tests for the Monte Carlo race simulator and SC hazard model."""

from __future__ import annotations

import numpy as np
import pytest

from f1se.sim.safety_car import SafetyCarModel
from f1se.sim.simulate import (
    Strategy,
    compare_strategies,
    simulate_race,
    stint_plan,
)


# A simple deterministic pace: 90s base + 0.05 s per lap of tyre age, no fuel.
def _pace(compound, tyre_age, lap):
    rate = {"SOFT": 0.10, "MEDIUM": 0.05, "HARD": 0.03}[compound]
    return 90.0 + rate * tyre_age


def test_strategy_validation():
    with pytest.raises(ValueError):
        Strategy(compounds=("SOFT", "HARD"), pit_laps=())  # mismatched lengths
    with pytest.raises(ValueError):
        Strategy(compounds=("S", "M", "H"), pit_laps=(30, 20))  # not ascending


def test_stint_plan_covers_every_lap_with_resetting_age():
    s = Strategy(compounds=("MEDIUM", "HARD"), pit_laps=(20,))
    plan = stint_plan(s, total_laps=50)
    assert len(plan) == 50
    # Stint 1: laps 1..20 medium, age 1..20; stint 2: laps 21..50 hard, age 1..30.
    assert plan[0] == ("MEDIUM", 1, False)
    assert plan[19] == ("MEDIUM", 20, True)   # pit in-lap
    assert plan[20] == ("HARD", 1, False)     # fresh tyre, age resets
    assert plan[49] == ("HARD", 30, False)


def test_no_sc_no_noise_is_exact():
    s = Strategy(compounds=("MEDIUM", "MEDIUM"), pit_laps=(25,))
    res = simulate_race(s, 50, _pace, sc_model=None, pace_noise_s=0.0,
                        pit_loss_s=20.0, n_runs=10)
    # Sum of green laps + one pit stop, identical across runs.
    plan = stint_plan(s, 50)
    expected = sum(_pace(c, a, i + 1) for i, (c, a, _) in enumerate(plan)) + 20.0
    assert np.allclose(res.samples, expected)
    assert res.p_safety_car == 0.0


def test_extra_stop_adds_one_pit_loss():
    one = Strategy(("MEDIUM", "MEDIUM"), (25,))
    two = Strategy(("MEDIUM", "MEDIUM", "MEDIUM"), (17, 34))
    kw = dict(sc_model=None, pace_noise_s=0.0, pit_loss_s=20.0, n_runs=5)
    # Same pace model; the only difference is one extra 20s stop.
    r1 = simulate_race(one, 50, _pace, **kw)
    r2 = simulate_race(two, 50, _pace, **kw)
    # Two-stop runs shorter stints (less degradation) but pays an extra stop.
    assert r2.mean - r1.mean == pytest.approx(20.0 + _deg_delta(one, two), abs=1e-6)


def _deg_delta(one, two):
    # Degradation-time difference between the two plans (deterministic).
    p1 = stint_plan(one, 50); p2 = stint_plan(two, 50)
    g1 = sum(_pace(c, a, i + 1) for i, (c, a, _) in enumerate(p1))
    g2 = sum(_pace(c, a, i + 1) for i, (c, a, _) in enumerate(p2))
    return g2 - g1


def test_softer_compound_degrades_faster_costs_more_over_long_stint():
    soft = Strategy(("SOFT",), ())
    hard = Strategy(("HARD",), ())
    kw = dict(sc_model=None, pace_noise_s=0.0, n_runs=3)
    assert simulate_race(soft, 50, _pace, **kw).mean > simulate_race(hard, 50, _pace, **kw).mean


def test_safety_car_increases_race_time_and_is_recorded():
    s = Strategy(("MEDIUM", "MEDIUM"), (25,))
    no_sc = simulate_race(s, 50, _pace, sc_model=None, pace_noise_s=0.0, n_runs=500)
    # Force SC almost every lap -> much slower, and p_safety_car high.
    heavy = SafetyCarModel(prob_per_lap=0.5, mean_duration=4)
    with_sc = simulate_race(s, 50, _pace, sc_model=heavy, pace_noise_s=0.0, n_runs=500, seed=1)
    assert with_sc.mean > no_sc.mean
    assert with_sc.p_safety_car > 0.9


def test_sc_mask_shape_and_rate():
    sc = SafetyCarModel(prob_per_lap=0.0)
    rng = np.random.default_rng(0)
    mask = sc.sample_mask(50, 100, rng)
    assert mask.shape == (100, 50)
    assert not mask.any()  # zero hazard -> no SC


def test_compare_strategies_sorted_by_mean():
    s1 = Strategy(("SOFT",), ())
    s2 = Strategy(("HARD",), ())
    ranked = compare_strategies([s1, s2], 50, _pace, sc_model=None, pace_noise_s=0.0, n_runs=3)
    assert [r.strategy.compounds for r in ranked] == [("HARD",), ("SOFT",)]
