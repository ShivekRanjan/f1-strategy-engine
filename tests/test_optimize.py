"""No-network tests for the Phase 4 strategy optimiser."""

from __future__ import annotations

import numpy as np
import pytest

from f1se.sim.optimize import enumerate_strategies, recommend_strategy
from f1se.sim.safety_car import SafetyCarModel


def _pace(compound, tyre_age, lap):
    rate = {"SOFT": 0.12, "MEDIUM": 0.05, "HARD": 0.03}[compound]
    return 90.0 + rate * tyre_age


def test_enumerate_respects_constraints():
    strats = enumerate_strategies(50, max_stops=2, pit_grid_step=5, min_stint=10)
    assert strats
    for s in strats:
        # 1..2 stops, ascending pit laps, >=2 distinct compounds, min stint honoured.
        assert 1 <= s.n_stops <= 2
        assert len(set(s.compounds)) >= 2
        bounds = [0, *s.pit_laps, 50]
        assert all(bounds[i + 1] - bounds[i] >= 10 for i in range(len(bounds) - 1))


def test_enumerate_excludes_single_compound_plans():
    strats = enumerate_strategies(50, max_stops=1, pit_grid_step=5, min_stint=10)
    assert all(len(set(s.compounds)) >= 2 for s in strats)  # dry-race 2-compound rule


def test_recommend_picks_lower_expected_time():
    # No SC, no noise -> deterministic; optimiser must return the true minimum.
    res = recommend_strategy(
        50, _pace, sc_model=None, pace_noise_s=0.0,
        objective="mean", pit_grid_step=5, min_stint=10, n_runs=4,
    )
    # Brute-force the same space to confirm the optimum.
    cands = enumerate_strategies(50, pit_grid_step=5, min_stint=10)
    from f1se.sim.simulate import simulate_race
    best = min(cands, key=lambda s: simulate_race(s, 50, _pace, sc_model=None,
                                                  pace_noise_s=0.0, n_runs=4).mean)
    best_t = simulate_race(best, 50, _pace, sc_model=None, pace_noise_s=0.0, n_runs=4).mean
    res_t = float(np.mean(res.best_samples))
    assert np.isclose(res_t, best_t, atol=1e-6)


def test_shortlist_ordered_and_best_wins_most():
    res = recommend_strategy(
        60, _pace, sc_model=SafetyCarModel(prob_per_lap=0.02), n_runs=1500,
        pit_grid_step=6, min_stint=10, top_k=4, seed=3,
    )
    scores = [row["score"] for row in res.shortlist]
    assert scores == sorted(scores)                 # shortlist sorted by objective
    assert res.shortlist[0]["rank"] == 1
    # The chosen best should win the majority of paired races vs the runner-up.
    assert res.shortlist[1]["win_prob_vs_best"] < 0.5
    assert res.n_evaluated > len(res.shortlist)


def test_risk_objective_can_change_the_pick():
    # mean vs p85 need not agree; both must return a valid enumerated strategy.
    common = dict(sc_model=SafetyCarModel(prob_per_lap=0.03, mean_duration=5),
                  n_runs=1500, pit_grid_step=6, min_stint=10, seed=7)
    mean_pick = recommend_strategy(60, _pace, objective="mean", **common).best
    p85_pick = recommend_strategy(60, _pace, objective="p85", **common).best
    space = set(enumerate_strategies(60, pit_grid_step=6, min_stint=10))
    assert mean_pick in space and p85_pick in space


def test_max_stint_constraint_prunes_long_stints():
    # Cap SOFT at 12 laps: no strategy may run a soft stint longer than that.
    limits = {"SOFT": 12, "MEDIUM": 40, "HARD": 40}
    strats = enumerate_strategies(50, max_stops=2, pit_grid_step=4, min_stint=8,
                                  max_stint=limits)
    assert strats
    for s in strats:
        bounds = [0, *s.pit_laps, 50]
        for k, comp in enumerate(s.compounds):
            length = bounds[k + 1] - bounds[k]
            assert length <= limits[comp]


def test_invalid_objective_raises():
    with pytest.raises(ValueError):
        recommend_strategy(50, _pace, objective="nope", n_runs=2)
