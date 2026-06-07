"""No-network tests for in-race re-optimisation."""

from __future__ import annotations

import numpy as np
import pytest

from f1se.sim.inrace import (
    RaceState,
    RemainingPlan,
    enumerate_remaining,
    recommend_remaining,
    remaining_arrays,
)


def _pace(compound, tyre_age, lap):
    rate = {"SOFT": 0.20, "MEDIUM": 0.06, "HARD": 0.03}[compound]
    return 90.0 + rate * tyre_age


def test_laps_remaining():
    assert RaceState(50, 20, "MEDIUM", 10).laps_remaining == 30


def test_remaining_arrays_ages_and_pit_mask():
    state = RaceState(total_laps=50, current_lap=20, current_compound="MEDIUM",
                      tyre_age=10, compounds_used=("MEDIUM",))
    plan = RemainingPlan(future_pits=(35,), future_compounds=("HARD",))
    green, pit = remaining_arrays(state, plan, _pace)
    assert len(green) == 30 and len(pit) == 30          # laps 21..50
    assert pit.sum() == 1 and pit[35 - 21]              # pit flagged on lap 35
    # Ongoing medium ages from 10: lap 21 -> age 11; fresh hard after the stop.
    assert np.isclose(green[0], 90.0 + 0.06 * 11)
    assert np.isclose(green[35 - 20], 90.0 + 0.03 * 1)  # lap 36, fresh hard age 1


def test_enumerate_enforces_two_compound_rule():
    # Only MEDIUM used so far -> "stay out" illegal; every plan adds a 2nd compound.
    only_medium = RaceState(50, 20, "MEDIUM", 10, compounds_used=("MEDIUM",))
    plans = enumerate_remaining(only_medium, max_future_stops=1, pit_grid_step=4, min_stint=6)
    assert RemainingPlan((), ()) not in plans
    for p in plans:
        assert "HARD" in p.future_compounds or "SOFT" in p.future_compounds

    # Two compounds already used -> "stay out" is a legal option.
    two_used = RaceState(50, 20, "HARD", 10, compounds_used=("SOFT", "HARD"))
    assert RemainingPlan((), ()) in enumerate_remaining(
        two_used, max_future_stops=1, pit_grid_step=4, min_stint=6)


def test_enumerate_respects_max_stint_on_ongoing_tyre():
    # Current HARD already aged 30; capping HARD at 40 forbids running it past lap 30.
    state = RaceState(60, 25, "HARD", 30, compounds_used=("SOFT", "HARD"))
    plans = enumerate_remaining(state, max_future_stops=1, pit_grid_step=2, min_stint=6,
                                max_stint={"HARD": 40, "MEDIUM": 30, "SOFT": 26})
    for p in plans:
        if p.future_pits:  # ongoing-stint length at the pit must stay <= 40
            assert 30 + (p.future_pits[0] - 25) <= 40


def test_recommend_pits_off_a_shot_tyre():
    # On a heavily worn soft with many laps left -> pitting to a durable tyre wins.
    state = RaceState(total_laps=55, current_lap=20, current_compound="SOFT",
                      tyre_age=20, compounds_used=("MEDIUM", "SOFT"))
    rec = recommend_remaining(state, _pace, sc_model=None, pace_noise_s=0.0,
                              n_runs=4, pit_grid_step=3, min_stint=6, pit_loss_s=20.0)
    assert rec.best.future_pits, "should choose to pit off a worn soft"
    assert rec.n_evaluated > 1
    assert rec.shortlist[0]["rank"] == 1


def test_recommend_stays_out_near_the_end():
    # A few laps left, 2-compound rule already met -> just stay out.
    state = RaceState(total_laps=55, current_lap=50, current_compound="HARD",
                      tyre_age=12, compounds_used=("MEDIUM", "HARD"))
    rec = recommend_remaining(state, _pace, sc_model=None, pace_noise_s=0.0,
                              n_runs=4, pit_grid_step=2, min_stint=4, pit_loss_s=20.0)
    assert rec.best.future_pits == ()
