"""No-network tests for the two-car undercut duel."""

from __future__ import annotations

import numpy as np

from f1se.sim.duel import CarPlan, car_lap_times, simulate_duel, undercut_decision


def _pace(compound, tyre_age, lap):
    # Fresh tyres much faster than worn; soft fastest fresh, degrades quickest.
    rate = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.04}[compound]
    base = {"SOFT": 89.5, "MEDIUM": 90.0, "HARD": 90.4}[compound]
    return base + rate * tyre_age


def test_car_lap_times_resets_age_after_pit():
    plan = CarPlan("MEDIUM", tyre_age=10, pit_lap=25, new_compound="HARD")
    t = car_lap_times(plan, current_lap=20, end_lap=50, pace_fn=_pace, pit_loss_s=20.0)
    assert len(t) == 30                                  # laps 21..50
    # Lap 25 (index 4) is the in-lap: medium age 15 + 20s pit loss.
    assert np.isclose(t[4], _pace("MEDIUM", 15, 25) + 20.0)
    # Lap 26 (index 5): fresh hard, age 1.
    assert np.isclose(t[5], _pace("HARD", 1, 26))


def test_no_noise_duel_is_deterministic_gap():
    you = CarPlan("HARD", 20, pit_lap=None, new_compound="HARD")
    rival = CarPlan("HARD", 20, pit_lap=None, new_compound="HARD")
    # Identical cars, you 2s behind, neither pits -> still exactly 2s behind.
    gaps = simulate_duel(you, rival, current_lap=20, total_laps=40, pace_fn=_pace,
                         gap_s=2.0, pace_noise_s=0.0, n_runs=5)
    assert np.allclose(gaps, 2.0)


def test_undercut_jumps_a_rival_staying_out_on_old_tyres():
    # You 2s behind on a worn medium; pit now to fresh soft. Rival stays out on a
    # 25-lap-old hard until lap 45. Fresh-tyre pace should jump you ahead.
    out = undercut_decision(
        current_lap=20, total_laps=55, pace_fn=_pace, gap_s=2.0,
        your_compound="MEDIUM", your_age=20, your_new_compound="SOFT",
        rival_compound="HARD", rival_age=25, rival_new_compound="HARD", rival_pit_lap=45,
        pace_noise_s=0.0, pit_loss_s=20.0,
    )
    assert out["undercut_works"]
    assert out["undercut"]["final_gap_s"] < 0          # you end ahead
    assert out["undercut"]["p_ahead"] == 1.0           # deterministic, clearly ahead


def test_cover_preferred_when_rival_can_react_immediately():
    # If the rival pits next lap, there's no undercut window — pitting "now" is
    # the same lap, so the undercut gains nothing and covering is the call.
    out = undercut_decision(
        current_lap=20, total_laps=55, pace_fn=_pace, gap_s=1.0,
        your_compound="HARD", your_age=10, your_new_compound="HARD",
        rival_compound="HARD", rival_age=10, rival_new_compound="HARD", rival_pit_lap=21,
        pace_noise_s=0.0, pit_loss_s=20.0,
    )
    assert not out["undercut_works"]
    assert abs(out["undercut_gain_s"]) < 0.5          # no real window -> ~no gain
    assert "Hold / cover" in out["verdict"]
