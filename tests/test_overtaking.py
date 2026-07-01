"""No-network tests for the track-position (overtaking) prior."""

from __future__ import annotations

import pandas as pd

from f1se.models.overtaking import OvertakingPrior
from f1se.sim.optimize import recommend_strategy


def _results():
    rows = []
    for rnd in range(1, 4):
        for g in range(1, 11):
            # "Hard": finish == grid (no shuffle -> hard to overtake).
            rows.append({"event_name": "Hard", "year": 2023, "round": rnd, "grid": g,
                         "position": g})
            # "Easy": order fully reversed (big shuffle -> easy to overtake).
            rows.append({"event_name": "Easy", "year": 2023, "round": rnd, "grid": g,
                         "position": 11 - g})
    return pd.DataFrame(rows)


def test_difficulty_and_penalty_track_overtaking_hardness():
    ov = OvertakingPrior.from_results(_results())
    assert ov.difficulty["Hard"] > ov.difficulty["Easy"]
    assert ov.penalty_per_stop("Hard") > ov.penalty_per_stop("Easy")
    # unknown track falls back to the default difficulty
    assert ov.penalty_per_stop("Unknown") == ov.sec_base + ov.sec_hard * ov.default_difficulty


def test_disabled_prior_is_free():
    assert OvertakingPrior.disabled().penalty_per_stop("anything") == 0.0


def _pace(compound, tyre_age, lap):
    rate = {"SOFT": 0.12, "MEDIUM": 0.06, "HARD": 0.04}[compound]
    base = {"SOFT": 89.5, "MEDIUM": 90.0, "HARD": 90.4}[compound]
    return base + rate * tyre_age


def test_stop_penalty_never_increases_the_chosen_stop_count():
    common = dict(pace_fn=_pace, n_runs=300, max_stops=2, pit_loss_s=20.0,
                  pace_noise_s=0.0, min_stint=8)
    free = recommend_strategy(50, stop_penalty_s=0.0, **common)
    penalised = recommend_strategy(50, stop_penalty_s=25.0, **common)
    assert penalised.best.n_stops <= free.best.n_stops
    # reported times stay raw (penalty is ranking-only), so mean is a real race time
    assert penalised.best_summary["mean_s"] > 1000
