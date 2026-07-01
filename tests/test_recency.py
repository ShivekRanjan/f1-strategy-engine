"""Recency-weighting — keep the models responsive to mid-season car upgrades."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import fit_linear_baseline, recency_weights
from f1se.standalone.podium import build_features


def test_recency_weights_halve_each_halflife():
    laps = pd.DataFrame({"year": 2024, "round": [1, 2, 3]})
    # halflife 1 race: newest (round 3) = 1.0, round 2 = 0.5, round 1 = 0.25.
    assert np.allclose(recency_weights(laps, 1.0), [0.25, 0.5, 1.0])
    assert recency_weights(laps, None) is None


def _stint(slope: float, rnd: int) -> pd.DataFrame:
    ages = np.arange(1, 16)
    return pd.DataFrame({
        "year": 2024, "round": rnd, "event_name": f"GP{rnd}", "driver": "X", "stint": 1,
        "compound": "MEDIUM", "tyre_life": ages.astype(float),
        "lap_time_fuel_corr_s": 90.0 + slope * ages,
    })


def test_recency_weighting_pulls_slope_toward_recent_races():
    # Round 1 degraded gently (0.05), round 2 steeply (0.15) — e.g. post-upgrade.
    df = pd.concat([_stint(0.05, 1), _stint(0.15, 2)], ignore_index=True)
    flat = fit_linear_baseline(df, min_laps=5).compound_slope["MEDIUM"]
    rec = fit_linear_baseline(df, min_laps=5, recency_halflife=1.0).compound_slope["MEDIUM"]
    assert rec > flat                       # recent steep race weighted more
    assert abs(rec - 0.15) < abs(flat - 0.15)   # ...and closer to the recent truth


def test_form_recency_reacts_faster_to_a_step_change():
    # A driver who jumps from the back (P15) to the front (P2) for the last races.
    pos = [15, 15, 15, 15, 2, 2]
    df = pd.DataFrame([
        {"year": 2024, "round": i, "event_name": f"GP{i}", "driver": "AAA", "team": "T",
         "grid": p, "position": p, "points": 26 if p == 2 else 0, "status": "Finished"}
        for i, p in enumerate(pos, start=1)
    ])
    flat = build_features(df)                       # flat rolling window
    rec = build_features(df, recency_halflife=2.0)  # exponential decay
    last_flat = flat[flat["round"] == 6]["driver_form_pos"].iloc[0]
    last_rec = rec[rec["round"] == 6]["driver_form_pos"].iloc[0]
    assert last_rec < last_flat   # recency sees the step change sooner (better form)
