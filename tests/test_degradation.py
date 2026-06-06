"""No-network tests for the Phase 2 within-stint degradation baseline."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import (
    cross_val_mae,
    fit_linear_baseline,
    naive_pace_loss_mae,
    pace_loss_mae,
    predict_corrected_laptime,
    predict_pace_loss,
)


def _dry_laps(slope_by_track, *, compound="MEDIUM", n=30, base=90.0):
    """Synthetic dry laps with a known per-track degradation slope (no noise).

    One stint per race (driver D, stint 1) so the fixed-effects fit can recover
    the slope from within-stint spread.
    """
    rows = []
    for rnd, (track, slope) in enumerate(slope_by_track.items(), start=1):
        for age in range(1, n + 1):
            rows.append(
                {
                    "year": 2023,
                    "round": rnd,
                    "driver": "D",
                    "stint": 1,
                    "event_name": track,
                    "compound": compound,
                    "tyre_life": float(age),
                    "lap_time_fuel_corr_s": base + slope * age,
                }
            )
    return pd.DataFrame(rows)


def test_baseline_recovers_known_slope_per_track():
    laps = _dry_laps({"Spain": 0.05, "Monaco": 0.02})
    model = fit_linear_baseline(laps, group_cols=("event_name", "compound"))
    assert np.isclose(predict_pace_loss(model, "MEDIUM", 10, track="Spain"), 0.5, atol=1e-6)
    assert np.isclose(predict_pace_loss(model, "MEDIUM", 10, track="Monaco"), 0.2, atol=1e-6)


def test_unseen_track_falls_back_to_compound_slope():
    laps = _dry_laps({"Spain": 0.05})
    model = fit_linear_baseline(laps, group_cols=("event_name", "compound"))
    loss = predict_pace_loss(model, "MEDIUM", 10, track="Monza")  # never seen
    assert np.isclose(loss, 0.5, atol=1e-6)


def test_predict_corrected_laptime_on_seen_track_uses_base_pace():
    laps = _dry_laps({"Spain": 0.05}, base=90.0)
    model = fit_linear_baseline(laps, group_cols=("event_name", "compound"))
    assert np.isclose(predict_corrected_laptime(model, "MEDIUM", 0, track="Spain"), 90.0, atol=1e-6)
    assert np.isclose(predict_corrected_laptime(model, "MEDIUM", 10, track="Spain"), 90.5, atol=1e-6)


def test_linear_beats_naive_when_degradation_is_real():
    laps = _dry_laps({"Spain": 0.08, "Monaco": 0.06})
    model = fit_linear_baseline(laps)
    # Perfect linear data -> near-zero pace-loss error, far below the slope-0 bar.
    assert pace_loss_mae(model, laps) < naive_pace_loss_mae(laps)
    assert pace_loss_mae(model, laps) < 1e-6


def test_cross_val_beats_naive_on_held_out_races():
    laps = _dry_laps({f"R{r}": 0.05 for r in range(6)})  # 6 races -> GroupKFold
    res = cross_val_mae(laps, n_splits=3, min_laps=10)
    assert res["n_folds"] == 3
    # Held-out race resolves via compound slope (0.05) -> recovers shape exactly.
    assert res["linear_mae"] < res["naive_mae"]
