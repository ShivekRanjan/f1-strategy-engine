"""No-network tests for the boosted degradation model and the head-to-head."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from f1se.models.degradation import fit_linear_baseline, linear_shape, shape_mae
from f1se.models.degradation_boosted import (
    cross_val_compare,
    fit_boosted,
    predict_shape,
)

FAST = {"n_estimators": 120, "max_depth": 3, "learning_rate": 0.1}


def _nonlinear_laps(n_races=8, n=30, curv=0.003, compound="MEDIUM"):
    """Dry laps whose degradation is quadratic in age (a 'cliff') -> a straight
    line must leave residual curvature a flexible model can exploit."""
    rng = np.random.default_rng(0)
    rows = []
    for rnd in range(1, n_races + 1):
        for age in range(1, n + 1):
            deg = curv * age**2  # nonlinear in tyre age
            rows.append(
                {
                    "year": 2023,
                    "round": rnd,
                    "driver": "D",
                    "stint": 1,
                    "event_name": f"R{rnd}",
                    "compound": compound,
                    "tyre_life": float(age),
                    "lap_time_fuel_corr_s": 90.0 + deg + rng.normal(0, 0.01),
                }
            )
    return pd.DataFrame(rows)


def test_fit_and_predict_shape_runs():
    laps = _nonlinear_laps()
    model = fit_boosted(laps, params=FAST)
    shape = predict_shape(model, laps)
    assert len(shape) == len(laps)
    assert np.isfinite(shape).all()


def test_boosted_beats_linear_on_nonlinear_degradation():
    laps = _nonlinear_laps()
    lm = fit_linear_baseline(laps, min_laps=10)
    bm = fit_boosted(laps, params=FAST)
    lin = shape_mae(linear_shape(lm, laps), laps)
    boost = shape_mae(predict_shape(bm, laps), laps)
    assert boost < lin  # curvature gives the flexible model the edge


def test_cross_val_compare_reports_all_models():
    laps = _nonlinear_laps()
    res = cross_val_compare(laps, n_splits=4, min_laps=10, params=FAST)
    for k in ("linear_mae", "boosted_mae", "naive_mae", "n_folds"):
        assert k in res
    assert res["n_folds"] == 4
    # Both modelled approaches should beat the no-degradation comparator.
    assert res["boosted_mae"] < res["naive_mae"]
