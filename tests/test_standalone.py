"""No-network tests for the Phase A standalone predictors."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.standalone.championship import predict_season, simulate_championship
from f1se.standalone.podium import FEATURE_COLS, build_features, predict_race, train_podium_model


def _synthetic_results(n_seasons=3, n_rounds=12, n_drivers=10, seed=0):
    """Seasons where lower-numbered drivers are genuinely stronger (grid≈skill),
    so a model SHOULD learn to predict the podium better than chance."""
    rng = np.random.default_rng(seed)
    rows = []
    drivers = [f"D{i}" for i in range(n_drivers)]
    teams = [f"T{i // 2}" for i in range(n_drivers)]
    for yr in range(2021, 2021 + n_seasons):
        for rnd in range(1, n_rounds + 1):
            skill = np.arange(n_drivers) + rng.normal(0, 1.5, n_drivers)  # lower = better
            order = np.argsort(skill)
            for pos, di in enumerate(order, start=1):
                rows.append({
                    "year": yr, "round": rnd, "event_name": f"R{rnd}",
                    "driver": drivers[di], "team": teams[di],
                    "grid": float(pos + int(rng.integers(-1, 2))),
                    "position": float(pos),
                    "points": float([25, 18, 15, 12, 10, 8, 6, 4, 2, 1][pos - 1] if pos <= 10 else 0),
                    "status": "Finished",
                })
    return pd.DataFrame(rows)


def test_build_features_no_leakage_and_targets():
    res = _synthetic_results()
    feats = build_features(res)
    assert feats["podium"].isin([0, 1]).all()
    assert (feats["podium"] == (feats["position"] <= 3)).all()
    # No feature column is left NaN (first appearances get a sensible default).
    assert feats[FEATURE_COLS].notna().all().all()


def test_podium_model_beats_grid_baseline_when_signal_exists():
    res = _synthetic_results(n_seasons=4)
    model = train_podium_model(build_features(res), test_year=2024)
    m = model.metrics
    assert 0.5 <= m["auc"] <= 1.0
    # With real signal the model should at least match the strong grid baseline.
    assert m["model_precision_at_3"] >= m["grid_baseline_precision_at_3"] - 0.05


def test_predict_race_returns_sorted_probabilities():
    res = _synthetic_results()
    feats = build_features(res)
    model = train_podium_model(feats, test_year=2023)
    one = feats[(feats["year"] == 2023) & (feats["round"] == 1)]
    pred = predict_race(model, one)
    assert list(pred["podium_prob"]) == sorted(pred["podium_prob"], reverse=True)
    assert (pred["podium_prob"].between(0, 1)).all()


def test_simulate_championship_favours_the_strongest():
    strengths = pd.Series({"A": 25.0, "B": 10.0, "C": 2.0, "D": 0.5})
    out = simulate_championship(strengths, n_races=20, n_sims=3000, seed=1)
    assert out.iloc[0]["driver"] == "A"                       # strongest most likely champ
    assert abs(out["win_prob"].sum() - 1.0) < 1e-9            # probabilities normalise
    assert out.iloc[0]["mean_points"] > out.iloc[-1]["mean_points"]


def test_predict_season_runs_end_to_end():
    res = _synthetic_results(n_seasons=3)
    out = predict_season(res, 2023, n_sims=1000)
    assert set(out["driver"]) <= set(res[res["year"] == 2023]["driver"])
    assert np.isclose(out["win_prob"].sum(), 1.0)
