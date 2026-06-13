"""No-network tests for the Phase 2.5 sequence (LSTM) lap-time model.

Windowing/baseline logic is tested with plain numpy/pandas; the torch-dependent
training path self-skips when torch isn't installed (it's the heavy ``[models]``
dependency CI omits), so the suite stays green everywhere.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1se.models.lap_time import (
    FEATURE_NAMES,
    NumpyLapForecaster,
    build_live_window,
    build_sequence_windows,
    evaluate_sequence_vs_baselines,
    fit_sequence_model,
    train_and_export,
)


def _synthetic_clean_laps(years=(2023, 2024, 2025), races_per_year=4, seed=0) -> pd.DataFrame:
    """Cleaned-shaped dry laps with a known degradation trend + noise.

    One stint per driver per race; lap-to-lap pace = base + slope*age + noise, so
    a sequence model has a real (but noisy) signal to forecast.
    """
    rng = np.random.default_rng(seed)
    base_by_comp = {"SOFT": 89.5, "MEDIUM": 90.0, "HARD": 90.4}
    slope_by_comp = {"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.04}
    rows = []
    for year in years:
        for rnd in range(1, races_per_year + 1):
            for drv in ("VER", "HAM", "LEC", "NOR"):
                comp = rng.choice(list(base_by_comp))
                n = int(rng.integers(10, 26))
                for lap in range(1, n + 1):
                    age = float(lap)
                    pace = base_by_comp[comp] + slope_by_comp[comp] * age + rng.normal(0, 0.15)
                    rows.append({
                        "year": year, "round": rnd, "event_name": f"GP{rnd}",
                        "driver": drv, "stint": 1, "lap_number": lap,
                        "tyre_life": age, "compound": comp,
                        "lap_time_fuel_corr_s": pace, "position": rng.integers(1, 21),
                    })
    return pd.DataFrame(rows)


def test_windows_shapes_and_targets_are_leakage_safe():
    laps = _synthetic_clean_laps()
    w = build_sequence_windows(laps, window=5)
    # Shape: (n, window, n_features) and the documented feature count.
    assert w.X.ndim == 3 and w.X.shape[1] == 5 and w.X.shape[2] == len(FEATURE_NAMES)
    assert len(w.y_next) == len(w.y_curr) == w.X.shape[0]
    # A stint of n laps yields n-window samples; total matches.
    expected = sum(max(0, g.shape[0] - 5) for _, g in
                   laps.groupby(["year", "round", "driver", "stint"]))
    assert w.X.shape[0] == expected


def test_window_curr_and_next_align_to_consecutive_laps():
    # One clean stint; verify y_curr is lap t and y_next is lap t+1 (no peeking).
    n = 12
    laps = pd.DataFrame({
        "year": 2023, "round": 1, "event_name": "GP1", "driver": "VER", "stint": 1,
        "lap_number": np.arange(1, n + 1), "tyre_life": np.arange(1, n + 1.0),
        "compound": "MEDIUM", "position": 3,
        "lap_time_fuel_corr_s": 90.0 + 0.05 * np.arange(1, n + 1),
    })
    w = build_sequence_windows(laps, window=5)
    # First sample: window ends at lap 5 -> predicts lap 6.
    assert np.isclose(w.y_curr[0], 90.0 + 0.05 * 5)
    assert np.isclose(w.y_next[0], 90.0 + 0.05 * 6)
    # Last sample predicts the final lap n.
    assert np.isclose(w.y_next[-1], 90.0 + 0.05 * n)


def test_short_stints_are_skipped():
    laps = _synthetic_clean_laps()
    # window larger than every stint -> no samples buildable.
    with pytest.raises(RuntimeError):
        build_sequence_windows(laps, window=100)


def test_fit_and_predict_are_deterministic_and_finite():
    torch = pytest.importorskip("torch")  # noqa: F841
    laps = _synthetic_clean_laps()
    w = build_sequence_windows(laps, window=5)
    m1 = fit_sequence_model(w, epochs=3, hidden=8, seed=7)
    m2 = fit_sequence_model(w, epochs=3, hidden=8, seed=7)
    p1, p2 = m1.predict_next(w), m2.predict_next(w)
    assert p1.shape == w.y_next.shape
    assert np.all(np.isfinite(p1))
    assert np.allclose(p1, p2)  # same seed -> identical model


def test_evaluate_returns_comparable_maes():
    pytest.importorskip("torch")
    laps = _synthetic_clean_laps()
    res = evaluate_sequence_vs_baselines(
        laps, train_max_year=2024, test_year=2025, epochs=5, hidden=8, seed=0
    )
    for k in ("persistence_mae", "rolling_slope_mae", "lstm_mae"):
        assert res[k] > 0 and np.isfinite(res[k])
    assert res["n_test"] > 0 and res["test_year"] == 2025
    assert "lstm_vs_persistence_pct" in res


def test_live_window_builds_or_returns_none():
    n = 8
    stint = pd.DataFrame({
        "year": 2025, "round": 1, "event_name": "GP1", "driver": "VER", "stint": 1,
        "lap_number": np.arange(1, n + 1), "tyre_life": np.arange(1, n + 1.0),
        "compound": "SOFT", "position": 4,
        "lap_time_fuel_corr_s": 90.0 + 0.08 * np.arange(1, n + 1),
    })
    built = build_live_window(stint, window=5)
    assert built is not None
    X, last = built
    assert X.shape == (1, 5, len(FEATURE_NAMES))
    assert np.isclose(last, stint["lap_time_fuel_corr_s"].iloc[-1])
    # Too few laps for the window -> None (graceful "not enough green laps yet").
    assert build_live_window(stint.head(3), window=5) is None


def test_numpy_forecaster_matches_torch_and_forecasts(tmp_path):
    pytest.importorskip("torch")
    laps = _synthetic_clean_laps()
    train = laps[laps["year"] <= 2024]
    art = tmp_path / "lstm.npz"
    model = train_and_export(train, art, epochs=5, hidden=8, seed=0)

    fc = NumpyLapForecaster.load(art)
    assert fc.feature_names == FEATURE_NAMES
    # Numpy inference must reproduce the torch forward pass.
    test_w = build_sequence_windows(laps[laps["year"] == 2025], window=fc.window)
    assert np.allclose(fc.predict_delta(test_w.X), model.predict_delta(test_w.X), atol=1e-4)
    # End-to-end forecast on a stint.
    stint = laps[laps["year"] == 2025].groupby(
        ["year", "round", "driver", "stint"], observed=True).get_group(
        next(iter(laps[laps["year"] == 2025].groupby(
            ["year", "round", "driver", "stint"], observed=True).groups)))
    out = fc.forecast_next_lap(stint)
    assert out["ok"] and np.isfinite(out["predicted_s"])
