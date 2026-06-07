"""No-network tests for polynomial (cliff) degradation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import shape_mae
from f1se.models.degradation_poly import fit_poly, poly_shape


def _laps(coef_by_compound, *, degree, n_races=6, n=30):
    """Synthetic dry laps: fuel-corrected pace = sum_k coef[k]*age**(k+1)."""
    rows = []
    for rnd in range(1, n_races + 1):
        for comp, coef in coef_by_compound.items():
            for age in range(1, n + 1):
                deg = sum(coef[k] * age ** (k + 1) for k in range(len(coef)))
                rows.append({
                    "year": 2023, "round": rnd, "driver": f"D{comp}", "stint": 1,
                    "event_name": f"R{rnd}", "compound": comp,
                    "tyre_life": float(age), "lap_time_fuel_corr_s": 90.0 + deg,
                })
    return pd.DataFrame(rows)


def test_quadratic_recovers_curvature():
    # MEDIUM: 0.02*age + 0.001*age^2 (accelerating). HARD: linear.
    laps = _laps({"MEDIUM": [0.02, 0.001], "HARD": [0.03, 0.0]}, degree=2)
    m = fit_poly(laps, degree=2, min_laps=10)
    assert np.allclose(m.compound_coeffs["MEDIUM"], [0.02, 0.001], atol=1e-4)
    assert abs(m.compound_coeffs["HARD"][1]) < 1e-4  # no curvature for hard


def test_pace_loss_uses_polynomial():
    laps = _laps({"MEDIUM": [0.02, 0.001]}, degree=2)
    m = fit_poly(laps, degree=2, min_laps=10)
    # age 20: 0.02*20 + 0.001*400 = 0.4 + 0.4 = 0.8
    assert np.isclose(m.pace_loss("MEDIUM", 20, track="R1"), 0.8, atol=1e-3)


def test_quadratic_beats_linear_on_curved_data():
    laps = _laps({"MEDIUM": [0.02, 0.0015], "HARD": [0.03, 0.001]}, degree=2)
    lin = fit_poly(laps, degree=1, min_laps=10)
    quad = fit_poly(laps, degree=2, min_laps=10)
    assert shape_mae(poly_shape(quad, laps), laps) < shape_mae(poly_shape(lin, laps), laps)


def test_degree1_poly_matches_linear_fe_slope():
    from f1se.models.degradation import fit_linear_baseline
    laps = _laps({"MEDIUM": [0.05, 0.0], "HARD": [0.03, 0.0]}, degree=2)
    poly1 = fit_poly(laps, degree=1, min_laps=10)
    lin = fit_linear_baseline(laps, min_laps=10)
    assert np.isclose(poly1.compound_coeffs["MEDIUM"][0], lin.compound_slope["MEDIUM"], atol=1e-6)
