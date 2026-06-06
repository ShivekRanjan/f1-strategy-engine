"""No-network tests for the Phase 1 EDA analysis functions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.eda import compound_degradation_summary, fit_stint_slopes, fuel_sensitivity


def _stint(driver, stint, compound, slope, n=10, intercept=90.0):
    """One synthetic stint with a known fuel-corrected degradation slope."""
    tyre_life = np.arange(1, n + 1, dtype=float)
    return pd.DataFrame(
        {
            "year": 2023,
            "round": 1,
            "driver": driver,
            "stint": stint,
            "compound": compound,
            "race": "2023 Test",
            "tyre_life": tyre_life,
            "lap_number": tyre_life,
            "lap_time_s": 90.0,  # raw constant
            # fuel falls 1.5 kg/lap -> used by the sensitivity test
            "fuel_mass_kg": 50.0 - 1.5 * (tyre_life - 1),
            "lap_time_fuel_corr_s": intercept + slope * tyre_life,
        }
    )


def test_fit_stint_slopes_recovers_known_slope():
    clean = _stint("VER", 1, "SOFT", slope=0.05)
    slopes = fit_stint_slopes(clean)
    assert len(slopes) == 1
    assert np.isclose(slopes["slope_s_per_lap"].iloc[0], 0.05, atol=1e-9)
    assert slopes["r2"].iloc[0] > 0.999


def test_fit_stint_slopes_skips_short_stints():
    clean = _stint("VER", 1, "SOFT", slope=0.05, n=4)  # below min_laps=6
    assert fit_stint_slopes(clean).empty


def test_compound_summary_orders_by_fastest_degrading():
    clean = pd.concat(
        [
            _stint("VER", 1, "SOFT", slope=0.10),
            _stint("HAM", 1, "HARD", slope=0.02),
        ],
        ignore_index=True,
    )
    summary = compound_degradation_summary(fit_stint_slopes(clean))
    # Soft degrades faster -> appears first.
    assert summary.index[0] == "SOFT"
    assert summary.loc["SOFT", "median_slope"] > summary.loc["HARD", "median_slope"]


def test_fuel_sensitivity_is_linear_in_beta():
    # Raw pace constant; fuel falls 1.5 kg/lap. Corrected slope vs tyre age must
    # equal beta * 1.5 (the analytical Δslope = Δβ · fuel_per_lap relationship).
    clean = _stint("VER", 1, "MEDIUM", slope=0.0)
    sens = fuel_sensitivity(clean, betas=[0.02, 0.04])
    s02 = sens[sens["sec_per_kg"] == 0.02]["median_slope"].iloc[0]
    s04 = sens[sens["sec_per_kg"] == 0.04]["median_slope"].iloc[0]
    assert np.isclose(s02, 0.02 * 1.5, atol=1e-9)
    assert np.isclose(s04, 0.04 * 1.5, atol=1e-9)
    # Doubling beta doubles the slope shift.
    assert np.isclose(s04, 2 * s02, atol=1e-9)
