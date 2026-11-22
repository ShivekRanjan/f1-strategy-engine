"""The 2026 fallback for unraced tracks, and the stint-derived cliff option."""

import pandas as pd

from f1se.models.cliff import CliffPrior
from f1se.models.era import fit_era_shrunk_degradation


def _stint(year, rnd, track, driver, comp, n, slope, base=90.0):
    return pd.DataFrame({
        "year": year, "round": rnd, "event_name": track, "driver": driver, "stint": 1,
        "compound": comp, "tyre_life": range(1, n + 1), "lap_number": range(1, n + 1),
        "lap_time_fuel_corr_s": [base + slope * a for a in range(1, n + 1)],
    })


def _laps():
    old = [_stint(2025, r, "Old Track", d, "SOFT", 15, 0.20) for r in (1, 2) for d in "ABC"]
    new = [_stint(2026, r, "Raced 2026", d, "SOFT", 15, 0.05) for r in (1, 2) for d in "ABC"]
    return pd.concat(old + new, ignore_index=True)


def test_replace_ignores_the_tracks_old_car_history():
    m = fit_era_shrunk_degradation(_laps(), era_delta="replace")
    assert abs(m.slope("SOFT", "Old Track") - 0.05) < 1e-9


def test_stint_median_shifts_the_old_rate():
    m = fit_era_shrunk_degradation(_laps(), era_delta="stint_median")
    assert abs(m.slope("SOFT", "Old Track") - (0.20 + (0.05 - 0.20))) < 1e-9


def test_cliff_from_stints_uses_demonstrated_lengths():
    laps = pd.concat([_stint(2026, r, "T", d, "SOFT", n, 0.05)
                      for r in range(1, 6) for d, n in zip("ABCD", (10, 20, 30, 40))],
                     ignore_index=True)
    c = CliffPrior.from_stints(laps, quantile=0.5)
    assert c.cliff_age["SOFT"] == 25.0
    assert CliffPrior.from_stints(laps.head(5)).cliff_age == CliffPrior().cliff_age  # too few stints
