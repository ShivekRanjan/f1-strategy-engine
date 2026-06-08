"""No-network tests for era-aware shrinkage degradation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.era import fit_era_shrunk_degradation


def _laps(year, slope, n_rounds, *, n=30, base=90.0, compound="MEDIUM", track="A"):
    """One stint per round at `track`, fuel-corrected pace = base + slope*age."""
    rows = []
    for rnd in range(1, n_rounds + 1):
        for age in range(1, n + 1):
            rows.append({"year": year, "round": rnd, "driver": "D", "stint": 1,
                         "event_name": track, "compound": compound,
                         "tyre_life": float(age), "lap_time_fuel_corr_s": base + slope * age})
    return pd.DataFrame(rows)


def test_no_target_data_returns_prior():
    laps = _laps(2024, 0.05, n_rounds=8)            # only pre-2026
    m = fit_era_shrunk_degradation(laps, target_min_year=2026)
    assert np.isclose(m.slope("MEDIUM", "A"), 0.05, atol=1e-3)


def test_shrinks_between_prior_and_2026():
    # Regime shift: prior degrades 0.05/lap, 2026 degrades 0.15/lap.
    prior = _laps(2024, 0.05, n_rounds=10)          # plentiful old data
    target = _laps(2026, 0.15, n_rounds=2)          # thin 2026 data (60 laps)
    laps = pd.concat([prior, target], ignore_index=True)
    m = fit_era_shrunk_degradation(laps, target_min_year=2026, shrinkage_laps=60)
    s = m.slope("MEDIUM", "A")
    assert 0.05 < s < 0.15                          # strictly between
    assert np.isclose(s, 0.10, atol=1e-3)           # n=60, k=60 -> midpoint


def test_more_2026_data_moves_toward_2026():
    prior = _laps(2024, 0.05, n_rounds=10)
    thin = fit_era_shrunk_degradation(
        pd.concat([prior, _laps(2026, 0.15, 2)], ignore_index=True),
        target_min_year=2026, shrinkage_laps=60).slope("MEDIUM", "A")
    rich = fit_era_shrunk_degradation(
        pd.concat([prior, _laps(2026, 0.15, 8)], ignore_index=True),
        target_min_year=2026, shrinkage_laps=60).slope("MEDIUM", "A")
    assert rich > thin                               # more 2026 data -> closer to 0.15
    assert rich < 0.15


def test_shrinkage_knob_controls_trust():
    laps = pd.concat([_laps(2024, 0.05, 10), _laps(2026, 0.15, 3)], ignore_index=True)
    trusting = fit_era_shrunk_degradation(laps, shrinkage_laps=30).slope("MEDIUM", "A")
    cautious = fit_era_shrunk_degradation(laps, shrinkage_laps=600).slope("MEDIUM", "A")
    assert trusting > cautious                       # small k trusts 2026 sooner
    assert cautious < 0.10 < trusting


def test_meta_records_shrinkage():
    laps = pd.concat([_laps(2024, 0.05, 8), _laps(2026, 0.15, 2)], ignore_index=True)
    m = fit_era_shrunk_degradation(laps)
    assert m.meta["era"] == "shrunk" and m.meta["n_target_laps"] == 60
