"""No-network tests for the cleaning + fuel-correction logic."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.data.clean import (
    FuelModel,
    add_fuel_correction,
    clean_laps,
    filter_racing_laps,
)
from f1se.data.loader import SessionRef, tidy_laps

REF = SessionRef(year=2023, round=1, event_name="Test GP", session="R")


def test_filter_drops_pit_sc_and_inaccurate_laps(fake_session):
    df = tidy_laps(fake_session, REF)
    clean = filter_racing_laps(df)
    # Only laps 3 and 4 survive (out/in/SC/inaccurate all removed).
    assert sorted(clean["lap_number"].tolist()) == [3, 4]


def test_filter_keeps_only_green_status(fake_session):
    df = tidy_laps(fake_session, REF)
    clean = filter_racing_laps(df)
    assert (clean["track_status"] == "1").all()


def test_fuel_correction_subtracts_penalty_and_isolates_tyre_trend():
    # Build a flat-pace stint: identical raw lap times, only fuel changing.
    # After correction, early (heavy) laps should be FASTER than late (light)
    # ones flips: corrected time should be monotonic, exposing the fuel trend
    # we removed. Here raw is constant, so corrected must RISE with lap number
    # (heavy laps were helped less once we strip the larger fuel credit).
    total = 50
    laps = pd.DataFrame(
        {
            "year": 2023,
            "round": 1,
            "lap_number": [10, 20, 30, 40],
            "lap_time_s": [90.0, 90.0, 90.0, 90.0],
        }
    )
    fuel = FuelModel(sec_per_kg=0.03, start_fuel_kg=110.0)
    out = add_fuel_correction(laps, fuel, total_laps=total)

    # Heavier (earlier) laps carry a bigger fuel credit subtracted, so corrected
    # times increase with lap number when raw pace is constant.
    corr = out["lap_time_fuel_corr_s"].tolist()
    assert corr == sorted(corr)
    # Spot-check the arithmetic on lap 10: fuel = 110*(50-10)/50 = 88 kg.
    expected = 90.0 - 0.03 * 88.0
    assert np.isclose(out.loc[out["lap_number"] == 10, "lap_time_fuel_corr_s"].iloc[0], expected)


def test_fuel_correction_infers_total_laps_per_race():
    laps = pd.DataFrame(
        {
            "year": [2023, 2023, 2024, 2024],
            "round": [1, 1, 1, 1],
            "lap_number": [1, 2, 1, 2],
            "lap_time_s": [90.0, 90.0, 90.0, 90.0],
        }
    )
    # Two races of different lengths -> max lap_number inferred per group.
    out = add_fuel_correction(laps)
    assert "fuel_mass_kg" in out.columns
    assert (out["fuel_mass_kg"] >= 0).all()


def test_clean_pipeline_adds_corrected_column(fake_session):
    df = tidy_laps(fake_session, REF)
    out = clean_laps(df, drop_outliers=False, total_laps=50)
    assert "lap_time_fuel_corr_s" in out.columns
    assert len(out) == 2  # same survivors as the filter test
