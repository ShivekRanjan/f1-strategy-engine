"""No-network tests for the tidy-lap flattening (loader)."""

from __future__ import annotations

import pandas as pd

from f1se.data.loader import LAP_SCHEMA, SessionRef, tidy_laps

REF = SessionRef(year=2023, round=1, event_name="Test GP", session="R")


def test_tidy_laps_conforms_to_schema(fake_session):
    df = tidy_laps(fake_session, REF)
    # Exactly the schema columns, in order.
    assert list(df.columns) == list(LAP_SCHEMA.keys())
    # Provenance stamped on every row.
    assert (df["year"] == 2023).all()
    assert (df["event_name"] == "Test GP").all()


def test_tidy_laps_converts_timedeltas_to_seconds(fake_session):
    df = tidy_laps(fake_session, REF)
    assert df["lap_time_s"].dtype == "float64"
    # Lap 3 was 90.0s in the fixture.
    assert df.loc[df["lap_number"] == 3, "lap_time_s"].iloc[0] == 90.0


def test_tidy_laps_derives_pit_flags(fake_session):
    df = tidy_laps(fake_session, REF)
    # Lap 1 is an out-lap, lap 6 an in-lap; nothing else.
    assert df.loc[df["lap_number"] == 1, "is_pit_out_lap"].iloc[0]
    assert df.loc[df["lap_number"] == 6, "is_pit_in_lap"].iloc[0]
    assert df["is_pit_out_lap"].sum() == 1
    assert df["is_pit_in_lap"].sum() == 1


def test_coerce_schema_rejects_missing_columns(fake_session):
    df = tidy_laps(fake_session, REF).drop(columns=["compound"])
    from f1se.data.loader import _coerce_schema

    try:
        _coerce_schema(df)
    except ValueError as e:
        assert "compound" in str(e)
    else:  # pragma: no cover
        raise AssertionError("expected ValueError for missing column")
