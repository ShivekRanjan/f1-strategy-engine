"""Shared fixtures: a synthetic FastF1-shaped ``Laps`` frame, no network."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


class FakeSession:
    """Minimal stand-in for a FastF1 ``Session`` — just exposes ``.laps``."""

    def __init__(self, laps: pd.DataFrame):
        self.laps = laps


def _td(seconds):
    return pd.to_timedelta(seconds, unit="s")


@pytest.fixture
def raw_laps() -> pd.DataFrame:
    """A small FastF1-shaped lap frame covering the cases cleaning must handle.

    One driver, one stint of 6 laps, plus deliberately dirty rows:
      - lap 1 : out-lap (has PitOutTime)        -> dropped by pit filter
      - lap 2 : safety car (TrackStatus "4")    -> dropped by green filter
      - lap 3 : clean green lap
      - lap 4 : clean green lap
      - lap 5 : flagged inaccurate              -> dropped by accuracy filter
      - lap 6 : in-lap (has PitInTime)          -> dropped by pit filter
    """
    n = 6
    return pd.DataFrame(
        {
            "Driver": ["VER"] * n,
            "DriverNumber": ["1"] * n,
            "Team": ["Red Bull Racing"] * n,
            "LapNumber": list(range(1, n + 1)),
            "Stint": [1] * n,
            "LapTime": _td([95.0, 120.0, 90.0, 90.5, 91.0, 105.0]),
            "Position": [1] * n,
            "Sector1Time": _td([30.0] * n),
            "Sector2Time": _td([30.0] * n),
            "Sector3Time": _td([30.0] * n),
            "Compound": ["MEDIUM"] * n,
            "TyreLife": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
            "FreshTyre": [True] * n,
            "PitOutTime": [_td(10.0), pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT],
            "PitInTime": [pd.NaT, pd.NaT, pd.NaT, pd.NaT, pd.NaT, _td(20.0)],
            "TrackStatus": ["1", "4", "1", "1", "1", "1"],
            "IsAccurate": [True, True, True, True, False, True],
        }
    )


@pytest.fixture
def fake_session(raw_laps):
    return FakeSession(raw_laps)
