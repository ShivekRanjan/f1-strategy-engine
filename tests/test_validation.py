"""No-network tests for the leakage-safe splitting."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1se.validation import (
    assert_no_race_leakage,
    forward_year_holdout,
    group_kfold_races,
    race_id,
)


def _laps(years_rounds, laps_each=10):
    """Build laps spanning given (year, round) pairs, laps_each rows each."""
    rows = []
    for yr, rnd in years_rounds:
        for i in range(laps_each):
            rows.append({"year": yr, "round": rnd, "lap_number": i + 1})
    return pd.DataFrame(rows)


def test_race_id_is_stable_and_zero_padded():
    df = _laps([(2023, 1), (2023, 12)], laps_each=1)
    assert race_id(df).tolist() == ["2023_01", "2023_12"]


def test_group_kfold_never_leaks_a_race():
    laps = _laps([(2023, r) for r in range(1, 7)])  # 6 races
    n = 0
    for train_idx, test_idx in group_kfold_races(laps, n_splits=3):
        train, test = laps.iloc[train_idx], laps.iloc[test_idx]
        assert_no_race_leakage(train, test)  # raises if a race spans both
        n += 1
    assert n == 3


def test_forward_year_holdout_splits_in_time():
    laps = _laps([(2022, 1), (2023, 1), (2024, 1)])
    train, test = forward_year_holdout(laps, test_year=2024)
    assert set(train["year"]) == {2022, 2023}
    assert set(test["year"]) == {2024}
    assert_no_race_leakage(train, test)


def test_forward_holdout_raises_on_empty_side():
    laps = _laps([(2022, 1), (2023, 1)])
    with pytest.raises(ValueError):
        forward_year_holdout(laps, test_year=2024)  # no 2024 data -> empty test


def test_assert_no_race_leakage_detects_overlap():
    laps = _laps([(2023, 1)])
    with pytest.raises(AssertionError):
        assert_no_race_leakage(laps, laps)  # same race on both sides
