"""Leakage-safe splitting — the project's headline methodological rule.

Laps within a race are highly correlated (same car, track, weather, fuel run).
A shuffled ``train_test_split`` over laps lets the model peek at the answer:
near-identical laps land in both train and test, so the score is inflated and
collapses on a genuinely unseen race. We never do that here.

Two complementary, defensible splits:

- :func:`group_kfold_races` — cross-validate with whole races as the group, so
  no race ever spans train and test. Measures "how well does this generalise to
  a race I didn't train on?".
- :func:`forward_year_holdout` — train on ``< test_year``, test on
  ``>= test_year`` (the project default: train ≤2023, test 2024). Measures the
  realistic "predict the future from the past" setting and guards against any
  subtle season-level leakage GroupKFold could miss.

Both key off :func:`race_id`, a stable per-race identifier.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def race_id(laps: pd.DataFrame) -> pd.Series:
    """Stable per-race id ("YYYY_RR") used as the grouping key."""
    return laps["year"].astype(int).astype(str) + "_" + laps["round"].astype(int).astype(str).str.zfill(2)


def group_kfold_races(
    laps: pd.DataFrame, n_splits: int = 5
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    """Yield ``(train_idx, test_idx)`` positional arrays, grouped by race.

    Thin wrapper over scikit-learn's ``GroupKFold`` with the race id as group,
    so no race appears in both train and test of any fold. ``n_splits`` is
    capped at the number of distinct races.
    """
    from sklearn.model_selection import GroupKFold

    groups = race_id(laps).to_numpy()
    n_races = len(np.unique(groups))
    n_splits = max(2, min(n_splits, n_races))

    gkf = GroupKFold(n_splits=n_splits)
    x = np.zeros(len(laps))  # GroupKFold ignores X values, only needs length.
    yield from gkf.split(x, groups=groups)


def forward_year_holdout(
    laps: pd.DataFrame, test_year: int = 2024
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split forward in time: train on ``year < test_year``, test on ``>=``.

    Returns ``(train, test)`` frames (copies). Raises if either side is empty —
    a silent empty test set is a classic way to fool yourself.
    """
    train = laps[laps["year"] < test_year].copy()
    test = laps[laps["year"] >= test_year].copy()
    if train.empty or test.empty:
        raise ValueError(
            f"forward holdout at {test_year} left an empty side "
            f"(train={len(train)}, test={len(test)}); check the year range loaded."
        )
    return train, test


def assert_no_race_leakage(train: pd.DataFrame, test: pd.DataFrame) -> None:
    """Raise if any race id appears in both splits. Cheap insurance in tests."""
    overlap = set(race_id(train)) & set(race_id(test))
    if overlap:
        raise AssertionError(f"race(s) leaked across split: {sorted(overlap)}")
