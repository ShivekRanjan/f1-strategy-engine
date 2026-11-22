"""Ingest only asks for races that have already run."""

import pandas as pd

from f1se.data.ingest import past_rounds


def _sched():
    return pd.DataFrame({
        "RoundNumber": [0, 1, 2, 3, 4],
        "Session5DateUtc": pd.to_datetime([
            None, "2026-11-08 17:00", "2026-11-22 04:00", "2026-11-29 16:00", None,
        ]),
    })


def test_future_and_testing_rounds_are_dropped():
    assert past_rounds(_sched(), pd.Timestamp("2026-11-22 12:00")) == [1, 2, 4]


def test_race_not_yet_published_is_dropped():
    # an hour after the start the timing isn't out yet
    assert 2 not in past_rounds(_sched(), pd.Timestamp("2026-11-22 05:00"))
