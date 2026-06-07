"""No-network test for the per-track pit-loss estimator."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.eda import estimate_pit_loss


def _race_laps(event, pit_lap, in_extra, out_extra, *, rnd=1, n_laps=30, green=90.0, status_in="1"):
    """One driver's race: green laps at `green`s, a pit at `pit_lap` adding
    `in_extra` to the in-lap and `out_extra` to the out-lap."""
    rows = []
    for lap in range(1, n_laps + 1):
        is_in = lap == pit_lap
        is_out = lap == pit_lap + 1
        t = green + (in_extra if is_in else out_extra if is_out else 0.0)
        rows.append({
            "year": 2023, "round": rnd, "event_name": event, "driver": "VER",
            "lap_number": lap, "lap_time_s": t,
            "is_pit_in_lap": is_in, "is_pit_out_lap": is_out,
            "track_status": status_in if is_in else "1",
        })
    return pd.DataFrame(rows)


def test_estimates_pit_loss_from_neighbours():
    # in-lap +12s, out-lap +9s vs green neighbours -> pit loss ~21s.
    df = _race_laps("Test GP", pit_lap=10, in_extra=12.0, out_extra=9.0)
    est = estimate_pit_loss(df)
    assert np.isclose(est["Test GP"], 21.0, atol=1e-6)
    assert np.isclose(est["_global"], 21.0, atol=1e-6)


def test_safety_car_stops_are_excluded():
    # The only stop is under SC (in-lap status contains '4') -> no estimate.
    df = _race_laps("SC GP", pit_lap=10, in_extra=12.0, out_extra=9.0, status_in="4")
    est = estimate_pit_loss(df)
    assert "SC GP" not in est


def test_per_track_medians_differ():
    a = _race_laps("Short Pitlane", pit_lap=10, in_extra=8.0, out_extra=6.0, rnd=1)
    b = _race_laps("Long Pitlane", pit_lap=10, in_extra=15.0, out_extra=12.0, rnd=2)
    est = estimate_pit_loss(pd.concat([a, b], ignore_index=True))
    assert est["Short Pitlane"] < est["Long Pitlane"]
    assert np.isclose(est["Short Pitlane"], 14.0, atol=1e-6)
    assert np.isclose(est["Long Pitlane"], 27.0, atol=1e-6)
