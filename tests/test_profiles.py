"""No-network tests for driver & constructor profiles.

Uses the committed results dataset when present; skips cleanly otherwise.
"""

from __future__ import annotations

import json

import pytest

from f1se.standalone.profiles import (
    constructor_profile,
    constructors_index,
    driver_profile,
    drivers_index,
)


def _drivers():
    idx = drivers_index()
    if idx is None:
        pytest.skip("results.parquet not available in this checkout")
    return idx


def test_drivers_index_ordered_recent_first():
    idx = _drivers()
    assert idx and {"driver", "team", "last_season", "seasons", "points", "wins"} <= set(idx[0])
    # most-recently-active drivers sort first
    last = [d["last_season"] for d in idx]
    assert last == sorted(last, reverse=True)
    json.dumps(idx)


def test_driver_profile_shape_and_totals():
    idx = _drivers()
    code = idx[0]["driver"]
    p = driver_profile(code)
    assert p["driver"] == code
    assert {"career", "by_season", "recent", "teammate_h2h"} <= set(p)
    car = p["career"]
    assert {"races", "wins", "podiums", "points", "avg_finish", "dnf"} <= set(car)
    # career totals equal the sum across seasons
    assert car["wins"] == sum(s["wins"] for s in p["by_season"])
    assert abs((car["points"] or 0) - sum(s["points"] or 0 for s in p["by_season"])) < 1e-6
    # recent is newest-first and at most 5
    assert len(p["recent"]) <= 5
    idxs = [r["season"] * 100 + r["round"] for r in p["recent"]]
    assert idxs == sorted(idxs, reverse=True)
    json.dumps(p)


def test_teammate_h2h_counts_are_bounded():
    idx = _drivers()
    # find a driver with at least one teammate H2H entry
    prof = next((driver_profile(d["driver"]) for d in idx
                 if driver_profile(d["driver"])["teammate_h2h"]), None)
    if prof is None:
        pytest.skip("no teammate H2H available")
    h = prof["teammate_h2h"][0]
    assert 0 <= h["quali_ahead"] <= h["quali_races"]
    assert 0 <= h["race_ahead"] <= h["race_races"]
    assert h["teammate"] != prof["driver"]


def test_constructor_profile_and_index():
    ci = constructors_index()
    assert ci and {"team", "last_season", "points", "wins"} <= set(ci[0])
    team = ci[0]["team"]
    p = constructor_profile(team)
    assert p["team"] == team
    assert {"career", "by_season", "drivers"} <= set(p)
    assert p["career"]["wins"] == sum(s["wins"] for s in p["by_season"])
    # drivers are ranked by points, best first
    pts = [d["points"] or 0 for d in p["drivers"]]
    assert pts == sorted(pts, reverse=True)
    json.dumps(p)


def test_unknown_driver_and_team_return_none():
    assert driver_profile("ZZZ") is None
    assert constructor_profile("Nonexistent Racing") is None
