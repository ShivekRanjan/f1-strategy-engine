"""No-network test for the standings + title-projection orchestration.

Uses the committed results dataset when present; skips cleanly otherwise. Checks
payload shape, ordering, and JSON-safety rather than exact numbers.
"""

from __future__ import annotations

import json

import pytest

from f1se.standalone.standings import compute_standings


def _payload():
    p = compute_standings(n_sims=200)
    if p is None:
        pytest.skip("results.parquet not available in this checkout")
    return p


def test_standings_payload_shape_and_order():
    p = _payload()
    assert {"season", "seasons", "drivers", "constructors", "ongoing"} <= set(p)
    assert p["season"] in p["seasons"]

    drv = p["drivers"]
    assert drv, "driver standings should be non-empty"
    assert {"pos", "driver", "team", "points", "wins", "podiums", "races"} <= set(drv[0])
    # ranked by points, best-first, with 1-indexed positions
    assert [d["pos"] for d in drv] == list(range(1, len(drv) + 1))
    pts = [d["points"] for d in drv]
    assert pts == sorted(pts, reverse=True)

    con = p["constructors"]
    assert con and {"pos", "team", "points", "wins"} <= set(con[0])
    cpts = [c["points"] for c in con]
    assert cpts == sorted(cpts, reverse=True)

    json.dumps(p)  # must be JSON-serialisable for the API


def test_win_prob_present_only_when_ongoing():
    p = _payload()
    leader = p["drivers"][0]
    if p["ongoing"]:
        assert 0.0 <= leader["win_prob"] <= 1.0
        # projected title odds should sum to ~1 across the field
        probs = [d.get("win_prob") or 0.0 for d in p["drivers"]]
        assert 0.9 <= sum(probs) <= 1.01
    else:
        assert "win_prob" not in leader


def test_historical_season_has_no_projection():
    p = _payload()
    seasons = p["seasons"]
    if len(seasons) < 2:
        pytest.skip("need >1 season to check a completed one")
    past = compute_standings(season=seasons[0], n_sims=200)
    assert past["season"] == seasons[0]
    assert past["ongoing"] is False
    assert "win_prob" not in past["drivers"][0]
