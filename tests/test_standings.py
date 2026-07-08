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


def test_sprint_points_count_toward_standings(tmp_path):
    """Sprint points must be in the totals — a GP-only tally can even swap
    positions (it did, on the real 2026 table: P2/P3)."""
    import pandas as pd

    results = pd.DataFrame({
        "year": 2030, "round": [1, 1, 2, 2], "event_name": "X GP",
        "driver": ["AAA", "BBB", "AAA", "BBB"], "team": ["T1", "T2", "T1", "T2"],
        "grid": 1.0, "position": [1, 2, 1, 2], "points": [25.0, 18.0, 25.0, 18.0],
        "status": "Finished",
    })
    # BBB wins both sprints 8-7: GP-only would be 50-36; official is 64-52.
    sprints = pd.DataFrame({
        "year": 2030, "round": [1, 1, 2, 2], "event_name": "X GP",
        "driver": ["BBB", "AAA", "BBB", "AAA"], "team": ["T2", "T1", "T2", "T1"],
        "position": [1, 2, 1, 2], "points": [8.0, 7.0, 8.0, 7.0],
    })
    results.to_parquet(tmp_path / "results.parquet")
    sprints.to_parquet(tmp_path / "sprint_points.parquet")

    p = compute_standings(data_dir=tmp_path, n_sims=200)
    assert p["includes_sprints"] is True
    by_driver = {d["driver"]: d for d in p["drivers"]}
    assert by_driver["AAA"]["points"] == 64.0 and by_driver["BBB"]["points"] == 52.0
    assert by_driver["AAA"]["wins"] == 2 and by_driver["BBB"]["wins"] == 0  # GP wins only
    by_team = {c["team"]: c for c in p["constructors"]}
    assert by_team["T1"]["points"] == 64.0 and by_team["T2"]["points"] == 52.0

    # Without the sprint file, totals fall back to GP-only (and say so).
    (tmp_path / "sprint_points.parquet").unlink()
    p2 = compute_standings(data_dir=tmp_path, n_sims=200)
    assert p2["includes_sprints"] is False
    assert {d["driver"]: d["points"] for d in p2["drivers"]} == {"AAA": 50.0, "BBB": 36.0}


def test_refresh_merges_a_new_round(tmp_path, monkeypatch):
    """refresh_standings tops up the committed data with a live-pulled round."""
    import pandas as pd

    from f1se.standalone import standings as S

    base = pd.DataFrame({
        "year": 2031, "round": [1, 1], "event_name": "R1",
        "driver": ["AAA", "BBB"], "team": ["T1", "T2"], "grid": 1.0,
        "position": [1, 2], "points": [25.0, 18.0], "status": "Finished",
    })
    base.to_parquet(tmp_path / "results.parquet")
    monkeypatch.setattr(S, "_resolve_results", lambda _dd=None: tmp_path / "results.parquet")

    # Simulate FastF1 returning a freshly-run round 2 where BBB wins.
    def fake_fetch(season, have):
        race = pd.DataFrame({
            "year": 2031, "round": 2, "event_name": "R2",
            "driver": ["BBB", "AAA"], "team": ["T2", "T1"], "grid": [1.0, 2.0],
            "position": [1, 2], "points": [25.0, 18.0], "status": "Finished",
        })
        empty_s = pd.DataFrame(columns=["year", "round", "event_name", "driver", "team",
                                        "position", "points"])
        return race, empty_s, [2]

    monkeypatch.setattr(S, "_fetch_new_rounds", fake_fetch)

    p = S.refresh_standings(2031, n_sims=200)
    assert p["refreshed"] is True and p["added_rounds"] == [2]
    assert p["races_done"] == 2
    by = {d["driver"]: d["points"] for d in p["drivers"]}
    assert by["AAA"] == 43.0 and by["BBB"] == 43.0   # both raced R1+R2


def test_historical_season_has_no_projection():
    p = _payload()
    seasons = p["seasons"]
    if len(seasons) < 2:
        pytest.skip("need >1 season to check a completed one")
    past = compute_standings(season=seasons[0], n_sims=200)
    assert past["season"] == seasons[0]
    assert past["ongoing"] is False
    assert "win_prob" not in past["drivers"][0]
