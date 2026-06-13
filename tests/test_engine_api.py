"""No-network tests for the engine orchestration and the thin API layer.

Uses a synthetic engine (no parquet/data dependency) so these stay fast and
deterministic; the real ``from_processed`` path is exercised by running the apps.
"""

from __future__ import annotations

import pytest

from f1se.engine import StrategyEngine
from f1se.models.degradation import DegradationModel


def _synthetic_engine() -> StrategyEngine:
    track = "Test GP"
    model = DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={},
        intercepts={(track, "SOFT"): 89.5, (track, "MEDIUM"): 90.0, (track, "HARD"): 90.4},
        compound_slope={"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03},
        global_slope=0.05,
    )
    return StrategyEngine(
        deg_model=model,
        total_laps_by_track={track: 40},
        stint_limits={"SOFT": 22, "MEDIUM": 30, "HARD": 38},
    )


def test_engine_tracks_and_race_info():
    eng = _synthetic_engine()
    assert eng.tracks() == ["Test GP"]
    info = eng.race_info("Test GP")
    assert info["total_laps"] == 40
    assert "sc_prob_per_lap" in info and "pit_loss_s" in info


def test_engine_recommend_returns_ranked_shortlist():
    eng = _synthetic_engine()
    rec = eng.recommend("Test GP", n_runs=300, top_k=4, seed=1)
    assert rec["n_evaluated"] > 0
    assert len(rec["shortlist"]) == 4
    assert len(rec["best"]["compounds"]) >= 2          # >=1 stop, >=2 compounds
    scores = [row["mean_s"] for row in rec["shortlist"]]
    assert rec["best"]["mean_s"] <= max(scores)         # best is among the best


def test_engine_simulate_returns_summary_and_histogram():
    eng = _synthetic_engine()
    sim = eng.simulate("Test GP", ("MEDIUM", "HARD"), (20,), n_runs=300, hist_bins=30)
    assert sim["mean_s"] > 0 and sim["p90_s"] >= sim["p10_s"]
    assert len(sim["hist_counts"]) == 30
    assert len(sim["hist_edges"]) == 31


def test_engine_recommend_live_from_current_state():
    eng = _synthetic_engine()
    out = eng.recommend_live("Test GP", current_lap=15, current_compound="SOFT",
                             tyre_age=15, compounds_used=("MEDIUM", "SOFT"), n_runs=300)
    assert out["laps_remaining"] == 40 - 15
    assert out["n_evaluated"] > 0
    assert out["shortlist"][0]["rank"] == 1
    assert "best_plan" in out


def test_engine_reliable_tracks_filter_and_flag():
    model = _synthetic_engine().deg_model
    eng = StrategyEngine(deg_model=model, total_laps_by_track={"A": 50, "B": 50},
                         well_sampled_tracks={"A"})
    assert eng.tracks() == ["A", "B"]
    assert eng.tracks(reliable_only=True) == ["A"]          # B filtered out
    assert eng.is_well_sampled("A") and not eng.is_well_sampled("B")
    # Empty well-sampled set is a no-op guard (returns all tracks).
    assert _synthetic_engine().tracks(reliable_only=True) == ["Test GP"]


def test_engine_undercut_returns_both_options():
    eng = _synthetic_engine()
    out = eng.undercut("Test GP", current_lap=15, gap_s=2.0,
                       your_compound="MEDIUM", your_age=15, your_new_compound="SOFT",
                       rival_compound="HARD", rival_age=20, rival_new_compound="HARD",
                       rival_pit_lap=25, n_runs=300)
    assert "undercut" in out and "cover" in out
    assert "verdict" in out and isinstance(out["undercut_works"], bool)
    for opt in ("undercut", "cover"):
        assert 0.0 <= out[opt]["p_ahead"] <= 1.0


def test_engine_unknown_track_raises():
    with pytest.raises(KeyError):
        _synthetic_engine().recommend("Nowhere GP")


def test_api_endpoints_with_synthetic_engine():
    from fastapi.testclient import TestClient

    from f1se import api

    api.app.dependency_overrides[api.get_engine] = _synthetic_engine
    client = TestClient(api.app)
    try:
        assert client.get("/health").json() == {"status": "ok"}
        assert client.get("/tracks").json() == {"tracks": ["Test GP"]}
        assert client.get("/race/Test GP").json()["total_laps"] == 40
        assert client.get("/race/Nowhere").status_code == 404

        r = client.post("/recommend", json={"track": "Test GP", "n_runs": 300, "top_k": 3})
        assert r.status_code == 200
        body = r.json()
        assert len(body["shortlist"]) == 3 and len(body["best"]["compounds"]) >= 2

        s = client.post("/simulate", json={"track": "Test GP", "compounds": ["MEDIUM", "HARD"],
                                           "pit_laps": [20], "n_runs": 300})
        assert s.status_code == 200 and s.json()["mean_s"] > 0

        live = client.post("/recommend_live", json={
            "track": "Test GP", "current_lap": 15, "current_compound": "SOFT",
            "tyre_age": 15, "compounds_used": ["MEDIUM", "SOFT"], "n_runs": 300})
        assert live.status_code == 200 and live.json()["laps_remaining"] == 25
    finally:
        api.app.dependency_overrides.clear()
