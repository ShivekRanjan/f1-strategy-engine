"""No-network tests for the engine orchestration and the thin API layer.

Uses a synthetic engine (no parquet/data dependency) so these stay fast and
deterministic; the real ``from_processed`` path is exercised by running the apps.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from f1se.engine import StrategyEngine
from f1se.models.degradation import DegradationModel

TRACK = "Test GP"


def _synthetic_laps() -> pd.DataFrame:
    """One driver, one 30-lap stint at Test GP 2024 — enough for the replay views."""
    n = 30
    lap = np.arange(1, n + 1)
    return pd.DataFrame({
        "year": 2024, "round": 1, "event_name": TRACK, "driver": "AAA", "stint": 1,
        "lap_number": lap, "tyre_life": lap.astype(float), "compound": "MEDIUM",
        "lap_time_s": 90.0 + 0.05 * lap, "lap_time_fuel_corr_s": 90.0 + 0.05 * lap,
        "position": 5,
    })


def _synthetic_engine() -> StrategyEngine:
    model = DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={},
        intercepts={(TRACK, "SOFT"): 89.5, (TRACK, "MEDIUM"): 90.0, (TRACK, "HARD"): 90.4},
        compound_slope={"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03},
        global_slope=0.05,
    )
    return StrategyEngine(
        deg_model=model,
        total_laps_by_track={TRACK: 40},
        stint_limits={"SOFT": 22, "MEDIUM": 30, "HARD": 38},
        laps=_synthetic_laps(),
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
    # Lap-by-lap gap paths for the crossover chart: aligned series that start
    # from (roughly) the entered 2.0s gap on the first lap after "now".
    t = out["trajectory"]
    assert len(t["laps"]) == len(t["undercut"]) == len(t["cover"]) > 0
    assert t["laps"][0] == 16 and t["your_pit_lap"] == 16 and t["rival_pit_lap"] == 25
    assert abs(t["cover"][0] - 2.0) < 1.0


def test_engine_unknown_track_raises():
    with pytest.raises(KeyError):
        _synthetic_engine().recommend("Nowhere GP")


def test_engine_replay_data_serving():
    eng = _synthetic_engine()
    assert eng.seasons(TRACK) == [2024]
    assert eng.all_seasons() == [2024]
    assert eng.circuits_for_season(2024) == [TRACK]
    assert eng.circuits_for_season(2023) == []          # season-first nav: empty year
    assert eng.replay_drivers(TRACK, 2024) == ["AAA"]
    hist = eng.lap_history(TRACK, 2024, "AAA")
    assert hist["lap_min"] == 1 and hist["lap_max"] == 30 and len(hist["laps"]) == 30
    assert hist["laps"][0]["compound"] == "MEDIUM"
    with pytest.raises(KeyError):
        eng.lap_history(TRACK, 2024, "ZZZ")


def test_engine_tracks_info_and_degradation_curves():
    eng = _synthetic_engine()
    assert eng.tracks_info() == [{"track": TRACK, "total_laps": 40, "well_sampled": False}]
    curves = eng.degradation_curves(TRACK)
    assert set(curves["compounds"]) == {"SOFT", "MEDIUM", "HARD"}
    soft = curves["compounds"]["SOFT"]
    assert soft["max_age"] == 22 and len(soft["ages"]) == 23
    assert soft["linear"][0] == 0.0 and soft["linear"][-1] > 0          # rises with age
    assert all(c >= lin for c, lin in zip(soft["cliff"], soft["linear"]))  # cliff adds loss
    assert soft["cliff"][-1] > soft["linear"][-1]                       # onset 18 < max 22
    with pytest.raises(KeyError):
        eng.degradation_curves("Nowhere")


def test_engine_live_replay_state_rec_and_nowcast():
    out = _synthetic_engine().live_replay(TRACK, 2024, "AAA", current_lap=15, n_runs=300)
    assert out["state"]["current_lap"] == 15 and out["state"]["laps_remaining"] == 25
    assert out["recommendation"] is not None and out["recommendation"]["laps_remaining"] == 25
    assert out["nowcast"] is None          # no forecaster on the synthetic engine


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

        r = client.post("/recommend", json={"track": "Test GP", "n_runs": 300, "top_k": 3,
                                            "track_temp": 20})
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

        uc = client.post("/undercut", json={
            "track": "Test GP", "current_lap": 15, "gap_s": 2.0,
            "your_compound": "MEDIUM", "your_age": 15, "your_new_compound": "SOFT",
            "rival_compound": "HARD", "rival_age": 20, "rival_new_compound": "HARD",
            "rival_pit_lap": 25, "n_runs": 300})
        assert uc.status_code == 200 and "verdict" in uc.json()

        assert client.get("/tracks_info").json()["tracks"][0]["total_laps"] == 40
        deg = client.get("/degradation/Test GP").json()
        assert set(deg["compounds"]) == {"SOFT", "MEDIUM", "HARD"}
        assert client.get("/degradation/Nowhere").status_code == 404

        assert client.get("/seasons/Test GP").json()["seasons"] == [2024]
        assert client.get("/seasons").json()["seasons"] == [2024]
        assert client.get("/circuits/2024").json()["circuits"] == ["Test GP"]
        assert client.get("/circuits/2023").json()["circuits"] == []
        assert client.get("/drivers/Test GP/2024").json()["drivers"] == ["AAA"]
        lh = client.get("/laps/Test GP/2024/AAA").json()
        assert lh["lap_max"] == 30 and len(lh["laps"]) == 30

        rep = client.post("/live", json={"track": "Test GP", "season": 2024,
                                         "driver": "AAA", "current_lap": 15, "n_runs": 300})
        assert rep.status_code == 200
        assert rep.json()["state"]["laps_remaining"] == 25 and rep.json()["nowcast"] is None
    finally:
        api.app.dependency_overrides.clear()
