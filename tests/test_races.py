"""No-network test for the Race Hub card (finishing order + pre-race prediction).

Uses the committed results dataset when present; skips cleanly otherwise.
"""

from __future__ import annotations

import json

import pytest

from f1se.standalone.races import _results_and_features, race_card


def _any_race() -> tuple[int, str]:
    rf = _results_and_features()
    if rf is None:
        pytest.skip("results.parquet not available in this checkout")
    results, _ = rf
    # A non-earliest season so a pre-race prediction (trained on prior years) exists.
    seasons = sorted(int(y) for y in results["year"].dropna().unique())
    season = seasons[-1]
    track = str(results[results["year"] == season]["event_name"].iloc[0])
    return season, track


def test_race_card_shape_and_result_order():
    season, track = _any_race()
    card = race_card(season, track)
    assert card is not None
    assert card["season"] == season and card["event_name"] == track
    assert isinstance(card["round"], int)

    res = card["result"]
    assert res, "finishing order should be non-empty"
    assert {"pos", "driver", "team", "grid", "points", "status", "gained"} <= set(res[0])
    # Finishers are ordered by position; DNFs (pos=None) sort to the end.
    finish = [row["pos"] for row in res if row["pos"] is not None]
    assert finish == sorted(finish)
    # actual_podium is the top-3 finishers, in order.
    assert card["actual_podium"] == [r["driver"] for r in res if r["pos"] and r["pos"] <= 3][:3]

    json.dumps(card)  # JSON-safe for the API


def test_race_card_prediction_is_forward_and_flags_actual():
    season, track = _any_race()
    card = race_card(season, track)
    pred = card["prediction"]
    if pred is None:
        pytest.skip("earliest season has no prior data to train a prediction")
    preds = pred["predictions"]
    assert preds and {"driver", "team", "grid", "podium_prob", "actual"} <= set(preds[0])
    # ranked best-first by podium probability
    probs = [p["podium_prob"] for p in preds]
    assert probs == sorted(probs, reverse=True)
    # hit_at_3 counts how many of the top-3 predicted actually podiumed (0..3)
    assert 0 <= pred["hit_at_3"] <= 3
    assert pred["hit_at_3"] == sum(1 for p in preds[:3] if p["actual"])
    assert 0.0 <= pred["auc"] <= 1.0


def test_unknown_race_returns_none():
    assert race_card(1998, "Nowhere Grand Prix") is None
