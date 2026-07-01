"""No-network test for the outcome-predictor orchestration.

Uses the committed results dataset when present (CI has it); skips cleanly if a
checkout lacks it. Verifies the payload shape and JSON-safety, not exact numbers.
"""

from __future__ import annotations

import json

from f1se.standalone.outcome import compute_outcome, predict_upcoming


def test_compute_outcome_payload_shape():
    payload = compute_outcome(n_sims=300)
    if payload is None:
        import pytest

        pytest.skip("results.parquet not available in this checkout")

    assert {"test_year", "ongoing", "championship", "model_metrics", "rounds"} <= set(payload)
    assert payload["championship"], "championship list should be non-empty"
    leader = payload["championship"][0]
    assert 0.0 <= leader["win_prob"] <= 1.0 and isinstance(leader["driver"], str)
    assert 0.0 <= payload["model_metrics"]["auc"] <= 1.0
    if payload["rounds"]:
        pred = payload["rounds"][0]["predictions"][0]
        assert {"driver", "team", "grid", "podium_prob", "actual"} <= set(pred)
    json.dumps(payload)  # must be JSON-serialisable for the API


def test_predict_upcoming_next_round():
    payload = predict_upcoming()
    if payload is None:
        import pytest

        pytest.skip("results.parquet not available in this checkout")

    assert payload["grid_source"] == "form"
    assert payload["next_round"] >= 2 and payload["predictions"]
    top = payload["predictions"][0]
    assert {"driver", "team", "grid", "podium_prob"} <= set(top)
    assert 0.0 <= top["podium_prob"] <= 1.0
    # best-first ordering
    probs = [p["podium_prob"] for p in payload["predictions"]]
    assert probs == sorted(probs, reverse=True)
    # a custom grid override changes the outcome (put the last driver on pole)
    last = payload["predictions"][-1]["driver"]
    bumped = predict_upcoming(grid={last: 1})
    by_driver = {p["driver"]: p for p in bumped["predictions"]}
    assert by_driver[last]["grid"] == 1
    assert bumped["grid_source"] == "custom"
    json.dumps(payload)
