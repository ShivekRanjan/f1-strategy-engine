"""No-network test for the outcome-predictor orchestration.

Uses the committed results dataset when present (CI has it); skips cleanly if a
checkout lacks it. Verifies the payload shape and JSON-safety, not exact numbers.
"""

from __future__ import annotations

import json

from f1se.standalone.outcome import compute_outcome


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
