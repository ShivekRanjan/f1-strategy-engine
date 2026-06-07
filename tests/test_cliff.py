"""No-network tests for the tyre-cliff prior."""

from __future__ import annotations

import numpy as np

from f1se.models.cliff import CliffPrior
from f1se.models.degradation import DegradationModel
from f1se.sim.simulate import pace_fn_from_model


def test_cliff_zero_before_onset_and_convex_after():
    c = CliffPrior(cliff_age={"SOFT": 18.0}, rate=0.05, power=2.0)
    assert c.extra_loss("SOFT", 10) == 0.0          # within window
    assert c.extra_loss("SOFT", 18) == 0.0          # at onset
    # 3 laps past: 0.05 * 3^2 = 0.45; 6 past: 0.05 * 36 = 1.8 (accelerating).
    assert np.isclose(c.extra_loss("SOFT", 21), 0.45)
    assert np.isclose(c.extra_loss("SOFT", 24), 1.80)
    assert c.extra_loss("SOFT", 24) > 2 * c.extra_loss("SOFT", 21)  # convex


def test_unknown_compound_has_no_cliff():
    c = CliffPrior(cliff_age={"SOFT": 18.0})
    assert c.extra_loss("HARD", 50) == 0.0          # no onset defined -> inf


def test_disabled_prior_is_noop():
    c = CliffPrior.disabled()
    assert c.extra_loss("SOFT", 60) == 0.0


def _model():
    # Minimal degradation model: flat pace, so we isolate the cliff's effect.
    return DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={}, intercepts={}, compound_slope={}, global_slope=0.0,
    )


def test_pace_fn_adds_cliff_extra():
    model = _model()
    cliff = CliffPrior(cliff_age={"SOFT": 18.0}, rate=0.05, power=2.0)
    no_cliff = pace_fn_from_model(model, "X", 60, cliff=None)
    with_cliff = pace_fn_from_model(model, "X", 60, cliff=cliff)
    # Below onset: identical. Above onset: cliff adds the extra loss.
    assert with_cliff("SOFT", 10, 30) == no_cliff("SOFT", 10, 30)
    assert np.isclose(with_cliff("SOFT", 24, 30) - no_cliff("SOFT", 24, 30), 1.80)
