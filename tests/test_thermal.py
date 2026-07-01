"""No-network tests for the thermal (track-temperature) degradation prior."""

from __future__ import annotations

from f1se.models.degradation import DegradationModel
from f1se.models.thermal import ThermalPrior
from f1se.sim.simulate import pace_fn_from_model


def _model() -> DegradationModel:
    return DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={}, intercepts={("T", "MEDIUM"): 90.0},
        compound_slope={"MEDIUM": 0.06}, global_slope=0.05,
        track_base={"T": 90.0}, global_base=90.0,
    )


def test_slope_delta_direction():
    tp = ThermalPrior(sensitivity=0.01, ref_temp=36.0)
    assert tp.slope_delta(None) == 0.0
    assert tp.slope_delta(36.0) == 0.0
    assert tp.slope_delta(20.0) < 0.0          # cool -> less degradation
    assert tp.slope_delta(50.0) > 0.0          # hot -> more degradation
    assert ThermalPrior.disabled().slope_delta(50.0) == 0.0


def test_pace_fn_delta_shifts_degradation_and_is_backward_compatible():
    m = _model()
    hot = pace_fn_from_model(m, "T", 50, deg_slope_delta=+0.02)
    cool = pace_fn_from_model(m, "T", 50, deg_slope_delta=-0.02)
    # At the same tyre age a hotter track has degraded more (slower lap).
    assert hot("MEDIUM", 20, 10) > cool("MEDIUM", 20, 10)
    # delta = 0 reproduces the original pace exactly.
    base = pace_fn_from_model(m, "T", 50)
    zero = pace_fn_from_model(m, "T", 50, deg_slope_delta=0.0)
    assert abs(base("MEDIUM", 15, 5) - zero("MEDIUM", 15, 5)) < 1e-9


def test_delta_cannot_make_tyres_gain_pace():
    # A large negative delta clamps effective slope at 0 (no negative degradation).
    m = _model()
    cool = pace_fn_from_model(m, "T", 50, deg_slope_delta=-1.0)
    assert cool("MEDIUM", 30, 10) >= cool("MEDIUM", 1, 10) - 1e-9
