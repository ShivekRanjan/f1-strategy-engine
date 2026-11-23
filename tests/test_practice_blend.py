"""Practice long runs blended into an unraced track's wear (models/practice.py)."""

import pandas as pd

from f1se.models.degradation import DegradationModel
from f1se.models.practice import OFFSET, K, apply_for_unraced, with_practice


def _model(slope=0.05):
    return DegradationModel(group_cols=("event_name", "compound"),
                            slopes={("Lusail", "SOFT"): slope}, intercepts={},
                            compound_slope={"SOFT": slope}, global_slope=slope)


def _practice(n, slope, track="Lusail", year=2026):
    return pd.DataFrame({"year": [year], "round": [22], "event_name": [track],
                         "compound": ["SOFT"], "n_laps": [n], "slope_s_per_lap": [slope]})


def test_blend_weights_practice_by_its_laps():
    m = with_practice(_model(0.05), _practice(150, 0.15), "Lusail")
    assert abs(m.slope("SOFT", "Lusail") - (150 * (0.15 - OFFSET) + K * 0.05) / (150 + K)) < 1e-12


def test_no_practice_leaves_the_model_alone():
    m = _model(0.05)
    assert with_practice(m, _practice(150, 0.15, track="Elsewhere"), "Lusail") is m


def test_only_unraced_tracks_take_practice():
    raced = pd.DataFrame({"year": [2026], "event_name": ["Lusail"]})
    m = apply_for_unraced(_model(0.05), _practice(150, 0.15), raced, 2026)
    assert m.slope("SOFT", "Lusail") == 0.05
