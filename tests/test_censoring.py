"""No-network tests for the censoring guardrail (avoidance caps + adjustments)."""

from __future__ import annotations

import pandas as pd

from f1se.models.censoring import AvoidancePrior, apply_avoidance_adjustments
from f1se.models.degradation import DegradationModel


def _era_laps(track="T", comp="SOFT", n_stints=3, stint_len=5, slope=0.15, base=90.0):
    """Synthetic era laps: short stints of ``comp`` + long MEDIUM stints at ``track``."""
    rows = []
    for i in range(n_stints):
        for a in range(1, stint_len + 1):
            rows.append({"year": 2026, "round": i + 1, "driver": f"D{i}", "stint": 1,
                         "event_name": track, "compound": comp, "tyre_life": float(a),
                         "lap_time_fuel_corr_s": base + slope * a})
    for i in range(3):  # well-supported anchor compound with long stints
        for a in range(1, 25):
            rows.append({"year": 2026, "round": i + 1, "driver": f"D{i}", "stint": 2,
                         "event_name": track, "compound": "MEDIUM", "tyre_life": float(a),
                         "lap_time_fuel_corr_s": 91.0 + 0.05 * a})
    return pd.DataFrame(rows)


def test_caps_flag_only_avoided_compounds():
    laps = _era_laps(stint_len=5)
    caps = AvoidancePrior().track_caps(laps)
    assert caps == {("T", "SOFT"): 7}            # max age 5 + margin 2
    # MEDIUM ran to age 24 >= threshold -> uncapped; single-stint compounds ignored
    one = _era_laps(n_stints=1)
    assert ("T", "SOFT") not in AvoidancePrior().track_caps(one)
    assert AvoidancePrior.disabled().track_caps(laps) == {}


def _model(soft_slope=0.05, soft_base=89.0):
    return DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={("T", "SOFT"): soft_slope, ("T", "MEDIUM"): 0.05},
        intercepts={("T", "SOFT"): soft_base, ("T", "MEDIUM"): 91.0},
        compound_slope={"SOFT": 0.06, "MEDIUM": 0.05}, global_slope=0.05,
    )


def test_avoidance_unshrinks_slope_and_reanchors_base():
    laps = _era_laps(slope=0.15)                  # true (avoided) slope 0.15
    model = _model(soft_slope=0.05, soft_base=89.0)   # fiction: gentle + fast
    prior = DegradationModel(                     # old era measured soft 0.4s SLOWER
        group_cols=("event_name", "compound"), slopes={},
        intercepts={("T", "SOFT"): 90.75, ("T", "MEDIUM"): 90.35},
        compound_slope={}, global_slope=0.05,
    )
    out = apply_avoidance_adjustments(model, laps, prior_model=prior)
    assert out.slope("SOFT", "T") > 0.14          # raw era estimate restored
    # base re-anchored: 2026 MEDIUM level + old-era (soft - medium) gap
    assert abs(out.intercepts[("T", "SOFT")] - (91.0 + 0.40)) < 0.02
    assert "avoidance_adjustments" in out.meta
    # untouched: the well-supported anchor compound
    assert out.slope("MEDIUM", "T") == model.slope("MEDIUM", "T")
    assert out.intercepts[("T", "MEDIUM")] == model.intercepts[("T", "MEDIUM")]


def test_never_lowers_a_slope():
    laps = _era_laps(slope=0.02)                  # raw estimate GENTLER than fitted
    model = _model(soft_slope=0.10)
    out = apply_avoidance_adjustments(model, laps, prior_model=None)
    assert out.slope("SOFT", "T") == 0.10         # kept the worse (higher) fitted slope


def test_engine_caps_apply_to_current_era_only():
    from f1se.engine import StrategyEngine

    model = _model()
    eng = StrategyEngine(
        deg_model=model, deg_model_2026=model,
        total_laps_by_track={"T": 50},
        stint_limits={"SOFT": 22, "MEDIUM": 30, "HARD": 38},
        stint_caps={("T", "SOFT"): 7},
    )
    lim26 = eng._stint_limits_for("T", 2026)
    assert lim26["SOFT"] == 7 and lim26["MEDIUM"] == 30
    lim25 = eng._stint_limits_for("T", 2025)      # old-era sims: no 2026 caps
    assert lim25["SOFT"] == 22
    assert eng._stint_limits_for("U", 2026)["SOFT"] == 22   # other tracks untouched


def test_capped_compound_not_planned_beyond_cap():
    from f1se.engine import StrategyEngine

    model = DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={}, intercepts={("T", "SOFT"): 89.5, ("T", "MEDIUM"): 90.0,
                               ("T", "HARD"): 90.4},
        compound_slope={"SOFT": 0.09, "MEDIUM": 0.05, "HARD": 0.03},
        global_slope=0.05,
    )
    eng = StrategyEngine(
        deg_model=model, deg_model_2026=model,
        total_laps_by_track={"T": 40},
        stint_limits={"SOFT": 22, "MEDIUM": 30, "HARD": 38},
        stint_caps={("T", "SOFT"): 10},
    )
    rec = eng.recommend("T", season=2026, n_runs=300, top_k=10, seed=1)
    for row in rec["shortlist"]:
        bounds = [0, *row["pit_laps"], 40]
        for k, comp in enumerate(row["compounds"]):
            if comp == "SOFT":
                assert bounds[k + 1] - bounds[k] <= 10
