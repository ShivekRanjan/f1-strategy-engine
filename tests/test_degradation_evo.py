"""No-network tests for the track-evolution-corrected degradation model.

These tests are the correctness backbone: on synthetic data with KNOWN
degradation and evolution slopes, the joint model must recover both, and the
within-stint baseline must show the predicted bias (it recovers deg + evo).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import fit_linear_baseline
from f1se.models.degradation_evo import fit_evolution_model


def _make_race(round_no, evo, deg_by_compound, *, n_cars=8, event="R1", base=90.0):
    """One race: each car runs 2 stints (different compounds, varied start laps).

    y = car_base + evo*race_lap + deg[compound]*tyre_age  (noiseless).
    Varying the first-stint length across cars gives the cross-stint variation
    that lets the joint model separate evolution from degradation.
    """
    comps = list(deg_by_compound)
    rows = []
    for d in range(n_cars):
        car_base = base + 0.5 * d           # distinct per-car pace
        L1 = 12 + d                          # first stint length varies per car
        # stint 1
        for age in range(1, L1 + 1):
            lap = age
            comp = comps[0]
            rows.append((round_no, f"D{d}", 1, event, comp, float(age), lap,
                         car_base + evo * lap + deg_by_compound[comp] * age))
        # stint 2 (different compound), starts at lap L1+1
        L2 = 20
        for age in range(1, L2 + 1):
            lap = L1 + age
            comp = comps[1]
            rows.append((round_no, f"D{d}", 2, event, comp, float(age), lap,
                         car_base + evo * lap + deg_by_compound[comp] * age))
    df = pd.DataFrame(
        rows,
        columns=["round", "driver", "stint", "event_name", "compound",
                 "tyre_life", "lap_number", "lap_time_fuel_corr_s"],
    )
    df["year"] = 2023
    return df


def test_recovers_known_degradation_and_evolution():
    deg = {"MEDIUM": 0.05, "HARD": 0.04}
    laps = _make_race(1, evo=-0.02, deg_by_compound=deg)
    m = fit_evolution_model(laps)
    assert np.isclose(m.deg_slope["MEDIUM"], 0.05, atol=1e-4)
    assert np.isclose(m.deg_slope["HARD"], 0.04, atol=1e-4)
    assert np.isclose(m.evo_slope["2023_01"], -0.02, atol=1e-4)
    assert m.r2 > 0.999


def test_within_stint_baseline_is_biased_by_evolution():
    # The naive within-stint slope should recover (deg + evo), i.e. be biased
    # LOW by |evo| when the track improves (evo < 0). The joint model corrects it.
    deg = {"MEDIUM": 0.06, "HARD": 0.05}
    evo = -0.02
    laps = _make_race(1, evo=evo, deg_by_compound=deg)

    naive = fit_linear_baseline(laps, group_cols=("event_name", "compound"), min_laps=5)
    corrected = fit_evolution_model(laps)

    for comp in deg:
        # baseline ~ true deg + evo (biased), corrected ~ true deg
        assert np.isclose(naive.compound_slope[comp], deg[comp] + evo, atol=1e-3)
        assert np.isclose(corrected.deg_slope[comp], deg[comp], atol=1e-3)
        # correction moves the estimate UP by |evo|
        assert corrected.deg_slope[comp] > naive.compound_slope[comp]


def test_per_race_evolution_recovered_across_two_races():
    deg = {"MEDIUM": 0.05, "HARD": 0.04}
    r1 = _make_race(1, evo=-0.02, deg_by_compound=deg, event="R1")
    r2 = _make_race(2, evo=-0.05, deg_by_compound=deg, event="R2")
    laps = pd.concat([r1, r2], ignore_index=True)
    m = fit_evolution_model(laps)
    assert np.isclose(m.evo_slope["2023_01"], -0.02, atol=1e-4)
    assert np.isclose(m.evo_slope["2023_02"], -0.05, atol=1e-4)
