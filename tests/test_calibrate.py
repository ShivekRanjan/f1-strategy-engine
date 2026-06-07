"""No-network tests for effective fuel-coefficient calibration."""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.calibrate import calibrate_effective_fuel


def _raw_race(round_no, beta, deg_by_compound, *, start_fuel=110.0, n_cars=10, base=90.0):
    """Raw laps: lap_time_s = base_car + beta*fuel_mass(lap) + deg*tyre_age.

    fuel_mass = start*(N-lap)/N, where N (race distance) is the max lap any car
    reaches — self-consistent, as in a real race where the leader runs all N
    laps. Two stints/car with widely varied first-stint lengths give the
    cross-stint structure that identifies the lap-trend.
    """
    comps = list(deg_by_compound)
    # Race distance = the longest car's finishing lap (self-consistent fuel-N).
    N = (8 + 3 * (n_cars - 1)) + 20
    rows = []
    for d in range(n_cars):
        car_base = base + 0.4 * d
        L1 = 8 + 3 * d  # wide spread of first-stint lengths -> varied pit laps
        # Alternate compound order per car so compound is NOT confounded with
        # race phase (otherwise base-grip absorbs the fuel/lap trend).
        order = [comps[0], comps[1]] if d % 2 == 0 else [comps[1], comps[0]]
        # stint 1
        for age in range(1, L1 + 1):
            lap = age
            mass = start_fuel * (N - lap) / N
            c = order[0]
            rows.append((round_no, f"D{d}", 1, c, float(age), lap,
                         car_base + beta * mass + deg_by_compound[c] * age))
        # stint 2
        for age in range(1, 21):
            lap = L1 + age
            mass = start_fuel * (N - lap) / N
            c = order[1]
            rows.append((round_no, f"D{d}", 2, c, float(age), lap,
                         car_base + beta * mass + deg_by_compound[c] * age))
    df = pd.DataFrame(rows, columns=["round", "driver", "stint", "compound",
                                     "tyre_life", "lap_number", "lap_time_s"])
    df["year"] = 2023
    df["event_name"] = f"R{round_no}"
    return df


def test_recovers_true_fuel_coefficient_and_degradation():
    beta_true = 0.030
    deg = {"MEDIUM": 0.05, "HARD": 0.04}
    laps = _raw_race(1, beta_true, deg, start_fuel=110.0)
    cal = calibrate_effective_fuel(laps, start_fuel_kg=110.0)
    # Noiseless -> implied beta and degradation recovered essentially exactly.
    assert np.isclose(cal.median_beta, beta_true, atol=1e-3)
    assert np.isclose(cal.deg_slope["MEDIUM"], 0.05, atol=1e-3)
    assert np.isclose(cal.deg_slope["HARD"], 0.04, atol=1e-3)


def test_implied_beta_scales_with_assumed_start_fuel():
    # beta and start_fuel are confounded through their product in fuel mass:
    # assuming half the start fuel implies double the coefficient.
    deg = {"MEDIUM": 0.05, "HARD": 0.04}
    laps = _raw_race(1, 0.03, deg, start_fuel=110.0)
    full = calibrate_effective_fuel(laps, start_fuel_kg=110.0).median_beta
    half = calibrate_effective_fuel(laps, start_fuel_kg=55.0).median_beta
    assert np.isclose(half, 2 * full, rtol=1e-3)


def test_recovers_per_race_across_two_races():
    deg = {"MEDIUM": 0.05, "HARD": 0.04}
    r1 = _raw_race(1, 0.030, deg, n_cars=10)
    r2 = _raw_race(2, 0.035, deg, n_cars=12)
    laps = pd.concat([r1, r2], ignore_index=True)
    cal = calibrate_effective_fuel(laps, start_fuel_kg=110.0)
    assert np.isclose(cal.implied_beta_by_race["2023_01"], 0.030, atol=2e-3)
    assert np.isclose(cal.implied_beta_by_race["2023_02"], 0.035, atol=2e-3)
