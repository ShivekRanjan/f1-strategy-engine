"""Fuel-coefficient calibration — what the data can (and cannot) tell us.

The fuel correction in :mod:`f1se.data.clean` assumes ``sec_per_kg = 0.03`` (a
well-established engineering figure). Phase 2 showed the *degradation magnitude*
is dominated by this assumption, so it's worth asking what the data implies.

Method & its honest limit
-------------------------
We fit the joint model on **raw** lap time (per-car-race base + per-race linear
lap-trend ``γ`` + per-compound tyre-age slope). ``γ`` is identifiable from the
pace jumps at pit stops — tyre age resets there while fuel keeps falling — so it
is *the net race-lap pace trend*. Attributing all of it to fuel gives an implied
coefficient::

    β_implied = -γ · N_race / start_fuel_kg     (γ < 0 when pace improves)

BUT ``γ`` bundles fuel **with** track evolution and late-race management, which
are all smooth in race lap and not separable without external info. So
``β_implied`` is an *effective* coefficient, biased by whatever evo/management
remain. The robust, genuinely useful output is the **assumption-free degradation
slope** (``deg_slope`` from the raw fit), which removes the whole lap-trend
without committing to a fuel value. We report ``β_implied`` mainly to sanity-check
the physics 0.03 against the data, not to replace it blindly.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1se.data.clean import FuelModel
from f1se.models.degradation_evo import _race_key, fit_evolution_model


@dataclass(frozen=True)
class FuelCalibration:
    """Result of backing an effective fuel coefficient out of the lap-trend."""

    implied_beta_by_race: dict[str, float]
    mean_beta: float
    median_beta: float
    std_beta: float
    deg_slope: dict[str, float]   # assumption-free degradation (raw joint fit)
    physics_beta: float
    meta: dict = field(default_factory=dict)


def calibrate_effective_fuel(
    laps: pd.DataFrame,
    *,
    start_fuel_kg: float | None = None,
    raw_col: str = "lap_time_s",
) -> FuelCalibration:
    """Estimate the effective fuel coefficient + assumption-free degradation.

    Parameters
    ----------
    laps
        Cleaned dry laps carrying raw ``lap_time_s``, ``tyre_life``,
        ``lap_number``, ``driver``, ``compound``.
    start_fuel_kg
        Assumed start fuel mass (defaults to :class:`FuelModel`'s 110 kg).
    """
    fuel = FuelModel()
    start = float(start_fuel_kg if start_fuel_kg is not None else fuel.start_fuel_kg)

    model = fit_evolution_model(laps, target_col=raw_col)

    rk = _race_key(laps)
    n_by_race = laps.assign(_rk=rk).groupby("_rk")["lap_number"].max()

    implied: dict[str, float] = {}
    for rid, gamma in model.evo_slope.items():
        n = float(n_by_race.get(rid, np.nan))
        if not np.isfinite(n) or n <= 0:
            continue
        implied[rid] = -gamma * n / start

    betas = np.array(list(implied.values()), dtype=float)
    return FuelCalibration(
        implied_beta_by_race=implied,
        mean_beta=float(np.mean(betas)) if betas.size else float("nan"),
        median_beta=float(np.median(betas)) if betas.size else float("nan"),
        std_beta=float(np.std(betas)) if betas.size else float("nan"),
        deg_slope=model.deg_slope,
        physics_beta=fuel.sec_per_kg,
        meta={"n_races": len(implied), "start_fuel_kg": start, "r2": model.r2},
    )
