"""Phase 2 (correctness) — degradation with explicit track-evolution correction.

The within-stint baseline (:mod:`f1se.models.degradation`) isolates degradation
from per-stint base pace, but it does **not** separate degradation from *track
evolution* — the grip improvement as the circuit rubbers in over a race. Because
high tyre age correlates with later race laps (long stints finish late), the
baseline attributes some of that improving grip to the tyre and so
**under-estimates degradation** (and produces the unphysical late-stint dip).

This module fits, after fuel correction, the decomposition::

    fuel_corrected_time ≈ base(car, race) + evo·race_lap + deg·tyre_age + comp_offset

  * base(car, race) — one intercept per (race, driver): the car's race pace
    (absorbs track base pace, fuel-start, car/driver). NOT per stint.
  * evo·race_lap   — per-race linear track-evolution slope (s per race lap;
    expected negative = improving).
  * deg·tyre_age   — per-compound degradation slope (s per lap of tyre age) —
    the de-biased quantity we want.
  * comp_offset    — per-compound base-grip offset (a soft is faster when fresh).

Identifiability
---------------
Within one stint, ``race_lap`` and ``tyre_age`` move together, so a *per-stint*
intercept would confound evo and deg (only their sum is identified — exactly what
the baseline measures). Using a per-**car-race** intercept instead, drivers who
run two or more stints starting at different race laps provide the cross-stint
variation that separates the two linear slopes. The fit is ordinary least
squares; rank is handled by the least-norm ``lstsq`` solution.

Prediction note: ``evo`` is a per-race nuisance estimated to *de-bias* deg; it is
not used to predict an unseen race. The transferable output for the simulator is
the per-compound degradation slope.

IMPORTANT CAVEAT — what the linear ``evo`` term really is
---------------------------------------------------------
Fuel mass is itself linear in race lap (``mass ∝ N − lap``), so the fuel
correction already baked a linear-in-lap slope into ``fuel_corrected_time``. A
*second* linear-in-lap term (this ``evo``) is therefore **collinear with the fuel
correction within a race** and is NOT separately identifiable from it. So
``evo_slope`` does not isolate track evolution — it captures the *net* residual
linear-in-lap trend, conflating (a) real track evolution, (b) any miscalibration
of the 0.03 s/kg fuel coefficient, and (c) late-race pace/thermal management.
Empirically it comes out mostly *positive* (pace fading over the race), i.e. NOT
dominated by classic rubber-in. To genuinely separate track evolution from fuel
you need a *nonlinear* (concave) evolution form, whose curvature is distinct from
fuel's linear shape. Treat this model as a diagnostic of the net lap-trend, not
as a validated track-evolution estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1se.models.degradation import AGE_COL, TARGET_COL

LAP_COL = "lap_number"


@dataclass(frozen=True)
class EvoDegradationModel:
    """Fitted degradation (per compound) and track-evolution (per race) slopes."""

    deg_slope: dict[str, float]      # compound -> s/lap of tyre age (de-biased)
    evo_slope: dict[str, float]      # race id  -> s/lap of race progress
    compound_offset: dict[str, float]
    r2: float
    meta: dict

    def degradation(self, compound: str, tyre_age: float) -> float:
        """De-biased predicted pace loss (s) vs a fresh tyre."""
        return float(self.deg_slope.get(compound, np.mean(list(self.deg_slope.values()))) * tyre_age)


def _race_key(laps: pd.DataFrame) -> pd.Series:
    return laps["year"].astype(int).astype(str) + "_" + laps["round"].astype(int).astype(str).str.zfill(2)


def fit_evolution_model(
    laps: pd.DataFrame, *, target_col: str = TARGET_COL
) -> EvoDegradationModel:
    """Jointly fit per-compound degradation and per-race lap-trend (OLS).

    Expects cleaned dry laps with ``tyre_life``, ``lap_number``, ``driver`` and
    ``compound`` plus ``target_col``.

    ``target_col`` selects what the lap-trend (``evo``) absorbs:
      * ``lap_time_fuel_corr_s`` (default) — fuel already removed, so ``evo`` is
        the *residual* trend (track evo + fuel miscalibration + management).
      * ``lap_time_s`` (raw) — ``evo`` is the *total* race-lap trend including
        fuel; used by :mod:`f1se.models.calibrate` to back out the effective
        fuel coefficient and an assumption-free degradation estimate.
    """
    for col in (target_col, AGE_COL, LAP_COL, "driver", "compound"):
        if col not in laps.columns:
            raise ValueError(f"evolution fit needs column '{col}'")

    df = laps.copy()
    race = _race_key(df)
    car = race + "|" + df["driver"].astype(str)  # per car-race intercept

    # Design blocks:
    car_d = pd.get_dummies(car, prefix="base")                       # base pace per car-race
    evo = pd.get_dummies(race, prefix="evo").mul(df[LAP_COL].to_numpy(), axis=0)   # per-race evo slope
    deg = pd.get_dummies(df["compound"], prefix="deg").mul(df[AGE_COL].to_numpy(), axis=0)  # per-compound deg
    comp = pd.get_dummies(df["compound"], prefix="comp", drop_first=True)          # compound base offset

    X = pd.concat([car_d, evo, deg, comp], axis=1).astype(float)
    y = df[target_col].to_numpy(float)

    # Column scaling before lstsq: the lap/age columns span ~1-70 while the
    # dummies are 0/1, a dynamic range that makes the solve ill-conditioned on
    # the (small-singular-value) lap-trend direction. Scale to unit norm, solve,
    # then unscale — recovers the slopes precisely without changing the model.
    Xv = X.to_numpy()
    norms = np.linalg.norm(Xv, axis=0)
    norms[norms == 0] = 1.0
    beta_scaled, *_ = np.linalg.lstsq(Xv / norms, y, rcond=None)
    beta = beta_scaled / norms
    coef = dict(zip(X.columns, beta))

    deg_slope = {c[len("deg_"):]: coef[c] for c in X.columns if c.startswith("deg_")}
    evo_slope = {c[len("evo_"):]: coef[c] for c in X.columns if c.startswith("evo_")}
    compound_offset = {c[len("comp_"):]: coef[c] for c in X.columns if c.startswith("comp_")}

    resid = y - X.to_numpy() @ beta
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum(resid**2)) / ss_tot if ss_tot > 0 else float("nan")

    return EvoDegradationModel(
        deg_slope=deg_slope,
        evo_slope=evo_slope,
        compound_offset=compound_offset,
        r2=r2,
        meta={"n_laps": len(df), "n_cars": car_d.shape[1], "n_races": evo.shape[1]},
    )
