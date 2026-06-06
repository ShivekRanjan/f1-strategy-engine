"""Phase 2 — tyre degradation model.

Models how much pace a tyre loses as it ages. The deliverable the simulator
calls is :func:`predict_pace_loss`: "given this compound at this age on this
track, how much slower than fresh am I?"

Why a within-stint (fixed-effects) fit
--------------------------------------
A naive first attempt — regress *absolute* fuel-corrected lap time on tyre age —
fails the leakage-safe evaluation badly, because the intercept (base pace) is
track/car/fuel specific. Hold out a whole race (its track unseen in training)
and the absolute prediction is off by seconds. Pooling across tracks also
conflates a track's pace *level* with tyre age, inflating slopes.

The degradation *rate* is what generalises. So we estimate the slope **within
each stint** (demean age and pace per stint, removing the per-stint level), then
pool those within-stint relationships per group. The intercept is treated as a
nuisance the simulator supplies separately. We therefore predict — and evaluate
on — *pace loss relative to the stint's own level*, not absolute lap time.

Build order (project rule): this linear, fixed-effects baseline is the number a
later hierarchical/boosted model (XGBoost) must beat on the GroupKFold-by-race
and forward-in-time splits (:mod:`f1se.validation`). Never a shuffled lap split.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1se.validation import group_kfold_races

DEFAULT_GROUP_COLS = ("event_name", "compound")
STINT_KEYS = ("year", "round", "driver", "stint")
AGE_COL = "tyre_life"
TARGET_COL = "lap_time_fuel_corr_s"


@dataclass(frozen=True)
class DegradationModel:
    """Per-group within-stint degradation slope, with fallbacks for unseen groups.

    ``slopes`` maps a group key (tuple of ``group_cols`` values) to the
    degradation rate in s/lap. ``intercepts`` stores a representative base pace
    per group purely for plotting/absolute prediction *on a seen track* — it is
    not used for pace-loss prediction or evaluation. Resolution falls back
    compound-only, then global, for groups never fit in training.
    """

    group_cols: tuple[str, ...]
    slopes: dict[tuple, float]
    intercepts: dict[tuple, float]
    compound_slope: dict[str, float]
    global_slope: float
    meta: dict = field(default_factory=dict)

    def _key(self, compound: str, track: str | None) -> tuple | None:
        if track is None:
            return None
        return tuple({"event_name": track, "compound": compound}[c] for c in self.group_cols)

    def slope(self, compound: str, track: str | None = None) -> float:
        """Resolve the degradation rate (s/lap), most-specific-first."""
        key = self._key(compound, track)
        if key is not None and key in self.slopes:
            return self.slopes[key]
        if self.group_cols == ("compound",) and (compound,) in self.slopes:
            return self.slopes[(compound,)]
        if compound in self.compound_slope:
            return self.compound_slope[compound]
        return self.global_slope


def _stint_demeaned(laps: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (age, pace) demeaned within each stint — the fixed-effects transform."""
    g = laps.groupby(list(STINT_KEYS), observed=True)
    age_dm = (laps[AGE_COL] - g[AGE_COL].transform("mean")).to_numpy(float)
    corr_dm = (laps[TARGET_COL] - g[TARGET_COL].transform("mean")).to_numpy(float)
    return age_dm, corr_dm


def _fe_slope(age_dm: np.ndarray, corr_dm: np.ndarray) -> float | None:
    """Within-group fixed-effects slope = Σ(x̃ỹ)/Σ(x̃²); None if no age spread."""
    denom = float(np.sum(age_dm**2))
    if denom == 0:
        return None
    return float(np.sum(age_dm * corr_dm) / denom)


def fit_linear_baseline(
    laps: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_laps: int = 20,
) -> DegradationModel:
    """Fit per-group within-stint degradation slopes (the baseline).

    Expects cleaned dry laps with ``lap_time_fuel_corr_s``, ``tyre_life`` and the
    stint keys (run ``clean_laps(..., dry_only=True)`` first). Groups thinner than
    ``min_laps`` or with no within-stint age spread are skipped and resolved by
    fallback.
    """
    for col in (*group_cols, *STINT_KEYS, AGE_COL, TARGET_COL):
        if col not in laps.columns:
            raise ValueError(f"degradation fit needs column '{col}'")

    laps = laps.copy()
    age_dm, corr_dm = _stint_demeaned(laps)
    laps["_age_dm"], laps["_corr_dm"] = age_dm, corr_dm

    slopes: dict[tuple, float] = {}
    intercepts: dict[tuple, float] = {}
    for key, grp in laps.groupby(list(group_cols), observed=True):
        if len(grp) < min_laps:
            continue
        s = _fe_slope(grp["_age_dm"].to_numpy(float), grp["_corr_dm"].to_numpy(float))
        if s is None:
            continue
        key = key if isinstance(key, tuple) else (key,)
        slopes[key] = s
        # Representative base pace (for plotting/absolute pred on a seen track).
        intercepts[key] = float(grp[TARGET_COL].mean() - s * grp[AGE_COL].mean())

    compound_slope: dict[str, float] = {}
    for comp, grp in laps.groupby("compound", observed=True):
        if len(grp) < min_laps:
            continue
        s = _fe_slope(grp["_age_dm"].to_numpy(float), grp["_corr_dm"].to_numpy(float))
        if s is not None:
            compound_slope[str(comp)] = s

    gslope = _fe_slope(age_dm, corr_dm)
    global_slope = gslope if gslope is not None else 0.0

    return DegradationModel(
        group_cols=tuple(group_cols),
        slopes=slopes,
        intercepts=intercepts,
        compound_slope=compound_slope,
        global_slope=global_slope,
        meta={"n_groups": len(slopes), "min_laps": min_laps, "n_train_laps": len(laps)},
    )


def predict_pace_loss(
    model: DegradationModel,
    compound: str,
    tyre_age: float,
    *,
    track: str | None = None,
) -> float:
    """Predicted fuel-corrected pace loss (s) versus a fresh (age-0) tyre."""
    return float(model.slope(compound, track) * tyre_age)


def predict_corrected_laptime(
    model: DegradationModel,
    compound: str,
    tyre_age: float,
    *,
    track: str | None = None,
) -> float:
    """Absolute fuel-corrected lap time (s) — only meaningful on a *seen* track.

    Uses the stored representative base pace; for an unseen track the base pace
    is unknown (that's the simulator's job), so prefer :func:`predict_pace_loss`.
    """
    key = model._key(compound, track)
    intercept = model.intercepts.get(key) if key is not None else None
    if intercept is None:  # no base pace for this track -> fall back to 0 origin
        intercept = 0.0
    return float(intercept + model.slope(compound, track) * tyre_age)


# --- evaluation: on within-stint pace loss, the track-level-free target --------

def pace_loss_mae(model: DegradationModel, laps: pd.DataFrame) -> float:
    """MAE of predicted vs actual within-stint pace deviation.

    Demeans each stint, so the metric measures how well the model captures the
    *degradation shape*, independent of the stint's (track/car/fuel) base level.
    """
    age_dm, corr_dm = _stint_demeaned(laps)
    slopes = np.array(
        [model.slope(c, t) for c, t in zip(laps["compound"], laps["event_name"])]
    )
    pred_dm = slopes * age_dm
    return float(np.mean(np.abs(pred_dm - corr_dm)))


def naive_pace_loss_mae(laps: pd.DataFrame) -> float:
    """The dumb comparator: predict *no* degradation (slope 0) -> mean |deviation|."""
    _, corr_dm = _stint_demeaned(laps)
    return float(np.mean(np.abs(corr_dm)))


def cross_val_mae(
    laps: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_laps: int = 20,
    n_splits: int = 5,
) -> dict[str, float]:
    """Leakage-safe CV (GroupKFold by race) of pace-loss MAE: linear vs naive.

    Because each track typically appears once, held-out races resolve via the
    compound-level slope — so this measures whether the degradation *rate* learnt
    on other races transfers to an unseen one.
    """
    lin_scores: list[float] = []
    naive_scores: list[float] = []
    for train_idx, test_idx in group_kfold_races(laps, n_splits=n_splits):
        train, test = laps.iloc[train_idx], laps.iloc[test_idx]
        model = fit_linear_baseline(train, group_cols=group_cols, min_laps=min_laps)
        lin_scores.append(pace_loss_mae(model, test))
        naive_scores.append(naive_pace_loss_mae(test))

    lin = float(np.mean(lin_scores))
    naive = float(np.mean(naive_scores))
    return {
        "linear_mae": lin,
        "naive_mae": naive,
        "improvement_pct": 100.0 * (naive - lin) / naive if naive else float("nan"),
        "n_folds": len(lin_scores),
    }
