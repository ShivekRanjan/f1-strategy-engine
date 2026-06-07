"""Phase 2 (second half) — boosted degradation model, vs the linear baseline.

A gradient-boosted (XGBoost) degradation model that must *earn its place* by
beating the linear baseline (:mod:`f1se.models.degradation`) on the identical
leakage-safe split. It learns a flexible degradation shape g(age, compound,
track) instead of a single slope, so it can capture curvature and the late-stint
"cliff" that a straight line cannot.

Fairness, by construction:
  * Same target — within-stint pace deviation (the fixed-effects target the
    baseline uses), so neither model is rewarded for knowing base pace.
  * Same metric — :func:`f1se.models.degradation.shape_mae`, which demeans each
    model's prediction per stint (granting both the same free per-stint
    intercept) before scoring.
  * Same folds — :func:`f1se.validation.group_kfold_races` (no race leakage).

Held-out races have an unseen track category; XGBoost handles that natively
(default split direction), so like the baseline it then leans on age + compound.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import (
    AGE_COL,
    DEFAULT_GROUP_COLS,
    _stint_demeaned,
    fit_linear_baseline,
    linear_shape,
    naive_pace_loss_mae,
    shape_mae,
)
from f1se.validation import group_kfold_races

CATEGORICAL = ["compound", "event_name"]
FEATURES = [AGE_COL, "compound", "event_name"]

# Modest, regularised defaults — the dataset is small (thousands of laps) and we
# care about generalisation to unseen races, not squeezing in-sample fit.
DEFAULT_PARAMS: dict = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "tree_method": "hist",
    "enable_categorical": True,
    "random_state": 0,
}


def _prep_features(laps: pd.DataFrame) -> pd.DataFrame:
    X = laps[FEATURES].copy()
    for c in CATEGORICAL:
        # Preserve an existing category universe (set by harmonise_categories);
        # only derive from this frame if it isn't already categorical. This lets
        # a held-out track be a known-but-unobserved category at predict time.
        if not isinstance(X[c].dtype, pd.CategoricalDtype):
            X[c] = X[c].astype("category")
    return X


def harmonise_categories(laps: pd.DataFrame) -> pd.DataFrame:
    """Cast categorical features to a shared category universe over the full frame.

    Call this *before* splitting so train and test slices share categories and
    XGBoost never sees an unknown category at predict time.
    """
    out = laps.copy()
    for c in CATEGORICAL:
        out[c] = out[c].astype("category")
    return out


def fit_boosted(laps: pd.DataFrame, *, params: dict | None = None):
    """Fit XGBoost on the within-stint pace-deviation target.

    Returns the fitted ``XGBRegressor``. Trains on the demeaned target so it
    learns degradation *shape*, not absolute pace.
    """
    import xgboost as xgb

    p = {**DEFAULT_PARAMS, **(params or {})}
    X = _prep_features(laps)
    _, corr_dm = _stint_demeaned(laps)  # target = within-stint deviation
    model = xgb.XGBRegressor(**p)
    model.fit(X, corr_dm)
    return model


def predict_shape(model, laps: pd.DataFrame) -> np.ndarray:
    """Per-row degradation shape g(age, compound, track) from the boosted model."""
    return model.predict(_prep_features(laps))


def cross_val_compare(
    laps: pd.DataFrame,
    *,
    n_splits: int = 5,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_laps: int = 20,
    params: dict | None = None,
) -> dict[str, float]:
    """Leakage-safe head-to-head: linear vs boosted vs naive (pace-loss MAE).

    Trains both models on each GroupKFold-by-race training split and scores them
    on the held-out race with the shared :func:`shape_mae` metric.
    """
    laps = harmonise_categories(laps)  # shared category universe across folds
    lin, boost, naive = [], [], []
    for train_idx, test_idx in group_kfold_races(laps, n_splits=n_splits):
        train, test = laps.iloc[train_idx], laps.iloc[test_idx]

        lm = fit_linear_baseline(train, group_cols=group_cols, min_laps=min_laps)
        bm = fit_boosted(train, params=params)

        lin.append(shape_mae(linear_shape(lm, test), test))
        boost.append(shape_mae(predict_shape(bm, test), test))
        naive.append(naive_pace_loss_mae(test))

    lin_mae, boost_mae, naive_mae = map(lambda s: float(np.mean(s)), (lin, boost, naive))
    return {
        "linear_mae": lin_mae,
        "boosted_mae": boost_mae,
        "naive_mae": naive_mae,
        "boosted_vs_linear_pct": 100.0 * (lin_mae - boost_mae) / lin_mae if lin_mae else float("nan"),
        "boosted_vs_naive_pct": 100.0 * (naive_mae - boost_mae) / naive_mae if naive_mae else float("nan"),
        "n_folds": len(lin),
    }
