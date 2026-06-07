"""Phase A — podium predictor (will a driver finish top-3?).

A results-only classifier with the same discipline as the main project: features
use only information available *before* the race (grid + rolling prior form, no
leakage), and we validate forward in time (train on earlier seasons, test on the
latest) against a strong dumb baseline — "the top 3 on the grid finish on the
podium". The natural metric is per-race precision@3: of our top-3 predicted, how
many actually podiumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

FEATURE_COLS = ["grid", "driver_form_pos", "driver_form_pts", "driver_podium_rate", "team_form_pts"]


@dataclass(frozen=True)
class PodiumModel:
    clf: object
    feature_cols: tuple[str, ...]
    metrics: dict


def _finished(status: pd.Series) -> pd.Series:
    s = status.astype("string").fillna("")
    return s.str.contains("Finished", na=False) | s.str.contains(r"\+\d+ Lap", regex=True, na=False)


def build_features(results: pd.DataFrame, *, form_window: int = 5) -> pd.DataFrame:
    """Add the leakage-safe features and the podium target.

    Rolling form is computed per driver/team over *previous* races only
    (``shift(1)`` before the rolling window), so no future information leaks.
    """
    df = results.copy()
    df["podium"] = (df["position"] <= 3).fillna(False).astype(int)
    df["race_idx"] = df["year"].astype(int) * 100 + df["round"].astype(int)

    df = df.sort_values(["driver", "race_idx"])
    g = df.groupby("driver", observed=True)
    df["driver_form_pos"] = g["position"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    df["driver_form_pts"] = g["points"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())
    df["driver_podium_rate"] = g["podium"].transform(
        lambda s: s.shift(1).rolling(form_window, min_periods=1).mean())

    df = df.sort_values(["team", "race_idx"])
    df["team_form_pts"] = df.groupby("team", observed=True)["points"].transform(
        lambda s: s.shift(1).rolling(form_window * 2, min_periods=1).mean())

    # Sensible fills for a driver/team's first appearances (no prior history).
    df["driver_form_pos"] = df["driver_form_pos"].fillna(15.0)   # assume midfield/back
    df["driver_form_pts"] = df["driver_form_pts"].fillna(0.0)
    df["driver_podium_rate"] = df["driver_podium_rate"].fillna(0.0)
    df["team_form_pts"] = df["team_form_pts"].fillna(0.0)
    df["grid"] = df["grid"].replace(0, 20).fillna(20)            # 0 = pit-lane start
    return df.sort_values(["race_idx", "position"]).reset_index(drop=True)


def _precision_at_3(df: pd.DataFrame, score_col: str, *, ascending: bool) -> float:
    """Mean over races of (# of our top-3-by-score that actually podiumed) / 3."""
    hits = []
    for _, race in df.groupby("race_idx"):
        top3 = race.sort_values(score_col, ascending=ascending).head(3)
        hits.append(top3["podium"].sum() / 3.0)
    return float(np.mean(hits)) if hits else float("nan")


def train_podium_model(features: pd.DataFrame, *, test_year: int) -> PodiumModel:
    """Fit a gradient-boosted podium classifier; validate on ``test_year``."""
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import roc_auc_score

    train = features[features["year"] < test_year]
    test = features[features["year"] == test_year].copy()
    if train.empty or test.empty:
        raise ValueError(f"forward split at {test_year} left an empty side")

    clf = GradientBoostingClassifier(n_estimators=200, max_depth=3,
                                     learning_rate=0.05, random_state=0)
    clf.fit(train[FEATURE_COLS], train["podium"])
    test["score"] = clf.predict_proba(test[FEATURE_COLS])[:, 1]

    metrics = {
        "test_year": test_year,
        "n_train": len(train),
        "n_test": len(test),
        "auc": float(roc_auc_score(test["podium"], test["score"])),
        "model_precision_at_3": _precision_at_3(test, "score", ascending=False),
        "grid_baseline_precision_at_3": _precision_at_3(test, "grid", ascending=True),
    }
    return PodiumModel(clf=clf, feature_cols=tuple(FEATURE_COLS), metrics=metrics)


def predict_race(model: PodiumModel, race_features: pd.DataFrame) -> pd.DataFrame:
    """Return drivers with their podium probability for one race, best first."""
    out = race_features.copy()
    out["podium_prob"] = model.clf.predict_proba(out[list(model.feature_cols)])[:, 1]
    cols = ["driver", "team", "grid", "podium_prob"]
    return out[[c for c in cols if c in out.columns]].sort_values("podium_prob", ascending=False)
