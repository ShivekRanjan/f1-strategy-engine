"""Phase 2 — tyre degradation model. STUB (fixed signatures).

Models fuel-corrected lap time as a function of tyre age, compound, track, and
conditions. The deliverable is a function the simulator can call to ask "given
this compound at this age on this track, how much pace am I losing?"

Build order (a project rule — beat a dumb baseline first):
  1. Per-stint linear degradation (slope + intercept per compound) — the baseline.
  2. A pooled / hierarchical or boosted model (XGBoost) that shares strength
     across stints and conditions, only once it beats the baseline.

Validation is the part interviewers will probe — NO shuffled split. Laps within
a race are correlated, so we split by race (GroupKFold on a race id) AND keep a
forward-in-time holdout (train <=2023, test 2024). See :mod:`f1se.eval` (TODO).
"""

from __future__ import annotations

import pandas as pd


def fit_linear_baseline(laps: pd.DataFrame) -> "DegradationModel":  # noqa: F821
    """Fit per-compound linear degradation (the baseline). TODO Phase 2."""
    raise NotImplementedError("Phase 2: per-stint linear degradation baseline")


def predict_pace_loss(
    model: "DegradationModel",  # noqa: F821
    compound: str,
    tyre_age: int,
    *,
    track: str | None = None,
) -> float:
    """Predicted fuel-corrected pace loss (s) vs a fresh tyre. TODO Phase 2."""
    raise NotImplementedError("Phase 2: degradation prediction")
