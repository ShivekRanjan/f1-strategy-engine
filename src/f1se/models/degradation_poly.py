"""Phase 2 (correctness) — polynomial degradation, to test for a tyre 'cliff'.

The linear baseline assumes pace loss grows at a constant rate with tyre age.
Real degradation *accelerates* late in a stint (the cliff), which a linear model
misses — and which made the optimiser over-prefer soft tyres. This module fits a
per-compound polynomial (default quadratic) of fuel-corrected pace vs tyre age,
within-stint (fixed effects, same as the linear baseline), so we can:

  * estimate the curvature (a positive age^2 term = accelerating degradation), and
  * test on the leakage-safe forward holdout whether curvature actually
    generalises or is just overfitting sparse late-stint data.

Same within-stint target and :func:`f1se.models.degradation.shape_mae` metric as
the linear baseline, so degree-1 here reproduces the linear result and degree-2
is a like-for-like comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from f1se.models.degradation import (
    AGE_COL,
    DEFAULT_GROUP_COLS,
    STINT_KEYS,
    TARGET_COL,
    _demean_within_stint,
    _stint_demeaned,
)


@dataclass(frozen=True)
class PolyDegradationModel:
    """Per-group polynomial degradation coefficients (no intercept; FE target)."""

    degree: int
    group_cols: tuple[str, ...]
    coeffs: dict[tuple, np.ndarray]
    compound_coeffs: dict[str, np.ndarray]
    global_coeffs: np.ndarray
    meta: dict = field(default_factory=dict)

    def _coef(self, compound: str, track: str | None) -> np.ndarray:
        if track is not None:
            key = tuple({"event_name": track, "compound": compound}[c] for c in self.group_cols)
            if key in self.coeffs:
                return self.coeffs[key]
        if compound in self.compound_coeffs:
            return self.compound_coeffs[compound]
        return self.global_coeffs

    def pace_loss(self, compound: str, tyre_age: float, *, track: str | None = None) -> float:
        """Predicted pace loss (s) vs fresh: sum_k b_k * age**k."""
        b = self._coef(compound, track)
        return float(sum(b[k] * tyre_age ** (k + 1) for k in range(len(b))))


def _powers(age: np.ndarray, degree: int) -> np.ndarray:
    """Column stack of age**1 .. age**degree."""
    return np.column_stack([age ** k for k in range(1, degree + 1)])


def _fe_coeffs(Xdm: np.ndarray, ydm: np.ndarray) -> np.ndarray | None:
    """Least-squares coefficients of ydm ~ Xdm (no intercept), column-scaled."""
    if Xdm.shape[0] < Xdm.shape[1] + 1:
        return None
    norms = np.linalg.norm(Xdm, axis=0)
    if np.any(norms == 0):
        return None
    b_scaled, *_ = np.linalg.lstsq(Xdm / norms, ydm, rcond=None)
    return b_scaled / norms


def fit_poly(
    laps: pd.DataFrame,
    *,
    degree: int = 2,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_laps: int = 30,
) -> PolyDegradationModel:
    """Fit per-group within-stint polynomial degradation (degree>=1)."""
    for col in (*group_cols, *STINT_KEYS, AGE_COL, TARGET_COL):
        if col not in laps.columns:
            raise ValueError(f"poly degradation fit needs column '{col}'")

    laps = laps.reset_index(drop=True)
    age = laps[AGE_COL].to_numpy(float)
    # Demean each power column within stint (fixed effects on base pace).
    powers = _powers(age, degree)
    Xdm = np.column_stack([_demean_within_stint(powers[:, j], laps) for j in range(degree)])
    _, ydm = _stint_demeaned(laps)

    coeffs: dict[tuple, np.ndarray] = {}
    for key, idx in laps.groupby(list(group_cols), observed=True).indices.items():
        if len(idx) < min_laps:
            continue
        b = _fe_coeffs(Xdm[idx], ydm[idx])
        if b is not None:
            coeffs[key if isinstance(key, tuple) else (key,)] = b

    compound_coeffs: dict[str, np.ndarray] = {}
    for comp, idx in laps.groupby("compound", observed=True).indices.items():
        if len(idx) >= min_laps:
            b = _fe_coeffs(Xdm[idx], ydm[idx])
            if b is not None:
                compound_coeffs[str(comp)] = b

    gb = _fe_coeffs(Xdm, ydm)
    global_coeffs = gb if gb is not None else np.zeros(degree)

    return PolyDegradationModel(
        degree=degree,
        group_cols=tuple(group_cols),
        coeffs=coeffs,
        compound_coeffs=compound_coeffs,
        global_coeffs=global_coeffs,
        meta={"n_groups": len(coeffs), "min_laps": min_laps, "n_train_laps": len(laps)},
    )


def poly_shape(model: PolyDegradationModel, laps: pd.DataFrame) -> np.ndarray:
    """Per-row degradation shape g(age) for the shared shape_mae metric."""
    age = laps[AGE_COL].to_numpy(float)
    out = np.zeros(len(laps))
    for i, (c, a, t) in enumerate(zip(laps["compound"], age, laps["event_name"])):
        b = model._coef(c, t)
        out[i] = sum(b[k] * a ** (k + 1) for k in range(len(b)))
    return out
