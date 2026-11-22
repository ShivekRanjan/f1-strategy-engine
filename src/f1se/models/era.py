"""Era-aware degradation via partial pooling (shrinkage) across a rule change.

2026 is a regulation reset, so a model trained on the old cars is biased for
2026, while 2026 data alone is too thin early in the season. We blend them:
each per-group degradation slope and base pace is a precision-weighted average of
the **2026 estimate** and the **pre-2026 prior**, weighted by how many 2026 laps
we actually have::

    shrunk = (n_2026 * estimate_2026 + k * prior) / (n_2026 + k)

Early in 2026 (little data) it leans on the prior; as 2026 races accumulate it
converges to the 2026-specific value. ``k`` (``shrinkage_laps``) is the
"how much to trust the old cars" knob — smaller = trust 2026 data sooner. This is
the same partial-pooling idea used for the per-track safety-car model, applied
across the regulation boundary. Only the *regime-sensitive* components (pace,
degradation) are shrunk; pit loss / safety-car / fuel transfer and use all data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.models.degradation import (
    AGE_COL,
    DEFAULT_GROUP_COLS,
    TARGET_COL,
    DegradationModel,
    _fe_slope,
    _stint_demeaned,
    fit_linear_baseline,
    recency_weights,
)


def fit_era_shrunk_degradation(
    laps: pd.DataFrame,
    *,
    target_min_year: int = 2026,
    group_cols: tuple[str, ...] = DEFAULT_GROUP_COLS,
    min_laps: int = 20,
    shrinkage_laps: float | dict[str, float] = 150.0,
    recency_halflife: float | None = None,
    era_delta: str = "replace",
) -> DegradationModel:
    """Degradation model for the target era, shrunk toward the pre-era prior.

    Fits the prior on ``year < target_min_year`` and blends in per-group
    estimates from ``year >= target_min_year`` weighted by 2026 lap counts.
    Falls back to the plain prior if there is no target-era data yet.

    ``recency_halflife`` (races) recency-weights the target-era estimate so the
    most recent races dominate — keeping the slopes responsive to mid-season car
    upgrades rather than averaging the whole season flat.
    """
    laps = laps.dropna(subset=[AGE_COL, TARGET_COL])
    prior_laps = laps[laps["year"] < target_min_year]
    target_laps = laps[laps["year"] >= target_min_year]
    # The pre-era prior stays a flat, stable baseline; recency-weighting applies
    # only to the *target* era, where mid-season upgrades actually move the pace.
    prior = fit_linear_baseline(prior_laps, group_cols=group_cols, min_laps=min_laps)
    if target_laps.empty:
        return prior

    tl = target_laps.reset_index(drop=True)
    age_dm, corr_dm = _stint_demeaned(tl)
    tl["_age_dm"], tl["_corr_dm"] = age_dm, corr_dm
    w_tl = recency_weights(tl, recency_halflife)
    tl["_w"] = 1.0 if w_tl is None else w_tl
    use_w = recency_halflife is not None

    def _w(grp: pd.DataFrame) -> np.ndarray | None:
        return grp["_w"].to_numpy(float) if use_w else None

    def _wbase(grp: pd.DataFrame, slope: float) -> float:
        wt = grp["_w"].to_numpy(float) if use_w else None
        ybar = float(np.average(grp[TARGET_COL].to_numpy(float), weights=wt))
        abar = float(np.average(grp[AGE_COL].to_numpy(float), weights=wt))
        return ybar - slope * abar

    def _k_for(compound: str | None) -> float:
        """Prior strength for this group.

        ``shrinkage_laps`` may be a scalar (one k for everything, the original
        behaviour) or a per-compound mapping. The mapping exists because lap
        counts differ sharply by compound — the hards carry several times the
        evidence of the softs — so in principle each deserves its own prior
        weight. Whether that is worth doing is an empirical question, answered
        in `analysis/phase_2026_shrinkage_percompound.py`.
        """
        if isinstance(shrinkage_laps, dict):
            return float(shrinkage_laps.get(str(compound), shrinkage_laps.get("_default", 150.0)))
        return float(shrinkage_laps)

    def _blend(n: int, est: float, prior_val: float, compound: str | None = None) -> float:
        k = _k_for(compound)
        return (n * est + k * prior_val) / (n + k)

    def _kd(key) -> dict:
        return dict(zip(group_cols, key if isinstance(key, tuple) else (key,)))

    slopes = dict(prior.slopes)
    intercepts = dict(prior.intercepts)
    n_2026: dict[tuple, int] = {}

    for key, grp in tl.groupby(list(group_cols), observed=True):
        s_t = _fe_slope(grp["_age_dm"].to_numpy(float), grp["_corr_dm"].to_numpy(float), _w(grp))
        if s_t is None:
            continue
        kd = _kd(key)
        track, compound = kd.get("event_name"), kd.get("compound")
        key = key if isinstance(key, tuple) else (key,)
        n = len(grp)
        s_prior = prior.raw_slope(compound, track)
        slopes[key] = _blend(n, s_t, s_prior, compound)
        # Base pace: target mean de-aged at the (shrunk) slope, blended with prior.
        b_t = _wbase(grp, slopes[key])
        b_prior = prior.intercepts.get(key, prior.track_base.get(track, prior.global_base))
        intercepts[key] = _blend(n, b_t, b_prior, compound)
        n_2026[key] = n

    # Per-track base + per-compound slope fallbacks, also shrunk.
    track_base = dict(prior.track_base)
    for ev, grp in tl.groupby("event_name", observed=True):
        b_t = _wbase(grp, prior.global_slope)
        track_base[str(ev)] = _blend(len(grp), b_t, prior.track_base.get(str(ev), prior.global_base))

    compound_slope = dict(prior.compound_slope)
    for comp, grp in tl.groupby("compound", observed=True):
        s_t = _fe_slope(grp["_age_dm"].to_numpy(float), grp["_corr_dm"].to_numpy(float), _w(grp))
        if s_t is not None:
            s_prior = prior.compound_slope.get(str(comp), prior.global_slope)
            compound_slope[str(comp)] = _blend(len(grp), s_t, s_prior, str(comp))

    # Propagate the 2026 era shift to tracks NOT yet raced in 2026: their
    # pre-2026 per-track slope is nudged by how much that compound's degradation
    # moved under the new regs (the recency-weighted compound-level delta). Without
    # this, a track like Austria — raced in 2023-25 but not yet in 2026 — would
    # keep its stale pre-reset slope and ignore the regime change entirely.
    if "compound" in group_cols:
        ci = group_cols.index("compound")
        if era_delta == "stint_median":
            deltas = _stint_median_deltas(prior_laps, target_laps)
        elif era_delta == "replace":
            # No borrowing from the track's old-car history: a track not yet
            # raced in the target era gets the era's median-stint rate outright.
            level = _stint_medians(target_laps)
            deltas = level
        elif era_delta == "pooled":
            deltas = {c: compound_slope[c] - prior.compound_slope.get(c, prior.global_slope)
                      for c in compound_slope}
        else:
            raise ValueError("era_delta must be 'pooled', 'stint_median' or 'replace', "
                             f"not {era_delta!r}")
        for key, s in prior.slopes.items():
            if key in n_2026:
                continue                       # already has a direct 2026 estimate
            comp = key[ci] if isinstance(key, tuple) else None
            if comp in deltas:
                slopes[key] = deltas[comp] if era_delta == "replace" else s + deltas[comp]

    return DegradationModel(
        group_cols=tuple(group_cols),
        slopes=slopes,
        intercepts=intercepts,
        compound_slope=compound_slope,
        global_slope=prior.global_slope,
        track_base=track_base,
        global_base=prior.global_base,
        meta={
            "era": "shrunk",
            "target_min_year": target_min_year,
            "shrinkage_laps": shrinkage_laps,
            "recency_halflife": recency_halflife,
            "era_delta": era_delta,
            "n_target_laps": int(len(target_laps)),
            "n_shrunk_groups": len(n_2026),
        },
    )


def _stint_median_deltas(prior_laps: pd.DataFrame, target_laps: pd.DataFrame) -> dict[str, float]:
    """Per-compound era shift as the change in the *median stint's* slope.

    The pooled fixed-effects slope weights a stint by its age spread, roughly
    the cube of its length, so it measures long stints. When the field changes
    how long it runs a compound, the pooled slope moves even if the tyre
    didn't: 2026 teams ran 3.5 times as many soft laps in 25-lap-plus stints
    as in 2025, and the pooled soft shift came out at -0.040 s/lap against -0.027
    for the median stint. Every stint counts once here.
    """
    before, after = _stint_medians(prior_laps), _stint_medians(target_laps)
    return {c: after[c] - before[c] for c in after if c in before}


def _stint_medians(laps: pd.DataFrame) -> dict[str, float]:
    """Median per-stint degradation slope for each compound."""
    from f1se.eda import fit_stint_slopes

    sl = fit_stint_slopes(laps.assign(race=laps["event_name"]))
    med = sl.groupby("compound", observed=True)["slope_s_per_lap"].median()
    return {str(c): float(v) for c, v in med.items()}
