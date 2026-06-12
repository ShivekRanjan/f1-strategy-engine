"""Phase A — championship predictor via Monte Carlo season simulation.

Estimates each driver's title probability by simulating the season many times.
Per-race finishing orders are sampled from a Plackett-Luce model (drivers ranked
by a "strength"), using the Gumbel-max trick so the whole thing is vectorised.
This reuses the project's theme: outcomes as distributions, not point predictions.

Strengths come from recent average points (results-only), so a season can be
projected from prior form — the same forward-in-time spirit as everywhere else.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# F1 points for finishing positions 1..10 (fastest-lap/sprint points ignored).
POINTS_TABLE = np.array([25, 18, 15, 12, 10, 8, 6, 4, 2, 1], dtype=float)


def driver_strengths(results: pd.DataFrame, *, before_year: int, window: int = 22) -> pd.Series:
    """Strength per driver = recent average points before ``before_year`` (+ floor).

    The small floor keeps every driver's Plackett-Luce weight positive so even a
    backmarker can, rarely, score.
    """
    hist = results[results["year"] < before_year].copy()
    hist["race_idx"] = hist["year"] * 100 + hist["round"]
    recent = hist.sort_values("race_idx").groupby("driver", observed=True).tail(window)
    avg_pts = recent.groupby("driver", observed=True)["points"].mean()
    return (avg_pts + 0.5).rename("strength")


def simulate_championship(
    strengths: pd.Series, n_races: int, *, starting_points: pd.Series | None = None,
    strength_samples: np.ndarray | None = None, n_sims: int = 5000, seed: int = 0,
) -> pd.DataFrame:
    """Monte Carlo title probabilities for the given field over ``n_races``.

    ``starting_points`` (optional, per driver) seeds the standings — use it for an
    in-season projection (points already scored + the remaining races simulated).
    ``strength_samples`` (optional, ``(n_sims, n_drivers)``) propagates *parameter*
    uncertainty: each simulation uses its own strength draw (e.g. a bootstrap over
    the races seen so far) instead of treating form as known exactly — without it,
    a mid-season leader can show an overconfident ~100% title probability.
    Returns a frame (driver, win_prob, mean_points) sorted by win probability.
    """
    drivers = list(strengths.index)
    n_drivers = len(drivers)
    rng = np.random.default_rng(seed)
    if strength_samples is not None:
        if strength_samples.shape != (n_sims, n_drivers):
            raise ValueError("strength_samples must be (n_sims, n_drivers)")
        log_s = np.log(np.maximum(strength_samples, 1e-9))
    else:
        log_s = np.log(strengths.to_numpy(dtype=float))[None, :]  # broadcast over sims

    base = (starting_points.reindex(drivers).fillna(0.0).to_numpy(dtype=float)
            if starting_points is not None else np.zeros(n_drivers))
    totals = np.tile(base, (n_sims, 1)).astype(float)
    k = min(len(POINTS_TABLE), n_drivers)
    for _ in range(n_races):
        # Gumbel-max sampling of a finishing order ~ Plackett-Luce(strengths).
        scores = log_s + rng.gumbel(size=(n_sims, n_drivers))
        order = np.argsort(-scores, axis=1)            # finishing order (driver indices)
        rows = np.arange(n_sims)[:, None]
        totals[rows, order[:, :k]] += POINTS_TABLE[:k] # award points to top finishers

    champ = totals.argmax(axis=1)
    win_prob = np.bincount(champ, minlength=n_drivers) / n_sims
    return (pd.DataFrame({"driver": drivers, "win_prob": win_prob,
                          "mean_points": totals.mean(axis=0)})
            .sort_values("win_prob", ascending=False).reset_index(drop=True))


def predict_season(results: pd.DataFrame, season: int, *, n_sims: int = 5000, seed: int = 0) -> pd.DataFrame:
    """Project ``season``'s title race from prior-season form (drivers who raced it)."""
    field = sorted(results[results["year"] == season]["driver"].dropna().unique())
    strengths = driver_strengths(results, before_year=season)
    # Restrict to the season's actual field; unseen drivers get the floor strength.
    strengths = strengths.reindex(field).fillna(0.5)
    n_races = int(results[results["year"] == season]["round"].nunique())
    return simulate_championship(strengths, n_races, n_sims=n_sims, seed=seed)


def project_ongoing_season(
    results: pd.DataFrame, season: int, *, total_races: int = 24,
    n_sims: int = 5000, seed: int = 0,
) -> pd.DataFrame:
    """Mid-season title projection: current points + simulate the remaining races.

    Strengths come from *this season's* form so far (crucial after a regulation
    reset, when last season's order no longer applies) — and, because only a few
    races of evidence exist, each simulation BOOTSTRAPS the driver's results to
    propagate how uncertain that form estimate still is. Without this, a leader
    shows a dishonest ~100% title probability after a handful of rounds.
    Returns (driver, win_prob, mean_points) plus the current standings, with
    ``races_done`` in ``.attrs``.
    """
    sr = results[results["year"] == season]
    done = int(sr["round"].nunique())
    field = sorted(sr["driver"].dropna().unique())
    current = sr.groupby("driver", observed=True)["points"].sum().reindex(field).fillna(0.0)
    strengths = (sr.groupby("driver", observed=True)["points"].mean() + 0.5).reindex(field).fillna(0.5)
    remaining = max(total_races - done, 0)

    # Bootstrap strength draws: resample each driver's per-race points (with
    # replacement) once per simulation -> (n_sims, n_drivers) strength matrix.
    rng = np.random.default_rng(seed + 1)
    samples = np.full((n_sims, len(field)), 0.5)
    for j, drv in enumerate(field):
        pts = sr[sr["driver"] == drv]["points"].to_numpy(dtype=float)
        if pts.size:
            idx = rng.integers(0, pts.size, size=(n_sims, pts.size))
            samples[:, j] = pts[idx].mean(axis=1) + 0.5

    out = simulate_championship(strengths, remaining, starting_points=current,
                                strength_samples=samples, n_sims=n_sims, seed=seed)
    out = out.merge(current.rename("points_now").reset_index(), on="driver", how="left")
    out.attrs["races_done"] = done
    out.attrs["total_races"] = total_races
    return out
