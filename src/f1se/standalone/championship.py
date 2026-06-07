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
    strengths: pd.Series, n_races: int, *, n_sims: int = 5000, seed: int = 0
) -> pd.DataFrame:
    """Monte Carlo title probabilities for the given field over ``n_races``.

    Returns a frame (driver, win_prob, mean_points) sorted by win probability.
    """
    drivers = list(strengths.index)
    log_s = np.log(strengths.to_numpy(dtype=float))
    n_drivers = len(drivers)
    rng = np.random.default_rng(seed)

    totals = np.zeros((n_sims, n_drivers))
    k = min(len(POINTS_TABLE), n_drivers)
    for _ in range(n_races):
        # Gumbel-max sampling of a finishing order ~ Plackett-Luce(strengths).
        scores = log_s[None, :] + rng.gumbel(size=(n_sims, n_drivers))
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
