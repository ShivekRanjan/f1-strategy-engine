"""Championship standings + a live title projection, from the results dataset.

Driver and constructor standings are a points tally of a season — **including
sprint-race points** (from ``sprint_points.parquet``; a GP-only tally is simply
wrong on sprint seasons: in 2026 it even swapped P2/P3). Wins/podiums follow the
official convention and count Grands Prix only. The projection reuses the
Monte-Carlo season simulator (:func:`project_ongoing_season`) so the "who wins
the title" number carries honest uncertainty rather than a naive points
extrapolation. Results-only (no lap data), so it lives beside the outcome
predictor and is served without the :class:`~f1se.engine.StrategyEngine`.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from f1se.standalone.championship import project_ongoing_season
from f1se.standalone.outcome import _f, _resolve_results


def _season_rows(results: pd.DataFrame, season: int) -> pd.DataFrame:
    return results[results["year"] == season]


def load_sprints(results_fp: Path) -> pd.DataFrame | None:
    """The sprint-points table living beside results.parquet (None if absent)."""
    fp = results_fp.parent / "sprint_points.parquet"
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    return df if not df.empty else None


def _sprint_sums(sprints: pd.DataFrame | None, season: int, by: str) -> pd.Series:
    """Sprint points for ``season`` summed per driver/team (empty Series if none)."""
    if sprints is None:
        return pd.Series(dtype=float)
    ss = sprints[sprints["year"] == season]
    if ss.empty:
        return pd.Series(dtype=float)
    return ss.groupby(by, observed=True)["points"].sum()


def driver_standings(results: pd.DataFrame, season: int,
                     sprints: pd.DataFrame | None = None) -> list[dict]:
    """Points tally for ``season`` (GP + sprint), best-first.

    Wins/podiums/races count Grands Prix only (the official convention — a
    sprint win is not a race win); sprint points are added to the totals.
    """
    sr = _season_rows(results, season)
    pos = pd.to_numeric(sr["position"], errors="coerce")
    agg = (
        sr.assign(_pos=pos, _win=(pos == 1), _pod=(pos <= 3))
        .groupby("driver", observed=True)
        .agg(points=("points", "sum"), wins=("_win", "sum"),
             podiums=("_pod", "sum"), races=("round", "nunique"))
        .reset_index()
    )
    spr = _sprint_sums(sprints, season, "driver")
    agg["points"] = agg["points"] + agg["driver"].map(spr).fillna(0.0)
    # Latest team a driver drove for (handles a mid-season seat change).
    team = (sr.sort_values("round").groupby("driver", observed=True).tail(1)
            .set_index("driver")["team"])
    agg["team"] = agg["driver"].map(lambda d: str(team.get(d, "")))
    agg = agg.sort_values(["points", "wins"], ascending=False).reset_index(drop=True)
    return [
        {"pos": i + 1, "driver": str(r.driver), "team": r.team,
         "points": _f(r.points), "wins": int(r.wins), "podiums": int(r.podiums),
         "races": int(r.races)}
        for i, r in enumerate(agg.itertuples(index=False))
    ]


def constructor_standings(results: pd.DataFrame, season: int,
                          sprints: pd.DataFrame | None = None) -> list[dict]:
    """Constructor (team) points tally for ``season`` (GP + sprint), best-first."""
    sr = _season_rows(results, season)
    pos = pd.to_numeric(sr["position"], errors="coerce")
    agg = (
        sr.assign(_win=(pos == 1), _pod=(pos <= 3))
        .groupby("team", observed=True)
        .agg(points=("points", "sum"), wins=("_win", "sum"), podiums=("_pod", "sum"))
        .reset_index()
    )
    spr = _sprint_sums(sprints, season, "team")
    agg["points"] = agg["points"] + agg["team"].map(spr).fillna(0.0)
    agg = agg.sort_values(["points", "wins"], ascending=False).reset_index(drop=True)
    return [
        {"pos": i + 1, "team": str(r.team), "points": _f(r.points),
         "wins": int(r.wins), "podiums": int(r.podiums)}
        for i, r in enumerate(agg.itertuples(index=False))
    ]


def compute_standings(
    data_dir: str | Path | None = None, season: int | None = None, *, n_sims: int = 4000,
) -> dict | None:
    """Standings payload for ``season`` (default: latest), or ``None`` if no data.

    When the requested season is the latest and still in progress, each driver's
    title-win probability from the Monte-Carlo projection is attached — the same
    engine the Outcome tab uses, so the standings page shows not just *where* the
    title race stands but *how likely* each contender is to finish on top.
    """
    fp = _resolve_results(data_dir)
    if fp is None:
        return None
    results = pd.read_parquet(fp)
    sprints = load_sprints(fp)
    seasons = sorted(int(y) for y in results["year"].dropna().unique())
    if not seasons:
        return None
    latest = seasons[-1]
    season = int(season) if season is not None else latest
    if season not in seasons:
        return None

    total_races = int(results.groupby("year")["round"].nunique().max())
    done = int(_season_rows(results, season)["round"].nunique())
    # "Ongoing" mirrors the outcome predictor: a live projection only makes sense
    # for the current season before it's effectively decided.
    ongoing = season == latest and done < total_races - 2

    drivers = driver_standings(results, season, sprints)
    if ongoing:
        proj = project_ongoing_season(
            results, season, total_races=total_races, n_sims=n_sims,
            extra_points=_sprint_sums(sprints, season, "driver"),
        )
        win_prob = {str(r.driver): _f(r.win_prob) for r in proj.itertuples(index=False)}
        for d in drivers:
            d["win_prob"] = win_prob.get(d["driver"])

    return {
        "season": season,
        "seasons": seasons,
        "latest": latest,
        "races_done": done,
        "total_races": total_races,
        "ongoing": ongoing,
        "includes_sprints": sprints is not None,
        "drivers": drivers,
        "constructors": constructor_standings(results, season, sprints),
    }


@lru_cache(maxsize=8)
def cached_standings(season: int | None = None) -> dict | None:
    """Process-wide cached standings (the projection runs a Monte-Carlo sim)."""
    return compute_standings(season=season)
