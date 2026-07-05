"""Driver & constructor profile pages, from the results dataset.

Career and per-season aggregates, recent form, and the classic teammate
head-to-head (who out-qualified / out-raced whom). Results-only, so it lives
beside the other standalone predictors and is served without the engine.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from f1se.standalone.outcome import _f, _resolve_results


@lru_cache(maxsize=1)
def _results() -> pd.DataFrame | None:
    fp = _resolve_results(None)
    if fp is None:
        return None
    df = pd.read_parquet(fp).copy()
    df["_pos"] = pd.to_numeric(df["position"], errors="coerce")
    df["_grid"] = pd.to_numeric(df["grid"], errors="coerce")
    df["_pts"] = pd.to_numeric(df["points"], errors="coerce").fillna(0.0)
    df["race_idx"] = df["year"].astype(int) * 100 + df["round"].astype(int)
    return df


@lru_cache(maxsize=1)
def _sprints() -> pd.DataFrame | None:
    """Sprint points (they count toward championship totals; GP-only is wrong)."""
    from f1se.standalone.standings import load_sprints

    fp = _resolve_results(None)
    return load_sprints(fp) if fp is not None else None


def _sprint_pts(by: str, key: str, year: int | None = None) -> float:
    """Sprint points summed for one driver/team (optionally one season)."""
    sp = _sprints()
    if sp is None:
        return 0.0
    rows = sp[sp[by] == key]
    if year is not None:
        rows = rows[rows["year"] == year]
    return float(rows["points"].sum())


def _latest_team(sub: pd.DataFrame) -> str:
    return str(sub.sort_values("race_idx").iloc[-1]["team"])


def _status_finished(status: pd.Series) -> pd.Series:
    """True where the car was running at the flag ('Finished' or '+N Laps')."""
    s = status.astype("string").fillna("")
    return s.str.contains("Finished", na=False) | s.str.contains(r"\+\d+ Lap", regex=True, na=False)


# --- indexes (for the pickers) ----------------------------------------------
def drivers_index() -> list[dict] | None:
    df = _results()
    if df is None:
        return None
    rows = []
    for code, sub in df.groupby("driver", observed=True):
        seasons = sorted(int(y) for y in sub["year"].unique())
        rows.append({
            "driver": str(code), "team": _latest_team(sub),
            "last_season": seasons[-1], "seasons": seasons,
            "points": _f(sub["_pts"].sum() + _sprint_pts("driver", str(code))),
            "wins": int((sub["_pos"] == 1).sum()),
        })
    # Most-recently-active first, then by career points (current grid surfaces on top).
    rows.sort(key=lambda r: (-r["last_season"], -(r["points"] or 0)))
    return rows


def constructors_index() -> list[dict] | None:
    df = _results()
    if df is None:
        return None
    rows = []
    for team, sub in df.groupby("team", observed=True):
        seasons = sorted(int(y) for y in sub["year"].unique())
        rows.append({
            "team": str(team), "last_season": seasons[-1], "seasons": seasons,
            "points": _f(sub["_pts"].sum() + _sprint_pts("team", str(team))),
            "wins": int((sub["_pos"] == 1).sum()),
        })
    rows.sort(key=lambda r: (-r["last_season"], -(r["points"] or 0)))
    return rows


# --- driver profile ---------------------------------------------------------
def _season_line(s: pd.DataFrame) -> dict:
    pos = s["_pos"]
    classified = pos.notna()
    return {
        "races": int(s["race_idx"].nunique()),
        "wins": int((pos == 1).sum()),
        "podiums": int((pos <= 3).sum()),
        "points": _f(s["_pts"].sum()),
        "avg_grid": _f(s["_grid"].mean()) if s["_grid"].notna().any() else None,
        "avg_finish": _f(pos.mean()) if classified.any() else None,
        "best": int(pos.min()) if classified.any() else None,
        # DNF = not running at the flag, from race status (a classified retirement
        # still carries a finishing position, so status is the honest signal).
        "dnf": int((~_status_finished(s["status"])).sum()),
    }


def _teammate_h2h(df: pd.DataFrame, code: str, season: int) -> list[dict]:
    """Per-teammate qualifying & race head-to-head over ``season``."""
    s = df[(df["driver"] == code) & (df["year"] == season)]
    if s.empty:
        return []
    mates: dict[str, dict] = {}
    for _, me in s.iterrows():
        rnd, team = me["round"], me["team"]
        race = df[(df["year"] == season) & (df["round"] == rnd)
                  & (df["team"] == team) & (df["driver"] != code)]
        for _, m in race.iterrows():
            d = str(m["driver"])
            rec = mates.setdefault(d, {
                "teammate": d, "quali_races": 0, "quali_ahead": 0,
                "race_races": 0, "race_ahead": 0, "pts_self": 0.0, "pts_mate": 0.0,
            })
            if pd.notna(me["_grid"]) and pd.notna(m["_grid"]):
                rec["quali_races"] += 1
                rec["quali_ahead"] += int(me["_grid"] < m["_grid"])
            if pd.notna(me["_pos"]) and pd.notna(m["_pos"]):
                rec["race_races"] += 1
                rec["race_ahead"] += int(me["_pos"] < m["_pos"])
            rec["pts_self"] += float(me["_pts"])
            rec["pts_mate"] += float(m["_pts"])
    out = sorted(mates.values(), key=lambda x: -(x["quali_races"] + x["race_races"]))
    for r in out:
        r["pts_self"] = _f(r["pts_self"])
        r["pts_mate"] = _f(r["pts_mate"])
    return out


def driver_profile(code: str) -> dict | None:
    df = _results()
    if df is None:
        return None
    sub = df[df["driver"] == code]
    if sub.empty:
        return None
    seasons = sorted(int(y) for y in sub["year"].unique())

    by_season = []
    for yr in seasons:
        s = sub[sub["year"] == yr]
        line = {"season": yr, "team": _latest_team(s), **_season_line(s)}
        line["points"] = _f((line["points"] or 0) + _sprint_pts("driver", code, yr))
        by_season.append(line)

    recent = sub.sort_values("race_idx").tail(5)
    recent_out = [{
        "season": int(r["year"]), "round": int(r["round"]), "event_name": str(r["event_name"]),
        "grid": None if pd.isna(r["_grid"]) else int(r["_grid"]),
        "position": None if pd.isna(r["_pos"]) else int(r["_pos"]),
        "points": _f(r["_pts"]), "status": str(r["status"]),
    } for _, r in recent.iterrows()]
    recent_out.reverse()  # most recent first

    career = {"races": int(sub["race_idx"].nunique()), **_season_line(sub)}
    career["points"] = _f((career["points"] or 0) + _sprint_pts("driver", code))
    return {
        "driver": str(code), "team": _latest_team(sub), "seasons": seasons,
        "career": career,
        "by_season": by_season, "recent": recent_out,
        "h2h_season": seasons[-1], "teammate_h2h": _teammate_h2h(df, code, seasons[-1]),
    }


# --- constructor profile ----------------------------------------------------
def constructor_profile(team: str) -> dict | None:
    df = _results()
    if df is None:
        return None
    sub = df[df["team"] == team]
    if sub.empty:
        return None
    seasons = sorted(int(y) for y in sub["year"].unique())

    by_season = []
    for yr in seasons:
        s = sub[sub["year"] == yr]
        pos = s["_pos"]
        by_season.append({
            "season": yr, "races": int(s["race_idx"].nunique()),
            "wins": int((pos == 1).sum()), "podiums": int((pos <= 3).sum()),
            "points": _f(s["_pts"].sum() + _sprint_pts("team", team, yr)),
            "best": int(pos.min()) if pos.notna().any() else None,
            "drivers": sorted(str(d) for d in s["driver"].dropna().unique()),
        })

    drivers = []
    for d, ds in sub.groupby("driver", observed=True):
        drivers.append({"driver": str(d), "points": _f(ds["_pts"].sum()),
                        "wins": int((ds["_pos"] == 1).sum()),
                        "seasons": sorted(int(y) for y in ds["year"].unique())})
    drivers.sort(key=lambda x: -(x["points"] or 0))

    pos = sub["_pos"]
    return {
        "team": str(team), "seasons": seasons,
        "career": {"races": int(sub["race_idx"].nunique()), "wins": int((pos == 1).sum()),
                   "podiums": int((pos <= 3).sum()),
                   "points": _f(sub["_pts"].sum() + _sprint_pts("team", team)),
                   "best": int(pos.min()) if pos.notna().any() else None},
        "by_season": by_season, "drivers": drivers,
    }


# --- cached wrappers --------------------------------------------------------
@lru_cache(maxsize=1)
def cached_drivers_index() -> list[dict] | None:
    return drivers_index()


@lru_cache(maxsize=1)
def cached_constructors_index() -> list[dict] | None:
    return constructors_index()


@lru_cache(maxsize=64)
def cached_driver_profile(code: str) -> dict | None:
    return driver_profile(code)


@lru_cache(maxsize=64)
def cached_constructor_profile(team: str) -> dict | None:
    return constructor_profile(team)
