"""Outcome predictor orchestration — championship projection + per-race podium.

Assembles the standalone championship simulator and the podium classifier into a
single JSON-friendly payload for the API (and, previously, the Streamlit app).
Kept here, not in the API layer, so it's reusable and tested in one place.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.standalone.championship import predict_season, project_ongoing_season
from f1se.standalone.podium import build_features, predict_race, train_podium_model


def _resolve_results(data_dir: str | Path | None) -> Path | None:
    candidates = []
    if data_dir is not None:
        candidates.append(Path(data_dir) / "results.parquet")
    candidates += [Path.cwd() / "data" / "processed" / "results.parquet",
                   PROJECT_ROOT / "data" / "processed" / "results.parquet"]
    for c in candidates:
        if c.exists():
            return c
    return None


def _f(x) -> float | None:
    return None if pd.isna(x) else float(x)


def compute_outcome(data_dir: str | Path | None = None, *, n_sims: int = 5000) -> dict | None:
    """Build the full outcome payload, or ``None`` if the results dataset is absent.

    Mirrors the old Streamlit ``load_outcome``: trains the podium model with a
    forward holdout on the latest season, projects (or simulates) the title race,
    and returns per-round podium predictions with the actual podium flagged.
    """
    fp = _resolve_results(data_dir)
    if fp is None:
        return None
    results = pd.read_parquet(fp)
    # Recency-weight form (halflife ~4 races) so a team's mid-season upgrade step
    # shows up quickly instead of being diluted by a flat window.
    feats = build_features(results, recency_halflife=4.0)
    test_year = int(results["year"].max())
    model = train_podium_model(feats, test_year=test_year)

    full = int(results.groupby("year")["round"].nunique().max())
    done = int(results[results["year"] == test_year]["round"].nunique())
    ongoing = done < full - 2
    champ = (project_ongoing_season(results, test_year, total_races=full, n_sims=n_sims)
             if ongoing else predict_season(results, test_year, n_sims=n_sims))

    has_points = "points" in champ.columns
    championship = [
        {"driver": str(r.driver), "win_prob": _f(r.win_prob),
         "points": _f(r.points) if has_points else None}
        for r in champ.head(8).itertuples(index=False)
    ]

    test = feats[feats["year"] == test_year]
    rounds_out = []
    for rnd in sorted(test["round"].unique()):
        race = test[test["round"] == rnd]
        pred = predict_race(model, race).head(8)
        podium = set(race[race["podium"] == 1]["driver"])
        rounds_out.append({
            "round": int(rnd),
            "event_name": str(race["event_name"].iloc[0]),
            "predictions": [
                {"driver": str(p.driver), "team": str(p.team), "grid": int(p.grid),
                 "podium_prob": _f(p.podium_prob), "actual": bool(p.driver in podium)}
                for p in pred.itertuples(index=False)
            ],
        })

    mtr = model.metrics
    return {
        "test_year": test_year, "ongoing": ongoing, "done": done, "full": full,
        "championship": championship,
        "model_metrics": {
            "auc": _f(mtr["auc"]),
            "model_precision_at_3": _f(mtr["model_precision_at_3"]),
            "grid_baseline_precision_at_3": _f(mtr["grid_baseline_precision_at_3"]),
        },
        "rounds": rounds_out,
    }


@lru_cache(maxsize=1)
def cached_outcome() -> dict | None:
    """Process-wide cached outcome payload (heavy: trains a model + Monte Carlo)."""
    return compute_outcome()


# --------------------------------------------------------------------------- #
# Predicting the NEXT (not-yet-raced) round — a real forward prediction.       #
# The podium model is grid + form only (no circuit feature), and an upcoming   #
# race has no grid until qualifying, so the grid defaults to each driver's     #
# current qualifying form and is overridable. Form is grid-independent, so it  #
# is computed once (cached) and only the grid varies per call — fast enough    #
# to re-predict live as the user edits the grid.                               #
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=1)
def _upcoming_context() -> dict | None:
    fp = _resolve_results(None)
    if fp is None:
        return None
    results = pd.read_parquet(fp)
    season = int(results["year"].max())
    done = results[results["year"] == season]
    if done.empty:
        return None
    next_round = int(done["round"].max()) + 1

    # Default grid = each driver's average grid so far (their qualifying form), ranked.
    avg_grid = done.groupby("driver", observed=True)["grid"].mean().sort_values()
    default_grid = {str(d): i + 1 for i, d in enumerate(avg_grid.index)}
    team = (done.sort_values("round").groupby("driver", observed=True).tail(1)
            .set_index("driver")["team"])

    rows = [{
        "year": season, "round": next_round, "event_name": "Next Race", "driver": d,
        "team": str(team.get(d, "")), "grid": float(default_grid[d]),
        "position": float("nan"), "points": float("nan"), "status": "Finished",
    } for d in default_grid]
    feats = build_features(pd.concat([results, pd.DataFrame(rows)], ignore_index=True),
                           recency_halflife=4.0)
    model = train_podium_model(feats, test_year=season)   # trains on < season only
    nxt = feats[(feats["year"] == season) & (feats["round"] == next_round)]
    return {
        "season": season, "next_round": next_round, "default_grid": default_grid,
        "clf": model.clf, "feature_cols": list(model.feature_cols),
        "rows": nxt[["driver", "team", *model.feature_cols]].reset_index(drop=True),
    }


def predict_upcoming(grid: dict[str, int] | None = None) -> dict | None:
    """Predict the next round's podium probabilities from current form.

    ``grid`` optionally overrides start positions (``{driver: grid_pos}``); any
    driver not listed keeps the form-based default. Returns drivers best-first.
    """
    ctx = _upcoming_context()
    if ctx is None:
        return None
    used = {**ctx["default_grid"], **{str(k): int(v) for k, v in (grid or {}).items()}}
    rows = ctx["rows"].copy()
    rows["grid"] = rows["driver"].map(lambda d: float(used.get(str(d), 20)))
    rows["podium_prob"] = ctx["clf"].predict_proba(rows[ctx["feature_cols"]])[:, 1]
    rows = rows.sort_values("podium_prob", ascending=False)
    return {
        "season": ctx["season"], "next_round": ctx["next_round"],
        "grid_source": "custom" if grid else "form",
        "predictions": [
            {"driver": str(r.driver), "team": str(r.team), "grid": int(r.grid),
             "podium_prob": _f(r.podium_prob)}
            for r in rows.itertuples(index=False)
        ],
    }
