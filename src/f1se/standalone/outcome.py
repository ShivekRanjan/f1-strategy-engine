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
