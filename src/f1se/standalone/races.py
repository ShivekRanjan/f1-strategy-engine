"""Race Hub — one race's full story, assembled from the results dataset.

Pairs the actual finishing order with the podium model's *pre-race* prediction —
trained only on prior seasons (the same forward-in-time discipline as the Outcome
tab), so the card honestly shows what the model expected *before* the race and
what actually happened. Results-only; the strategy call and degradation curves
the Race Hub also shows are fetched by the frontend from the engine endpoints.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from f1se.standalone.outcome import _f, _resolve_results
from f1se.standalone.podium import build_features, predict_race, train_podium_model


@lru_cache(maxsize=1)
def _results_and_features() -> tuple[pd.DataFrame, pd.DataFrame] | None:
    fp = _resolve_results(None)
    if fp is None:
        return None
    results = pd.read_parquet(fp)
    feats = build_features(results, recency_halflife=4.0)
    return results, feats


@lru_cache(maxsize=8)
def _season_model(season: int):
    """Podium model trained on seasons < ``season`` (None if no prior data)."""
    rf = _results_and_features()
    if rf is None:
        return None
    _, feats = rf
    try:
        return train_podium_model(feats, test_year=season)
    except ValueError:
        return None  # earliest season — no prior races to train on


def race_card(season: int, track: str) -> dict | None:
    """Finishing order + pre-race podium prediction for one race, or ``None``."""
    rf = _results_and_features()
    if rf is None:
        return None
    results, feats = rf
    race = feats[(feats["year"] == season) & (feats["event_name"] == track)]
    if race.empty:
        return None
    rnd = int(race["round"].iloc[0])

    # Actual finishing order (raw results — keep grid 0 = pit-lane start as-is).
    r = results[(results["year"] == season) & (results["event_name"] == track)].copy()
    r["_pos"] = pd.to_numeric(r["position"], errors="coerce")
    r = r.sort_values("_pos", na_position="last")
    result = []
    for row in r.itertuples(index=False):
        pos = None if pd.isna(row.position) else int(row.position)
        grid = None if pd.isna(row.grid) else int(row.grid)
        result.append({
            "pos": pos, "driver": str(row.driver), "team": str(row.team),
            "grid": grid, "points": _f(row.points), "status": str(row.status),
            # positions gained (grid -> finish); None for a DNF or pit-lane start
            "gained": (grid - pos) if (pos is not None and grid) else None,
        })
    actual_podium = [x["driver"] for x in result if x["pos"] is not None and x["pos"] <= 3]

    # Pre-race prediction: the podium model trained only on earlier seasons.
    prediction = None
    model = _season_model(season)
    if model is not None:
        pred = predict_race(model, race)
        podset = set(actual_podium)
        preds = [
            {"driver": str(p.driver), "team": str(p.team), "grid": int(p.grid),
             "podium_prob": _f(p.podium_prob), "actual": bool(p.driver in podset)}
            for p in pred.itertuples(index=False)
        ]
        prediction = {
            "predictions": preds,
            "hit_at_3": sum(1 for p in preds[:3] if p["actual"]),
            "auc": _f(model.metrics["auc"]),
        }

    return {
        "season": int(season), "round": rnd, "event_name": str(track),
        "result": result, "actual_podium": actual_podium, "prediction": prediction,
    }


@lru_cache(maxsize=64)
def cached_race_card(season: int, track: str) -> dict | None:
    """Process-wide cached race card (trains/reuses the season's podium model)."""
    return race_card(season, track)
