"""Score pre-registered calls against the committed result.

Only ever *appends* an outcome to a locked file — ``predictions`` and
``called_podium`` are never touched, because rewriting a prediction after
seeing the result would defeat the entire point of pre-registering it.

    python analysis/score_predictions.py

Scores every unscored file whose round is now present in the results dataset,
against both the model's call and the naive grid baseline it has to beat.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from f1se.config import PROJECT_ROOT


def _score_strategy(rec: dict, results: pd.DataFrame, laps: pd.DataFrame | None) -> bool:
    """Append the strategy outcome once the race's lap data exists. True if it did.

    Lap data arrives separately from results (it is pulled from a home
    connection, see scripts/refresh_laps.ps1), so this is independent of the
    podium scoring and may land a run later.
    """
    from collections import Counter

    call = rec.get("strategy")
    if not call or rec.get("strategy_result") or laps is None:
        return False
    race = laps[(laps["year"] == rec["season"]) & (laps["round"] == rec["round"])]
    win = results[(results["year"] == rec["season"]) & (results["round"] == rec["round"])
                  & (results["position"] == 1)]
    if race.empty or win.empty:
        return False
    stints = race.groupby("driver", observed=True)["stint"].nunique() - 1
    field = int(Counter(stints).most_common(1)[0][0])
    winner = str(win["driver"].iloc[0])
    win_stops = int(stints.get(winner, -1))
    rec["strategy_result"] = {
        "field_stops": field, "winner": winner, "winner_stops": win_stops,
        "matches_field": call["stops"] == field, "matches_winner": call["stops"] == win_stops,
    }
    r = rec["strategy_result"]
    print(f"  strategy: called {call['stops']} stop(s); field {field}, winner {winner} {win_stops}"
          f" -> field {'OK' if r['matches_field'] else 'miss'}, winner {'OK' if r['matches_winner'] else 'miss'}")
    return True


def main() -> int:
    fp = Path(PROJECT_ROOT) / "data" / "processed" / "results.parquet"
    if not fp.exists():
        print("no results dataset")
        return 1
    results = pd.read_parquet(fp)

    out_dir = Path(PROJECT_ROOT) / "predictions"
    files = sorted(out_dir.glob("*.json")) if out_dir.exists() else []
    if not files:
        print("no pre-registered predictions")
        return 0

    laps_fp = Path(PROJECT_ROOT) / "data" / "processed" / "dry_laps.parquet"
    laps = pd.read_parquet(laps_fp) if laps_fp.exists() else None

    for f in files:
        rec = json.loads(f.read_text(encoding="utf-8"))
        if _score_strategy(rec, results, laps):
            f.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")
        if rec.get("scored"):
            print(f"{f.name}: already scored — hit@3 {rec['result']['hit_at_3']}/3")
            continue
        race = results[(results["year"] == rec["season"]) & (results["round"] == rec["round"])]
        if race.empty:
            print(f"{f.name}: round {rec['round']} hasn't run yet")
            continue

        podium = list(race.sort_values("position").head(3)["driver"])
        called = rec["called_podium"]
        # The bar the model has to clear: just take the front row + P3.
        baseline = list(race.sort_values("grid").head(3)["driver"])
        finish = dict(zip(race["driver"], race["position"]))

        rec["result"] = {
            "actual_podium": podium,
            "hit_at_3": len(set(called) & set(podium)),
            "grid_baseline_podium": baseline,
            "grid_baseline_hit_at_3": len(set(baseline) & set(podium)),
            "called_finished": {
                p["driver"]: (int(finish[p["driver"]]) if p["driver"] in finish
                              and pd.notna(finish[p["driver"]]) else None)
                for p in rec["predictions"][:3]
            },
        }
        rec["scored"] = True
        f.write_text(json.dumps(rec, indent=2) + "\n", encoding="utf-8")

        r = rec["result"]
        print(f"{f.name}")
        print(f"  called  {called} -> hit@3 {r['hit_at_3']}/3")
        print(f"  actual  {podium}")
        print(f"  grid baseline {baseline} -> hit@3 {r['grid_baseline_hit_at_3']}/3")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
