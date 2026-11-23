r"""Learn the per-stop cost from the season so far, and only from it (METHODOLOGY §23).

The free-air simulator charges a stop its pit-lane time plus a hand-set
penalty (models/overtaking.py, 3.5-7.5 s) for what it can't see: traffic, a
slow stop, a badly timed safety car. In 2026 the field one-stopped in 16 of 21
races, and most of the engine's two-stop calls won by only 8-14 s on paper. So
how much more a team needs to see before taking the extra stop may be a
property of the season, not a constant.

Before each round, an extra per-stop cost is chosen from ``GRID``: whichever
value best matches the field's stop counts in that season's *earlier* rounds,
ties going to the value nearest zero. Every call is made before its race, so
this is a forward test even on races examined before. Developed on 2023-25
(each season learning from itself) before 2026.

Input: the engine's best one-stop vs two-stop margins per race, from
analysis/strategy_bias_diagnosis.py (``gap`` = two-stop minus one-stop rank
score, negative = the engine prefers two).

    .venv\Scripts\python.exe analysis/stop_penalty_walkforward.py
"""
from __future__ import annotations

import pandas as pd

from f1se.config import PROJECT_ROOT

CSV = PROJECT_ROOT / "analysis" / "figures" / "strategy_bias_diagnosis.csv"
GRID = (-10, -5, 0, 5, 10, 15, 20, 25, 30)


def calls(gap: pd.Series, extra: float) -> pd.Series:
    return (gap + extra < 0).map({True: 2, False: 1})


def walk_forward(season: pd.DataFrame) -> pd.DataFrame:
    season = season.sort_values("round").reset_index(drop=True)
    target = season["field"].clip(upper=2)
    chosen = []
    for i in range(len(season)):
        past = season.iloc[:i]
        if past.empty:
            chosen.append(0)
            continue
        hits = {x: int((calls(past["gap"], x) == target.iloc[:i]).sum()) for x in GRID}
        top = max(hits.values())
        chosen.append(min((x for x in GRID if hits[x] == top), key=abs))
    season["extra"] = chosen
    season["call"] = [calls(pd.Series([g]), x).iloc[0] for g, x in zip(season["gap"], chosen)]
    season["base_call"] = calls(season["gap"], 0)
    season["target"] = target
    return season


def main() -> None:
    d = pd.read_csv(CSV)
    out = pd.concat([walk_forward(g) for _, g in d.groupby("year")], ignore_index=True)
    out["era"] = out["year"].map(lambda y: "2026" if y >= 2026 else "2023-25")
    for era, g in out.groupby("era"):
        base = int((g.base_call == g.target).sum())
        learned = int((g.call == g.target).sum())
        print(f"{era}: fixed penalty {base}/{len(g)}  learned from the season so far {learned}/{len(g)}")
    for y, g in out.groupby("year"):
        print(f"  {y}: extra per-stop cost by round {list(g.extra)}")
    out.to_csv(PROJECT_ROOT / "analysis" / "figures" / "stop_penalty_walkforward.csv", index=False)


if __name__ == "__main__":
    main()
