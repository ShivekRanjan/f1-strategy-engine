r"""Strategy changes are developed on 2023-25, then scored once on 2026 (METHODOLOGY §20).

The 2026 stop counts have been looked at too often to choose anything on. The
54 dry 2023-25 races are independent of them, so a structural change to the
engine (one that isn't about the 2026 cars) is chosen here and only then
scored on 2026.

    .venv\Scripts\python.exe analysis/strategy_dev.py VARIANT [--years 2023,2024,2025]

Each race is predicted leave-one-out: its laps are removed before the
degradation model and any data-derived prior are fitted, because a cliff
derived from stint lengths would otherwise be read off the very strategy it
is meant to predict.
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT  # noqa: E402
from f1se.engine import StrategyEngine  # noqa: E402
from f1se.models.cliff import CliffPrior  # noqa: E402
from f1se.models.compound_pace import (  # noqa: E402
    pooled_offsets,
    season_offsets,
    with_compound_offsets,
)

PROC = PROJECT_ROOT / "data" / "processed"
FIG = Path(__file__).parent / "figures"


def variant_engine(name: str, eng: StrategyEngine, seen: pd.DataFrame, year: int,
                   offsets: pd.DataFrame | None = None) -> StrategyEngine:
    """Apply a named variant, using only the laps (and per-race offsets) the model may see."""
    era = seen[seen.year >= 2026] if year >= 2026 else seen[seen.year < 2026]
    if name == "current":
        return eng
    if name == "pace_lapfe":
        o = offsets[(offsets.year >= 2026) == (year >= 2026)]
        o = o.merge(seen[["year", "round"]].drop_duplicates(), on=["year", "round"])
        off = pooled_offsets(o)
        if year >= 2026:
            return replace(eng, deg_model_2026=with_compound_offsets(eng.deg_model_2026, off))
        return replace(eng, deg_model=with_compound_offsets(eng.deg_model, off))
    if name == "used_sets":
        from f1se.eda import used_set_ages
        ages = used_set_ages(era)
        return replace(eng, start_age=ages, start_age_2026=ages)
    if name == "no_cliff":
        return replace(eng, cliff=CliffPrior.disabled(), cliff_2026=CliffPrior.disabled())
    if name.startswith("cliff_q"):
        c = CliffPrior.from_stints(era, quantile=int(name[7:]) / 100)
        return replace(eng, cliff=c, cliff_2026=c)
    raise ValueError(f"unknown variant {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("variant")
    ap.add_argument("--years", default="2023,2024,2025")
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]

    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from strategy_bias_diagnosis import loo_engine
    from thermal_out_of_sample import weather

    dry = pd.read_parquet(PROC / "dry_laps.parquet")
    base = StrategyEngine.from_processed()
    offsets = season_offsets(dry) if a.variant == "pace_lapfe" else None
    rows = []
    for (year, rnd), race in dry[dry.year.isin(years)].groupby(["year", "round"]):
        year, rnd = int(year), int(rnd)
        track = str(race["event_name"].iloc[0])
        if track not in base.total_laps_by_track:
            continue
        if year < 2026:
            temp, rained = weather(year, rnd)
            if temp is None or rained:
                continue
        seen = dry[~((dry.year == year) & (dry["round"] == rnd))]
        eng = variant_engine(a.variant, loo_engine(base, dry, year, rnd), seen, year, offsets)
        from f1se.data.race_rules import max_stint_laps
        cap = max_stint_laps(year, str(race["event_name"].iloc[0]))
        rec = eng.recommend(track, season=year, n_runs=2000, max_stint_laps=cap,
                            max_stops=3 if cap else 2)["best"]
        field = int(Counter(race.groupby("driver", observed=True)["stint"].nunique() - 1)
                    .most_common(1)[0][0])
        rows.append({"year": year, "round": rnd, "race": track, "field": field,
                     "engine": len(rec["pit_laps"])})
    df = pd.DataFrame(rows)
    tag = "2026" if years == [2026] else "2023-25"
    df.to_csv(FIG / f"strategy_dev_{a.variant}_{tag}.csv", index=False)
    hit = int((df.engine == df.field).sum())
    over, under = int((df.engine > df.field).sum()), int((df.engine < df.field).sum())
    print(f"{a.variant:<12} {tag}: match {hit}/{len(df)}  over {over}  under {under}")


if __name__ == "__main__":
    main()
