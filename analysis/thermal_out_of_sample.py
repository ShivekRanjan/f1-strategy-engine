"""Does the thermal prior hold up on races that never shaped it?

    python analysis/thermal_out_of_sample.py

The 16-race 2026 backtest found two over-stopping misses at the season's hottest
races (Monza 55 C, Madrid 53 C), both above anything the thermal prior was
validated on. Diagnosing and fixing that *from those two races* would be fitting
the test. This runs the same leave-one-race-out strategy backtest on 2023-25
instead — races the prior's sensitivity was never tuned on — with the prior on
and off, banded by track temperature. If the linear prior over-stops on hot
races here too, the cause is established on independent data.

Wet races are skipped (any rainfall in FastF1's weather feed): stint counts
there reflect weather, not tyre wear.
"""

from __future__ import annotations

import sys
from collections import Counter
from dataclasses import replace
from pathlib import Path

import pandas as pd

from f1se.config import PROJECT_ROOT, enable_cache
from f1se.engine import StrategyEngine
from f1se.models.degradation import fit_linear_baseline
from f1se.models.thermal import ThermalPrior

PROC = PROJECT_ROOT / "data" / "processed"


def weather(year: int, rnd: int) -> tuple[float | None, bool]:
    """(mean track temp C, did it rain) for a race, from FastF1 weather."""
    enable_cache()
    import fastf1
    try:
        s = fastf1.get_session(year, rnd, "R")
        s.load(laps=False, telemetry=False, weather=True, messages=False)
        w = s.weather_data
        return float(w["TrackTemp"].mean()), bool(w["Rainfall"].any())
    except Exception:
        return None, False


BANDS = [(-1, 30, "cool  <30"), (30, 45, "mid 30-45"), (45, 99, "hot  45+")]


def main() -> None:
    dry = pd.read_parquet(PROC / "dry_laps.parquet")
    base = StrategyEngine.from_processed()
    pre = dry[dry["year"] < 2026]

    rows = []
    for (year, rnd), race in pre.groupby(["year", "round"]):
        year, rnd = int(year), int(rnd)
        track = str(race["event_name"].iloc[0])
        if track not in base.total_laps_by_track:
            continue
        temp, rained = weather(year, rnd)
        if temp is None or rained:
            continue
        field = int(Counter(race.groupby("driver", observed=True)["stint"].nunique() - 1)
                    .most_common(1)[0][0])
        loo = fit_linear_baseline(pre[~((pre["year"] == year) & (pre["round"] == rnd))])
        eng = replace(base, deg_model=loo)
        on = len(eng.recommend(track, season=year, n_runs=2000, track_temp=temp)["best"]["pit_laps"])
        off = len(replace(eng, thermal=ThermalPrior.disabled())
                  .recommend(track, season=year, n_runs=2000)["best"]["pit_laps"])
        rows.append({"year": year, "race": track, "temp": temp, "field": field,
                     "thermal_on": on, "thermal_off": off})
        print(f"{year} {track[:24]:<25}{temp:5.0f}C  field {field}  on {on}  off {off}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(Path(__file__).parent / "figures" / "thermal_out_of_sample.csv", index=False)
    print(f"\n{len(df)} dry races, 2023-25, leave-one-race-out\n")
    print(f"{'band':<11}{'n':>4}{'on: match':>11}{'on: over':>10}{'off: match':>12}{'off: over':>11}")
    for lo, hi, name in BANDS:
        b = df[(df.temp > lo) & (df.temp <= hi)]
        if b.empty:
            continue
        mo, oo = int((b.thermal_on == b.field).sum()), int((b.thermal_on > b.field).sum())
        mf, of = int((b.thermal_off == b.field).sum()), int((b.thermal_off > b.field).sum())
        print(f"{name:<11}{len(b):>4}{mo:>8}/{len(b):<2}{oo:>8}{mf:>9}/{len(b):<2}{of:>9}")
    for col in ("thermal_on", "thermal_off"):
        print(f"overall {col:<12} match {int((df[col] == df.field).sum())}/{len(df)}")


if __name__ == "__main__" and "--slopes" not in sys.argv:
    main()


def slope_test() -> None:
    """The prior's actual claim: hot tracks wear tyres faster, by `sensitivity`
    s/lap per C. Test it directly — regress the leave-one-out model's slope error
    on track temperature, over the same independent 2023-25 races. The fitted
    coefficient *is* the sensitivity the data supports."""
    import numpy as np

    from f1se.eda import fit_stint_slopes

    temps = pd.read_csv(Path(__file__).parent / "figures" / "thermal_out_of_sample.csv")
    dry = pd.read_parquet(PROC / "dry_laps.parquet")
    pre = dry[dry["year"] < 2026]
    rows = []
    for t in temps.itertuples():
        race = pre[(pre["year"] == t.year) & (pre["event_name"] == t.race)]
        loo = fit_linear_baseline(pre[~((pre["year"] == t.year) & (pre["event_name"] == t.race))])
        med = fit_stint_slopes(race.assign(race=t.race)).groupby("compound")["slope_s_per_lap"].median()
        for comp, actual in med.items():
            rows.append({"race": f"{t.year} {t.race}", "dt": t.temp - ThermalPrior().ref_temp,
                         "resid": actual - loo.slope(comp, t.race)})
    r = pd.DataFrame(rows)
    fit = lambda d: np.polyfit(d.dt, d.resid, 1)[0]  # noqa: E731
    races = r.race.unique()
    rng = np.random.default_rng(0)
    boot = [fit(r[r.race.isin(rng.choice(races, len(races)))]) for _ in range(2000)]
    lo, hi = np.percentile(boot, [2.5, 97.5])
    print(f"\nslope error vs track temp, {len(r)} race-compounds over {len(races)} races")
    print(f"  fitted sensitivity {fit(r):+.4f} s/lap per C   95% CI [{lo:+.4f}, {hi:+.4f}]")
    print(f"  the prior assumes  {ThermalPrior().sensitivity:+.4f}")


if __name__ == "__main__" and "--slopes" in sys.argv:
    slope_test()
