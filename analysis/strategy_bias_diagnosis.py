"""Why does the engine two-stop 2026 races the field one-stops? (METHODOLOGY §19)

Every one of the seven 2026 stop-count misses in §18 is the same error: the
engine plans two stops where the field ran one. This script does not score
anything against the field. It asks the engine to explain *itself*: for each
race it takes the engine's own best one-stop and best two-stop plan (same
leave-one-race-out model as the backtest) and splits the expected-time gap
between them into the terms the simulator adds up:

    base     compound pace on a fresh tyre (the fitted intercepts)
    wear     linear degradation, slope x tyre age
    cliff    the hand-set cliff prior (models/cliff.py), not fitted
    pit      green-flag pit-lane loss, one per stop
    penalty  the hand-set per-stop track-position cost (models/overtaking.py)
    sc/vsc   everything else in the simulated mean: cheaper stops under
             neutralisation, plus Monte Carlo noise

A positive number is time that term charges the two-stop over the one-stop.
Whichever terms are reliably negative are what buy the second stop.

    .venv\\Scripts\\python.exe analysis/strategy_bias_diagnosis.py [--years 2026]

``--years 2023,2024,2025`` runs the same decomposition on old-car races, where
§17 found the engine's misses lean the other way.
"""
from __future__ import annotations

import argparse
import warnings
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT  # noqa: E402
from f1se.engine import StrategyEngine  # noqa: E402
from f1se.models.censoring import AvoidancePrior, apply_avoidance_adjustments  # noqa: E402
from f1se.models.cliff import CliffPrior  # noqa: E402
from f1se.models.degradation import fit_linear_baseline, predict_corrected_laptime  # noqa: E402
from f1se.models.era import fit_era_shrunk_degradation  # noqa: E402
from f1se.sim.simulate import stint_plan  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
OUT = PROJECT_ROOT / "analysis" / "figures" / "strategy_bias_diagnosis.csv"
TERMS = ["base", "wear", "cliff", "pit", "penalty", "sc_other"]


def loo_engine(base: StrategyEngine, dry: pd.DataFrame, year: int, rnd: int) -> StrategyEngine:
    """The engine with its degradation refit without race (year, rnd)."""
    dry_loo = dry[~((dry.year == year) & (dry["round"] == rnd))]
    if year >= 2026:
        loo = fit_era_shrunk_degradation(dry_loo, target_min_year=2026, recency_halflife=4.0)
        era = dry_loo[dry_loo["year"] >= 2026]
        loo = apply_avoidance_adjustments(loo, era, prior_model=base.deg_model)
        return replace(base, deg_model_2026=loo, stint_caps=AvoidancePrior().track_caps(era))
    return replace(base, deg_model=fit_linear_baseline(dry_loo[dry_loo.year < 2026]))


def terms(eng: StrategyEngine, track: str, season: int, plan: dict) -> dict:
    """Deterministic per-term totals for one plan."""
    from f1se.sim.simulate import Strategy
    model = eng._model_for(season)
    total = eng._total_laps(track)
    strat = Strategy(tuple(plan["compounds"]), tuple(plan["pit_laps"]))
    cliff = CliffPrior()
    base = wear = cl = 0.0
    for comp, age, _ in stint_plan(strat, total):
        base += predict_corrected_laptime(model, comp, 0, track=track)
        wear += model.slope(comp, track) * age
        cl += cliff.extra_loss(comp, age)
    n = strat.n_stops
    return {"base": base, "wear": wear, "cliff": cl,
            "pit": n * eng._pit_loss(track),
            "penalty": n * eng.overtaking.penalty_per_stop(track),
            "mean": plan["mean_s"]}


def best_by_stops(eng: StrategyEngine, track: str, season: int, n_runs: int) -> dict[int, dict]:
    """The engine's best plan at each stop count, each from its own search."""
    out = {}
    for k in (1, 2):
        try:
            rec = eng.recommend(track, season=season, n_runs=n_runs, max_stops=k, top_k=64)
        except ValueError:      # no plan at this stop count fits the stint limits
            continue
        plans = [p for p in rec["shortlist"] if p["n_stops"] == k]
        if plans:
            out[k] = plans[0]
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", default="2026")
    ap.add_argument("--runs", type=int, default=1500)
    a = ap.parse_args()
    years = [int(y) for y in a.years.split(",")]

    dry = pd.read_parquet(PROC / "dry_laps.parquet")
    base = StrategyEngine.from_processed()
    rows = []
    for year in years:
        for rnd in sorted(dry.loc[dry.year == year, "round"].unique()):
            race = dry[(dry.year == year) & (dry["round"] == rnd)]
            track = str(race["event_name"].iloc[0])
            if track not in base.total_laps_by_track:
                continue
            eng = loo_engine(base, dry, year, int(rnd))
            best = best_by_stops(eng, track, year, a.runs)
            if 1 not in best or 2 not in best:
                continue
            t1, t2 = (terms(eng, track, year, best[k]) for k in (1, 2))
            d = {k: t2[k] - t1[k] for k in ("base", "wear", "cliff", "pit", "penalty")}
            d["sc_other"] = (t2["mean"] - t1["mean"]) - (d["base"] + d["wear"] + d["cliff"] + d["pit"])
            gap = sum(d.values())          # >0: one-stop ranks better
            field = int(Counter(race.groupby("driver", observed=True)["stint"].nunique() - 1)
                        .most_common(1)[0][0])
            rows.append({"year": year, "round": int(rnd), "race": track, "field": field,
                         "engine": 2 if gap < 0 else 1, "gap": gap, **d,
                         "plan1": "-".join(c[0] for c in best[1]["compounds"]),
                         "plan2": "-".join(c[0] for c in best[2]["compounds"])})
            r = rows[-1]
            print(f"{year} {track[:22]:<23} fld {field} eng {r['engine']}  gap {gap:+6.1f}  "
                  + "  ".join(f"{t} {d[t]:+6.1f}" for t in TERMS)
                  + f"   {r['plan1']:<6}{r['plan2']}", flush=True)

    df = pd.DataFrame(rows)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, mode="a" if OUT.exists() and years != [2026] else "w",
              header=not (OUT.exists() and years != [2026]), index=False)
    print("\nmean two-stop minus one-stop, by term (s):")
    for grp, sub in [("all", df), ("engine 2-stops", df[df.engine == 2]),
                     ("engine 1-stops", df[df.engine == 1])]:
        if len(sub):
            print(f"  {grp:<16} n={len(sub):>2}  " +
                  "  ".join(f"{t} {sub[t].mean():+6.1f}" for t in TERMS))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    np.seterr(all="ignore")
    main()
