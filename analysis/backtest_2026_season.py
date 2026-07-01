"""Season-wide 2026 backtest — verify every 2026 race against ground truth.

For each 2026 race we hold, checks the model on three fronts the user asked for:
strategy, tyre selection, and pace (degradation). Crucially it's **leave-one-race-
out**: the degradation model that predicts race R is refit *without* R's laps, so
each check is genuinely out-of-sample even though R is now in the dataset.

    .venv\\Scripts\\python.exe analysis/backtest_2026_season.py
"""
from __future__ import annotations

import warnings
from collections import Counter
from dataclasses import replace

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT
from f1se.eda import fit_stint_slopes
from f1se.engine import StrategyEngine
from f1se.models.era import fit_era_shrunk_degradation

PROC = PROJECT_ROOT / "data" / "processed"
COMPS = ["SOFT", "MEDIUM", "HARD"]


def winner_strategy(race_laps: pd.DataFrame, driver: str):
    """Compound sequence + approx pit laps for one driver, from cleaned laps."""
    d = race_laps[race_laps["driver"] == driver].sort_values("lap_number")
    seq, pits = [], []
    for _, g in d.groupby("stint"):
        seq.append(str(g["compound"].iloc[0]))
        pits.append(int(g["lap_number"].max()))
    return seq, pits[:-1]  # drop the final stint's last lap (the flag, not a pit)


def main() -> None:
    dry = pd.read_parquet(PROC / "dry_laps.parquet")
    results = pd.read_parquet(PROC / "results.parquet")
    base = StrategyEngine.from_processed()

    r26 = dry[dry["year"] == 2026]
    rounds = sorted(r26["round"].unique())
    print(f"2026 races in dataset: {len(rounds)} (rounds {rounds[0]}–{rounds[-1]})\n")

    hdr = (f"{'race':<22}{'eng':>5}{'win':>5}{'fld':>5}  "
           f"{'engine pick':<14}{'winner':<12}{'vsWin':>7}{'vsFld':>7}{'degMAE':>8}")
    print(hdr)
    print("-" * len(hdr))

    stops_hit = tyre_hit = field_hit = deg_maes = 0
    n = 0
    for rnd in rounds:
        race = r26[r26["round"] == rnd]
        track = str(race["event_name"].iloc[0])
        win = results[(results.year == 2026) & (results["round"] == rnd) & (results.position == 1)]
        if win.empty or track not in base.total_laps_by_track:
            continue
        winner = str(win["driver"].iloc[0])
        wseq, wpits = winner_strategy(race, winner)
        if not wseq:
            continue

        # Leave-one-race-out degradation model, then recommend for this track.
        loo = fit_era_shrunk_degradation(
            dry[~((dry.year == 2026) & (dry["round"] == rnd))],
            target_min_year=2026, recency_halflife=4.0)
        eng = replace(base, deg_model_2026=loo)
        rec = eng.recommend(track, season=2026, n_runs=3000)["best"]

        # Degradation: actual (this race) vs leave-one-out model, per compound.
        slopes = fit_stint_slopes(race.assign(race=track))
        med = slopes.groupby("compound")["slope_s_per_lap"].median()
        errs = [abs(med[c] - loo.slope(c, track)) for c in COMPS if c in med.index]
        deg_mae = float(np.mean(errs)) if errs else float("nan")

        # Field-dominant stop count (the strategy most of the field actually ran).
        field_stops = (race.groupby("driver", observed=True)["stint"].nunique() - 1)
        field_mode = int(Counter(field_stops).most_common(1)[0][0])

        eng_stops = len(rec["pit_laps"])
        stops_match = eng_stops == len(wpits)
        tyre_match = Counter(rec["compounds"]) == Counter(wseq)
        field_match = eng_stops == field_mode
        stops_hit += stops_match
        tyre_hit += tyre_match
        field_hit += field_match
        deg_maes += deg_mae
        n += 1

        print(f"{track[:21]:<22}{eng_stops:>5}{len(wpits):>5}{field_mode:>5}  "
              f"{'-'.join(c[0] for c in rec['compounds']):<14}{'-'.join(c[0] for c in wseq):<12}"
              f"{'OK' if stops_match else 'x':>7}{'OK' if field_match else 'x':>7}{deg_mae:>8.4f}")

    print("-" * len(hdr))
    print(f"\nAggregate over {n} races  (eng/win/fld = engine/winner/field-dominant stop count):")
    print(f"  stop-count vs winner        : {stops_hit}/{n} ({100*stops_hit/n:.0f}%)")
    print(f"  stop-count vs field-dominant : {field_hit}/{n} ({100*field_hit/n:.0f}%)")
    print(f"  exact tyre set vs winner     : {tyre_hit}/{n} ({100*tyre_hit/n:.0f}%)")
    print(f"  degradation MAE (leave-one-out): {deg_maes/n:.4f} s/lap")
    print("\nRead honestly: the engine computes the free-air TIME-optimal plan; it doesn't model\n"
          "track position / overtaking cost, so at durable-tyre circuits where the field 1-stops to\n"
          "hold position, it tends to recommend an extra stop. It matches best where the field truly\n"
          "2-stopped. Degradation itself is close (leave-one-out).")


if __name__ == "__main__":
    main()
