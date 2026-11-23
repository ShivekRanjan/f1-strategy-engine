r"""Does a weekend's practice predict its race's tyre wear? (METHODOLOGY §22)

The engine's blind spot is a track's first race of a season, where its wear
rate has to be borrowed: since §20, the era's median-stint rate for the
compound. Practice long runs (analysis/practice_longruns.py) come before the
race, so they can stand in for it without leakage. Practice reads high, though,
because teams push harder than in a race, so two numbers are needed: practice's
offset from race wear, and how much weight it gets against the fallback:

    predicted = (n * (practice - offset) + k * fallback) / (n + k)

where n is the practice long-run laps on that compound. Both are fitted on
2023-25 only. 2026 is then scored once with them, against the fallback alone,
on the same yardstick as the backtest: each race's median stint slope per
compound.

    .venv\Scripts\python.exe analysis/practice_wear_test.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT  # noqa: E402
from f1se.eda import fit_stint_slopes  # noqa: E402

PROC = PROJECT_ROOT / "data" / "processed"
COMPS = ("SOFT", "MEDIUM", "HARD")
K_GRID = (25, 50, 100, 150, 300, 600, 1200)


def table() -> pd.DataFrame:
    """One row per (race, compound): race wear, LOO era fallback, practice summary."""
    d = pd.read_parquet(PROC / "dry_laps.parquet").dropna(subset=["tyre_life", "lap_time_fuel_corr_s"])
    d = d[d.compound.isin(COMPS)]
    st = fit_stint_slopes(d.assign(race=d.event_name))
    st = st[st.slope_s_per_lap.abs() < 0.5]
    st["era"] = np.where(st.year >= 2026, "2026", "2023-25")
    fp = pd.read_parquet(PROC / "practice_longruns.parquet")
    fp = fp[fp.slope_s_per_lap.abs() < 0.5]
    rows = []
    for (y, r, c), g in st.groupby(["year", "round", "compound"]):
        era = g.era.iloc[0]
        others = st[(st.era == era) & (st.compound == c) & ~((st.year == y) & (st["round"] == r))]
        f = fp[(fp.year == y) & (fp["round"] == r) & (fp.compound == c)]
        rows.append({"year": y, "round": r, "compound": c, "era": era,
                     "actual": g.slope_s_per_lap.median(),
                     "fallback": others.slope_s_per_lap.median(),
                     "practice": f.slope_s_per_lap.median() if len(f) else np.nan,
                     "n": int(f.n_laps.sum())})
    return pd.DataFrame(rows)


def predict(t: pd.DataFrame, offset: float, k: float) -> pd.Series:
    has = t.practice.notna()
    blend = (t.n * (t.practice - offset) + k * t.fallback) / (t.n + k)
    return blend.where(has, t.fallback)


def main() -> None:
    t = table()
    dev, test = t[t.era == "2023-25"], t[t.era == "2026"]
    dev_fp = dev.dropna(subset=["practice"])
    offset = float((dev_fp.practice - dev_fp.actual).median())
    print(f"2023-25: {len(dev_fp)} (race, compound) pairs with practice; "
          f"practice - race offset (median) {offset:+.4f} s/lap, "
          f"corr {dev_fp[['practice', 'actual']].corr().iloc[0, 1]:.2f}")
    print("2023-25 MAE by k (lower k = trust practice more):")
    base = (dev.fallback - dev.actual).abs().mean()
    best_k, best = None, base
    for k in K_GRID:
        mae = (predict(dev, offset, k) - dev.actual).abs().mean()
        print(f"  k={k:<5} {mae:.4f}")
        if mae < best:
            best_k, best = k, mae
    print(f"  fallback alone {base:.4f}  -> chosen k = {best_k}")

    print("\n2026, scored once with the 2023-25 offset and k:")
    fb = (test.fallback - test.actual).abs().mean()
    if best_k is None:
        print(f"  practice never helped on 2023-25; nothing to test. fallback {fb:.4f}")
        return
    pr = (predict(test, offset, best_k) - test.actual).abs().mean()
    print(f"  fallback alone {fb:.4f}   practice-informed {pr:.4f}   "
          f"({'better' if pr < fb else 'worse'}; {test.practice.notna().sum()} of {len(test)} pairs have practice)")
    t.to_csv(PROJECT_ROOT / "analysis" / "figures" / "practice_wear_test.csv", index=False)


if __name__ == "__main__":
    main()
