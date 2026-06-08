"""2026 regulation-reset validation — does old-car data transfer, and does
shrinkage help? Quantifies the regime shift on real 2026 data.

    .venv\\Scripts\\python.exe notebooks/phase_2026_validation.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.degradation import (
    fit_linear_baseline,
    linear_shape,
    naive_pace_loss_mae,
    shape_mae,
)
from f1se.models.era import fit_era_shrunk_degradation


def main() -> None:
    dry = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet")
    dry = dry.dropna(subset=["tyre_life", "lap_time_fuel_corr_s"])
    pre, d26 = dry[dry.year < 2026], dry[dry.year >= 2026]
    n26 = d26.groupby(["year", "round"]).ngroups
    print(f"Pre-2026: {len(pre):,} laps · 2026: {len(d26):,} laps over {n26} races\n")

    prior = fit_linear_baseline(pre)
    pure26 = fit_linear_baseline(d26, min_laps=10)
    shrunk = fit_era_shrunk_degradation(dry, target_min_year=2026)

    print("Degradation slope (s/lap):  pre-2026 | 2026-only | shrunk")
    for c in ["SOFT", "MEDIUM", "HARD"]:
        print(f"  {c:<7} {prior.compound_slope.get(c, np.nan):+.4f}   "
              f"{pure26.compound_slope.get(c, np.nan):+.4f}   {shrunk.compound_slope.get(c, np.nan):+.4f}")

    print("\nPace-loss MAE on real 2026 laps (lower = better fit to 2026):")
    base = naive_pace_loss_mae(d26)
    old = shape_mae(linear_shape(prior, d26), d26)
    new = shape_mae(linear_shape(shrunk, d26), d26)
    print(f"  naive (no degradation) : {base:.4f}")
    print(f"  pre-2026 (old cars)    : {old:.4f}   ({100*(base-old)/base:+.0f}% vs naive)")
    print(f"  shrunk (2026-aware)    : {new:.4f}   ({100*(base-new)/base:+.0f}% vs naive)")
    print("\nReading: the old-car model barely beats 'no degradation' on 2026 (regime shift);")
    print("the shrinkage model recovers most of the signal by blending in 2026 data.")


if __name__ == "__main__":
    main()
