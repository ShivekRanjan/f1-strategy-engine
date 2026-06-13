"""Phase 2 (correctness) — forward-in-time validation on the full dataset.

Loads the bulk dry dataset (run ``python -m f1se.data.ingest`` first), then:
  1. Forward holdout: train on <=2023, test on 2024 (the realistic
     'predict the future' split). Same circuit in a later year is a SEEN
     (track, compound) group, so per-track degradation transfers across seasons.
  2. Reports per-(track, compound) degradation slopes and how much of the 2024
     test set is covered by track-specific vs compound-fallback slopes.
  3. A fuel-assumption sensitivity band on the headline compound slopes.

    .venv\\Scripts\\python.exe notebooks/phase2_forward.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.degradation import (
    AGE_COL,
    TARGET_COL,
    fit_linear_baseline,
    linear_shape,
    naive_pace_loss_mae,
    pace_loss_mae,
    shape_mae,
)
from f1se.validation import assert_no_race_leakage, forward_year_holdout, group_kfold_races

DATA = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"


def _refit_corrected(laps: pd.DataFrame, sec_per_kg: float) -> pd.DataFrame:
    """Recompute fuel-corrected pace at a different sec_per_kg (for sensitivity)."""
    out = laps.copy()
    out[TARGET_COL] = out["lap_time_s"] - sec_per_kg * out["fuel_mass_kg"]
    return out


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}\nRun:  python -m f1se.data.ingest")

    laps = pd.read_parquet(DATA)
    yrs = sorted(laps["year"].unique())
    n_races = laps.groupby(["year", "round"]).ngroups
    print(f"Loaded {len(laps):,} dry laps, {n_races} races, years {yrs}.")

    # --- 1. Forward holdout: train <=2023, test 2024 --------------------------
    train, test = forward_year_holdout(laps, test_year=2024)
    assert_no_race_leakage(train, test)
    print(f"\nForward holdout: train {len(train):,} laps (<=2023), "
          f"test {len(test):,} laps (2024).")

    model = fit_linear_baseline(train)  # per-(event_name, compound)
    lin = pace_loss_mae(model, test)
    naive = naive_pace_loss_mae(test)
    print(f"  linear per-(track,compound) MAE : {lin:.4f} s")
    print(f"  naive (no-deg)                  : {naive:.4f} s")
    print(f"  improvement                     : {100*(naive-lin)/naive:+.1f}%")

    # Coverage: which 2024 (track,compound) groups were seen in <=2023 training?
    train_keys = set(model.slopes.keys())
    test_keys = set(map(tuple, test[["event_name", "compound"]].drop_duplicates().to_numpy()))
    seen = test_keys & train_keys
    print(f"  test (track,compound) groups seen in train: {len(seen)}/{len(test_keys)} "
          f"(rest use compound-level fallback)")

    # --- 2. GroupKFold over all years, for comparison -------------------------
    gk_lin, gk_naive = [], []
    for tr, te in group_kfold_races(laps, n_splits=5):
        m = fit_linear_baseline(laps.iloc[tr])
        gk_lin.append(pace_loss_mae(m, laps.iloc[te]))
        gk_naive.append(naive_pace_loss_mae(laps.iloc[te]))
    print(f"\nGroupKFold (all years, 5 folds): linear {np.mean(gk_lin):.4f}s "
          f"vs naive {np.mean(gk_naive):.4f}s  ({100*(np.mean(gk_naive)-np.mean(gk_lin))/np.mean(gk_naive):+.1f}%)")

    # --- 3. Headline compound slopes + fuel-assumption sensitivity band -------
    print("\nCompound degradation slope (s/lap) across fuel assumption sec_per_kg:")
    betas = [0.02, 0.03, 0.04]
    rows = {}
    for b in betas:
        m = fit_linear_baseline(_refit_corrected(laps, b))
        rows[b] = m.compound_slope
    comps = ["SOFT", "MEDIUM", "HARD"]
    header = "  " + "compound".ljust(9) + "".join(f"b={b:<7}" for b in betas)
    print(header)
    for c in comps:
        line = "  " + c.ljust(9) + "".join(f"{rows[b].get(c, float('nan')):<9.4f}" for b in betas)
        print(line)
    print("  (spread across b shows how the fuel assumption moves the estimate)")


if __name__ == "__main__":
    main()
