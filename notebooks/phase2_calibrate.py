"""Phase 2 (correctness) — what does the data imply about the fuel coefficient?

Backs an *effective* fuel coefficient out of the net race-lap pace trend and
reports the *assumption-free* degradation slopes. Compares both to the physics
value (0.03 s/kg) and to the 0.03-based within-stint degradation.

    .venv\\Scripts\\python.exe notebooks/phase2_calibrate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.calibrate import calibrate_effective_fuel
from f1se.models.degradation import fit_linear_baseline

DATA = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"
FIG_DIR = Path(__file__).parent / "figures"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}\nRun:  python -m f1se.data.ingest")
    laps = pd.read_parquet(DATA)
    print(f"Loaded {len(laps):,} dry laps over {laps.groupby(['year','round']).ngroups} races.")

    cal = calibrate_effective_fuel(laps)
    betas = np.array(list(cal.implied_beta_by_race.values()))
    print(f"\nEffective fuel coefficient implied by the net race-lap trend:")
    print(f"  physics assumption : {cal.physics_beta:.3f} s/kg")
    print(f"  implied  median    : {cal.median_beta:.3f} s/kg")
    print(f"  implied  mean+/-sd : {cal.mean_beta:.3f} +/- {cal.std_beta:.3f}")
    print(f"  range (10-90 pct)  : {np.percentile(betas,10):.3f} .. {np.percentile(betas,90):.3f}")
    print(f"  (CAVEAT: bundles fuel + track-evo + late-race management; effective, not pure)")

    # Assumption-free degradation (raw joint fit) vs the 0.03-based within-stint.
    naive = fit_linear_baseline(laps)
    print("\nDegradation slope (s/lap): assumption-free (joint) vs 0.03-based (within-stint):")
    print(f"  {'compound':<8} {'assump-free':>12} {'0.03-based':>12}")
    for c in ["SOFT", "MEDIUM", "HARD"]:
        af = cal.deg_slope.get(c, float("nan"))
        wb = naive.compound_slope.get(c, float("nan"))
        print(f"  {c:<8} {af:>12.4f} {wb:>12.4f}")

    # Histogram of per-race implied beta.
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(betas, bins=15, color="#1f6feb", alpha=0.8, edgecolor="white")
    ax.axvline(cal.physics_beta, color="red", ls="--", lw=2, label="physics 0.03")
    ax.axvline(cal.median_beta, color="black", lw=2, label=f"implied median {cal.median_beta:.3f}")
    ax.set_xlabel("Implied effective fuel coefficient (s/kg)")
    ax.set_ylabel("races")
    ax.set_title("Per-race effective fuel coefficient vs the physics value")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phase2_fuel_calibration.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
