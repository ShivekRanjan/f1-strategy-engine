"""Phase 2 (correctness) — does track-evolution correction change degradation?

Fits the joint (base + evolution + degradation) model on the dry races and
compares the de-biased per-compound degradation slopes against the uncorrected
within-stint baseline. Also reports the estimated per-circuit track-evolution
slope so we can sanity-check its sign and size (should be improving = negative,
on the order of tenths of a second across a race).

    .venv\\Scripts\\python.exe notebooks/phase2_evolution.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f1se.eda import RaceSpec, load_clean_races
from f1se.models.degradation import fit_linear_baseline
from f1se.models.degradation_evo import fit_evolution_model

FIG_DIR = Path(__file__).parent / "figures"
RACES = [
    RaceSpec(2023, "Spain"), RaceSpec(2023, "Monaco"), RaceSpec(2023, "Italy"),
    RaceSpec(2023, "Hungary"), RaceSpec(2023, "Austria"), RaceSpec(2023, "Britain"),
]
COMPOUND_COLORS = {"SOFT": "#e2231a", "MEDIUM": "#f6c700", "HARD": "#b0b0b0"}


def main() -> None:
    print("Loading dry races (cached after first call)...")
    clean = load_clean_races(RACES, dry_only=True)

    evo = fit_evolution_model(clean)
    naive = fit_linear_baseline(clean)  # within-stint, uncorrected
    print(f"\nJoint model R^2 = {evo.r2:.3f} on {evo.meta['n_laps']} laps "
          f"({evo.meta['n_cars']} car-races, {evo.meta['n_races']} races).")

    # Track evolution per race: slope (s/lap) and total over a ~full race length.
    print("\nEstimated track evolution (negative = track improving):")
    laps_per_race = clean.groupby(["year", "round"])["lap_number"].max()
    for rid, slope in sorted(evo.evo_slope.items()):
        yr, rnd = rid.split("_")
        ev = clean[(clean["year"] == int(yr)) & (clean["round"] == int(rnd))]
        name = ev["event_name"].iloc[0] if len(ev) else rid
        nlaps = int(laps_per_race.get((int(yr), int(rnd)), 0))
        print(f"  {name:<26} {slope:+.4f} s/lap   (~{slope*nlaps:+.2f} s over {nlaps} laps)")

    # Degradation: corrected vs uncorrected.
    print("\nDegradation slope (s/lap) — uncorrected (within-stint) vs evolution-corrected:")
    print(f"  {'compound':<8} {'uncorrected':>12} {'corrected':>12} {'change':>10}")
    comps = ["SOFT", "MEDIUM", "HARD"]
    for c in comps:
        unc = naive.compound_slope.get(c, float("nan"))
        cor = evo.deg_slope.get(c, float("nan"))
        print(f"  {c:<8} {unc:>12.4f} {cor:>12.4f} {cor - unc:>+10.4f}")

    # Bar comparison.
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(comps))
    unc = [naive.compound_slope.get(c, np.nan) for c in comps]
    cor = [evo.deg_slope.get(c, np.nan) for c in comps]
    ax.bar(x - 0.2, unc, 0.4, label="uncorrected (within-stint)", color="#bbbbbb")
    ax.bar(x + 0.2, cor, 0.4, label="evolution-corrected", color="#1f6feb")
    ax.set_xticks(x); ax.set_xticklabels(comps)
    ax.set_ylabel("Degradation slope (s/lap)")
    ax.set_title("Tyre degradation: correcting for track evolution")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phase2_evolution_correction.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
