"""Phase 2 — fit and evaluate the linear degradation baseline.

Loads dry-only laps across several races, fits the per-(track, compound) linear
baseline, and evaluates it the honest way: GroupKFold by race (no leakage),
compared against a naive "predict the group mean / no degradation" comparator.
The baseline only earns its place if it beats that comparator on held-out races.

Also plots the fitted degradation lines over the data for one track, so you can
eyeball that the fit is sane.

Network on first run (cached after). Figure -> notebooks/figures/.

    .venv\\Scripts\\python.exe notebooks/phase2_degradation.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f1se.eda import RaceSpec, load_clean_races
from f1se.models.degradation import (
    cross_val_mae,
    fit_linear_baseline,
    predict_corrected_laptime,
)

FIG_DIR = Path(__file__).parent / "figures"
COMPOUND_COLORS = {"SOFT": "#e2231a", "MEDIUM": "#f6c700", "HARD": "#b0b0b0"}

RACES = [
    RaceSpec(2023, "Spain"),
    RaceSpec(2023, "Monaco"),
    RaceSpec(2023, "Italy"),
    RaceSpec(2023, "Hungary"),
    RaceSpec(2023, "Austria"),
    RaceSpec(2023, "Britain"),
]


def plot_fit_for_track(clean, model, track: str, out: Path) -> Path:
    sub = clean[clean["event_name"].str.contains(track, case=False, na=False)]
    if sub.empty:  # substring missed (e.g. "Spain" vs "Spanish Grand Prix")
        biggest = clean["event_name"].value_counts().idxmax()
        sub = clean[clean["event_name"] == biggest]
    track = sub["event_name"].iloc[0]
    fig, ax = plt.subplots(figsize=(9, 6))
    for comp, grp in sub.groupby("compound", observed=True):
        color = COMPOUND_COLORS.get(str(comp), "#444")
        ax.scatter(grp["tyre_life"], grp["lap_time_fuel_corr_s"], s=10, alpha=0.25, color=color)
        ages = np.linspace(grp["tyre_life"].min(), grp["tyre_life"].max(), 50)
        line = [predict_corrected_laptime(model, str(comp), a, track=sub["event_name"].iloc[0])
                for a in ages]
        ax.plot(ages, line, color=color, lw=2.5, label=str(comp))
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_title(f"Linear degradation baseline fit — {track}")
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    print("Loading dry races (cached after first call)...")
    clean = load_clean_races(RACES, dry_only=True)

    model = fit_linear_baseline(clean)
    print(f"\nFitted baseline: {model.meta['n_groups']} (track,compound) groups "
          f"on {model.meta['n_train_laps']} laps.")

    # Within-stint degradation rate (s/lap) per compound.
    print("\nCompound degradation rate (s/lap, within-stint fixed-effects fit):")
    for comp, slope in sorted(model.compound_slope.items()):
        print(f"  {comp:<8} {slope:+.4f}")

    print("\nLeakage-safe evaluation (GroupKFold by race; metric = pace-loss MAE):")
    res = cross_val_mae(clean, n_splits=5)
    print(f"  linear baseline MAE : {res['linear_mae']:.4f} s")
    print(f"  naive (no-deg)      : {res['naive_mae']:.4f} s")
    print(f"  improvement         : {res['improvement_pct']:.1f}%  over {res['n_folds']} folds")

    fig = plot_fit_for_track(clean, model, "Spain", FIG_DIR / "phase2_baseline_fit.png")
    print(f"\nFigure saved: {fig}")
    verdict = "BEATS" if res["linear_mae"] < res["naive_mae"] else "DOES NOT beat"
    print(f"\nVerdict: linear baseline {verdict} the naive comparator on held-out races.")


if __name__ == "__main__":
    main()
