"""Phase 1 EDA — generalization, per-stint compound ordering, fuel sensitivity.

Runs the three checks that decide whether the cleaning is trustworthy enough to
model on:

  1. Generalization  : do degradation curves hold across track types
                       (Barcelona high-deg, Monaco/Monza low-deg)?
  2. Per-stint view  : fit a slope per stint to expose compound ordering that
                       pooling all laps hides.
  3. Fuel sensitivity: how much does the measured degradation slope move as the
                       fuel assumption (sec_per_kg) varies — and does it match
                       the analytical prediction Δslope = Δβ · fuel_per_lap?

Network on first run (cached after). Figures -> notebooks/figures/.

    .venv\\Scripts\\python.exe notebooks/phase1_eda.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.data.clean import FuelModel
from f1se.eda import (
    RaceSpec,
    compound_degradation_summary,
    fit_stint_slopes,
    fuel_sensitivity,
    load_clean_races,
)

FIG_DIR = Path(__file__).parent / "figures"
COMPOUND_COLORS = {
    "SOFT": "#e2231a",
    "MEDIUM": "#f6c700",
    "HARD": "#b0b0b0",
    "INTERMEDIATE": "#3aa655",
    "WET": "#1f6feb",
}
COMPOUND_ORDER = ["SOFT", "MEDIUM", "HARD", "INTERMEDIATE", "WET"]

# A deliberate spread of track types to test generalization.
RACES = [
    RaceSpec(2023, "Spain"),   # Barcelona - abrasive, high degradation
    RaceSpec(2023, "Monaco"),  # street - very low degradation, tyre management
    RaceSpec(2023, "Italy"),   # Monza - low downforce, low degradation
]


def _present_compounds(slopes: pd.DataFrame) -> list[str]:
    have = set(slopes["compound"].unique())
    return [c for c in COMPOUND_ORDER if c in have]


def plot_stint_slopes(slopes: pd.DataFrame, out: Path) -> Path:
    """Strip plot of per-stint degradation slopes by compound, faceted by race."""
    races = sorted(slopes["race"].unique())
    fig, axes = plt.subplots(1, len(races), figsize=(5 * len(races), 5), sharey=True)
    if len(races) == 1:
        axes = [axes]

    compounds = _present_compounds(slopes)
    for ax, race in zip(axes, races):
        sub = slopes[slopes["race"] == race]
        for i, comp in enumerate(compounds):
            pts = sub[sub["compound"] == comp]["slope_s_per_lap"]
            jitter = np.random.default_rng(0).normal(0, 0.06, len(pts))
            ax.scatter(np.full(len(pts), i) + jitter, pts, s=28, alpha=0.7,
                       color=COMPOUND_COLORS.get(comp, "#444"))
            if len(pts):
                ax.hlines(pts.median(), i - 0.25, i + 0.25, color="black", lw=2)
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xticks(range(len(compounds)))
        ax.set_xticklabels(compounds, rotation=20)
        ax.set_title(race)
        ax.grid(True, axis="y", alpha=0.3)
    axes[0].set_ylabel("Degradation slope (s / lap)\nhigher = degrades faster")
    fig.suptitle("Per-stint degradation slope by compound (black bar = median)")
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def plot_fuel_sensitivity(sens: pd.DataFrame, fuel_per_lap: float, out: Path) -> Path:
    """Median degradation slope vs sec_per_kg, per compound, with default marked."""
    fig, ax = plt.subplots(figsize=(9, 6))
    for comp in _present_compounds(sens.rename(columns={"compound": "compound"})):
        sub = sens[sens["compound"] == comp].sort_values("sec_per_kg")
        ax.plot(sub["sec_per_kg"], sub["median_slope"], marker="o",
                color=COMPOUND_COLORS.get(comp, "#444"), label=comp)
    ax.axvline(0.03, color="grey", ls="--", lw=1, label="default 0.03")
    ax.set_xlabel("Fuel assumption  sec_per_kg")
    ax.set_ylabel("Median degradation slope (s / lap)")
    ax.set_title(
        "Fuel sensitivity of measured degradation\n"
        f"analytical: d(slope) = d(beta) * fuel_per_lap ~ d(beta) * {fuel_per_lap:.2f}"
    )
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    print("Loading races (cached after first call)...")
    clean = load_clean_races(RACES)

    # --- 2. Per-stint slopes + compound ordering -------------------------------
    slopes = fit_stint_slopes(clean)
    print(f"\nFitted {len(slopes)} stint slopes (>=6 laps each).")

    print("\nCompound degradation ordering (overall, fastest-degrading first):")
    print(compound_degradation_summary(slopes).round(4).to_string())

    print("\nPer track (median slope s/lap, generalization check):")
    pivot = (
        slopes.pivot_table(index="compound", columns="race",
                           values="slope_s_per_lap", aggfunc="median", observed=True)
        .reindex(_present_compounds(slopes))
    )
    print(pivot.round(4).to_string())

    plot_stint_slopes(slopes, FIG_DIR / "phase1_stint_slopes.png")

    # --- 3. Fuel sensitivity ---------------------------------------------------
    betas = [0.00, 0.02, 0.03, 0.04, 0.06]
    sens = fuel_sensitivity(clean, betas)

    # Analytical prediction: fuel_per_lap = start_fuel / total_laps (per race).
    fuel = FuelModel()
    total_laps = clean.groupby(["year", "round"])["lap_number"].max().mean()
    fuel_per_lap = fuel.start_fuel_kg / total_laps

    print(f"\nFuel sensitivity (median slope s/lap by sec_per_kg):")
    print(sens.pivot_table(index="compound", columns="sec_per_kg",
                           values="median_slope", observed=True).round(4).to_string())
    print(f"\nAnalytical: each +0.01 in sec_per_kg should raise every slope by "
          f"~{0.01 * fuel_per_lap:.4f} s/lap (fuel_per_lap ~ {fuel_per_lap:.2f} kg).")

    plot_fuel_sensitivity(sens, fuel_per_lap, FIG_DIR / "phase1_fuel_sensitivity.png")

    print(f"\nFigures saved to {FIG_DIR}")


if __name__ == "__main__":
    main()
