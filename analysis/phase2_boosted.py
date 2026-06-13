"""Phase 2 (second half) — boosted vs linear degradation, leakage-safe head-to-head.

Loads the dry races, runs GroupKFold-by-race comparison of the linear baseline,
the XGBoost model, and the no-degradation comparator (identical target/metric/
folds), then plots the learnt degradation curve of each model for one compound so
any nonlinearity the boosted model finds is visible.

    .venv\\Scripts\\python.exe notebooks/phase2_boosted.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.eda import RaceSpec, load_clean_races
from f1se.models.degradation import fit_linear_baseline
from f1se.models.degradation_boosted import (
    cross_val_compare,
    fit_boosted,
    harmonise_categories,
    predict_shape,
)

FIG_DIR = Path(__file__).parent / "figures"
RACES = [
    RaceSpec(2023, "Spain"), RaceSpec(2023, "Monaco"), RaceSpec(2023, "Italy"),
    RaceSpec(2023, "Hungary"), RaceSpec(2023, "Austria"), RaceSpec(2023, "Britain"),
]
COMPOUND_COLORS = {"SOFT": "#e2231a", "MEDIUM": "#f6c700", "HARD": "#b0b0b0"}


def plot_learnt_curves(laps, lm, bm, track: str, out: Path) -> Path:
    """Pace-loss vs age curves (normalised to 0 at age 1) for linear vs boosted."""
    sub = laps[laps["event_name"] == track]
    ages = np.arange(1, int(sub["tyre_life"].max()) + 1)
    fig, ax = plt.subplots(figsize=(9, 6))
    for comp in [c for c in COMPOUND_COLORS if c in set(sub["compound"])]:
        grid = pd.DataFrame({
            "year": 2023, "round": sub["round"].iloc[0], "driver": "X", "stint": 1,
            "event_name": pd.Categorical([track] * len(ages),
                                         categories=laps["event_name"].cat.categories),
            "compound": pd.Categorical([comp] * len(ages),
                                       categories=laps["compound"].cat.categories),
            "tyre_life": ages.astype(float),
        })
        lin = lm.slope(comp, track) * ages
        boost = predict_shape(bm, grid)
        lin -= lin[0]
        boost = boost - boost[0]
        color = COMPOUND_COLORS[comp]
        ax.plot(ages, lin, color=color, ls="--", lw=1.8, label=f"{comp} linear")
        ax.plot(ages, boost, color=color, lw=2.5, label=f"{comp} boosted")
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Predicted pace loss vs fresh (s)")
    ax.set_title(f"Learnt degradation curves — {track}\n(dashed=linear, solid=boosted)")
    ax.legend(fontsize=8, ncol=3)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    print("Loading dry races (cached after first call)...")
    clean = load_clean_races(RACES, dry_only=True)

    print("\nLeakage-safe head-to-head (GroupKFold by race; pace-loss MAE):")
    res = cross_val_compare(clean, n_splits=5)
    print(f"  naive (no-deg)   : {res['naive_mae']:.4f} s")
    print(f"  linear baseline  : {res['linear_mae']:.4f} s")
    print(f"  XGBoost          : {res['boosted_mae']:.4f} s")
    print(f"  boosted vs linear: {res['boosted_vs_linear_pct']:+.1f}%")
    print(f"  boosted vs naive : {res['boosted_vs_naive_pct']:+.1f}%   ({res['n_folds']} folds)")

    # Fit both on all data for the curve plot.
    cat = harmonise_categories(clean)
    lm = fit_linear_baseline(cat)
    bm = fit_boosted(cat)
    track = cat["event_name"].value_counts().idxmax()
    fig = plot_learnt_curves(cat, lm, bm, track, FIG_DIR / "phase2_boosted_curves.png")
    print(f"\nFigure saved: {fig}")

    if res["boosted_mae"] < res["linear_mae"]:
        print("\nVerdict: XGBoost beats the linear baseline on held-out races.")
    else:
        print("\nVerdict: XGBoost does NOT beat linear -> degradation is ~linear here; "
              "the simpler model wins (Occam). A defensible result either way.")


if __name__ == "__main__":
    main()
