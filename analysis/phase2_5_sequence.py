"""Phase 2.5 — sequence (LSTM) lap-time model, leakage-safe head-to-head.

The deep-learning chapter. Forecasts the next lap's fuel-corrected time from the
recent sequence of laps in a stint, and proves it earns its place against two
dumb baselines on a forward-in-time split:

  * persistence    — next lap == last lap (delta 0)
  * rolling-slope  — extrapolate the within-window OLS trend one lap
  * LSTM           — predicts the lap-to-lap delta from the sequence

Train on year <= 2024, test on 2025 (2026 excluded — regulation reset). Prints
MAE in seconds on the absolute next-lap time, and saves a figure: the MAE
head-to-head plus the model tracking one real 2025 stint lap by lap.

    .venv\\Scripts\\python.exe analysis/phase2_5_sequence.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.lap_time import (
    build_sequence_windows,
    fit_sequence_model,
    train_and_export,
)

DATA = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"
ARTIFACT = PROJECT_ROOT / "data" / "processed" / "lstm_nextlap.npz"
FIG_DIR = Path(__file__).parent / "figures"
WINDOW = 5
SEED = 0


def _mae(p, t) -> float:
    return float(np.mean(np.abs(np.asarray(p) - np.asarray(t))))


def _race_grouped_val_split(w, val_frac=0.15, seed=SEED):
    from f1se.models.lap_time import _subset

    rng = np.random.default_rng(seed)
    races = np.unique(w.race_ids)
    val = set(rng.choice(races, size=max(1, int(len(races) * val_frac)), replace=False))
    is_val = np.array([r in val for r in w.race_ids])
    return _subset(w, ~is_val), _subset(w, is_val)


def _example_stint_panel(ax, laps_2025, model) -> None:
    """Overlay actual vs one-step LSTM/persistence predictions on a long 2025 stint."""
    stints = laps_2025.groupby(["year", "round", "driver", "stint"], observed=True)
    # Longest stint makes the cleanest illustration.
    key = max(stints.groups, key=lambda k: len(stints.get_group(k)))
    stint = stints.get_group(key).sort_values("lap_number")
    w = build_sequence_windows(stint, window=WINDOW)
    age_next = stint["tyre_life"].to_numpy(float)[WINDOW:]
    lstm = model.predict_next(w)

    ax.plot(stint["tyre_life"], stint["lap_time_fuel_corr_s"], "o-", color="#222",
            ms=4, lw=1.4, label="actual")
    ax.plot(age_next, w.y_curr, "s", color="#b0b0b0", ms=5, label="persistence (last lap)")
    ax.plot(age_next, lstm, "^", color="#1f6feb", ms=6, label="LSTM one-step")
    comp = str(stint["compound"].iloc[0])
    drv = str(stint["driver"].iloc[0])
    ax.set_title(f"One-step forecast on a real 2025 stint\n({drv}, {comp}, {key[2]} R{key[1]})", fontsize=10)
    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}\nRun:  python -m f1se.data.ingest")
    laps = pd.read_parquet(DATA)
    train = laps[laps["year"] <= 2024]
    test = laps[laps["year"] == 2025]
    print(f"Loaded {len(laps):,} dry laps. Train <=2024: {len(train):,}  Test 2025: {len(test):,}")

    train_w = build_sequence_windows(train, window=WINDOW)
    test_w = build_sequence_windows(test, window=WINDOW)
    sub_tr, sub_val = _race_grouped_val_split(train_w)

    print("Training LSTM (CPU)...")
    model = fit_sequence_model(sub_tr, val_windows=sub_val, epochs=40, hidden=32, seed=SEED)

    truth = test_w.y_next
    maes = {
        "persistence": _mae(test_w.y_curr, truth),
        "rolling-slope": _mae(test_w.slope_next, truth),
        "LSTM": _mae(model.predict_next(test_w), truth),
    }
    print("\nOne-step-ahead next-lap MAE (test = 2025 season):")
    for k, v in maes.items():
        print(f"  {k:14s} {v:.4f} s")
    imp = 100.0 * (maes["persistence"] - maes["LSTM"]) / maes["persistence"]
    print(f"  LSTM vs persistence  : {imp:+.1f}%")
    print(f"  LSTM vs rolling-slope: {100.0 * (maes['rolling-slope'] - maes['LSTM']) / maes['rolling-slope']:+.1f}%")
    print(f"  (n_test={len(truth):,}, epochs_run={model.meta['epochs_run']})")

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    colors = ["#b0b0b0", "#e2231a", "#1f6feb"]
    bars = ax1.bar(list(maes), list(maes.values()), color=colors, edgecolor="white")
    ax1.bar_label(bars, fmt="%.3f s", padding=3, fontsize=9)
    ax1.set_ylabel("Next-lap MAE (s) — lower is better")
    ax1.set_title("Forecasting the next lap, 2025 held out\n(LSTM trained on ≤2024)", fontsize=10)
    ax1.set_ylim(0, max(maes.values()) * 1.25)
    ax1.grid(True, axis="y", alpha=0.3)
    _example_stint_panel(ax2, test, model)

    verdict = ("LSTM beats both baselines" if maes["LSTM"] < min(maes["persistence"], maes["rolling-slope"])
               else "LSTM does NOT beat the baselines — the simpler predictor wins")
    fig.suptitle(f"Phase 2.5 — sequence lap-time model: {verdict}", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phase2_5_sequence.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved: {out}")

    if maes["LSTM"] < maes["persistence"]:
        print("\nVerdict: the sequence model adds value - it forecasts lap-to-lap pace "
              "changes better than assuming no change. A documented DL win.")
    else:
        print("\nVerdict: the sequence model does NOT beat persistence here — within-stint "
              "lap times are near-random-walk; the simpler predictor wins (Occam). Honest either way.")

    # Export the deployed nowcaster: trained on ALL seasons (the delta target
    # transfers across the 2026 reset), serialised torch-free for the app.
    print("\nTraining deploy model on all seasons and exporting...")
    train_and_export(laps, ARTIFACT, epochs=40, hidden=32, seed=SEED)
    print(f"Artifact saved: {ARTIFACT} ({ARTIFACT.stat().st_size / 1024:.1f} KB) "
          "- the app loads this via NumpyLapForecaster (no torch needed).")


if __name__ == "__main__":
    main()
