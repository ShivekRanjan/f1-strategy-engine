"""Phase 4 — optimise pit strategy for a real race, with uncertainty.

Fits the degradation model, builds the per-lap pace function, then searches the
whole strategy space (stops x pit laps x compound sequence) against a stochastic
safety car. Reports the recommended plan, a ranked shortlist with paired
win-probabilities, and how the pick changes under a risk-averse objective.

    .venv\\Scripts\\python.exe notebooks/phase4_optimize.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.degradation import fit_linear_baseline
from f1se.sim.optimize import recommend_strategy
from f1se.sim.safety_car import SafetyCarModel
from f1se.sim.simulate import Strategy, pace_fn_from_model

DATA = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"
FIG_DIR = Path(__file__).parent / "figures"
TRACK = "Spanish Grand Prix"


def _lbl(compounds, pit_laps) -> str:
    return "->".join(c[0] for c in compounds) + f" @{','.join(map(str, pit_laps))}"


def _print_shortlist(res) -> None:
    print(f"{'#':>2} {'plan':<20}{'mean':>9}{'p50':>9}{'p90':>9}{'P(beat best)':>14}")
    for r in res.shortlist:
        print(f"{r['rank']:>2} {_lbl(r['compounds'], r['pit_laps']):<20}"
              f"{r['mean_s']:>9.1f}{r['p50_s']:>9.1f}{r['p90_s']:>9.1f}"
              f"{r['win_prob_vs_best']*100:>13.0f}%")


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}\nRun:  python -m f1se.data.ingest")
    laps = pd.read_parquet(DATA)
    model = fit_linear_baseline(laps)

    ev = laps[laps["event_name"] == TRACK]
    total_laps = int(ev["lap_number"].max())
    pace_fn = pace_fn_from_model(model, TRACK, total_laps)
    sc_model = SafetyCarModel.from_rate(0.7, total_laps)

    common = dict(sc_model=sc_model, n_runs=4000, pit_grid_step=3, min_stint=9, seed=42)

    print(f"{TRACK}: {total_laps} laps\n")
    mean_res = recommend_strategy(total_laps, pace_fn, objective="mean", **common)
    print(f"Searched {mean_res.n_evaluated} strategies.\n")
    print("=== Objective: minimise EXPECTED time ===")
    print(f"Recommended: {_lbl(mean_res.best.compounds, mean_res.best.pit_laps)}")
    _print_shortlist(mean_res)

    print("\n=== Objective: minimise the 85th-percentile time (risk-averse) ===")
    p85_res = recommend_strategy(total_laps, pace_fn, objective="p85", **common)
    print(f"Recommended: {_lbl(p85_res.best.compounds, p85_res.best.pit_laps)}")
    _print_shortlist(p85_res)

    same = p85_res.best == mean_res.best
    print(f"\nRisk-averse pick {'matches' if same else 'DIFFERS from'} the expected-time pick.")

    # Plot: recommended vs the shortlist's slowest, distributions.
    fig, ax = plt.subplots(figsize=(9, 6))
    base = mean_res.best_summary["mean_s"]
    ax.hist(mean_res.best_samples - base, bins=60, alpha=0.6,
            label=f"recommended {_lbl(mean_res.best.compounds, mean_res.best.pit_laps)}")
    # A clearly worse one-stop for contrast.
    worse = Strategy(("HARD", "MEDIUM"), (total_laps // 2,))
    from f1se.sim.simulate import simulate_race
    w = simulate_race(worse, total_laps, pace_fn, sc_model=sc_model, n_runs=4000, seed=42)
    ax.hist(w.samples - base, bins=60, alpha=0.5, label=f"baseline {_lbl(worse.compounds, worse.pit_laps)}")
    ax.set_xlabel(f"Total race time minus {base:.0f}s")
    ax.set_ylabel("sampled races")
    ax.set_title(f"Optimised vs baseline strategy — {TRACK}")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phase4_optimised_strategy.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
