"""Phase 3 — simulate and compare pit strategies for a real race, with uncertainty.

Fits the degradation model on the full dataset, builds a per-lap pace function
for one circuit, and Monte-Carlos several candidate strategies against a
stochastic safety car. Reports the outcome distribution per strategy and, using
common random numbers (each strategy faces the SAME sampled races), the
probability the best plan actually beats its closest rival.

    .venv\\Scripts\\python.exe notebooks/phase3_simulate.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.models.degradation import fit_linear_baseline
from f1se.sim.safety_car import SafetyCarModel
from f1se.sim.simulate import Strategy, pace_fn_from_model, simulate_race

DATA = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"
FIG_DIR = Path(__file__).parent / "figures"
TRACK = "Spanish Grand Prix"


def main() -> None:
    if not DATA.exists():
        raise SystemExit(f"dataset not found: {DATA}\nRun:  python -m f1se.data.ingest")
    laps = pd.read_parquet(DATA)
    model = fit_linear_baseline(laps)

    ev = laps[laps["event_name"] == TRACK]
    total_laps = int(ev["lap_number"].max())
    print(f"{TRACK}: {total_laps} laps. Simulating strategies...\n")

    pace_fn = pace_fn_from_model(model, TRACK, total_laps)
    sc_model = SafetyCarModel.from_rate(sc_periods_per_race=0.7, total_laps=total_laps)

    h = total_laps // 2
    t = total_laps // 3
    candidates = [
        Strategy(("MEDIUM", "HARD"), (h,)),
        Strategy(("HARD", "MEDIUM"), (h + 4,)),
        Strategy(("SOFT", "HARD"), (h - 8,)),
        Strategy(("MEDIUM", "MEDIUM", "HARD"), (t, 2 * t)),
        Strategy(("SOFT", "MEDIUM", "HARD"), (t - 4, 2 * t)),
    ]

    results = [
        simulate_race(s, total_laps, pace_fn, sc_model=sc_model, n_runs=8000, seed=42)
        for s in candidates
    ]
    results.sort(key=lambda r: r.mean)

    def label(s: Strategy) -> str:
        return "->".join(c[0] for c in s.compounds) + f" @{','.join(map(str, s.pit_laps))}"

    print(f"{'strategy':<22}{'mean':>9}{'p10':>9}{'p50':>9}{'p90':>9}{'stops':>7}")
    for r in results:
        print(f"{label(r.strategy):<22}{r.mean:>9.1f}{r.quantile(.1):>9.1f}"
              f"{r.quantile(.5):>9.1f}{r.quantile(.9):>9.1f}{r.strategy.n_stops:>7}")

    # Paired (common random numbers): how often does the best beat the runner-up?
    best, runner = results[0], results[1]
    p_win = float(np.mean(best.samples < runner.samples))
    margin = runner.mean - best.mean
    print(f"\nBest: {label(best.strategy)}  (mean {best.mean:.1f}s, "
          f"p(SC)={best.p_safety_car:.2f})")
    print(f"Beats runner-up {label(runner.strategy)} by {margin:.1f}s on average, "
          f"and wins {100*p_win:.0f}% of sampled races (paired).")

    # Distribution plot for the top 3 strategies (time relative to overall best mean).
    base = min(r.mean for r in results)
    fig, ax = plt.subplots(figsize=(9, 6))
    for r in results[:3]:
        ax.hist(r.samples - base, bins=60, alpha=0.5, label=label(r.strategy))
    ax.set_xlabel(f"Total race time minus {base:.0f}s")
    ax.set_ylabel("sampled races")
    ax.set_title(f"Strategy outcome distributions — {TRACK}\n(stochastic safety car; lower is better)")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    out = FIG_DIR / "phase3_strategy_distributions.png"
    fig.savefig(out, dpi=130)
    plt.close(fig)
    print(f"\nFigure saved: {out}")


if __name__ == "__main__":
    main()
