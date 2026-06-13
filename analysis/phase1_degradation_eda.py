"""Phase 1 EDA — the cleaning-validation "money plot".

Pulls one real race, runs it through the cleaning pipeline, and plots
fuel-corrected lap time vs tyre age, split by compound. This is the check the
whole project hinges on: **before any modelling**, confirm the curves actually
look like degradation (corrected pace rising with tyre age, softer compounds
degrading faster). If they don't, the fix is in the cleaning, not the model.

This makes a live FastF1 call (cached after the first run). Run it directly:

    .venv\\Scripts\\python.exe notebooks/phase1_degradation_eda.py --year 2023 --gp Spain

Outputs a PNG to notebooks/figures/.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt

from f1se.data.clean import clean_laps
from f1se.data.loader import load_session_laps

# Plot softs->hards in a consistent, intuitive colour order.
COMPOUND_COLORS = {
    "SOFT": "#e2231a",
    "MEDIUM": "#f6c700",
    "HARD": "#b0b0b0",
    "INTERMEDIATE": "#3aa655",
    "WET": "#1f6feb",
}


def plot_degradation(clean, title: str, out_path: Path) -> Path:
    """Scatter + per-compound rolling median of fuel-corrected pace vs tyre age."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for compound, grp in clean.groupby("compound", observed=True):
        color = COMPOUND_COLORS.get(str(compound), "#444444")
        ax.scatter(
            grp["tyre_life"], grp["lap_time_fuel_corr_s"],
            s=8, alpha=0.25, color=color, label=None,
        )
        # Median pace by integer tyre age — the trend the eye should follow.
        med = grp.groupby("tyre_life", observed=True)["lap_time_fuel_corr_s"].median()
        ax.plot(med.index, med.values, color=color, lw=2.2, label=str(compound))

    ax.set_xlabel("Tyre age (laps)")
    ax.set_ylabel("Fuel-corrected lap time (s)")
    ax.set_title(title)
    ax.legend(title="Compound")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)
    return out_path


def main() -> None:
    p = argparse.ArgumentParser(description="Phase 1 degradation EDA")
    p.add_argument("--year", type=int, default=2023)
    p.add_argument("--gp", default="Spain", help="round number or event name")
    p.add_argument("--session", default="R")
    args = p.parse_args()

    print(f"Pulling {args.year} {args.gp} {args.session} (cached after first call)...")
    raw = load_session_laps(args.year, args.gp, args.session)
    clean = clean_laps(raw)
    print(f"  {len(raw)} raw laps -> {len(clean)} clean racing laps")

    out = Path(__file__).parent / "figures" / f"degradation_{args.year}_{args.gp}.png"
    saved = plot_degradation(
        clean,
        title=f"Fuel-corrected pace vs tyre age — {args.year} {args.gp}",
        out_path=out,
    )
    print(f"Saved: {saved}")
    print("Sanity check: corrected pace should RISE with tyre age; softer = steeper.")


if __name__ == "__main__":
    main()
