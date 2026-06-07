"""Generate the README hero image from the live engine output.

Two panels: the learnt per-compound degradation curves (the model) and the
recommended strategy's outcome distribution (the decision, with uncertainty).

    .venv\\Scripts\\python.exe notebooks/make_hero.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from f1se.engine import StrategyEngine
from f1se.models.cliff import CliffPrior
from f1se.models.degradation import predict_corrected_laptime

TRACK = "Spanish Grand Prix"
COMPOUND_COLORS = {"SOFT": "#e2231a", "MEDIUM": "#f6c700", "HARD": "#cfcfcf"}
OUT = Path(__file__).resolve().parents[1] / "assets" / "hero.png"


def main() -> None:
    engine = StrategyEngine.from_processed()
    info = engine.race_info(TRACK)
    total_laps = info["total_laps"]
    rec = engine.recommend(TRACK, objective="mean", n_runs=6000)
    best = rec["best"]
    sim = engine.simulate(TRACK, tuple(best["compounds"]), tuple(best["pit_laps"]), n_runs=8000)

    plt.style.use("dark_background")
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(13, 5.2), facecolor="#0d1117")
    for ax in (axL, axR):
        ax.set_facecolor("#0d1117")

    # Left: degradation curves (pace loss vs tyre age), with the cliff prior.
    cliff = CliffPrior()
    ages = np.arange(0, 31)
    for comp, color in COMPOUND_COLORS.items():
        base = predict_corrected_laptime(engine.deg_model, comp, 0, track=TRACK)
        loss = [predict_corrected_laptime(engine.deg_model, comp, a, track=TRACK) - base
                + cliff.extra_loss(comp, a) for a in ages]
        axL.plot(ages, loss, color=color, lw=2.6, label=comp)
    axL.set_title("Tyre degradation model", color="white", fontsize=13, weight="bold")
    axL.set_xlabel("Tyre age (laps)"); axL.set_ylabel("Pace loss vs fresh (s)")
    axL.legend(title="Compound", facecolor="#161b22", labelcolor="white")
    axL.grid(alpha=0.15)

    # Right: recommended strategy outcome distribution.
    edges = np.array(sim["hist_edges"]); centers = (edges[:-1] + edges[1:]) / 2
    axR.bar(centers - sim["mean_s"], sim["hist_counts"], width=(edges[1] - edges[0]),
            color="#1f6feb", alpha=0.85)
    axR.axvline(0, color="#e2231a", lw=2, label="expected")
    axR.set_title("Recommended-strategy outcome (Monte Carlo)", color="white",
                  fontsize=13, weight="bold")
    axR.set_xlabel("Race time vs expected (s)"); axR.set_ylabel("sampled races")
    plan = " → ".join(best["compounds"]) + f"  ·  pit lap {', '.join(map(str, best['pit_laps']))}"
    axR.legend(facecolor="#161b22", labelcolor="white")
    axR.grid(alpha=0.15)

    fig.suptitle(f"F1 Strategy Engine  —  {TRACK}:  {plan}",
                 color="white", fontsize=15, weight="bold", y=0.99)
    fig.text(0.5, 0.005,
             f"{total_laps} laps · pit loss {info['pit_loss_s']:.0f}s · "
             f"P(safety car) {sim['p_safety_car']:.0%} · searched {rec['n_evaluated']} strategies · "
             f"\"what should the team do — with quantified uncertainty\"",
             ha="center", color="#9da7b3", fontsize=9)
    fig.tight_layout(rect=[0, 0.02, 1, 0.96])
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140, facecolor="#0d1117")
    plt.close(fig)
    print(f"Saved hero -> {OUT}")


if __name__ == "__main__":
    main()
