"""Track-position prior — the un-modelled cost of an extra pit stop.

The strategy optimiser simulates a car in **free air**: it costs a stop the pit
lane time and nothing else. Reality charges more for every stop — the out-lap on
cold tyres, rejoining into traffic, and execution risk (a slow stop, an unsafe
release, a mistimed safety car). So the free-air optimum over-stops: it takes the
marginally-faster two-stop that real teams decline for a lower-risk one-stop.

Backtesting 2026 confirmed the over-stopping is mostly at *easy*-to-overtake
circuits — so it's NOT primarily a track-position effect. Hence the prior is
mostly a **uniform** per-stop cost (``sec_base``), plus a **small** term that
grows where the circuit is hard to overtake (``sec_hard`` × difficulty), for the
genuinely processional tracks (Monaco, Suzuka) where losing position really bites.

This is a labelled, tunable **assumption** — the same epistemic status as the
cliff and fuel priors, not a fitted quantity. ``difficulty`` is *data-informed*:
how little the finishing order shuffles vs the grid (position retention).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd


@dataclass(frozen=True)
class OvertakingPrior:
    """Per-stop time cost the free-air simulator omits.

    ``penalty_per_stop`` seconds are added to a strategy's score per pit stop when
    ranking (not to the reported race time), so extra stops must *earn* their place.
    """

    difficulty: dict[str, float] = field(default_factory=dict)   # 0 (easy) .. 1 (hard)
    default_difficulty: float = 0.5
    sec_base: float = 3.5      # per-stop cost even where overtaking is easy
    sec_hard: float = 4.0      # extra per-stop cost at maximal overtaking difficulty

    @classmethod
    def disabled(cls) -> OvertakingPrior:
        return cls(sec_base=0.0, sec_hard=0.0)

    def penalty_per_stop(self, track: str | None) -> float:
        d = self.difficulty.get(track, self.default_difficulty) if track else self.default_difficulty
        return self.sec_base + self.sec_hard * d

    @classmethod
    def from_results(cls, results: pd.DataFrame, **kwargs) -> OvertakingPrior:
        """Derive per-track difficulty from grid→finish position retention.

        Median |finish − grid| per circuit over classified finishers; a small
        shuffle ⇒ hard to overtake ⇒ high difficulty. Normalised to [0, 1].
        """
        fin = results[results["position"].notna() & results["grid"].notna() & (results["grid"] > 0)]
        move = (fin.assign(_m=(fin["position"] - fin["grid"]).abs())
                .groupby("event_name", observed=True)["_m"].median())
        if move.empty:
            return cls(**kwargs)
        lo, hi = float(move.min()), float(move.max())
        span = hi - lo
        difficulty = {
            str(ev): (1.0 if span == 0 else float((hi - m) / span))
            for ev, m in move.items()
        }
        return cls(difficulty=difficulty, **kwargs)
