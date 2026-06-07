"""Phase 3 — safety-car hazard model.

A safety car (SC) bunches the field and makes a pit stop much cheaper (everyone
is already slow), so SC timing is the single biggest source of strategy
uncertainty. We model it as a simple per-lap hazard: each racing lap can *trigger*
an SC with probability ``prob_per_lap``, and an SC then lasts ``mean_duration``
laps. Sampling many races gives the distribution of SC scenarios the simulator
rolls strategies against.

Defaults are literature-ballpark (~0.6–0.8 SC periods per race); they can be
recalibrated from data later via :meth:`SafetyCarModel.from_rate`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class SafetyCarModel:
    """Per-lap safety-car hazard.

    Attributes
    ----------
    prob_per_lap
        Probability an SC is triggered on any given racing lap.
    mean_duration
        Number of laps an SC stays out once triggered.
    """

    prob_per_lap: float = 0.013
    mean_duration: int = 4

    @classmethod
    def from_rate(cls, sc_periods_per_race: float, total_laps: int, mean_duration: int = 4):
        """Build from an expected number of SC periods per race of ``total_laps``."""
        return cls(prob_per_lap=sc_periods_per_race / total_laps, mean_duration=mean_duration)

    def sample_mask(self, total_laps: int, n_runs: int, rng: np.random.Generator) -> np.ndarray:
        """Return a boolean ``(n_runs, total_laps)`` mask of laps run under SC.

        A trigger on lap ``l`` puts laps ``l .. l+mean_duration-1`` under SC
        (overlapping triggers simply merge).
        """
        triggers = rng.random((n_runs, total_laps)) < self.prob_per_lap
        D = max(1, int(self.mean_duration))
        csum = np.cumsum(triggers.astype(np.int32), axis=1)
        shifted = np.zeros_like(csum)
        if D < total_laps:
            shifted[:, D:] = csum[:, :-D]
        # Number of triggers in the trailing D-lap window; >0 => SC active.
        return (csum - shifted) > 0
