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
import pandas as pd

# FastF1 track-status code for a full safety car (codes can concatenate per lap).
SC_CODE = "4"


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

    @classmethod
    def from_track_status(cls, status: pd.DataFrame) -> SafetyCarModel:
        """Calibrate the hazard from observed per-lap ``track_status`` data.

        ``status`` needs columns ``year, round, lap_number, track_status`` (one
        row per driver-lap, unfiltered). Replaces the literature default with the
        measured per-lap trigger rate and mean SC duration; see
        :func:`safety_car_summary` for the diagnostics behind it.
        """
        summary = safety_car_summary(status)
        prob = summary["n_periods"] / summary["total_race_laps"]
        return cls(prob_per_lap=float(prob), mean_duration=int(round(summary["mean_duration"])))

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


def calibrate_per_track(
    status: pd.DataFrame, *, shrinkage_laps: float = 150.0
) -> dict[str, SafetyCarModel]:
    """Per-circuit calibrated SC models, shrunk toward the global rate.

    With only ~2 races per track, a raw per-track hazard is noisy (a track with
    no observed SC isn't truly zero-probability). We partially pool: the per-lap
    hazard is a weighted blend of the track's own rate and the global rate, with
    ``shrinkage_laps`` of "global prior" laps. More observed laps -> more trust in
    the track's own data.
    """
    glob = safety_car_summary(status)
    global_prob = glob["n_periods"] / glob["total_race_laps"] if glob["total_race_laps"] else 0.0
    global_dur = glob["mean_duration"] or 4.0

    out: dict[str, SafetyCarModel] = {}
    for event, g in status.groupby("event_name"):
        s = safety_car_summary(g)
        laps = s["total_race_laps"]
        prob = (s["n_periods"] + shrinkage_laps * global_prob) / (laps + shrinkage_laps)
        dur = s["mean_duration"] if s["n_periods"] > 0 else global_dur
        out[str(event)] = SafetyCarModel(prob_per_lap=float(prob), mean_duration=int(round(dur)))
    return out


def track_sc_model(status: pd.DataFrame, event_name: str) -> SafetyCarModel:
    """Calibrated SC model for one circuit, falling back to the global rate.

    Convenience for the simulator/optimiser: returns the (shrunk) per-track model
    if the circuit is present in ``status``, else the global calibration.
    """
    per_track = calibrate_per_track(status)
    return per_track.get(event_name) or SafetyCarModel.from_track_status(status)


def sc_laps_in_race(race_status: pd.DataFrame) -> list[int]:
    """Race laps run under safety car: those where most cars show the SC code."""
    by_lap = race_status.groupby("lap_number")["track_status"].apply(
        lambda s: s.astype("string").str.contains(SC_CODE, na=False).mean()
    )
    return sorted(int(lap) for lap, frac in by_lap.items() if frac > 0.5)


def sc_period_durations(sc_laps: list[int]) -> list[int]:
    """Durations (in laps) of each contiguous safety-car period."""
    if not sc_laps:
        return []
    durations, start, prev = [], sc_laps[0], sc_laps[0]
    for lap in sc_laps[1:]:
        if lap == prev + 1:
            prev = lap
        else:
            durations.append(prev - start + 1)
            start = prev = lap
    durations.append(prev - start + 1)
    return durations


def safety_car_summary(status: pd.DataFrame) -> dict:
    """Aggregate safety-car statistics across races from per-lap track status."""
    n_periods = 0
    all_durations: list[int] = []
    total_race_laps = 0
    races_with_sc = 0
    n_races = 0
    for _, race in status.groupby(["year", "round"]):
        n_races += 1
        total_race_laps += int(race["lap_number"].max())
        durations = sc_period_durations(sc_laps_in_race(race))
        if durations:
            races_with_sc += 1
        n_periods += len(durations)
        all_durations.extend(durations)
    return {
        "n_races": n_races,
        "n_periods": n_periods,
        "periods_per_race": n_periods / n_races if n_races else 0.0,
        "mean_duration": float(np.mean(all_durations)) if all_durations else 0.0,
        "total_race_laps": total_race_laps,
        "pct_races_with_sc": 100.0 * races_with_sc / n_races if n_races else 0.0,
    }
