"""StrategyEngine — the orchestration layer the API and UI both call.

This is where the pieces are assembled: the fitted degradation model, the
per-track safety-car and pit-loss calibrations, the stint-length guards, and the
optimiser. It exposes a small, JSON-friendly surface (:meth:`recommend`,
:meth:`simulate`, :meth:`race_info`, :meth:`tracks`).

Keeping this in the package — not in ``api.py`` or the Streamlit app — is the
architecture rule: those are thin presentation layers; all engine logic lives
here so it's tested once and reused by both. Build via :meth:`from_processed`
(loads the parquet datasets) or construct directly from components in tests.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.eda import compound_stint_limits, estimate_pit_loss
from f1se.models.cliff import CliffPrior
from f1se.models.degradation import DegradationModel, fit_linear_baseline
from f1se.sim.optimize import recommend_strategy
from f1se.sim.safety_car import SafetyCarModel, calibrate_per_track
from f1se.sim.simulate import Strategy, pace_fn_from_model, simulate_race

DEFAULT_PIT_LOSS_S = 21.0


@dataclass
class StrategyEngine:
    """Assembled strategy engine over a fitted model + per-track calibrations."""

    deg_model: DegradationModel
    total_laps_by_track: dict[str, int]
    sc_models: dict[str, SafetyCarModel] = field(default_factory=dict)
    pit_loss_by_track: dict[str, float] = field(default_factory=dict)
    stint_limits: dict[str, int] = field(default_factory=dict)
    global_sc: SafetyCarModel = field(default_factory=SafetyCarModel)
    global_pit_loss: float = DEFAULT_PIT_LOSS_S

    # ---- construction --------------------------------------------------------
    @staticmethod
    def _resolve_data_dir(data_dir=None) -> Path:
        """Find the processed-data dir robustly across run contexts.

        Order: explicit arg, ``F1SE_DATA_DIR`` env, CWD/data/processed (how the
        app/API and cloud hosts run), then the repo root. CWD-relative matters
        because a non-editable install (e.g. Streamlit Cloud) puts the package in
        site-packages, where the repo-root path would not point at the data.
        """
        if data_dir is not None:
            return Path(data_dir)
        candidates = []
        if os.environ.get("F1SE_DATA_DIR"):
            candidates.append(Path(os.environ["F1SE_DATA_DIR"]))
        candidates += [Path.cwd() / "data" / "processed", PROJECT_ROOT / "data" / "processed"]
        for c in candidates:
            if (c / "dry_laps.parquet").exists():
                return c
        return candidates[-1]  # let the subsequent read raise a clear error

    @classmethod
    def from_processed(cls, data_dir=None) -> "StrategyEngine":
        """Build from the processed parquet datasets (degradation + calibrations)."""
        data_dir = cls._resolve_data_dir(data_dir)
        dry = pd.read_parquet(data_dir / "dry_laps.parquet")
        deg_model = fit_linear_baseline(dry)
        total_laps = dry.groupby("event_name", observed=True)["lap_number"].max().astype(int).to_dict()
        limits = compound_stint_limits(dry)

        sc_models: dict[str, SafetyCarModel] = {}
        global_sc = SafetyCarModel()
        status_fp = data_dir / "track_status.parquet"
        if status_fp.exists():
            status = pd.read_parquet(status_fp)
            sc_models = calibrate_per_track(status)
            global_sc = SafetyCarModel.from_track_status(status)

        pit_by_track: dict[str, float] = {}
        global_pit = DEFAULT_PIT_LOSS_S
        racelaps_fp = data_dir / "race_laps.parquet"
        if racelaps_fp.exists():
            pit_by_track = estimate_pit_loss(pd.read_parquet(racelaps_fp))
            global_pit = pit_by_track.pop("_global", DEFAULT_PIT_LOSS_S)

        return cls(
            deg_model=deg_model,
            total_laps_by_track={str(k): int(v) for k, v in total_laps.items()},
            sc_models=sc_models,
            pit_loss_by_track=pit_by_track,
            stint_limits=limits,
            global_sc=global_sc,
            global_pit_loss=float(global_pit),
        )

    # ---- per-track parameters ------------------------------------------------
    def tracks(self) -> list[str]:
        return sorted(self.total_laps_by_track)

    def _total_laps(self, track: str) -> int:
        if track not in self.total_laps_by_track:
            raise KeyError(f"unknown track: {track!r}")
        return self.total_laps_by_track[track]

    def _sc_model(self, track: str) -> SafetyCarModel:
        return self.sc_models.get(track, self.global_sc)

    def _pit_loss(self, track: str) -> float:
        return self.pit_loss_by_track.get(track, self.global_pit_loss)

    def race_info(self, track: str) -> dict:
        sc = self._sc_model(track)
        return {
            "track": track,
            "total_laps": self._total_laps(track),
            "sc_prob_per_lap": round(sc.prob_per_lap, 4),
            "sc_mean_duration": sc.mean_duration,
            "pit_loss_s": round(self._pit_loss(track), 1),
            "stint_limits": self.stint_limits,
        }

    # ---- core operations -----------------------------------------------------
    def _pace_fn(self, track: str, total_laps: int, use_cliff: bool):
        cliff = CliffPrior() if use_cliff else None
        return pace_fn_from_model(self.deg_model, track, total_laps, cliff=cliff)

    def recommend(
        self,
        track: str,
        *,
        objective: str = "mean",
        use_cliff: bool = True,
        max_stops: int = 2,
        n_runs: int = 2000,
        top_k: int = 5,
        seed: int = 0,
    ) -> dict:
        """Recommend a strategy for ``track`` with a ranked, uncertainty-aware shortlist."""
        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff)
        pit_loss = self._pit_loss(track)
        res = recommend_strategy(
            total_laps, pace_fn,
            sc_model=self._sc_model(track), objective=objective,
            n_runs=n_runs, top_k=top_k, seed=seed,
            max_stops=max_stops, max_stint=self.stint_limits,
            pit_loss_s=pit_loss, pit_loss_sc_s=round(pit_loss * 0.5, 1),
        )
        return {
            **self.race_info(track),
            "objective": objective,
            "use_cliff": use_cliff,
            "n_evaluated": res.n_evaluated,
            "best": res.best_summary,
            "shortlist": res.shortlist,
        }

    def simulate(
        self,
        track: str,
        compounds: tuple[str, ...],
        pit_laps: tuple[int, ...],
        *,
        use_cliff: bool = True,
        n_runs: int = 4000,
        seed: int = 0,
        hist_bins: int = 40,
    ) -> dict:
        """Simulate one explicit strategy; return summary + a histogram for plotting."""
        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff)
        pit_loss = self._pit_loss(track)
        strat = Strategy(compounds=tuple(compounds), pit_laps=tuple(pit_laps))
        result = simulate_race(
            strat, total_laps, pace_fn, sc_model=self._sc_model(track),
            n_runs=n_runs, seed=seed,
            pit_loss_s=pit_loss, pit_loss_sc_s=round(pit_loss * 0.5, 1),
        )
        counts, edges = np.histogram(result.samples, bins=hist_bins)
        return {
            "track": track,
            "total_laps": total_laps,
            "compounds": list(compounds),
            "pit_laps": list(pit_laps),
            **result.summary(),
            "hist_counts": counts.tolist(),
            "hist_edges": edges.tolist(),
        }
