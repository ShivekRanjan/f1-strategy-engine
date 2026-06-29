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
    deg_model_2026: DegradationModel | None = None
    sc_models: dict[str, SafetyCarModel] = field(default_factory=dict)
    pit_loss_by_track: dict[str, float] = field(default_factory=dict)
    stint_limits: dict[str, int] = field(default_factory=dict)
    global_sc: SafetyCarModel = field(default_factory=SafetyCarModel)
    global_pit_loss: float = DEFAULT_PIT_LOSS_S
    well_sampled_tracks: set = field(default_factory=set)
    # Lap data + next-lap forecaster, kept so the API can serve the race-replay
    # views and the LSTM nowcast (None for component-built engines in tests).
    laps: pd.DataFrame | None = None
    forecaster: object | None = None

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
    def from_processed(cls, data_dir=None) -> StrategyEngine:
        """Build from the processed parquet datasets (degradation + calibrations)."""
        data_dir = cls._resolve_data_dir(data_dir)
        dry = pd.read_parquet(data_dir / "dry_laps.parquet")
        # Era-aware: 2026 is a regulation reset. Fit the historical model on the
        # old cars, and (if 2026 data exists) a shrunk model that blends 2026 with
        # the old-era prior. Components below (SC, pit loss) transfer and use all.
        has_2026 = bool((dry["year"] >= 2026).any())
        deg_model = fit_linear_baseline(dry[dry["year"] < 2026] if has_2026 else dry)
        deg_model_2026 = None
        if has_2026:
            from f1se.models.era import fit_era_shrunk_degradation
            deg_model_2026 = fit_era_shrunk_degradation(dry, target_min_year=2026)
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

        # A track is "well sampled" only if all three dry compounds have a fitted
        # per-track pace. Tracks missing one (usually softs, which run few laps)
        # lean on the rougher base-pace fallback there and are flagged so the UI
        # can warn — predictions stay realistic (the fallback prevents the old
        # zero-base collapse) but are less precise on the missing compound.
        fitted_counts: dict[str, int] = {}
        for ev, _comp in deg_model.intercepts:
            fitted_counts[str(ev)] = fitted_counts.get(str(ev), 0) + 1
        well_sampled = {ev for ev, n in fitted_counts.items() if n >= 3}

        # Next-lap forecaster (Phase 2.5 LSTM, exported torch-free) — optional.
        forecaster = None
        npz = data_dir / "lstm_nextlap.npz"
        if npz.exists():
            from f1se.models.lap_time import NumpyLapForecaster
            forecaster = NumpyLapForecaster.load(npz)

        return cls(
            deg_model=deg_model,
            deg_model_2026=deg_model_2026,
            total_laps_by_track={str(k): int(v) for k, v in total_laps.items()},
            sc_models=sc_models,
            pit_loss_by_track=pit_by_track,
            stint_limits=limits,
            global_sc=global_sc,
            global_pit_loss=float(global_pit),
            well_sampled_tracks=well_sampled,
            laps=dry,
            forecaster=forecaster,
        )

    # ---- per-track parameters ------------------------------------------------
    def tracks(self, *, reliable_only: bool = False) -> list[str]:
        ts = sorted(self.total_laps_by_track)
        if reliable_only and self.well_sampled_tracks:
            ts = [t for t in ts if t in self.well_sampled_tracks]
        return ts

    def is_well_sampled(self, track: str) -> bool:
        return track in self.well_sampled_tracks

    def _total_laps(self, track: str) -> int:
        if track not in self.total_laps_by_track:
            raise KeyError(f"unknown track: {track!r}")
        return self.total_laps_by_track[track]

    def _sc_model(self, track: str, scale: float = 1.0) -> SafetyCarModel:
        """The track's calibrated SC hazard, optionally scaled (0 = no safety car,
        2 = doubled risk) — used for 'what would change this call' sensitivity."""
        sc = self.sc_models.get(track, self.global_sc)
        if scale == 1.0:
            return sc
        return SafetyCarModel(prob_per_lap=sc.prob_per_lap * scale,
                              mean_duration=sc.mean_duration)

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
            "well_sampled": self.is_well_sampled(track),
        }

    # ---- core operations -----------------------------------------------------
    def _model_for(self, season: int | None):
        """Pick the era-appropriate degradation model (2026 -> shrunk, else prior)."""
        if season is not None and season >= 2026 and self.deg_model_2026 is not None:
            return self.deg_model_2026
        return self.deg_model

    def _pace_fn(self, track: str, total_laps: int, use_cliff: bool, season: int | None = None):
        cliff = CliffPrior() if use_cliff else None
        return pace_fn_from_model(self._model_for(season), track, total_laps, cliff=cliff)

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
        season: int | None = None,
        sc_scale: float = 1.0,
    ) -> dict:
        """Recommend a strategy for ``track`` with a ranked, uncertainty-aware shortlist.

        ``sc_scale`` scales the calibrated safety-car hazard (0 = assume no SC,
        2 = double the risk) for sensitivity analysis of the recommendation.
        """
        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff, season)
        pit_loss = self._pit_loss(track)
        res = recommend_strategy(
            total_laps, pace_fn,
            sc_model=self._sc_model(track, sc_scale), objective=objective,
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

    def recommend_live(
        self,
        track: str,
        current_lap: int,
        current_compound: str,
        tyre_age: int,
        *,
        compounds_used: tuple[str, ...] = (),
        objective: str = "mean",
        use_cliff: bool = True,
        n_runs: int = 2000,
        top_k: int = 5,
        seed: int = 0,
        season: int | None = None,
    ) -> dict:
        """In-race: recommend the best strategy for the REMAINING laps from now."""
        from f1se.sim.inrace import RaceState, recommend_remaining

        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff, season)
        pit_loss = self._pit_loss(track)
        used = tuple(compounds_used) or (current_compound,)
        state = RaceState(total_laps, current_lap, current_compound, tyre_age, used)
        rec = recommend_remaining(
            state, pace_fn, sc_model=self._sc_model(track), objective=objective,
            n_runs=n_runs, top_k=top_k, seed=seed, max_stint=self.stint_limits,
            pit_loss_s=pit_loss, pit_loss_sc_s=round(pit_loss * 0.5, 1),
        )
        return {
            "track": track,
            "total_laps": total_laps,
            "current_lap": current_lap,
            "laps_remaining": state.laps_remaining,
            "current_compound": current_compound,
            "tyre_age": tyre_age,
            "best_plan": rec.shortlist[0]["plan"],
            "best_future_pits": list(rec.best.future_pits),
            "best_future_compounds": list(rec.best.future_compounds),
            "n_evaluated": rec.n_evaluated,
            "shortlist": rec.shortlist,
        }

    def undercut(
        self,
        track: str,
        *,
        current_lap: int,
        gap_s: float,
        your_compound: str,
        your_age: int,
        your_new_compound: str,
        rival_compound: str,
        rival_age: int,
        rival_new_compound: str,
        rival_pit_lap: int,
        season: int | None = None,
        use_cliff: bool = True,
        n_runs: int = 2000,
    ) -> dict:
        """Should you pit now to undercut a rival? (two-car crossover analysis)."""
        from f1se.sim.duel import undercut_decision

        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff, season)
        return undercut_decision(
            current_lap=current_lap, total_laps=total_laps, pace_fn=pace_fn, gap_s=gap_s,
            your_compound=your_compound, your_age=your_age, your_new_compound=your_new_compound,
            rival_compound=rival_compound, rival_age=rival_age,
            rival_new_compound=rival_new_compound, rival_pit_lap=rival_pit_lap,
            pit_loss_s=self._pit_loss(track), n_runs=n_runs,
        )

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
        season: int | None = None,
    ) -> dict:
        """Simulate one explicit strategy; return summary + a histogram for plotting."""
        total_laps = self._total_laps(track)
        pace_fn = self._pace_fn(track, total_laps, use_cliff, season)
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

    # ---- race-replay data serving (for the live view) ------------------------
    def _require_laps(self) -> pd.DataFrame:
        if self.laps is None:
            raise RuntimeError("engine has no lap data — build it via from_processed()")
        return self.laps

    def seasons(self, track: str) -> list[int]:
        """Seasons with data for ``track`` (ascending)."""
        laps = self._require_laps()
        yrs = laps.loc[laps["event_name"] == track, "year"].dropna().unique()
        return sorted(int(y) for y in yrs)

    def replay_drivers(self, track: str, season: int) -> list[str]:
        """Drivers who have laps in ``track`` for ``season``."""
        laps = self._require_laps()
        race = laps[(laps["event_name"] == track) & (laps["year"] == season)]
        if race.empty:
            raise KeyError(f"no laps for {track!r} {season}")
        return sorted(d for d in race["driver"].dropna().astype(str).unique())

    def _driver_laps(self, track: str, season: int, driver: str) -> pd.DataFrame:
        laps = self._require_laps()
        dl = laps[(laps["event_name"] == track) & (laps["year"] == season)
                  & (laps["driver"] == driver)].sort_values("lap_number")
        if dl.empty:
            raise KeyError(f"no laps for {driver!r} in {track!r} {season}")
        return dl

    def lap_history(self, track: str, season: int, driver: str) -> dict:
        """One driver's clean racing laps — for the replay chart and slider."""
        dl = self._driver_laps(track, season, driver)
        total_laps = self.total_laps_by_track.get(track, int(dl["lap_number"].max()))

        def _f(x):  # NaN -> None, numpy -> python float
            return None if pd.isna(x) else float(x)

        laps_out = [
            {
                "lap": int(r.lap_number),
                "stint": None if pd.isna(r.stint) else int(r.stint),
                "compound": str(r.compound),
                "tyre_age": _f(r.tyre_life),
                "lap_time_s": _f(r.lap_time_s),
                "lap_time_fuel_corr_s": _f(r.lap_time_fuel_corr_s),
                "position": None if pd.isna(r.position) else int(r.position),
            }
            for r in dl.itertuples(index=False)
        ]
        return {
            "track": track, "season": int(season), "driver": driver,
            "total_laps": int(total_laps),
            "lap_min": int(dl["lap_number"].min()), "lap_max": int(dl["lap_number"].max()),
            "laps": laps_out,
        }

    def _nowcast(self, hist: pd.DataFrame) -> dict | None:
        """LSTM next-lap nowcast from the current stint's laps (None if unavailable)."""
        if self.forecaster is None or hist.empty:
            return None
        stint = hist[hist["stint"] == int(hist.iloc[-1]["stint"])]
        return self.forecaster.forecast_next_lap(stint)

    def live_replay(
        self,
        track: str,
        season: int,
        driver: str,
        current_lap: int,
        *,
        n_runs: int = 2000,
        use_cliff: bool = True,
        objective: str = "mean",
    ) -> dict:
        """Replay state at ``current_lap``: derived state + remaining-strategy call
        + the next-lap nowcast. The one call the live view makes per lap."""
        from f1se.live import state_from_laps

        dl = self._driver_laps(track, season, driver)
        total_laps = self.total_laps_by_track.get(track, int(dl["lap_number"].max()))
        hist = dl[dl["lap_number"] <= current_lap]
        if hist.empty:
            raise ValueError(f"no laps up to lap {current_lap}")
        state = state_from_laps(hist, total_laps)

        out: dict = {
            "track": track, "season": int(season), "driver": driver,
            "state": {
                "current_lap": state.current_lap, "total_laps": total_laps,
                "current_compound": state.current_compound, "tyre_age": state.tyre_age,
                "laps_remaining": state.laps_remaining,
                "compounds_used": list(state.compounds_used),
            },
            "recommendation": None,
            "nowcast": self._nowcast(hist),
        }
        if state.laps_remaining >= 3:
            out["recommendation"] = self.recommend_live(
                track, state.current_lap, state.current_compound, state.tyre_age,
                compounds_used=state.compounds_used, n_runs=n_runs, season=int(season),
                use_cliff=use_cliff, objective=objective,
            )
        return out
