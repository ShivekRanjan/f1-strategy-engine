"""Phase 3 (fidelity) — calibrate the safety-car model from real data.

Safety cars dominate strategy-outcome uncertainty, yet our hazard model used
literature-ballpark defaults. This measures the truth from per-lap track status
across 2023-24, replaces the defaults, and shows the effect on a race's SC risk.

    .venv\\Scripts\\python.exe notebooks/phase3_sc_calibrate.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from f1se.config import PROJECT_ROOT
from f1se.sim.safety_car import (
    SafetyCarModel,
    calibrate_per_track,
    safety_car_summary,
    sc_laps_in_race,
    sc_period_durations,
)

STATUS = PROJECT_ROOT / "data" / "processed" / "track_status.parquet"


def main() -> None:
    if not STATUS.exists():
        raise SystemExit(f"not found: {STATUS}\nRun:  python -m f1se.data.ingest status")
    status = pd.read_parquet(STATUS)

    s = safety_car_summary(status)
    print(f"Calibrated from {s['n_races']} races ({s['total_race_laps']} race-laps):")
    print(f"  races with >=1 SC : {s['pct_races_with_sc']:.0f}%")
    print(f"  SC periods / race : {s['periods_per_race']:.2f}")
    print(f"  mean SC duration  : {s['mean_duration']:.1f} laps")
    print(f"  per-lap hazard    : {s['n_periods']/s['total_race_laps']:.4f}")

    calibrated = SafetyCarModel.from_track_status(status)
    default = SafetyCarModel()  # prob_per_lap=0.013, mean_duration=4
    print("\nHazard model: default (assumed) vs calibrated (measured):")
    print(f"  prob_per_lap : {default.prob_per_lap:.4f}  ->  {calibrated.prob_per_lap:.4f}")
    print(f"  mean_duration: {default.mean_duration}      ->  {calibrated.mean_duration}")

    # Per-circuit SC rate — strategy risk is very track-dependent.
    rows = []
    for (yr, rnd), race in status.groupby(["year", "round"]):
        durs = sc_period_durations(sc_laps_in_race(race))
        rows.append({"event": race["event_name"].iloc[0], "year": int(yr),
                     "sc_periods": len(durs), "sc_laps": sum(durs)})
    by_track = (pd.DataFrame(rows).groupby("event")["sc_periods"].mean()
                .sort_values(ascending=False))
    print("\nMost & least SC-prone circuits (avg SC periods/race, 2023-24):")
    for ev, v in pd.concat([by_track.head(5), by_track.tail(3)]).items():
        print(f"  {ev:<28} {v:.1f}")

    # Per-track models (shrunk toward global for small samples).
    per_track = calibrate_per_track(status)
    print("\nPer-track calibrated hazard (shrunk; prob_per_lap):")
    for ev in ["Australian Grand Prix", "Spanish Grand Prix", "Monaco Grand Prix"]:
        if ev in per_track:
            print(f"  {ev:<26} {per_track[ev].prob_per_lap:.4f}")

    # Effect on the Spanish GP (which had ZERO SCs in 2023-24): default vs the
    # track-calibrated model. Probability of >=1 SC over 66 laps.
    import numpy as np
    rng = np.random.default_rng(0)
    spain = per_track.get("Spanish Grand Prix", calibrated)
    for name, m in [("default (assumed)", default),
                    ("global calibrated", calibrated),
                    ("Spain-specific", spain)]:
        p = float(m.sample_mask(66, 5000, rng).any(axis=1).mean())
        print(f"  P(>=1 SC) over 66 laps, {name:<18}: {p:.2f}")


if __name__ == "__main__":
    main()
