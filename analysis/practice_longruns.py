r"""Practice long runs: a track's tyre wear measured before its race (METHODOLOGY §22).

A track's first race of a season is the engine's blind spot: there are no laps
from it in the target era, so its wear rate has to be borrowed (§19, §20). Teams
don't borrow. They measure it in practice, where long runs on each compound come
before the race. Using them is not leakage: they happen first.

This pulls the practice session a strategist would use (FP2, or FP1 on a sprint
weekend, the only practice there), keeps clean stints of at least ``MIN_RUN``
laps, and fits a slope to each. Practice fuel loads are unknown, so the slope is
corrected only for the fuel burned *within* the run, at the same 0.03 s/kg and
1.69 kg/lap the race model already assumes.

    .venv\Scripts\python.exe analysis/practice_longruns.py 2026 [2025 ...]

Needs F1 TV access (the timing archive), so run it from a home connection.
Results are written per session, so a run cut short by the rate limit resumes.
"""
from __future__ import annotations

import sys
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT, enable_cache  # noqa: E402
from f1se.data.clean import filter_racing_laps  # noqa: E402
from f1se.data.loader import SessionRef, tidy_laps  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "processed" / "by_race_practice"
OUT = PROJECT_ROOT / "data" / "processed" / "practice_longruns.parquet"
MIN_RUN = 6
PUSH_BAND = 0.015            # keep laps within 1.5% of the run's median pace
FUEL_S_PER_LAP = 0.03 * 1.69     # pace gained per lap as fuel burns (race model's figures)


def practice_session(event) -> str:
    """FP2 normally; FP1 on a sprint weekend, where it is the only practice."""
    return "FP1" if "sprint" in str(event.get("EventFormat", "")).lower() else "FP2"


def long_runs(laps: pd.DataFrame) -> pd.DataFrame:
    """One row per clean practice stint of at least MIN_RUN laps, with its slope.

    Practice stints mix push laps with cool-down laps, traffic and engine-mode
    experiments, so only laps within ``PUSH_BAND`` of the run's median pace are
    kept, and the slope is Theil-Sen (one bad lap can't tilt it). On 2026 this
    raised the practice-to-race correlation from 0.24 to 0.41. That choice was
    made looking at 2026, which METHODOLOGY §22 records.
    """
    from scipy.stats import theilslopes

    clean = filter_racing_laps(laps, dry_only=True)
    rows = []
    for (drv, stint), g in clean.groupby(["driver", "stint"], observed=True):
        g = g.dropna(subset=["tyre_life", "lap_time_s"]).sort_values("tyre_life")
        g = g[g["lap_time_s"] < g["lap_time_s"].median() * (1 + PUSH_BAND)]
        if len(g) < MIN_RUN or np.ptp(g["tyre_life"]) == 0:
            continue
        raw = theilslopes(g["lap_time_s"].to_numpy(float), g["tyre_life"].to_numpy(float))[0]
        rows.append({"driver": str(drv), "stint": int(stint), "compound": str(g["compound"].iloc[0]),
                     "n_laps": len(g), "slope_s_per_lap": float(raw + FUEL_S_PER_LAP)})
    return pd.DataFrame(rows)


def pull(year: int) -> None:
    enable_cache()
    import fastf1

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    sched = fastf1.get_event_schedule(year, include_testing=False)
    now = pd.Timestamp.now(tz="UTC").tz_localize(None)
    for _, ev in sched[sched["RoundNumber"] >= 1].iterrows():
        rnd = int(ev["RoundNumber"])
        fp = OUT_DIR / f"{year}_{rnd:02d}.parquet"
        if (fp.exists() and "--reparse" not in sys.argv) or pd.Timestamp(ev["Session5DateUtc"]) > now:
            continue
        name = practice_session(ev)
        try:
            ses = fastf1.get_session(year, rnd, name)
            ses.load(laps=True, telemetry=False, weather=False, messages=False)
            ref = SessionRef(year=year, round=rnd, event_name=str(ev["EventName"]), session=name)
            runs = long_runs(tidy_laps(ses, ref))
        except Exception as e:  # pragma: no cover - network
            if "RateLimit" in type(e).__name__:
                print(f"rate limit reached at {year} r{rnd:02d}; re-run later to resume", flush=True)
                return
            print(f"  ! {year} r{rnd:02d} {name} FAILED: {e}", flush=True)
            continue
        runs = runs.assign(year=year, round=rnd, event_name=str(ev["EventName"]), session=name)
        runs.to_parquet(fp)
        print(f"  {year} r{rnd:02d} {name}: {len(runs)} long runs", flush=True)


def combine() -> pd.DataFrame:
    from f1se.data.circuits import apply_circuit_keys

    files = sorted(OUT_DIR.glob("*.parquet"))
    df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()
    if not df.empty:
        df = apply_circuit_keys(df)
        df.to_parquet(OUT)
    return df


if __name__ == "__main__":
    for y in [int(a) for a in sys.argv[1:] if not a.startswith("-")] or [2026]:
        pull(y)
    out = combine()
    print(f"DONE: {len(out)} long runs from {out.groupby(['year', 'round']).ngroups if len(out) else 0} sessions")
