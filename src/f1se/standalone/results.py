"""Build a results-only dataset (grid, finish, points) from FastF1 sessions.

The Ergast/Jolpica historical API is unreachable here, so we derive results from
FastF1 race sessions (minimal load — no telemetry). Resumable per race.

    python -m f1se.standalone.results 2021 2022 2023 2024
"""

from __future__ import annotations

import sys

import pandas as pd

from f1se.config import PROJECT_ROOT, enable_cache

PROCESSED = PROJECT_ROOT / "data" / "processed"
BY_RACE_RESULTS = PROCESSED / "by_race_results"


def season_rounds(year: int) -> list[int]:
    enable_cache()
    import fastf1

    sched = fastf1.get_event_schedule(year, include_testing=False)
    return [int(r) for r in sched["RoundNumber"] if int(r) >= 1]


def build_results_dataset(years: list[int], *, out_name: str = "results.parquet") -> pd.DataFrame:
    """Pull race results for each season into one tidy table."""
    BY_RACE_RESULTS.mkdir(parents=True, exist_ok=True)
    enable_cache()
    import fastf1

    frames: list[pd.DataFrame] = []
    for year in years:
        try:
            rounds = season_rounds(year)
        except Exception as e:  # pragma: no cover - network
            print(f"! schedule {year} failed: {e}", flush=True)
            continue
        for rnd in rounds:
            fp = BY_RACE_RESULTS / f"{year}_{rnd:02d}.parquet"
            if fp.exists():
                frames.append(pd.read_parquet(fp))
                continue
            try:
                s = fastf1.get_session(year, rnd, "R")
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                r = s.results
                df = pd.DataFrame({
                    "year": int(year),
                    "round": int(s.event["RoundNumber"]),
                    "event_name": str(s.event["EventName"]),
                    "driver": r["Abbreviation"].astype("string"),
                    "team": r["TeamName"].astype("string"),
                    "grid": pd.to_numeric(r["GridPosition"], errors="coerce"),
                    "position": pd.to_numeric(r["Position"], errors="coerce"),
                    "points": pd.to_numeric(r["Points"], errors="coerce"),
                    "status": r["Status"].astype("string"),
                })
                df.to_parquet(fp)
                frames.append(df)
                print(f"  results {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} results FAILED: {e}", flush=True)

    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(PROCESSED / out_name)
    print(f"\nDONE: results for {len(frames)} races -> {PROCESSED / out_name}", flush=True)
    return full


def build_sprint_dataset(years: list[int], *, out_name: str = "sprint_points.parquet") -> pd.DataFrame:
    """Pull sprint-race results (they award championship points the GP table misses).

    Sprint weekends award 8-7-6-5-4-3-2-1 to the top 8 — official standings
    include them, so a GP-only tally is simply wrong on sprint seasons (in 2026
    it even swapped P2/P3). Resumable per race, like the GP puller.
    """
    by_race = PROCESSED / "by_race_sprints"
    by_race.mkdir(parents=True, exist_ok=True)
    enable_cache()
    import fastf1

    frames: list[pd.DataFrame] = []
    for year in years:
        try:
            sched = fastf1.get_event_schedule(year, include_testing=False)
        except Exception as e:  # pragma: no cover - network
            print(f"! schedule {year} failed: {e}", flush=True)
            continue
        sprint_rounds = [int(r) for r, fmt in zip(sched["RoundNumber"], sched["EventFormat"])
                         if int(r) >= 1 and "sprint" in str(fmt).lower()]
        for rnd in sprint_rounds:
            fp = by_race / f"{year}_{rnd:02d}.parquet"
            if fp.exists():
                frames.append(pd.read_parquet(fp))
                continue
            try:
                s = fastf1.get_session(year, rnd, "S")
                s.load(laps=False, telemetry=False, weather=False, messages=False)
                r = s.results
                df = pd.DataFrame({
                    "year": int(year),
                    "round": rnd,
                    "event_name": str(s.event["EventName"]),
                    "driver": r["Abbreviation"].astype("string"),
                    "team": r["TeamName"].astype("string"),
                    "position": pd.to_numeric(r["Position"], errors="coerce"),
                    "points": pd.to_numeric(r["Points"], errors="coerce").fillna(0.0),
                })
                df.to_parquet(fp)
                frames.append(df)
                print(f"  sprint {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} sprint FAILED: {e}", flush=True)

    full = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(
        columns=["year", "round", "event_name", "driver", "team", "position", "points"])
    full.to_parquet(PROCESSED / out_name)
    print(f"\nDONE: sprints for {len(frames)} events -> {PROCESSED / out_name}", flush=True)
    return full


if __name__ == "__main__":
    args = sys.argv[1:]
    if args and args[0] == "sprints":
        yrs = [int(a) for a in args[1:]] or [2023, 2024, 2025, 2026]
        build_sprint_dataset(yrs)
    else:
        yrs = [int(a) for a in args] or [2021, 2022, 2023, 2024]
        build_results_dataset(yrs)
