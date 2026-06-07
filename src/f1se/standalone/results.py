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


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or [2021, 2022, 2023, 2024]
    build_results_dataset(yrs)
