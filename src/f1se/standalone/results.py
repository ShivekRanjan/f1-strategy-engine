"""Build a results-only dataset (grid, finish, points) from FastF1 sessions.

The Ergast/Jolpica historical API is unreachable here, so we derive results from
FastF1 race sessions (minimal load — no telemetry). Resumable per race.

    python -m f1se.standalone.results 2021 2022 2023 2024
"""

from __future__ import annotations

import sys

import pandas as pd

from f1se.config import PROJECT_ROOT, enable_cache
from f1se.data.circuits import apply_circuit_keys

PROCESSED = PROJECT_ROOT / "data" / "processed"
BY_RACE_RESULTS = PROCESSED / "by_race_results"


# Current scales, confirmed against officially scored 2026 rounds. Only used
# when a source gave no points at all, which only happens for sessions so recent
# that FastF1 derived the order from lap times — so older seasons, with their
# fastest-lap point, always keep the official figures.
RACE_POINTS = [25, 18, 15, 12, 10, 8, 6, 4, 2, 1]
SPRINT_POINTS = [8, 7, 6, 5, 4, 3, 2, 1]


def _fill_points(df: pd.DataFrame, scale: list[int]) -> pd.DataFrame:
    """Points from finishing position, when the source supplied none.

    A laps-derived classification has positions but zero points, and a race
    where nobody scores is impossible — so all-zero means "not provided", never
    "nobody scored". Left alone, the standings would silently drop the round.
    """
    if df.empty or df["points"].fillna(0).any():
        return df
    df = df.copy()
    df["points"] = [float(scale[int(p) - 1]) if pd.notna(p) and 1 <= p <= len(scale) else 0.0
                    for p in df["position"]]
    return df


def _usable_cache(fp) -> pd.DataFrame | None:
    """A cached race, or None if it must be (re-)pulled.

    An empty or unclassified file is a miss, not a hit. A pull in June cached
    empty frames for every race not yet run, and because any existing file was
    trusted forever, R07-R22 stayed frozen empty for four months — later races
    were never even attempted. Treating them as misses makes that cache heal.
    """
    if not fp.exists():
        return None
    df = pd.read_parquet(fp)
    return None if df.empty or df["position"].isna().all() else df


def _upsert_years(frames: list[pd.DataFrame], years: list[int], out_name: str) -> pd.DataFrame:
    """Replace only ``years`` in the combined file and keep every other season.

    Both builders used to write just the years they were asked for over the
    whole file, so ``results 2026`` silently deleted 2023-25 — the defect that
    once cost 60,928 laps in ``data/ingest.py``. Upsert rather than rebuild from
    the per-race cache, because the cache still holds 2021-22 (2021 an all-NaN
    broken pull) that were deliberately dropped from the combined file.
    """
    new = apply_circuit_keys(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()
    out = PROCESSED / out_name
    if not out.exists():
        new.to_parquet(out)
        return new
    old = pd.read_parquet(out)
    full = pd.concat([old[~old["year"].isin(years)], new], ignore_index=True)
    if len(full) < len(old):
        # A failed network pull must not erase what is already on disk.
        raise RuntimeError(f"refusing to shrink {out_name}: {len(old)} rows -> {len(full)}")
    full.to_parquet(out)
    return full


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
            if (cached := _usable_cache(fp)) is not None:
                frames.append(_fill_points(cached, RACE_POINTS))
                continue
            try:
                s = fastf1.get_session(year, rnd, "R")
                # laps + messages: for a recent session Ergast has nothing yet, and
                # FastF1 can only derive the classification from lap times plus the
                # race-control messages. Without them positions come back all-NaN
                # (same fix as the qualifying grid in outcome.py).
                s.load(laps=True, telemetry=False, weather=False, messages=True)
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
                if df["position"].isna().all():
                    # Not run yet, or a light load returning just the entry list.
                    # Caching it would freeze an all-NaN race in forever — how
                    # the broken 2021 cache was made. Skip; the next run retries.
                    print(f"  - {year} r{rnd:02d}: no classification yet, skipped", flush=True)
                    continue
                df = _fill_points(df, RACE_POINTS)
                df.to_parquet(fp)
                frames.append(df)
                print(f"  results {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} results FAILED: {e}", flush=True)

    full = _upsert_years(frames, years, out_name)
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
            if (cached := _usable_cache(fp)) is not None:
                frames.append(_fill_points(cached, SPRINT_POINTS))
                continue
            try:
                s = fastf1.get_session(year, rnd, "S")
                # laps + messages: for a recent session Ergast has nothing yet, and
                # FastF1 can only derive the classification from lap times plus the
                # race-control messages. Without them positions come back all-NaN
                # (same fix as the qualifying grid in outcome.py).
                s.load(laps=True, telemetry=False, weather=False, messages=True)
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
                if df["position"].isna().all():
                    # Not run yet, or a light load returning just the entry list.
                    # Caching it would freeze an all-NaN race in forever — how
                    # the broken 2021 cache was made. Skip; the next run retries.
                    print(f"  - {year} r{rnd:02d}: no classification yet, skipped", flush=True)
                    continue
                df = _fill_points(df, SPRINT_POINTS)
                df.to_parquet(fp)
                frames.append(df)
                print(f"  sprint {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} sprint FAILED: {e}", flush=True)

    full = _upsert_years(frames, years, out_name)
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
