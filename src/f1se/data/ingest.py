"""Bulk ingestion — pull whole seasons into a cleaned, dry-only lap dataset.

Pulls each race of the requested years, runs the cleaning pipeline
(``dry_only=True``), and writes one parquet per race under
``data/processed/by_race/`` plus a concatenated dataset. Resumable: a race whose
parquet already exists is loaded from disk instead of re-pulled, so an
interrupted run picks up where it left off. FastF1's own cache also persists the
raw API responses, so even a fresh parse is cheap the second time.

    .venv\\Scripts\\python.exe -m f1se.data.ingest          # all 2023 + 2024
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from f1se.config import PROJECT_ROOT, enable_cache
from f1se.data.clean import clean_laps
from f1se.data.loader import load_session_laps

PROCESSED = PROJECT_ROOT / "data" / "processed"
BY_RACE = PROCESSED / "by_race"


def season_rounds(year: int) -> list[int]:
    """Round numbers of the championship races in ``year`` (excludes testing)."""
    enable_cache()
    import fastf1

    sched = fastf1.get_event_schedule(year, include_testing=False)
    return [int(r) for r in sched["RoundNumber"] if int(r) >= 1]


def build_dry_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "dry_laps.parquet",
) -> pd.DataFrame:
    """Pull + clean every race of ``years`` into one dry-only lap dataset."""
    BY_RACE.mkdir(parents=True, exist_ok=True)
    frames: list[pd.DataFrame] = []
    pulled = skipped = failed = 0

    for year in years:
        try:
            rounds = season_rounds(year)
        except Exception as e:  # pragma: no cover - network
            print(f"! schedule {year} failed: {e}", flush=True)
            continue
        print(f"\n=== {year}: {len(rounds)} races ===", flush=True)

        for rnd in rounds:
            fp = BY_RACE / f"{year}_{rnd:02d}.parquet"
            if fp.exists():
                frames.append(pd.read_parquet(fp))
                skipped += 1
                print(f"  cached {year} r{rnd:02d}", flush=True)
                continue
            try:
                raw = load_session_laps(year, rnd, session)
                clean = clean_laps(raw, dry_only=True)
                clean.to_parquet(fp)
                frames.append(clean)
                pulled += 1
                print(f"  pulled {year} r{rnd:02d}: {len(clean):>4} dry laps "
                      f"({clean['event_name'].iloc[0] if len(clean) else '??'})", flush=True)
            except Exception as e:  # pragma: no cover - network/availability
                failed += 1
                print(f"  ! {year} r{rnd:02d} FAILED: {e}", flush=True)

    if not frames:
        raise RuntimeError("no races ingested")

    full = pd.concat(frames, ignore_index=True)
    PROCESSED.mkdir(parents=True, exist_ok=True)
    full.to_parquet(PROCESSED / out_name)
    print(f"\nDONE: {len(full)} laps from {len(frames)} races "
          f"(pulled {pulled}, cached {skipped}, failed {failed})", flush=True)
    print(f"Saved -> {PROCESSED / out_name}", flush=True)
    return full


BY_RACE_STATUS = PROCESSED / "by_race_status"


def build_track_status_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "track_status.parquet",
) -> pd.DataFrame:
    """Pull per-lap ``track_status`` for every race (for safety-car calibration).

    Loads laps only (no telemetry/weather) for speed, and keeps just the columns
    needed to detect safety-car laps. Resumable per race.
    """
    BY_RACE_STATUS.mkdir(parents=True, exist_ok=True)
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
            fp = BY_RACE_STATUS / f"{year}_{rnd:02d}.parquet"
            if fp.exists():
                frames.append(pd.read_parquet(fp))
                continue
            try:
                ses = fastf1.get_session(year, rnd, session)
                ses.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = ses.laps
                df = pd.DataFrame({
                    "year": int(year),
                    "round": int(ses.event["RoundNumber"]),
                    "event_name": str(ses.event["EventName"]),
                    "lap_number": laps["LapNumber"].astype("int64"),
                    "driver": laps["Driver"].astype("string"),
                    "track_status": laps["TrackStatus"].astype("string"),
                })
                df.to_parquet(fp)
                frames.append(df)
                print(f"  status {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} status FAILED: {e}", flush=True)

    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(PROCESSED / out_name)
    print(f"\nDONE: track status for {len(frames)} races -> {PROCESSED / out_name}", flush=True)
    return full


BY_RACE_PITLAPS = PROCESSED / "by_race_pitlaps"


def build_race_laps_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "race_laps.parquet",
) -> pd.DataFrame:
    """Pull per-lap times + pit flags + status for every race (for pit-loss calib).

    Laps-only load (fast). Keeps the columns needed to estimate per-track pit
    loss from in/out-lap times relative to neighbouring green laps. Resumable.
    """
    BY_RACE_PITLAPS.mkdir(parents=True, exist_ok=True)
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
            fp = BY_RACE_PITLAPS / f"{year}_{rnd:02d}.parquet"
            if fp.exists():
                frames.append(pd.read_parquet(fp))
                continue
            try:
                ses = fastf1.get_session(year, rnd, session)
                ses.load(laps=True, telemetry=False, weather=False, messages=False)
                laps = ses.laps
                df = pd.DataFrame({
                    "year": int(year),
                    "round": int(ses.event["RoundNumber"]),
                    "event_name": str(ses.event["EventName"]),
                    "driver": laps["Driver"].astype("string"),
                    "lap_number": laps["LapNumber"].astype("int64"),
                    "lap_time_s": laps["LapTime"].dt.total_seconds(),
                    "is_pit_in_lap": laps["PitInTime"].notna(),
                    "is_pit_out_lap": laps["PitOutTime"].notna(),
                    "track_status": laps["TrackStatus"].astype("string"),
                })
                df.to_parquet(fp)
                frames.append(df)
                print(f"  laps {year} r{rnd:02d}: {df['event_name'].iloc[0]}", flush=True)
            except Exception as e:  # pragma: no cover - network
                print(f"  ! {year} r{rnd:02d} laps FAILED: {e}", flush=True)

    full = pd.concat(frames, ignore_index=True)
    full.to_parquet(PROCESSED / out_name)
    print(f"\nDONE: race laps for {len(frames)} races -> {PROCESSED / out_name}", flush=True)
    return full


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    if cmd == "status":
        build_track_status_dataset([int(a) for a in args[1:]] or [2023, 2024])
    elif cmd == "racelaps":
        build_race_laps_dataset([int(a) for a in args[1:]] or [2023, 2024])
    else:
        build_dry_dataset([int(a) for a in args] or [2023, 2024])
