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
from f1se.data.circuits import apply_circuit_keys
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


def combine_cached_races(by_race_dir=None) -> pd.DataFrame:
    """Every race cached under ``by_race/``, concatenated into one dataset.

    The per-race parquets are the real store; the combined file is a derived
    artifact. Rebuilding it from *all* of them is what keeps ingesting one
    season from throwing away every other season (see ``build_dry_dataset``).
    """
    d = BY_RACE if by_race_dir is None else Path(by_race_dir)
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise RuntimeError(f"no cached races under {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def _write_combined(frames, cache_dir, out_name: str, *, rebuild: bool, unit: str = "rows"):
    """Rebuild a combined artifact from its per-race cache and write it safely.

    Shared by all three ingest entry points, which had the same defect: each
    concatenated only the years it was asked to pull and wrote that over the
    combined file, discarding every other season without a word.
    """
    full = pd.concat(frames, ignore_index=True) if rebuild else combine_cached_races(cache_dir)
    full = apply_circuit_keys(full)   # models key on the track, not the event
    out_fp = PROCESSED / out_name
    if out_fp.exists():
        try:
            prev = len(pd.read_parquet(out_fp))
        except Exception:  # pragma: no cover - unreadable/corrupt previous file
            prev = 0
        if len(full) < prev:
            # Never silently shrink: invisible at the time, and it surfaces later
            # as a model that quietly got worse for no traceable reason.
            msg = (f"refusing to shrink {out_name}: {prev} {unit} on disk, "
                   f"{len(full)} in the new build")
            if not rebuild:
                raise RuntimeError(msg + " (cache may be incomplete)")
            print(f"  !! {msg} — writing anyway because rebuild=True", flush=True)

    PROCESSED.mkdir(parents=True, exist_ok=True)
    full.to_parquet(out_fp)
    seasons = sorted(int(y) for y in full["year"].unique()) if "year" in full else []
    print(f"      seasons in dataset: {seasons}", flush=True)
    print(f"Saved -> {out_fp}", flush=True)
    return full


def build_dry_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "dry_laps.parquet",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Pull + clean every race of ``years``, and rewrite the combined dataset.

    ``years`` selects what gets *pulled*. The combined file is then rebuilt from
    every race cached under ``by_race/`` — so ingesting one season **adds** to
    the dataset instead of replacing it.

    This is deliberate. The obvious reading of ``ingest 2026`` is "add 2026",
    and the previous behaviour instead wrote a file containing 2026 alone,
    silently dropping the pre-2026 laps that the era-shrinkage prior is built
    on. The per-race parquets survived, so it was recoverable — but only
    because the combined artifact happened to be committed.

    ``rebuild=True`` restores the narrow behaviour: write only ``years``. Use it
    when you genuinely want a subset, and note that it will shrink the file.
    """
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

    # Newly-pulled races were written to by_race/ above, so recombining from
    # the cache picks them up along with every previously-ingested season.
    full = _write_combined(frames, BY_RACE, out_name, rebuild=rebuild, unit="laps")
    print(f"DONE: {len(full)} laps from {full.groupby(['year', 'round']).ngroups} races "
          f"(pulled {pulled}, cached {skipped}, failed {failed})", flush=True)
    return full


BY_RACE_STATUS = PROCESSED / "by_race_status"


def build_track_status_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "track_status.parquet",
    rebuild: bool = False,
) -> pd.DataFrame:
    """Pull per-lap ``track_status`` for every race (for safety-car calibration).

    Loads laps only (no telemetry/weather) for speed, and keeps just the columns
    needed to detect safety-car laps. Resumable per race.

    As with :func:`build_dry_dataset`, ``years`` selects what to *pull*; the
    combined file is rebuilt from the whole cache so one season's ingest can't
    discard the others. ``rebuild=True`` writes only ``years``.
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

    full = _write_combined(frames, BY_RACE_STATUS, out_name, rebuild=rebuild, unit="rows")
    print(f"DONE: track status for {full.groupby(['year', 'round']).ngroups} races", flush=True)
    return full


BY_RACE_PITLAPS = PROCESSED / "by_race_pitlaps"


def build_race_laps_dataset(
    years: list[int],
    *,
    session: str = "R",
    out_name: str = "race_laps.parquet",
    rebuild: bool = False,
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

    full = _write_combined(frames, BY_RACE_PITLAPS, out_name, rebuild=rebuild, unit="rows")
    print(f"DONE: race laps for {full.groupby(['year', 'round']).ngroups} races", flush=True)
    return full


if __name__ == "__main__":
    args = sys.argv[1:]
    cmd = args[0] if args else ""
    rebuild = "--rebuild" in args
    yrs = [int(a) for a in args[1:] if not a.startswith("-")]
    if cmd == "status":
        build_track_status_dataset(yrs or [2023, 2024], rebuild=rebuild)
    elif cmd == "racelaps":
        build_race_laps_dataset(yrs or [2023, 2024], rebuild=rebuild)
    else:
        build_dry_dataset([int(a) for a in args if not a.startswith("-")] or [2023, 2024],
                          rebuild=rebuild)
