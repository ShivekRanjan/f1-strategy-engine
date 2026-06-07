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


if __name__ == "__main__":
    yrs = [int(a) for a in sys.argv[1:]] or [2023, 2024]
    build_dry_dataset(yrs)
