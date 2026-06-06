"""Pull FastF1 sessions into a tidy, typed lap-level table.

The loader does exactly one thing: turn a FastF1 ``Session`` into a flat
``pandas.DataFrame`` with one row per driver-lap and a stable, documented schema.
It does **no** filtering or correction — that is :mod:`f1se.data.clean`'s job.
Keeping the two separate means the cleaning rules are testable in isolation
against fixture data with no network.

Tidy schema (see :data:`LAP_SCHEMA`)
------------------------------------
Identity      : year, round, event_name, session
Driver        : driver, driver_number, team
Lap           : lap_number, stint, lap_time_s, position
Sectors       : sector1_s, sector2_s, sector3_s
Tyre          : compound, tyre_life, fresh_tyre
Pit           : is_pit_out_lap, is_pit_in_lap
Status        : track_status, is_accurate

``track_status`` is FastF1's per-lap status string (concatenated single-digit
codes). It is carried through verbatim so the cleaning layer can decide what to
drop — green-flag laps are ``"1"``; safety car / VSC / red flag laps contain
``"4"`` / ``"6"``/``"7"`` / ``"5"`` respectively.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from f1se.config import enable_cache

# Column order / dtypes the rest of the pipeline relies on.
LAP_SCHEMA: dict[str, str] = {
    "year": "int64",
    "round": "int64",
    "event_name": "string",
    "session": "string",
    "driver": "string",
    "driver_number": "string",
    "team": "string",
    "lap_number": "int64",
    "stint": "int64",
    "lap_time_s": "float64",
    "position": "Int64",
    "sector1_s": "float64",
    "sector2_s": "float64",
    "sector3_s": "float64",
    "compound": "string",
    "tyre_life": "float64",
    "fresh_tyre": "boolean",
    "is_pit_out_lap": "boolean",
    "is_pit_in_lap": "boolean",
    "track_status": "string",
    "is_accurate": "boolean",
}


@dataclass(frozen=True)
class SessionRef:
    """A resolved reference to one session, for provenance in the tidy table."""

    year: int
    round: int
    event_name: str
    session: str


def _td_to_seconds(s: pd.Series) -> pd.Series:
    """Convert a timedelta column to float seconds (NaT -> NaN)."""
    return s.dt.total_seconds()


def tidy_laps(session, ref: SessionRef) -> pd.DataFrame:
    """Flatten a loaded FastF1 ``Session`` into the tidy lap schema.

    Split out from :func:`load_session_laps` so it can be unit-tested against a
    fixture ``Laps`` frame with no network call.

    Parameters
    ----------
    session
        A FastF1 ``Session`` whose ``.load()`` has already been called, or any
        object exposing a ``.laps`` DataFrame with FastF1's column names.
    ref
        Provenance (year/round/event/session) stamped onto every row.
    """
    laps = session.laps

    out = pd.DataFrame(
        {
            "year": ref.year,
            "round": ref.round,
            "event_name": ref.event_name,
            "session": ref.session,
            "driver": laps["Driver"],
            "driver_number": laps["DriverNumber"],
            "team": laps["Team"],
            "lap_number": laps["LapNumber"],
            "stint": laps["Stint"],
            "lap_time_s": _td_to_seconds(laps["LapTime"]),
            "position": laps["Position"] if "Position" in laps else pd.NA,
            "sector1_s": _td_to_seconds(laps["Sector1Time"]),
            "sector2_s": _td_to_seconds(laps["Sector2Time"]),
            "sector3_s": _td_to_seconds(laps["Sector3Time"]),
            "compound": laps["Compound"],
            "tyre_life": laps["TyreLife"],
            "fresh_tyre": laps["FreshTyre"],
            # A lap is an out-lap if it has a PitOutTime, an in-lap if PitInTime.
            "is_pit_out_lap": laps["PitOutTime"].notna(),
            "is_pit_in_lap": laps["PitInTime"].notna(),
            "track_status": laps["TrackStatus"],
            "is_accurate": laps["IsAccurate"] if "IsAccurate" in laps else pd.NA,
        }
    )

    return _coerce_schema(out)


def _coerce_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Reorder to :data:`LAP_SCHEMA` and coerce dtypes; fail loudly on missing cols."""
    missing = [c for c in LAP_SCHEMA if c not in df.columns]
    if missing:
        raise ValueError(f"tidy frame missing required columns: {missing}")
    df = df[list(LAP_SCHEMA.keys())].copy()
    return df.astype(LAP_SCHEMA)


def load_session_laps(
    year: int,
    gp: int | str,
    session: str = "R",
    *,
    cache_dir: str | None = None,
) -> pd.DataFrame:
    """Load one session from FastF1 and return the tidy lap-level table.

    Parameters
    ----------
    year
        Championship year (FastF1 timing data is reliable from ~2018 on).
    gp
        Round number (e.g. ``9``) or event name (e.g. ``"Spain"``).
    session
        Session code: ``"R"`` (race), ``"Q"``, ``"FP1"`` ... Defaults to race.
    cache_dir
        Optional override for the FastF1 cache location.

    Returns
    -------
    pandas.DataFrame
        One row per driver-lap, conforming to :data:`LAP_SCHEMA`.

    Notes
    -----
    This performs network I/O on a cache miss; it is intentionally **not** unit
    tested against the live API. See ``tests/test_loader.py`` for the no-network
    path that exercises :func:`tidy_laps`.
    """
    enable_cache(cache_dir)

    import fastf1

    ses = fastf1.get_session(year, gp, session)
    ses.load()  # telemetry/laps/weather; cached after first call.

    event = ses.event
    ref = SessionRef(
        year=int(year),
        round=int(event["RoundNumber"]),
        event_name=str(event["EventName"]),
        session=str(session),
    )
    return tidy_laps(ses, ref)
