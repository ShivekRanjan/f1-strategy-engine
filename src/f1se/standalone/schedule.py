"""Race calendar / schedule for a season (FastF1 event schedule).

Gives the full season calendar — rounds, circuits, session times, sprint format
— with each round flagged done/upcoming and the next session counted down. This
is the one standalone that touches FastF1 at request time (schedules aren't in
the results parquet); it's cached per process and degrades to ``None`` offline.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from f1se.config import enable_cache


def _iso_utc(value) -> str | None:
    """A tz-naive UTC session time -> ISO string with explicit +00:00."""
    if value is None or pd.isna(value):
        return None
    ts = pd.Timestamp(value)
    ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts.tz_convert("UTC")
    return ts.isoformat()


def season_schedule(year: int) -> list[dict] | None:
    """The season's rounds with circuits + session times, or ``None`` offline."""
    try:
        enable_cache()
        import fastf1

        sched = fastf1.get_event_schedule(year, include_testing=False)
    except Exception:  # pragma: no cover - network / offline
        return None

    rounds: list[dict] = []
    for _, ev in sched.iterrows():
        rnd = int(ev["RoundNumber"])
        if rnd < 1:
            continue
        sessions = []
        for i in range(1, 6):
            name = ev.get(f"Session{i}")
            date = _iso_utc(ev.get(f"Session{i}DateUtc"))
            if isinstance(name, str) and name and date:
                sessions.append({"name": name, "date": date})
        edate = ev.get("EventDate")
        rounds.append({
            "round": rnd,
            "event_name": str(ev["EventName"]),
            "country": str(ev["Country"]),
            "location": str(ev["Location"]),
            "event_date": pd.Timestamp(edate).date().isoformat() if pd.notna(edate) else None,
            "format": str(ev.get("EventFormat", "")),
            "sessions": sessions,
        })
    return rounds or None


def calendar_payload(year: int) -> dict | None:
    """Schedule + done/upcoming flags + the next round and next session to run."""
    rounds = season_schedule(year)
    if rounds is None:
        return None
    now = pd.Timestamp.now(tz="UTC")

    next_round: int | None = None
    next_session: dict | None = None
    for r in rounds:
        # Race time = last listed session (the Grand Prix); fall back to event date.
        race_iso = r["sessions"][-1]["date"] if r["sessions"] else (
            f"{r['event_date']}T00:00:00+00:00" if r["event_date"] else None)
        r["done"] = bool(race_iso and pd.Timestamp(race_iso) < now)
        if not r["done"] and next_round is None:
            next_round = r["round"]
        for s in r["sessions"]:
            sdt = pd.Timestamp(s["date"])
            if sdt > now and (next_session is None or sdt < pd.Timestamp(next_session["date"])):
                next_session = {"round": r["round"], "event_name": r["event_name"],
                                "name": s["name"], "date": s["date"]}

    return {"season": year, "rounds": rounds, "next_round": next_round,
            "next_session": next_session}


@lru_cache(maxsize=8)
def cached_calendar(year: int) -> dict | None:
    return calendar_payload(year)
