"""Tests for the season calendar.

The done/next-session logic is tested offline with a synthetic schedule; the live
FastF1 fetch is marked ``network``.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest

from f1se.standalone import schedule as S


def test_iso_utc_localises_naive_times():
    assert S._iso_utc(None) is None
    out = S._iso_utc(pd.Timestamp("2026-07-03 11:30:00"))
    assert out.startswith("2026-07-03T11:30:00") and out.endswith("+00:00")


def test_calendar_flags_done_and_finds_next(monkeypatch):
    past = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=10)
    soon = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=2)
    later = pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=3)
    synthetic = [
        {"round": 1, "event_name": "Past GP", "country": "X", "location": "x",
         "event_date": past.date().isoformat(), "format": "conventional",
         "sessions": [{"name": "Race", "date": past.isoformat()}]},
        {"round": 2, "event_name": "Next GP", "country": "Y", "location": "y",
         "event_date": later.date().isoformat(), "format": "sprint_qualifying",
         "sessions": [{"name": "Practice 1", "date": soon.isoformat()},
                      {"name": "Race", "date": later.isoformat()}]},
    ]
    monkeypatch.setattr(S, "season_schedule", lambda year: synthetic)

    p = S.calendar_payload(2026)
    assert p["rounds"][0]["done"] is True
    assert p["rounds"][1]["done"] is False
    assert p["next_round"] == 2
    assert p["next_session"]["name"] == "Practice 1" and p["next_session"]["round"] == 2
    json.dumps(p)


def test_calendar_none_when_schedule_unavailable(monkeypatch):
    monkeypatch.setattr(S, "season_schedule", lambda year: None)
    assert S.calendar_payload(2026) is None


def test_cached_calendar_expires_so_next_race_flags_stay_fresh(monkeypatch):
    """The done/next flags are time-dependent — the cache must expire, not pin them."""
    calls = {"n": 0}

    def fake_payload(year):
        calls["n"] += 1
        return {"season": year, "rounds": [], "next_round": calls["n"], "next_session": None}

    monkeypatch.setattr(S, "calendar_payload", fake_payload)
    S._CACHE.clear()
    try:
        assert S.cached_calendar(2026)["next_round"] == 1
        assert S.cached_calendar(2026)["next_round"] == 1      # within TTL: cached
        S._CACHE[2026] = (S._CACHE[2026][0] - S._TTL_S - 1, S._CACHE[2026][1])
        assert S.cached_calendar(2026)["next_round"] == 2      # expired: recomputed
        # an offline miss is not cached — the next request retries
        monkeypatch.setattr(S, "calendar_payload", lambda year: None)
        S._CACHE.clear()
        assert S.cached_calendar(2026) is None
        assert 2026 not in S._CACHE
    finally:
        S._CACHE.clear()


@pytest.mark.network
def test_season_schedule_live_2026():
    rounds = S.season_schedule(2026)
    assert rounds and len(rounds) > 10
    r = rounds[0]
    assert {"round", "event_name", "country", "sessions"} <= set(r)
    assert r["sessions"], "each round should list its sessions"
