"""Pre-registration must never lock a call it can't defend.

It is about to run unattended, so the two refusals matter more than the happy
path: no locking before the real grid exists, and none after lights out.
"""

import sys
from datetime import UTC, datetime, timedelta

import pytest

from f1se.config import PROJECT_ROOT

sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
import preregister as P  # noqa: E402


def _run(monkeypatch, tmp_path, *, grid, starts_in_h):
    start = (datetime.now(UTC) + timedelta(hours=starts_in_h)).isoformat()
    monkeypatch.setattr(P, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(P, "_upcoming_context", lambda: None)
    monkeypatch.setattr(P, "predict_upcoming", lambda: {
        "season": 2026, "next_round": 99, "grid_source": grid,
        "predictions": [{"driver": "AAA", "team": "T", "grid": 1, "podium_prob": 0.9}]})
    monkeypatch.setattr(P, "cached_calendar", lambda s: {"rounds": [
        {"round": 99, "event_name": "Test GP", "sessions": [{"date": start}]}]})
    return P.main(), list((tmp_path / "predictions").glob("*.json"))


@pytest.mark.parametrize("grid,starts_in_h,code,written", [
    ("qualifying", 20, 0, 1),   # the real grid, before the race: lock it
    ("form", 20, 0, 0),         # quali not published yet: wait, don't lock
    ("qualifying", -2, 1, 0),   # race already started: stale data, refuse loudly
    ("qualifying", -24 * 5, 1, 0),  # a week-old round: the refresh has stopped
])
def test_locks_only_off_the_real_grid_and_only_before_the_race(
        monkeypatch, tmp_path, grid, starts_in_h, code, written):
    rc, files = _run(monkeypatch, tmp_path, grid=grid, starts_in_h=starts_in_h)
    assert (rc, len(files)) == (code, written)


def test_an_already_locked_call_does_not_hide_a_stalled_refresh(monkeypatch, tmp_path):
    """Oct-Nov 2026: refresh failed every Monday, so this kept finding Singapore
    already locked and reported success 30 times while three races went by."""
    (tmp_path / "predictions").mkdir()
    (tmp_path / "predictions" / "2026_r99_test_gp.json").write_text("{}")
    rc, _ = _run(monkeypatch, tmp_path, grid="qualifying", starts_in_h=-24 * 5)
    assert rc == 1
    rc, _ = _run(monkeypatch, tmp_path, grid="qualifying", starts_in_h=-6)   # race just ran
    assert rc == 0
