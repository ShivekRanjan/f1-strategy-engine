"""Live race feed adapter + state extraction for the in-race predictor.

Two pieces:

- :func:`state_from_laps` — turn the laps a driver has completed *so far* into a
  :class:`~f1se.sim.inrace.RaceState`. Used by both the replay demo (cached laps)
  and live mode (freshly recorded laps).
- :func:`record_live_timing` — RACE-DAY ONLY: record F1's official live timing
  stream via FastF1's SignalR client. It produces data only while a session is
  actually running, so it can't be exercised outside a live session; during a
  race you record to a file, reload it periodically, and feed the latest state to
  :meth:`f1se.engine.StrategyEngine.recommend_live`.
"""

from __future__ import annotations

import pandas as pd

from f1se.sim.inrace import RaceState


def state_from_laps(driver_laps: pd.DataFrame, total_laps: int) -> RaceState:
    """Build the current :class:`RaceState` from one driver's completed laps.

    ``driver_laps`` must have ``lap_number``, ``compound`` and ``tyre_life`` and
    cover only laps run up to "now" (sorted or not).
    """
    if driver_laps.empty:
        raise ValueError("no laps to derive state from")
    laps = driver_laps.sort_values("lap_number")
    last = laps.iloc[-1]
    used = tuple(pd.unique(laps["compound"].dropna().astype(str)))
    return RaceState(
        total_laps=int(total_laps),
        current_lap=int(last["lap_number"]),
        current_compound=str(last["compound"]),
        tyre_age=int(last["tyre_life"]),
        compounds_used=used,
    )


def record_live_timing(output_file: str) -> None:  # pragma: no cover - race-day only
    """Record the live F1 timing stream to ``output_file`` (run DURING a session).

    Produces data only while a session is live. Afterwards, load it with FastF1's
    ``livetiming`` data source and extract laps as usual, then call
    :func:`state_from_laps` + ``StrategyEngine.recommend_live`` on each update.
    """
    from fastf1.livetiming.client import SignalRClient

    client = SignalRClient(output_file)
    client.start()
