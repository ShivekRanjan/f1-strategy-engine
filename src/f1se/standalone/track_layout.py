"""Circuit outline SVGs from FastF1 position telemetry (V2 UI: track shape next
to the circuit name in Strategy's picker and the Race Hub header).

A single fastest-lap X/Y trace is normalised into a compact SVG path string —
no server-side SVG rendering needed, the frontend just drops the path into an
``<svg>``. Committed as a tiny lookup, refreshed with
``python -m f1se.standalone.track_layout``.
"""

from __future__ import annotations

from functools import lru_cache

import pandas as pd

from f1se.config import PROJECT_ROOT, enable_cache
from f1se.data.circuits import circuit_key

PROCESSED = PROJECT_ROOT / "data" / "processed"
OUT = PROCESSED / "track_layouts.parquet"

VIEWBOX = 200  # square viewBox; path coordinates are normalised into [0, VIEWBOX]
MARGIN = 10
MAX_POINTS = 120  # downsample — a circuit outline doesn't need every telemetry sample


def _svg_path(xs: list[float], ys: list[float]) -> str:
    """Normalise a lap's X/Y trace into an SVG path string, y-flipped (SVG is top-down)."""
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    span = max(x1 - x0, y1 - y0) or 1.0
    scale = (VIEWBOX - 2 * MARGIN) / span
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

    def norm(x: float, y: float) -> tuple[float, float]:
        return (
            round(VIEWBOX / 2 + (x - cx) * scale, 1),
            round(VIEWBOX / 2 - (y - cy) * scale, 1),  # flip Y
        )

    pts = [norm(x, y) for x, y in zip(xs, ys)]
    if len(pts) > MAX_POINTS:
        step = len(pts) / MAX_POINTS
        pts = [pts[int(i * step)] for i in range(MAX_POINTS)]
    d = f"M {pts[0][0]},{pts[0][1]} " + " ".join(f"L {x},{y}" for x, y in pts[1:]) + " Z"
    return d


def build_track_layouts(seasons: list[int] | None = None) -> pd.DataFrame:
    """One SVG path per circuit, from that circuit's most recent race fastest lap."""
    enable_cache()
    import fastf1

    seasons = seasons or [2026, 2025, 2024, 2023]
    rows: dict[str, dict] = {}  # track -> row; first (newest) season wins
    for yr in seasons:
        try:
            sched = fastf1.get_event_schedule(yr, include_testing=False)
            now = pd.Timestamp.now()
            done = sched[(sched["RoundNumber"] >= 1) & (sched["EventDate"] < now)]
            rounds = [int(r) for r in done["RoundNumber"]]
        except Exception:  # pragma: no cover - network
            continue
        for rnd in rounds:
            try:
                ev = sched[sched["RoundNumber"] == rnd].iloc[0]
                track = circuit_key(yr, str(ev["EventName"]))
                if track in rows:
                    continue
                s = fastf1.get_session(yr, rnd, "R")
                s.load(laps=True, telemetry=True, weather=False, messages=False)
                lap = s.laps.pick_fastest()
                tel = lap.get_telemetry()
                xs, ys = list(tel["X"]), list(tel["Y"])
                if len(xs) < 10:
                    continue
                rows[track] = {"track": track, "path": _svg_path(xs, ys), "viewbox": VIEWBOX}
                print(f"  {track}: {len(xs)} points -> {len(rows[track]['path'].split())} nodes", flush=True)
            except Exception:  # pragma: no cover - network / missing telemetry
                continue

    if not rows:
        raise RuntimeError("no track layouts pulled")
    df = pd.DataFrame(rows.values())
    # Merge, newest wins: rebuilding one season must not drop every circuit
    # that season did not visit.
    if OUT.exists():
        old = pd.read_parquet(OUT)
        df = pd.concat([old[~old["track"].isin(df["track"])], df], ignore_index=True)
    df.to_parquet(OUT)
    print(f"DONE: {len(df)} track layouts -> {OUT}", flush=True)
    return df


@lru_cache(maxsize=1)
def track_layouts() -> dict[str, dict]:
    """``{track: {path, viewbox}}`` from the committed lookup ({} if absent)."""
    if not OUT.exists():
        return {}
    df = pd.read_parquet(OUT)
    return {
        str(r.track): {"path": str(r.path), "viewbox": int(r.viewbox)}
        for r in df.itertuples(index=False)
    }


if __name__ == "__main__":
    build_track_layouts()
