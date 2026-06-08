"""Live / Replay race page — in-race re-optimisation as the race unfolds.

Replays a real race lap-by-lap from cached data (works anytime): advance the
"current lap" and watch the recommended strategy for the *remaining* laps update
as tyres age. On race day the same engine call is driven by live timing instead
(see f1se.live.record_live_timing). Part of:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from f1se.config import PROJECT_ROOT
from f1se.engine import StrategyEngine
from f1se.live import state_from_laps

DRY = PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet"
if not DRY.exists() and (Path.cwd() / "data/processed/dry_laps.parquet").exists():
    DRY = Path.cwd() / "data/processed/dry_laps.parquet"

st.set_page_config(page_title="Live Race", page_icon="🔴", layout="wide")
st.title("🔴 Live Race — in-race strategy")
st.caption("Replays a real race lap-by-lap; the engine re-optimises the *remaining* "
           "strategy from the current state each lap. On race day this is fed by live timing.")


@st.cache_resource
def _engine() -> StrategyEngine:
    return StrategyEngine.from_processed()


@st.cache_data
def _laps() -> pd.DataFrame:
    return pd.read_parquet(DRY)


if not DRY.exists():
    st.warning("Dataset not found — run the ingestion first (see README).")
    st.stop()

engine, laps = _engine(), _laps()

c1, c2, c3 = st.columns(3)
_tracks = engine.tracks()
track = c1.selectbox("Circuit", _tracks,
                     format_func=lambda t: t if engine.is_well_sampled(t) else f"{t}  ⚠ limited data",
                     index=_tracks.index("Spanish Grand Prix") if "Spanish Grand Prix" in _tracks else 0)
ev = laps[laps["event_name"] == track]
seasons = sorted(ev["year"].unique())
year = c2.selectbox("Season", seasons, index=len(seasons) - 1)
race = ev[ev["year"] == year]
drivers = sorted(race["driver"].dropna().unique())
driver = c3.selectbox("Driver", drivers,
                      index=drivers.index("VER") if "VER" in drivers else 0)

dl = race[race["driver"] == driver]
total_laps = engine.total_laps_by_track.get(track, int(ev["lap_number"].max()))
max_lap = int(dl["lap_number"].max())
cur = st.slider("Current lap (drag to 'play' the race)", int(dl["lap_number"].min()),
                max_lap, min(max_lap, max(2, max_lap // 3)))

so_far = dl[dl["lap_number"] <= cur]
state = state_from_laps(so_far, total_laps)

m1, m2, m3, m4 = st.columns(4)
m1.metric("Lap", f"{state.current_lap} / {total_laps}")
m2.metric("Current tyre", state.current_compound)
m3.metric("Tyre age", f"{state.tyre_age} laps")
m4.metric("Laps remaining", state.laps_remaining)

if state.laps_remaining < 3:
    st.info("Race effectively over — nothing left to optimise.")
    st.stop()

with st.spinner("Re-optimising from current state..."):
    rec = engine.recommend_live(
        track, state.current_lap, state.current_compound, state.tyre_age,
        compounds_used=state.compounds_used, n_runs=2000)

st.success(f"**Recommended from here:**  {rec['best_plan']}")
st.caption(f"Compounds used so far: {', '.join(state.compounds_used)} · "
           f"evaluated {rec['n_evaluated']} remaining plans")

df = pd.DataFrame(rec["shortlist"])
df["P(beat best)"] = (df["win_prob_vs_best"] * 100).round(0).astype(int).astype(str) + "%"
df["remaining time"] = df["mean_remaining_s"].round(1)
st.dataframe(df[["rank", "plan", "remaining time", "P(beat best)"]].set_index("rank"),
             use_container_width=True)

st.caption("Tip: drag the lap slider forward — watch the recommendation switch to "
           "'pit now' as the current tyre ages, or 'stay out' near the finish.")
