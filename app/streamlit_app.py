"""F1 Strategy Engine — one tabbed interface over f1se.engine + the standalone
outcome predictors. Thin presentation layer only; all logic lives in the package.

    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1se.config import PROJECT_ROOT
from f1se.engine import StrategyEngine
from f1se.live import state_from_laps
from f1se.standalone.championship import predict_season
from f1se.standalone.podium import build_features, predict_race, train_podium_model

st.set_page_config(page_title="F1 Strategy Engine", page_icon="🏎️", layout="wide")
PROC = PROJECT_ROOT / "data" / "processed"
if not (PROC / "dry_laps.parquet").exists() and (Path.cwd() / "data/processed/dry_laps.parquet").exists():
    PROC = Path.cwd() / "data" / "processed"


# --- cached loaders ---------------------------------------------------------
@st.cache_resource
def load_engine() -> StrategyEngine:
    return StrategyEngine.from_processed()


@st.cache_data
def load_dry() -> pd.DataFrame:
    return pd.read_parquet(PROC / "dry_laps.parquet")


@st.cache_resource
def load_outcome():
    fp = PROC / "results.parquet"
    if not fp.exists():
        return None
    results = pd.read_parquet(fp)
    feats = build_features(results)
    test_year = int(results["year"].max())
    model = train_podium_model(feats, test_year=test_year)
    champ = predict_season(results, test_year, n_sims=5000)
    return results, feats, model, test_year, champ


# --- helpers ----------------------------------------------------------------
def clock(seconds: float) -> str:
    """Seconds -> H:MM:SS (or M:SS for short), the human-readable race time."""
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_plan(compounds, pit_laps) -> str:
    return " → ".join(compounds) + (f"  (pit lap {', '.join(map(str, pit_laps))})" if pit_laps else "")


def dist_figure(sim: dict) -> go.Figure:
    edges = np.array(sim["hist_edges"]) / 60.0          # seconds -> minutes
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=sim["hist_counts"], marker_color="#1f6feb"))
    fig.add_vline(x=sim["p50_s"] / 60.0, line_dash="dash", line_color="white",
                  annotation_text="median")
    fig.update_layout(xaxis_title="Total race time (minutes)", yaxis_title="sampled races",
                      height=340, margin=dict(t=30, b=10), showlegend=False)
    return fig


def track_label(engine, t: str) -> str:
    return t if engine.is_well_sampled(t) else f"{t}  ⚠ limited data"


# --- Tab 1: Strategy --------------------------------------------------------
def strategy_tab(engine: StrategyEngine) -> None:
    tracks = engine.tracks()
    c1, c2, c3, c4 = st.columns([3, 3, 2, 2])
    track = c1.selectbox("Circuit", tracks, format_func=lambda t: track_label(engine, t),
                         index=tracks.index("Spanish Grand Prix") if "Spanish Grand Prix" in tracks else 0)
    objective = c2.selectbox("Objective", ["mean", "median", "p85"],
                             format_func={"mean": "Minimise expected time",
                                          "median": "Minimise median time",
                                          "p85": "Risk-averse (85th pct)"}.get)
    max_stops = c3.slider("Max stops", 1, 3, 2)
    n_runs = c4.select_slider("MC runs", [1000, 2000, 4000, 8000], value=2000)
    use_cliff = st.checkbox("Apply tyre-cliff prior", value=True,
                            help="Domain assumption: degradation accelerates past a per-compound age.")

    info = engine.race_info(track)
    if not info["well_sampled"]:
        st.warning("⚠ Limited data for this circuit — a compound lacks fitted per-track pace, "
                   "so its predictions are rougher (still realistic).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Race distance", f"{info['total_laps']} laps")
    m2.metric("Safety-car risk", f"{info['sc_prob_per_lap']*info['total_laps']*100:.0f}% / race")
    m3.metric("Pit loss", f"{info['pit_loss_s']:.1f} s")

    with st.spinner("Searching strategies..."):
        rec = engine.recommend(track, objective=objective, use_cliff=use_cliff,
                               max_stops=max_stops, n_runs=n_runs)
    best = rec["best"]
    st.success(f"**{fmt_plan(best['compounds'], best['pit_laps'])}**  —  "
               f"expected **{clock(best['mean_s'])}**  (p50 {clock(best['p50_s'])}, p90 {clock(best['p90_s'])})")
    st.caption(f"Searched {rec['n_evaluated']} strategies · objective: {objective}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Shortlist** (paired win-probability vs the recommendation)")
        df = pd.DataFrame(rec["shortlist"])
        df["plan"] = [fmt_plan(c, p) for c, p in zip(df["compounds"], df["pit_laps"])]
        for col in ("mean_s", "p50_s", "p90_s"):
            df[col.replace("_s", "")] = df[col].map(clock)
        df["P(beat best)"] = (df["win_prob_vs_best"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(df[["rank", "plan", "mean", "p50", "p90", "P(beat best)"]].set_index("rank"),
                     use_container_width=True)
    with right:
        sim = engine.simulate(track, tuple(best["compounds"]), tuple(best["pit_laps"]),
                              use_cliff=use_cliff, n_runs=max(n_runs, 4000))
        st.plotly_chart(dist_figure(sim), use_container_width=True)
        st.caption(f"P(safety car) = {sim['p_safety_car']:.0%} · "
                   f"spread (p90−p10) = {(sim['p90_s']-sim['p10_s']):.0f}s")


# --- Tab 2: Outcome predictor ----------------------------------------------
def outcome_tab() -> None:
    loaded = load_outcome()
    if loaded is None:
        st.warning("Results dataset not found — run `python -m f1se.standalone.results 2021 2022 2023 2024`.")
        return
    results, feats, model, test_year, champ = loaded
    mtr = model.metrics

    st.markdown(f"#### Podium predictor — forward-tested on {test_year}")
    c1, c2, c3 = st.columns(3)
    c1.metric("ROC-AUC", f"{mtr['auc']:.3f}")
    c2.metric("Model precision@3", f"{mtr['model_precision_at_3']:.0%}")
    c3.metric("Grid-baseline precision@3", f"{mtr['grid_baseline_precision_at_3']:.0%}",
              delta=f"{(mtr['model_precision_at_3']-mtr['grid_baseline_precision_at_3'])*100:+.0f} pts")

    test = feats[feats["year"] == test_year]
    rounds = sorted(test["round"].unique())
    rnd = st.select_slider(f"{test_year} round", rounds, value=rounds[0])
    race = test[test["round"] == rnd]
    pred = predict_race(model, race).head(8).copy()
    podium = set(race[race["podium"] == 1]["driver"])
    pred["actual"] = pred["driver"].map(lambda d: "🏆" if d in podium else "")
    pred["podium prob"] = (pred["podium_prob"] * 100).round(0).astype(int).astype(str) + "%"
    st.markdown(f"**{race['event_name'].iloc[0]}** — predicted podium probabilities")
    st.dataframe(pred[["driver", "team", "grid", "podium prob", "actual"]],
                 use_container_width=True, hide_index=True)

    st.markdown(f"#### Championship projection — {test_year} (Monte Carlo from prior form)")
    top = champ.head(8).iloc[::-1]
    fig = go.Figure(go.Bar(x=top["win_prob"] * 100, y=top["driver"], orientation="h",
                           marker_color="#e2231a",
                           text=[f"{p*100:.0f}%" for p in top["win_prob"]], textposition="outside"))
    fig.update_layout(xaxis_title="Title probability (%)", height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)


# --- Tab 3: Live race -------------------------------------------------------
def live_tab(engine: StrategyEngine, laps: pd.DataFrame) -> None:
    tracks = engine.tracks()
    c1, c2, c3 = st.columns(3)
    track = c1.selectbox("Circuit ", tracks, format_func=lambda t: track_label(engine, t),
                         index=tracks.index("Spanish Grand Prix") if "Spanish Grand Prix" in tracks else 0,
                         key="live_track")
    ev = laps[laps["event_name"] == track]
    seasons = sorted(ev["year"].unique())
    year = c2.selectbox("Season", seasons, index=len(seasons) - 1)
    race = ev[ev["year"] == year]
    drivers = sorted(race["driver"].dropna().unique())
    driver = c3.selectbox("Driver", drivers, index=drivers.index("VER") if "VER" in drivers else 0)

    dl = race[race["driver"] == driver]
    total_laps = engine.total_laps_by_track.get(track, int(ev["lap_number"].max()))
    lo, hi = int(dl["lap_number"].min()), int(dl["lap_number"].max())
    cur = st.slider("Current lap (drag to 'play' the race)", lo, hi, min(hi, max(2, hi // 3)))

    state = state_from_laps(dl[dl["lap_number"] <= cur], total_laps)
    a, b, c, d = st.columns(4)
    a.metric("Lap", f"{state.current_lap} / {total_laps}")
    b.metric("Current tyre", state.current_compound)
    c.metric("Tyre age", f"{state.tyre_age} laps")
    d.metric("Laps remaining", state.laps_remaining)

    if state.laps_remaining < 3:
        st.info("Race effectively over — nothing left to optimise.")
        return
    with st.spinner("Re-optimising from current state..."):
        rec = engine.recommend_live(track, state.current_lap, state.current_compound,
                                    state.tyre_age, compounds_used=state.compounds_used, n_runs=2000)
    st.success(f"**Recommended from here:**  {rec['best_plan']}")
    st.caption(f"Compounds used so far: {', '.join(state.compounds_used)} · "
               f"evaluated {rec['n_evaluated']} remaining plans")
    df = pd.DataFrame(rec["shortlist"])
    df["P(beat best)"] = (df["win_prob_vs_best"] * 100).round(0).astype(int).astype(str) + "%"
    df["remaining"] = df["mean_remaining_s"].map(clock)
    st.dataframe(df[["rank", "plan", "remaining", "P(beat best)"]].set_index("rank"),
                 use_container_width=True)


# --- layout -----------------------------------------------------------------
def main() -> None:
    st.title("🏎️ F1 Strategy Engine")
    st.caption("Not *who will win* — *what should the team do*. Pit strategy, outcomes, and "
               "live in-race calls, all with quantified uncertainty.")
    engine = load_engine()
    t1, t2, t3 = st.tabs(["🏁 Strategy", "🏆 Outcome Predictor", "🔴 Live Race"])
    with t1:
        strategy_tab(engine)
    with t2:
        outcome_tab()
    with t3:
        live_tab(engine, load_dry())


if __name__ == "__main__":
    main()
