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
from f1se.standalone.championship import predict_season, project_ongoing_season
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


@st.cache_data(show_spinner=False)
def cached_recommend(track: str, objective: str, use_cliff: bool, max_stops: int,
                     n_runs: int, season: int, sc_scale: float = 1.0) -> dict:
    """Memoise recommendations — repeat clicks with the same settings are instant
    (matters on free-tier CPU, where a fresh search takes a few seconds)."""
    return load_engine().recommend(track, objective=objective, use_cliff=use_cliff,
                                   max_stops=max_stops, n_runs=n_runs, season=season,
                                   sc_scale=sc_scale)


@st.cache_data(show_spinner=False)
def cached_simulate(track: str, compounds: tuple, pit_laps: tuple, use_cliff: bool,
                    n_runs: int, season: int) -> dict:
    return load_engine().simulate(track, compounds, pit_laps, use_cliff=use_cliff,
                                  n_runs=n_runs, season=season)


@st.cache_resource
def load_forecaster():
    """Torch-free next-lap forecaster (Phase 2.5 LSTM, exported to numpy weights).
    Returns None if the artifact isn't present, so the app degrades gracefully."""
    from f1se.models.lap_time import NumpyLapForecaster

    fp = PROC / "lstm_nextlap.npz"
    return NumpyLapForecaster.load(fp) if fp.exists() else None


@st.cache_resource
def load_outcome():
    fp = PROC / "results.parquet"
    if not fp.exists():
        return None
    results = pd.read_parquet(fp)
    feats = build_features(results)
    test_year = int(results["year"].max())
    model = train_podium_model(feats, test_year=test_year)
    full = int(results.groupby("year")["round"].nunique().max())          # ~full-season length
    done = int(results[results["year"] == test_year]["round"].nunique())
    ongoing = done < full - 2
    if ongoing:                                                            # mid-season (e.g. 2026)
        champ = project_ongoing_season(results, test_year, total_races=full, n_sims=5000)
    else:
        champ = predict_season(results, test_year, n_sims=5000)
    return results, feats, model, test_year, champ, ongoing, done, full


# --- helpers ----------------------------------------------------------------
def clock(seconds: float) -> str:
    """Seconds -> H:MM:SS (or M:SS for short), the human-readable race time."""
    s = int(round(seconds))
    h, m, sec = s // 3600, (s % 3600) // 60, s % 60
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m}:{sec:02d}"


def fmt_plan(compounds, pit_laps) -> str:
    return " → ".join(compounds) + (f"  (pit lap {', '.join(map(str, pit_laps))})" if pit_laps else "")


def beats_pick(rank: int, prob: float) -> str:
    """Plain-English version of the paired win-probability column."""
    return "★ our pick" if rank == 1 else f"wins {prob*100:.0f}% of races"


def dist_figure(sim: dict) -> go.Figure:
    edges = np.array(sim["hist_edges"]) / 60.0          # seconds -> minutes
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=sim["hist_counts"], marker_color="#1f6feb"))
    fig.add_vline(x=sim["p50_s"] / 60.0, line_dash="dash", line_color="white",
                  annotation_text="median")
    fig.update_layout(xaxis_title="Total race time (minutes)", yaxis_title="sampled races",
                      height=340, margin=dict(t=30, b=10), showlegend=False)
    return fig


def track_label(engine, t: str, tracks_2026=frozenset()) -> str:
    tag = "  🆕 2026" if t in tracks_2026 else ""
    if not engine.is_well_sampled(t):
        tag += "  ⚠ limited data"
    return t + tag


def default_track_index(tracks: list[str], tracks_2026) -> int:
    """Default to a 2026 circuit (so 2026 mode shows on load), else Spain, else 0."""
    for pref in ["Japanese Grand Prix", "Miami Grand Prix", "Monaco Grand Prix"]:
        if pref in tracks_2026 and pref in tracks:
            return tracks.index(pref)
    if "Spanish Grand Prix" in tracks:
        return tracks.index("Spanish Grand Prix")
    return 0


# --- Tab 1: Strategy --------------------------------------------------------
def strategy_tab(engine: StrategyEngine, laps: pd.DataFrame) -> None:
    tracks = engine.tracks()
    tracks_2026 = frozenset(laps[laps["year"] >= 2026]["event_name"].unique())
    st.caption(f"🆕 Circuits already raced under 2026 regs: **{', '.join(sorted(tracks_2026))}** "
               f"— pick one and set Season to 2026 to see new-regs mode.")
    c1, c2, c3, c4 = st.columns([3, 2, 3, 2])
    track = c1.selectbox("Circuit", tracks, format_func=lambda t: track_label(engine, t, tracks_2026),
                         index=default_track_index(tracks, tracks_2026), key="s_track")
    years = sorted(laps[laps["event_name"] == track]["year"].unique())
    # Key depends on the track so Season resets to the circuit's latest season
    # (e.g. 2026 for a 2026 circuit) instead of stickily keeping a prior value.
    season = c2.selectbox("Season", years, index=len(years) - 1, key=f"s_season_{track}",
                          format_func=lambda y: f"{y}  (new regs)" if y >= 2026 else str(y))
    objective = c3.selectbox("Objective", ["mean", "median", "p85"], key="s_obj",
                             format_func={"mean": "Minimise expected time",
                                          "median": "Minimise median time",
                                          "p85": "Risk-averse (85th pct)"}.get)
    max_stops = c4.slider("Max stops", 1, 3, 2, key="s_stops")
    with st.expander("Advanced settings"):
        n_runs = st.select_slider("Monte Carlo runs", [1000, 2000, 4000, 8000], value=2000,
                                  key="s_runs", help="More runs = smoother estimates, slower search.")
        use_cliff = st.checkbox("Apply tyre-cliff prior", value=True, key="s_cliff",
                                help="Domain assumption: degradation accelerates past a per-compound age.")

    if season >= 2026 and engine.deg_model_2026 is not None:
        n26 = engine.deg_model_2026.meta.get("n_target_laps", 0)
        st.info(f"🆕 **2026 mode** — new-regulation cars. Degradation blends {season} data "
                f"(~{n26} laps so far) with the pre-2026 prior, shrinking toward 2026 as the "
                f"season runs. Early-season numbers are necessarily uncertain.")

    info = engine.race_info(track)
    if not info["well_sampled"]:
        st.warning("⚠ Limited data for this circuit — a compound lacks fitted per-track pace, "
                   "so its predictions are rougher (still realistic).")
    m1, m2, m3 = st.columns(3)
    m1.metric("Race distance", f"{info['total_laps']} laps")
    # P(>=1 SC) — matches the simulator's reported p_safety_car, unlike the raw
    # expected-count (hazard x laps), which overstates and confused the two.
    p_any_sc = 1 - (1 - info["sc_prob_per_lap"]) ** info["total_laps"]
    m2.metric("Safety-car chance", f"{p_any_sc*100:.0f}%",
              help="Probability of at least one safety car this race (calibrated per circuit).")
    m3.metric("Pit loss", f"{info['pit_loss_s']:.1f} s")

    with st.spinner("Searching strategies..."):
        rec = cached_recommend(track, objective, use_cliff, max_stops, n_runs, int(season))
    best = rec["best"]
    st.success(f"**{fmt_plan(best['compounds'], best['pit_laps'])}**  —  "
               f"expected **{clock(best['mean_s'])}**  ·  typical {clock(best['p50_s'])}  ·  "
               f"bad luck {clock(best['p90_s'])}")
    st.caption(f"Searched {rec['n_evaluated']} strategies · objective: {objective}")

    # One-line takeaway: is the pick clear-cut, or are the top plans near-tied?
    df = pd.DataFrame(rec["shortlist"])
    df["plan"] = [fmt_plan(c, p) for c, p in zip(df["compounds"], df["pit_laps"])]
    df["expected"] = df["mean_s"].map(clock)
    df["typical"] = df["p50_s"].map(clock)
    df["bad luck"] = df["p90_s"].map(clock)
    df["how it compares"] = [beats_pick(r, p) for r, p in zip(df["rank"], df["win_prob_vs_best"])]
    show_cols = ["rank", "plan", "expected", "typical", "bad luck", "how it compares"]
    if len(df) > 1:
        runner_up_wins = float(df.iloc[1]["win_prob_vs_best"])
        spread = float(df["mean_s"].max() - df["mean_s"].min())
        if runner_up_wins >= 0.30:
            st.info(f"ℹ️ The top plans are near-tied (all within ~{spread:.0f}s) — the call is "
                    "robust to plan details; safety-car timing matters more than which you pick.")
        else:
            st.info(f"ℹ️ The pick is clear-cut — the runner-up wins only "
                    f"{runner_up_wins*100:.0f}% of simulated races.")

        # "What would change this call?" — counterfactual with no safety car.
        if info["sc_prob_per_lap"] > 0:
            no_sc = cached_recommend(track, objective, use_cliff, max_stops, n_runs,
                                     int(season), 0.0)
            nb = no_sc["best"]
            if (tuple(nb["compounds"]), tuple(nb["pit_laps"])) != \
               (tuple(best["compounds"]), tuple(best["pit_laps"])):
                st.caption(f"🔀 **What would change this?** With *no* safety car, the call flips to "
                           f"**{fmt_plan(nb['compounds'], nb['pit_laps'])}** — so this recommendation "
                           "is partly a hedge against the SC risk above.")
            else:
                st.caption("🔀 **What would change this?** The pick holds even assuming no safety "
                           "car — it's driven by pace and degradation, not SC hedging.")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Closest alternatives**")
        st.dataframe(df.head(3)[show_cols].set_index("rank"), use_container_width=True)
        with st.expander(f"All {len(df)} shortlisted plans + column guide"):
            st.dataframe(df[show_cols].set_index("rank"), use_container_width=True)
            st.caption("**expected** = average race time · **typical** = middle outcome · "
                       "**bad luck** = a rough race (worst ~10%). **'how it compares'** = how often "
                       "that plan would actually finish ahead of our pick across thousands of "
                       "simulated races — lower means our pick is more clearly best.")
    with right:
        sim = cached_simulate(track, tuple(best["compounds"]), tuple(best["pit_laps"]),
                              use_cliff, max(n_runs, 4000), int(season))
        st.plotly_chart(dist_figure(sim), use_container_width=True)
        st.caption(f"P(safety car) = {sim['p_safety_car']:.0%} · "
                   f"spread (p90−p10) = {(sim['p90_s']-sim['p10_s']):.0f}s")


# --- Tab 2: Outcome predictor ----------------------------------------------
def outcome_tab() -> None:
    loaded = load_outcome()
    if loaded is None:
        st.warning("Results dataset not found — run `python -m f1se.standalone.results 2023 2024 2025 2026`.")
        return
    results, feats, model, test_year, champ, ongoing, done, full = loaded
    mtr = model.metrics

    # Championship first — it's the headline (visual, correct, and live for an
    # ongoing season). Per-race podium detail follows below.
    if ongoing:
        st.markdown(f"#### Championship projection — {test_year} (live, after {done} of {full} races)")
        st.caption("Current points + the remaining races simulated, using **this season's** form "
                   "(after a regulation reset, last year's order no longer applies). Each "
                   "simulation bootstraps driver strength from the few races so far, so the "
                   "odds honestly reflect how little evidence the season has produced yet.")
    else:
        st.markdown(f"#### Championship projection — {test_year} (Monte Carlo from prior form)")
    top = champ.head(8).iloc[::-1]

    def _pct(p: float) -> str:  # don't round a real 0.4% chance down to "0%"
        return f"{p*100:.0f}%" if p >= 0.10 or p == 0 else f"{p*100:.1f}%"

    fig = go.Figure(go.Bar(x=top["win_prob"] * 100, y=top["driver"], orientation="h",
                           marker_color="#e2231a",
                           text=[_pct(p) for p in top["win_prob"]], textposition="outside"))
    fig.update_layout(xaxis_title="Title probability (%)", height=340, margin=dict(t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.markdown(f"#### Podium predictor — forward-tested on {test_year}")
    test = feats[feats["year"] == test_year]
    rounds = sorted(test["round"].unique())
    rnd = st.select_slider(f"{test_year} round", rounds, value=rounds[0], key="o_round")
    race = test[test["round"] == rnd]
    pred = predict_race(model, race).head(8).copy()
    podium = set(race[race["podium"] == 1]["driver"])
    pred["actual"] = pred["driver"].map(lambda d: "🏆" if d in podium else "")
    pred["podium prob"] = (pred["podium_prob"] * 100).round(0).astype(int).astype(str) + "%"
    st.markdown(f"**{race['event_name'].iloc[0]}** — predicted podium probabilities")
    st.dataframe(pred[["driver", "team", "grid", "podium prob", "actual"]],
                 use_container_width=True, hide_index=True)

    with st.expander("Model quality (honest numbers)"):
        c1, c2, c3 = st.columns(3)
        c1.metric("ROC-AUC", f"{mtr['auc']:.3f}",
                  help="How well the model ranks podium vs non-podium drivers (1.0 = perfect).")
        c2.metric("Model precision@3", f"{mtr['model_precision_at_3']:.0%}")
        c3.metric("Grid-baseline precision@3", f"{mtr['grid_baseline_precision_at_3']:.0%}")
        st.caption(f"Forward-tested on {test_year} ({done if ongoing else full} races"
                   f"{' so far — small sample, so precision@3 is noisy' if ongoing else ''}). "
                   "Grid position is itself the strongest podium signal; the model's value is the "
                   "*calibrated probability* per driver, not reshuffling the grid's top 3.")


# --- Tab 3: Race replay / live ----------------------------------------------
def live_tab(engine: StrategyEngine, laps: pd.DataFrame) -> None:
    st.caption("**Replay mode**: any past race, lap by lap — the engine re-optimises the remaining "
               "strategy from the current state each lap. On race day the same engine call is fed "
               "by F1's live timing stream instead (`f1se.live.record_live_timing`).")
    tracks = engine.tracks()
    tracks_2026 = frozenset(laps[laps["year"] >= 2026]["event_name"].unique())
    c1, c2, c3 = st.columns(3)
    track = c1.selectbox("Circuit", tracks, format_func=lambda t: track_label(engine, t, tracks_2026),
                         index=default_track_index(tracks, tracks_2026), key="l_track")
    ev = laps[laps["event_name"] == track]
    seasons = sorted(ev["year"].unique())
    year = c2.selectbox("Season", seasons, index=len(seasons) - 1, key=f"l_season_{track}")
    race = ev[ev["year"] == year]
    drivers = sorted(race["driver"].dropna().unique())
    driver = c3.selectbox("Driver", drivers, index=drivers.index("VER") if "VER" in drivers else 0,
                          key=f"l_driver_{track}_{year}")

    dl = race[race["driver"] == driver]
    total_laps = engine.total_laps_by_track.get(track, int(ev["lap_number"].max()))
    lo, hi = int(dl["lap_number"].min()), int(dl["lap_number"].max())
    cur = st.slider("Current lap (drag to 'play' the race)", lo, hi, min(hi, max(2, hi // 3)),
                    key=f"l_lap_{track}_{year}_{driver}")

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
                                    state.tyre_age, compounds_used=state.compounds_used,
                                    n_runs=2000, season=int(year))
    st.success(f"**Recommended from here:**  {rec['best_plan']}")
    st.caption(f"Compounds used so far: {', '.join(state.compounds_used)} · "
               f"evaluated {rec['n_evaluated']} remaining plans")
    df = pd.DataFrame(rec["shortlist"])
    df["how it compares"] = [beats_pick(r, p) for r, p in zip(df["rank"], df["win_prob_vs_best"])]
    df["time left"] = df["mean_remaining_s"].map(clock)
    st.dataframe(df[["rank", "plan", "time left", "how it compares"]].set_index("rank"),
                 use_container_width=True)
    st.caption("**time left** = average time to the flag · **'how it compares'** = how often that "
               "plan would beat our pick across simulated races (lower → our pick is clearly best).")

    _nextlap_nowcast(dl, cur)


def _nextlap_nowcast(driver_laps: pd.DataFrame, cur: int) -> None:
    """Phase 2.5 LSTM readout: forecast the next green lap's pace from the recent
    sequence of laps in the current stint. A *separate* model from the strategy
    sim — surfaced here because the live tab is exactly its use case."""
    forecaster = load_forecaster()
    if forecaster is None:
        return
    st.divider()
    st.markdown("##### 🔮 Next-lap pace nowcast — *LSTM sequence model*")
    hist = driver_laps[driver_laps["lap_number"] <= cur].sort_values("lap_number")
    stint_laps = hist[hist["stint"] == int(hist.iloc[-1]["stint"])]
    fc = forecaster.forecast_next_lap(stint_laps)
    if not fc["ok"]:
        st.caption(f"_{fc['reason']} — it reads the last {forecaster.window} green laps of the current stint._")
        return
    f1, f2 = st.columns([1, 1])
    f1.metric("Predicted next lap (fuel-adjusted)", f"{fc['predicted_s']:.1f} s",
              delta=f"{fc['delta_s']:+.2f} s vs last lap", delta_color="inverse")
    read = ("🔻 tyres fading" if fc["delta_s"] > 0.05
            else "🔺 still coming in / improving" if fc["delta_s"] < -0.05
            else "➡ holding steady")
    f2.metric("What the sequence implies", read)
    st.caption(f"Forecasts the next green lap *if you stay out*, from the recent lap **sequence** "
               f"(not just tyre age). The dumb baseline just repeats the last lap "
               f"({fc['last_s']:.1f}s); on held-out 2025 this LSTM beat that by ~8.5%. "
               f"Independent of the strategy search above.")


# --- Tab 4: Undercut duel ---------------------------------------------------
def undercut_tab(engine: StrategyEngine, laps: pd.DataFrame) -> None:
    st.caption("**Should you pit now to jump a rival?** A two-car *cumulative-time* model of the "
               "undercut — fresh-tyre pace vs the gap and pit loss, judged at the crossover once "
               "both have stopped. (Free-air: it sizes the time delta, not dirty-air overtaking.)")
    tracks = engine.tracks()
    c1, c2, c3 = st.columns(3)
    track = c1.selectbox("Circuit", tracks, format_func=lambda t: track_label(engine, t),
                         index=default_track_index(tracks, frozenset()), key="u_track")
    total = engine.total_laps_by_track.get(track, 60)
    cur = c2.slider("Current lap", 2, total - 4, min(total // 3, total - 5), key="u_lap")
    gap = c3.slider("Gap to rival (s, + = you're behind)", -5.0, 25.0, 2.0, 0.5, key="u_gap")

    you_c, riv_c = st.columns(2)
    comps = ["SOFT", "MEDIUM", "HARD"]
    with you_c:
        st.markdown("**You**")
        yc = st.selectbox("Current tyre", comps, index=1, key="u_yc")
        ya = st.slider("Tyre age", 1, total, min(15, total), key="u_ya")
        ynew = st.selectbox("Pit to", comps, index=2, key="u_ynew")
    with riv_c:
        st.markdown("**Rival**")
        rc = st.selectbox("Current tyre", comps, index=2, key="u_rc")
        ra = st.slider("Tyre age", 1, total, min(15, total), key="u_ra")
        rnew = st.selectbox("Pit to", comps, index=1, key="u_rnew")
        rpit = st.slider("Rival's expected pit lap", cur + 1, total - 1,
                         min(cur + 8, total - 1), key="u_rpit")

    res = engine.undercut(track, current_lap=cur, gap_s=gap, your_compound=yc, your_age=ya,
                          your_new_compound=ynew, rival_compound=rc, rival_age=ra,
                          rival_new_compound=rnew, rival_pit_lap=rpit)
    (st.success if res["undercut_works"] else st.info)(f"**{res['verdict']}**")

    def _gap_txt(g): return f"{abs(g):.1f}s {'ahead' if g < 0 else 'behind'}"
    m1, m2 = st.columns(2)
    m1.metric("Pit now (undercut)", _gap_txt(res["undercut"]["final_gap_s"]),
              help=f"ends ahead {res['undercut']['p_ahead']*100:.0f}% of simulated races")
    m2.metric("Cover (pit with rival)", _gap_txt(res["cover"]["final_gap_s"]),
              help=f"ends ahead {res['cover']['p_ahead']*100:.0f}% of simulated races")
    st.caption(f"Undercutting nets **{res['undercut_gain_s']:+.1f}s** vs covering, measured at the "
               "crossover. Positive = the undercut is faster.")


# --- layout -----------------------------------------------------------------
def main() -> None:
    st.title("🏎️ F1 Strategy Engine")
    st.caption("Not *who will win* — *what should the team do*. Pit strategy, outcomes, and "
               "live in-race calls, all with quantified uncertainty.")
    engine = load_engine()
    dry = load_dry()
    t1, t2, t3, t4 = st.tabs(["🏁 Strategy", "🆚 Undercut", "🏆 Outcome Predictor",
                              "🔴 Race Replay / Live"])
    with t1:
        strategy_tab(engine, dry)
    with t2:
        undercut_tab(engine, dry)
    with t3:
        outcome_tab()
    with t4:
        live_tab(engine, dry)


if __name__ == "__main__":
    main()
