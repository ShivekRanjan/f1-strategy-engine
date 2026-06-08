"""Phase 5 — Streamlit UI (thin layer over :class:`f1se.engine.StrategyEngine`).

A demo front-end: pick a race + objective, get the recommended strategy and the
outcome distribution, or simulate your own plan. All logic lives in the engine;
this file only renders. Run:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1se.engine import StrategyEngine

st.set_page_config(page_title="F1 Strategy Engine", page_icon="🏎️", layout="wide")


@st.cache_resource
def load_engine() -> StrategyEngine:
    return StrategyEngine.from_processed()


def _fmt_plan(compounds, pit_laps) -> str:
    return " → ".join(compounds) + (f"  (pit lap {', '.join(map(str, pit_laps))})" if pit_laps else "")


def _dist_figure(sim: dict, label: str) -> go.Figure:
    edges = np.array(sim["hist_edges"])
    centers = (edges[:-1] + edges[1:]) / 2
    fig = go.Figure(go.Bar(x=centers, y=sim["hist_counts"], name=label, marker_color="#1f6feb"))
    fig.add_vline(x=sim["p50_s"], line_dash="dash", line_color="black",
                  annotation_text="median")
    fig.update_layout(xaxis_title="Total race time (s)", yaxis_title="sampled races",
                      height=360, margin=dict(t=30, b=10), showlegend=False)
    return fig


def main() -> None:
    st.title("🏎️ F1 Strategy Engine")
    st.caption("Not *who will win* — *what should the team do*. Pit strategy with quantified uncertainty.")

    engine = load_engine()

    with st.sidebar:
        st.header("Race & objective")
        # Mark circuits that lack full 3-compound data (predictions are rougher).
        all_tracks = engine.tracks()
        labels = {t: (t if engine.is_well_sampled(t) else f"{t}  ⚠ limited data")
                  for t in all_tracks}
        track = st.selectbox("Circuit", all_tracks, format_func=lambda t: labels[t],
                             index=all_tracks.index("Spanish Grand Prix")
                             if "Spanish Grand Prix" in all_tracks else 0)
        objective = st.selectbox("Objective", ["mean", "median", "p85"],
                                 format_func={"mean": "Minimise expected time",
                                              "median": "Minimise median time",
                                              "p85": "Risk-averse (85th pct)"}.get)
        max_stops = st.slider("Max stops", 1, 3, 2)
        use_cliff = st.checkbox("Apply tyre-cliff prior", value=True,
                                help="Domain assumption: degradation accelerates past a per-compound age.")
        n_runs = st.select_slider("Monte Carlo runs", [1000, 2000, 4000, 8000], value=2000)

    info = engine.race_info(track)
    if not info["well_sampled"]:
        st.warning("⚠ Limited data for this circuit — at least one compound has no "
                   "fitted per-track pace, so its predictions lean on a fallback and "
                   "are rougher. Numbers stay realistic but treat them with caution.")
    c1, c2, c3 = st.columns(3)
    c1.metric("Race distance", f"{info['total_laps']} laps")
    c2.metric("Safety-car risk", f"{info['sc_prob_per_lap']*info['total_laps']*100:.0f}% / race",
              help=f"{info['sc_prob_per_lap']:.4f} per lap (calibrated)")
    c3.metric("Pit loss", f"{info['pit_loss_s']:.1f} s", help="estimated from in/out-lap times")

    st.subheader("Recommended strategy")
    with st.spinner("Searching strategies..."):
        rec = engine.recommend(track, objective=objective, use_cliff=use_cliff,
                               max_stops=max_stops, n_runs=n_runs)
    best = rec["best"]
    st.success(f"**{_fmt_plan(best['compounds'], best['pit_laps'])}**  —  "
               f"expected {best['mean_s']:.1f}s  (p50 {best['p50_s']:.1f}, p90 {best['p90_s']:.1f})")
    st.caption(f"Searched {rec['n_evaluated']} strategies · objective: {objective}")

    left, right = st.columns([3, 2])
    with left:
        st.markdown("**Shortlist** (paired win-probability vs the recommendation)")
        df = pd.DataFrame(rec["shortlist"])
        df["plan"] = [_fmt_plan(c, p) for c, p in zip(df["compounds"], df["pit_laps"])]
        df["P(beat best)"] = (df["win_prob_vs_best"] * 100).round(0).astype(int).astype(str) + "%"
        st.dataframe(df[["rank", "plan", "mean_s", "p50_s", "p90_s", "P(beat best)"]]
                     .round(1).set_index("rank"), use_container_width=True)
    with right:
        sim_best = engine.simulate(track, tuple(best["compounds"]), tuple(best["pit_laps"]),
                                   use_cliff=use_cliff, n_runs=max(n_runs, 4000))
        st.plotly_chart(_dist_figure(sim_best, "recommended"), use_container_width=True)
        st.caption(f"P(safety car) = {sim_best['p_safety_car']:.0%} · "
                   f"spread (p90−p10) = {sim_best['p90_s']-sim_best['p10_s']:.0f}s")

    # --- custom strategy ------------------------------------------------------
    st.divider()
    st.subheader("Try your own strategy")
    cc1, cc2 = st.columns(2)
    with cc1:
        n_stops = st.slider("Stops", 1, 3, 2, key="custom_stops")
        compounds = [st.selectbox(f"Stint {i+1} compound", ["SOFT", "MEDIUM", "HARD"],
                                  index=[1, 2, 0][i % 3], key=f"c{i}") for i in range(n_stops + 1)]
    with cc2:
        default_pits = [round(info["total_laps"] * (i + 1) / (n_stops + 1)) for i in range(n_stops)]
        pit_laps = [st.number_input(f"Pit {i+1} (lap)", 1, info["total_laps"] - 1, default_pits[i],
                                    key=f"p{i}") for i in range(n_stops)]
    if st.button("Simulate", type="primary"):
        try:
            sim = engine.simulate(track, tuple(compounds), tuple(sorted(pit_laps)),
                                  use_cliff=use_cliff, n_runs=max(n_runs, 4000))
            m1, m2, m3 = st.columns(3)
            m1.metric("Expected time", f"{sim['mean_s']:.1f}s")
            m2.metric("Median (p50)", f"{sim['p50_s']:.1f}s")
            m3.metric("Worst-case (p90)", f"{sim['p90_s']:.1f}s")
            st.plotly_chart(_dist_figure(sim, "your strategy"), use_container_width=True)
        except ValueError as e:
            st.error(str(e))


if __name__ == "__main__":
    main()
