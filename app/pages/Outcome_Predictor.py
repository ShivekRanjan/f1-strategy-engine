"""Phase A — standalone Outcome Predictor page (podium + championship).

Results-only, separate from the strategy engine. Shows the forward-validated
podium classifier (vs a grid baseline) and the Monte Carlo championship
projection. Run as part of:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from f1se.config import PROJECT_ROOT
from f1se.standalone.championship import predict_season
from f1se.standalone.podium import build_features, predict_race, train_podium_model

RESULTS = PROJECT_ROOT / "data" / "processed" / "results.parquet"
if not RESULTS.exists() and (Path.cwd() / "data/processed/results.parquet").exists():
    RESULTS = Path.cwd() / "data/processed/results.parquet"

st.set_page_config(page_title="Outcome Predictor", page_icon="🏆", layout="wide")
st.title("🏆 Outcome Predictor")
st.caption("Standalone, results-only models — separate from the strategy engine. "
           "Podium classifier (forward-validated) + Monte Carlo championship projection.")


@st.cache_resource
def _load():
    results = pd.read_parquet(RESULTS)
    feats = build_features(results)
    test_year = int(results["year"].max())
    model = train_podium_model(feats, test_year=test_year)
    champ = predict_season(results, test_year, n_sims=5000)
    return results, feats, model, test_year, champ


if not RESULTS.exists():
    st.warning("Results dataset not found. Build it with:  "
               "`python -m f1se.standalone.results 2021 2022 2023 2024`")
    st.stop()

results, feats, model, test_year, champ = _load()
m = model.metrics

# --- Podium predictor -------------------------------------------------------
st.subheader(f"Podium predictor — tested forward on {test_year}")
c1, c2, c3 = st.columns(3)
c1.metric("ROC-AUC", f"{m['auc']:.3f}")
c2.metric("Model precision@3", f"{m['model_precision_at_3']:.0%}",
          help="Of our top-3 predicted per race, share that actually finished on the podium.")
c3.metric("Grid-baseline precision@3", f"{m['grid_baseline_precision_at_3']:.0%}",
          delta=f"{(m['model_precision_at_3']-m['grid_baseline_precision_at_3'])*100:+.0f} pts",
          help="The dumb baseline: top 3 on the grid.")

test = feats[feats["year"] == test_year]
rounds = sorted(test["round"].unique())
rnd = st.select_slider(f"{test_year} round", rounds, value=rounds[0])
race = test[test["round"] == rnd]
pred = predict_race(model, race).head(8).copy()
actual_podium = set(race[race["podium"] == 1]["driver"])
pred["actual podium"] = pred["driver"].map(lambda d: "🏆" if d in actual_podium else "")
pred["podium prob"] = (pred["podium_prob"] * 100).round(0).astype(int).astype(str) + "%"
st.markdown(f"**{race['event_name'].iloc[0]}** — predicted podium probabilities")
st.dataframe(pred[["driver", "team", "grid", "podium prob", "actual podium"]]
             .reset_index(drop=True), use_container_width=True, hide_index=True)

# --- Championship projection ------------------------------------------------
st.divider()
st.subheader(f"Championship projection — {test_year} (Monte Carlo, from prior form)")
top = champ.head(8).iloc[::-1]
fig = go.Figure(go.Bar(x=top["win_prob"] * 100, y=top["driver"], orientation="h",
                       marker_color="#e2231a",
                       text=[f"{p*100:.0f}%" for p in top["win_prob"]], textposition="outside"))
fig.update_layout(xaxis_title="Title probability (%)", height=380,
                  margin=dict(t=10, l=10, r=30, b=10))
st.plotly_chart(fig, use_container_width=True)
st.caption("Each season simulated 5,000× — finishing orders sampled (Plackett-Luce) "
           "from driver strengths estimated on prior-season points.")
