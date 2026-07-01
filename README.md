# 🏎️ F1 Strategy Engine

[![CI](https://github.com/ShivekRanjan/f1-strategy-engine/actions/workflows/ci.yml/badge.svg)](https://github.com/ShivekRanjan/f1-strategy-engine/actions/workflows/ci.yml)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue)](pyproject.toml)
[![Frontend: React + Vite](https://img.shields.io/badge/frontend-React%20%2B%20Vite-61dafb?logo=react&logoColor=white)](frontend/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

**Not *who will win* — *what should the team do*.** A pit-strategy decision engine
for Formula 1: given a race situation, it recommends when to stop and which tyre
compounds to fit, with quantified uncertainty — including the ongoing **2026
season**, modelled across the regulation reset.

<!-- TODO: add a screenshot/GIF of the React UI to assets/ (Live Race tab, lap slider). -->

A **React + Vite** frontend over a **FastAPI** service wrapping the engine — four views:

| Tab | What it does |
|---|---|
| 🏁 **Strategy** | Searches ~1,000+ pit strategies per race via Monte Carlo (stochastic safety cars, calibrated per circuit) and recommends the best plan — with the honest spread: *typical* race vs *bad-luck* race, and whether the call is clear-cut or a coin-flip |
| 🆚 **Undercut** | The two-car question: with a rival *N* seconds ahead, should you pit now to undercut or hold and cover? Models the cumulative-time crossover and returns the verdict with how many seconds it gains |
| 🏆 **Outcome Predictor** | Podium probabilities (forward-tested, never a shuffled split) + a live championship projection that **bootstraps driver-strength uncertainty** so a 6-race leader doesn't show a dishonest 100% |
| 🔴 **Live Race** | Replays any race lap-by-lap and **re-optimises the remaining strategy from the current state** each lap — the same engine call a live-timing feed drives on race day. Also shows a **next-lap pace nowcast** from the Phase 2.5 LSTM |

## The models I built, tested, and kept the receipts for

Every number in this engine was either calibrated from data or explicitly
labelled as an assumption. Sophisticated models were adopted **only when they
beat a simpler baseline on a leakage-safe split** — most didn't, and those are
documented rather than deleted; one did, and it's here too:

| Finding | Evidence |
|---|---|
| **XGBoost lost to a linear baseline** on held-out races (0.42 vs 0.40 MAE) — tyre degradation is ~linear in the observed range, and the flexible model chased noise in sparse late-stint laps | identical leakage-safe folds, target, and metric for both |
| **The tyre "cliff" cannot be fitted from race data** — teams pit before it, so it's censored out of every public dataset. A quadratic fit was *worse* out-of-sample. Ships as an explicit, tunable physical prior instead | forward holdout, train ≤2023 → test 2024 |
| **The 0.03 s/kg fuel assumption survived calibration** — backing an effective coefficient out of 43 races' pace trends gives a median of 0.031 | per-race implied-β distribution |
| **Validation is leakage-safe by construction** — laps within a race are near-duplicates, so splits are GroupKFold-by-race plus a forward-in-time holdout; a shuffled split would inflate every score | `f1se/validation.py`, tested |
| **2026's regulation reset breaks old models** — a pre-2026 degradation model barely beats "no degradation" on 2026 laps (+3%); blending 2026 data with the old prior via shrinkage recovers the signal (+16%) | `analysis/phase_2026_validation.py` |
| **An LSTM *did* earn its place** — for next-lap pace forecasting it beats persistence by ~8.5% (0.306 vs 0.335s MAE on held-out 2025) by damping per-lap noise and anticipating tyre warm-up. The one case where complexity won, on the same footing — and it's live in the app (exported torch-free to a 28 KB numpy artifact) | `analysis/phase2_5_sequence.py` |
| **Validated on a race the models never saw** (Austrian GP 2026, in no training data) — LSTM nowcast **+18%** vs persistence, podium model **2/3** vs the grid's 1/3; a strategy miss surfaced (and fixed) a real degradation gap, and the model even flagged the underused softs a driver called out post-race | `analysis/backtest_austria_2026.py` |

Full receipts — figures, numbers, and how to reproduce each one — in
**[docs/METHODOLOGY.md](docs/METHODOLOGY.md)**.

## How it works

```
FastF1 ──▶ data (load, clean, fuel-correct) ──▶ models (degradation, era-shrinkage, cliff prior,
                                                       │         LSTM next-lap forecaster)
        calibrations (safety-car hazard, pit loss) ──▶ sim (Monte Carlo, optimiser, in-race)
                                                       │
                                          engine.StrategyEngine (orchestration)
                                                       │
                                            api.py (FastAPI, thin)
                                                       │  HTTP / JSON
                                          frontend/ (React + Vite + Tailwind)
```

The modelling lives in plain, tested functions; `api.py` is a thin wrapper over
one `StrategyEngine`, and the React frontend is a pure client of that API. Per-
circuit safety-car risk and pit loss are **measured** from 76 races of track-
status and in/out-lap data — not assumed.

## Quickstart

Two processes — the API and the frontend. **Backend:**

```bash
py -3.12 -m venv .venv && .venv\Scripts\Activate.ps1   # (or python3.12 -m venv on unix)
pip install -e ".[app,dev]" scikit-learn   # scikit-learn powers the outcome predictor
pytest                                     # 104 no-network tests
uvicorn f1se.api:app --reload              # REST API + Swagger at localhost:8000/docs
pip install -e ".[models]"                 # optional: torch etc. to retrain the LSTM (heavy)
```

**Frontend** (Node 18+), in a second terminal — the committed ~1.5 MB datasets make it run instantly:

```bash
cd frontend
npm install
npm run dev                                # UI at localhost:5173, talks to the API above
```

Rebuild the datasets from source (network; FastF1-cached, resumable):

```bash
python -m f1se.data.ingest               # dry laps (degradation model)
python -m f1se.data.ingest status        # track status (safety-car calibration)
python -m f1se.data.ingest racelaps      # pit-loss calibration
python -m f1se.standalone.results        # race results (outcome predictors)
```

Example API call:

```bash
curl -X POST localhost:8000/recommend -H 'Content-Type: application/json' \
  -d '{"track": "Japanese Grand Prix", "objective": "p85"}'
```

## Deploy

Full stack locally in one command: **`docker compose up`** → API on `:8000`,
frontend on `:5173`.

For a hosted demo, deploy the two pieces independently:

- **API** → Render (`render.yaml` + `Dockerfile` included; injects `$PORT`, serves
  `f1se.api:app`). Set `F1SE_CORS_ORIGINS` to your frontend origin.
- **Frontend** → Vercel or Netlify (Vite static build). Set the project root to
  `frontend/` and `VITE_API_BASE` to the deployed API URL. See
  [frontend/README.md](frontend/README.md).

## Data

[FastF1](https://docs.fastf1.dev/) timing, tyre, and track-status data,
2023–2026. The raw cache is git-ignored; the small processed datasets are
committed so the API and CI run without network.

## License

[MIT](LICENSE)
