# F1 Strategy Engine

![F1 Strategy Engine](assets/hero.png)

A pit-strategy recommendation engine for Formula 1. Given a race situation, it
recommends **what the team should do** — when to pit and which tyre compounds to
fit — with quantified uncertainty.

The framing is deliberate: *not "who will win," but "what should the team do."*
That's how race engineers actually think — a decision problem under uncertainty,
not a finishing-position classifier.

## How it works

```
FastF1  ->  data (loader, clean)  ->  models (degradation, lap_time)
                                          |
                                          v
                              sim (simulate, optimize)
                                          |
                          +---------------+---------------+
                          v                               v
                   api.py (FastAPI)              app/ (Streamlit)
```

The ML lives in plain functions; the API and UI are thin layers on top, so the
same core can be demoed as a service or a UI with no rework.

## Quickstart

```powershell
py -3.12 -m venv .venv
.venv\Scripts\Activate.ps1
pip install -e ".[dev]"      # base + dev tooling
pytest                       # no-network smoke tests
```

Pull and clean a real race (network; cached after the first call):

```python
from f1se.data.loader import load_session_laps
from f1se.data.clean import clean_laps

laps = load_session_laps(2023, "Spain", "R")   # tidy lap-level table
clean = clean_laps(laps)                        # filtered + fuel-corrected
```

`clean` carries `lap_time_fuel_corr_s` — fuel-burn trend removed, so the residual
is tyre degradation.

## Run the engine (Phase 5)

Build the datasets once (network; cached), then launch the UI or the API:

```powershell
pip install -e ".[app]"
python -m f1se.data.ingest            # dry laps (degradation model)
python -m f1se.data.ingest status     # track status (safety-car calibration)
python -m f1se.data.ingest racelaps   # lap times (pit-loss calibration)

streamlit run app/streamlit_app.py        # interactive strategy explorer
uvicorn f1se.api:app --reload             # REST API at /docs
```

Both are thin layers over `f1se.engine.StrategyEngine`, which assembles the
fitted degradation model, the per-track safety-car and pit-loss calibrations,
the stint guards, and the optimiser. Example API call:

```bash
curl -X POST localhost:8000/recommend \
  -H 'Content-Type: application/json' \
  -d '{"track": "Spanish Grand Prix", "objective": "mean"}'
```

## Deploy (Phase 6)

The three small processed datasets (~1.3 MB) are committed, so the app runs
without hitting FastF1.

**Docker** (the API + UI over one image):

```bash
docker compose up          # UI :8501, API :8000/docs
# or a single service:
docker build -t f1se . && docker run -p 8501:8501 f1se
```

**Free hosting:**
- **Streamlit Community Cloud** (no Docker) — point it at this repo, set the main
  file to `app/streamlit_app.py`; it installs from `requirements.txt`.
- **Render / Fly.io** — deploy the `Dockerfile` (`render.yaml` included; `$PORT`
  is honoured).

## Roadmap

- [x] **Phase 0** — scaffold, FastF1 caching, tidy loader, cleaning + tests
- [ ] **Phase 1** — cleaning validation + fuel correction + EDA ← *current*
- [x] **Phase 2** — tyre degradation model *(minimum shippable)* — within-stint linear baseline (beats naive 23% on held-out races); XGBoost evaluated head-to-head but did **not** beat it (degradation is ~linear in-range), so the simpler model ships
- [ ] **Phase 2.5** — sequence lap-time model (LSTM/ConvLSTM)
- [x] **Phase 3** — Monte Carlo simulator + safety-car hazard model — vectorised race-time simulation with stochastic SC; common-random-numbers paired strategy comparison (distributions, not points)
- [x] **Phase 4** — strategy optimiser — searches stops × pit laps × compound sequence vs stochastic SC; pluggable objective (expected/median/risk-averse p85), paired win-probabilities *(stretch: in-race re-optimisation)*
- [x] **Phase 5** — FastAPI service + Streamlit UI — thin layers over `f1se.engine.StrategyEngine`; recommend/simulate endpoints + interactive strategy explorer
- [ ] **Phase 6** — Docker + live deploy + hero-image README
- [ ] **Phase A** *(optional)* — standalone podium predictor

## Data

[FastF1](https://docs.fastf1.dev/) — telemetry, timing, weather, tyre/stint data
from ~2018 on (backed by the Jolpica-F1 API, the Ergast successor). Caching is
always on; the cache is git-ignored and regenerable.

## License

MIT
