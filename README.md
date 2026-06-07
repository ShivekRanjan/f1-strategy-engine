# F1 Strategy Engine

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

## Roadmap

- [x] **Phase 0** — scaffold, FastF1 caching, tidy loader, cleaning + tests
- [ ] **Phase 1** — cleaning validation + fuel correction + EDA ← *current*
- [x] **Phase 2** — tyre degradation model *(minimum shippable)* — within-stint linear baseline (beats naive 23% on held-out races); XGBoost evaluated head-to-head but did **not** beat it (degradation is ~linear in-range), so the simpler model ships
- [ ] **Phase 2.5** — sequence lap-time model (LSTM/ConvLSTM)
- [ ] **Phase 3** — Monte Carlo simulator + safety-car hazard model
- [ ] **Phase 4** — strategy optimiser
- [ ] **Phase 5** — FastAPI service + Streamlit UI
- [ ] **Phase 6** — Docker + live deploy + hero-image README
- [ ] **Phase A** *(optional)* — standalone podium predictor

## Data

[FastF1](https://docs.fastf1.dev/) — telemetry, timing, weather, tyre/stint data
from ~2018 on (backed by the Jolpica-F1 API, the Ergast successor). Caching is
always on; the cache is git-ignored and regenerable.

## License

MIT
