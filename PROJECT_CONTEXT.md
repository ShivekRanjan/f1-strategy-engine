# F1 Strategy Engine — Project Context

The narrative/decisions document: what the project is, why it's scoped this way,
the decisions behind it, and where things stand. (`CLAUDE.md` holds the operating
rules for the coding agent; `README.md` is the public-facing intro.)

---

## What it is

A **pit-strategy recommendation engine** for Formula 1. Given a race situation,
it recommends what the team should do — when to pit and which tyre compounds to
fit — with quantified uncertainty. The deliberate framing: **not "who will win,"
but "what should the team do."** A decision problem, not a finishing-position
classifier.

## Why this project

A **portfolio piece for a job hunt**, targeting a mix of roles — data science /
ML, ML engineering, possibly full-stack. That drove two choices:

1. Show **depth and judgement**, not `model.fit()`. Every layer has a decision
   you can defend under interview scrutiny.
2. **Decoupled architecture** so it hedges across role types: the same ML core
   demos as a service (ML-eng signal) or a UI (DS signal) with no rework.

## Scope — one flagship, not five projects

Three of the four common "F1 ML ideas" are components of this one engine, not
separate projects:

| Common idea | Where it lives here |
|---|---|
| Tyre Degradation Predictor | The degradation model (Phase 2) — core of the engine |
| Lap Time Predictor | The sequence lap-time model (Phase 2.5) — feeds the simulator |
| Pit Stop Strategy Optimiser | The simulator + optimiser (Phases 3–4) — the payoff |
| Race Outcome / Podium Predictor | Standalone warm-up (Phase A) — the one separate piece |

**Final scope: one flagship app plus one small optional standalone tab.**

## Architecture

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

Modelling logic never goes in `api.py` or the Streamlit app. This decoupling is
the hedge for the undecided role target.

## Tech stack

- **Language/data:** Python 3.12, FastF1, pandas, pyarrow.
- **Modelling (Phase 2+):** scikit-learn, XGBoost, PyTorch (sequence model).
- **App (Phase 5):** FastAPI + Streamlit.
- **Deploy (Phase 6):** Docker + a free host. A live link beats any README.

## Roadmap

- [x] **Phase 0** — repo scaffold, FastF1 caching, tidy lap-level loader, cleaning module + tests
- [ ] **Phase 1** — cleaning validation + fuel correction + EDA ← **current** *(biggest schedule risk)*
- [ ] **Phase 2** — degradation model *(minimum shippable)*
- [ ] **Phase 2.5** — sequence lap-time model (LSTM/ConvLSTM) *(the DL differentiator)*
- [ ] **Phase 3** — Monte Carlo race simulator + safety-car hazard model
- [ ] **Phase 4** — strategy optimiser *(stretch: in-race re-optimisation)*
- [ ] **Phase 5** — FastAPI service + Streamlit UI
- [ ] **Phase 6** — Docker + live deploy + hero-image README
- [ ] **Phase A** *(optional)* — standalone podium predictor

Cut-lines: **Phase 2 = shippable, through Phase 4 = strong, Phases 5–6 = the
hedge that opens the most doors.** Roughly 7–9 weeks part-time.

## Key technical decisions & rules

- **No validation leakage.** Split by race (GroupKFold on race id) AND keep a
  forward-in-time holdout (train ≤2023, test 2024). The decision interviewers
  will probe — be ready to justify it.
- **Model the tyre, not the fuel tank.** Fuel correction in `data/clean.py`
  (done) so the residual signal is tyre degradation, not an emptying tank.
- **Filter SC/VSC/in-out laps before modelling.** Done in `clean_laps`; these
  are the #1 reason degradation curves look like noise.
- **Beat a dumb baseline first.** Per-stint linear degradation before any
  hierarchical/boosted model; a naive predictor before the sequence model.
- **Quantify uncertainty.** The simulator returns distributions, not points.
- **Build the ML before the app.** Never start Phase 5 over a model that doesn't
  work yet.
- **Caching stays on.** FastF1 is slow and rate-limited.

### Known confounders to name in the EDA
- **Track evolution:** grip improves through a session (laps get faster) while
  tyre deg makes them slower — they partially cancel. If corrected curves look
  flat, suspect this before suspecting a bug.
- **Fuel correction is an assumption** (~0.03 s/lap/kg, 110 kg start), surfaced
  as `FuelModel` parameters so it can be cited and sensitivity-tested.

## Data sources

- **FastF1** — telemetry, timing, weather, tyre/stint data from ~2018 on. The
  engine runs on this.
- **Jolpica-F1 API** — historical results back to 1950; Ergast successor (Ergast
  shut down early 2025). FastF1 uses it under the hood.
- **Kaggle "F1 World Championship 1950–2024"** — results-only, no telemetry.
  Fine for the Phase A podium predictor; cannot power the engine.

## Repo structure

```
f1-strategy-engine/
├── CLAUDE.md              # operating rules for Claude Code
├── PROJECT_CONTEXT.md     # this file
├── README.md              # public-facing intro + roadmap
├── pyproject.toml         # deps, split into base / [models] / [app] / [dev]
├── src/f1se/
│   ├── config.py          # FastF1 cache (always on)            (DONE)
│   ├── data/
│   │   ├── loader.py      # FastF1 pull -> tidy lap-level table  (DONE, tested)
│   │   └── clean.py       # filtering + fuel correction          (DONE, tested)
│   ├── models/
│   │   ├── degradation.py # Phase 2 (stub)
│   │   └── lap_time.py    # Phase 2.5 sequence model (stub)
│   ├── sim/
│   │   ├── simulate.py    # Phase 3 (stub)
│   │   └── optimize.py    # Phase 4 (stub)
│   └── api.py             # Phase 5 FastAPI (stub)
├── app/streamlit_app.py   # Phase 5 UI (stub)
├── notebooks/             # EDA goes here
└── tests/                 # no-network smoke tests (passing)
```

## Where things stand

**Phase 0 is complete and on disk:** scaffold, venv (3.12), FastF1 caching, the
tidy loader, and the cleaning + fuel-correction module — all with passing
no-network tests (9 green). Model/sim/api files are documented stubs with fixed
signatures.

**Next up is Phase 1:** pull a season of real data, then validate the cleaning
via EDA — plot fuel-corrected lap time vs tyre age by compound and confirm the
curves actually look like degradation *before* any modelling begins. If they
don't, the fix is in the cleaning, not the model.
