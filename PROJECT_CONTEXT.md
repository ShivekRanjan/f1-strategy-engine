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
classifier. It later grew into a full **F1 OS** around that engine (see the
scope section) — but the strategy engine remains the identity and the moat.

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

**Original scope: one flagship app plus one small optional standalone tab.**

### The later expansion: the F1 OS (July 2026)

Once the engine was validated and deployed-ready, scope was deliberately
expanded into a full **"F1 OS"** — one app for everything F1 — as a product
decision (the user's call, made knowing the breadth-vs-depth trade-off). The
guardrail that kept it from diluting the ML story: every new section **reuses a
validated model** rather than adding a shallow feature — standings carry the
championship simulator's title odds, the Race Hub shows each race's pre-race
podium prediction scored against reality, the calendar surfaces the next-race
prediction. The two genuinely new surfaces (news, calendar) are the only ones
that touch the network at runtime, and the "hard" ideas were scoped honestly:
news is RSS headlines + link-out only (no scraping), and live timing was *not*
faked — real-time data only exists during a session, so the app counts down and
replays instead, and says so in the UI.

## Architecture

```
FastF1  ->  data (loader, clean)  ->  models (degradation, era shrinkage,
                                          |    cliff/thermal/overtaking priors,
                                          v    LSTM next-lap forecaster)
                              sim (simulate, optimize, inrace, duel)
                                          |
                              engine.StrategyEngine (orchestration)
                                          |
   standalone/ (outcome, standings,       |
   races, profiles, news, schedule) --> api.py (FastAPI, thin)
                                          |  HTTP / JSON
                              frontend/ (React + Vite + Tailwind, 9 sections)
```

Modelling logic never goes in `api.py` or the React frontend. This decoupling is
the hedge for the undecided role target — and it paid off twice: the UI was
swapped from Streamlit to a React client with zero engine changes, and the F1 OS
sections were added as thin `standalone/` modules + endpoints + views without
touching the engine either.

## Tech stack

- **Language/data:** Python 3.12, FastF1, pandas, pyarrow.
- **Modelling (Phase 2+):** scikit-learn, XGBoost, PyTorch (sequence model —
  training only; inference is a torch-free numpy export).
- **App (Phase 5):** FastAPI service + a React + Vite + TypeScript + Tailwind
  frontend (Recharts for charts); feedparser for the news RSS.
- **Deploy (Phase 6):** Docker / Render (API) + Vercel-Netlify (frontend).

## Roadmap

**Engine (all done):**

- [x] **Phase 0** — repo scaffold, FastF1 caching, tidy lap-level loader, cleaning module + tests
- [x] **Phase 1** — cleaning validation + fuel correction + EDA
- [x] **Phase 2** — degradation model (linear beat XGBoost on leakage-safe folds)
- [x] **Phase 2.5** — LSTM next-lap model (+8.5% vs persistence; exported torch-free)
- [x] **Phase 3** — Monte Carlo race simulator + per-track safety-car hazard
- [x] **Phase 4** — strategy optimiser, incl. in-race re-optimisation + undercut duel
- [x] **Phase 5** — FastAPI service + React (Vite/TS/Tailwind) frontend (replaced Streamlit)
- [x] **Phase A** — standalone podium predictor + championship simulator
- [x] **2026 validation** — Austria backtest, season-wide leave-one-race-out,
      recency weighting, overtaking + thermal priors (over-stopping fixed 4/8 → 7/8)

**F1 OS expansion (all done, July 2026):**

- [x] Standings + live title odds · Race Hub · Drivers & Teams profiles
- [x] News (RSS) · Calendar + live countdown + next-race prediction
- [x] Grouped-sidebar OS shell, hash routing (deep links), code-split bundles

**Remaining (Phase 6, user-side):** live deploy (Render API + Vercel frontend —
configs are committed and ready), hero GIF/screenshot for the README, GitHub
About/topics/pin.

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
├── README.md              # public-facing intro
├── pyproject.toml         # deps, split into base / [models] / [app] / [dev]
├── Dockerfile / render.yaml / docker-compose.yml   # deploy configs
├── src/f1se/
│   ├── config.py          # FastF1 cache (always on)
│   ├── data/              # loader, clean (fuel correction), ingest
│   ├── models/            # degradation, era shrinkage, cliff/thermal/
│   │                      #   overtaking priors, LSTM lap_time (+ the rejected
│   │                      #   alternates: boosted, poly, evo — kept as receipts)
│   ├── sim/               # simulate, optimize, inrace, duel, safety_car
│   ├── engine.py          # StrategyEngine — the one orchestration layer
│   ├── standalone/        # results-only features: outcome (podium +
│   │                      #   championship), standings, races (Race Hub),
│   │                      #   profiles, news (RSS), schedule (calendar)
│   └── api.py             # FastAPI — thin over engine + standalone
├── frontend/              # React + Vite + TS UI — 9 views, grouped sidebar,
│                          #   hash routing, code-split chunks
├── data/processed/        # small committed datasets (app + CI run offline)
├── analysis/              # EDA + phase scripts + 2026 backtests
├── docs/METHODOLOGY.md    # the receipts — every accepted/rejected model
└── tests/                 # 136 no-network tests; network ones opt-in
```

## Where things stand

**Everything through the F1 OS is built, tested, and pushed** (July 2026): the
validated engine (degradation + priors + LSTM + Monte-Carlo optimiser), the
FastAPI service, and the nine-section React app — 136 no-network tests green,
CI runs both the Python suite and the frontend build.

The 2026 season is handled across the regulation reset (era shrinkage +
recency weighting) and validated the honest way: a leave-one-race-out season
backtest, whose over-stopping miss was root-caused to **weather** and fixed
with the thermal prior (stop-count match 4/8 → 7/8; Canada remains the one
documented miss).

**What remains is Phase 6, and it's user-side:** deploy the two pieces (Render
API + Vercel frontend — configs committed), record the README hero GIF, and
fill in the GitHub About panel. No engineering blockers.
