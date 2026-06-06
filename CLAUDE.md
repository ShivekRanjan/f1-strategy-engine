# CLAUDE.md — operating rules for the coding agent

Rules for working in this repo. The narrative/decisions live in
`PROJECT_CONTEXT.md`; the public intro in `README.md`. Read those for *why*;
this file is *how*.

## Environment
- **Python 3.12** in `.venv` (pinned `>=3.12,<3.13` for wheel availability across
  torch / xgboost; 3.14 has no stable wheels yet).
- Activate: `.venv\Scripts\Activate.ps1` (PowerShell) or call
  `.venv\Scripts\python.exe` directly.
- Install: `pip install -e .` (base) or `pip install -e ".[models,app,dev]"`.

## Architecture rule — keep it decoupled
Modelling logic lives in `src/f1se/{models,sim}` as plain functions.
**`api.py` and `app/streamlit_app.py` are thin layers — never put modelling
logic in them.** This decoupling is the hedge across DS / ML-eng / full-stack
role targets; don't collapse it.

## Hard rules
- **No validation leakage.** Never use a shuffled `train_test_split` on laps —
  they're correlated within a race. Split by race (GroupKFold on race id) AND
  keep a forward-in-time holdout (train ≤2023, test 2024).
- **Model the tyre, not the fuel tank.** Fuel correction happens in
  `data/clean.py` (done). Degradation models consume `lap_time_fuel_corr_s`,
  never raw `lap_time_s`.
- **Filter SC/VSC/in-out laps before any modelling.** `clean_laps` does this;
  don't model on un-cleaned data.
- **Beat a dumb baseline first.** Linear per-stint degradation before boosted/
  hierarchical; a naive predictor before the sequence model.
- **Quantify uncertainty.** The simulator returns distributions, not points.
- **Build the ML before the app.** No Phase 5 over a model that doesn't work.
- **Caching stays on.** Always go through `f1se.config.enable_cache`.

## Testing
- `pytest` runs the no-network suite by default (`-m "not network"` via config).
- Network/integration tests must be marked `@pytest.mark.network`.
- Cleaning/loader logic is tested against synthetic FastF1-shaped fixtures
  (`tests/conftest.py`) — keep that path green and fast.

## Commits
- Small, phase-scoped commits. Don't commit the cache or data artifacts
  (`.gitignore` covers them).
