# Changelog

## v1.0.0 — the F1 OS (2026-07-10)

The project's first tagged release: the pit-strategy engine, grown into a full
F1 OS, deployed and synced through the 2026 British GP (round 9).

### The engine (the moat)
- Tyre-degradation model — per-circuit/compound, era-shrunk across the 2026
  regulation reset, recency-weighted for mid-season upgrades.
- Monte-Carlo race simulator + strategy optimiser (coarse-to-fine search),
  per-circuit safety-car hazard and pit loss **measured** from 78+ races.
- Labelled priors for what data can't show: tyre cliff, thermal (track-temp)
  degradation, track-position cost per stop, and **compound censoring**
  (avoidance-aware stint caps + slope/base repair).
- LSTM next-lap forecaster (+8.5% vs persistence), exported torch-free.
- Podium + championship models, always validated forward-in-time; title odds
  bootstrap driver-strength uncertainty.

### The OS
Home (next race + countdown + the model's podium call) · Strategy · Undercut ·
Calendar · Race Hub (pre-race prediction vs actual result, scored hit@3) ·
Live Race replay with LSTM nowcast · Standings (sprint-inclusive, live
title odds, one-click refresh from FastF1) · Drivers & Teams · News · About.

### Numbers (leakage-safe, forward-tested)
- Strategy stop-count: 8/9 vs the field's dominant choice, 7/9 vs the winner
  (leave-one-race-out over 2026).
- Podium model ROC-AUC 0.93 (forward split); degradation MAE 0.069 s/lap on
  unseen races.
- 146 no-network tests; CI runs the Python suite and the frontend build.

Full receipts and the accepted/rejected-model history: `docs/METHODOLOGY.md`.
