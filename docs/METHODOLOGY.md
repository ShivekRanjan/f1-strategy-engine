# Methodology — how every number was earned

This project's claim is not "the model is accurate"; it's that **every number is
either calibrated from data or explicitly labelled as an assumption** — and the
models that failed to earn their place are documented rather than deleted. This
page is the evidence. Each section is reproducible from a script in
[`analysis/`](../analysis/).

---

## 1. Validation that can't cheat

Laps within a race are near-duplicates (same car, track, weather, fuel run). A
shuffled train/test split puts lap 30 of a race in training and lap 31 in test,
and every score inflates. All evaluation here uses two splits instead
([`f1se/validation.py`](../src/f1se/validation.py), tested):

- **GroupKFold by race** — no race ever spans train and test;
- **forward-in-time holdout** — train on past seasons, test on a future one.

This discipline caught a real bug immediately: the first degradation model
predicted *absolute* lap time and scored 0.69s MAE in-sample — but **7.5s on
held-out races**, because base pace is track-specific and an unseen track has no
intercept. The fix (predict *within-stint pace loss*, leaving base pace to the
simulator) is what generalises: 0.40s on races the model never saw.

## 2. XGBoost lost to a straight line — and why that's the right answer

Identical leakage-safe folds, identical within-stint target, identical metric:

| Model | Pace-loss MAE, held-out races |
|---|---|
| Naive (no degradation) | 0.526 s |
| **Linear per (track, compound)** | **0.404 s** |
| XGBoost | 0.422 s |

The learnt-curve plot shows why: the boosted model tracks the line where data is
dense (tyre age ≤ ~30 laps), then chases noise in sparse, confounded late-stint
laps. Degradation is ~linear in the observed range, so added flexibility buys
variance, not signal. A synthetic test with genuinely curved degradation confirms
the comparison *can* detect curvature when it exists — there just isn't any here.

![Linear vs boosted degradation curves](../analysis/figures/phase2_boosted_curves.png)

*Reproduce: `analysis/phase2_boosted.py`*

## 3. The tyre "cliff" is censored out of every public dataset

Teams pit before tyres fall off the cliff, so race data — anyone's race data —
contains almost no cliff laps. Fitting a quadratic anyway makes the model
**worse** on the forward holdout (0.497 vs 0.484 MAE) and produces physically
backwards curvature (hards come out *concave*). Practice sessions don't rescue
it: their long runs are no longer than race stints, and fuel loads are unknown.

So the cliff ships as an explicit, tunable **physical prior**
([`models/cliff.py`](../src/f1se/models/cliff.py)) — extra pace loss beyond a
per-compound onset age — with the same epistemic status as the fuel coefficient:
an assumption, labelled, adjustable, never presented as a measurement. Paired
with data-driven per-compound stint-length caps, it shifted the Spanish GP
recommendation from a 42-soft-lap plan to an 18-soft-lap plan whose soft stint
ends exactly at the cliff onset — the realistic behaviour.

*Reproduce: `analysis/phase2_forward.py` (quadratic test), `analysis/phase4_optimize.py` (effect on strategy)*

## 4. The fuel assumption survived its audit

Fuel burn makes every lap faster as the race runs, masking tyre degradation. The
correction assumes **0.03 s of lap time per kg of fuel** — a rule of thumb worth
auditing, since the measured degradation slope roughly *doubles* across the
plausible β range. Two checks:

1. **Sensitivity is analytic**: within a stint, the corrected slope shifts by
   exactly Δβ · (fuel burned per lap) ≈ Δβ · 1.69 — verified empirically, so the
   assumption's influence is known, not vague.
2. **Calibration**: backing an *effective* coefficient out of the net race-lap
   pace trend (identified from pit-stop pace jumps, where tyre age resets but
   fuel keeps falling) across 43 races gives **median β = 0.031** — right on the
   physics value. The per-race spread is wide (evolution and lift-and-coast
   confound any single race), so only the pooled median is trusted.

![Implied fuel coefficient per race](../analysis/figures/phase2_fuel_calibration.png)

*Reproduce: `analysis/phase1_eda.py` (sensitivity), `analysis/phase2_calibrate.py` (calibration)*

## 5. A decomposition that didn't work — kept as a diagnostic

An attempt to separate *track evolution* (grip improving as the circuit rubbers
in) from degradation failed honestly: a linear-in-lap evolution term is
**collinear with the fuel correction** (fuel mass is also linear in lap), so the
fitted "evolution" absorbed fuel miscalibration and late-race management instead
— coming out *positive* (pace fading), the opposite of rubber-in. The module
survives as a documented diagnostic of the net lap-trend, not as a correction.
The lesson generalised: for this data, the within-stint bundled estimate is the
*right* predictive object for a strategy simulator anyway.

*Reproduce: `analysis/phase2_evolution.py`*

## 6. Safety cars and pit loss are measured, not assumed

Safety cars dominate outcome uncertainty — the race-time distribution is
multi-modal, with clusters ~2 minutes apart corresponding to 0/1/2 SC periods:

![Strategy outcome distributions](../analysis/figures/phase3_strategy_distributions.png)

The hazard model was therefore calibrated from 76 races of per-lap track-status
data: **0.0105 SC triggers per lap, mean duration 4.1 laps** (the literature
default of 0.013/4 was close). The real gain is **per-circuit** rates with
partial pooling — Australia/Canada/Qatar average ~1.5 SC periods per race while
Spain had zero in 2023–24, and shrinkage keeps the zero-observation tracks at a
sensible non-zero hazard. The same treatment gives per-circuit pit loss from
in/out-lap deltas (Spa 19.0s … Spain 23.4s … Singapore 29.4s — matching the
known pit-lane geometry).

*Reproduce: `analysis/phase3_sc_calibrate.py`*

## 7. 2026: modelling across a regulation reset

2026 rewrote the cars, so pre-2026 models are biased and 2026 data is scarce —
a textbook bias–variance trade resolved component-wise:

- **Transferable components** (pit-lane loss, SC hazard, fuel physics) pool all
  seasons.
- **Regime-sensitive components** (base pace, degradation) use a **shrinkage
  estimator**: each per-group slope is a precision-weighted blend of the 2026
  estimate and the pre-2026 prior, converging to 2026 truth as races accumulate.

Measured on real 2026 laps, the regime shift is large and the fix works:

| Degradation model | Pace-loss MAE on 2026 laps | vs naive |
|---|---|---|
| Naive (no degradation) | 0.590 s | — |
| Pre-2026 (old cars) | 0.573 s | **+3%** — barely useful |
| **Shrunk (2026-aware)** | **0.495 s** | **+16%** |

The championship projection applies the same humility: with only a handful of
2026 rounds, each simulation **bootstraps driver strength** from the races seen
so far, so a dominant leader shows ~99% — not a dishonest 100% — and close form
yields genuinely open odds.

*Reproduce: `analysis/phase_2026_validation.py`*

---

### The pattern

Five times in this project, the sophisticated option (boosted trees, a fitted
cliff, a recalibrated fuel coefficient, an evolution decomposition, trusting
six races of 2026 form) was built, evaluated honestly, and **rejected in favour
of a simpler, better-validated alternative** — with the evidence kept. That's
the methodology: parsimony plus domain knowledge, verified at every step.
