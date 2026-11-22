"""Fresh-tyre pace gaps between compounds, measured on the same lap (METHODOLOGY §20).

NOT USED BY THE ENGINE. Tested and rejected: it halved the error on each race's
measured compound gap and lifted 2023-25 stop counts from 32/54 to 35/54, but
on 2026 it took stop counts from 15/21 to 13/21. A fair soft pace on top of a
soft whose wear is still under-predicted makes soft-heavy two-stops win again.
Kept, like the other rejected alternatives, so the result can be re-run.

The degradation model's per-(track, compound) base pace is a group mean de-aged
at the fitted slope. That makes each compound's base pace depend on *when* in
the race it was run: softs and hards run later than mediums, on a rubbered-in
track and lighter car, so their base pace absorbs track evolution and any fuel
error. Per track it came out noisy: Monza 2026 had a fresh soft 1.45 s/lap
quicker than a medium, Zandvoort 2.07 s/lap slower. Over 35 soft laps, the Monza
figure alone bought the engine a second stop.

Here the gap is measured from cars on *different compounds on the same lap*:

    lap_time = lap effect + driver effect + compound offset + compound slope * age

The lap effect absorbs fuel load, track evolution and any neutralisation; the
driver effect absorbs car speed. Raw lap time is used, so no fuel assumption
enters. A race's offsets are noisy, so the model uses the median over the era's
races, the same at every track.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd

from f1se.models.degradation import DegradationModel

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
REFERENCE = "MEDIUM"


def race_offsets(race: pd.DataFrame) -> dict[str, float]:
    """One race's fresh-tyre offset of each compound vs MEDIUM (s/lap)."""
    r = race.dropna(subset=["lap_time_s", "tyre_life"])
    r = r[r["compound"].isin(COMPOUNDS)].reset_index(drop=True)
    present = set(r["compound"])
    if REFERENCE not in present or len(present) < 2:
        return {}
    others = [c for c in COMPOUNDS if c != REFERENCE and c in present]
    comp = r["compound"].astype(str).to_numpy()
    age = r["tyre_life"].to_numpy(float)
    cols = [
        pd.get_dummies(r["lap_number"].astype(int), dtype=float).to_numpy(),
        pd.get_dummies(r["driver"].astype(str), drop_first=True, dtype=float).to_numpy(),
        np.column_stack([(comp == c).astype(float) for c in others]),
        np.column_stack([(comp == c) * age for c in COMPOUNDS if c in present]),
    ]
    coef, *_ = np.linalg.lstsq(np.hstack(cols), r["lap_time_s"].to_numpy(float), rcond=None)
    start = cols[0].shape[1] + cols[1].shape[1]
    return {c: float(coef[start + i]) for i, c in enumerate(others)}


def season_offsets(laps: pd.DataFrame) -> pd.DataFrame:
    """Per-race offsets for every race in ``laps``: columns year, round, compound, offset."""
    rows = []
    for (year, rnd), race in laps.groupby(["year", "round"], observed=True):
        for c, v in race_offsets(race).items():
            rows.append({"year": int(year), "round": int(rnd), "compound": c, "offset": v})
    return pd.DataFrame(rows, columns=["year", "round", "compound", "offset"])


def pooled_offsets(per_race: pd.DataFrame) -> dict[str, float]:
    """The median race's offset per compound; MEDIUM is 0 by definition."""
    out = {REFERENCE: 0.0}
    if not per_race.empty:
        out.update(per_race.groupby("compound")["offset"].median().astype(float).to_dict())
    return out


def with_compound_offsets(model: DegradationModel, offsets: dict[str, float]) -> DegradationModel:
    """Rebuild each track's base pace as its MEDIUM pace plus the measured offset.

    Only the gaps between compounds change. The track's overall pace level and
    every degradation slope are untouched, so pace-loss evaluation is unaffected.
    """
    if model.group_cols != ("event_name", "compound"):
        return model
    tracks = {k[0] for k in model.intercepts} | set(model.track_base)
    intercepts = dict(model.intercepts)
    for t in tracks:
        ref = model.intercepts.get((t, REFERENCE), model.track_base.get(t))
        if ref is None:
            continue
        for c in COMPOUNDS:
            intercepts[(t, c)] = ref + offsets.get(c, 0.0)
    return replace(model, intercepts=intercepts,
                   meta={**model.meta, "compound_offsets": {k: round(v, 3) for k, v in offsets.items()}})


def apply_era_offsets(model: DegradationModel, era_laps: pd.DataFrame) -> DegradationModel:
    """Fit the era's pooled offsets from ``era_laps`` and apply them to ``model``.

    ``era_laps`` must hold only laps the model may see: in a leave-one-out
    backtest, the predicted race is excluded before this is called.
    """
    return with_compound_offsets(model, pooled_offsets(season_offsets(era_laps)))
