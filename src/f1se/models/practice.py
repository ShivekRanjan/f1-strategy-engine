"""Practice long runs as wear evidence for a track not yet raced (METHODOLOGY §22).

A track's first race of a season has no laps of its own in the target era, so
its wear rate is borrowed (models/era.py). That weekend's practice long runs
come before the race, and blending them in cut the wear-prediction error by 10%
on 2026, with both constants fitted on 2023-25 only
(analysis/practice_wear_test.py):

    slope = (n * (practice - OFFSET) + K * fallback) / (n + K)

``n`` is the practice long-run laps on the compound. Practice reads high
(teams push harder than in a race), hence ``OFFSET``. ``K`` came out at the
same 150 laps the era blend already uses.
"""

from __future__ import annotations

from dataclasses import replace

import pandas as pd

from f1se.models.degradation import DegradationModel

OFFSET = 0.048     # s/lap; practice minus race, median over 2023-25
K = 150.0          # laps; chosen on 2023-25 from a grid, equal to the era blend's k
MAX_ABS = 0.5      # a practice stint slope beyond this is a timing artefact


def with_practice(model: DegradationModel, practice: pd.DataFrame, track: str) -> DegradationModel:
    """Blend this weekend's practice long runs into ``track``'s slopes."""
    if practice is None or practice.empty or model.group_cols != ("event_name", "compound"):
        return model
    p = practice[(practice["event_name"] == track) & (practice["slope_s_per_lap"].abs() < MAX_ABS)]
    if p.empty:
        return model
    slopes = dict(model.slopes)
    used = {}
    for comp, g in p.groupby("compound", observed=True):
        n = float(g["n_laps"].sum())
        fallback = model.raw_slope(str(comp), track)
        fp = float(g["slope_s_per_lap"].median()) - OFFSET
        slopes[(track, str(comp))] = (n * fp + K * fallback) / (n + K)
        used[str(comp)] = int(n)
    return replace(model, slopes=slopes,
                   meta={**model.meta, "practice": {**model.meta.get("practice", {}), track: used}})


def apply_for_unraced(model: DegradationModel, practice: pd.DataFrame | None,
                      era_laps: pd.DataFrame, season: int) -> DegradationModel:
    """Apply practice to every ``season`` track that has no race laps of its own yet."""
    if practice is None or practice.empty:
        return model
    raced = set(era_laps.loc[era_laps["year"] == season, "event_name"].astype(str))
    for track in practice.loc[practice["year"] == season, "event_name"].astype(str).unique():
        if track not in raced:
            model = with_practice(model, practice[practice["year"] == season], track)
    return model
