"""Clean tidy lap data so the residual signal is *tyre degradation*.

Two jobs, kept as separate, independently-testable functions:

1. :func:`filter_racing_laps` — drop laps that don't represent a car running
   freely on a green track: in/out laps, safety-car / VSC / red-flag laps,
   FastF1-flagged inaccurate laps, and null lap times. **This is the step that
   makes degradation curves legible** — un-filtered SC laps are the number-one
   reason "lap time vs tyre age" looks like noise.

2. :func:`add_fuel_correction` — remove the fuel-burn trend. Cars get lighter
   and faster all race; left in, that trend swamps tyre degradation. We model
   it explicitly and subtract it, leaving a residual that should *rise* with
   tyre age (the thing we actually want to model in Phase 2).

Both are wrapped by :func:`clean_laps`. Every assumption is a named, overridable
parameter — see :class:`FuelModel` — so it can be cited and sensitivity-tested
rather than buried as a magic number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# FastF1 track-status digit codes. A lap is fully green only if its status is
# exactly "1"; anything else means a yellow/SC/VSC/red condition touched the lap.
GREEN_FLAG = "1"

# Slick (dry) compounds. The Phase 1 EDA showed INTERMEDIATE/WET stints produce
# strongly *negative* fuel-corrected slopes — that's a drying track, not tyre
# degradation. The dry strategy model trains on slicks only.
DRY_COMPOUNDS = ("SOFT", "MEDIUM", "HARD")


@dataclass(frozen=True)
class FuelModel:
    """Assumptions behind the fuel correction.

    These are *rules of thumb*, deliberately surfaced as parameters so they can
    be justified ("~0.03 s/lap per kg is the commonly cited figure") and varied
    in a sensitivity check rather than hard-coded.

    Attributes
    ----------
    sec_per_kg
        Lap-time penalty per kilogram of fuel carried (seconds). Default 0.03.
    start_fuel_kg
        Fuel mass at the start of the race (regulation max is 110 kg).
    """

    sec_per_kg: float = 0.03
    start_fuel_kg: float = 110.0

    def fuel_mass_kg(self, lap_number: pd.Series, total_laps: int) -> pd.Series:
        """Estimated fuel remaining at the *start* of ``lap_number``.

        Assumes a linear burn: full tank on lap 1, ~empty on the final lap.
        """
        laps_remaining = (total_laps - lap_number).clip(lower=0)
        return self.start_fuel_kg * laps_remaining / total_laps

    def penalty_s(self, lap_number: pd.Series, total_laps: int) -> pd.Series:
        """Seconds of lap time attributable to fuel load on ``lap_number``."""
        return self.sec_per_kg * self.fuel_mass_kg(lap_number, total_laps)


def filter_racing_laps(
    df: pd.DataFrame,
    *,
    require_green: bool = True,
    require_accurate: bool = True,
    drop_pit_laps: bool = True,
    dry_only: bool = False,
) -> pd.DataFrame:
    """Keep only laps that represent a car running freely on a clean track.

    Parameters
    ----------
    df
        Tidy lap frame (see :data:`f1se.data.loader.LAP_SCHEMA`).
    require_green
        Drop any lap whose ``track_status`` is not exactly green (``"1"``),
        i.e. anything touched by yellow / safety car / VSC / red flag.
    require_accurate
        Drop laps FastF1 flagged ``is_accurate == False`` (timing glitches,
        missing sectors, etc.).
    drop_pit_laps
        Drop in-laps and out-laps (their times include pit travel/stop and are
        meaningless for degradation).
    dry_only
        Keep only slick compounds (:data:`DRY_COMPOUNDS`). Off by default so EDA
        can still see wet/inter; the dry degradation model turns it on.

    Returns
    -------
    pandas.DataFrame
        Filtered copy. A boolean is never silently coerced — missing flags are
        treated conservatively (kept only when explicitly ``True``).
    """
    mask = df["lap_time_s"].notna() & (df["lap_time_s"] > 0)

    if drop_pit_laps:
        mask &= ~df["is_pit_out_lap"].fillna(False)
        mask &= ~df["is_pit_in_lap"].fillna(False)

    if require_green:
        mask &= df["track_status"].astype("string") == GREEN_FLAG

    if require_accurate and "is_accurate" in df.columns:
        mask &= df["is_accurate"].fillna(False)

    if dry_only:
        mask &= df["compound"].isin(DRY_COMPOUNDS)

    return df.loc[mask].reset_index(drop=True)


def drop_stint_outliers(
    df: pd.DataFrame,
    *,
    group_cols: tuple[str, ...] = ("year", "round", "driver", "stint"),
    n_mad: float = 5.0,
) -> pd.DataFrame:
    """Drop within-stint lap-time outliers using a robust (MAD) threshold.

    Even after status filtering, the odd lap is wrecked by traffic or a lock-up.
    We flag laps more than ``n_mad`` median-absolute-deviations above the stint
    median (only slow outliers — a freakishly *fast* lap is rarely a data error
    worth discarding, and clipping both tails would bias the degradation slope).
    """
    if df.empty:
        return df.copy()

    # Vectorised via transform (no groupby.apply) — avoids operating on the
    # grouping columns and keeps the mask index-aligned to df.
    g = df.groupby(list(group_cols))["lap_time_s"]
    med = g.transform("median")
    mad = g.transform(lambda s: (s - s.median()).abs().median())
    # 1.4826 scales MAD to a std-equivalent for ~normal data.
    threshold = med + n_mad * 1.4826 * mad
    # Keep when within threshold, or when MAD is degenerate (0 / NaN: too few
    # distinct laps to judge an outlier — don't drop).
    keep = (df["lap_time_s"] <= threshold) | (mad == 0) | mad.isna()
    return df.loc[keep].reset_index(drop=True)


def add_fuel_correction(
    df: pd.DataFrame,
    fuel: FuelModel | None = None,
    *,
    total_laps: int | None = None,
    group_cols: tuple[str, ...] = ("year", "round"),
) -> pd.DataFrame:
    """Add ``fuel_mass_kg`` and ``lap_time_fuel_corr_s`` columns.

    The corrected lap time removes the fuel penalty, expressing every lap as if
    the car were on an empty tank (end-of-race reference)::

        lap_time_fuel_corr_s = lap_time_s - sec_per_kg * fuel_mass_kg

    so that the remaining lap-to-lap rise reflects tyre degradation, not an
    emptying tank.

    Parameters
    ----------
    fuel
        Fuel assumptions. Defaults to :class:`FuelModel` (0.03 s/kg, 110 kg).
    total_laps
        Race distance in laps. If ``None``, inferred per race group as the max
        observed ``lap_number`` (works on cleaned data where the leader runs
        the full distance).
    group_cols
        Columns identifying a single race, used when inferring ``total_laps``.
    """
    fuel = fuel or FuelModel()
    out = df.copy()

    if total_laps is not None:
        n = pd.Series(total_laps, index=out.index)
    else:
        n = out.groupby(list(group_cols))["lap_number"].transform("max")

    out["fuel_mass_kg"] = fuel.fuel_mass_kg(out["lap_number"], n)
    out["lap_time_fuel_corr_s"] = out["lap_time_s"] - fuel.sec_per_kg * out["fuel_mass_kg"]
    return out


def clean_laps(
    df: pd.DataFrame,
    fuel: FuelModel | None = None,
    *,
    drop_outliers: bool = True,
    total_laps: int | None = None,
    dry_only: bool = False,
) -> pd.DataFrame:
    """Full cleaning pipeline: filter → (outliers) → fuel-correct.

    This is the single entry point Phase 1 EDA and Phase 2 modelling should call.
    Pass ``dry_only=True`` for the dry degradation model (drops wet/inter laps).
    """
    out = filter_racing_laps(df, dry_only=dry_only)
    if drop_outliers:
        out = drop_stint_outliers(out)
    out = add_fuel_correction(out, fuel, total_laps=total_laps)
    return out
