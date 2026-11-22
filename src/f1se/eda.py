"""Phase 1 EDA helpers — kept in the package so they're reusable and testable.

Three analyses, each a pure function over cleaned lap frames:

- :func:`load_clean_races` — pull + clean several races into one labelled frame.
- :func:`fit_stint_slopes` — per (race, driver, stint) linear degradation slope,
  which exposes compound ordering far more cleanly than pooling all laps.
- :func:`fuel_sensitivity` — how the measured degradation slope moves as the
  fuel assumption (``sec_per_kg``) varies.

Analytical note behind the sensitivity check
---------------------------------------------
Within one stint, ``lap_number`` increases 1:1 with ``tyre_life``, and the fuel
correction is ``-sec_per_kg * fuel_mass_kg`` where ``fuel_mass_kg`` falls
linearly at ``start_fuel/total_laps`` kg per lap. So the corrected-pace slope is::

    corrected_slope = raw_slope + sec_per_kg * (start_fuel / total_laps)

i.e. the fuel assumption shifts every stint's measured degradation slope by a
*known, additive* amount ``Δβ · fuel_per_lap``. :func:`fuel_sensitivity` verifies
this empirically — a tidy "I understand exactly how my assumption biases the
result" talking point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from f1se.data.clean import FuelModel, clean_laps
from f1se.data.loader import load_session_laps

STINT_KEYS = ("year", "round", "driver", "stint")
GREEN_FLAG = "1"   # FastF1 track-status code for a fully green lap
SC_CODE = "4"      # ... and for a full safety car


@dataclass(frozen=True)
class RaceSpec:
    """One race to pull, with a short label for plots/tables."""

    year: int
    gp: str | int
    session: str = "R"

    @property
    def label(self) -> str:
        return f"{self.year} {self.gp}"


def load_clean_races(
    specs: list[RaceSpec],
    fuel: FuelModel | None = None,
    *,
    dry_only: bool = False,
) -> pd.DataFrame:
    """Pull and clean several races; return one frame with a ``race`` label column.

    Network on a cache miss; cached thereafter. Races that fail to load are
    skipped with a printed warning rather than aborting the whole batch.
    Pass ``dry_only=True`` for the dry degradation model (drops wet/inter laps).
    """
    frames: list[pd.DataFrame] = []
    for spec in specs:
        try:
            raw = load_session_laps(spec.year, spec.gp, spec.session)
        except Exception as e:  # pragma: no cover - network/availability
            print(f"  ! skipped {spec.label}: {e}")
            continue
        clean = clean_laps(raw, fuel, dry_only=dry_only)
        clean["race"] = spec.label
        frames.append(clean)
        print(f"  loaded {spec.label}: {len(raw)} raw -> {len(clean)} clean laps")

    if not frames:
        raise RuntimeError("no races loaded")
    return pd.concat(frames, ignore_index=True)


def estimate_pit_loss(
    race_laps: pd.DataFrame, *, min_loss: float = 5.0, max_loss: float = 60.0
) -> dict[str, float]:
    """Estimate per-circuit green-flag pit loss (s) from in/out-lap times.

    For each green-flag pit stop, the time lost is approximated as how much
    slower the in-lap and out-lap are than their immediate green neighbours::

        pit_loss = (in_lap - lap_before) + (out_lap - lap_after)

    Stops under safety car are excluded (they're cheaper and unrepresentative).
    Returns the median per ``event_name`` plus a ``"_global"`` median fallback.
    ``race_laps`` needs columns: year, round, event_name, driver, lap_number,
    lap_time_s, is_pit_in_lap, is_pit_out_lap, track_status.
    """
    def _green(row) -> bool:
        return (str(row["track_status"]) == GREEN_FLAG and not bool(row["is_pit_in_lap"])
                and not bool(row["is_pit_out_lap"]) and pd.notna(row["lap_time_s"]))

    losses: dict[str, list[float]] = {}
    for (_, _, _), g in race_laps.groupby(["year", "round", "driver"], observed=True):
        g = g.set_index("lap_number").sort_index()
        event = str(g["event_name"].iloc[0])
        for lap in g.index[g["is_pit_in_lap"].fillna(False)]:
            need = [lap - 1, lap, lap + 1, lap + 2]
            if any(n not in g.index for n in need):
                continue
            before, in_, out_, after = (g.loc[n] for n in need)
            if not bool(out_["is_pit_out_lap"]) or not _green(before) or not _green(after):
                continue
            # Skip any stop whose in- or out-lap wasn't fully green. This used to
            # skip only the full safety car ("4"), so stops under a virtual
            # safety car ("6"/"7") or yellows were scored as green ones, and
            # their slow neutralised laps read as pit-lane time: Monza 2026's
            # VSC stops came out at 45-58 s against ~25 s green (METHODOLOGY §19).
            if set(str(in_["track_status"])) - {GREEN_FLAG} or                     set(str(out_["track_status"])) - {GREEN_FLAG}:
                continue
            if any(pd.isna(x["lap_time_s"]) for x in (in_, out_)):
                continue
            loss = (in_["lap_time_s"] - before["lap_time_s"]) + (out_["lap_time_s"] - after["lap_time_s"])
            if min_loss < loss < max_loss:
                losses.setdefault(event, []).append(float(loss))

    out = {ev: float(np.median(v)) for ev, v in losses.items() if v}
    allv = [x for v in losses.values() for x in v]
    if allv:
        out["_global"] = float(np.median(allv))
    return out


def compound_stint_limits(clean: pd.DataFrame, *, quantile: float = 0.9) -> dict[str, int]:
    """Observed max stint length per compound (a ``quantile`` of stint lengths).

    The tyre 'cliff' is censored out of race data — teams pit before it — so the
    degradation model has no support beyond the stint lengths actually run. These
    limits let the optimiser respect that support (don't extrapolate a too-gentle
    linear fit into long-stint territory the model never saw), which is the
    principled fix for the optimiser over-preferring soft tyres.
    """
    stint_max = clean.groupby(list(STINT_KEYS), observed=True).agg(
        compound=("compound", "first"), max_age=("tyre_life", "max")
    )
    return {
        str(c): int(np.ceil(g["max_age"].quantile(quantile)))
        for c, g in stint_max.groupby("compound", observed=True)
    }


def compound_set_limits(clean: pd.DataFrame, *, quantile: float = 0.999) -> dict[str, int]:
    """How many stints of one compound a driver can actually run in a race.

    A strategy needs a *fresh set* per stint, and the weekend allocation is
    finite — 13 sets (12 on a sprint weekend), of which practice and qualifying
    consume most before the race. The optimiser otherwise has no notion of this
    and can propose a plan requiring more sets of a compound than a team could
    physically have left.

    Rather than invent an allocation model, this reads the constraint off what
    teams actually managed: the observed count of distinct stints per
    driver-race-compound already reflects whatever sets they had. Over 2023-26,
    1 stint is typical, 2 common, 3 rare (~1.4%) and 4 all but unheard of
    (~0.07%), so a high quantile is the right cut — this is a feasibility guard
    against absurd plans, not a tuning knob.
    """
    per = clean.groupby(["year", "round", "driver", "compound"], observed=True)["stint"].nunique()
    return {
        str(c): max(1, int(np.ceil(g.quantile(quantile))))
        for c, g in per.groupby("compound", observed=True)
    }


def fit_stint_slopes(clean: pd.DataFrame, *, min_laps: int = 6) -> pd.DataFrame:
    """Fit a linear degradation slope per stint on fuel-corrected pace.

    One row per (year, round, driver, stint) with enough laps:

    ===============  ========================================================
    slope_s_per_lap  degradation rate (s of fuel-corrected pace gained / lap)
    intercept_s      fitted pace at tyre age 0
    r2               goodness of the linear fit
    n_laps           laps used in the fit
    compound, race   carried through for grouping
    ===============  ========================================================

    Per-stint fitting removes the pace offset between cars/tracks, so a
    compound's *degradation shape* is comparable across the field.
    """
    rows: list[dict] = []
    for keys, grp in clean.groupby(list(STINT_KEYS), observed=True):
        if len(grp) < min_laps:
            continue
        x = grp["tyre_life"].to_numpy(dtype=float)
        y = grp["lap_time_fuel_corr_s"].to_numpy(dtype=float)
        if np.ptp(x) == 0:  # no spread in tyre age -> can't fit a slope
            continue
        slope, intercept = np.polyfit(x, y, 1)
        resid = y - (slope * x + intercept)
        ss_res = float(np.sum(resid**2))
        ss_tot = float(np.sum((y - y.mean()) ** 2))
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
        rows.append(
            {
                **dict(zip(STINT_KEYS, keys)),
                "compound": grp["compound"].iloc[0],
                "race": grp["race"].iloc[0] if "race" in grp else None,
                "n_laps": len(grp),
                "slope_s_per_lap": float(slope),
                "intercept_s": float(intercept),
                "r2": r2,
            }
        )
    return pd.DataFrame(rows)


def compound_degradation_summary(slopes: pd.DataFrame) -> pd.DataFrame:
    """Median degradation slope (+ spread) per compound, ordered fastest-degrading."""
    g = slopes.groupby("compound", observed=True)["slope_s_per_lap"]
    summary = pd.DataFrame(
        {
            "n_stints": g.size(),
            "median_slope": g.median(),
            "q25": g.quantile(0.25),
            "q75": g.quantile(0.75),
        }
    )
    return summary.sort_values("median_slope", ascending=False)


def fuel_sensitivity(
    clean: pd.DataFrame,
    betas: list[float],
    *,
    base_fuel: FuelModel | None = None,
    min_laps: int = 6,
) -> pd.DataFrame:
    """Median per-compound degradation slope as ``sec_per_kg`` varies.

    Recomputes the corrected pace for each ``beta`` directly from the preserved
    raw ``lap_time_s`` and ``fuel_mass_kg`` (which is independent of beta), then
    refits per-stint slopes. Returns long-form rows: (beta, compound, median_slope).
    """
    if "fuel_mass_kg" not in clean.columns:
        raise ValueError("clean frame must carry fuel_mass_kg (run clean_laps first)")

    out: list[dict] = []
    for beta in betas:
        df = clean.copy()
        df["lap_time_fuel_corr_s"] = df["lap_time_s"] - beta * df["fuel_mass_kg"]
        slopes = fit_stint_slopes(df, min_laps=min_laps)
        med = slopes.groupby("compound", observed=True)["slope_s_per_lap"].median()
        for compound, slope in med.items():
            out.append({"sec_per_kg": beta, "compound": compound, "median_slope": slope})
    return pd.DataFrame(out)
