"""Avoidance-aware stint caps — the censoring guardrail for compound choice.

The Silverstone 2026 failure mode: teams avoided softs (13 laps all season,
longest stint 13), so the fitted soft slope there is fiction inherited from the
old-regs prior — and the optimiser happily planned 21-lap soft stints no 2026
car has ever attempted. The censoring can't be fixed by re-anchoring the slope:
the *global* 2026 soft slope is itself contaminated (softs only appear in short
end-of-race dashes, so pooled soft degradation looks gentler than hard's).

So instead of fabricating a slope, constrain the *plan* to the evidence: when a
compound has been run at a track in the target era, but only ever in short
stints, cap the optimiser's stint length for that (track, compound) at slightly
beyond the longest stint actually demonstrated. Teams' avoidance is data.

Deliberately conservative:
- needs ``min_stints`` distinct stints (one odd stint proves nothing);
- a compound *never* run at a track is left uncapped (pure absence is
  ambiguous — could be pace, allocation, or chance, not degradation);
- compounds with any stint at/over ``max_age_threshold`` are left uncapped
  (the field demonstrated real stint lengths; the model's fit has support).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import pandas as pd

from f1se.models.degradation import DegradationModel, _fe_slope, _stint_demeaned

AGE_COL = "tyre_life"
STINT_KEYS = ("year", "round", "driver", "stint")


@dataclass(frozen=True)
class AvoidancePrior:
    """Caps a compound's plannable stint length where the era's field avoided it."""

    max_age_threshold: int = 15   # all stints shorter than this => "avoided beyond"
    min_stints: int = 2           # need >= this many stints to trust the signal
    margin: int = 2               # allow slightly beyond the longest observed stint
    enabled: bool = True

    @classmethod
    def disabled(cls) -> AvoidancePrior:
        return cls(enabled=False)

    def track_caps(self, era_laps: pd.DataFrame) -> dict[tuple[str, str], int]:
        """``{(track, compound): max plannable stint}`` from the era's real running."""
        if not self.enabled or era_laps is None or era_laps.empty:
            return {}
        need = {"event_name", "compound", AGE_COL, *STINT_KEYS}
        if not need <= set(era_laps.columns):
            return {}
        laps = era_laps.dropna(subset=[AGE_COL])
        caps: dict[tuple[str, str], int] = {}
        grouped = laps.groupby(["event_name", "compound"], observed=True)
        for (track, comp), grp in grouped:
            n_stints = grp.groupby(list(STINT_KEYS), observed=True).ngroups
            max_age = int(grp[AGE_COL].max())
            if n_stints >= self.min_stints and max_age < self.max_age_threshold:
                caps[(str(track), str(comp))] = max_age + self.margin
        return caps


def apply_avoidance_adjustments(
    model: DegradationModel,
    era_laps: pd.DataFrame,
    *,
    prior_model: DegradationModel | None = None,
    prior: AvoidancePrior | None = None,
) -> DegradationModel:
    """Repair the fitted quantities for avoided (track, compound) groups.

    Era shrinkage blends thin target-era estimates toward the *old-regs* prior —
    the right trade-off in general, but affirmatively wrong for an avoided
    compound, where every fitted quantity is fiction in a different direction:

    - **Slope**: blended toward the friendly old-era prior (Silverstone softs:
      raw 2026 estimate ~0.16 s/lap crushed to ~0.057). The field's avoidance
      *corroborates* the noisy direct estimate, so use it when it's worse.
      Never lowers a slope.
    - **Base pace**: the few laps that exist are end-of-race dashes (evolved
      track, fresh rubber), inflating the compound's advantage (Silverstone
      softs looked 1.35 s/lap faster than mediums; the old era, with real soft
      running, measured them 0.38 s *slower*). Rebuild the intercept as the
      era's best-supported compound at that track plus the *old-era measured
      gap* — compound offsets are far more stable across reg changes than the
      levels themselves.

    Together with the stint caps: short demonstrated stints stay plannable,
    fictional long ones on fictional pace die.
    """
    prior = prior or AvoidancePrior()
    caps = prior.track_caps(era_laps)
    if not caps or model.group_cols != ("event_name", "compound"):
        return model

    laps = era_laps.dropna(subset=[AGE_COL, "lap_time_fuel_corr_s"]).reset_index(drop=True)
    age_dm, corr_dm = _stint_demeaned(laps)
    laps = laps.assign(_age_dm=age_dm, _corr_dm=corr_dm)

    slopes = dict(model.slopes)
    intercepts = dict(model.intercepts)
    adjusted: dict[str, dict] = {}
    for (track, comp) in caps:
        grp = laps[(laps["event_name"] == track) & (laps["compound"] == comp)]
        key = (track, comp)
        note: dict = {}

        # (a) slope: trust the raw era estimate when it's worse than the blend.
        if not grp.empty:
            raw = _fe_slope(grp["_age_dm"].to_numpy(float), grp["_corr_dm"].to_numpy(float))
            if raw is not None and raw > model.slope(comp, track):
                slopes[key] = raw
                note["slope"] = round(raw, 4)

        # (b) base pace: anchor to the era's best-supported compound at this
        # track, offset by the old-era measured gap between the two compounds.
        if prior_model is not None:
            at_track = laps[(laps["event_name"] == track) & (laps["compound"] != comp)]
            if not at_track.empty:
                anchor = str(at_track.groupby("compound", observed=True).size().idxmax())
                b_anchor = model.intercepts.get((track, anchor))
                p_self = prior_model.intercepts.get((track, comp))
                p_anchor = prior_model.intercepts.get((track, anchor))
                if None not in (b_anchor, p_self, p_anchor):
                    intercepts[key] = b_anchor + (p_self - p_anchor)
                    note["base_anchor"] = anchor
                    note["base"] = round(intercepts[key], 3)
        if note:
            adjusted[f"{track}|{comp}"] = note

    if not adjusted:
        return model
    return replace(model, slopes=slopes, intercepts=intercepts,
                   meta={**model.meta, "avoidance_adjustments": adjusted})
