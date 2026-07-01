"""Thermal prior — track temperature shifts tyre degradation.

Tyre degradation is strongly temperature-dependent: a hot track thermally
overworks the tyre and it falls off faster; a cool track preserves it. The fitted
degradation model pools across all conditions, so it predicts an *average*-
temperature wear rate — which **over**-predicts degradation on a cool day and
**under**-predicts it on a hot one. Backtesting 2026 showed exactly this: the
model's degradation error lines up with track temperature, and the resulting
mis-estimate is what made the optimiser over-stop at cool races (China 23°C,
Canada 18°C) and under-rate the genuine multi-stop at hot ones (Barcelona 50°C).

So we shift the per-compound degradation *slope* by ``sensitivity`` seconds/lap
for every °C the track is above (hotter ⇒ more deg) or below (cooler ⇒ less) a
reference temperature at which the base model is calibrated. This is a labelled,
tunable **assumption** — the same status as the cliff and fuel priors, its
direction physics and its magnitude informed by the observed deg-vs-temp
relationship, not presented as a fitted measurement.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ThermalPrior:
    """Track-temperature adjustment to the degradation slope (s/lap per °C).

    Attributes
    ----------
    sensitivity
        Extra degradation slope (s/lap) per °C of track temperature.
    ref_temp
        Track temperature (°C) at which the base degradation model is unbiased —
        roughly the mean of the training races (deg-error ≈ 0 near here).
    """

    sensitivity: float = 0.006
    ref_temp: float = 36.0

    @classmethod
    def disabled(cls) -> ThermalPrior:
        return cls(sensitivity=0.0)

    def slope_delta(self, track_temp: float | None) -> float:
        """Slope adjustment (s/lap) for a given track temp; 0 if temp unknown."""
        if track_temp is None:
            return 0.0
        return self.sensitivity * (float(track_temp) - self.ref_temp)
