"""Tyre-'cliff' prior — DOMAIN KNOWLEDGE, not fitted from data.

The accelerating end-of-life degradation (the 'cliff') is censored out of race
data: teams pit before it, so it cannot be estimated from observed laps (we
proved this in :mod:`f1se.models.degradation_poly` — a quadratic fit was worse
out-of-sample and showed no consistent curvature). Yet the cliff is real physics,
and ignoring it makes the optimiser over-prefer long stints on soft tyres.

So we add it as an explicit, tunable **prior** — the same status as the 0.03 s/kg
fuel coefficient: an assumption, clearly labelled, seeded from public Pirelli /
engineering guidance, NOT presented as a measurement. Pace loss gains an extra
convex term beyond a per-compound onset age::

    extra(age) = rate * max(age - cliff_age[compound], 0) ** power

Zero within a tyre's normal window, then accelerating — soft tyres cliff earliest.
Set :meth:`CliffPrior.disabled` to turn it off and recover the pure data model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


def _default_cliff_ages() -> dict[str, float]:
    # Onset (laps of tyre age) where degradation starts to accelerate. Seeded
    # from typical F1 stint behaviour: softs fall off earliest, hards last
    # longest. These are ASSUMPTIONS — tune per track / from team knowledge.
    return {"SOFT": 18.0, "MEDIUM": 28.0, "HARD": 38.0}


@dataclass(frozen=True)
class CliffPrior:
    """A physical, tunable tyre-cliff prior (not data-derived).

    Attributes
    ----------
    cliff_age
        Per-compound tyre age (laps) at which degradation starts accelerating.
    rate
        Seconds of extra pace loss per ``(lap beyond onset) ** power``.
    power
        Convexity of the cliff (2.0 = quadratic acceleration).
    """

    cliff_age: dict[str, float] = field(default_factory=_default_cliff_ages)
    rate: float = 0.05
    power: float = 2.0

    @classmethod
    def disabled(cls) -> CliffPrior:
        """A no-op prior (no cliff) — recovers the pure data-fitted model."""
        return cls(cliff_age={}, rate=0.0)

    @classmethod
    def from_stints(cls, laps, quantile: float = 0.75, **kwargs) -> CliffPrior:
        """Onsets where the field has demonstrably raced without one.

        Each compound's onset is the ``quantile`` of its stint lengths in
        ``laps``: that share of stints ended sooner, and the rest ran past it,
        so no cliff can sit much before it. The fixed defaults (soft 18 laps)
        predate 2026, where a quarter of soft stints ran past 25. Pass only the
        laps the model may see: stint lengths are the strategy itself, so the
        race being predicted must not be among them.
        """
        keys = ["year", "round", "driver", "stint"]
        st = laps.groupby(keys, observed=True).agg(c=("compound", "first"), n=("tyre_life", "max"))
        ages = {str(c): float(g["n"].quantile(quantile))
                for c, g in st.groupby("c", observed=True) if len(g) >= 20}
        return cls(cliff_age=ages or _default_cliff_ages(), **kwargs)

    def extra_loss(self, compound: str, tyre_age: float) -> float:
        """Extra pace loss (s) from the cliff at this compound/age (>= 0)."""
        onset = self.cliff_age.get(compound, float("inf"))
        over = max(float(tyre_age) - onset, 0.0)
        return self.rate * over ** self.power
