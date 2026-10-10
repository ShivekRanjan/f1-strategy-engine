"""Circuit identity, kept separate from the event name.

Every model here keys on the *track* — degradation, pit loss, safety-car hazard,
stint limits, overtaking difficulty. FastF1 names *events*, and in 2026 the two
came apart: the Spanish GP moved from Barcelona to Madrid (Barcelona kept a race,
as the "Barcelona GP"), and the Bahrain GP was held in Malaysia. Keyed on event
name, Madrid and Kuala Lumpur each inherited three years of another circuit's
history, while Barcelona's own history sat orphaned under a name it no longer
races under. Both new venues were strategy misses in the 16-race backtest.

The combined datasets store this key in ``event_name``, so every consumer is
circuit-correct without knowing this module exists. Only rows where the name and
the venue disagree are rewritten; everything else keys on its own event name.
Sakhir keeps "Bahrain Grand Prix" so a future return there reuses its history.
"""

from __future__ import annotations

import pandas as pd

#: event name -> (circuit key, seasons it applies to)
_RENAMES: dict[str, tuple[str, range]] = {
    "Spanish Grand Prix": ("Barcelona Grand Prix", range(1950, 2026)),
    "Bahrain Grand Prix": ("Bahrain Grand Prix (Malaysia)", range(2026, 2027)),
}


def circuit_key(year: int, event_name: str) -> str:
    key, years = _RENAMES.get(event_name, (event_name, range(0)))
    return key if int(year) in years else event_name


def apply_circuit_keys(df: pd.DataFrame) -> pd.DataFrame:
    """``df`` with ``event_name`` rewritten to circuit keys. Idempotent."""
    if "event_name" not in df or "year" not in df:
        return df
    df = df.copy()
    names = df["event_name"].astype(str)
    for name, (key, years) in _RENAMES.items():
        names = names.mask((names == name) & df["year"].isin(years), key)
    df["event_name"] = names.astype(df["event_name"].dtype)
    return df
