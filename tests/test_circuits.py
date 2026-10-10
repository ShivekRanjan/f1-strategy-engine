"""Models key on the circuit, not the event name.

In 2026 the Spanish GP moved to Madrid and the Bahrain GP was held in Malaysia.
Keyed on event name, both new venues inherited another track's history.
"""

import pandas as pd

from f1se.data.circuits import apply_circuit_keys, circuit_key
from f1se.standalone.races import race_card


def test_relocated_events_get_their_own_circuit_and_barcelona_reunites():
    assert circuit_key(2024, "Spanish Grand Prix") == "Barcelona Grand Prix"   # was Barcelona
    assert circuit_key(2026, "Barcelona Grand Prix") == "Barcelona Grand Prix"
    assert circuit_key(2026, "Spanish Grand Prix") == "Spanish Grand Prix"      # Madrid: no prior
    assert circuit_key(2025, "Bahrain Grand Prix") == "Bahrain Grand Prix"      # Sakhir keeps it
    assert circuit_key(2026, "Bahrain Grand Prix") == "Bahrain Grand Prix (Malaysia)"
    assert circuit_key(2026, "Italian Grand Prix") == "Italian Grand Prix"


def test_apply_is_idempotent_and_preserves_dtype():
    df = pd.DataFrame({"year": [2024, 2026, 2026],
                       "event_name": pd.array(["Spanish Grand Prix", "Spanish Grand Prix",
                                               "Bahrain Grand Prix"], dtype="string")})
    once = apply_circuit_keys(df)
    assert list(once["event_name"]) == ["Barcelona Grand Prix", "Spanish Grand Prix",
                                        "Bahrain Grand Prix (Malaysia)"]
    assert once["event_name"].dtype == df["event_name"].dtype
    assert apply_circuit_keys(once).equals(once)


def test_the_committed_data_has_no_circuit_split_across_venues():
    """The modelling data must already hold circuit keys, not raw event names."""
    from f1se.config import PROJECT_ROOT

    d = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet")
    pre = set(d.loc[d["year"] < 2026, "event_name"].astype(str))
    assert "Spanish Grand Prix" not in pre, "Madrid would inherit Barcelona's history"
    assert "Barcelona Grand Prix" in pre, "Barcelona's history must sit under its own key"


def test_race_card_resolves_the_name_the_calendar_shows():
    """The calendar says "Bahrain Grand Prix" for 2026; the data says Malaysia."""
    a = race_card(2026, "Bahrain Grand Prix")
    b = race_card(2026, "Bahrain Grand Prix (Malaysia)")
    assert a is not None and a == b
    assert race_card(2024, "Spanish Grand Prix") == race_card(2024, "Barcelona Grand Prix")
