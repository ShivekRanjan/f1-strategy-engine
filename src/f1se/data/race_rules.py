"""Race-specific regulations the engine cannot infer (data/race_rules.json).

Entered by hand from an official FIA or Pirelli announcement, with its source,
never guessed: a mandated stint cap decides the stop count by itself (Qatar
2023: 18 laps, 2025: 25 laps).
"""

from __future__ import annotations

import json

from f1se.config import PROJECT_ROOT

RULES = PROJECT_ROOT / "data" / "race_rules.json"


def max_stint_laps(season: int, event: str) -> int | None:
    """The mandated per-set stint cap for this race, if one was announced."""
    if not RULES.exists():
        return None
    rules = json.loads(RULES.read_text(encoding="utf-8"))
    return rules.get(str(season), {}).get(event, {}).get("max_stint_laps")
