"""No-network tests for the coarse-to-fine strategy search.

The screen must be a pure speed-up: the winner (and the reported numbers for
it) have to match the exhaustive full-precision search.
"""

from __future__ import annotations

import numpy as np

from f1se.models.degradation import DegradationModel
from f1se.sim.optimize import enumerate_strategies, recommend_strategy
from f1se.sim.simulate import pace_fn_from_model


def _pace_fn(total_laps: int = 50):
    model = DegradationModel(
        group_cols=("event_name", "compound"),
        slopes={},
        intercepts={("T", "SOFT"): 89.4, ("T", "MEDIUM"): 90.0, ("T", "HARD"): 90.5},
        compound_slope={"SOFT": 0.10, "MEDIUM": 0.06, "HARD": 0.03},
        global_slope=0.05,
    )
    return pace_fn_from_model(model, "T", total_laps)


def test_screened_search_matches_full_search():
    pace = _pace_fn()
    kwargs = dict(n_runs=2000, top_k=5, seed=3, max_stops=2)
    fast = recommend_strategy(50, pace, **kwargs)                     # screening on (default)
    full = recommend_strategy(50, pace, screen_runs=0, **kwargs)      # exhaustive
    assert fast.best == full.best
    # Reported numbers for the winner come from the same full-precision draw.
    assert np.allclose(fast.best_samples, full.best_samples)
    # n_evaluated reports the whole search space, not just the survivors.
    assert fast.n_evaluated == full.n_evaluated > 64


def test_screening_skipped_when_it_cannot_help():
    pace = _pace_fn()
    # Small n_runs (the no-network tests' regime): behaviour must be identical
    # with or without the screen parameters present.
    a = recommend_strategy(50, pace, n_runs=300, seed=1, max_stops=1)
    b = recommend_strategy(50, pace, n_runs=300, seed=1, max_stops=1, screen_runs=0)
    assert a.best == b.best and a.n_evaluated == b.n_evaluated


def test_three_stop_layer_uses_coarser_grid():
    # 1-2 stop candidates are unchanged by the adaptive grid; the 3-stop layer
    # is generated on a coarser step so the space stays tractable.
    two = enumerate_strategies(57, max_stops=2)
    three = enumerate_strategies(57, max_stops=3)
    assert set(two) <= set(three)                       # coarsening never touches <=2-stop
    n_three_only = len(three) - len(two)
    assert 0 < n_three_only < 5000                      # vs ~9.4k on the fine grid
