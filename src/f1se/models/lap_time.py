"""Phase 2.5 — sequence lap-time model (LSTM/ConvLSTM). STUB.

The DL differentiator. Predicts the next lap time (or a short horizon) from the
recent sequence of laps for a car — capturing momentum, traffic, and warm-up
effects a per-lap regressor misses. Feeds the simulator with richer per-lap pace.

Beat a naive baseline first (e.g. "next lap == last green lap, fuel-adjusted")
before claiming the sequence model adds value. Same race-grouped, forward-in-time
validation as the degradation model — never a shuffled split across laps.
"""

from __future__ import annotations

import pandas as pd


def fit_sequence_model(laps: pd.DataFrame, **kwargs) -> LapTimeModel:  # noqa: F821
    """Train the sequence lap-time model. TODO Phase 2.5."""
    raise NotImplementedError("Phase 2.5: sequence lap-time model")


def predict_lap_times(
    model: LapTimeModel,  # noqa: F821
    history: pd.DataFrame,
    horizon: int = 1,
) -> pd.Series:
    """Predict the next ``horizon`` lap times from recent history. TODO Phase 2.5."""
    raise NotImplementedError("Phase 2.5: sequence lap-time prediction")
