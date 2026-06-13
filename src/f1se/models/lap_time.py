"""Phase 2.5 — sequence lap-time model (LSTM). The deep-learning chapter.

A recurrent (LSTM) model that forecasts the **next lap** from the recent
sequence of laps in a stint, capturing dynamics a per-lap regressor cannot see:
tyre warm-up at the start of a stint, traffic recovery, and compound-specific
curvature. It is the differentiator — *and it must earn its place* against dumb
baselines on a leakage-safe, forward-in-time split, exactly like the boosted
degradation model had to beat the linear one.

The task, framed to be honest
-----------------------------
One-step-ahead within-stint forecasting. Standing at lap ``t`` of a stint, having
observed laps ``1..t``, predict lap ``t+1``'s *fuel-corrected* time. The network
predicts the **lap-to-lap delta** ``Δ = y[t+1] - y[t]`` and we reconstruct
``ŷ[t+1] = y[t] + Δ̂``. Predicting the delta (not the absolute time) is the key
leakage guard: a stint's base pace is track/car/fuel specific, so a model that
learnt absolute pace would simply memorise circuits and collapse on a held-out
race. The delta strips the per-stint level out, so the model can only win by
predicting *change* — momentum and curvature — which is the whole point.

Baselines it must beat (all predict an absolute next-lap time, same metric):
  * **persistence** — ``ŷ[t+1] = y[t]`` (Δ̂ = 0). The canonical dumb forecaster.
  * **rolling-slope** — fit OLS to the window's (age, pace) and extrapolate one
    lap. This is the linear degradation model applied online; a genuinely strong
    classical comparator, not a strawman.

Split: forward-in-time (:mod:`f1se.validation`) — train on ``year <= train_max``
(default 2024), test on the next season (2025). Never a shuffled lap split. 2026
is excluded from this experiment because it is a regulation reset (a different
regime, handled by the shrinkage degradation model), which would confound a
"does the sequence model generalise forward?" read.

Torch is imported lazily (inside the fitting functions), so this module imports
fine without the ``[models]`` extra; only training/prediction needs it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from f1se.validation import race_id

STINT_KEYS = ("year", "round", "driver", "stint")
AGE_COL = "tyre_life"
TARGET_COL = "lap_time_fuel_corr_s"
COMPOUNDS = ("SOFT", "MEDIUM", "HARD")

# Per-lap input features fed to the LSTM. All are observable at or before the
# current lap and carry no absolute base-pace information (see module docstring):
#   d_prev        lap-to-lap fuel-corrected pace change into this lap
#   tyre_age_z    standardised tyre age
#   comp_*        compound one-hot (constant in a stint; sets the degradation rate)
#   stint_frac    lap-in-stint fraction (warm-up early, settled later)
#   d_position    track-position change vs the previous lap (a traffic proxy)
FEATURE_NAMES = ("d_prev", "tyre_age", "comp_SOFT", "comp_MEDIUM", "comp_HARD",
                 "stint_frac", "d_position")
DEFAULT_WINDOW = 5


# --------------------------------------------------------------------------- #
# Windowing — turn cleaned laps into (sequence, next-lap) supervised samples.  #
# --------------------------------------------------------------------------- #
@dataclass
class SequenceWindows:
    """Supervised one-step-ahead windows extracted from cleaned laps.

    ``X`` is ``(n_samples, window, n_features)``. For sample ``i`` the inputs are
    the last ``window`` laps of a stint ending at lap ``t``; the targets describe
    lap ``t+1``::

        y_next   absolute fuel-corrected time of lap t+1   (what we score on)
        y_curr   fuel-corrected time of lap t              (persistence baseline)
        slope_next  rolling-slope baseline's prediction of y[t+1]
        race_ids stable race id per sample (for leakage-safe grouping)
        years    season per sample (for the forward-in-time split)
    """

    X: np.ndarray
    y_next: np.ndarray
    y_curr: np.ndarray
    slope_next: np.ndarray
    race_ids: np.ndarray
    years: np.ndarray
    feature_names: tuple[str, ...] = FEATURE_NAMES
    window: int = DEFAULT_WINDOW


def _stint_arrays(grp: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Sorted (age, pace, position, d_prev) arrays for one stint."""
    grp = grp.sort_values("lap_number")
    age = grp[AGE_COL].to_numpy(float)
    pace = grp[TARGET_COL].to_numpy(float)
    pos = grp["position"].astype("float64").to_numpy() if "position" in grp else np.full(len(grp), np.nan)
    d_prev = np.diff(pace, prepend=pace[0])  # first lap: 0 change
    return age, pace, pos, d_prev


def build_sequence_windows(
    laps: pd.DataFrame,
    *,
    window: int = DEFAULT_WINDOW,
    min_stint_laps: int | None = None,
) -> SequenceWindows:
    """Slice cleaned dry laps into one-step-ahead windows, per stint.

    Expects the output of ``clean_laps(..., dry_only=True)`` (needs ``tyre_life``,
    ``lap_time_fuel_corr_s``, the stint keys, and ideally ``position``). A stint of
    ``n`` laps yields ``n - window`` samples; stints shorter than ``window + 1`` are
    skipped. Crucially, every feature for sample ``i`` is computed from laps at or
    before lap ``t`` — never from the lap being predicted.
    """
    for col in (*STINT_KEYS, AGE_COL, TARGET_COL):
        if col not in laps.columns:
            raise ValueError(f"sequence windowing needs column '{col}'")
    min_stint_laps = min_stint_laps if min_stint_laps is not None else window + 1

    laps = laps.dropna(subset=[AGE_COL, TARGET_COL]).copy()
    rid = race_id(laps)

    Xs: list[np.ndarray] = []
    y_next: list[float] = []
    y_curr: list[float] = []
    slope_next: list[float] = []
    rids: list[str] = []
    years: list[int] = []

    for keys, grp in laps.groupby(list(STINT_KEYS), observed=True):
        n = len(grp)
        if n < min_stint_laps:
            continue
        age, pace, pos, d_prev = _stint_arrays(grp)
        comp = str(grp["compound"].iloc[0])
        comp_oh = [float(comp == c) for c in COMPOUNDS]
        this_rid = str(rid.loc[grp.index[0]])
        year = int(keys[0])
        denom = max(age.max(), 1.0)
        d_pos = np.diff(pos, prepend=pos[0])
        d_pos = np.nan_to_num(d_pos, nan=0.0)

        # Window ends at index t (0-based); predict index t+1.
        for t in range(window - 1, n - 1):
            sl = slice(t - window + 1, t + 1)
            feats = np.column_stack([
                d_prev[sl],
                age[sl],
                np.full(window, comp_oh[0]),
                np.full(window, comp_oh[1]),
                np.full(window, comp_oh[2]),
                age[sl] / denom,
                d_pos[sl],
            ])
            Xs.append(feats.astype(np.float32))
            y_next.append(float(pace[t + 1]))
            y_curr.append(float(pace[t]))
            slope_next.append(_rolling_slope_pred(age[sl], pace[sl], age[t + 1]))
            rids.append(this_rid)
            years.append(year)

    if not Xs:
        raise RuntimeError("no sequence windows built — check window/min_stint_laps")

    return SequenceWindows(
        X=np.stack(Xs),
        y_next=np.asarray(y_next, np.float32),
        y_curr=np.asarray(y_curr, np.float32),
        slope_next=np.asarray(slope_next, np.float32),
        race_ids=np.asarray(rids, dtype=object),
        years=np.asarray(years, int),
        window=window,
    )


def _rolling_slope_pred(age: np.ndarray, pace: np.ndarray, next_age: float) -> float:
    """Rolling-slope baseline: OLS over the window, extrapolated to ``next_age``.

    Degenerate windows (no age spread) fall back to persistence (last pace).
    """
    if np.ptp(age) == 0:
        return float(pace[-1])
    slope, intercept = np.polyfit(age, pace, 1)
    return float(slope * next_age + intercept)


# --------------------------------------------------------------------------- #
# The model wrapper — torch module + the feature scaling it was trained with.  #
# --------------------------------------------------------------------------- #
@dataclass
class SequenceLapModel:
    """A trained sequence lap-time model: the LSTM plus its feature scaler.

    Predicts the next-lap **delta**; :meth:`predict_next` reconstructs the
    absolute next-lap time as ``y_curr + Δ̂``.
    """

    module: Any                       # torch.nn.Module (typed loosely to keep import lazy)
    feat_mean: np.ndarray
    feat_std: np.ndarray
    window: int = DEFAULT_WINDOW
    feature_names: tuple[str, ...] = FEATURE_NAMES
    meta: dict = field(default_factory=dict)

    def _scale(self, X: np.ndarray) -> np.ndarray:
        return (X - self.feat_mean) / self.feat_std

    def predict_delta(self, X: np.ndarray) -> np.ndarray:
        """Predicted lap-to-lap delta Δ̂ for each window (shape ``(n,)``)."""
        import torch

        self.module.eval()
        with torch.no_grad():
            xb = torch.from_numpy(self._scale(X).astype(np.float32))
            return self.module(xb).cpu().numpy().reshape(-1)

    def predict_next(self, windows: SequenceWindows) -> np.ndarray:
        """Predicted absolute fuel-corrected next-lap time for each window."""
        return windows.y_curr + self.predict_delta(windows.X)


def _make_lstm(n_features: int, hidden: int, layers: int, dropout: float):
    """Build the LSTM regressor. Torch + the class live here so the module
    imports without the ``[models]`` extra installed."""
    from torch import nn

    class _LapLSTM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=n_features, hidden_size=hidden, num_layers=layers,
                batch_first=True, dropout=dropout if layers > 1 else 0.0,
            )
            self.head = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, 1))

        def forward(self, x):
            out, _ = self.lstm(x)          # (B, T, H)
            return self.head(out[:, -1, :])  # last timestep -> delta

    return _LapLSTM()


def fit_sequence_model(
    windows: SequenceWindows,
    *,
    val_windows: SequenceWindows | None = None,
    hidden: int = 32,
    layers: int = 1,
    dropout: float = 0.0,
    lr: float = 1e-3,
    epochs: int = 40,
    batch_size: int = 256,
    patience: int = 6,
    seed: int = 0,
) -> SequenceLapModel:
    """Train the LSTM to predict the next-lap delta. CPU-friendly and seeded.

    Standardises features on the training windows (those stats travel with the
    returned model). If ``val_windows`` is given, early-stops on its delta MSE.
    """
    import torch
    from torch import nn

    torch.manual_seed(seed)
    np.random.seed(seed)

    X = windows.X
    feat_mean = X.reshape(-1, X.shape[-1]).mean(axis=0)
    feat_std = X.reshape(-1, X.shape[-1]).std(axis=0)
    feat_std[feat_std == 0] = 1.0

    def _scaled_tensor(w: SequenceWindows) -> tuple[Any, Any]:
        xb = torch.from_numpy(((w.X - feat_mean) / feat_std).astype(np.float32))
        yb = torch.from_numpy((w.y_next - w.y_curr).astype(np.float32)).reshape(-1, 1)
        return xb, yb

    xb, yb = _scaled_tensor(windows)
    ds = torch.utils.data.TensorDataset(xb, yb)
    g = torch.Generator().manual_seed(seed)
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, generator=g)

    model = _make_lstm(X.shape[-1], hidden, layers, dropout)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.SmoothL1Loss()  # Huber: robust to traffic-spiked lap outliers

    vxb = vyb = None
    if val_windows is not None:
        vxb, vyb = _scaled_tensor(val_windows)

    best_val = float("inf")
    best_state = None
    since_best = 0
    history: list[dict] = []

    for epoch in range(epochs):
        model.train()
        for xbatch, ybatch in loader:
            opt.zero_grad()
            loss = loss_fn(model(xbatch), ybatch)
            loss.backward()
            opt.step()

        if vxb is not None:
            model.eval()
            with torch.no_grad():
                val = float(nn.functional.mse_loss(model(vxb), vyb))
            history.append({"epoch": epoch, "val_mse": val})
            if val < best_val - 1e-6:
                best_val, best_state, since_best = val, {k: v.clone() for k, v in model.state_dict().items()}, 0
            else:
                since_best += 1
                if since_best >= patience:
                    break

    if best_state is not None:
        model.load_state_dict(best_state)

    return SequenceLapModel(
        module=model, feat_mean=feat_mean, feat_std=feat_std, window=windows.window,
        meta={"n_train": len(windows.y_next), "epochs_run": len(history) or epochs,
              "best_val_mse": best_val if best_state is not None else None,
              "hidden": hidden, "layers": layers},
    )


# --------------------------------------------------------------------------- #
# Leakage-safe head-to-head: LSTM vs persistence vs rolling-slope.             #
# --------------------------------------------------------------------------- #
def _mae(pred: np.ndarray, truth: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - truth)))


def evaluate_sequence_vs_baselines(
    laps: pd.DataFrame,
    *,
    train_max_year: int = 2024,
    test_year: int = 2025,
    val_frac: float = 0.15,
    window: int = DEFAULT_WINDOW,
    seed: int = 0,
    **fit_kwargs: Any,
) -> dict[str, Any]:
    """Forward-in-time head-to-head on one-step-ahead next-lap MAE.

    Train the LSTM on ``year <= train_max_year`` (a race-grouped slice held out
    for early stopping), then score persistence, rolling-slope, and the LSTM on
    ``test_year``. Every number is MAE in seconds on the *absolute* fuel-corrected
    next-lap time, so the three are directly comparable.
    """
    train_laps = laps[laps["year"] <= train_max_year]
    test_laps = laps[laps["year"] == test_year]
    if train_laps.empty or test_laps.empty:
        raise ValueError(f"empty split (train<= {train_max_year}, test={test_year}); check years loaded")

    train_w = build_sequence_windows(train_laps, window=window)
    test_w = build_sequence_windows(test_laps, window=window)

    # Race-grouped validation slice for early stopping (no race spans the split).
    rng = np.random.default_rng(seed)
    races = np.unique(train_w.race_ids)
    val_races = set(rng.choice(races, size=max(1, int(len(races) * val_frac)), replace=False))
    is_val = np.array([r in val_races for r in train_w.race_ids])
    sub_tr = _subset(train_w, ~is_val)
    sub_val = _subset(train_w, is_val)

    model = fit_sequence_model(sub_tr, val_windows=sub_val, seed=seed, **fit_kwargs)

    truth = test_w.y_next
    out = {
        "persistence_mae": _mae(test_w.y_curr, truth),
        "rolling_slope_mae": _mae(test_w.slope_next, truth),
        "lstm_mae": _mae(model.predict_next(test_w), truth),
        "n_test": int(len(truth)),
        "n_train": int(len(sub_tr.y_next)),
        "train_max_year": train_max_year,
        "test_year": test_year,
        "window": window,
        "model_meta": model.meta,
    }
    base = out["persistence_mae"]
    out["lstm_vs_persistence_pct"] = 100.0 * (base - out["lstm_mae"]) / base if base else float("nan")
    slope = out["rolling_slope_mae"]
    out["lstm_vs_slope_pct"] = 100.0 * (slope - out["lstm_mae"]) / slope if slope else float("nan")
    return out


def _subset(w: SequenceWindows, mask: np.ndarray) -> SequenceWindows:
    return SequenceWindows(
        X=w.X[mask], y_next=w.y_next[mask], y_curr=w.y_curr[mask],
        slope_next=w.slope_next[mask], race_ids=w.race_ids[mask], years=w.years[mask],
        feature_names=w.feature_names, window=w.window,
    )
