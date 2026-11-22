r"""Is there tyre management to model? Two signatures, tested within each race (METHODOLOGY §21).

If drivers manage tyres on long stints, two things should show:

1. **Wear**: a longer planned stint wears more slowly per lap.
2. **Pace**: a longer planned stint is slower from its first laps, which is the
   price of the lower wear.

Pooled across races, (1) looks strong, but stint length is not random: low-wear
tracks get long stints. So both are measured *within* a race:

- (1) stint slope on 1/stint length, demeaned within (race, compound), with a
  race bootstrap;
- (2) per race, lap time on lap + driver + compound + compound x age fixed
  effects plus stint length, with a bootstrap of the median race;
- (3) management that starts *mid-stint* would flatten the second half of long
  stints (a cliff would steepen it): second-half minus first-half slope in
  20+ lap stints, pace measured against the same-lap field median.

    .venv\Scripts\python.exe analysis/tyre_management_check.py
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import PROJECT_ROOT  # noqa: E402
from f1se.eda import fit_stint_slopes  # noqa: E402

COMPS = ("SOFT", "MEDIUM", "HARD")


def wear_gradient(d: pd.DataFrame) -> None:
    s = fit_stint_slopes(d.dropna(subset=["lap_time_fuel_corr_s"]).assign(race=d.event_name))
    s = s[s.slope_s_per_lap.abs() < 0.5]
    s["era"] = np.where(s.year >= 2026, "2026", "2023-25")
    s["invL"] = 1 / s.n_laps
    g = s.groupby(["year", "round", "compound"])
    s["y"] = s.slope_s_per_lap - g.slope_s_per_lap.transform("mean")
    s["x"] = s.invL - g.invL.transform("mean")
    rng = np.random.default_rng(0)
    print("(1) wear vs stint length, within race — 12-lap minus 30-lap stint, s/lap")
    for (era, comp), q in s.groupby(["era", "compound"]):
        def b(z):
            return (z.x * z.y).sum() / (z.x ** 2).sum()
        races = [r for _, r in q.groupby(["year", "round"])]
        bs = [b(pd.concat([races[i] for i in rng.integers(0, len(races), len(races))]))
              for _ in range(300)]
        k = 1 / 12 - 1 / 30
        lo, hi = np.percentile(bs, [2.5, 97.5]) * k
        print(f"  {era:<8}{comp:<7}{b(q) * k:+.3f}   95% CI [{lo:+.3f}, {hi:+.3f}]   stints {len(q)}")


def pace_cost(d: pd.DataFrame) -> None:
    d = d.dropna(subset=["lap_time_s"]).copy()
    d["L"] = d.groupby(["year", "round", "driver", "stint"]).tyre_life.transform("max")
    rows = []
    for (y, _rnd), g in d.groupby(["year", "round"]):
        g = g.reset_index(drop=True)
        comp, age = g.compound.astype(str).to_numpy(), g.tyre_life.to_numpy(float)
        X = np.hstack([
            pd.get_dummies(g.lap_number.astype(int), dtype=float).to_numpy(),
            pd.get_dummies(g.driver.astype(str), drop_first=True, dtype=float).to_numpy(),
            np.column_stack([(comp == c).astype(float) for c in ("SOFT", "HARD")]),
            np.column_stack([(comp == c) * age for c in COMPS]),
            (g.L.to_numpy(float) / 10)[:, None],
        ])
        yv = g.lap_time_s.to_numpy(float)
        keep = np.abs(yv - np.median(yv)) < 5
        coef, *_ = np.linalg.lstsq(X[keep], yv[keep], rcond=None)
        rows.append({"year": y, "per10": float(np.clip(coef[-1], -1, 1))})
    e = pd.DataFrame(rows)
    e["era"] = np.where(e.year >= 2026, "2026", "2023-25")
    rng = np.random.default_rng(0)
    print("(2) pace cost of +10 laps of planned stint, within race, s/lap")
    for era, q in e.groupby("era"):
        v = q.per10.to_numpy()
        bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        print(f"  {era:<8}median race {np.median(v):+.3f}   95% CI "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]   races {len(v)}")


def mid_stint(d: pd.DataFrame) -> None:
    d = d.dropna(subset=["lap_time_s"]).copy()
    d["rel"] = d.lap_time_s - d.groupby(["year", "round", "lap_number"]).lap_time_s.transform("median")
    rows = []
    for k, g in d.groupby(["year", "round", "driver", "stint"]):
        g = g.sort_values("tyre_life")
        if len(g) < 20 or g.rel.abs().max() > 5:
            continue
        a, y, h = g.tyre_life.to_numpy(float), g.rel.to_numpy(float), len(g) // 2
        rows.append({"year": k[0], "comp": g.compound.iloc[0],
                     "chg": np.polyfit(a[h:], y[h:], 1)[0] - np.polyfit(a[:h], y[:h], 1)[0]})
    r = pd.DataFrame(rows)
    r["era"] = np.where(r.year >= 2026, "2026", "2023-25")
    rng = np.random.default_rng(0)
    print("(3) second-half minus first-half slope, 20+ lap stints, s/lap (- flattens, + steepens)")
    for (era, comp), q in r.groupby(["era", "comp"]):
        v = q.chg.to_numpy()
        bs = [np.median(rng.choice(v, len(v))) for _ in range(2000)]
        print(f"  {era:<8}{comp:<7}{np.median(v):+.3f}   95% CI "
              f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]   stints {len(v)}")


if __name__ == "__main__":
    laps = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "dry_laps.parquet")
    laps = laps.dropna(subset=["tyre_life"])
    laps = laps[laps.compound.isin(COMPS)]
    wear_gradient(laps)
    pace_cost(laps)
    mid_stint(laps)
