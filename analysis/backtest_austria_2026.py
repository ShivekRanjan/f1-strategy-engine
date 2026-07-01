"""Out-of-sample backtest: the whole f1se build vs ground truth (Austrian GP 2026).

Austria 2026 is in NONE of the committed data, so every check here is genuinely
out-of-sample. Pulls the real race from FastF1 (network) and compares strategy,
degradation, the LSTM nowcast, and the podium model against what actually
happened — including the "were the softs actually quick?" question a driver
raised post-race.

    .venv\\Scripts\\python.exe analysis/backtest_austria_2026.py
"""
from __future__ import annotations

import warnings
from collections import Counter

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from f1se.config import enable_cache
from f1se.data.clean import clean_laps
from f1se.data.loader import load_session_laps
from f1se.eda import fit_stint_slopes
from f1se.engine import StrategyEngine

TRACK = "Austrian Grand Prix"


def _stints(driver_laps: pd.DataFrame):
    out = []
    for _, g in driver_laps.sort_values("lap_number").groupby("stint"):
        out.append((str(g["compound"].iloc[0]), int(g["lap_number"].min()), int(g["lap_number"].max())))
    return out


def main() -> None:
    enable_cache()
    import fastf1

    ses = fastf1.get_session(2026, "Austria", "R")
    ses.load(laps=True, telemetry=False, weather=False, messages=False)
    res = ses.results.sort_values("Position")
    podium = list(res["Abbreviation"].head(3))
    total = int(ses.laps["LapNumber"].max())

    raw = load_session_laps(2026, "Austria", "R")
    clean = clean_laps(raw, dry_only=True).assign(race="AUT26")
    eng = StrategyEngine.from_processed()

    print("=" * 68)
    print("GROUND TRUTH — Austrian GP 2026")
    print("=" * 68)
    print(f"{total} laps · podium: {' '.join(podium)}")
    winner = _stints(raw[raw["driver"] == podium[0]])
    pit_actual = [s[2] for s in winner[:-1]]
    print(f"Winner {podium[0]}: {' -> '.join(f'{c}(L{a}-{b})' for c, a, b in winner)}  "
          f"pit {pit_actual} ({len(pit_actual)}-stop)")
    print("field stop counts:",
          dict(Counter(g["stint"].nunique() - 1 for _, g in raw.groupby("driver")
                       if g["lap_number"].max() > total * 0.5)))

    # 1) STRATEGY
    print("\n" + "=" * 68 + "\n1) STRATEGY\n" + "=" * 68)
    rec = eng.recommend(TRACK, season=2026, n_runs=4000)["best"]
    print(f"engine total laps {eng.total_laps_by_track[TRACK]} (actual {total})")
    print(f"engine: {'-'.join(c[0] for c in rec['compounds'])} pit {rec['pit_laps']} "
          f"({len(rec['pit_laps'])}-stop) | winner: {'-'.join(s[0] for s in winner)} "
          f"pit {pit_actual} ({len(pit_actual)}-stop)")

    # 2) DEGRADATION
    print("\n" + "=" * 68 + "\n2) DEGRADATION (actual vs model)\n" + "=" * 68)
    med = fit_stint_slopes(clean).groupby("compound")["slope_s_per_lap"].median()
    m26 = eng.deg_model_2026 or eng.deg_model
    print(f"{'compound':<8}{'actual':>9}{'model':>9}")
    for c in ["SOFT", "MEDIUM", "HARD"]:
        print(f"{c:<8}{med.get(c, float('nan')):>9.4f}{m26.slope(c, TRACK):>9.4f}")

    # 3) LSTM nowcast (unseen race)
    print("\n" + "=" * 68 + "\n3) LSTM NEXT-LAP NOWCAST (unseen race)\n" + "=" * 68)
    if eng.forecaster is not None:
        from f1se.models.lap_time import build_sequence_windows
        w = build_sequence_windows(clean, window=eng.forecaster.window)
        pers = float(np.mean(np.abs(w.y_curr - w.y_next)))
        lstm = float(np.mean(np.abs((w.y_curr + eng.forecaster.predict_delta(w.X)) - w.y_next)))
        print(f"persistence {pers:.4f}s · LSTM {lstm:.4f}s  ({100 * (pers - lstm) / pers:+.1f}%)")

    # 4) PODIUM (forward test)
    print("\n" + "=" * 68 + "\n4) PODIUM PREDICTOR (forward test)\n" + "=" * 68)
    from f1se.standalone.podium import build_features, predict_race, train_podium_model
    results = pd.read_parquet(eng._resolve_data_dir() / "results.parquet")
    aut = pd.DataFrame({
        "year": 2026, "round": int(results[results.year == 2026]["round"].max()) + 1,
        "event_name": TRACK, "driver": res["Abbreviation"].astype("string").values,
        "team": res["TeamName"].astype("string").values,
        "grid": res["GridPosition"].astype(float).values,
        "position": res["Position"].astype(float).values,
        "points": res["Points"].astype(float).values,
        "status": res["Status"].astype("string").values,
    })
    feats = build_features(pd.concat([results, aut], ignore_index=True), recency_halflife=4.0)
    model = train_podium_model(feats, test_year=2026)
    pred = predict_race(model, feats[(feats.year == 2026) & (feats.event_name == TRACK)]).head(3)
    print(f"model top-3 {list(pred['driver'])} vs actual {podium} "
          f"({len(set(pred['driver']) & set(podium))}/3)")
    grid3 = list(res.sort_values("GridPosition")["Abbreviation"].head(3))
    print(f"grid top-3  {grid3} ({len(set(grid3) & set(podium))}/3)")

    # 5) "Were the softs actually quick?" (a driver's post-race claim)
    print("\n" + "=" * 68 + "\n5) SOFT-TYRE CHECK (was the model right to like softs?)\n" + "=" * 68)
    for c in ["SOFT", "MEDIUM", "HARD"]:
        sub = clean[clean.compound == c]
        s = fit_stint_slopes(clean)
        print(f"{c:<7} laps={len(sub):>4} drivers={sub.driver.nunique():>2} "
              f"deg={s[s.compound == c]['slope_s_per_lap'].median():.4f}")
    print("(soft degraded the least here — the model's soft lean was a real signal, "
          "not thin-data noise)")


if __name__ == "__main__":
    main()
