"""`results 2026` must refresh 2026 without deleting 2023-25 (it used to)."""

import pandas as pd
import pytest

import f1se.standalone.results as R


def _rows(year, n):
    return pd.DataFrame({"year": [year] * n, "round": range(1, n + 1)})


def test_upsert_keeps_other_seasons_and_refuses_to_shrink(tmp_path, monkeypatch):
    monkeypatch.setattr(R, "PROCESSED", tmp_path)
    pd.concat([_rows(2023, 3), _rows(2026, 2)]).to_parquet(tmp_path / "x.parquet")

    full = R._upsert_years([_rows(2026, 5)], [2026], "x.parquet")
    assert sorted(full.groupby("year").size().items()) == [(2023, 3), (2026, 5)]

    with pytest.raises(RuntimeError, match="refusing to shrink"):
        R._upsert_years([], [2026], "x.parquet")   # a failed pull


def test_an_empty_cached_race_is_a_miss_not_a_hit(tmp_path):
    """Empty files cached in June froze R07-R22 for four months."""
    empty, good = tmp_path / "e.parquet", tmp_path / "g.parquet"
    pd.DataFrame({"position": pd.Series([], dtype=float)}).to_parquet(empty)
    pd.DataFrame({"position": [1.0, 2.0]}).to_parquet(good)
    assert R._usable_cache(empty) is None
    assert R._usable_cache(tmp_path / "missing.parquet") is None
    assert len(R._usable_cache(good)) == 2
