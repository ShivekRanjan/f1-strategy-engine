"""A partial per-race cache (a fresh CI runner) must not shrink the dataset."""

import pandas as pd

import f1se.data.ingest as I


def test_combined_file_keeps_seasons_missing_from_the_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(I, "PROCESSED", tmp_path)
    cache = tmp_path / "by_race"
    cache.mkdir()
    rows = lambda y, n: pd.DataFrame({"year": [y] * n, "event_name": ["Italian Grand Prix"] * n})  # noqa: E731
    pd.concat([rows(2024, 5), rows(2026, 2)]).to_parquet(tmp_path / "x.parquet")
    rows(2026, 3).to_parquet(cache / "2026_01.parquet")          # cache: only 2026, now bigger

    full = I._write_combined([], cache, "x.parquet", rebuild=False)
    assert full.groupby("year").size().to_dict() == {2024: 5, 2026: 3}
