"""Tests for the news aggregator.

Parsing is tested against a static RSS fixture (no network); the live fetch is
marked ``network`` so the default suite stays offline and fast.
"""

from __future__ import annotations

import json

import pytest

from f1se.standalone.news import _clean, fetch_news, parse_feed

_RSS = b"""<?xml version="1.0"?>
<rss version="2.0"><channel>
  <title>Demo F1</title>
  <item>
    <title>Verstappen takes pole in Austria</title>
    <link>https://example.com/a</link>
    <pubDate>Sun, 29 Jun 2025 14:00:00 GMT</pubDate>
    <description><![CDATA[<p>A <b>dramatic</b> qualifying session.</p>]]></description>
  </item>
  <item>
    <title>Hamilton fastest in final practice</title>
    <link>https://example.com/b</link>
    <pubDate>Sat, 28 Jun 2025 11:30:00 GMT</pubDate>
    <description>Short summary here.</description>
  </item>
</channel></rss>"""


def test_parse_feed_extracts_fields_and_cleans_html():
    items = parse_feed(_RSS, "Demo")
    assert len(items) == 2
    first = items[0]
    assert first["title"] == "Verstappen takes pole in Austria"
    assert first["link"] == "https://example.com/a"
    assert first["source"] == "Demo"
    assert isinstance(first["ts"], int) and first["ts"] > 0
    # HTML tags stripped from the summary
    assert "<" not in first["summary"] and "dramatic" in first["summary"]
    json.dumps(items)


def test_parse_feed_skips_items_without_title_or_link():
    bad = b"""<?xml version="1.0"?><rss version="2.0"><channel>
      <item><link>https://x/</link></item>
      <item><title>No link</title></item>
    </channel></rss>"""
    assert parse_feed(bad, "Demo") == []


def test_clean_truncates_long_text():
    long = "word " * 100
    out = _clean(long, limit=40)
    assert len(out) <= 40 and out.endswith("…")


def test_cache_holds_full_list_and_limit_is_per_request():
    """A small-limit call must not shrink what a later larger-limit call returns."""
    import time

    from f1se.standalone import news as N

    full = [{"title": f"t{i}", "link": f"l{i}", "source": "S", "ts": 2000 - i, "summary": ""}
            for i in range(20)]
    N._CACHE.update(at=time.time(), payload={"items": full, "sources": ["S"], "fetched_at": 0.0})
    try:
        assert len(N.fetch_news(limit=3)["items"]) == 3
        big = N.fetch_news(limit=15)
        assert len(big["items"]) == 15 and big["cached"] is True
    finally:
        N._CACHE.update(at=0.0, payload=None)


@pytest.mark.network
def test_fetch_news_live_returns_items():
    payload = fetch_news(limit=10, force=True)
    assert payload["items"], "expected at least one live headline"
    assert payload["sources"]
    top = payload["items"][0]
    assert {"title", "link", "source", "ts", "summary"} <= set(top)
    # newest-first ordering
    ts = [i["ts"] or 0 for i in payload["items"]]
    assert ts == sorted(ts, reverse=True)
