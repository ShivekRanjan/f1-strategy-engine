"""F1 news aggregator — headlines from reputable F1 RSS feeds.

We surface *headline, source, timestamp, and a link out* only — never the full
article text (that stays the publisher's). Feeds are fetched in parallel with a
short timeout, merged newest-first, and cached briefly (news moves fast, but not
second-to-second). One feed failing never breaks the list.
"""

from __future__ import annotations

import calendar
import re
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from html import unescape

import feedparser

# Reputable F1 outlets that publish a working RSS/Atom feed (verified reachable).
FEEDS: list[tuple[str, str]] = [
    ("The Race", "https://www.the-race.com/feed/"),
    ("Autosport", "https://www.autosport.com/rss/f1/news/"),
    ("Motorsport.com", "https://www.motorsport.com/rss/f1/news/"),
    ("RaceFans", "https://www.racefans.net/feed/"),
    ("Formula1.com", "https://www.formula1.com/en/latest/all.xml"),
]

_UA = "Mozilla/5.0 (compatible; F1SE/0.1; +https://github.com/ShivekRanjan/f1-strategy-engine)"
_TIMEOUT_S = 7
_TTL_S = 600  # 10 min — news updates, but not per-second
_CACHE: dict = {"at": 0.0, "payload": None}

_TAG_RE = re.compile(r"<[^>]+>")


def _clean(text: str | None, limit: int = 240) -> str:
    """Strip HTML tags/entities from a feed summary and truncate to a snippet."""
    if not text:
        return ""
    txt = unescape(_TAG_RE.sub("", text)).strip()
    txt = re.sub(r"\s+", " ", txt)
    return txt[: limit - 1].rstrip() + "…" if len(txt) > limit else txt


def parse_feed(content: bytes | str, source: str) -> list[dict]:
    """Parse one feed's bytes into a list of headline dicts (no network)."""
    parsed = feedparser.parse(content)
    items = []
    for e in parsed.entries:
        title = unescape((e.get("title") or "").strip())
        link = e.get("link") or ""
        if not title or not link:
            continue
        tstruct = e.get("published_parsed") or e.get("updated_parsed")
        ts = calendar.timegm(tstruct) if tstruct else None
        items.append({
            "title": title,
            "link": link,
            "source": source,
            "ts": ts,
            "summary": _clean(e.get("summary")),
        })
    return items


def _fetch_one(feed: tuple[str, str]) -> list[dict]:
    source, url = feed
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_TIMEOUT_S) as resp:  # noqa: S310 (fixed hosts)
            return parse_feed(resp.read(), source)
    except Exception:  # pragma: no cover - network; one bad feed must not break the rest
        return []


def fetch_news(limit: int = 40, *, force: bool = False) -> dict:
    """Aggregate all feeds newest-first (cached ``_TTL_S``). Never raises.

    The cache holds the *full* merged list; ``limit`` is applied per request, so a
    small-limit call can't shrink what a later larger-limit call returns.
    """
    now = time.time()
    cached = _CACHE["payload"]
    if not force and cached and now - _CACHE["at"] < _TTL_S:
        return {"items": cached["items"][:limit], "sources": cached["sources"],
                "fetched_at": cached["fetched_at"], "cached": True}

    items: list[dict] = []
    sources_ok: list[str] = []
    with ThreadPoolExecutor(max_workers=len(FEEDS)) as ex:
        for feed, res in zip(FEEDS, ex.map(_fetch_one, FEEDS)):
            if res:
                items.extend(res)
                sources_ok.append(feed[0])

    items.sort(key=lambda x: x["ts"] or 0, reverse=True)
    if items:  # only cache a non-empty result (transient network blips retry next call)
        _CACHE.update(at=now, payload={"items": items, "sources": sources_ok, "fetched_at": now})
    return {"items": items[:limit], "sources": sources_ok, "fetched_at": now, "cached": False}
