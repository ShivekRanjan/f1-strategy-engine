"""Central configuration — most importantly, the FastF1 cache.

FastF1 is slow and rate-limited. Caching stays on everywhere: every entry point
that touches FastF1 imports :func:`enable_cache` so we never hammer the live API
twice for the same session. The cache directory is git-ignored and regenerable.
"""

from __future__ import annotations

import os
from pathlib import Path

# Project root = two levels up from this file (src/f1se/config.py -> repo root).
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Cache dir can be overridden via env var (useful for CI / containers where the
# repo may be read-only). Defaults to data/cache under the project root.
CACHE_DIR = Path(os.environ.get("F1SE_CACHE_DIR", PROJECT_ROOT / "data" / "cache"))

_cache_enabled = False


def enable_cache(cache_dir: Path | str | None = None) -> Path:
    """Enable the FastF1 on-disk cache (idempotent).

    Parameters
    ----------
    cache_dir
        Override the default cache location. Created if it does not exist.

    Returns
    -------
    Path
        The directory FastF1 is caching into.
    """
    global _cache_enabled

    target = Path(cache_dir) if cache_dir is not None else CACHE_DIR
    target.mkdir(parents=True, exist_ok=True)

    if not _cache_enabled:
        # Imported lazily so that non-FastF1 code paths (and tests that stub the
        # data layer) don't pay the heavy import cost.
        import fastf1

        fastf1.Cache.enable_cache(str(target))
        _cache_enabled = True

    return target
