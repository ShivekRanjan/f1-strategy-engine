"""Phase 5 — FastAPI service. STUB (thin layer; no modelling logic here).

Exposes the engine over HTTP so it can be demoed as a service (the ML-eng
signal). This file must stay thin: it validates requests, calls into
``f1se.sim``/``f1se.models``, and serialises the response. Any modelling logic
that creeps in here is a bug.
"""

from __future__ import annotations

# from fastapi import FastAPI  # enabled in Phase 5 (pip install .[app])
#
# app = FastAPI(title="F1 Strategy Engine", version="0.1.0")
#
# @app.post("/recommend")
# def recommend(...):
#     """Return a strategy recommendation with uncertainty. TODO Phase 5."""
#     ...
