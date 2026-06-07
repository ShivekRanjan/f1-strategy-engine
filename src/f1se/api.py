"""Phase 5 — FastAPI service (thin layer over :class:`f1se.engine.StrategyEngine`).

This file is intentionally thin: it validates requests, calls the engine, and
serialises the response. No modelling logic lives here — that's the engine's job.
Run:  uvicorn f1se.api:app --reload
"""

from __future__ import annotations

from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from f1se.engine import StrategyEngine

app = FastAPI(
    title="F1 Strategy Engine",
    version="0.1.0",
    description="Recommends pit strategy (when to stop, which compounds) with quantified uncertainty.",
)


@lru_cache(maxsize=1)
def get_engine() -> StrategyEngine:
    """Build the engine once (cached) from the processed datasets."""
    return StrategyEngine.from_processed()


class RecommendRequest(BaseModel):
    track: str
    objective: str = Field("mean", description="mean | median | p85 (risk-averse)")
    use_cliff: bool = True
    max_stops: int = Field(2, ge=1, le=3)
    n_runs: int = Field(2000, ge=200, le=20000)
    top_k: int = Field(5, ge=1, le=20)


class SimulateRequest(BaseModel):
    track: str
    compounds: list[str]
    pit_laps: list[int]
    use_cliff: bool = True
    n_runs: int = Field(4000, ge=200, le=20000)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/tracks")
def tracks(engine: StrategyEngine = Depends(get_engine)) -> dict:
    return {"tracks": engine.tracks()}


@app.get("/race/{track}")
def race_info(track: str, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.race_info(track)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.post("/recommend")
def recommend(req: RecommendRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.recommend(
            req.track, objective=req.objective, use_cliff=req.use_cliff,
            max_stops=req.max_stops, n_runs=req.n_runs, top_k=req.top_k,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/simulate")
def simulate(req: SimulateRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.simulate(
            req.track, tuple(req.compounds), tuple(req.pit_laps),
            use_cliff=req.use_cliff, n_runs=req.n_runs,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
