"""Phase 5 — FastAPI service (thin layer over :class:`f1se.engine.StrategyEngine`).

This file is intentionally thin: it validates requests, calls the engine, and
serialises the response. No modelling logic lives here — that's the engine's job.
Run:  uvicorn f1se.api:app --reload
"""

from __future__ import annotations

import os
from functools import lru_cache

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from f1se.engine import StrategyEngine

app = FastAPI(
    title="F1 Strategy Engine",
    version="0.1.0",
    description="Recommends pit strategy (when to stop, which compounds) with quantified uncertainty.",
)

# The React frontend runs on a different origin in dev (Vite :5173) and on its
# own host in prod. This is a public, read-only API with no credentials, so a
# permissive default is fine; override with F1SE_CORS_ORIGINS (comma-separated).
_origins = os.environ.get("F1SE_CORS_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in _origins],
    allow_methods=["*"],
    allow_headers=["*"],
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
    season: int | None = Field(None, description="e.g. 2026 selects the new-regs model")
    sc_scale: float = Field(1.0, ge=0.0, le=3.0, description="scale SC hazard (0 = no SC)")


class SimulateRequest(BaseModel):
    track: str
    compounds: list[str]
    pit_laps: list[int]
    use_cliff: bool = True
    n_runs: int = Field(4000, ge=200, le=20000)
    season: int | None = None


class LiveRequest(BaseModel):
    track: str
    current_lap: int = Field(..., ge=0)
    current_compound: str
    tyre_age: int = Field(..., ge=0)
    compounds_used: list[str] = []
    objective: str = "mean"
    use_cliff: bool = True
    n_runs: int = Field(2000, ge=200, le=20000)
    season: int | None = None


class LiveReplayRequest(BaseModel):
    track: str
    season: int
    driver: str
    current_lap: int = Field(..., ge=1)
    n_runs: int = Field(2000, ge=200, le=20000)
    use_cliff: bool = True
    objective: str = "mean"


class UndercutRequest(BaseModel):
    track: str
    current_lap: int = Field(..., ge=1)
    gap_s: float
    your_compound: str
    your_age: int = Field(..., ge=0)
    your_new_compound: str
    rival_compound: str
    rival_age: int = Field(..., ge=0)
    rival_new_compound: str
    rival_pit_lap: int = Field(..., ge=1)
    season: int | None = None
    n_runs: int = Field(2000, ge=200, le=20000)


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
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/recommend")
def recommend(req: RecommendRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.recommend(
            req.track, objective=req.objective, use_cliff=req.use_cliff,
            max_stops=req.max_stops, n_runs=req.n_runs, top_k=req.top_k,
            season=req.season, sc_scale=req.sc_scale,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/recommend_live")
def recommend_live(req: LiveRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.recommend_live(
            req.track, req.current_lap, req.current_compound, req.tyre_age,
            compounds_used=tuple(req.compounds_used), objective=req.objective,
            use_cliff=req.use_cliff, n_runs=req.n_runs, season=req.season,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/simulate")
def simulate(req: SimulateRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.simulate(
            req.track, tuple(req.compounds), tuple(req.pit_laps),
            use_cliff=req.use_cliff, n_runs=req.n_runs, season=req.season,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.post("/undercut")
def undercut(req: UndercutRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.undercut(
            req.track, current_lap=req.current_lap, gap_s=req.gap_s,
            your_compound=req.your_compound, your_age=req.your_age,
            your_new_compound=req.your_new_compound, rival_compound=req.rival_compound,
            rival_age=req.rival_age, rival_new_compound=req.rival_new_compound,
            rival_pit_lap=req.rival_pit_lap, season=req.season, n_runs=req.n_runs,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


# ---- race-replay (live view) data endpoints --------------------------------
@app.get("/seasons")
def all_seasons(engine: StrategyEngine = Depends(get_engine)) -> dict:
    """Every season in the data (for season-first navigation)."""
    return {"seasons": engine.all_seasons()}


@app.get("/circuits/{season}")
def circuits_for_season(season: int, engine: StrategyEngine = Depends(get_engine)) -> dict:
    """Circuits raced in a given season (the season-specific shortlist)."""
    return {"season": season, "circuits": engine.circuits_for_season(season)}


@app.get("/seasons/{track}")
def seasons(track: str, engine: StrategyEngine = Depends(get_engine)) -> dict:
    return {"track": track, "seasons": engine.seasons(track)}


@app.get("/drivers/{track}/{season}")
def drivers(track: str, season: int, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return {"track": track, "season": season, "drivers": engine.replay_drivers(track, season)}
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.get("/laps/{track}/{season}/{driver}")
def laps(track: str, season: int, driver: str,
         engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.lap_history(track, season, driver)
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e


@app.post("/live")
def live(req: LiveReplayRequest, engine: StrategyEngine = Depends(get_engine)) -> dict:
    try:
        return engine.live_replay(
            req.track, req.season, req.driver, req.current_lap,
            n_runs=req.n_runs, use_cliff=req.use_cliff, objective=req.objective,
        )
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e)) from e
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@app.get("/outcome")
def outcome() -> dict:
    from f1se.standalone.outcome import cached_outcome

    payload = cached_outcome()
    if payload is None:
        raise HTTPException(status_code=404, detail="results dataset not available")
    return payload
