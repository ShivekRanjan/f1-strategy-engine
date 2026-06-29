# F1 Strategy Engine — API container (FastAPI). The UI is the React app in
# frontend/, deployed separately (Vercel/Netlify) and pointed at this API via
# VITE_API_BASE. Run locally:
#   docker build -t f1se-api . && docker run -p 8000:8000 f1se-api
FROM python:3.12-slim

WORKDIR /app

# Only the [app] extra is needed at runtime, plus scikit-learn for the outcome
# predictor (engine uses base + numpy/pandas; no xgboost/torch). Deps first for
# better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[app]" "scikit-learn>=1.5,<2"

# The small pre-built datasets the engine, outcome predictor, and torch-free
# next-lap forecaster load, so the container starts instantly without FastF1.
COPY data/processed/dry_laps.parquet \
     data/processed/track_status.parquet \
     data/processed/race_laps.parquet \
     data/processed/results.parquet \
     data/processed/lstm_nextlap.npz \
     ./data/processed/

# FastF1 cache dir is writable (only used if data is regenerated at runtime).
ENV F1SE_CACHE_DIR=/tmp/f1se_cache

EXPOSE 8000
# Honour $PORT when a host injects one (Render/Fly), else default to 8000.
CMD ["sh", "-c", "uvicorn f1se.api:app --host 0.0.0.0 --port ${PORT:-8000}"]
