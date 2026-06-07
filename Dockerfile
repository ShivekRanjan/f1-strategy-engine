# F1 Strategy Engine — container image.
# Defaults to the Streamlit UI; override the command to run the API instead:
#   docker run -p 8000:8000 f1se uvicorn f1se.api:app --host 0.0.0.0 --port 8000
FROM python:3.12-slim

WORKDIR /app

# Only the [app] extra is needed at runtime (engine uses base + numpy/pandas;
# no xgboost/torch). Install deps first for better layer caching.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --no-cache-dir -e ".[app]"

# App code + the small pre-built datasets, so the container starts instantly
# without hitting FastF1 (the regenerable cache is NOT shipped).
COPY app ./app
COPY data/processed/dry_laps.parquet \
     data/processed/track_status.parquet \
     data/processed/race_laps.parquet \
     ./data/processed/

# FastF1 cache dir is writable (only used if data is regenerated at runtime).
ENV F1SE_CACHE_DIR=/tmp/f1se_cache

EXPOSE 8501
# Honour $PORT when a host injects one (Render/Fly), else default to 8501.
CMD ["sh", "-c", "streamlit run app/streamlit_app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true"]
