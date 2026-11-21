"""
api.py - tiny web API around forecast_service.py

    pip install fastapi uvicorn lightgbm pandas numpy requests
    export EIA_API_KEY=...
    uvicorn api:app --host 0.0.0.0 --port 8000

The website then calls  GET https://<your-server>/forecast  and gets the JSON.
EIA publishes once a week, so the result is cached and only recomputed
every 6 hours (or immediately with /forecast?refresh=true).
"""
import time

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from forecast_service import run_forecast

app = FastAPI(title="Tech Fem crude inventory forecast")
app.add_middleware(CORSMiddleware, allow_origins=["https://techfem.haweyaadv.ly"], allow_methods=["GET"])

CACHE_SECONDS = 6 * 3600
_cache = {"result": None, "time": 0.0}


@app.get("/forecast")
def forecast(refresh: bool = False):
    if refresh or _cache["result"] is None or time.time() - _cache["time"] > CACHE_SECONDS:
        _cache["result"] = run_forecast(output_file=None)
        _cache["time"] = time.time()
    return _cache["result"]
