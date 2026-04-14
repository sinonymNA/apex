"""
api/main.py — FastAPI backend for Apex Trading System.

CORS open. Auth via X-Dashboard-Secret header on all routes
except /api/health. Reads DASHBOARD_SECRET from env.

Routes:
  GET /api/health         -> {status, timestamp}
  GET /api/status         -> latest system_status row
  GET /api/summary        -> today's daily_summary row
  GET /api/trades         -> recent trades (limit param)
  GET /api/gates          -> latest gate_status row
  GET /api/risk-log       -> recent risk_checks (limit param)
  GET /api/anomalies      -> recent anomalies (limit param)
"""
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware

# load_dotenv BEFORE importing db so DATABASE_URL is set when engine is created
load_dotenv()

from worker.db import (  # noqa: E402
    get_gate_status,
    get_latest_status,
    get_recent_anomalies,
    get_recent_trades,
    get_risk_log,
    get_today_summary,
    init_db,
)

DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "")

app = FastAPI(
    title="Apex Trading System API",
    version="1.0.0",
    description="REST API for Apex Trading System dashboard",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Auth dependency ────────────────────────────────────────────────────────────
async def verify_auth(x_dashboard_secret: str = Header(default=None)):
    """Validate X-Dashboard-Secret header."""
    if not DASHBOARD_SECRET:
        # If secret not configured, allow all (dev mode)
        return
    if x_dashboard_secret != DASHBOARD_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Dashboard-Secret header")


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    init_db()


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check — no auth required."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "service": "apex-trading-api",
    }


@app.get("/api/status", dependencies=[Depends(verify_auth)])
async def get_status():
    """Return the latest system_status row."""
    data = get_latest_status()
    if not data:
        return {"status": "NO_DATA", "message": "No status rows yet"}
    return data


@app.get("/api/summary", dependencies=[Depends(verify_auth)])
async def get_summary():
    """Return today's daily_summary row (or computed live aggregate)."""
    data = get_today_summary()
    if not data:
        return {"message": "No trades today"}
    return data


@app.get("/api/trades", dependencies=[Depends(verify_auth)])
async def get_trades(limit: int = Query(default=50, ge=1, le=200)):
    """Return the N most recent completed trades."""
    return get_recent_trades(n=limit)


@app.get("/api/gates", dependencies=[Depends(verify_auth)])
async def get_gates():
    """Return the most recent gate_status row."""
    data = get_gate_status()
    if not data:
        return {"message": "No gate data yet"}
    return data


@app.get("/api/risk-log", dependencies=[Depends(verify_auth)])
async def get_risk_log_route(limit: int = Query(default=20, ge=1, le=100)):
    """Return the N most recent risk check records."""
    return get_risk_log(n=limit)


@app.get("/api/anomalies", dependencies=[Depends(verify_auth)])
async def get_anomalies(limit: int = Query(default=10, ge=1, le=50)):
    """Return the N most recent anomaly records."""
    return get_recent_anomalies(n=limit)
