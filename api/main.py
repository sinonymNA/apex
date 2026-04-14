"""
api/main.py — Unified FastAPI service for Apex Trading System.

Single Railway service: serves dashboard, API routes, and runs the trading
worker in a background daemon thread. Zero extra configuration required.

On startup:
  1. Initialise DB (SQLite fallback if DATABASE_URL not set)
  2. Train regime model if pkl missing (uses synthetic data if yfinance unavailable)
  3. Mount dashboard/build as static files at / (if built)
  4. Start the APScheduler trading worker in a background thread
  5. All API routes are under /api/*

Auth: X-Dashboard-Secret header required on /api/* routes (except /api/health).
      Set DASHBOARD_SECRET env var. If not set, auth is bypassed (dev mode).
"""
import os
import threading
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from loguru import logger

# load_dotenv BEFORE any internal imports that create DB engines at import time
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
_MODEL_PATH = Path(__file__).parent.parent / "models" / "regime_rf.pkl"
_DASHBOARD_BUILD = Path(__file__).parent.parent / "dashboard" / "build"


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Apex Trading System",
    version="1.0.0",
    docs_url="/api/docs",
    redoc_url=None,
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
    if not DASHBOARD_SECRET:
        return  # dev mode — no secret configured
    if x_dashboard_secret != DASHBOARD_SECRET:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Dashboard-Secret header")


# ── Background tasks ───────────────────────────────────────────────────────────
def _train_model_if_needed():
    """Train the regime classifier in a background thread if pkl is missing."""
    if _MODEL_PATH.exists():
        return
    logger.info("No regime model found — training with backtest data...")
    try:
        from backtest.run import main as run_backtest
        run_backtest()
        logger.info("Regime model training complete")
    except Exception as e:
        logger.warning(f"Regime model training failed: {e} — using 'Weak Trend' default")


def _start_worker():
    """Start the trading scheduler as a daemon thread (ignores errors)."""
    try:
        from worker.trader import start_background
        start_background()
    except Exception as e:
        logger.error(f"Failed to start trading worker: {e}")


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    # 1. Initialise database
    init_db()

    # 2. Train model in background (non-blocking)
    threading.Thread(target=_train_model_if_needed, daemon=True, name="model-trainer").start()

    # 3. Start trading worker (also non-blocking)
    threading.Thread(target=_start_worker, daemon=True, name="worker-launcher").start()

    # 4. Mount React dashboard as static files (if built)
    if _DASHBOARD_BUILD.exists():
        app.mount("/", StaticFiles(directory=str(_DASHBOARD_BUILD), html=True), name="dashboard")
        logger.info("Dashboard static files mounted at /")
    else:
        logger.info("No dashboard build found — serving API only")

    logger.info("Apex Trading System API ready")


# ── API Routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check — no auth required."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_ready": _MODEL_PATH.exists(),
        "dashboard_built": _DASHBOARD_BUILD.exists(),
    }


@app.get("/api/status", dependencies=[Depends(verify_auth)])
async def get_status():
    data = get_latest_status()
    return data or {"status": "NO_DATA", "message": "No status rows yet"}


@app.get("/api/summary", dependencies=[Depends(verify_auth)])
async def get_summary():
    data = get_today_summary()
    return data or {"message": "No trades today"}


@app.get("/api/trades", dependencies=[Depends(verify_auth)])
async def get_trades(limit: int = Query(default=50, ge=1, le=200)):
    return get_recent_trades(n=limit)


@app.get("/api/gates", dependencies=[Depends(verify_auth)])
async def get_gates():
    data = get_gate_status()
    return data or {"message": "No gate data yet"}


@app.get("/api/risk-log", dependencies=[Depends(verify_auth)])
async def get_risk_log_route(limit: int = Query(default=20, ge=1, le=100)):
    return get_risk_log(n=limit)


@app.get("/api/anomalies", dependencies=[Depends(verify_auth)])
async def get_anomalies(limit: int = Query(default=10, ge=1, le=50)):
    return get_recent_anomalies(n=limit)
