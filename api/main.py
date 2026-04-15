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
from fastapi.responses import FileResponse
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
_DASHBOARD_HTML = Path(__file__).parent.parent / "dashboard" / "index.html"


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

    # 4. Log whether dashboard HTML is present
    if _DASHBOARD_HTML.exists():
        logger.info("Dashboard HTML found — serving at /")
    else:
        logger.warning("dashboard/index.html not found — GET / will return 404")

    logger.info("Apex Trading System API ready")


# ── API Routes ─────────────────────────────────────────────────────────────────
@app.get("/api/health")
async def health():
    """Health check — no auth required."""
    return {
        "status": "ok",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_ready": _MODEL_PATH.exists(),
        "dashboard_ready": _DASHBOARD_HTML.exists(),
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


@app.get("/api/diagnostics", dependencies=[Depends(verify_auth)])
async def diagnostics():
    """Full systems check — verifies every component is reachable and configured."""
    checks = {}

    # 1. Database
    try:
        get_latest_status()  # any query works; will return None on empty DB
        from worker.db import engine
        with engine.connect() as conn:
            conn.execute(__import__("sqlalchemy").text("SELECT 1"))
        db_url = os.getenv("DATABASE_URL", "sqlite")
        db_type = "PostgreSQL" if db_url.startswith("postgres") else "SQLite"
        checks["database"] = {"ok": True, "detail": f"{db_type} connected"}
    except Exception as e:
        checks["database"] = {"ok": False, "detail": str(e)[:120]}

    # 2. Alpaca paper trading API
    alpaca_key = os.getenv("ALPACA_API_KEY", "")
    alpaca_secret = os.getenv("ALPACA_SECRET_KEY", "")
    if not alpaca_key or not alpaca_secret:
        checks["alpaca"] = {"ok": False, "detail": "ALPACA_API_KEY / ALPACA_SECRET_KEY not set"}
    else:
        try:
            from alpaca.trading.client import TradingClient
            client = TradingClient(alpaca_key, alpaca_secret, paper=True)
            account = client.get_account()
            equity = float(account.equity)
            checks["alpaca"] = {
                "ok": True,
                "detail": f"Paper account ACTIVE — equity ${equity:,.2f}",
            }
        except Exception as e:
            checks["alpaca"] = {"ok": False, "detail": f"Connection failed: {str(e)[:100]}"}

    # 3. Trading worker thread
    worker_alive = any(t.name == "apex-trader" for t in threading.enumerate())
    checks["worker"] = {
        "ok": worker_alive,
        "detail": "Scheduler running (will trade Mon–Fri 10:00–15:30 ET)" if worker_alive else "Thread not found — check Railway logs",
    }

    # 4. Regime model
    model_ok = _MODEL_PATH.exists()
    checks["regime_model"] = {
        "ok": model_ok,
        "detail": "Model loaded" if model_ok else "Training in background — defaulting to Weak Trend",
    }

    # 5. Email
    gmail_user = os.getenv("GMAIL_USER", "")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    notify = os.getenv("NOTIFY_EMAIL", "")
    email_ok = bool(gmail_user and gmail_pass and notify)
    if email_ok:
        checks["email"] = {"ok": True, "detail": f"{gmail_user} → {notify}"}
    else:
        missing = [v for v, k in [("GMAIL_USER", gmail_user), ("GMAIL_APP_PASSWORD", gmail_pass), ("NOTIFY_EMAIL", notify)] if not k]
        checks["email"] = {"ok": False, "detail": f"Missing: {', '.join(missing)}"}

    # 6. Dashboard secret
    secret_set = bool(os.getenv("DASHBOARD_SECRET"))
    checks["auth"] = {
        "ok": secret_set,
        "detail": "DASHBOARD_SECRET configured" if secret_set else "Not set — anyone can access the dashboard",
    }

    all_ok = all(v["ok"] for v in checks.values())
    return {"all_ok": all_ok, "checks": checks, "timestamp": datetime.now(timezone.utc).isoformat()}


# ── Dashboard catch-all (must be LAST so /api/* routes take precedence) ────────
@app.get("/", include_in_schema=False)
async def serve_root():
    if _DASHBOARD_HTML.exists():
        return FileResponse(str(_DASHBOARD_HTML))
    return {"message": "Apex Trading System — place dashboard/index.html to serve the UI"}


@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str):
    if _DASHBOARD_HTML.exists():
        return FileResponse(str(_DASHBOARD_HTML))
    raise HTTPException(status_code=404, detail="Not found")
