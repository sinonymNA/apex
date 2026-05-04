"""
api/main.py — Unified FastAPI service for Sable Stocks.

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
from fastapi.responses import FileResponse, Response, StreamingResponse
from loguru import logger

# load_dotenv BEFORE any internal imports that create DB engines at import time
load_dotenv()

from worker.db import (  # noqa: E402
    get_gate_status,
    get_last_near_miss,
    get_latest_status,
    get_recent_anomalies,
    get_recent_near_misses,
    get_recent_trades,
    get_revenue_summary,
    get_risk_log,
    get_today_summary,
    init_db,
)

DASHBOARD_SECRET = os.getenv("DASHBOARD_SECRET", "")
_MODEL_PATH = Path(__file__).parent.parent / "models" / "regime_rf.pkl"
_DASHBOARD_HTML = Path(__file__).parent.parent / "dashboard" / "index.html"


# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Sable Stocks",
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


def _start_discord_bot():
    """Start the SABLE Discord bot as a daemon thread (ignores errors)."""
    try:
        from discord_bot import start_bot
        start_bot()
    except Exception as e:
        logger.error(f"Failed to start Discord bot: {e}")


# ── Startup ────────────────────────────────────────────────────────────────────
@app.on_event("startup")
async def startup_event():
    # 1. Initialise database
    init_db()

    # 2. Train model in background (non-blocking)
    threading.Thread(target=_train_model_if_needed, daemon=True, name="model-trainer").start()

    # 3. Start trading worker (also non-blocking)
    threading.Thread(target=_start_worker, daemon=True, name="worker-launcher").start()

    # 4. Start Discord bot (also non-blocking)
    threading.Thread(target=_start_discord_bot, daemon=True, name="discord-launcher").start()

    # 4. Log whether dashboard HTML is present
    if _DASHBOARD_HTML.exists():
        logger.info("Dashboard HTML found — serving at /")
    else:
        logger.warning("dashboard/index.html not found — GET / will return 404")

    logger.info("Sable Stocks API ready")


# ── Helper ────────────────────────────────────────────────────────────────────
def _build_human_summary(row: dict) -> str:
    """Convert a near_miss_signals row into a plain-English sentence."""
    if not row:
        return "No signal data yet — bars are being evaluated every 5 minutes."
    reason = row.get("blocked_reason", "")
    pct    = abs(row.get("percent_to_breakout") or 0.0)
    vr     = row.get("volume_ratio") or 0.0
    regime = row.get("regime", "Unknown")
    sym    = row.get("symbol", "SPY")

    summaries = {
        "regime_blocked":      f"{sym} met conditions but regime was {regime} — entry blocked.",
        "outside_time_window": f"{sym} had activity outside the 10:00–15:30 ET window.",
        "max_trades_reached":  f"{sym} had a setup but daily trade limit was already reached.",
        "risk_blocked":        f"{sym} cleared all conditions but risk engine blocked the entry.",
        "volume_not_met":      f"{sym} cleared breakout level but volume was only {vr:.2f}x (need 1.5x).",
        "breakout_not_met":    f"{sym} came within {pct:.2f}% of breakout level (volume: {vr:.2f}x).",
    }
    return summaries.get(reason, f"{sym} near-miss — {pct:.2f}% from breakout, volume {vr:.2f}x.")


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


@app.get("/api/health-check")
async def health_check():
    """Full readiness check — tests Alpaca, TradersPost, scheduler, and position state."""
    import httpx
    import pytz
    from datetime import time as dt_time

    ET = pytz.timezone("America/New_York")
    now_et = datetime.now(ET)

    # ── Alpaca ────────────────────────────────────────────────────────────────
    alpaca_status = "error: API keys not set"
    try:
        from worker.trader import _get_trading_client
        client = _get_trading_client()
        if client is None:
            alpaca_status = "error: API keys not set"
        else:
            from alpaca.trading.enums import AccountStatus
            account = client.get_account()
            if account.status == AccountStatus.ACTIVE:
                alpaca_status = "connected"
            else:
                alpaca_status = f"error: {account.status}"
    except Exception as e:
        alpaca_status = f"error: {e}"

    # ── TradersPost — lightweight ping (GET, no trade triggered) ──────────────
    traderspost_status = "error: TRADERSPOST_WEBHOOK_URL not set"
    webhook_url = os.getenv("TRADERSPOST_WEBHOOK_URL")
    if webhook_url:
        try:
            async with httpx.AsyncClient() as http:
                # HEAD/GET to the URL — any HTTP response means the endpoint is up
                r = await http.get(webhook_url, timeout=5.0)
            # 4xx is fine (endpoint exists but rejects GETs); 5xx or timeout = error
            traderspost_status = "connected" if r.status_code < 500 else f"error: HTTP {r.status_code}"
        except Exception as e:
            traderspost_status = f"error: {e}"

    # ── Market open (regular session 9:30–16:00 ET, Mon–Fri) ─────────────────
    t = now_et.time()
    market_open = (
        now_et.weekday() < 5
        and dt_time(9, 30) <= t <= dt_time(16, 0)
    )

    # ── Current position — query Alpaca directly ──────────────────────────────
    current_position = None
    try:
        from worker.trader import _get_trading_client, SYMBOL
        pos_client = _get_trading_client()
        if pos_client is not None:
            positions = pos_client.get_all_positions()
            spy = next((p for p in positions if p.symbol == SYMBOL), None)
            if spy is not None:
                current_position = {
                    "symbol": spy.symbol,
                    "qty": int(spy.qty),
                    "avg_price": float(spy.avg_entry_price),
                }
    except Exception:
        pass

    # ── Kill switch & scheduler ───────────────────────────────────────────────
    kill_switch = False
    scheduler_status = "stopped"
    try:
        from worker import trader as _trader
        kill_switch = bool(_trader._state.get("kill_switch_active", False))
        if _trader._scheduler is not None and _trader._scheduler.running:
            scheduler_status = "running"
    except Exception:
        pass

    # ── Railway deployment stamp ──────────────────────────────────────────────
    railway_deployed_at = os.getenv(
        "RAILWAY_DEPLOYMENT_ID",
        os.getenv("RAILWAY_SNAPSHOT_ID", "unknown"),
    )

    ready_to_trade = (
        alpaca_status == "connected"
        and traderspost_status == "connected"
        and not kill_switch
        and scheduler_status == "running"
    )

    return {
        "alpaca": alpaca_status,
        "traderspost": traderspost_status,
        "market_open": market_open,
        "current_position": current_position,
        "kill_switch": kill_switch,
        "scheduler": scheduler_status,
        "es_front_month": "ESM2026",
        "railway_deployed_at": railway_deployed_at,
        "ready_to_trade": ready_to_trade,
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


@app.get("/api/revenue", dependencies=[Depends(verify_auth)])
async def get_revenue():
    """Aggregated P&L, trade stats, gate progress, and system state for Sable Agents."""
    return get_revenue_summary()


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
        db_type = "PostgreSQL" if ("postgresql" in db_url or "postgres" in db_url) else "SQLite"
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


@app.get("/api/debug/last-signal", dependencies=[Depends(verify_auth)])
async def get_last_signal():
    """Most recent near-miss bar evaluation with plain-English summary."""
    row = get_last_near_miss()
    if not row:
        return {
            "message": "No signal evaluation data yet.",
            "human_summary": "No signal data yet — bars are being evaluated every 5 minutes.",
        }
    row["human_summary"] = _build_human_summary(row)
    return row


@app.get("/api/near-misses", dependencies=[Depends(verify_auth)])
async def get_near_misses(limit: int = Query(default=20, ge=1, le=100)):
    """Recent near-miss signals (bars close to triggering but blocked)."""
    return get_recent_near_misses(n=limit)


@app.post("/api/debug/test-traderspost", dependencies=[Depends(verify_auth)])
async def test_traderspost():
    """Send a test buy signal to TradersPost and return the raw response."""
    import httpx

    webhook_url = os.getenv("TRADERSPOST_WEBHOOK_URL")
    if not webhook_url:
        return {"ok": False, "error": "TRADERSPOST_WEBHOOK_URL not set"}

    payload = {"ticker": "ESM2026", "action": "buy", "contracts": 1}
    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=5.0)
        return {
            "ok": True,
            "status_code": response.status_code,
            "response": response.text,
            "payload_sent": payload,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@app.post("/api/debug/send-test-email", dependencies=[Depends(verify_auth)])
async def send_test_email():
    """Send a test email using Resend (preferred) or SMTP fallback."""
    notify = os.getenv("NOTIFY_EMAIL") or os.getenv("GMAIL_USER", "")
    if not notify:
        return {"ok": False, "error": "NOTIFY_EMAIL not set in Railway Variables"}

    resend_key = os.getenv("RESEND_API_KEY", "")
    html = """<html><body style="font-family:monospace;background:#0d0d0d;color:#e0e0e0;padding:20px">
<h2 style="color:#00e676">&#9650; APEX &mdash; Email Test</h2>
<p>Email is working. Morning brief, noon update, and EOD summary will arrive automatically.</p>
<p style="color:#888;font-size:12px">Sent via Resend API</p>
</body></html>"""

    if resend_key:
        try:
            import requests as _req
            from_addr = os.getenv("RESEND_FROM", "Sable Stocks <onboarding@resend.dev>")
            resp = _req.post(
                "https://api.resend.com/emails",
                headers={"Authorization": f"Bearer {resend_key}", "Content-Type": "application/json"},
                json={"from": from_addr, "to": [notify], "subject": "Sable Stocks — Test Email", "html": html},
                timeout=15,
            )
            if resp.status_code in (200, 201):
                logger.info(f"Test email sent via Resend to {notify}")
                return {"ok": True, "sent_to": notify, "method": "resend"}
            else:
                err = resp.json().get("message", resp.text[:100]) if resp.text else str(resp.status_code)
                logger.error(f"Resend test email failed: {err}")
                return {"ok": False, "error": f"Resend error: {err}"}
        except Exception as e:
            logger.error(f"Resend test email exception: {e}")
            return {"ok": False, "error": str(e)}

    # SMTP fallback (blocked on Railway, works locally)
    import smtplib
    from email.mime.multipart import MIMEMultipart
    from email.mime.text import MIMEText
    gmail_user = os.getenv("GMAIL_USER", "")
    gmail_pass = os.getenv("GMAIL_APP_PASSWORD", "")
    if not gmail_user or not gmail_pass:
        return {"ok": False, "error": "Set RESEND_API_KEY (recommended) or GMAIL_USER + GMAIL_APP_PASSWORD. Railway blocks SMTP — Resend is required on Railway."}
    try:
        msg = MIMEMultipart("alternative")
        msg["Subject"] = "Sable Stocks — Test Email"
        msg["From"] = gmail_user
        msg["To"] = notify
        msg.attach(MIMEText(html, "html"))
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo(); server.starttls()
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, notify, msg.as_string())
        return {"ok": True, "sent_to": notify, "method": "smtp"}
    except Exception as e:
        logger.error(f"SMTP test email failed: {e}")
        return {"ok": False, "error": str(e)}


@app.post("/api/debug/send-morning-brief", dependencies=[Depends(verify_auth)])
async def trigger_morning_brief():
    """Immediately send the morning brief email using current DB state."""
    from worker.email_report import send_morning_brief
    from worker.db import get_latest_status, get_last_near_miss, get_recent_trades

    status     = get_latest_status()
    near_miss  = get_last_near_miss()
    trades     = get_recent_trades(n=1)

    session_day     = int(status.get("session_day") or 1) if status else 1
    regime          = str(status.get("regime") or "Weak Trend") if status else "Weak Trend"
    spy_price       = float(near_miss.get("close") or 0.0)
    breakout_level  = float(near_miss.get("breakout_level") or spy_price)
    atr             = float((trades[0].get("atr_at_entry") or 2.0) if trades else 2.0)

    try:
        send_morning_brief(
            session_day=session_day,
            regime=regime,
            spy_price=spy_price,
            breakout_level=breakout_level,
            atr=atr,
        )
        notify = os.getenv("NOTIFY_EMAIL") or os.getenv("GMAIL_USER", "")
        logger.info(f"Manual morning brief sent → {notify}")
        return {"ok": True, "sent_to": notify, "session_day": session_day, "regime": regime}
    except Exception as e:
        logger.error(f"Manual morning brief failed: {e}")
        return {"ok": False, "error": str(e)}


@app.get("/api/export/trades.csv", dependencies=[Depends(verify_auth)])
async def export_trades_csv():
    """Download all trades as CSV."""
    import csv, io
    trades = get_recent_trades(n=10_000)
    cols = ["id", "entry_time", "exit_time", "symbol", "direction",
            "entry_price", "exit_price", "stop_price", "target_price",
            "shares", "pnl_dollars", "pnl_r", "atr_at_entry",
            "volume_ratio", "exit_reason", "regime"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    if trades:
        w.writerows(trades)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="apex_trades.csv"'},
    )


@app.get("/api/export/near-misses.csv", dependencies=[Depends(verify_auth)])
async def export_near_misses_csv():
    """Download all near-miss signal records as CSV."""
    import csv, io
    rows = get_recent_near_misses(n=10_000)
    cols = ["id", "timestamp", "symbol", "close", "breakout_level",
            "percent_to_breakout", "volume", "required_volume",
            "volume_ratio", "regime", "blocked_reason", "trades_today"]
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    if rows:
        w.writerows(rows)
    return StreamingResponse(
        iter([buf.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": 'attachment; filename="apex_near_misses.csv"'},
    )


@app.get("/api/debug/pipeline-test", dependencies=[Depends(verify_auth)])
async def pipeline_test():
    """Run the full bar pipeline end-to-end and return verbose diagnostics."""
    import datetime as _dt
    import yfinance as yf
    from worker.strategy import MomentumBreakout
    from worker.db import log_near_miss, get_last_near_miss

    out = {}

    # Step 1: yfinance fetch
    try:
        df = yf.download("SPY", period="2d", interval="5m", auto_adjust=True, progress=False)
        raw_cols = list(df.columns)
        out["step1_fetch"] = {"ok": True, "rows": len(df), "columns_raw": str(raw_cols)}

        if hasattr(df.columns, "levels"):
            for _lvl in range(df.columns.nlevels):
                _cand = df.columns.get_level_values(_lvl)
                if "Close" in _cand:
                    df.columns = _cand
                    break
            out["step1_fetch"]["columns_after_fix"] = list(df.columns)
        out["step1_fetch"]["last_close"] = float(df["Close"].iloc[-1]) if "Close" in df.columns else None
    except Exception as e:
        return {"step1_fetch": {"ok": False, "error": str(e)}}

    # Step 2: compute_indicators
    try:
        strat = MomentumBreakout()
        df_ind = strat.compute_indicators(df)
        valid = df_ind.dropna(subset=["high_20", "volume_avg", "atr14"])
        out["step2_indicators"] = {
            "ok": True,
            "total_rows": len(df_ind),
            "valid_rows": len(valid),
            "last_high20": float(valid["high_20"].iloc[-1]) if not valid.empty else None,
            "last_atr14": float(valid["atr14"].iloc[-1]) if not valid.empty else None,
        }
    except Exception as e:
        return {**out, "step2_indicators": {"ok": False, "error": str(e)}}

    # Step 3: evaluate_signal_state
    try:
        nm = strat.evaluate_signal_state(df_ind)
        out["step3_signal_state"] = nm if nm else {"is_none": True}
    except Exception as e:
        return {**out, "step3_signal_state": {"ok": False, "error": str(e)}}

    # Step 4: DB write test
    try:
        log_near_miss({
            "timestamp": _dt.datetime.now(_dt.timezone.utc),
            "symbol": "SPY",
            "blocked_reason": "pipeline_test",
            "regime": "test",
            "trades_today": 0,
            **(nm or {}),
        })
        last = get_last_near_miss()
        out["step4_db_write"] = {"ok": True, "last_reason": last.get("blocked_reason")}
    except Exception as e:
        out["step4_db_write"] = {"ok": False, "error": str(e)}

    return out



@app.get("/", include_in_schema=False)
async def serve_root():
    if _DASHBOARD_HTML.exists():
        return FileResponse(str(_DASHBOARD_HTML))
    return {"message": "Sable Stocks — place dashboard/index.html to serve the UI"}


@app.get("/{path:path}", include_in_schema=False)
async def serve_spa(path: str):
    if _DASHBOARD_HTML.exists():
        return FileResponse(str(_DASHBOARD_HTML))
    raise HTTPException(status_code=404, detail="Not found")
