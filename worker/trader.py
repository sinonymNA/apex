"""
worker/trader.py — Main trading worker for Sable Stocks.

Runs Monday-Friday 9:25 AM - 4:05 PM ET using APScheduler.
Fetches SPY bars from Alpaca (yfinance fallback), runs MomentumBreakout strategy,
checks regime and risk, places paper orders via Alpaca, and logs everything.

Deploy as a Railway worker process. Handles SIGTERM gracefully.
"""
import json
import os
import signal
import sys
from datetime import datetime, time, timezone
from pathlib import Path

import pytz
import yfinance as yf
from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger
from apscheduler.triggers.interval import IntervalTrigger
from dotenv import load_dotenv
from loguru import logger

# load_dotenv BEFORE any internal imports that create DB engines
load_dotenv()

# Internal imports (after load_dotenv so DATABASE_URL is set)
from worker import db, risk
from worker.strategy import MomentumBreakout
from worker.email_report import send_daily_report, send_morning_brief, send_noon_update
from diagnostics.analyzer import analyze_anomaly
from models.regime_classifier import RegimeClassifier

# ── Configuration ─────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
SYMBOL = "SPY"
PAPER = True
SESSION_START = time(9, 25)
SESSION_END = time(16, 5)

# ── Global mutable state ──────────────────────────────────────────────────────
_state = {
    "daily_pnl": 0.0,
    "trade_count": 0,
    "consecutive_losses": 0,
    "peak_equity": 100_000.0,
    "current_equity": 100_000.0,
    "current_position": None,   # dict or None
    "is_paused": False,
    "kill_switch_active": False,
    "regime": "Weak Trend",
    "daily_trades": [],         # list of completed trade dicts for the day
    "session_day": 1,           # day number in 20-day evaluation period
}

# Lazy-initialized Alpaca client (not created until first use)
_trading_client = None
_strategy = MomentumBreakout()
_classifier = RegimeClassifier()
_scheduler = None


def _get_trading_client():
    """Lazy initialize the Alpaca TradingClient."""
    global _trading_client
    if _trading_client is None:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
        if not api_key or not secret_key:
            logger.warning("Alpaca API keys not set — order placement will be simulated")
            return None
        _trading_client = TradingClient(api_key, secret_key, paper=PAPER)
    return _trading_client


# ── Helper functions ──────────────────────────────────────────────────────────
def _now_et() -> datetime:
    return datetime.now(ET)


def _is_session_hours() -> bool:
    t = _now_et().time()
    return SESSION_START <= t <= SESSION_END


def _fetch_bars_alpaca() -> "pd.DataFrame | None":
    """Fetch 5-min SPY bars from Alpaca market data API (primary source)."""
    import pandas as pd
    api_key = os.getenv("ALPACA_API_KEY", "")
    secret_key = os.getenv("ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
    if not api_key or not secret_key:
        return None
    try:
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from datetime import timedelta

        client = StockHistoricalDataClient(api_key, secret_key)
        start = datetime.now(timezone.utc) - timedelta(days=5)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return None
        # Strip symbol level from MultiIndex (symbol, timestamp) → timestamp index
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(SYMBOL, level="symbol")
        # Alpaca returns lowercase; rename to standard uppercase OHLCV
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning(f"Alpaca data fetch failed: {e}")
        return None


def _fetch_bars_yfinance() -> "pd.DataFrame | None":
    """Fetch 5-min SPY bars from yfinance (fallback)."""
    try:
        df = yf.download(SYMBOL, period="2d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if hasattr(df.columns, "levels"):
            for _lvl in range(df.columns.nlevels):
                _candidate = df.columns.get_level_values(_lvl)
                if "Close" in _candidate:
                    df.columns = _candidate
                    break
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning(f"yfinance fetch failed: {e}")
        return None


def _fetch_bars():
    """Fetch recent SPY 5-minute bars. Tries Alpaca first, falls back to yfinance."""
    df = _fetch_bars_alpaca()
    if df is not None:
        return df
    logger.warning("Alpaca data unavailable — trying yfinance fallback")
    df = _fetch_bars_yfinance()
    if df is None:
        logger.error("Both Alpaca and yfinance data sources failed")
    return df


def _place_buy_order(signal: dict) -> dict | None:
    """
    Place a paper buy order via Alpaca.
    Returns order dict on success, None on failure.
    Simulates the order if Alpaca keys are missing.
    """
    client = _get_trading_client()
    levels = _strategy.get_levels(signal["price"], signal["atr"])

    if client is None:
        # Simulated paper order
        logger.info(f"[SIMULATED] BUY {risk.MAX_CONTRACTS} {SYMBOL} @ {signal['price']:.2f}")
        return {
            "id": f"sim_{datetime.now().timestamp()}",
            "symbol": SYMBOL,
            "qty": risk.MAX_CONTRACTS,
            "side": "buy",
            "status": "filled",
            "filled_avg_price": signal["price"],
        }

    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=SYMBOL,
            qty=risk.MAX_CONTRACTS,
            side=OrderSide.BUY,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_data=req)
        logger.info(f"BUY order placed: {order.id} — {risk.MAX_CONTRACTS} {SYMBOL}")
        return {
            "id": str(order.id),
            "symbol": SYMBOL,
            "qty": risk.MAX_CONTRACTS,
            "side": "buy",
            "status": str(order.status),
            "filled_avg_price": float(order.filled_avg_price or signal["price"]),
        }
    except Exception as e:
        logger.error(f"Failed to place buy order: {e}")
        return None


def _place_sell_order(qty: int) -> bool:
    """Place a market sell order to close the position."""
    client = _get_trading_client()
    if client is None:
        logger.info(f"[SIMULATED] SELL {qty} {SYMBOL}")
        return True
    try:
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce

        req = MarketOrderRequest(
            symbol=SYMBOL,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.DAY,
        )
        order = client.submit_order(order_data=req)
        logger.info(f"SELL order placed: {order.id}")
        return True
    except Exception as e:
        logger.error(f"Failed to place sell order: {e}")
        return False


def _log_near_miss_safe(nm: dict, regime: str, blocked_reason: str, trades_today: int):
    """Write a near-miss record to DB. Never raises — silently logs errors."""
    try:
        db.log_near_miss({
            "timestamp":           datetime.now(timezone.utc),
            "symbol":              SYMBOL,  # TODO SPY→ES: update SYMBOL constant
            "close":               nm.get("close"),
            "breakout_level":      nm.get("breakout_level"),
            "percent_to_breakout": nm.get("percent_to_breakout"),
            "volume":              nm.get("volume"),
            "required_volume":     nm.get("required_volume"),
            "volume_ratio":        nm.get("volume_ratio"),
            "regime":              regime,
            "blocked_reason":      blocked_reason,
            "trades_today":        trades_today,
        })
    except Exception as e:
        logger.error(f"Near-miss logging failed (non-fatal): {e}")


def _close_position(reason: str, exit_price: float):
    """Close the current position and log the trade."""
    pos = _state["current_position"]
    if pos is None:
        return

    entry_price = pos["entry"]
    stop_price = pos["stop"]
    qty = pos["qty"]

    pnl_dollars = (exit_price - entry_price) * qty
    atr = pos.get("atr", 0)
    risk_amount = abs(entry_price - stop_price) * qty
    pnl_r = pnl_dollars / risk_amount if risk_amount > 0 else 0.0

    # Update state
    _state["daily_pnl"] += pnl_dollars
    _state["current_equity"] += pnl_dollars
    _state["peak_equity"] = max(_state["peak_equity"], _state["current_equity"])

    if pnl_dollars > 0:
        _state["consecutive_losses"] = 0
    else:
        _state["consecutive_losses"] += 1

    trade_data = {
        "entry_time": pos.get("entry_time"),
        "exit_time": datetime.now(timezone.utc),
        "symbol": SYMBOL,
        "direction": "LONG",
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": pos.get("target"),
        "shares": qty,
        "pnl_dollars": pnl_dollars,
        "pnl_r": pnl_r,
        "atr_at_entry": atr,
        "volume_ratio": pos.get("volume_ratio", 0),
        "exit_reason": reason,
        "regime": _state["regime"],
        "consecutive_losses": _state["consecutive_losses"],
    }
    db.log_trade(trade_data)
    _state["daily_trades"].append(trade_data)
    _state["current_position"] = None
    _state["trade_count"] += 1

    logger.info(
        f"Trade closed: {reason} | P&L=${pnl_dollars:+.2f} | R={pnl_r:.2f} | "
        f"Consecutive losses: {_state['consecutive_losses']}"
    )

    # Place sell order (fire-and-forget — position may already be closed by stop)
    _place_sell_order(qty)

    # Check kill switch after trade
    ks = risk.check_kill_switch(
        _state["consecutive_losses"],
        _state["daily_pnl"],
        _state["peak_equity"],
        _state["current_equity"],
    )
    if ks["pause"] and not _state["kill_switch_active"]:
        _state["kill_switch_active"] = True
        _state["is_paused"] = True
        logger.critical(f"KILL SWITCH ACTIVATED: {ks['reason']}")
        analyze_anomaly("kill_switch", {
            "reason": ks["reason"],
            "daily_pnl": _state["daily_pnl"],
            "consecutive_losses": _state["consecutive_losses"],
            "current_equity": _state["current_equity"],
        })


# ── APScheduler jobs ──────────────────────────────────────────────────────────
def five_min_bar_job():
    """Main trading logic — runs every 5 minutes during session hours."""
    if _state["is_paused"]:
        return
    if not _is_session_hours():
        return

    now_et = _now_et()

    # ── Fetch data ────────────────────────────────────────────────────────────
    df = _fetch_bars()
    if df is None or len(df) < 30:
        logger.warning("Insufficient data — skipping bar")
        _log_near_miss_safe(
            {"close": None, "breakout_level": None, "percent_to_breakout": None,
             "volume": None, "required_volume": None, "volume_ratio": None},
            _state["regime"], "data_unavailable", _state["trade_count"],
        )
        return

    # ── Compute indicators ────────────────────────────────────────────────────
    try:
        df = _strategy.compute_indicators(df)
    except Exception as e:
        logger.error(f"compute_indicators error: {e}")
        _log_near_miss_safe(
            {"close": None, "breakout_level": None, "percent_to_breakout": None,
             "volume": None, "required_volume": None, "volume_ratio": None},
            _state["regime"], "indicator_error", _state["trade_count"],
        )
        return

    # ── Evaluate near-miss proximity (non-blocking, used for visibility) ─────
    try:
        _nm = _strategy.evaluate_signal_state(df)
    except Exception as e:
        logger.error(f"evaluate_signal_state error: {e}")
        _nm = None

    # ── Monitor open position ─────────────────────────────────────────────────
    if _state["current_position"] is not None:
        pos = _state["current_position"]
        current_price = float(df["Close"].iloc[-1])

        # Check stop
        if current_price <= pos["stop"]:
            _close_position("stop_hit", current_price)
            return

        # Check target
        if current_price >= pos["target"]:
            _close_position("target_hit", current_price)
            return

        # Check max hold (4 hours)
        entry_time = pos.get("entry_time")
        if entry_time:
            elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            if elapsed >= _strategy.MAX_HOLD_MINUTES:
                _close_position("max_hold_exceeded", current_price)
                return

        return  # Still holding, nothing to do

    # ── Regime check ──────────────────────────────────────────────────────────
    try:
        regime = _classifier.classify(df)
        _state["regime"] = regime
        if regime in ("Range-Bound", "Extreme Volatility"):
            logger.debug(f"Regime {regime} — no entry")
            db.log_risk_check("regime", "BLOCKED", f"Regime is {regime}")
            if _nm:  # log every bar so Last Signal Check always has data
                _log_near_miss_safe(_nm, regime, "regime_blocked", _state["trade_count"])
            return
    except Exception as e:
        logger.error(f"Regime classification error: {e}")
        regime = "Weak Trend"
        _state["regime"] = regime

    # ── Signal generation ─────────────────────────────────────────────────────
    signal = _strategy.generate_signals(df, now_et, _state["trade_count"])
    if signal is None:
        if _nm:  # log every bar so Last Signal Check always has data
            t = now_et.time()
            if not (time(9, 30) <= t < time(15, 30)):
                _reason = "outside_time_window"
            elif _state["trade_count"] >= risk.MAX_TRADES_PER_DAY:
                _reason = "max_trades_reached"
            elif _nm.get("price_near_miss") and not _nm.get("volume_near_miss"):
                _reason = "volume_not_met"
            else:
                _reason = "breakout_not_met"
            _log_near_miss_safe(_nm, _state["regime"], _reason, _state["trade_count"])
        return

    # ── Risk check ────────────────────────────────────────────────────────────
    risk_result = risk.pre_trade_check(
        daily_pnl=_state["daily_pnl"],
        trade_count=_state["trade_count"],
        time_et=now_et,
        consecutive_losses=_state["consecutive_losses"],
    )
    result_str = "APPROVED" if risk_result["approved"] else "BLOCKED"
    db.log_risk_check("pre_trade", result_str, risk_result["reason"])

    if not risk_result["approved"]:
        logger.debug(f"Risk check blocked: {risk_result['reason']}")
        if _nm:  # log every bar
            _log_near_miss_safe(_nm, _state["regime"], "risk_blocked", _state["trade_count"])
        return

    # ── Place order ───────────────────────────────────────────────────────────
    order = _place_buy_order(signal)
    if order is None:
        return

    levels = _strategy.get_levels(signal["price"], signal["atr"])
    _state["current_position"] = {
        "entry_time": datetime.now(timezone.utc),
        "entry": levels["entry"],
        "stop": levels["stop"],
        "target": levels["target"],
        "qty": risk.MAX_CONTRACTS,
        "atr": signal["atr"],
        "volume_ratio": signal.get("volume_ratio", 0),
        "order_id": order.get("id"),
    }
    logger.info(
        f"Position opened: {SYMBOL} @ {levels['entry']:.2f} | "
        f"Stop={levels['stop']:.2f} | Target={levels['target']:.2f} | "
        f"Regime={regime}"
    )


def update_status_job():
    """Update system_status table every 60 seconds."""
    try:
        db.log_status({
            "status": "PAUSED" if _state["is_paused"] else "RUNNING",
            "regime": _state["regime"],
            "trade_count_today": _state["trade_count"],
            "daily_pnl": _state["daily_pnl"],
            "consecutive_losses": _state["consecutive_losses"],
            "kill_switch_active": _state["kill_switch_active"],
            "session_day": _state["session_day"],
            "message": f"Equity=${_state['current_equity']:.0f} | Position={'OPEN' if _state['current_position'] else 'NONE'}",
        })
    except Exception as e:
        logger.error(f"update_status_job error: {e}")


def market_open_job():
    """Reset daily state at 9:25 AM ET."""
    logger.info("Market open — resetting daily state")
    _state["daily_pnl"] = 0.0
    _state["trade_count"] = 0
    _state["consecutive_losses"] = 0
    _state["is_paused"] = False
    _state["kill_switch_active"] = False
    _state["daily_trades"] = []
    # Note: peak_equity and current_equity carry over (drawdown is cumulative)

    db.log_status({
        "status": "RUNNING",
        "regime": _state["regime"],
        "trade_count_today": 0,
        "daily_pnl": 0.0,
        "consecutive_losses": 0,
        "kill_switch_active": False,
        "session_day": _state["session_day"],
        "message": "Session started",
    })


def end_of_day_job():
    """3:55 PM ET: force-close any open position, compute summary, send email."""
    logger.info("End of day — running EOD procedure")

    # Always query Alpaca directly — _state["current_position"] is not the source
    # of truth here. A crash or missed signal could leave an orphan broker position.
    client = _get_trading_client()
    if client is not None:
        try:
            positions = client.get_all_positions()
            if positions:
                logger.info(
                    f"EOD close: found {len(positions)} open positions, closing all"
                )
                client.close_all_positions(cancel_orders=True)
            else:
                logger.info("EOD close: no open positions at Alpaca")
        except Exception as e:
            logger.error(f"EOD Alpaca close failed: {e}")
    else:
        # Simulated mode — Alpaca unavailable, fall back to in-memory state
        if _state["current_position"] is not None:
            df = _fetch_bars()
            if df is not None and not df.empty:
                exit_price = float(df["Close"].iloc[-1])
            else:
                exit_price = _state["current_position"]["entry"]
            _close_position("end_of_day", exit_price)

    _state["current_position"] = None

    # Increment session day counter
    _state["session_day"] += 1

    # Compute and log daily summary
    today_trades = _state["daily_trades"]
    wins = [t for t in today_trades if (t.get("pnl_dollars") or 0) > 0]
    pnls = [t.get("pnl_dollars", 0) for t in today_trades]
    rs = [t.get("pnl_r", 0) for t in today_trades]
    equity_curve = []
    running = 0.0
    for p in pnls:
        running += p
        equity_curve.append(running)

    max_dd = 0.0
    if equity_curve:
        peak = equity_curve[0]
        for v in equity_curve:
            peak = max(peak, v)
            max_dd = min(max_dd, v - peak)

    regime_dist = {}
    for t in today_trades:
        r = t.get("regime", "Unknown")
        regime_dist[r] = regime_dist.get(r, 0) + 1

    db.log_daily_summary({
        "trade_date": datetime.now(ET).date(),
        "total_trades": len(today_trades),
        "winning_trades": len(wins),
        "gross_pnl": sum(pnls),
        "max_drawdown": max_dd,
        "win_rate": len(wins) / len(today_trades) if today_trades else 0.0,
        "avg_r": sum(rs) / len(rs) if rs else 0.0,
        "rule_violations": 0,
        "regime_distribution": json.dumps(regime_dist),
    })

    # Send daily email
    try:
        send_daily_report(trade_day_n=_state["session_day"])
    except Exception as e:
        logger.error(f"Email report failed: {e}")

    logger.info(
        f"EOD complete — Day {_state['session_day']} | "
        f"Trades={len(today_trades)} | P&L=${sum(pnls):+.2f}"
    )


def morning_brief_job():
    """9:25 AM ET Mon-Fri: fetch indicators and send the morning briefing email."""
    try:
        df = _fetch_bars()
        if df is None or len(df) < 30:
            logger.warning("Morning brief: insufficient bar data")
            return
        df_ind = _strategy.compute_indicators(df)
        regime = _classifier.classify(df_ind)
        _state["regime"] = regime

        valid = df_ind.dropna(subset=["high_20", "atr14"])
        if valid.empty:
            logger.warning("Morning brief: no valid indicator rows")
            return
        last          = valid.iloc[-1]
        spy_price     = float(last["Close"])
        breakout_level = float(last["high_20"])
        atr           = float(last["atr14"])

        send_morning_brief(
            session_day=_state["session_day"],
            regime=regime,
            spy_price=spy_price,
            breakout_level=breakout_level,
            atr=atr,
        )
    except Exception as e:
        logger.error(f"Morning brief job failed: {e}")


def noon_update_job():
    """12:00 PM ET Mon-Fri: send midday status update email."""
    try:
        df = _fetch_bars()
        spy_price = 0.0
        if df is not None and not df.empty:
            spy_price = float(df["Close"].iloc[-1])

        near_misses_am = db.get_today_near_misses()

        send_noon_update(
            session_day=_state["session_day"],
            daily_pnl=_state["daily_pnl"],
            trade_count=_state["trade_count"],
            regime=_state["regime"],
            spy_price=spy_price,
            in_position=_state["current_position"] is not None,
            near_misses_am=near_misses_am,
        )
    except Exception as e:
        logger.error(f"Noon update job failed: {e}")


# ── Startup & shutdown ────────────────────────────────────────────────────────
def _startup_catchup():
    """On boot: reconcile broker positions, then fire missed emails within a 45-min grace window."""

    # ── Position reconciliation (runs unconditionally, before scheduler starts) ─
    client = _get_trading_client()
    if client is not None:
        try:
            positions = client.get_all_positions()
            spy_pos = next((p for p in positions if p.symbol == SYMBOL), None)
            if spy_pos is not None:
                qty = int(spy_pos.qty)
                avg_price = float(spy_pos.avg_entry_price)
                logger.warning(
                    f"Orphan position detected on startup: {qty} shares {SYMBOL} "
                    f"@ ${avg_price:.2f}, closing immediately"
                )
                client.close_all_positions(cancel_orders=True)
                _state["current_position"] = None
            else:
                logger.info("Startup reconciliation: no open positions at Alpaca")
        except Exception as e:
            logger.error(f"Startup position reconciliation failed: {e}")

    # ── Missed-email catch-up (weekdays only, within 45-min grace windows) ──────
    now = _now_et()
    if now.weekday() >= 5:   # weekend — no emails
        return
    t = now.time()
    # Morning brief: fire only if within 45 min of 9:25 AM
    if time(9, 25) <= t < time(10, 10):
        logger.info("Startup catch-up: sending morning brief")
        try:
            morning_brief_job()
        except Exception as e:
            logger.error(f"Startup morning brief failed: {e}")
    # EOD: fire only if within 45 min of 3:55 PM
    elif time(15, 55) <= t < time(16, 40):
        logger.info("Startup catch-up: sending EOD report")
        try:
            end_of_day_job()
        except Exception as e:
            logger.error(f"Startup EOD failed: {e}")


def _handle_sigterm(signum, frame):
    """Graceful shutdown on SIGTERM."""
    logger.info("SIGTERM received — shutting down gracefully")
    _state["is_paused"] = True
    if _scheduler:
        _scheduler.shutdown(wait=False)
    sys.exit(0)


def main():
    global _scheduler

    # SIGTERM handler can only be registered from the main thread.
    # When running as a background daemon thread (start_background), skip it.
    import threading as _threading
    if _threading.current_thread() is _threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_sigterm)

    # Initialize DB
    db.init_db()

    # Load or train regime model if not present
    from pathlib import Path
    model_path = Path(__file__).parent.parent / "models" / "regime_rf.pkl"
    if not model_path.exists():
        logger.info("No regime model found — worker will use default 'Weak Trend'")
        logger.info("Run `python backtest/run.py` to train and save the model")

    logger.info("Sable Stocks worker starting...")
    _startup_catchup()

    _scheduler = BlockingScheduler(timezone="America/New_York")

    # Every 5 minutes during session hours (Mon-Fri)
    _scheduler.add_job(
        five_min_bar_job,
        CronTrigger(day_of_week="mon-fri", hour="9-16", minute="*/5"),
        id="five_min_bar",
        name="5-Minute Bar Job",
        misfire_grace_time=60,
    )

    # Status update every 60 seconds
    _scheduler.add_job(
        update_status_job,
        IntervalTrigger(seconds=60),
        id="update_status",
        name="Status Update",
    )

    # Market open reset at 9:25 AM ET Mon-Fri
    _scheduler.add_job(
        market_open_job,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=25),
        id="market_open",
        name="Market Open Reset",
    )

    # Morning brief email at 9:25 AM ET Mon-Fri
    _scheduler.add_job(
        morning_brief_job,
        CronTrigger(day_of_week="mon-fri", hour=9, minute=25),
        id="morning_brief",
        name="Morning Brief Email",
        misfire_grace_time=7200,  # fire if within 2 hours of scheduled time
    )

    # Noon update email at 12:00 PM ET Mon-Fri
    _scheduler.add_job(
        noon_update_job,
        CronTrigger(day_of_week="mon-fri", hour=12, minute=0),
        id="noon_update",
        name="Noon Update Email",
        misfire_grace_time=7200,
    )

    # End of day at 3:55 PM ET Mon-Fri — must fire before 4:00 PM close so that
    # the DAY-order sell executes within the regular session.
    _scheduler.add_job(
        end_of_day_job,
        CronTrigger(day_of_week="mon-fri", hour=15, minute=55),
        id="end_of_day",
        name="End of Day",
        misfire_grace_time=14400,  # fire if within 4 hours of scheduled time
    )

    logger.info(
        "Scheduler started — jobs: 5min_bar, status_update, market_open, end_of_day"
    )

    try:
        _scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Worker stopped")


def start_background():
    """
    Start the trading scheduler in a daemon background thread.
    Called by api/main.py on startup so the worker co-runs with the API.
    The thread is a daemon so it exits automatically when the main process exits.
    """
    import threading

    db.init_db()

    thread = threading.Thread(target=main, daemon=True, name="apex-trader")
    thread.start()
    logger.info(f"Trading worker started in background thread ({thread.name})")
    return thread


if __name__ == "__main__":
    main()
