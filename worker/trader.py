"""
worker/trader.py — Main trading worker for Sable Stocks.

Runs Monday-Friday 9:25 AM - 4:05 PM ET using APScheduler.
Fetches SPY bars from Alpaca (yfinance fallback), runs VWAPTrendPullback strategy,
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
from worker.strategy import VWAPTrendPullback, MultiSessionStrategy
from worker.email_report import send_daily_report, send_morning_brief, send_noon_update
from diagnostics.analyzer import analyze_anomaly
from models.regime_classifier import RegimeClassifier

# ── Configuration ─────────────────────────────────────────────────────────────
ET = pytz.timezone("America/New_York")
SYMBOL = "SPY"              # Alpaca paper account symbol (proxy tracking only)
ES_DATA_SYMBOL = "ES=F"    # yfinance symbol for ES monitoring
ES_POINT_VALUE = 50.0      # USD per point for E-mini S&P 500
MES_POINT_VALUE = 5.0      # USD per point for Micro E-mini S&P 500
MES_MAX_CONTRACTS = 5      # max MES contracts per trade
ES_CONTRACTS = 1           # ES contracts cap (MES sizing handled per-signal)
TRADERSPOST_TICKER = "MESM2026"  # MES June 2026 front month
PAPER = True
SESSION_START = time(9, 25)
SESSION_END = time(16, 5)

# ── Safety configuration (override via environment variables) ──────────────────
# MARKET_DATA_MODE: "PROXY" = SPY×10 surrogate | "FUTURES_DIRECT" = live MES/ES data
MARKET_DATA_MODE: str = os.getenv("MARKET_DATA_MODE", "PROXY")
# ALLOW_PROXY_TRADING: must be explicitly "true" to send real/eval orders in PROXY mode
ALLOW_PROXY_TRADING: bool = os.getenv("ALLOW_PROXY_TRADING", "false").lower() == "true"
# DRY_RUN: defaults true when PROXY+!allowProxyTrading; no orders sent when true
_dry_run_default = "true" if (MARKET_DATA_MODE == "PROXY" and not ALLOW_PROXY_TRADING) else "false"
DRY_RUN: bool = os.getenv("DRY_RUN", _dry_run_default).lower() == "true"
# ALLOW_SHORTS: shorts are disabled until TradersPost short-side execution is confirmed
ALLOW_SHORTS: bool = os.getenv("ALLOW_SHORTS", "false").lower() == "true"
# LONG_ONLY: additional guard — skip all short entries regardless of ALLOW_SHORTS
LONG_ONLY: bool = os.getenv("LONG_ONLY", "false").lower() == "true"

# ── Global mutable state ──────────────────────────────────────────────────────
_state = {
    "daily_pnl": 0.0,
    "trade_count": 0,
    "consecutive_losses": 0,
    "consecutive_wins": 0,
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
_strategy = MultiSessionStrategy()
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


def _fetch_bars_1min_alpaca() -> "pd.DataFrame | None":
    """Fetch 1-min SPY bars from Alpaca — primary signal source (proven gate setting)."""
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
        # 2 hours of 1-min bars: plenty for 20-bar lookback + ATR14 warmup
        start = datetime.now(timezone.utc) - timedelta(hours=2)
        req = StockBarsRequest(
            symbol_or_symbols=SYMBOL,
            timeframe=TimeFrame(1, TimeFrameUnit.Minute),
            start=start,
        )
        bars = client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return None
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(SYMBOL, level="symbol")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning(f"Alpaca 1-min data fetch failed: {e}")
        return None


def _fetch_bars_alpaca() -> "pd.DataFrame | None":
    """Fetch 5-min SPY bars from Alpaca — fallback if 1-min unavailable."""
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
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(SYMBOL, level="symbol")
        df = df.rename(columns={"open": "Open", "high": "High", "low": "Low",
                                 "close": "Close", "volume": "Volume"})
        df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]]
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning(f"Alpaca 5-min data fetch failed: {e}")
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


def _fetch_es_bars_yfinance() -> "pd.DataFrame | None":
    """Fetch 5-min ES=F bars from yfinance, filtered to regular trading hours."""
    import pandas as pd
    try:
        df = yf.download(ES_DATA_SYMBOL, period="5d", interval="5m",
                         auto_adjust=True, progress=False)
        if df.empty:
            return None
        if hasattr(df.columns, "levels"):
            for _lvl in range(df.columns.nlevels):
                _candidate = df.columns.get_level_values(_lvl)
                if "Close" in _candidate:
                    df.columns = _candidate
                    break
        df = df.dropna(subset=["Close", "Volume"])
        df = df[df["Volume"] > 0]
        # Restrict to regular trading hours so overnight bars don't skew rolling highs
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.index = df.index.tz_convert("America/New_York")
        df = df.between_time("09:30", "16:00")
        return df if len(df) >= 30 else None
    except Exception as e:
        logger.warning(f"ES=F yfinance fetch failed: {e}")
        return None


def _fetch_bars():
    """Fetch SPY bars for signal generation.
    Primary: 1-min bars from Alpaca (proven gate setting — 20-bar = 20-min window).
    Fallback: 5-min bars from Alpaca, then yfinance.
    """
    df = _fetch_bars_1min_alpaca()
    if df is not None:
        return df
    logger.warning("Alpaca 1-min unavailable — falling back to 5-min bars")
    df = _fetch_bars_alpaca()
    if df is not None:
        return df
    logger.warning("Alpaca 5-min unavailable — trying yfinance fallback")
    df = _fetch_bars_yfinance()
    if df is None:
        logger.error("All data sources failed")
    return df


def _is_order_allowed() -> tuple[bool, str]:
    """
    Check whether real/eval orders may be sent.

    Returns (True, "OK") or (False, reason_string).
    Called before every TradersPost signal dispatch.
    """
    if DRY_RUN:
        return False, "DRY_RUN mode active — no orders sent"
    if MARKET_DATA_MODE == "PROXY" and not ALLOW_PROXY_TRADING:
        return False, (
            "Proxy market data mode blocks real orders "
            "(set ALLOW_PROXY_TRADING=true to override)"
        )
    return True, "OK"


async def send_traderspost_signal(
    action: str,
    contracts: int = 1,
    order_type: str = "market",
    limit_price: float = 0.0,
    intent: str = "",
):
    """
    Send trading signal to TradersPost/Tradovate.

    intent values: open_long | close_long | open_short | close_short | close_all
    Included as a payload field so the webhook log clearly identifies the trade side.
    TradersPost ignores unknown fields so this is safe.
    """
    import httpx

    # Safety gate — block proxy-mode real orders
    allowed, gate_reason = _is_order_allowed()
    if not allowed:
        logger.info(
            f"TradersPost signal SKIPPED [{intent or action}]: {gate_reason}"
        )
        return

    webhook_url = os.getenv("TRADERSPOST_WEBHOOK_URL")
    if not webhook_url:
        logger.warning("TRADERSPOST_WEBHOOK_URL not set, skipping")
        return

    payload: dict = {
        "ticker": TRADERSPOST_TICKER,
        "action": action,
        "contracts": contracts,
    }
    if order_type == "limit" and limit_price > 0:
        payload["orderType"] = "limit"
        payload["limitPrice"] = limit_price
    if intent:
        payload["intent"] = intent   # informational; ignored by TP if unsupported

    try:
        async with httpx.AsyncClient() as client:
            response = await client.post(webhook_url, json=payload, timeout=5.0)
            logger.info(
                f"TradersPost signal sent [{intent or action}] | "
                f"action={action} contracts={contracts} | "
                f"Status: {response.status_code} | Response: {response.text}"
            )
    except Exception as e:
        logger.error(f"TradersPost signal failed: {e}")


def _get_es_bid_ask() -> tuple:
    """Return (bid, ask) for ES front month.
    1. yfinance ES=F  (direct; often blocked on Railway)
    2. Alpaca latest SPY bar × 10  (fast single-bar call)
    3. Last bar from _fetch_bars() × 10  (full bars fallback)
    """
    # 1. yfinance ES=F
    try:
        df = yf.download("ES=F", period="1d", interval="1m", progress=False, auto_adjust=True)
        if not df.empty:
            price = float(df["Close"].iloc[-1])
            return price - 0.25, price + 0.25
    except Exception:
        pass

    # 2. Alpaca latest bar for SPY — single fast API call
    try:
        api_key = os.getenv("ALPACA_API_KEY", "")
        secret_key = os.getenv("ALPACA_SECRET_KEY", os.getenv("ALPACA_API_SECRET", ""))
        if api_key and secret_key:
            from alpaca.data.historical import StockHistoricalDataClient
            from alpaca.data.requests import StockLatestBarRequest
            _data_client = StockHistoricalDataClient(api_key, secret_key)
            req = StockLatestBarRequest(symbol_or_symbols=SYMBOL)
            bars = _data_client.get_stock_latest_bar(req)
            spy_price = float(bars[SYMBOL].close)
            es = round(spy_price * 10, 2)
            return es - 0.25, es + 0.25
    except Exception as e:
        logger.debug(f"Alpaca latest bar fallback failed: {e}")

    # 3. Full bars fetch × 10
    df = _fetch_bars()
    if df is not None and not df.empty:
        es = round(float(df["Close"].iloc[-1]) * 10, 2)
        return es - 0.25, es + 0.25

    return 0.0, 0.0


def _fire_traderspost(
    action: str,
    contracts: int = 1,
    order_type: str = "market",
    limit_price: float = 0.0,
    intent: str = "",
):
    """Non-blocking sync wrapper — spawns a daemon thread to run the async signal."""
    import asyncio
    import threading

    threading.Thread(
        target=lambda: asyncio.run(
            send_traderspost_signal(action, contracts, order_type, limit_price, intent)
        ),
        daemon=True,
    ).start()


def _fire_traderspost_exit_with_fallback(limit_price: float, intent: str = "close_long"):
    """Limit sell immediately, then market exit after 30 s if limit didn't fill."""
    import threading, time

    _fire_traderspost("sell", 1, "limit", limit_price, intent)

    def _fallback():
        time.sleep(30)
        _fire_traderspost("exit", 0, "market", 0.0, intent)
        logger.info("TradersPost 30s fallback: market exit sent")

    threading.Thread(target=_fallback, daemon=True).start()


def _place_buy_order(signal: dict) -> dict | None:
    """
    Place a paper buy order via Alpaca.
    Returns order dict on success, None on failure.
    Simulates the order if Alpaca keys are missing.
    """
    client = _get_trading_client()
    levels = _strategy.get_levels(signal["price"], signal.get("atr", 0.0))

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
        close = nm.get("close")
        vwap = nm.get("vwap")
        pct = ((close - vwap) / vwap * 100) if (close and vwap) else None
        db.log_near_miss({
            "timestamp":           datetime.now(timezone.utc),
            "symbol":              TRADERSPOST_TICKER,
            "close":               close,
            "breakout_level":      nm.get("or_high") or vwap,  # OR high as breakout reference
            "percent_to_breakout": pct,
            "volume":              nm.get("ema9"),              # repurpose field for EMA9
            "required_volume":     nm.get("ema21"),             # repurpose field for EMA21
            "volume_ratio":        nm.get("vwap_crossings", 0),
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

    direction = pos.get("direction", "LONG")
    entry_price = pos["entry"]   # SPY price used for stop/target monitoring
    stop_price = pos["stop"]
    qty = pos["qty"]             # MES contracts

    # P&L in MES dollars ($5/point, SPY×10 = ES proxy).
    es_entry = pos.get("es_entry", 0.0)
    es_bid, es_ask = _get_es_bid_ask()

    if es_entry > 0 and (es_bid > 0 or es_ask > 0):
        # LONG exit: sell at bid; SHORT exit: buy at ask
        if direction == "LONG":
            es_exit = es_bid if es_bid > 0 else es_ask
            pnl_dollars = (es_exit - es_entry) * qty * MES_POINT_VALUE
        else:
            es_exit = es_ask if es_ask > 0 else es_bid
            pnl_dollars = (es_entry - es_exit) * qty * MES_POINT_VALUE
    else:
        logger.warning("ES price unavailable at close — P&L approximated from SPY data")
        spy_move = exit_price - entry_price
        if direction == "SHORT":
            spy_move = -spy_move
        pnl_dollars = spy_move * 10 * qty * MES_POINT_VALUE

    stop_dist = pos.get("stop_distance", abs(entry_price - stop_price))
    risk_amount = max(stop_dist * 10 * qty * MES_POINT_VALUE, 1.0)
    atr = pos.get("atr", 0)
    pnl_r = pnl_dollars / risk_amount if risk_amount > 0 else 0.0

    # Update state
    _state["daily_pnl"] += pnl_dollars
    _state["current_equity"] += pnl_dollars
    _state["peak_equity"] = max(_state["peak_equity"], _state["current_equity"])

    if pnl_dollars > 0:
        _state["consecutive_losses"] = 0
        _state["consecutive_wins"] += 1
    else:
        _state["consecutive_losses"] += 1
        _state["consecutive_wins"] = 0

    trade_data = {
        "entry_time": pos.get("entry_time"),
        "exit_time": datetime.now(timezone.utc),
        "symbol": TRADERSPOST_TICKER,
        "direction": direction,
        "entry_price": entry_price,
        "exit_price": exit_price,
        "stop_price": stop_price,
        "target_price": pos.get("target"),
        "shares": qty,
        "pnl_dollars": pnl_dollars,
        "pnl_r": pnl_r,
        "atr_at_entry": atr,
        "volume_ratio": pos.get("stop_distance", 0),
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

    # Close Alpaca paper position (LONG only — SHORT not tracked in Alpaca SPY account)
    # Always sell exactly MAX_CONTRACTS (1 SPY share), not MES contract count
    if direction == "LONG":
        _place_sell_order(risk.MAX_CONTRACTS)

    # Fire TradersPost exit with explicit intent labels
    _es_bid2, _es_ask2 = _get_es_bid_ask()
    if reason == "end_of_day":
        _fire_traderspost("exit", 0, intent="close_all")
    elif direction == "LONG":
        _es_limit_sell = round(_es_ask2 - 0.25, 2) if _es_ask2 > 0 else 0.0
        if _es_limit_sell > 0:
            _fire_traderspost_exit_with_fallback(_es_limit_sell, intent="close_long")
        else:
            _fire_traderspost("sell", qty, intent="close_long")
    else:
        # SHORT exit: buy back at limit (bid + 0.25)
        _es_limit_buy = round(_es_bid2 + 0.25, 2) if _es_bid2 > 0 else 0.0
        _fire_traderspost("buy", qty, "limit" if _es_limit_buy > 0 else "market",
                          _es_limit_buy, intent="close_short")

    # Discord trade exit notification
    try:
        import discord_bot as _db
        entry_time = pos.get("entry_time")
        if entry_time:
            elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds()
            mins, secs = divmod(int(elapsed), 60)
            duration = f"{mins}m {secs}s"
        else:
            duration = "unknown"
        reason_labels = {
            "stop_hit": "Stop hit", "target_hit": "Target hit",
            "max_hold_exceeded": "Time exit", "end_of_day": "EOD close",
        }
        running_pnl = _state["current_equity"] - 100_000.0
        _db.post_trade_exit(
            pnl=pnl_dollars,
            entry=entry_price,
            exit_price=exit_price,
            duration=duration,
            running_pnl=running_pnl,
            reason=reason_labels.get(reason, reason),
            streak=_state["consecutive_wins"],
        )
    except Exception as _e:
        logger.warning(f"Discord exit notification failed: {_e}")

    # Milestone check
    try:
        import discord_bot as _db
        _db.check_milestones(_state["current_equity"] - 100_000.0)
    except Exception as _e:
        logger.warning(f"Milestone check failed: {_e}")

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
        _fire_traderspost("exit", 0, intent="close_all")
        try:
            import discord_bot as _db
            daily_loss = _state["daily_pnl"]
            buffer = max(0.0, 1500 + daily_loss)
            _db.post_kill_switch(daily_loss=daily_loss, buffer=buffer)
        except Exception as _e:
            logger.warning(f"Discord kill switch notification failed: {_e}")


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
            {"close": None, "vwap": None, "ema9": None, "ema21": None,
             "or_high": None, "or_low": None, "vwap_crossings": 0},
            _state["regime"], "data_unavailable", _state["trade_count"],
        )
        return

    # ── Compute indicators ────────────────────────────────────────────────────
    try:
        df = _strategy.compute_indicators(df)
    except Exception as e:
        logger.error(f"compute_indicators error: {e}")
        _log_near_miss_safe(
            {"close": None, "vwap": None, "ema9": None, "ema21": None,
             "or_high": None, "or_low": None, "vwap_crossings": 0},
            _state["regime"], "indicator_error", _state["trade_count"],
        )
        return

    # ── Evaluate near-miss proximity (non-blocking, used for visibility) ─────
    try:
        _nm = _strategy.evaluate_signal_state(df)
    except Exception as e:
        logger.error(f"evaluate_signal_state error: {e}")
        _nm = None

    # ── Monitor open position (directional) ──────────────────────────────────
    if _state["current_position"] is not None:
        pos = _state["current_position"]
        current_price = float(df["Close"].iloc[-1])
        direction = pos.get("direction", "LONG")

        if direction == "LONG":
            if current_price <= pos["stop"]:
                _close_position("stop_hit", current_price)
                return
            if current_price >= pos["target"]:
                _close_position("target_hit", current_price)
                return
        else:  # SHORT
            if current_price >= pos["stop"]:
                _close_position("stop_hit", current_price)
                return
            if current_price <= pos["target"]:
                _close_position("target_hit", current_price)
                return

        # Move stop to breakeven (+0.02 buffer) once price moves 1R in our favour
        entry     = pos["entry"]
        stop_dist = abs(entry - pos["stop"])
        if stop_dist > 0:
            if direction == "LONG" and current_price >= entry + stop_dist and pos["stop"] < entry:
                pos["stop"] = round(entry + 0.02, 2)
                logger.info(f"TRAIL: breakeven stop → {pos['stop']:.2f} (entry={entry:.2f})")
            elif direction == "SHORT" and current_price <= entry - stop_dist and pos["stop"] > entry:
                pos["stop"] = round(entry - 0.02, 2)
                logger.info(f"TRAIL: breakeven stop → {pos['stop']:.2f} (entry={entry:.2f})")

        entry_time = pos.get("entry_time")
        if entry_time:
            elapsed = (datetime.now(timezone.utc) - entry_time).total_seconds() / 60
            if elapsed >= _strategy.MAX_HOLD_MINUTES:
                _close_position("max_hold_exceeded", current_price)
                return

        return  # Still holding

    # ── Regime check (logging only — does NOT block entries) ──────────────────
    # Classifier was trained on synthetic data and cannot be trusted to filter trades.
    # It runs in the background so logs show what it would have called.
    try:
        regime = _classifier.classify(df)
        _state["regime"] = regime
        would_block = regime in ("Range-Bound", "Extreme Volatility")
        db.log_risk_check("regime", "INFO", f"Regime={regime} | would_have_blocked={would_block}")
    except Exception as e:
        logger.error(f"Regime classification error: {e}")
        regime = "Weak Trend"
        _state["regime"] = regime

    # ── Signal generation ─────────────────────────────────────────────────────
    eval_pnl = _state["current_equity"] - 100_000.0
    signal = _strategy.generate_signals(
        df, now_et, _state["trade_count"],
        daily_pnl=_state["daily_pnl"],
        current_equity=_state["current_equity"],
        peak_equity=_state["peak_equity"],
    )
    if signal is None:
        # Per-candle decision log
        if _nm:
            t_now = now_et.time()
            in_window = time(9, 45) <= t_now < time(11, 30)
            or_break = _nm.get("or_long_break") or _nm.get("or_short_break")
            vwap_aligned = _nm.get("above_vwap") or not _nm.get("above_vwap", True)
            ema_aligned = _nm.get("ema_bullish") is not None
            chop = _nm.get("chop_blocked", False)
            at_limit = _state["trade_count"] >= risk.MAX_TRADES_PER_DAY

            if not in_window:
                _reason = "outside_time_window"
            elif at_limit:
                _reason = "max_trades_reached"
            elif chop:
                _reason = "vwap_chop"
            elif not or_break:
                _reason = "no_or_breakout"
            elif not (_nm.get("long_pullback_valid") or _nm.get("short_pullback_valid")):
                _reason = "no_pullback_candle"
            else:
                _reason = "confirmation_not_met"

            if _nm.get("is_near_miss"):
                logger.info(
                    f"NEAR MISS: {now_et.strftime('%H:%M:%S')} | "
                    f"SPY={_nm.get('close', 0):.2f} VWAP={_nm.get('vwap', 0):.2f} | "
                    f"EMA9={_nm.get('ema9', 0):.2f} EMA21={_nm.get('ema21', 0):.2f} | "
                    f"OR_H={_nm.get('or_high') or 'none'} OR_L={_nm.get('or_low') or 'none'} | "
                    f"AboveVWAP={_nm.get('above_vwap')} EMABull={_nm.get('ema_bullish')} | "
                    f"LongPB={_nm.get('long_pullback_valid')} ShortPB={_nm.get('short_pullback_valid')} | "
                    f"Chop={_nm.get('vwap_crossings', 0)}x | Missing: {_reason}"
                )
            _log_near_miss_safe(_nm, regime, _reason, _state["trade_count"])
        return

    # ── Risk check ────────────────────────────────────────────────────────────
    risk_result = risk.pre_trade_check(
        daily_pnl=_state["daily_pnl"],
        trade_count=_state["trade_count"],
        time_et=now_et,
        consecutive_losses=_state["consecutive_losses"],
        eval_pnl=eval_pnl,
    )
    result_str = "APPROVED" if risk_result["approved"] else "BLOCKED"
    db.log_risk_check("pre_trade", result_str, risk_result["reason"])

    if not risk_result["approved"]:
        logger.debug(f"Risk check blocked: {risk_result['reason']}")
        if _nm:  # log every bar
            _log_near_miss_safe(_nm, _state["regime"], "risk_blocked", _state["trade_count"])
        return

    # ── Short direction guard ─────────────────────────────────────────────────
    direction = signal["direction"]
    if direction == "SHORT" and (LONG_ONLY or not ALLOW_SHORTS):
        if LONG_ONLY:
            logger.info(
                f"SHORT signal at {now_et.strftime('%H:%M:%S')} skipped: "
                f"long-only mode active (LONG_ONLY=true)"
            )
        else:
            logger.info(
                f"SHORT signal at {now_et.strftime('%H:%M:%S')} skipped: "
                f"shorts disabled by config (ALLOW_SHORTS=false)"
            )
        return

    # ── Signal fired log ──────────────────────────────────────────────────────
    logger.info(
        f"SIGNAL FIRED [{direction}] [{signal.get('strategy', 'VWAP')}]: "
        f"{now_et.strftime('%H:%M:%S')} | "
        f"SPY={signal['price']:.2f} | "
        f"OR={signal.get('or_high') or 0:.2f}/{signal.get('or_low') or 0:.2f} | "
        f"VWAP={signal.get('vwap', 0):.2f} EMA9={signal.get('ema9', 0):.2f} "
        f"EMA21={signal.get('ema21', 0):.2f} | "
        f"ATR={signal.get('atr', 0):.4f}(raw={signal.get('raw_atr', 0):.4f}) | "
        f"Stop={signal['stop']:.2f} Target={signal['target']:.2f} "
        f"Dist={signal['stop_distance']:.4f} | "
        f"Phase={signal.get('phase', '?')} MaxRisk=${signal.get('max_risk', 0):.0f} "
        f"Actual=${signal.get('risk_actual', 0):.0f} | "
        f"Contracts={signal['contracts']} MES | "
        f"VWAPx={signal.get('vwap_crossings', 'N/A')} | "
        f"Regime={regime}"
    )

    # ── Place entry order ─────────────────────────────────────────────────────
    _es_bid, _es_ask = _get_es_bid_ask()

    if direction == "LONG":
        order = _place_buy_order(signal)
        if order is None:
            return
        _es_entry_price = _es_bid if _es_bid > 0 else _es_ask
        _limit_price = round(_es_bid + 0.25, 2) if _es_bid > 0 else 0.0
        _fire_traderspost("buy", signal["contracts"],
                          "limit" if _limit_price > 0 else "market",
                          _limit_price, intent="open_long")
    else:
        # SHORT: skip Alpaca paper order (SPY short tracking unreliable); use TradersPost only
        order = {"id": f"short_{datetime.now(timezone.utc).timestamp()}"}
        _es_entry_price = _es_ask if _es_ask > 0 else _es_bid
        _limit_price = round(_es_ask - 0.25, 2) if _es_ask > 0 else 0.0
        _fire_traderspost("sell", signal["contracts"],
                          "limit" if _limit_price > 0 else "market",
                          _limit_price, intent="open_short")

    _state["current_position"] = {
        "entry_time": datetime.now(timezone.utc),
        "direction": direction,
        "entry": signal["price"],       # SPY price — used for stop/target monitoring
        "stop": signal["stop"],
        "target": signal["target"],
        "qty": signal["contracts"],     # MES contracts
        "atr": signal.get("atr", 0.0),
        "stop_distance": signal["stop_distance"],
        "order_id": order.get("id"),
        "es_entry": _es_entry_price,    # ES proxy price for P&L calculation
    }
    logger.info(
        f"Position opened [{direction}]: {SYMBOL} @ {signal['price']:.2f} | "
        f"Stop={signal['stop']:.2f} | Target={signal['target']:.2f} | "
        f"Contracts={signal['contracts']} MES | ES_entry={_es_entry_price:.2f}"
    )
    try:
        import discord_bot as _db
        running_pnl = _state["current_equity"] - 100_000.0
        _db.post_trade_entry(
            price=signal["price"],
            stop=signal["stop"],
            target=signal["target"],
            running_pnl=running_pnl,
        )
    except Exception as _e:
        logger.warning(f"Discord entry notification failed: {_e}")


_status_db_error_count: int = 0


def update_status_job():
    """Update system_status table every 60 seconds."""
    global _status_db_error_count
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
        _status_db_error_count = 0  # reset on success
    except Exception as e:
        _status_db_error_count += 1
        # Log the first 3 failures, then every 30th to avoid flooding during outages
        if _status_db_error_count <= 3 or _status_db_error_count % 30 == 0:
            logger.error(f"update_status_job error (#{_status_db_error_count}): {e}")


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
                _fire_traderspost("exit", 0, intent="close_all")
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

        valid = df_ind.dropna(subset=["vwap", "atr14"])
        if valid.empty:
            logger.warning("Morning brief: no valid indicator rows")
            return
        last          = valid.iloc[-1]
        spy_price     = float(last["Close"])
        breakout_level = float(last["vwap"])   # VWAP as the morning reference level
        atr           = float(last["atr14"])

        send_morning_brief(
            session_day=_state["session_day"],
            regime=regime,
            spy_price=spy_price,
            breakout_level=breakout_level,
            atr=atr,
        )
        try:
            import discord_bot as _db
            _db.post_morning_brief(
                session_day=_state["session_day"],
                regime=regime,
                spy_price=spy_price,
                breakout_level=breakout_level,
                atr=atr,
            )
        except Exception as _e:
            logger.warning(f"Discord morning brief failed: {_e}")
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
        try:
            import discord_bot as _db
            _db.post_noon_update(
                session_day=_state["session_day"],
                daily_pnl=_state["daily_pnl"],
                trade_count=_state["trade_count"],
                regime=_state["regime"],
                spy_price=spy_price,
                in_position=_state["current_position"] is not None,
            )
        except Exception as _e:
            logger.warning(f"Discord noon update failed: {_e}")
    except Exception as e:
        logger.error(f"Noon update job failed: {e}")


def regime_log_job():
    """Every 30 minutes: log current regime status and whether it would have blocked a trade."""
    if not _is_session_hours():
        return
    try:
        df = _fetch_bars()
        if df is None or len(df) < 30:
            return
        df = _strategy.compute_indicators(df)
        regime = _classifier.classify(df)
        _state["regime"] = regime
        would_block = regime in ("Range-Bound", "Extreme Volatility")

        valid = df.dropna(subset=["atr14", "vwap"])
        atr = float(valid["atr14"].iloc[-1]) if not valid.empty else 0.0

        # Simple ADX proxy: ratio of directional move to ATR over last 14 bars
        closes = df["Close"].tail(14)
        net_move = abs(float(closes.iloc[-1]) - float(closes.iloc[0]))
        adx_proxy = round(net_move / atr, 2) if atr > 0 else 0.0

        logger.info(
            f"REGIME CHECK: {_now_et().strftime('%H:%M')} | "
            f"Current regime: {regime} | "
            f"Would have blocked trade: {'YES' if would_block else 'NO'} | "
            f"ATR: {atr:.4f} | "
            f"ADX proxy: {adx_proxy}"
        )
    except Exception as e:
        logger.warning(f"regime_log_job error: {e}")


def discord_summary_job():
    """4:30 PM ET Mon-Fri: post daily summary to Discord with a Claude assessment."""
    try:
        import anthropic
        import discord_bot as _db

        today_trades = _state["daily_trades"]
        wins = [t for t in today_trades if (t.get("pnl_dollars") or 0) > 0]
        losses = [t for t in today_trades if (t.get("pnl_dollars") or 0) <= 0]
        daily_pnl = _state["daily_pnl"]
        total_pnl = _state["current_equity"] - 100_000.0
        buffer = max(0.0, 1500 + daily_pnl)

        assessment = "System ran as expected. Stay focused on the process."
        try:
            client = anthropic.Anthropic()
            msg = client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=120,
                system=_db.SABLE_SYSTEM,
                messages=[{
                    "role": "user",
                    "content": (
                        f"Write exactly 2 sentences assessing today's trading session.\n"
                        f"Trades: {len(today_trades)} | Wins: {len(wins)} | Losses: {len(losses)}\n"
                        f"Daily P&L: ${daily_pnl:+.2f} | Eval total: ${total_pnl:.2f} / $3,000\n"
                        f"Be honest and specific. End with one thing to focus on tomorrow."
                    ),
                }],
            )
            assessment = msg.content[0].text.strip()
        except Exception as e:
            logger.error(f"Claude daily assessment failed: {e}")

        _db.post_daily_summary(
            day_n=_state["session_day"],
            trades=len(today_trades),
            wins=len(wins),
            losses=len(losses),
            daily_pnl=daily_pnl,
            total_pnl=total_pnl,
            buffer=buffer,
            assessment=assessment,
        )
    except Exception as e:
        logger.error(f"discord_summary_job error: {e}")


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

    # ── Data pipeline smoke test ──────────────────────────────────────────────
    _data_ok = False
    _data_detail = "no data"
    try:
        _test_df = _fetch_bars_1min_alpaca()
        if _test_df is not None and len(_test_df) >= 30:
            _data_ok = True
            _data_detail = (
                f"{len(_test_df)} bars, last close ${float(_test_df['Close'].iloc[-1]):.2f}"
            )
        else:
            _data_detail = "0 bars returned — will retry on first tick"
    except Exception as _e:
        _data_detail = f"fetch failed: {_e}"

    _es_bid, _es_ask = _get_es_bid_ask()
    _es_detail = (
        f"bid={_es_bid:.2f} / ask={_es_ask:.2f} (SPY×10 proxy)"
        if _es_bid > 0 else "unavailable — will retry on each trade"
    )

    # ── Premarket readiness log ───────────────────────────────────────────────
    _order_allowed, _order_reason = _is_order_allowed()
    _order_status = "ENABLED" if _order_allowed else f"BLOCKED — {_order_reason}"
    logger.info("=" * 55)
    logger.info("  SABLE TRADING SYSTEM — PREMARKET READINESS")
    logger.info(f"  {datetime.now(ET).strftime('%Y-%m-%d %H:%M:%S')} ET")
    logger.info("=" * 55)
    logger.info(f"  Instrument:          {TRADERSPOST_TICKER} (MES ${MES_POINT_VALUE:.0f}/point)")
    logger.info(f"  Data source:         SPY 1-min Alpaca bars (×10 proxy)")
    logger.info(f"  marketDataMode:      {MARKET_DATA_MODE}")
    logger.info(f"  allowProxyTrading:   {str(ALLOW_PROXY_TRADING).lower()}")
    logger.info(f"  dryRun:              {str(DRY_RUN).lower()}")
    logger.info(f"  allowShorts:         {str(ALLOW_SHORTS).lower()}")
    logger.info(f"  longOnly:            {str(LONG_ONLY).lower()}")
    logger.info("  " + "-" * 51)
    logger.info(f"  Real/eval orders:    {_order_status}")
    logger.info("  " + "-" * 51)
    logger.info(f"  Data pipeline:       {'OK — ' + _data_detail if _data_ok else 'WARN — ' + _data_detail}")
    logger.info(f"  ES price:            {_es_detail}")
    logger.info("=" * 55)
    if MARKET_DATA_MODE == "PROXY":
        logger.warning(
            "WARNING: Using SPY×10 proxy for MES pricing. "
            "Signals may not match actual futures candles. "
            "Real/eval orders disabled unless ALLOW_PROXY_TRADING=true."
        )

    _startup_catchup()

    _scheduler = BlockingScheduler(timezone="America/New_York")

    # Every 1 minute during session hours (Mon-Fri) — proven gate setting
    _scheduler.add_job(
        five_min_bar_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour="9-16", minute="*"),
        id="five_min_bar",
        name="1-Minute Bar Job",
        misfire_grace_time=30,
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
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour=9, minute=25),
        id="market_open",
        name="Market Open Reset",
    )

    # Morning brief email at 9:25 AM ET Mon-Fri
    _scheduler.add_job(
        morning_brief_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour=9, minute=25),
        id="morning_brief",
        name="Morning Brief Email",
        misfire_grace_time=7200,  # fire if within 2 hours of scheduled time
    )

    # Noon update email at 12:00 PM ET Mon-Fri
    _scheduler.add_job(
        noon_update_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour=12, minute=0),
        id="noon_update",
        name="Noon Update Email",
        misfire_grace_time=7200,
    )

    # End of day at 3:55 PM ET Mon-Fri — must fire before 4:00 PM close so that
    # the DAY-order sell executes within the regular session.
    _scheduler.add_job(
        end_of_day_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour=15, minute=55),
        id="end_of_day",
        name="End of Day",
        misfire_grace_time=14400,  # fire if within 4 hours of scheduled time
    )

    # Regime status log every 30 minutes during session (diagnostic — never blocks)
    _scheduler.add_job(
        regime_log_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour="9-16", minute="*/30"),
        id="regime_log",
        name="Regime Log",
    )

    # Discord daily summary at 4:30 PM ET Mon-Fri (after market close)
    _scheduler.add_job(
        discord_summary_job,
        CronTrigger(day_of_week="mon-fri", timezone="America/New_York", hour=16, minute=30),
        id="discord_summary",
        name="Discord Daily Summary",
        misfire_grace_time=3600,
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

    try:
        import discord_bot
        discord_bot.set_state_getter(lambda: _state)
    except Exception:
        pass

    thread = threading.Thread(target=main, daemon=True, name="apex-trader")
    thread.start()
    logger.info(f"Trading worker started in background thread ({thread.name})")
    return thread


if __name__ == "__main__":
    main()
