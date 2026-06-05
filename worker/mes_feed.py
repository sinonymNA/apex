"""
worker/mes_feed.py — Real-time MES 1-minute bar feed via Tradovate WebSocket.

Runs a persistent background thread that streams live CME futures bars directly
from Tradovate's market data WebSocket, replacing the SPY×10 proxy.

Bars are buffered in memory and exposed as a pandas DataFrame with prices
divided by 10 so the existing SPY-scale strategy, stop/target math, and
TradersPost ×10 conversions all continue to work without modification.

Usage:
    from worker.mes_feed import start, is_ready, get_bars_df

    # Call once at startup (no-op if TRADOVATE_USERNAME not set):
    started = start()

    # Each polling cycle:
    df = get_bars_df()   # None if <30 bars buffered
"""
import asyncio
import os
import threading
from loguru import logger
import pandas as pd


# ── Module-level bar buffer ────────────────────────────────────────────────────
_bars: list[dict] = []
_bars_lock = threading.Lock()

_feed_started = False
_feed_thread: "threading.Thread | None" = None


# ── Callback fired by TradovateMarketData when a bar closes ───────────────────
def _on_bar_closed(symbol: str, closed_bars: list[dict]) -> None:
    """Replace the buffer with the latest closed bars list."""
    global _bars
    with _bars_lock:
        _bars = list(closed_bars)
    if closed_bars:
        logger.debug(
            f"MES feed: {len(closed_bars)} bars buffered "
            f"(last={closed_bars[-1].get('timestamp', '?')})"
        )


# ── Public API ─────────────────────────────────────────────────────────────────
def start() -> bool:
    """
    Start the Tradovate market data feed in a background daemon thread.

    Returns True if started (or already running), False if TRADOVATE_USERNAME
    is not set (Alpaca proxy fallback will be used instead).
    """
    global _feed_started, _feed_thread

    if not os.getenv("TRADOVATE_USERNAME", ""):
        logger.info(
            "MES feed: TRADOVATE_USERNAME not configured — "
            "feed not started, Alpaca SPY proxy will be used as fallback"
        )
        return False

    if _feed_started:
        return True

    _feed_started = True
    ticker = os.getenv("TRADERSPOST_TICKER", "MESM2026")

    def _run() -> None:
        from mirror.agent import TradovateMarketData
        client = TradovateMarketData(
            on_bar_closed=_on_bar_closed,
            symbols=[ticker],
            history_bars=150,   # 2.5 h of 1-min bars — enough for EMA50 warmup
        )
        # run_forever() reconnects on disconnect and never returns normally
        asyncio.run(client.run_forever())

    _feed_thread = threading.Thread(target=_run, daemon=True, name="mes-feed")
    _feed_thread.start()
    logger.info(f"MES feed: background thread started (symbol={ticker})")
    return True


def is_ready() -> bool:
    """True once 30+ bars have been buffered (enough for strategy indicators)."""
    with _bars_lock:
        return len(_bars) >= 30


def bar_count() -> int:
    with _bars_lock:
        return len(_bars)


def get_bars_df() -> "pd.DataFrame | None":
    """
    Return buffered MES 1-min bars as a DataFrame scaled to SPY-proxy values
    (MES prices ÷ 10). Returns None if fewer than 30 bars are buffered.

    The ÷ 10 scaling means:
      - The existing strategy (ATR, EMA, VWAP, pullback detection) needs no changes
      - signal["stop"] and signal["target"] remain in SPY scale
      - signal["stop"] × 10 continues to give correct MES bracket prices
      - P&L math (spy_move × 10 × qty × $5) remains correct
    """
    with _bars_lock:
        bars = list(_bars)

    if len(bars) < 30:
        return None

    df = pd.DataFrame(bars)
    if df.empty:
        return None

    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.set_index("timestamp").sort_index()

    df = df.rename(columns={
        "open": "Open",
        "high": "High",
        "low": "Low",
        "close": "Close",
        "volume": "Volume",
    })

    # Ensure Volume column exists (Tradovate may not always populate it)
    if "Volume" not in df.columns:
        df["Volume"] = 0.0
    else:
        df["Volume"] = df["Volume"].fillna(0.0)

    # Scale MES prices to SPY-proxy range (÷ 10)
    for col in ("Open", "High", "Low", "Close"):
        if col in df.columns:
            df[col] = df[col] / 10.0

    return df if len(df) >= 30 else None
