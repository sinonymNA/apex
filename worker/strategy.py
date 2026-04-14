"""
worker/strategy.py — Trading strategy for Apex Trading System.

INSTRUMENT: SPY during paper phase.
Switch to ES after 20-day gate + Polygon data purchase.
Only change needed: symbol, data source. Logic identical.
"""
from datetime import datetime, time

import pandas as pd
import numpy as np
from loguru import logger

try:
    from ta.volatility import AverageTrueRange
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("ta library not available — ATR will use manual calculation")

from worker.risk import MAX_TRADES_PER_DAY


class MomentumBreakout:
    """
    20-bar momentum breakout strategy with ATR-based stops and targets.

    Entry conditions (ALL must be true):
      - Close > 20-bar rolling high (breakout)
      - Volume >= 1.5x 20-bar average volume
      - Time is between 10:00 AM and 3:30 PM ET
      - Trades today < MAX_TRADES_PER_DAY

    Risk levels:
      - Stop:   entry - (1.0 x ATR14)
      - Target: entry + (2.0 x ATR14)
      - Max hold: 4 hours (240 minutes)
    """

    LOOKBACK = 20
    VOLUME_MULTIPLIER = 1.5
    STOP_ATR = 1.0
    TARGET_ATR = 2.0
    MAX_HOLD_MINUTES = 240
    ENTRY_START = time(10, 0)
    ENTRY_END = time(15, 30)

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add technical indicator columns to df.

        Expects uppercase OHLCV columns: Open, High, Low, Close, Volume
        Returns the same DataFrame with added columns:
          high_20, volume_avg, atr14

        Note: .shift(1) on high_20 prevents lookahead bias in backtesting.
        """
        df = df.copy()

        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # 20-bar rolling high — shifted 1 bar to avoid lookahead
        df["high_20"] = df["High"].rolling(self.LOOKBACK).max().shift(1)

        # 20-bar average volume — shifted 1 bar
        df["volume_avg"] = df["Volume"].rolling(self.LOOKBACK).mean().shift(1)

        # ATR14 — try ta library first, fall back to manual rolling calculation
        use_ta = _TA_AVAILABLE and len(df) >= 15
        atr_computed = False
        if use_ta:
            try:
                atr_indicator = AverageTrueRange(
                    high=df["High"],
                    low=df["Low"],
                    close=df["Close"],
                    window=14,
                    fillna=False,
                )
                raw_atr = atr_indicator.average_true_range()
                # ta library fills warmup period with 0.0 — convert to NaN for consistency
                df["atr14"] = raw_atr.where(raw_atr > 0, float("nan"))
                atr_computed = True
            except Exception:
                atr_computed = False

        if not atr_computed:
            # Manual ATR calculation (fallback: ta unavailable or too few bars)
            prev_close = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14).mean()

        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        time_et: datetime,
        trades_today: int,
    ) -> dict | None:
        """
        Evaluate the last bar for an entry signal.

        Args:
            df: DataFrame with indicators already computed (from compute_indicators)
            time_et: Current time in ET timezone
            trades_today: Number of trades already placed today

        Returns:
            {"signal": "BUY", "price": float, "atr": float} or None
        """
        # Time window check
        t = time_et.time() if hasattr(time_et, "time") else time_et
        if not (self.ENTRY_START <= t < self.ENTRY_END):
            return None

        # Trade count check
        if trades_today >= MAX_TRADES_PER_DAY:
            return None

        # Need indicator columns
        required_cols = {"high_20", "volume_avg", "atr14", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            logger.warning("compute_indicators() must be called before generate_signals()")
            return None

        # Drop rows with NaN indicators and get the last valid bar
        valid = df.dropna(subset=["high_20", "volume_avg", "atr14"])
        if valid.empty:
            return None

        last = valid.iloc[-1]

        # Breakout check: Close must exceed the prior period's 20-bar high
        if last["Close"] <= last["high_20"]:
            return None

        # Volume confirmation
        if last["Volume"] < self.VOLUME_MULTIPLIER * last["volume_avg"]:
            return None

        # ATR must be positive
        if last["atr14"] <= 0 or np.isnan(last["atr14"]):
            return None

        return {
            "signal": "BUY",
            "price": float(last["Close"]),
            "atr": float(last["atr14"]),
            "volume_ratio": float(last["Volume"] / last["volume_avg"]) if last["volume_avg"] > 0 else 0.0,
        }

    def get_levels(self, entry_price: float, atr: float) -> dict:
        """
        Compute stop and target levels from entry price and ATR.

        Returns:
            {"entry": float, "stop": float, "target": float}
        """
        return {
            "entry": round(entry_price, 2),
            "stop": round(entry_price - self.STOP_ATR * atr, 2),
            "target": round(entry_price + self.TARGET_ATR * atr, 2),
        }
