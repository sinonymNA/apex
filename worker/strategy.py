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
    Momentum breakout strategy with ATR-based stops and targets.

    Entry conditions (ALL must be true):
      - Close > 20-bar rolling high (breakout above 20-minute high on 1-min bars)
      - Volume >= 1.5x 20-bar average volume
      - Time is between 9:30 AM and 3:30 PM ET
      - Trades today < MAX_TRADES_PER_DAY

    Risk levels (proven gate settings):
      - Stop:   entry - clamp(ATR14, 0.10, 0.50) × 1.0
      - Target: entry + clamp(ATR14, 0.10, 0.50) × 2.0  (always 2:1)
      - ATR floor prevents stops < 0.10 SPY pts ($50 ES risk)
      - ATR cap prevents stops > 0.50 SPY pts ($250 ES risk)
      - Max hold: 4 hours (240 minutes)
    """

    LOOKBACK = 20
    VOLUME_MULTIPLIER = 1.5
    STOP_ATR = 1.0
    TARGET_ATR = 2.0
    ATR_MIN = 0.10   # floor: prevents sub-$50 ES stops in dead markets
    ATR_MAX = 0.50   # cap: prevents over-$250 ES stops in spike volatility
    MAX_HOLD_MINUTES = 240
    ENTRY_START = time(9, 30)
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
        raw_atr = float(last["atr14"])
        if raw_atr <= 0 or np.isnan(raw_atr):
            return None

        # Clamp ATR to proven guardrails: floor 0.10, cap 0.50 SPY points
        effective_atr = max(self.ATR_MIN, min(self.ATR_MAX, raw_atr))

        return {
            "signal": "BUY",
            "price": float(last["Close"]),
            "atr": effective_atr,
            "raw_atr": raw_atr,
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

    def evaluate_signal_state(self, df: pd.DataFrame) -> dict | None:
        """
        Evaluate last bar's proximity to an entry signal without generating one.
        Called by the worker every bar to log near-miss visibility data.

        Returns dict with near-miss flags, or None if insufficient data.
        # TODO SPY→ES: Symbol is passed from the caller; this method is symbol-agnostic.
        """
        required_cols = {"high_20", "volume_avg", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            return None

        valid = df.dropna(subset=["high_20", "volume_avg"])
        if valid.empty:
            return None

        last = valid.iloc[-1]
        close        = float(last["Close"])
        high_20      = float(last["high_20"])
        volume       = float(last["Volume"])
        volume_avg   = float(last["volume_avg"])
        required_vol = self.VOLUME_MULTIPLIER * volume_avg  # 1.5x

        pct_to_breakout = (close - high_20) / high_20 * 100 if high_20 > 0 else None
        vol_ratio       = volume / required_vol if required_vol > 0 else 0.0

        # Near-miss conditions:
        #   a) close within 0.3% below breakout level
        #   b) volume at least 80% of required breakout volume
        price_near  = pct_to_breakout is not None and pct_to_breakout >= -0.3
        volume_near = vol_ratio >= 0.8

        return {
            "close":               close,
            "breakout_level":      high_20,
            "percent_to_breakout": pct_to_breakout,
            "volume":              volume,
            "required_volume":     required_vol,
            "volume_ratio":        vol_ratio,
            "price_near_miss":     price_near,
            "volume_near_miss":    volume_near,
            "is_near_miss":        price_near or volume_near,
        }
