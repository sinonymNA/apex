"""
worker/strategy.py — Trading strategy for Apex Trading System.

INSTRUMENT: MES (Micro E-mini S&P 500) — $5/point
Using SPY 1-min bars (×10 proxy) for signal generation.
Strategy: VWAP Trend Pullback — high-quality morning setups 9:45–11:30 AM ET.
"""
from datetime import datetime, time, date
import math
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

try:
    from ta.volatility import AverageTrueRange
    _TA_AVAILABLE = True
except ImportError:
    _TA_AVAILABLE = False
    logger.warning("ta library not available — ATR will use manual calculation")

import pytz

ET = pytz.timezone("America/New_York")


class VWAPTrendPullback:
    """
    MES/ES VWAP Trend Pullback Strategy.

    Trades high-quality morning trend pullbacks between 9:45 AM and 11:30 AM ET.
    Uses a 2-bar confirmation: prior bar is a pullback candle near EMA9/VWAP,
    current bar confirms by breaking the pullback candle high (long) or low (short).

    Entry conditions (ALL must be true):
      - Time 9:45–11:30 AM ET
      - Opening range (9:30–9:45) is complete
      - Price broke above OR high (long) or below OR low (short)
      - Price above VWAP + EMA9 > EMA21 (long) | below VWAP + EMA9 < EMA21 (short)
      - Prior bar touched EMA9 or VWAP and closed as a rejection candle
      - Current bar breaks above prior bar high (long) / below prior bar low (short)
      - VWAP crossed <= 3 times in last 30 bars (chop filter)
      - ATR14 valid

    Risk phases (based on cumulative eval P&L):
      Phase 1 ($0–$1k):    $250 max risk, $500 max daily loss
      Phase 2 ($1k–$2.2k): $200 max risk, $400 max daily loss
      Phase 3 ($2.2k–$3k): $150 max risk, $300 max daily loss

    Drawdown gates (from $2,000 trailing limit):
      < $600 remaining: block all trades
      $600–$1,000:      cap risk at $150
      $1,000–$1,500:    cap risk at $200
      > $1,500:         use phase risk
    """

    OR_START = time(9, 30)
    OR_END = time(9, 45)
    ENTRY_START = time(9, 45)
    ENTRY_END = time(11, 30)

    STOP_ATR_MIN = 0.8     # minimum stop width as ATR multiple
    STOP_ATR_MAX = 1.5     # maximum stop width as ATR multiple
    TARGET_R = 2.0         # reward:risk ratio

    ATR_MIN = 0.10         # SPY-point floor for ATR
    ATR_MAX = 0.80         # SPY-point cap for ATR

    VWAP_CHOP_WINDOW = 30  # bars to look back for chop detection
    VWAP_CHOP_MAX = 3      # max VWAP crossings before blocking entry

    PULLBACK_PROXIMITY_PCT = 0.003  # 0.3% tolerance for "near EMA9/VWAP"

    MAX_HOLD_MINUTES = 90  # max hold time in minutes

    MES_POINT_VALUE = 5.0
    MES_MAX_CONTRACTS = 5
    ES_POINT_VALUE = 50.0
    ES_MAX_CONTRACTS = 1

    def __init__(self):
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_date: Optional[date] = None

    def _update_opening_range(self, df: pd.DataFrame) -> None:
        """Update opening range state from bars in the 9:30–9:45 window."""
        today = datetime.now(ET).date()
        if self._or_date == today:
            return
        try:
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                et_times = idx.tz_convert(ET)
            else:
                et_times = idx
            or_mask = (et_times.time >= self.OR_START) & (et_times.time < self.OR_END)
            or_bars = df[or_mask]
            if not or_bars.empty:
                self._or_high = float(or_bars["High"].max())
                self._or_low = float(or_bars["Low"].min())
                self._or_date = today
                logger.info(
                    f"Opening range set: high={self._or_high:.2f} "
                    f"low={self._or_low:.2f} "
                    f"size={self._or_high - self._or_low:.4f} SPY pts"
                )
        except Exception as e:
            logger.debug(f"Opening range update skipped: {e}")

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add VWAP, EMA9, EMA21, ATR14 columns to df.

        Expects uppercase OHLCV columns. Returns df copy with added columns:
          vwap, ema9, ema21, atr14
        """
        df = df.copy()
        required = {"Open", "High", "Low", "Close", "Volume"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")

        # VWAP — cumulative from session open (9:30 AM ET)
        try:
            idx = df.index
            if hasattr(idx, "tz") and idx.tz is not None:
                et_times = idx.tz_convert(ET)
            else:
                et_times = idx
            session_mask = et_times.time >= self.OR_START
            tp = (df["High"] + df["Low"] + df["Close"]) / 3
            tp_vol = (tp * df["Volume"]).where(session_mask, other=np.nan)
            vol_s = df["Volume"].where(session_mask, other=np.nan)
            df["vwap"] = tp_vol.cumsum() / vol_s.cumsum()
        except Exception:
            tp = (df["High"] + df["Low"] + df["Close"]) / 3
            df["vwap"] = (tp * df["Volume"]).cumsum() / df["Volume"].cumsum()

        # EMA 9 and EMA 21
        df["ema9"] = df["Close"].ewm(span=9, adjust=False).mean()
        df["ema21"] = df["Close"].ewm(span=21, adjust=False).mean()

        # ATR 14
        atr_computed = False
        if _TA_AVAILABLE and len(df) >= 15:
            try:
                atr_obj = AverageTrueRange(
                    high=df["High"], low=df["Low"], close=df["Close"],
                    window=14, fillna=False,
                )
                raw = atr_obj.average_true_range()
                df["atr14"] = raw.where(raw > 0, other=np.nan)
                atr_computed = True
            except Exception:
                pass

        if not atr_computed:
            prev_close = df["Close"].shift(1)
            tr = pd.concat([
                df["High"] - df["Low"],
                (df["High"] - prev_close).abs(),
                (df["Low"] - prev_close).abs(),
            ], axis=1).max(axis=1)
            df["atr14"] = tr.rolling(14).mean()

        self._update_opening_range(df)
        return df

    def _count_vwap_crossings(self, df: pd.DataFrame) -> int:
        """Count VWAP crossings in the last VWAP_CHOP_WINDOW bars."""
        window = df.tail(self.VWAP_CHOP_WINDOW + 1).dropna(subset=["vwap", "Close"])
        if len(window) < 2:
            return 0
        above = (window["Close"] > window["vwap"]).astype(int)
        return int(above.diff().abs().sum())

    def _get_phase_risk(self, eval_pnl: float) -> tuple[int, float]:
        """Return (phase_number, max_risk_dollars) based on cumulative eval P&L."""
        if eval_pnl >= 2200:
            return 3, 150.0
        elif eval_pnl >= 1000:
            return 2, 200.0
        else:
            return 1, 250.0

    def _apply_drawdown_gate(
        self, max_risk: float, current_equity: float, peak_equity: float
    ) -> Optional[float]:
        """
        Apply trailing drawdown gates.
        Returns adjusted max_risk, or None to block trading.
        """
        dd_line = peak_equity - 2000.0
        distance = current_equity - dd_line
        if distance < 600:
            return None
        elif distance < 1000:
            return min(max_risk, 150.0)
        elif distance < 1500:
            return min(max_risk, 200.0)
        return max_risk

    def _size_contracts(
        self, stop_distance_spy: float, max_risk: float, instrument: str = "MES"
    ) -> tuple[int, float]:
        """
        Size contracts from stop distance (SPY points) and max risk dollars.
        SPY × 10 ≈ ES, so stop in ES/MES points = stop_spy × 10.
        """
        es_pts = stop_distance_spy * 10
        if instrument == "MES":
            risk_per = es_pts * self.MES_POINT_VALUE
            cap = self.MES_MAX_CONTRACTS
        else:
            risk_per = es_pts * self.ES_POINT_VALUE
            cap = self.ES_MAX_CONTRACTS
        if risk_per <= 0:
            return 0, 0.0
        n = min(cap, math.floor(max_risk / risk_per))
        return n, n * risk_per

    def _check_long_pullback(self, pullback: pd.Series) -> Optional[dict]:
        """
        Check if bar qualifies as a long pullback candle.
        Must touch near EMA9 or VWAP from above and close bullish in upper half.
        """
        pb_o = float(pullback["Open"])
        pb_h = float(pullback["High"])
        pb_l = float(pullback["Low"])
        pb_c = float(pullback["Close"])
        pb_ema9 = float(pullback["ema9"])
        pb_vwap = float(pullback["vwap"])

        if any(np.isnan(v) for v in [pb_o, pb_h, pb_l, pb_c, pb_ema9, pb_vwap]):
            return None

        # Low must touch near EMA9 or VWAP (within 0.3%)
        near_ema9 = pb_l <= pb_ema9 * (1 + self.PULLBACK_PROXIMITY_PCT)
        near_vwap = pb_l <= pb_vwap * (1 + self.PULLBACK_PROXIMITY_PCT)
        if not (near_ema9 or near_vwap):
            return None

        # Must close bullish, in upper half of bar range
        bar_range = pb_h - pb_l
        if bar_range <= 0:
            return None
        if not (pb_c > pb_o and (pb_c - pb_l) / bar_range >= 0.5):
            return None

        return {"high": pb_h, "low": pb_l}

    def _check_short_pullback(self, pullback: pd.Series) -> Optional[dict]:
        """
        Check if bar qualifies as a short pullback candle.
        Must touch near EMA9 or VWAP from below and close bearish in lower half.
        """
        pb_o = float(pullback["Open"])
        pb_h = float(pullback["High"])
        pb_l = float(pullback["Low"])
        pb_c = float(pullback["Close"])
        pb_ema9 = float(pullback["ema9"])
        pb_vwap = float(pullback["vwap"])

        if any(np.isnan(v) for v in [pb_o, pb_h, pb_l, pb_c, pb_ema9, pb_vwap]):
            return None

        # High must touch near EMA9 or VWAP (within 0.3%)
        near_ema9 = pb_h >= pb_ema9 * (1 - self.PULLBACK_PROXIMITY_PCT)
        near_vwap = pb_h >= pb_vwap * (1 - self.PULLBACK_PROXIMITY_PCT)
        if not (near_ema9 or near_vwap):
            return None

        # Must close bearish, in lower half of bar range
        bar_range = pb_h - pb_l
        if bar_range <= 0:
            return None
        if not (pb_c < pb_o and (pb_h - pb_c) / bar_range >= 0.5):
            return None

        return {"high": pb_h, "low": pb_l}

    def generate_signals(
        self,
        df: pd.DataFrame,
        time_et: datetime,
        trades_today: int,
        daily_pnl: float = 0.0,
        current_equity: float = 100_000.0,
        peak_equity: float = 100_000.0,
    ) -> dict | None:
        """
        Evaluate last two bars for a VWAP pullback entry signal.

        Returns signal dict with direction/stop/target/contracts, or None.
        """
        t = time_et.time() if hasattr(time_et, "time") else time_et
        if not (self.ENTRY_START <= t < self.ENTRY_END):
            return None

        if self._or_high is None or self._or_low is None:
            return None

        from worker.risk import MAX_TRADES_PER_DAY
        if trades_today >= MAX_TRADES_PER_DAY:
            return None

        required_cols = {"vwap", "ema9", "ema21", "atr14", "Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            logger.warning("compute_indicators() must be called before generate_signals()")
            return None

        valid = df.dropna(subset=["vwap", "ema9", "ema21", "atr14"])
        if len(valid) < 2:
            return None

        current = valid.iloc[-1]
        pullback = valid.iloc[-2]

        close = float(current["Close"])
        vwap = float(current["vwap"])
        ema9 = float(current["ema9"])
        ema21 = float(current["ema21"])
        raw_atr = float(current["atr14"])

        if np.isnan(raw_atr) or raw_atr <= 0:
            return None
        effective_atr = max(self.ATR_MIN, min(self.ATR_MAX, raw_atr))

        # Chop filter
        vwap_crossings = self._count_vwap_crossings(valid)
        if vwap_crossings > self.VWAP_CHOP_MAX:
            return None

        # Phase + drawdown-gated risk
        eval_pnl = current_equity - 100_000.0
        phase, max_risk = self._get_phase_risk(eval_pnl)
        max_risk = self._apply_drawdown_gate(max_risk, current_equity, peak_equity)
        if max_risk is None:
            return None

        above_vwap = close > vwap
        ema_bullish = ema9 > ema21

        base = {
            "atr": effective_atr,
            "raw_atr": raw_atr,
            "phase": phase,
            "max_risk": max_risk,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "vwap": round(vwap, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "vwap_crossings": vwap_crossings,
            "pullback_high": float(pullback["High"]),
            "pullback_low": float(pullback["Low"]),
        }

        # LONG setup
        if close > self._or_high and above_vwap and ema_bullish:
            pb = self._check_long_pullback(pullback)
            if pb is not None and close > pb["high"]:
                raw_stop = pb["low"]
                stop = max(raw_stop, close - self.STOP_ATR_MAX * effective_atr)
                stop = min(stop, close - self.STOP_ATR_MIN * effective_atr)
                stop_dist = close - stop
                if stop_dist > 0:
                    target = close + self.TARGET_R * stop_dist
                    n, risk_actual = self._size_contracts(stop_dist, max_risk, "MES")
                    if n >= 1:
                        return {
                            **base,
                            "strategy": "MorningVWAP",
                            "signal": "BUY",
                            "direction": "LONG",
                            "price": close,
                            "stop": round(stop, 2),
                            "target": round(target, 2),
                            "stop_distance": round(stop_dist, 4),
                            "contracts": n,
                            "risk_actual": round(risk_actual, 2),
                        }

        # SHORT setup
        if close < self._or_low and not above_vwap and not ema_bullish:
            pb = self._check_short_pullback(pullback)
            if pb is not None and close < pb["low"]:
                raw_stop = pb["high"]
                stop = min(raw_stop, close + self.STOP_ATR_MAX * effective_atr)
                stop = max(stop, close + self.STOP_ATR_MIN * effective_atr)
                stop_dist = stop - close
                if stop_dist > 0:
                    target = close - self.TARGET_R * stop_dist
                    n, risk_actual = self._size_contracts(stop_dist, max_risk, "MES")
                    if n >= 1:
                        return {
                            **base,
                            "strategy": "MorningVWAP",
                            "signal": "SELL",
                            "direction": "SHORT",
                            "price": close,
                            "stop": round(stop, 2),
                            "target": round(target, 2),
                            "stop_distance": round(stop_dist, 4),
                            "contracts": n,
                            "risk_actual": round(risk_actual, 2),
                        }

        return None

    def get_levels(self, entry_price: float, atr: float, direction: str = "LONG") -> dict:
        """Compute stop/target from entry + ATR. Fallback if signal lacks levels."""
        if direction == "LONG":
            stop = round(entry_price - self.STOP_ATR_MIN * atr, 2)
            dist = entry_price - stop
            target = round(entry_price + self.TARGET_R * dist, 2)
        else:
            stop = round(entry_price + self.STOP_ATR_MIN * atr, 2)
            dist = stop - entry_price
            target = round(entry_price - self.TARGET_R * dist, 2)
        return {"entry": round(entry_price, 2), "stop": stop, "target": target}

    def evaluate_signal_state(self, df: pd.DataFrame) -> dict | None:
        """
        Evaluate current bar proximity to a signal (for per-candle logging).
        Returns state dict or None if data is insufficient.
        """
        required = {"vwap", "ema9", "ema21", "Close", "High", "Low", "Open"}
        if not required.issubset(df.columns):
            return None
        valid = df.dropna(subset=["vwap", "ema9", "ema21"])
        if len(valid) < 2:
            return None

        current = valid.iloc[-1]
        pullback = valid.iloc[-2]
        close = float(current["Close"])
        vwap = float(current["vwap"])
        ema9 = float(current["ema9"])
        ema21 = float(current["ema21"])
        above_vwap = close > vwap
        ema_bullish = ema9 > ema21
        or_long_ok = self._or_high is not None and close > self._or_high
        or_short_ok = self._or_low is not None and close < self._or_low
        long_pb = self._check_long_pullback(pullback) if (above_vwap and ema_bullish) else None
        short_pb = self._check_short_pullback(pullback) if (not above_vwap and not ema_bullish) else None
        vwap_crossings = self._count_vwap_crossings(valid)

        return {
            "close": close,
            "vwap": round(vwap, 2),
            "ema9": round(ema9, 2),
            "ema21": round(ema21, 2),
            "above_vwap": above_vwap,
            "ema_bullish": ema_bullish,
            "or_high": self._or_high,
            "or_low": self._or_low,
            "or_long_break": or_long_ok,
            "or_short_break": or_short_ok,
            "pullback_high": float(pullback["High"]),
            "pullback_low": float(pullback["Low"]),
            "long_pullback_valid": long_pb is not None,
            "short_pullback_valid": short_pb is not None,
            "vwap_crossings": vwap_crossings,
            "chop_blocked": vwap_crossings > self.VWAP_CHOP_MAX,
            "is_near_miss": (
                (or_long_ok and above_vwap and ema_bullish) or
                (or_short_ok and not above_vwap and not ema_bullish)
            ),
        }


class OpeningRangeBreakout:
    """
    Opening Range Breakout — first directional move out of the 9:30–9:45 OR.
    Entry window: 9:45–10:15 AM ET. Fires at most once per session.
    Stop = opposite OR extreme + buffer. Target = 2× stop distance.
    Volume confirmation: current bar >= 1.15× 20-bar average volume.
    """

    OR_END          = time(9, 45)
    ENTRY_START     = time(9, 45)
    ENTRY_END       = time(10, 15)

    MIN_OR_RANGE    = 0.15    # filter dead opens (SPY pts)
    MAX_OR_RANGE    = 1.80    # filter chaotic opens
    VOL_MULTIPLIER  = 1.15    # volume confirmation threshold
    TARGET_R        = 2.0
    STOP_BUFFER     = 0.03    # SPY pts beyond OR boundary for stop placement

    MES_POINT_VALUE   = 5.0
    MES_MAX_CONTRACTS = 5

    def __init__(self):
        self._fired_today: Optional[date] = None
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._or_date: Optional[date] = None

    def sync_or(
        self,
        or_high: Optional[float],
        or_low: Optional[float],
        or_date: Optional[date],
    ) -> None:
        self._or_high = or_high
        self._or_low = or_low
        self._or_date = or_date

    def generate_signals(
        self,
        df: pd.DataFrame,
        time_et: datetime,
        current_equity: float = 100_000.0,
        peak_equity: float = 100_000.0,
    ) -> dict | None:
        t = time_et.time() if hasattr(time_et, "time") else time_et
        if not (self.ENTRY_START <= t < self.ENTRY_END):
            return None

        today = time_et.date() if hasattr(time_et, "date") else date.today()
        if self._fired_today == today:
            return None

        if self._or_high is None or self._or_low is None:
            return None

        or_range = self._or_high - self._or_low
        if not (self.MIN_OR_RANGE <= or_range <= self.MAX_OR_RANGE):
            return None

        required = {"Open", "High", "Low", "Close", "Volume", "vwap", "ema9", "ema21", "atr14"}
        if not required.issubset(df.columns):
            return None

        valid = df.dropna(subset=["vwap", "ema9", "ema21", "atr14"])
        if len(valid) < 20:
            return None

        current = valid.iloc[-1]
        close   = float(current["Close"])
        vwap    = float(current["vwap"])
        ema9    = float(current["ema9"])
        ema21   = float(current["ema21"])
        volume  = float(current["Volume"])
        raw_atr = float(current["atr14"])
        avg_vol = float(valid["Volume"].tail(20).mean())

        if np.isnan(raw_atr) or raw_atr <= 0:
            return None
        if avg_vol > 0 and volume < self.VOL_MULTIPLIER * avg_vol:
            return None

        # Phase-based sizing (inline — no inheritance required)
        eval_pnl = current_equity - 100_000.0
        if eval_pnl >= 2200:
            phase, max_risk = 3, 150.0
        elif eval_pnl >= 1000:
            phase, max_risk = 2, 200.0
        else:
            phase, max_risk = 1, 250.0

        # Trailing drawdown gate (mirrors VWAPTrendPullback._apply_drawdown_gate)
        dd_line  = peak_equity - 2000.0
        distance = current_equity - dd_line
        if distance < 600:
            return None
        elif distance < 1000:
            max_risk = min(max_risk, 150.0)
        elif distance < 1500:
            max_risk = min(max_risk, 200.0)

        def _size(stop_dist: float) -> tuple[int, float]:
            es_pts   = stop_dist * 10
            risk_per = es_pts * self.MES_POINT_VALUE
            if risk_per <= 0:
                return 0, 0.0
            n = min(self.MES_MAX_CONTRACTS, math.floor(max_risk / risk_per))
            return n, n * risk_per

        vol_ratio = round(volume / avg_vol, 2) if avg_vol > 0 else 0.0

        base = {
            "strategy":       "ORB",
            "atr":            round(raw_atr, 6),
            "raw_atr":        round(raw_atr, 6),
            "phase":          phase,
            "max_risk":       max_risk,
            "or_high":        self._or_high,
            "or_low":         self._or_low,
            "vwap":           round(vwap, 2),
            "ema9":           round(ema9, 2),
            "ema21":          round(ema21, 2),
            "vwap_crossings": 0,
            "pullback_high":  close,
            "pullback_low":   close,
            "volume_ratio":   vol_ratio,
            "or_range":       round(or_range, 4),
        }

        # LONG breakout
        if close > self._or_high:
            stop      = round(self._or_low - self.STOP_BUFFER, 2)
            stop_dist = close - stop
            if stop_dist > 0:
                target           = round(close + self.TARGET_R * stop_dist, 2)
                n, risk_actual   = _size(stop_dist)
                if n >= 1:
                    self._fired_today = today
                    return {
                        **base,
                        "signal":        "BUY",
                        "direction":     "LONG",
                        "price":         close,
                        "stop":          stop,
                        "target":        target,
                        "stop_distance": round(stop_dist, 4),
                        "contracts":     n,
                        "risk_actual":   round(risk_actual, 2),
                    }

        # SHORT breakout
        if close < self._or_low:
            stop      = round(self._or_high + self.STOP_BUFFER, 2)
            stop_dist = stop - close
            if stop_dist > 0:
                target           = round(close - self.TARGET_R * stop_dist, 2)
                n, risk_actual   = _size(stop_dist)
                if n >= 1:
                    self._fired_today = today
                    return {
                        **base,
                        "signal":        "SELL",
                        "direction":     "SHORT",
                        "price":         close,
                        "stop":          stop,
                        "target":        target,
                        "stop_distance": round(stop_dist, 4),
                        "contracts":     n,
                        "risk_actual":   round(risk_actual, 2),
                    }

        return None


class AfternoonVWAP(VWAPTrendPullback):
    """
    Afternoon VWAP trend pullback (1:00–3:45 PM ET).
    No opening range breakout requirement — relies on VWAP/EMA alignment alone.
    Tighter chop filter (2 crossings vs 3) to survive midday noise.
    """

    ENTRY_START      = time(13, 0)
    ENTRY_END        = time(15, 45)

    STOP_ATR_MIN     = 0.5
    STOP_ATR_MAX     = 1.2
    ATR_MAX          = 0.70
    VWAP_CHOP_WINDOW = 15
    VWAP_CHOP_MAX    = 2

    def generate_signals(
        self,
        df: pd.DataFrame,
        time_et: datetime,
        trades_today: int,
        daily_pnl: float = 0.0,
        current_equity: float = 100_000.0,
        peak_equity: float = 100_000.0,
    ) -> dict | None:
        t = time_et.time() if hasattr(time_et, "time") else time_et
        if not (self.ENTRY_START <= t < self.ENTRY_END):
            return None

        from worker.risk import MAX_TRADES_PER_DAY
        if trades_today >= MAX_TRADES_PER_DAY:
            return None

        required_cols = {"vwap", "ema9", "ema21", "atr14", "Open", "High", "Low", "Close", "Volume"}
        if not required_cols.issubset(df.columns):
            return None

        valid = df.dropna(subset=["vwap", "ema9", "ema21", "atr14"])
        if len(valid) < 2:
            return None

        current = valid.iloc[-1]
        pullback = valid.iloc[-2]

        close   = float(current["Close"])
        vwap    = float(current["vwap"])
        ema9    = float(current["ema9"])
        ema21   = float(current["ema21"])
        raw_atr = float(current["atr14"])

        if np.isnan(raw_atr) or raw_atr <= 0:
            return None
        effective_atr = max(self.ATR_MIN, min(self.ATR_MAX, raw_atr))

        vwap_crossings = self._count_vwap_crossings(valid)
        if vwap_crossings > self.VWAP_CHOP_MAX:
            return None

        eval_pnl = current_equity - 100_000.0
        phase, max_risk = self._get_phase_risk(eval_pnl)
        max_risk = self._apply_drawdown_gate(max_risk, current_equity, peak_equity)
        if max_risk is None:
            return None

        above_vwap  = close > vwap
        ema_bullish = ema9 > ema21

        base = {
            "strategy":       "AfternoonVWAP",
            "atr":            effective_atr,
            "raw_atr":        raw_atr,
            "phase":          phase,
            "max_risk":       max_risk,
            "or_high":        self._or_high,
            "or_low":         self._or_low,
            "vwap":           round(vwap, 2),
            "ema9":           round(ema9, 2),
            "ema21":          round(ema21, 2),
            "vwap_crossings": vwap_crossings,
            "pullback_high":  float(pullback["High"]),
            "pullback_low":   float(pullback["Low"]),
        }

        # LONG — no OR breakout requirement
        if above_vwap and ema_bullish:
            pb = self._check_long_pullback(pullback)
            if pb is not None and close > pb["high"]:
                raw_stop  = pb["low"]
                stop      = max(raw_stop, close - self.STOP_ATR_MAX * effective_atr)
                stop      = min(stop, close - self.STOP_ATR_MIN * effective_atr)
                stop_dist = close - stop
                if stop_dist > 0:
                    target         = close + self.TARGET_R * stop_dist
                    n, risk_actual = self._size_contracts(stop_dist, max_risk, "MES")
                    if n >= 1:
                        return {
                            **base,
                            "signal":        "BUY",
                            "direction":     "LONG",
                            "price":         close,
                            "stop":          round(stop, 2),
                            "target":        round(target, 2),
                            "stop_distance": round(stop_dist, 4),
                            "contracts":     n,
                            "risk_actual":   round(risk_actual, 2),
                        }

        # SHORT — no OR breakout requirement
        if not above_vwap and not ema_bullish:
            pb = self._check_short_pullback(pullback)
            if pb is not None and close < pb["low"]:
                raw_stop  = pb["high"]
                stop      = min(raw_stop, close + self.STOP_ATR_MAX * effective_atr)
                stop      = max(stop, close + self.STOP_ATR_MIN * effective_atr)
                stop_dist = stop - close
                if stop_dist > 0:
                    target         = close - self.TARGET_R * stop_dist
                    n, risk_actual = self._size_contracts(stop_dist, max_risk, "MES")
                    if n >= 1:
                        return {
                            **base,
                            "signal":        "SELL",
                            "direction":     "SHORT",
                            "price":         close,
                            "stop":          round(stop, 2),
                            "target":        round(target, 2),
                            "stop_distance": round(stop_dist, 4),
                            "contracts":     n,
                            "risk_actual":   round(risk_actual, 2),
                        }

        return None


class MultiSessionStrategy:
    """
    Combines three strategies in priority order each bar:
      1. ORB        (9:45–10:15) — opening range breakout, fires once/day
      2. MorningVWAP (9:45–11:30) — VWAP pullback with OR confirmation
      3. AfternoonVWAP (13:00–15:45) — VWAP pullback, no OR requirement

    Opening range state is computed once by _morning and shared to the others.
    """

    def __init__(self):
        self._morning   = VWAPTrendPullback()
        self._orb       = OpeningRangeBreakout()
        self._afternoon = AfternoonVWAP()

    @property
    def MAX_HOLD_MINUTES(self) -> int:
        return self._morning.MAX_HOLD_MINUTES

    def compute_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._morning.compute_indicators(df)
        self._orb.sync_or(self._morning._or_high, self._morning._or_low, self._morning._or_date)
        # Share OR state with afternoon strategy too
        self._afternoon._or_high = self._morning._or_high
        self._afternoon._or_low  = self._morning._or_low
        self._afternoon._or_date = self._morning._or_date
        return df

    def generate_signals(
        self,
        df: pd.DataFrame,
        time_et: datetime,
        trades_today: int,
        daily_pnl: float = 0.0,
        current_equity: float = 100_000.0,
        peak_equity: float = 100_000.0,
    ) -> dict | None:
        # 1. ORB — highest priority during 9:45–10:15 window
        sig = self._orb.generate_signals(df, time_et, current_equity, peak_equity)
        if sig is not None:
            return sig

        # 2. Morning VWAP pullback
        sig = self._morning.generate_signals(
            df, time_et, trades_today,
            daily_pnl=daily_pnl,
            current_equity=current_equity,
            peak_equity=peak_equity,
        )
        if sig is not None:
            return sig

        # 3. Afternoon VWAP pullback
        return self._afternoon.generate_signals(
            df, time_et, trades_today,
            daily_pnl=daily_pnl,
            current_equity=current_equity,
            peak_equity=peak_equity,
        )

    def evaluate_signal_state(self, df: pd.DataFrame) -> dict | None:
        return self._morning.evaluate_signal_state(df)

    def get_levels(self, entry_price: float, atr: float, direction: str = "LONG") -> dict:
        return self._morning.get_levels(entry_price, atr, direction)
