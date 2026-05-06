"""
mirror/strategy.py — Mirror System v2 strategy engine.

Scores 1-minute futures candles for high-quality trend pullback setups.
All scoring is purely analytical — no orders, no broker calls.

Scoring breakdown (each component 0-25, total 0-100):
  1. Trend quality   — structure, EMA alignment, not a pure spike
  2. Pullback quality — 3-5 controlled pullback candles near EMA zone
  3. Confirmation    — strong reversal candle closing in right direction
  4. Location/risk   — not overextended, not preceded by exhaustion run
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Optional

SCORE_THRESHOLD: int = int(os.getenv("SCORE_THRESHOLD", "75"))


# ── Candle helpers ─────────────────────────────────────────────────────────────

def _ema(values: list[float], span: int) -> list[float]:
    """Exponential moving average series from a list of values."""
    if not values:
        return []
    k = 2.0 / (span + 1)
    result = [values[0]]
    for v in values[1:]:
        result.append(v * k + result[-1] * (1.0 - k))
    return result


def _is_bullish(c: dict) -> bool:
    return c["close"] > c["open"]


def _is_bearish(c: dict) -> bool:
    return c["close"] < c["open"]


def _body(c: dict) -> float:
    return abs(c["close"] - c["open"])


def _avg_body(candles: list[dict]) -> float:
    bodies = [_body(c) for c in candles]
    return sum(bodies) / len(bodies) if bodies else 1.0


def _atr(candles: list[dict]) -> float:
    """Simple average true range over the given candles."""
    if not candles:
        return 1.0
    trs = []
    for i, c in enumerate(candles):
        hl = c["high"] - c["low"]
        if i > 0:
            pc = candles[i - 1]["close"]
            trs.append(max(hl, abs(c["high"] - pc), abs(c["low"] - pc)))
        else:
            trs.append(hl)
    return (sum(trs) / len(trs)) or 1.0


# ── Result dataclass ───────────────────────────────────────────────────────────

@dataclass
class SetupResult:
    direction: str          # "LONG" or "SHORT"
    score: int              # 0-100
    trend_score: int
    pullback_score: int
    confirm_score: int
    location_score: int
    pullback_count: int
    confirm_vs_avg_body: float
    extension_risk: str     # "low", "moderate", "high"
    trend_read: str
    setup_summary: str
    ema9: float
    ema21: float


# ── Strategy class ─────────────────────────────────────────────────────────────

class MirrorStrategy:
    """
    Mirror System v2 — detects trend pullback setups on 1-minute futures candles.

    Structure required:
      [Trend candles] → [3-5 pullback candles] → [Confirmation candle]

    The confirmation candle is candles[-1] (the most recently closed bar).
    """

    MIN_PULLBACK = 3
    MAX_PULLBACK = 6
    MIN_TREND_CANDLES = 4
    EMA_EXTENSION_CAP = 2.0   # ATR multiples from EMA9 = overextended

    def analyze(self, candles: list[dict]) -> list[SetupResult]:
        """
        Analyze the candle list for LONG and SHORT setups.

        candles: list of dicts {open, high, low, close, timestamp}
                 ordered oldest-first, all CLOSED.
        Returns list of SetupResult with score >= 0 (filter by SCORE_THRESHOLD externally).
        """
        min_needed = self.MIN_TREND_CANDLES + self.MIN_PULLBACK + 1
        if len(candles) < min_needed:
            return []

        results = []
        for direction in ("LONG", "SHORT"):
            setup = self._score_setup(candles, direction)
            if setup is not None:
                results.append(setup)
        return results

    # ── Internal scoring ───────────────────────────────────────────────────────

    def _score_setup(self, candles: list[dict], direction: str) -> Optional[SetupResult]:
        """Build and score a directional setup. Returns None if structure is absent."""
        # Compute EMA series over all candles
        closes = [c["close"] for c in candles]
        ema9_series = _ema(closes, 9)
        ema21_series = _ema(closes, 21)
        ema9 = ema9_series[-1]
        ema21 = ema21_series[-1]
        atr14 = _atr(candles[-14:]) if len(candles) >= 14 else _atr(candles)
        avg_body_all = _avg_body(candles)

        # Confirmation = last closed candle
        confirm = candles[-1]

        # Early rejection: confirmation must be in the right direction
        if direction == "LONG" and not _is_bullish(confirm):
            return None
        if direction == "SHORT" and not _is_bearish(confirm):
            return None

        # Find pullback segment ending just before the confirmation candle
        pre_confirm = candles[:-1]
        pb_candles = self._find_pullback(pre_confirm, direction, avg_body_all)
        if pb_candles is None or len(pb_candles) < self.MIN_PULLBACK:
            return None

        # Trend candles are everything before the pullback
        n_trend = len(candles) - len(pb_candles) - 1  # exclude PB + confirm
        if n_trend < self.MIN_TREND_CANDLES:
            return None
        trend_candles = candles[:n_trend]

        # EMA values at the end of the trend segment (before pullback started)
        ema9_at_trend = ema9_series[n_trend - 1]
        ema21_at_trend = ema21_series[n_trend - 1]

        # Score each component
        t_score, trend_read = self._score_trend(
            trend_candles, ema9_at_trend, ema21_at_trend, direction
        )
        p_score = self._score_pullback(pb_candles, ema9, ema21, direction)
        c_score, confirm_vs_avg = self._score_confirmation(confirm, pb_candles, direction)
        l_score, ext_risk = self._score_location(
            confirm, ema9, ema21, atr14, trend_candles, direction
        )

        total = t_score + p_score + c_score + l_score
        summary = (
            f"Trend {t_score}/25 | Pullback {p_score}/25 ({len(pb_candles)} bars) | "
            f"Confirm {c_score}/25 | Location {l_score}/25"
        )

        return SetupResult(
            direction=direction,
            score=total,
            trend_score=t_score,
            pullback_score=p_score,
            confirm_score=c_score,
            location_score=l_score,
            pullback_count=len(pb_candles),
            confirm_vs_avg_body=round(confirm_vs_avg, 2),
            extension_risk=ext_risk,
            trend_read=trend_read,
            setup_summary=summary,
            ema9=round(ema9, 4),
            ema21=round(ema21, 4),
        )

    def _find_pullback(
        self,
        candles: list[dict],
        direction: str,
        avg_body: float,
    ) -> Optional[list[dict]]:
        """
        Scan backwards from the end of candles to find consecutive pullback bars.
        For LONG: red or small-body candles = pullback (price retracing down).
        For SHORT: green or small-body candles = pullback (price retracing up).
        Returns the pullback candle list (oldest first) or None.
        """
        pb = []
        small_body_threshold = avg_body * 0.4

        for i in range(len(candles) - 1, max(len(candles) - self.MAX_PULLBACK - 1, -1), -1):
            c = candles[i]
            if direction == "LONG":
                is_pb = _is_bearish(c) or _body(c) <= small_body_threshold
            else:
                is_pb = _is_bullish(c) or _body(c) <= small_body_threshold

            if is_pb:
                pb.insert(0, c)
            else:
                break

        return pb if pb else None

    def _score_trend(
        self,
        trend_candles: list[dict],
        ema9_at_end: float,
        ema21_at_end: float,
        direction: str,
    ) -> tuple[int, str]:
        """Score trend quality 0-25."""
        score = 0
        notes: list[str] = []

        highs = [c["high"] for c in trend_candles]
        lows = [c["low"] for c in trend_candles]
        closes = [c["close"] for c in trend_candles]
        n = len(trend_candles)

        # Structure: count HH+HL (long) or LH+LL (short) transitions
        hh = sum(1 for i in range(1, n) if highs[i] > highs[i - 1])
        hl = sum(1 for i in range(1, n) if lows[i] > lows[i - 1])
        lh = sum(1 for i in range(1, n) if highs[i] < highs[i - 1])
        ll = sum(1 for i in range(1, n) if lows[i] < lows[i - 1])
        denom = max((n - 1) * 2, 1)

        if direction == "LONG":
            ratio = (hh + hl) / denom
        else:
            ratio = (lh + ll) / denom

        if ratio >= 0.60:
            score += 10
            notes.append("strong structure")
        elif ratio >= 0.40:
            score += 6
            notes.append("moderate structure")
        else:
            score += 2
            notes.append("weak structure")

        # EMA alignment at the end of the trend segment
        last_close = closes[-1]
        if direction == "LONG":
            if last_close > ema9_at_end:
                score += 8
                notes.append("above EMA9")
            elif last_close > ema21_at_end:
                score += 4
                notes.append("above EMA21")
            if ema9_at_end > ema21_at_end:
                score += 7
                notes.append("EMA9>EMA21")
            elif ema9_at_end >= ema21_at_end * 0.9995:
                score += 3
                notes.append("EMA9 reclaiming")
        else:
            if last_close < ema9_at_end:
                score += 8
                notes.append("below EMA9")
            elif last_close < ema21_at_end:
                score += 4
                notes.append("below EMA21")
            if ema9_at_end < ema21_at_end:
                score += 7
                notes.append("EMA9<EMA21")
            elif ema9_at_end <= ema21_at_end * 1.0005:
                score += 3
                notes.append("EMA9 reclaiming down")

        # Penalise a pure one-direction spike (≥90% same-colour candles = no structure)
        bull_n = sum(1 for c in trend_candles if _is_bullish(c))
        bear_n = sum(1 for c in trend_candles if _is_bearish(c))
        if max(bull_n, bear_n) >= n * 0.90:
            score -= 5
            notes.append("spike risk")

        return max(0, min(25, score)), " | ".join(notes) or "unclear"

    def _score_pullback(
        self,
        pb_candles: list[dict],
        ema9: float,
        ema21: float,
        direction: str,
    ) -> int:
        """Score pullback quality 0-25."""
        score = 0
        count = len(pb_candles)

        # Ideal 3-5 bars
        if 3 <= count <= 5:
            score += 15
        elif count == 2:
            score += 8
        elif count >= 6:
            score += 5

        # Controlled pace: total move < sum of individual ranges × 0.7 (not a dump)
        pb_ranges = [c["high"] - c["low"] for c in pb_candles]
        avg_range = sum(pb_ranges) / len(pb_ranges) if pb_ranges else 0
        pb_closes = [c["close"] for c in pb_candles]
        total_move = abs(pb_closes[-1] - pb_closes[0]) if len(pb_closes) > 1 else 0
        if avg_range > 0 and total_move < avg_range * count * 0.7:
            score += 5

        # Approaches EMA zone without aggressively breaking it
        if direction == "LONG":
            pb_low = min(c["low"] for c in pb_candles)
            ema_low = min(ema9, ema21)
            ema_high = max(ema9, ema21)
            if ema_low * 0.998 <= pb_low <= ema_high * 1.003:
                score += 5   # touched the zone cleanly
            elif pb_low >= ema_low * 0.994:
                score += 2   # slightly below EMA but not too far
        else:
            pb_high = max(c["high"] for c in pb_candles)
            ema_low = min(ema9, ema21)
            ema_high = max(ema9, ema21)
            if ema_low * 0.997 <= pb_high <= ema_high * 1.002:
                score += 5
            elif pb_high <= ema_high * 1.006:
                score += 2

        return max(0, min(25, score))

    def _score_confirmation(
        self,
        confirm: dict,
        pb_candles: list[dict],
        direction: str,
    ) -> tuple[int, float]:
        """
        Score confirmation candle 0-25.
        Returns (score, confirm_vs_avg_pullback_body).
        """
        avg_pb_body = _avg_body(pb_candles)
        c_body = _body(confirm)
        vs_avg = c_body / avg_pb_body if avg_pb_body > 0 else 0.0

        # Must be in the right direction
        if direction == "LONG" and not _is_bullish(confirm):
            return 0, vs_avg
        if direction == "SHORT" and not _is_bearish(confirm):
            return 0, vs_avg

        score = 5  # direction matches

        # Body size: 1.5-3× avg pullback body = healthy confirmation
        if 1.5 <= vs_avg <= 3.0:
            score += 8
        elif 1.2 <= vs_avg < 1.5 or 3.0 < vs_avg <= 4.5:
            score += 5
        elif vs_avg >= 1.0:
            score += 2

        # Closes near its high (long) / low (short) — in upper/lower 30% of range
        candle_range = confirm["high"] - confirm["low"]
        if candle_range > 0:
            if direction == "LONG":
                pct_from_low = (confirm["close"] - confirm["low"]) / candle_range
                if pct_from_low >= 0.70:
                    score += 7
                elif pct_from_low >= 0.50:
                    score += 4
            else:
                pct_from_high = (confirm["high"] - confirm["close"]) / candle_range
                if pct_from_high >= 0.70:
                    score += 7
                elif pct_from_high >= 0.50:
                    score += 4

        # Closes beyond the pullback extreme (breaks the micro-high/low)
        if direction == "LONG":
            pb_high = max(c["high"] for c in pb_candles)
            if confirm["close"] > pb_high:
                score += 5
        else:
            pb_low = min(c["low"] for c in pb_candles)
            if confirm["close"] < pb_low:
                score += 5

        return max(0, min(25, score)), vs_avg

    def _score_location(
        self,
        confirm: dict,
        ema9: float,
        ema21: float,
        atr: float,
        trend_candles: list[dict],
        direction: str,
    ) -> tuple[int, str]:
        """
        Score location / extension / risk filter 0-25.
        Starts at 25 and deducts for problems.
        """
        score = 25
        ext_risk = "low"

        price = confirm["close"]

        # How far is price from EMA9 in ATR multiples?
        ema_dist_atr = abs(price - ema9) / atr if atr > 0 else 0
        if ema_dist_atr > self.EMA_EXTENSION_CAP:
            score -= 15
            ext_risk = "high"
        elif ema_dist_atr > 1.3:
            score -= 7
            ext_risk = "moderate"

        # Exhaustion: 4-5 consecutive same-direction candles at the tail of trend
        if trend_candles:
            tail = trend_candles[-5:]
            if direction == "LONG":
                consec = sum(1 for c in tail if _is_bullish(c))
            else:
                consec = sum(1 for c in tail if _is_bearish(c))
            if consec >= 5:
                score -= 8
                ext_risk = "high"
            elif consec >= 4:
                score -= 4
                ext_risk = max(ext_risk, "moderate",
                               key=lambda x: ["low", "moderate", "high"].index(x))

        # Confirmation candle is not a giant emotional spike (> 3× ATR height)
        confirm_range = confirm["high"] - confirm["low"]
        if confirm_range > atr * 3:
            score -= 7
            ext_risk = "high"

        return max(0, min(25, score)), ext_risk
