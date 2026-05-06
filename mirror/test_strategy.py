"""
mirror/test_strategy.py — Dry-run tests for MirrorStrategy scoring.

Run with: pytest mirror/test_strategy.py -v
"""
from datetime import datetime, timezone, timedelta

import pytest

from mirror.strategy import MirrorStrategy, SCORE_THRESHOLD


# ── Candle factories ───────────────────────────────────────────────────────────

def _ts(i: int) -> str:
    base = datetime(2024, 1, 2, 14, 30, tzinfo=timezone.utc)
    return (base + timedelta(minutes=i)).strftime("%Y-%m-%dT%H:%M:%S.000Z")


def _candle(o, h, l, c, i=0):
    return {"open": o, "high": h, "low": l, "close": c, "timestamp": _ts(i)}


def _bull(base, body=2.0, wick=0.4, i=0):
    """Bullish candle: opens at base, closes at base+body."""
    return _candle(base, base + body + wick, base - wick, base + body, i)


def _bear(top, body=2.0, wick=0.4, i=0):
    """Bearish candle: opens at top, closes at top-body."""
    return _candle(top, top + wick, top - body - wick, top - body, i)


def _doji(base, i=0):
    return _candle(base, base + 0.3, base - 0.3, base + 0.05, i)


# ── Setup builders ─────────────────────────────────────────────────────────────

def _long_setup():
    """
    Clean LONG sequence:
      8 bullish trend candles (HH+HL) → 4 bearish pullback → 1 strong bullish confirm
    """
    candles = []
    price = 5000.0
    # Trend: 8 bullish candles stepping up 2pts each
    for i in range(8):
        candles.append(_bull(price, body=2.0, wick=0.4, i=i))
        price += 2.0
    # Pullback: 4 bearish candles drifting down ~1pt each
    pb_base = price
    for j in range(4):
        candles.append(_bear(pb_base - j * 1.0, body=0.8, wick=0.2, i=8 + j))
    # Confirmation: large bullish candle, closes well above pullback high
    pb_top = pb_base + 0.2
    candles.append(_candle(
        pb_base - 3.8,          # open near pullback low
        pb_base + 2.5,          # high above pullback high
        pb_base - 4.2,          # low just below open
        pb_base + 2.3,          # close near high
        i=12,
    ))
    return candles


def _short_setup():
    """
    Clean SHORT sequence:
      8 bearish trend candles (LH+LL) → 4 bullish pullback → 1 strong bearish confirm
    """
    candles = []
    price = 5020.0
    for i in range(8):
        candles.append(_bear(price, body=2.0, wick=0.4, i=i))
        price -= 2.0
    pb_base = price
    for j in range(4):
        candles.append(_bull(pb_base + j * 1.0, body=0.8, wick=0.2, i=8 + j))
    # Confirmation: large bearish candle, closes well below pullback low
    pb_bottom = pb_base - 0.2
    candles.append(_candle(
        pb_base + 3.8,
        pb_base + 4.2,
        pb_base - 2.5,
        pb_base - 2.3,
        i=12,
    ))
    return candles


# ── Tests ──────────────────────────────────────────────────────────────────────

class TestLongSetup:
    def test_detects_long(self):
        strat = MirrorStrategy()
        results = strat.analyze(_long_setup())
        longs = [r for r in results if r.direction == "LONG"]
        assert longs, "Expected a LONG setup to be detected"

    def test_long_score_reasonable(self):
        strat = MirrorStrategy()
        results = strat.analyze(_long_setup())
        longs = [r for r in results if r.direction == "LONG"]
        assert longs
        # Synthetic clean setup should score at least 40/100
        assert longs[0].score >= 40, f"Score unexpectedly low: {longs[0].score}"

    def test_long_score_components_sum(self):
        strat = MirrorStrategy()
        for r in strat.analyze(_long_setup()):
            assert r.score == r.trend_score + r.pullback_score + r.confirm_score + r.location_score
            assert 0 <= r.trend_score <= 25
            assert 0 <= r.pullback_score <= 25
            assert 0 <= r.confirm_score <= 25
            assert 0 <= r.location_score <= 25
            assert 0 <= r.score <= 100


class TestShortSetup:
    def test_detects_short(self):
        strat = MirrorStrategy()
        results = strat.analyze(_short_setup())
        shorts = [r for r in results if r.direction == "SHORT"]
        assert shorts, "Expected a SHORT setup to be detected"

    def test_short_score_reasonable(self):
        strat = MirrorStrategy()
        results = strat.analyze(_short_setup())
        shorts = [r for r in results if r.direction == "SHORT"]
        assert shorts
        assert shorts[0].score >= 40, f"Score unexpectedly low: {shorts[0].score}"

    def test_short_score_components_sum(self):
        strat = MirrorStrategy()
        for r in strat.analyze(_short_setup()):
            assert r.score == r.trend_score + r.pullback_score + r.confirm_score + r.location_score


class TestRejections:
    def test_insufficient_candles_returns_empty(self):
        strat = MirrorStrategy()
        # MIN_TREND_CANDLES(4) + MIN_PULLBACK(3) + 1 = 8 needed
        candles = [_bull(5000.0 + i * 0.5, i=i) for i in range(6)]
        assert strat.analyze(candles) == []

    def test_wrong_direction_confirm_blocked(self):
        strat = MirrorStrategy()
        candles = _long_setup()
        # Replace confirmation (last candle) with a bearish bar
        last = candles[-1]
        candles[-1] = _bear(last["high"] + 1.0, body=3.0, i=12)
        results = strat.analyze(candles)
        longs = [r for r in results if r.direction == "LONG"]
        assert not longs, "Bearish confirm should block LONG setup"

    def test_flat_candles_score_low(self):
        strat = MirrorStrategy()
        # All candles at same price — no trend, no pullback structure
        candles = [_candle(5000, 5001, 4999, 5000, i) for i in range(15)]
        results = strat.analyze(candles)
        for r in results:
            assert r.score < SCORE_THRESHOLD, (
                f"Flat candles scored {r.score} >= threshold {SCORE_THRESHOLD}"
            )

    def test_single_spike_penalised(self):
        """≥90% same-colour candles in trend should receive a spike penalty."""
        strat = MirrorStrategy()
        candles = []
        price = 5000.0
        # 10 all-bullish candles (100% bull → spike risk)
        for i in range(10):
            candles.append(_bull(price, body=2.0, i=i))
            price += 2.0
        # 3 bearish pullback
        for j in range(3):
            candles.append(_bear(price - j * 0.5, body=0.5, i=10 + j))
        # 1 bullish confirm
        candles.append(_bull(price - 1.5, body=3.0, i=13))
        results = strat.analyze(candles)
        longs = [r for r in results if r.direction == "LONG"]
        if longs:
            # Spike penalty should reduce trend_score
            assert longs[0].trend_score < 25, "Spike risk should reduce trend score"


class TestSetupResult:
    def test_result_fields_populated(self):
        strat = MirrorStrategy()
        results = strat.analyze(_long_setup())
        for r in results:
            assert r.direction in ("LONG", "SHORT")
            assert isinstance(r.pullback_count, int)
            assert r.pullback_count >= strat.MIN_PULLBACK
            assert isinstance(r.extension_risk, str)
            assert r.extension_risk in ("low", "moderate", "high")
            assert isinstance(r.trend_read, str) and r.trend_read
            assert isinstance(r.setup_summary, str) and r.setup_summary
            assert r.ema9 > 0
            assert r.ema21 > 0
            assert r.confirm_vs_avg_body >= 0.0

    def test_pullback_count_within_bounds(self):
        strat = MirrorStrategy()
        results = strat.analyze(_long_setup())
        for r in results:
            assert strat.MIN_PULLBACK <= r.pullback_count <= strat.MAX_PULLBACK
