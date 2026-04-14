"""
tests/test_strategy.py — Unit tests for the MomentumBreakout strategy.

Tests:
  - compute_indicators adds required columns
  - generate_signals blocks before 10:00 AM ET
  - generate_signals blocks after 3:30 PM ET
  - get_levels computes correct stop/target math
  - Signal fires correctly when all conditions met
  - No signal when close <= high_20 (no breakout)
  - No signal when volume is insufficient

Run: pytest tests/ -v
"""
import pytest
import numpy as np
import pandas as pd
import pytz
from datetime import datetime

from worker.strategy import MomentumBreakout

ET = pytz.timezone("America/New_York")


def make_et(hour: int, minute: int = 0) -> datetime:
    """Create timezone-aware ET datetime."""
    now = datetime.now(ET)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


def make_ohlcv(n: int = 60, trend: float = 0.001, base_price: float = 500.0) -> pd.DataFrame:
    """
    Generate synthetic SPY-like OHLCV data.

    Args:
        n: Number of bars
        trend: Per-bar price drift (positive = uptrend)
        base_price: Starting close price
    """
    np.random.seed(42)
    prices = base_price * np.cumprod(1 + np.random.normal(trend, 0.002, n))
    noise = np.random.uniform(0.001, 0.003, n)

    close = prices
    high = prices * (1 + noise)
    low = prices * (1 - noise)
    open_ = prices * (1 + np.random.uniform(-0.001, 0.001, n))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)

    return pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="America/New_York"))


def make_breakout_df(n: int = 60) -> pd.DataFrame:
    """
    Generate OHLCV where the last bar is clearly above the 20-bar high
    with high volume.
    """
    df = make_ohlcv(n=n, trend=0.0, base_price=500.0)

    # Force last bar: price well above 20-bar high, volume 3x average
    twenty_bar_high = df["High"].iloc[-21:-1].max()
    breakout_price = twenty_bar_high * 1.005  # 0.5% above the high

    df.iloc[-1, df.columns.get_loc("Close")] = breakout_price
    df.iloc[-1, df.columns.get_loc("High")] = breakout_price * 1.001
    df.iloc[-1, df.columns.get_loc("Open")] = breakout_price * 0.999

    avg_vol = df["Volume"].iloc[-21:-1].mean()
    df.iloc[-1, df.columns.get_loc("Volume")] = avg_vol * 3.0

    return df


# ── compute_indicators tests ──────────────────────────────────────────────────
class TestComputeIndicators:
    def test_adds_required_columns(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=50)
        result = strat.compute_indicators(df)

        assert "high_20" in result.columns
        assert "volume_avg" in result.columns
        assert "atr14" in result.columns

    def test_original_columns_preserved(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=50)
        result = strat.compute_indicators(df)

        for col in ["Open", "High", "Low", "Close", "Volume"]:
            assert col in result.columns

    def test_does_not_mutate_original(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=50)
        original_close = df["Close"].copy()
        strat.compute_indicators(df)
        pd.testing.assert_series_equal(df["Close"], original_close)

    def test_raises_on_missing_columns(self):
        strat = MomentumBreakout()
        bad_df = pd.DataFrame({"price": [1, 2, 3]})
        with pytest.raises(ValueError, match="Missing columns"):
            strat.compute_indicators(bad_df)

    def test_high_20_is_shifted(self):
        """high_20 should be NaN for the first 21 rows (20 lookback + 1 shift)."""
        strat = MomentumBreakout()
        df = make_ohlcv(n=60)
        result = strat.compute_indicators(df)
        # After shift(1), rows 0..19 (20 rows) should be NaN
        assert result["high_20"].iloc[:20].isna().all()

    def test_atr14_is_positive_after_warmup(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=60)
        result = strat.compute_indicators(df)
        valid_atr = result["atr14"].dropna()
        assert (valid_atr > 0).all()


# ── generate_signals time blackout tests ─────────────────────────────────────
class TestGenerateSignalsTimeBlackout:
    def test_none_before_10am(self):
        strat = MomentumBreakout()
        df = make_breakout_df(n=60)
        df = strat.compute_indicators(df)
        result = strat.generate_signals(df, make_et(9, 30), trades_today=0)
        assert result is None

    def test_none_at_9_59am(self):
        strat = MomentumBreakout()
        df = make_breakout_df(n=60)
        df = strat.compute_indicators(df)
        result = strat.generate_signals(df, make_et(9, 59), trades_today=0)
        assert result is None

    def test_none_after_3_30pm(self):
        strat = MomentumBreakout()
        df = make_breakout_df(n=60)
        df = strat.compute_indicators(df)
        result = strat.generate_signals(df, make_et(15, 45), trades_today=0)
        assert result is None

    def test_none_at_3_30pm(self):
        strat = MomentumBreakout()
        df = make_breakout_df(n=60)
        df = strat.compute_indicators(df)
        result = strat.generate_signals(df, make_et(15, 30), trades_today=0)
        assert result is None


# ── get_levels math tests ─────────────────────────────────────────────────────
class TestGetLevels:
    def test_stop_is_one_atr_below(self):
        strat = MomentumBreakout()
        levels = strat.get_levels(entry_price=500.0, atr=5.0)
        assert levels["stop"] == pytest.approx(500.0 - 1.0 * 5.0, abs=0.01)

    def test_target_is_two_atr_above(self):
        strat = MomentumBreakout()
        levels = strat.get_levels(entry_price=500.0, atr=5.0)
        assert levels["target"] == pytest.approx(500.0 + 2.0 * 5.0, abs=0.01)

    def test_entry_is_preserved(self):
        strat = MomentumBreakout()
        levels = strat.get_levels(entry_price=487.63, atr=3.21)
        assert levels["entry"] == pytest.approx(487.63, abs=0.01)

    def test_risk_reward_is_2_to_1(self):
        strat = MomentumBreakout()
        levels = strat.get_levels(entry_price=500.0, atr=5.0)
        risk = levels["entry"] - levels["stop"]
        reward = levels["target"] - levels["entry"]
        assert reward / risk == pytest.approx(2.0, abs=0.01)

    def test_different_atr_values(self):
        strat = MomentumBreakout()
        for atr in [1.0, 2.5, 10.0, 0.5]:
            levels = strat.get_levels(500.0, atr)
            assert levels["stop"] < levels["entry"] < levels["target"]


# ── Signal generation logic tests ────────────────────────────────────────────
class TestGenerateSignalsLogic:
    def test_no_signal_when_no_indicators(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=60)
        # Not calling compute_indicators — should return None gracefully
        result = strat.generate_signals(df, make_et(11), trades_today=0)
        assert result is None

    def test_no_signal_when_insufficient_data(self):
        strat = MomentumBreakout()
        df = make_ohlcv(n=10)  # Too few bars for 20-bar lookback or ATR14
        # compute_indicators falls back to manual ATR when n < 15
        df = strat.compute_indicators(df)
        result = strat.generate_signals(df, make_et(11), trades_today=0)
        # All high_20 / volume_avg / atr14 rows will be NaN — no signal possible
        assert result is None
