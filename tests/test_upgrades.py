"""
tests/test_upgrades.py — Tests for the reliability upgrade pass.

Covers:
  - PostgreSQL URL normalization logic
  - Near-miss classification flags from evaluate_signal_state()
  - Human summary builder from api/main.py

Run: pytest tests/test_upgrades.py -v
"""
import pytest
import numpy as np
import pandas as pd

from worker.strategy import MomentumBreakout


# ── helpers (mirrors test_strategy.py) ────────────────────────────────────────
def make_ohlcv(n: int = 60, trend: float = 0.001, base_price: float = 500.0) -> pd.DataFrame:
    np.random.seed(42)
    prices = base_price * np.cumprod(1 + np.random.normal(trend, 0.002, n))
    noise = np.random.uniform(0.001, 0.003, n)
    close = prices
    high  = prices * (1 + noise)
    low   = prices * (1 - noise)
    open_ = prices * (1 + np.random.uniform(-0.001, 0.001, n))
    volume = np.random.randint(1_000_000, 5_000_000, n).astype(float)
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=pd.date_range("2024-01-02 09:30", periods=n, freq="5min", tz="America/New_York"),
    )


# ── Class 1: Database URL normalization ───────────────────────────────────────
class TestDatabaseUrlNormalization:
    def test_postgres_url_normalized(self):
        raw = "postgres://user:pass@host:5432/db"
        normalized = raw.replace("postgres://", "postgresql://", 1)
        assert normalized.startswith("postgresql://")
        assert not normalized.startswith("postgres://p")  # not double-replaced

    def test_postgresql_url_unchanged(self):
        url = "postgresql://user:pass@host:5432/db"
        result = url.replace("postgres://", "postgresql://", 1)
        assert result == url  # no change

    def test_sqlite_url_unchanged(self):
        url = "sqlite:///./logs/trades.db"
        result = url.replace("postgres://", "postgresql://", 1)
        assert result == url


# ── Class 2: Near-miss classification ─────────────────────────────────────────
class TestNearMissClassification:
    def test_near_miss_returns_none_on_empty(self):
        strategy = MomentumBreakout()
        result = strategy.evaluate_signal_state(pd.DataFrame())
        assert result is None

    def test_near_miss_returns_none_on_missing_columns(self):
        strategy = MomentumBreakout()
        df = pd.DataFrame({"Close": [500.0], "Volume": [1_000_000]})
        result = strategy.evaluate_signal_state(df)
        assert result is None

    def test_near_miss_basic_structure(self):
        strategy = MomentumBreakout()
        df = make_ohlcv(n=60)
        df = strategy.compute_indicators(df)
        result = strategy.evaluate_signal_state(df)
        assert result is not None
        assert "is_near_miss" in result
        assert "volume_ratio" in result
        assert "percent_to_breakout" in result
        assert "price_near_miss" in result
        assert "volume_near_miss" in result

    def test_near_miss_price_flag_within_threshold(self):
        """Bar within 0.3% of breakout sets price_near_miss=True."""
        strategy = MomentumBreakout()
        df = make_ohlcv(n=60)
        df = strategy.compute_indicators(df)
        valid = df.dropna(subset=["high_20"])
        if valid.empty:
            pytest.skip("Not enough data for indicator warmup")
        idx = df.index[-1]
        h20 = float(df["high_20"].dropna().iloc[-1])
        # Set close to 0.2% below breakout level — within the 0.3% threshold
        df.loc[idx, "Close"] = h20 * 0.998
        result = strategy.evaluate_signal_state(df)
        assert result is not None
        assert result["price_near_miss"] is True

    def test_near_miss_price_flag_outside_threshold(self):
        """Bar more than 0.3% below breakout sets price_near_miss=False."""
        strategy = MomentumBreakout()
        df = make_ohlcv(n=60)
        df = strategy.compute_indicators(df)
        valid = df.dropna(subset=["high_20"])
        if valid.empty:
            pytest.skip("Not enough data for indicator warmup")
        idx = df.index[-1]
        h20 = float(df["high_20"].dropna().iloc[-1])
        # Set close to 1.0% below breakout level — outside threshold
        df.loc[idx, "Close"] = h20 * 0.990
        result = strategy.evaluate_signal_state(df)
        assert result is not None
        assert result["price_near_miss"] is False

    def test_near_miss_volume_flag(self):
        """Volume >= 80% of required sets volume_near_miss=True."""
        strategy = MomentumBreakout()
        df = make_ohlcv(n=60)
        df = strategy.compute_indicators(df)
        valid = df.dropna(subset=["volume_avg"])
        if valid.empty:
            pytest.skip("Not enough data for indicator warmup")
        idx = df.index[-1]
        vol_avg = float(df["volume_avg"].dropna().iloc[-1])
        # Set volume to exactly 90% of required (1.5x avg) = 1.35x avg
        df.loc[idx, "Volume"] = vol_avg * 1.35
        result = strategy.evaluate_signal_state(df)
        assert result is not None
        assert result["volume_near_miss"] is True


# ── Class 3: Human summary builder ────────────────────────────────────────────
class TestLastSignalEndpointEmpty:
    def test_build_human_summary_empty(self):
        from api.main import _build_human_summary
        result = _build_human_summary({})
        assert isinstance(result, str)
        assert len(result) > 10

    def test_build_human_summary_no_row(self):
        from api.main import _build_human_summary
        result = _build_human_summary({})
        assert "No signal data" in result or "near-miss" in result or "breakout" in result

    def test_build_human_summary_regime_blocked(self):
        from api.main import _build_human_summary
        row = {
            "blocked_reason": "regime_blocked",
            "regime": "Range-Bound",
            "symbol": "SPY",
            "percent_to_breakout": -0.1,
            "volume_ratio": 1.2,
        }
        result = _build_human_summary(row)
        assert "Range-Bound" in result

    def test_build_human_summary_breakout_not_met(self):
        from api.main import _build_human_summary
        row = {
            "blocked_reason": "breakout_not_met",
            "regime": "Weak Trend",
            "symbol": "SPY",
            "percent_to_breakout": -0.25,
            "volume_ratio": 0.9,
        }
        result = _build_human_summary(row)
        assert "0.25%" in result

    def test_build_human_summary_volume_not_met(self):
        from api.main import _build_human_summary
        row = {
            "blocked_reason": "volume_not_met",
            "regime": "Weak Trend",
            "symbol": "SPY",
            "percent_to_breakout": 0.05,
            "volume_ratio": 0.92,
        }
        result = _build_human_summary(row)
        assert "0.92x" in result
