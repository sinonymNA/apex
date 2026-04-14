"""
tests/test_risk.py — Unit tests for the Apex Trading System risk engine.

Tests every risk scenario:
  - Daily loss limit
  - Kill switch at 3 consecutive losses
  - Trailing drawdown breach
  - Trade count maximum
  - Time blackout before 10:00 AM ET
  - Time blackout after 3:30 PM ET
  - Drawdown status levels
  - Happy path (all checks pass)

Run: pytest tests/ -v
"""
import pytest
import pytz
from datetime import datetime, time

from worker.risk import (
    pre_trade_check,
    check_kill_switch,
    get_drawdown_status,
    MAX_DAILY_LOSS,
    TRAILING_DD_LIMIT,
    KILL_CONSECUTIVE_LOSSES,
    MAX_TRADES_PER_DAY,
)

ET = pytz.timezone("America/New_York")


def make_et(hour: int, minute: int = 0) -> datetime:
    """Create a timezone-aware datetime in ET at the given hour:minute."""
    now = datetime.now(ET)
    return now.replace(hour=hour, minute=minute, second=0, microsecond=0)


# ── pre_trade_check tests ──────────────────────────────────────────────────────
class TestDailyLossLimit:
    def test_blocks_at_exact_limit(self):
        result = pre_trade_check(
            daily_pnl=MAX_DAILY_LOSS,
            trade_count=0,
            time_et=make_et(11),
            consecutive_losses=0,
        )
        assert result["approved"] is False
        assert "loss" in result["reason"].lower()

    def test_blocks_below_limit(self):
        result = pre_trade_check(
            daily_pnl=MAX_DAILY_LOSS - 100,
            trade_count=0,
            time_et=make_et(11),
            consecutive_losses=0,
        )
        assert result["approved"] is False

    def test_allows_above_limit(self):
        """Just above the limit should not be blocked by this check alone."""
        result = pre_trade_check(
            daily_pnl=MAX_DAILY_LOSS + 1,
            trade_count=0,
            time_et=make_et(11),
            consecutive_losses=0,
        )
        # Could be blocked by other checks, but NOT by the loss limit check
        assert "loss limit" not in result["reason"].lower()


class TestKillSwitchConsecutiveLosses:
    def test_blocks_at_kill_threshold(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(11),
            consecutive_losses=KILL_CONSECUTIVE_LOSSES,
        )
        assert result["approved"] is False
        assert "consecutive" in result["reason"].lower() or "kill" in result["reason"].lower()

    def test_allows_one_below_threshold(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(11),
            consecutive_losses=KILL_CONSECUTIVE_LOSSES - 1,
        )
        # Should not be blocked by consecutive losses check
        assert "consecutive" not in result["reason"].lower() or result["approved"] is True


class TestTrailingDrawdownBreach:
    def test_pause_at_trailing_dd_limit(self):
        peak = 100_000.0
        current = peak + TRAILING_DD_LIMIT  # exactly at limit
        result = check_kill_switch(
            consecutive_losses=0,
            daily_pnl=0.0,
            peak_equity=peak,
            current_equity=current,
        )
        assert result["pause"] is True
        assert "drawdown" in result["reason"].lower()

    def test_pause_below_trailing_dd_limit(self):
        peak = 100_000.0
        current = peak + TRAILING_DD_LIMIT - 1  # worse than limit
        result = check_kill_switch(
            consecutive_losses=0,
            daily_pnl=0.0,
            peak_equity=peak,
            current_equity=current,
        )
        assert result["pause"] is True

    def test_no_pause_above_trailing_dd_limit(self):
        peak = 100_000.0
        current = peak - 1000  # small drawdown, within limits
        result = check_kill_switch(
            consecutive_losses=0,
            daily_pnl=0.0,
            peak_equity=peak,
            current_equity=current,
        )
        assert result["pause"] is False


class TestTradeCountMax:
    def test_blocks_at_max(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=MAX_TRADES_PER_DAY,
            time_et=make_et(11),
            consecutive_losses=0,
        )
        assert result["approved"] is False
        assert "max" in result["reason"].lower() or "trade" in result["reason"].lower()

    def test_allows_below_max(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=MAX_TRADES_PER_DAY - 1,
            time_et=make_et(11),
            consecutive_losses=0,
        )
        # Not blocked by trade count
        assert "max trades" not in result["reason"].lower() or result["approved"] is True


class TestTimeBlackout:
    def test_blocks_before_10am(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(9, 30),
            consecutive_losses=0,
        )
        assert result["approved"] is False
        assert "early" in result["reason"].lower() or "10" in result["reason"]

    def test_blocks_at_exactly_10am_minus_1_min(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(9, 59),
            consecutive_losses=0,
        )
        assert result["approved"] is False

    def test_allows_at_10am(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(10, 0),
            consecutive_losses=0,
        )
        # Should not be blocked by the time check
        assert "early" not in result["reason"].lower()
        assert "10" not in result["reason"] or result["approved"] is True

    def test_blocks_after_3_30pm(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(15, 45),
            consecutive_losses=0,
        )
        assert result["approved"] is False
        assert "late" in result["reason"].lower() or "3:30" in result["reason"]

    def test_blocks_at_exactly_3_30pm(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(15, 30),
            consecutive_losses=0,
        )
        assert result["approved"] is False

    def test_allows_at_3_29pm(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(15, 29),
            consecutive_losses=0,
        )
        # Not blocked by the afternoon time check
        assert "3:30" not in result["reason"] or result["approved"] is True


# ── Happy path ────────────────────────────────────────────────────────────────
class TestApprovedNormalConditions:
    def test_all_checks_pass(self):
        result = pre_trade_check(
            daily_pnl=-100.0,
            trade_count=1,
            time_et=make_et(11, 30),
            consecutive_losses=1,
        )
        assert result["approved"] is True
        assert result["reason"] == "All checks passed"

    def test_zero_pnl_approved(self):
        result = pre_trade_check(
            daily_pnl=0.0,
            trade_count=0,
            time_et=make_et(14, 0),
            consecutive_losses=0,
        )
        assert result["approved"] is True


# ── get_drawdown_status tests ─────────────────────────────────────────────────
class TestDrawdownStatus:
    def test_normal_at_zero(self):
        assert get_drawdown_status(0) == "NORMAL"

    def test_normal_at_minus_750(self):
        assert get_drawdown_status(-750) == "NORMAL"

    def test_warning_at_minus_751(self):
        assert get_drawdown_status(-751) == "WARNING"

    def test_warning_at_minus_1125(self):
        assert get_drawdown_status(-1125) == "WARNING"

    def test_critical_at_minus_1126(self):
        assert get_drawdown_status(-1126) == "CRITICAL"

    def test_critical_at_minus_1500(self):
        assert get_drawdown_status(-1500) == "CRITICAL"

    def test_emergency_at_minus_1501(self):
        assert get_drawdown_status(-1501) == "EMERGENCY"

    def test_emergency_at_large_loss(self):
        assert get_drawdown_status(-5000) == "EMERGENCY"


# ── check_kill_switch additional tests ────────────────────────────────────────
class TestKillSwitchFunction:
    def test_kills_on_consecutive_losses(self):
        result = check_kill_switch(
            consecutive_losses=KILL_CONSECUTIVE_LOSSES,
            daily_pnl=0.0,
            peak_equity=100_000,
            current_equity=100_000,
        )
        assert result["pause"] is True

    def test_kills_on_daily_loss(self):
        result = check_kill_switch(
            consecutive_losses=0,
            daily_pnl=MAX_DAILY_LOSS,
            peak_equity=100_000,
            current_equity=100_000,
        )
        assert result["pause"] is True

    def test_no_kill_all_clear(self):
        result = check_kill_switch(
            consecutive_losses=1,
            daily_pnl=-500,
            peak_equity=100_000,
            current_equity=99_000,
        )
        assert result["pause"] is False
        assert result["reason"] == "No kill switch triggered"
