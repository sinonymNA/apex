"""
worker/risk.py — Risk engine for Apex Trading System.

All constants are hardcoded. No runtime override. Ever.
Pure functions — no DB imports, no external side effects.
This isolation makes unit testing trivial.
"""
from datetime import datetime, time
import pytz

# ── Hardcoded constants — DO NOT OVERRIDE AT RUNTIME ─────────────────────────
MAX_DAILY_LOSS = -1500          # Stop trading for the day if daily P&L hits this
TRAILING_DD_LIMIT = -2800       # Kill switch if drawdown from peak exceeds this
MAX_CONTRACTS = 2               # Maximum position size (shares for SPY paper phase)
MAX_TRADES_PER_DAY = 3          # Hard cap on trades per session
NEWS_BLACKOUT_PRE_MIN = 5       # Minutes before known news event to block entry
NEWS_BLACKOUT_POST_MIN = 8      # Minutes after known news event to block entry
KILL_CONSECUTIVE_LOSSES = 3     # Pause if this many losses in a row

ET = pytz.timezone("America/New_York")

# ── Static news blackout schedule ─────────────────────────────────────────────
# Format: (month, weekday_of_month, hour_et, minute_et)
# weekday_of_month = 0 means "every day of that month at that time"
# weekday_of_month = 1 means "first Friday" (for NFP), etc.
# TODO: integrate live economic calendar API (e.g., Tradier, Polygon)
_RECURRING_HIGH_IMPACT = [
    # NFP — first Friday of every month, 8:30 AM ET
    {"label": "NFP", "hour": 8, "minute": 30, "day_type": "first_friday"},
    # CPI — usually second Wednesday, 8:30 AM ET (approximated as any Wednesday 8:30)
    {"label": "CPI/PPI", "hour": 8, "minute": 30, "day_type": "wednesday"},
    # FOMC — no fixed schedule; operator should pause manually
    # Market open volatility — always block before 10:00 AM (handled by time window)
]


def _in_news_blackout(time_et: datetime) -> tuple[bool, str]:
    """Return (True, label) if current time is within a news blackout window."""
    t = time_et if isinstance(time_et, time) else time_et.time()
    h, m = t.hour, t.minute
    current_minutes = h * 60 + m

    weekday = time_et.weekday() if hasattr(time_et, "weekday") else 0  # 0=Monday, 4=Friday

    for event in _RECURRING_HIGH_IMPACT:
        event_minutes = event["hour"] * 60 + event["minute"]
        window_start = event_minutes - NEWS_BLACKOUT_PRE_MIN
        window_end = event_minutes + NEWS_BLACKOUT_POST_MIN

        day_match = False
        if event["day_type"] == "first_friday":
            # Approximate: any Friday in the first 7 days of the month
            day_match = (weekday == 4 and time_et.day <= 7)
        elif event["day_type"] == "wednesday":
            day_match = (weekday == 2)
        elif event["day_type"] == "any":
            day_match = True

        if day_match and window_start <= current_minutes <= window_end:
            return True, event["label"]

    return False, ""


def pre_trade_check(
    daily_pnl: float,
    trade_count: int,
    time_et: datetime,
    consecutive_losses: int,
) -> dict:
    """
    Run all pre-trade risk checks in priority order.
    Returns the first failing check.

    Args:
        daily_pnl: Cumulative P&L for the session (negative = loss)
        trade_count: Number of completed trades today
        time_et: Current datetime in ET timezone
        consecutive_losses: Number of consecutive losing trades

    Returns:
        {"approved": bool, "reason": str}
    """
    # 1. Daily loss limit
    if daily_pnl <= MAX_DAILY_LOSS:
        return {"approved": False, "reason": f"Daily loss limit reached (${daily_pnl:.0f} <= ${MAX_DAILY_LOSS})"}

    # 2. Max trades per day
    if trade_count >= MAX_TRADES_PER_DAY:
        return {"approved": False, "reason": f"Max trades per day reached ({trade_count}/{MAX_TRADES_PER_DAY})"}

    # 3. Time window — before 10:00 AM ET
    t = time_et.time() if hasattr(time_et, "time") else time_et
    if t < time(10, 0):
        return {"approved": False, "reason": f"Too early — market opens at 10:00 AM ET (current: {t.strftime('%H:%M')})"}

    # 4. Time window — after 3:30 PM ET
    if t >= time(15, 30):
        return {"approved": False, "reason": f"Too late — no entries after 3:30 PM ET (current: {t.strftime('%H:%M')})"}

    # 5. Consecutive losses kill switch
    if consecutive_losses >= KILL_CONSECUTIVE_LOSSES:
        return {
            "approved": False,
            "reason": f"Kill switch: {consecutive_losses} consecutive losses (limit: {KILL_CONSECUTIVE_LOSSES})",
        }

    # 6. News blackout
    in_blackout, event_label = _in_news_blackout(time_et)
    if in_blackout:
        return {"approved": False, "reason": f"News blackout window: {event_label}"}

    return {"approved": True, "reason": "All checks passed"}


def check_kill_switch(
    consecutive_losses: int,
    daily_pnl: float,
    peak_equity: float,
    current_equity: float,
) -> dict:
    """
    Check if the kill switch should be triggered.

    Args:
        consecutive_losses: Current streak of losing trades
        daily_pnl: Today's cumulative P&L
        peak_equity: Highest account equity reached
        current_equity: Current account equity

    Returns:
        {"pause": bool, "reason": str}
    """
    # 1. Consecutive losses
    if consecutive_losses >= KILL_CONSECUTIVE_LOSSES:
        return {
            "pause": True,
            "reason": f"Kill switch: {consecutive_losses} consecutive losses",
        }

    # 2. Daily loss limit
    if daily_pnl <= MAX_DAILY_LOSS:
        return {
            "pause": True,
            "reason": f"Kill switch: daily loss limit ${daily_pnl:.0f} <= ${MAX_DAILY_LOSS}",
        }

    # 3. Trailing drawdown from peak
    trailing_dd = current_equity - peak_equity
    if trailing_dd <= TRAILING_DD_LIMIT:
        return {
            "pause": True,
            "reason": f"Kill switch: trailing drawdown ${trailing_dd:.0f} <= ${TRAILING_DD_LIMIT}",
        }

    return {"pause": False, "reason": "No kill switch triggered"}


def get_drawdown_status(daily_pnl: float) -> str:
    """
    Classify current drawdown severity.

    Returns:
        "NORMAL" | "WARNING" | "CRITICAL" | "EMERGENCY"
    """
    if daily_pnl >= -750:
        return "NORMAL"
    elif daily_pnl >= -1125:
        return "WARNING"
    elif daily_pnl >= -1500:
        return "CRITICAL"
    else:
        return "EMERGENCY"
