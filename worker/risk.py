"""
worker/risk.py — Risk engine for Apex Trading System.

All constants are hardcoded. No runtime override. Ever.
Pure functions — no DB imports, no external side effects.
This isolation makes unit testing trivial.
"""
from datetime import datetime, time
import pytz

# ── Funded account parameters ─────────────────────────────────────────────────
# Calibrated for Apex $50K MES evaluation account:
#   Profit target:   $3,000 (evaluation passes when cumulative P&L reaches this)
#   Trailing DD:     $2,500 (broker blows the account if equity drops this far from peak)
#   Consistency:     40%    (Tradeify rule: no single day > 40% of TOTAL cumulative P&L)
FUNDED_PROFIT_TARGET = 3_000.0   # stop trading and withdraw when eval_pnl >= this
FUNDED_TRAILING_DD   = 2_500.0   # broker's hard limit (informational — we stop earlier)
# Tradeify's consistency rule is 40% of TOTAL cumulative profit (not a fixed daily cap).
# We keep a 33% guard so the biggest day stays comfortably under the line at the $3K pass.
CONSISTENCY_LIMIT    = 0.33      # 33% of target ($990) — margin under Tradeify's 40% rule

# ── Kill switch constants — DO NOT OVERRIDE AT RUNTIME ───────────────────────
MAX_DAILY_LOSS     = -2_000      # Phase 1 default; overridden by get_phase_limits() per trade
TRAILING_DD_LIMIT  = -2_300      # Our kill switch: $200 safety buffer inside the $2,500 funded limit
MAX_CONTRACTS      = 1           # SPY paper proxy shares (MES size controlled per-signal)
MAX_TRADES_PER_DAY = 4           # Hard cap on trades per session
NEWS_BLACKOUT_PRE_MIN  = 5       # Minutes before known news event to block entry
NEWS_BLACKOUT_POST_MIN = 8       # Minutes after known news event to block entry
KILL_CONSECUTIVE_LOSSES = 3      # Pause if this many losses in a row

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


def get_phase_limits(eval_pnl: float) -> dict:
    """
    Return risk limits for the current eval phase based on cumulative P&L.

    Calibrated for Tradeify $50K Select ($3K target, 40%-of-total consistency rule):
      Phase 1 ($0–$1,000):    full risk, $1,000/day cap
      Phase 2 ($1,000–$2,400): reduced risk, $900/day
      Phase 3 ($2,400–$3,000): conservative, $750/day  (final stretch)

    Daily profit caps are deliberately generous so winning days can compound through
    2–3 trades instead of locking after the first. The CONSISTENCY_LIMIT guard ($990)
    is the real ceiling — it keeps any single day under Tradeify's 40%-of-total rule.
    Daily loss limits remain the hard protective floor and are unchanged.

    Returns:
        {"phase": int, "max_risk": float, "daily_loss": float, "daily_profit_target": float}
    """
    if eval_pnl >= 2_400:
        return {"phase": 3, "max_risk": 750.0, "daily_loss": -500.0, "daily_profit_target": 750.0}
    elif eval_pnl >= 1_000:
        return {"phase": 2, "max_risk": 1_000.0, "daily_loss": -550.0, "daily_profit_target": 900.0}
    else:
        return {"phase": 1, "max_risk": 1_250.0, "daily_loss": -600.0, "daily_profit_target": 1_000.0}


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
    eval_pnl: float = 0.0,
    funded_mode: bool = False,
) -> dict:
    """
    Run all pre-trade risk checks in priority order.
    Returns the first failing check.

    Args:
        daily_pnl: Today's session P&L (negative = loss)
        trade_count: Number of completed trades today
        time_et: Current datetime in ET timezone
        consecutive_losses: Number of consecutive losing trades
        eval_pnl: Cumulative eval P&L (used for phase-based daily limit)
        funded_mode: True on funded account — skips eval target gate, daily profit cap,
                     and consistency cap (no eval rules apply post-passing)

    Returns:
        {"approved": bool, "reason": str}
    """
    # 0. Evaluation complete — stop all trading immediately (eval mode only)
    if not funded_mode and eval_pnl >= FUNDED_PROFIT_TARGET:
        return {"approved": False, "reason": f"EVALUATION PASSED — ${eval_pnl:.0f} >= ${FUNDED_PROFIT_TARGET:.0f} target. Stop trading and withdraw!"}

    # 1. Phase-based daily loss limit (and profit target in eval mode)
    phase_limits = get_phase_limits(eval_pnl)
    daily_limit  = phase_limits["daily_loss"]
    daily_target = phase_limits["daily_profit_target"]
    if daily_pnl <= daily_limit:
        return {"approved": False, "reason": f"Daily loss limit reached (${daily_pnl:.0f} <= ${daily_limit:.0f}, Phase {phase_limits['phase']})"}
    if not funded_mode and daily_pnl >= daily_target:
        return {"approved": False, "reason": f"Daily profit target reached (${daily_pnl:.0f} >= ${daily_target:.0f}, Phase {phase_limits['phase']}) — locking in the day"}

    # 1b. Consistency rule — eval mode only (not applicable on funded account)
    if not funded_mode:
        consistency_cap = FUNDED_PROFIT_TARGET * CONSISTENCY_LIMIT  # $570
        if daily_pnl >= consistency_cap:
            return {"approved": False, "reason": f"Consistency cap: ${daily_pnl:.0f} >= ${consistency_cap:.0f} (19% of ${FUNDED_PROFIT_TARGET:.0f} target)"}

    # 2. Max trades per day
    if trade_count >= MAX_TRADES_PER_DAY:
        return {"approved": False, "reason": f"Max trades per day reached ({trade_count}/{MAX_TRADES_PER_DAY})"}

    # 3. Time window — before 9:30 AM ET (market open)
    t = time_et.time() if hasattr(time_et, "time") else time_et
    if t < time(9, 30):
        return {"approved": False, "reason": f"Too early — market opens at 9:30 AM ET (current: {t.strftime('%H:%M')})"}

    # 4. Time window — after 3:45 PM ET
    if t >= time(15, 45):
        return {"approved": False, "reason": f"Too late — no entries after 3:45 PM ET (current: {t.strftime('%H:%M')})"}

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
