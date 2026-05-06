"""
mirror/discord_alerts.py — Discord webhook sender for Mirror Agent.

Message types (in priority order):
  BUY / SELL       — score ≥ SCORE_THRESHOLD (default 75). Full detail, actionable.
  SETUP FORMING    — score ≥ SETUP_FORMING_MIN (default 55). Setup developing.
  WAIT             — score < SETUP_FORMING_MIN. Structure found but weak.
  (silence)        — analyze() returned no result. Nothing forming.

If MIRROR_WAIT_SILENT=true, WAIT messages are logged but not posted to Discord.
"""
import os
from datetime import datetime, timezone, timedelta

import httpx
from loguru import logger

_WEBHOOK_URL: str = (
    os.getenv("DISCORD_MIRROR_WEBHOOK_URL")
    or os.getenv("DISCORD_WEBHOOK_URL")
    or ""
)
ALERT_COOLDOWN_MINUTES: int = int(os.getenv("ALERT_COOLDOWN_MINUTES", "10"))
SETUP_FORMING_MIN: int = int(os.getenv("SETUP_FORMING_MIN", "55"))
MIRROR_WAIT_SILENT: bool = os.getenv("MIRROR_WAIT_SILENT", "false").lower() == "true"

_WAIT_COOLDOWN = 15    # minutes between WAIT posts per symbol
_FORMING_COOLDOWN = 5  # minutes between SETUP FORMING posts per (symbol, direction)

# cooldown trackers
_last_alert: dict[tuple, datetime] = {}    # (symbol, direction) → BUY/SELL
_last_forming: dict[tuple, datetime] = {}  # (symbol, direction) → SETUP FORMING
_last_wait: dict[str, datetime] = {}       # symbol → WAIT


# ── Internal helpers ───────────────────────────────────────────────────────────

def _post(payload: dict) -> None:
    if not _WEBHOOK_URL:
        logger.warning("Mirror: DISCORD_MIRROR_WEBHOOK_URL not set — Discord message skipped")
        return
    try:
        r = httpx.post(_WEBHOOK_URL, json=payload, timeout=5.0)
        if r.status_code not in (200, 204):
            logger.warning(f"Discord webhook {r.status_code}: {r.text[:200]}")
    except Exception as e:
        logger.error(f"Discord webhook failed: {e}")


def _wait_reason(ts: int, ps: int, cs: int, ls: int) -> str:
    """Build a short human-readable reason string from score components."""
    weak: list[str] = []

    if ts <= 5:
        weak.append("no trend")
    elif ts < 12:
        weak.append("weak trend")

    if ps <= 5:
        weak.append("no pullback")
    elif ps < 12:
        weak.append("shallow pullback")

    if cs <= 5:
        weak.append("no confirmation")
    elif cs < 12:
        weak.append("weak confirmation")

    if ls <= 5:
        weak.append("overextended")
    elif ls < 12:
        weak.append("poor location")

    if len(weak) >= 3:
        return "chop, " + weak[-1]
    return ", ".join(weak) if weak else "marginal setup"


def _forming_desc(ts: int, ps: int, cs: int, ls: int, direction: str, pb_count: int) -> str:
    """Build a medium-length description of what's present and what's missing."""
    dir_word = "bullish" if direction == "LONG" else "bearish"
    parts: list[str] = []

    if ts >= 16:
        parts.append(f"trend {dir_word}")
    elif ts >= 10:
        parts.append(f"trend weakly {dir_word}")
    else:
        parts.append("trend unclear")

    if ps >= 16:
        parts.append(f"{pb_count}-candle pullback valid")
    elif ps >= 10:
        parts.append("pullback forming")
    else:
        parts.append("pullback shallow")

    if cs >= 16:
        parts.append("confirmation strong")
    elif cs >= 10:
        parts.append("confirmation forming")
    else:
        parts.append("confirmation not closed")

    if ls < 10:
        parts.append("location risk")

    return ", ".join(parts)


# ── Public senders ─────────────────────────────────────────────────────────────

def send_startup(symbols: list[str]) -> None:
    _post({
        "content": (
            "🟢 **Mirror Agent online.**\n"
            f"Mode: **ALERT-ONLY** — No live trades enabled.\n"
            f"Watching: `{'`, `'.join(symbols)}`\n"
            "Paper simulation active. Alerts fire when Mirror System v2 score ≥ threshold."
        )
    })


def send_error(msg: str) -> None:
    _post({"content": f"🔴 **Mirror Agent error:**\n{msg}"})


def send_wait(
    *,
    symbol: str,
    direction: str,
    score: int,
    trend_score: int,
    pullback_score: int,
    confirm_score: int,
    location_score: int,
) -> bool:
    """
    Post a WAIT message (low score, no actionable setup).
    Always logs internally. Skips Discord if MIRROR_WAIT_SILENT=true.
    Returns True if posted to Discord.
    """
    reason = _wait_reason(trend_score, pullback_score, confirm_score, location_score)
    logger.debug(f"WAIT [{symbol} {direction} {score}/100] {reason}")

    if MIRROR_WAIT_SILENT:
        return False

    now = datetime.now(timezone.utc)
    last = _last_wait.get(symbol)
    if last and (now - last) < timedelta(minutes=_WAIT_COOLDOWN):
        return False
    _last_wait[symbol] = now

    _post({
        "content": (
            f"⏸ **WAIT — {symbol}**\n"
            f"Score: {score}/100\n"
            f"Reason: {reason}."
        )
    })
    return True


def send_setup_forming(
    *,
    symbol: str,
    direction: str,
    score: int,
    trend_score: int,
    pullback_score: int,
    confirm_score: int,
    location_score: int,
    pullback_count: int,
) -> bool:
    """
    Post a WAIT — SETUP FORMING message (score approaching threshold).
    Returns True if posted, False if suppressed by cooldown.
    """
    key = (symbol, direction)
    now = datetime.now(timezone.utc)
    last = _last_forming.get(key)
    if last and (now - last) < timedelta(minutes=_FORMING_COOLDOWN):
        return False
    _last_forming[key] = now

    desc = _forming_desc(
        trend_score, pullback_score, confirm_score, location_score,
        direction, pullback_count,
    )
    _post({
        "embeds": [{
            "title": f"👁  WAIT — SETUP FORMING — {symbol} {direction}",
            "description": f"Score: **{score}/100**\n{desc}.",
            "color": 0xF0B429,
            "timestamp": now.isoformat(),
        }]
    })
    return True


def send_alert(
    *,
    symbol: str,
    direction: str,
    score: int,
    price: float,
    stop: float,
    target: float,
    trend_read: str,
    pullback_count: int,
    confirm_vs_avg: float,
    extension_risk: str,
    setup_summary: str,
    ema9: float,
    ema21: float,
    trend_score: int = 0,
    pullback_score: int = 0,
    confirm_score: int = 0,
    location_score: int = 0,
) -> bool:
    """
    Post a BUY / SELL alert (score ≥ SCORE_THRESHOLD). Full detail.
    Returns True if posted, False if suppressed by cooldown.
    """
    key = (symbol, direction)
    now = datetime.now(timezone.utc)
    last = _last_alert.get(key)
    if last and (now - last) < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
        rem = ALERT_COOLDOWN_MINUTES - int((now - last).total_seconds() // 60)
        logger.debug(f"Alert suppressed ({symbol} {direction}) — cooldown {rem}m")
        return False
    _last_alert[key] = now

    action = "BUY" if direction == "LONG" else "SELL"
    emoji = "🟢" if direction == "LONG" else "🔴"
    color = 0x00CC44 if direction == "LONG" else 0xCC2200
    dir_word = "bullish" if direction == "LONG" else "bearish"

    confirm_str = (
        f"{confirm_vs_avg:.1f}× avg body" if confirm_vs_avg >= 1.0 else "confirmed"
    )
    setup_line = (
        f"Trend {dir_word} · {pullback_count}-candle pullback · "
        f"{confirm_str} confirmation closed"
    )

    stop_dist = abs(price - stop)
    tgt_dist = abs(target - price)
    rr = f"1:{tgt_dist / stop_dist:.1f}" if stop_dist > 0 else "n/a"
    levels_line = (
        f"Entry ~{price:.2f}  ·  Stop ~{stop:.2f}  ·  Target ~{target:.2f}  ·  R:R {rr}"
    )

    _post({
        "embeds": [{
            "title": f"{emoji}  {action} — {symbol}",
            "description": (
                f"**Score: {score}/100**\n"
                f"{setup_line}\n"
                f"{levels_line}\n"
                f"**Check chart now.**"
            ),
            "color": color,
            "footer": {
                "text": (
                    f"EMA9 {ema9:.2f} · EMA21 {ema21:.2f} · "
                    f"Extension {extension_risk.upper()} · {setup_summary}"
                )
            },
            "timestamp": now.isoformat(),
        }]
    })
    return True


def send_paper_result(
    *,
    symbol: str,
    direction: str,
    outcome: str,
    points: float,
    duration_minutes: float,
    entry: float,
    exit_price: float,
) -> None:
    icons = {"WIN": "✅", "LOSS": "❌", "TIMEOUT": "⏱️", "AMBIGUOUS": "⚠️"}
    icon = icons.get(outcome, "•")
    sign = "+" if points > 0 else ""
    _post({
        "content": (
            f"{icon} **Mirror paper result:** `{symbol}` {direction} — **{outcome}** "
            f"`{sign}{points:.2f} pts` | "
            f"Entry {entry:.2f} → {exit_price:.2f} | "
            f"{duration_minutes:.0f}m held"
        )
    })
