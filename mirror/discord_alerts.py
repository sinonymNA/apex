"""
mirror/discord_alerts.py — Discord webhook sender for Mirror Agent.

Uses DISCORD_MIRROR_WEBHOOK_URL (falls back to DISCORD_WEBHOOK_URL).
Enforces per-symbol-per-direction cooldown (default 10 minutes).

This is a simple synchronous httpx poster — does NOT require a bot token
and does NOT depend on the main discord_bot.py.
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

# cooldown tracker: (symbol, direction) → last alert UTC datetime
_last_alert: dict[tuple, datetime] = {}


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


def send_startup(symbols: list[str]) -> None:
    """Send startup notification — called once when the agent comes online."""
    _post({
        "content": (
            "🟢 **Mirror Agent online.**\n"
            f"Mode: **ALERT-ONLY** — No live trades enabled.\n"
            f"Watching: `{'`, `'.join(symbols)}`\n"
            "Paper simulation active. Alerts fire when Mirror System v2 score ≥ threshold."
        )
    })


def send_error(msg: str) -> None:
    """Send an error/warning to Discord."""
    _post({"content": f"🔴 **Mirror Agent error:**\n{msg}"})


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
) -> bool:
    """
    Send a setup alert embed.
    Returns True if sent, False if suppressed by cooldown.
    """
    key = (symbol, direction)
    now = datetime.now(timezone.utc)
    last = _last_alert.get(key)
    if last and (now - last) < timedelta(minutes=ALERT_COOLDOWN_MINUTES):
        rem = ALERT_COOLDOWN_MINUTES - int((now - last).total_seconds() // 60)
        logger.debug(f"Alert suppressed ({symbol} {direction}) — cooldown {rem}m")
        return False

    _last_alert[key] = now

    stop_dist = abs(price - stop)
    tgt_dist = abs(target - price)
    rr = f"1:{tgt_dist / stop_dist:.1f}" if stop_dist > 0 else "n/a"
    emoji = "🟢📈" if direction == "LONG" else "🔴📉"

    _post({
        "embeds": [{
            "title": f"{emoji}  Mirror System v2 — {symbol} {direction}",
            "color": 0x00CC44 if direction == "LONG" else 0xCC2200,
            "fields": [
                {"name": "Score",              "value": f"**{score}/100**",        "inline": True},
                {"name": "Direction",          "value": direction,                 "inline": True},
                {"name": "Entry (now)",        "value": f"{price:.2f}",            "inline": True},
                {"name": "Planned Stop",       "value": f"{stop:.2f}",             "inline": True},
                {"name": "Planned Target",     "value": f"{target:.2f}",           "inline": True},
                {"name": "R:R",                "value": rr,                        "inline": True},
                {"name": "EMA9 / EMA21",       "value": f"{ema9:.2f} / {ema21:.2f}", "inline": True},
                {"name": "Extension Risk",     "value": extension_risk.upper(),    "inline": True},
                {"name": "Pullback Candles",   "value": str(pullback_count),       "inline": True},
                {"name": "Confirm Body",       "value": f"{confirm_vs_avg:.1f}× avg PB body", "inline": True},
                {"name": "Trend Read",         "value": trend_read,                "inline": False},
                {"name": "Score Breakdown",    "value": setup_summary,             "inline": False},
            ],
            "footer": {"text": "⚠️  ALERT ONLY — check chart manually before entering."},
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
    """Send a paper simulation result."""
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
