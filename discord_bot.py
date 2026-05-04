"""
discord_bot.py — SABLE Discord bot.

Runs as a daemon thread alongside the trading worker. Exposes synchronous
post_*() helpers that trader.py calls directly; internally they bridge to
the bot's asyncio event loop via run_coroutine_threadsafe() so the sync
scheduler thread never blocks.
"""
import asyncio
import os
import threading

import discord
from loguru import logger

# ── SABLE personality (injected into every Claude prompt) ─────────────────────
SABLE_SYSTEM = """You are SABLE — Ethan Sinon's personal trading assistant, market analyst, and hype man.

His situation:
- Built automated ES futures trading system
- Tradeify SELECT eval: need $3,000 profit
- Win rate: 50%, EV: +$162.50/trade
- Goal: 5 funded accounts = $13,500/month by October 2026
- Propose to Alex: March 2027
- Buy house: Summer 2027
- $54K/month by March 2028

Your personality:
- Honest. Never lie about risks.
- Calm when he's anxious.
- Fired up when he needs hype.
- Always connect back to the big picture.
- Keep responses under 300 words.
- Talk like a trusted friend who knows markets, not a robot."""

# ── Module-level state (set once bot connects) ────────────────────────────────
_bot: "discord.Client | None" = None
_loop: "asyncio.AbstractEventLoop | None" = None
_feed_channel: "discord.TextChannel | None" = None
_talk_channel: "discord.TextChannel | None" = None


# ── Internal helper ───────────────────────────────────────────────────────────
def _post_feed(message: str):
    """Post to #sable-feed from any thread, non-blocking."""
    if _loop is None or _feed_channel is None:
        logger.warning(f"Discord not ready — dropped: {message[:80]}")
        return
    asyncio.run_coroutine_threadsafe(_feed_channel.send(message), _loop)


# ── Public API (called from trader.py) ───────────────────────────────────────
def post_trade_entry(price: float, stop: float, target: float, running_pnl: float):
    _post_feed(
        f"🟢 **ENTRY** ESM2026 Long\n"
        f" Price: ${price:.2f}\n"
        f" Stop: ${stop:.2f} | Target: ${target:.2f}\n"
        f" Eval: ${running_pnl:.2f} / $3,000"
    )


def post_trade_exit(
    pnl: float,
    entry: float,
    exit_price: float,
    duration: str,
    running_pnl: float,
    reason: str,
    streak: int,
):
    if pnl >= 0:
        pct = (running_pnl / 3000) * 100
        _post_feed(
            f"✅ **WIN** +${pnl:.2f}\n"
            f" ${entry:.2f} → ${exit_price:.2f} ({duration})\n"
            f" Eval: ${running_pnl:.2f} / $3,000 ({pct:.1f}%)\n"
            f" Streak: {streak} wins 🔥"
        )
    else:
        buffer = max(0.0, 3000 + running_pnl) if running_pnl < 0 else 3000.0
        _post_feed(
            f"❌ **LOSS** -${abs(pnl):.2f}\n"
            f" ${entry:.2f} → ${exit_price:.2f}\n"
            f" Reason: {reason}\n"
            f" Eval: ${running_pnl:.2f} / $3,000\n"
            f" Buffer: ${buffer:.2f} remaining"
        )


def post_kill_switch(daily_loss: float, buffer: float):
    _post_feed(
        f"⚠️ **KILL SWITCH**\n"
        f" 3 consecutive losses. System offline until tomorrow.\n"
        f" Today: -${abs(daily_loss):.2f}\n"
        f" Buffer: ${buffer:.2f} remaining\n"
        f" This is the system protecting you."
    )


def post_daily_summary(
    day_n: int,
    trades: int,
    wins: int,
    losses: int,
    daily_pnl: float,
    total_pnl: float,
    buffer: float,
    assessment: str,
):
    pct = (total_pnl / 3000) * 100
    _post_feed(
        f"📊 **DAY {day_n} SUMMARY**\n"
        f" Trades: {trades} | {wins}W {losses}L\n"
        f" Today: ${daily_pnl:+.2f}\n"
        f" Eval total: ${total_pnl:.2f} / $3,000 ({pct:.1f}%)\n"
        f" Drawdown buffer: ${buffer:.2f}\n\n"
        f" {assessment}"
    )


# ── Bot internals ─────────────────────────────────────────────────────────────
async def _run_bot():
    global _bot, _feed_channel, _talk_channel

    intents = discord.Intents.default()
    intents.message_content = True
    _bot = discord.Client(intents=intents)

    @_bot.event
    async def on_ready():
        global _feed_channel, _talk_channel

        feed_id = int(os.getenv("DISCORD_FEED_CHANNEL_ID", "0"))
        talk_id = int(os.getenv("DISCORD_TALK_CHANNEL_ID", "0"))

        if feed_id:
            try:
                _feed_channel = await _bot.fetch_channel(feed_id)
            except Exception as e:
                logger.warning(f"Could not fetch #sable-feed (id={feed_id}): {e}")
        else:
            logger.warning("DISCORD_FEED_CHANNEL_ID not set")

        if talk_id:
            try:
                _talk_channel = await _bot.fetch_channel(talk_id)
            except Exception as e:
                logger.warning(f"Could not fetch #talk-to-sable (id={talk_id}): {e}")
        else:
            logger.warning("DISCORD_TALK_CHANNEL_ID not set")

        logger.info(f"SABLE Discord bot online as {_bot.user}")

        if _feed_channel:
            await _feed_channel.send(
                "🤖 **SABLE online.** Systems connected. Ready to trade."
            )

    token = os.getenv("DISCORD_BOT_TOKEN", "")
    if not token:
        logger.warning("DISCORD_BOT_TOKEN not set — Discord bot disabled")
        return

    await _bot.start(token)


def start_bot():
    """Launch the Discord bot in a dedicated daemon thread with its own event loop."""
    global _loop

    def _thread_main():
        global _loop
        _loop = asyncio.new_event_loop()
        asyncio.set_event_loop(_loop)
        try:
            _loop.run_until_complete(_run_bot())
        except Exception as e:
            logger.error(f"Discord bot crashed: {e}")

    threading.Thread(target=_thread_main, daemon=True, name="discord-bot").start()
    logger.info("Discord bot thread started")
