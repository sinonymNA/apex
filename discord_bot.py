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
NEWS_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s=SPY&region=US&lang=en-US"
)
NEWS_KEYWORDS = {
    "fed", "federal reserve", "iran", "inflation", "jobs", "gdp",
    "s&p 500", "s&p500", "recession", "rally", "crash", "oil",
    "rates", "interest rate", "earnings", "unemployment", "cpi", "pce",
}

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

# ── Module-level state ────────────────────────────────────────────────────────
_bot: "discord.Client | None" = None
_loop: "asyncio.AbstractEventLoop | None" = None
_feed_channel: "discord.TextChannel | None" = None
_talk_channel: "discord.TextChannel | None" = None
_state_getter = None  # set by trader.py so we can read live trading state


def set_state_getter(fn):
    """Called by trader.py on startup to share live _state without circular imports."""
    global _state_getter
    _state_getter = fn


def _build_context() -> str:
    """Build a plain-English context block from live trader state."""
    from datetime import date, timedelta

    if _state_getter is None:
        return ""

    state = _state_getter()
    eval_pnl   = state["current_equity"] - 100_000.0
    daily_pnl  = state["daily_pnl"]
    session_day = state["session_day"]
    streak     = state.get("consecutive_wins", 0)

    # Projected pass date based on current daily pace
    projected_pass = projected_payout = "TBD"
    if session_day > 0 and eval_pnl > 0:
        daily_rate = eval_pnl / session_day
        if daily_rate > 0:
            days_left = max(0, (3000 - eval_pnl) / daily_rate)
            projected_pass   = (date.today() + timedelta(days=days_left)).strftime("%B %d")
            projected_payout = (date.today() + timedelta(days=days_left + 14)).strftime("%B %d")

    today_trades = state.get("daily_trades", [])
    wins = sum(1 for t in today_trades if (t.get("pnl_dollars") or 0) > 0)
    trade_line = f"{len(today_trades)} trades ({wins}W {len(today_trades)-wins}L)" if today_trades else "0 trades"

    pos = state.get("current_position")
    position_line = (
        f"In position: {pos['qty']} contracts @ ${pos['entry']:.2f}"
        if pos else "No open position"
    )

    return (
        f"Current eval P&L: ${eval_pnl:+.2f} / $3,000\n"
        f"Today's P&L: ${daily_pnl:+.2f}\n"
        f"Today's trades: {trade_line}\n"
        f"Win streak: {streak}\n"
        f"Days in eval: {session_day}\n"
        f"Position: {position_line}\n"
        f"Projected pass date: {projected_pass}\n"
        f"Projected first payout: {projected_payout}\n"
        f"Projected $10K/month: October 2026"
    )


async def _respond_to_ethan(user_message: str) -> str:
    """Send Ethan's message to Claude with SABLE personality + live context."""
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return "ANTHROPIC_API_KEY not set — I can't think right now."

    context = _build_context()
    prompt = f"{context}\n\nEthan says: {user_message}" if context else user_message

    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=400,
            system=SABLE_SYSTEM,
            messages=[{"role": "user", "content": prompt}],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"Claude response failed: {e}")
        return "Something went wrong on my end. Check the logs."


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


# ── News feed ─────────────────────────────────────────────────────────────────
async def _analyze_news(headline: str) -> str:
    import anthropic

    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return ""
    try:
        client = anthropic.AsyncAnthropic(api_key=api_key)
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=SABLE_SYSTEM + "\nYou are in analyst mode.",
            messages=[{
                "role": "user",
                "content": (
                    f"News: {headline}\n\n"
                    "Write exactly 3 sentences:\n"
                    "1. What happened in plain English\n"
                    "2. How it affects ES momentum trading today\n"
                    "3. What Ethan should expect from his system\n"
                    "Be specific. End with what to watch."
                ),
            }],
        )
        return msg.content[0].text.strip()
    except Exception as e:
        logger.error(f"News analysis failed: {e}")
        return ""


async def _fetch_and_post_news():
    import httpx
    import pytz
    from xml.etree.ElementTree import fromstring
    from worker.db import get_today_news_count, is_news_url_posted, mark_news_url_posted

    ET_tz = pytz.timezone("America/New_York")
    now_et = __import__("datetime").datetime.now(ET_tz)

    if now_et.weekday() >= 5 or not (9 <= now_et.hour < 17):
        return
    if get_today_news_count() >= 6:
        return
    if _feed_channel is None:
        return

    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(
                NEWS_RSS_URL,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=10.0,
            )
        if r.status_code != 200:
            logger.warning(f"RSS fetch returned {r.status_code}")
            return

        root = fromstring(r.text)
        for item in root.findall(".//item"):
            title_el = item.find("title")
            link_el  = item.find("link")
            if title_el is None or link_el is None:
                continue

            headline = (title_el.text or "").strip()
            url      = (link_el.text or "").strip()
            if not headline or not url:
                continue

            if not any(kw in headline.lower() for kw in NEWS_KEYWORDS):
                continue
            if is_news_url_posted(url):
                continue

            analysis = await _analyze_news(headline)
            ts = now_et.strftime("%-I:%M %p ET")
            body = f"📰 **{headline}**"
            if analysis:
                body += f"\n{analysis}"
            body += f"\n_{ts}_"

            await _feed_channel.send(body)
            mark_news_url_posted(url, headline)
            break  # one post per 20-min check

    except Exception as e:
        logger.error(f"News fetch/post failed: {e}")


async def _news_loop():
    await _bot.wait_until_ready()
    while not _bot.is_closed():
        await _fetch_and_post_news()
        await asyncio.sleep(20 * 60)


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

        asyncio.create_task(_news_loop())

    @_bot.event
    async def on_message(message):
        if message.author.bot:
            return
        if _talk_channel is None or message.channel.id != _talk_channel.id:
            return

        async with message.channel.typing():
            reply = await _respond_to_ethan(message.content)

        await message.channel.send(reply)

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
