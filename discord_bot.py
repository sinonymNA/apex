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
# Eval P&L threshold (dollars) → (db key, message)
# None threshold = triggered externally via post_milestone()
MILESTONES = {
    "eval_25":   (750,  "📈 **25% there.** Keep going."),
    "eval_50":   (1500, "🔥 **Halfway to funded.**"),
    "eval_75":   (2250, "💪 **75%. Almost there.**"),
    "eval_100":  (3000, "🎉 **EVAL PASSED.**\n Account activating.\n First payout incoming."),
    "funded":    (None, "✅ **FUNDED.** Real money mode."),
    "payout":    (None, "💰 **FIRST DOLLAR.**\n This is real. This is working."),
    "10k_month": (None, "🚀 **TEN THOUSAND.**\n Alex doesn't know yet."),
    "all_5":     (None, "👑 **FULL STACK.**\n $13,500/month. October arrived."),
}

NEWS_RSS_URL = (
    "https://feeds.finance.yahoo.com/rss/2.0/headline"
    "?s=SPY&region=US&lang=en-US"
)
NEWS_KEYWORDS = {
    "fed", "federal reserve", "iran", "inflation", "jobs", "gdp",
    "s&p 500", "s&p500", "recession", "rally", "crash", "oil",
    "rates", "interest rate", "earnings", "unemployment", "cpi", "pce",
}

SABLE_SYSTEM = """You are SABLE — Ethan Sinon's AI trading assistant.

His situation:
- Running a Tradeify $50K Select eval account ($3,000 profit target, $2,500 trailing DD limit)
- Trading MES (Micro E-mini S&P 500) via TradersPost → Tradovate
- Strategy: VWAP pullback + Opening Range Breakout, 6 MES contracts max
- Goal: pass the eval and reach funded status

Your job:
- Give him real-time trade updates, P&L status, and risk alerts
- Answer questions about the eval, trade setup, and bot behavior
- Be direct and concise — he's watching live trades

Your personality:
- Practical and fast. He wants info, not fluff.
- Keep responses under 200 words.
- Talk like a sharp friend who knows trading."""

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
    """Build a plain-English context block from live trader state + DB diagnostics."""
    from datetime import date, timedelta
    from worker.db import get_today_near_misses, get_risk_log, get_recent_trades

    if _state_getter is None:
        return ""

    state = _state_getter()
    eval_pnl    = state["current_equity"] - 50_000.0
    daily_pnl   = state["daily_pnl"]
    session_day = state["session_day"]
    streak      = state.get("consecutive_wins", 0)

    # Projected pass date based on current daily pace
    projected_pass = projected_payout = "TBD"
    if session_day > 0 and eval_pnl > 0:
        daily_rate = eval_pnl / session_day
        if daily_rate > 0:
            days_left = max(0, (3000 - eval_pnl) / daily_rate)
            projected_pass   = (date.today() + timedelta(days=days_left)).strftime("%B %d")
            projected_payout = (date.today() + timedelta(days=days_left + 14)).strftime("%B %d")

    today_trades = state.get("daily_trades", [])
    wins  = sum(1 for t in today_trades if (t.get("pnl_dollars") or 0) > 0)
    trade_line = f"{len(today_trades)} trades ({wins}W {len(today_trades)-wins}L)" if today_trades else "0 trades"

    pos = state.get("current_position")
    position_line = (
        f"In position: {pos['qty']} contracts @ ${pos['entry']:.2f}"
        if pos else "No open position"
    )

    # Near-miss summary — why did bars not trigger today?
    near_miss_summary = "No bar data yet"
    try:
        nm_rows = get_today_near_misses()
        if nm_rows:
            from collections import Counter
            reasons = Counter(r.get("blocked_reason", "unknown") for r in nm_rows)
            near_miss_summary = (
                f"{len(nm_rows)} bars evaluated — "
                + ", ".join(f"{v}x {k}" for k, v in reasons.most_common())
            )
            last = nm_rows[0]
            if last.get("close") and last.get("breakout_level"):
                pct = last.get("percent_to_breakout", 0) or 0
                near_miss_summary += f"\nLast bar: SPY ${last['close']:.2f}, {abs(pct):.2f}% {'below' if pct < 0 else 'above'} breakout"
    except Exception:
        pass

    # Recent trades (last 5)
    recent_trade_lines = "None yet"
    try:
        recent = get_recent_trades(n=5)
        if recent:
            recent_trade_lines = "\n".join(
                f"  {t.get('exit_reason','?')} ${t.get('pnl_dollars',0):+.2f} @ {t.get('exit_price','?')}"
                for t in recent
            )
    except Exception:
        pass

    return (
        f"Current eval P&L: ${eval_pnl:+.2f} / $3,000\n"
        f"Today's P&L: ${daily_pnl:+.2f}\n"
        f"Today's trades: {trade_line}\n"
        f"Win streak: {streak}\n"
        f"Days in eval: {session_day}\n"
        f"Position: {position_line}\n"
        f"Projected pass date: {projected_pass}\n"
        f"Projected first payout: {projected_payout}\n"
        f"Projected $10K/month: October 2026\n"
        f"\nToday's signal activity:\n{near_miss_summary}\n"
        f"\nRecent trades:\n{recent_trade_lines}"
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
def post_trade_entry(
    price: float,
    stop: float,
    target: float,
    running_pnl: float,
    direction: str = "LONG",
    grade: str = "A",
    strategy: str = "",
    contracts: int = 1,
):
    icon  = "🟢" if direction == "LONG" else "🔴"
    side  = "LONG" if direction == "LONG" else "SHORT"
    risk  = abs(price - stop) * 10 * 5 * contracts  # SPY×10 × $5/pt × contracts
    strat = f" | {strategy}" if strategy else ""
    _post_feed(
        f"{icon} **{side} ENTRY** — {grade} grade{strat}\n"
        f" Price: ${price:.2f} | {contracts} MES\n"
        f" Stop: ${stop:.2f} | Target: ${target:.2f}\n"
        f" Risk: ${risk:.0f} | Eval: ${running_pnl:+,.0f} / $3,000"
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


def post_morning_brief(session_day: int, regime: str, spy_price: float,
                       breakout_level: float, atr: float):
    pct_to_breakout = ((breakout_level - spy_price) / spy_price) * 100
    _post_feed(
        f"🌅 **MORNING BRIEF — Day {session_day}**\n"
        f" SPY: ${spy_price:.2f}\n"
        f" Breakout level: ${breakout_level:.2f} (+{pct_to_breakout:.1f}%)\n"
        f" ATR: ${atr:.2f}\n"
        f" Regime: {regime}\n"
        f" Entry window opens 9:30 AM ET"
    )


def post_noon_update(session_day: int, daily_pnl: float, trade_count: int,
                     regime: str, spy_price: float, in_position: bool):
    position_line = "📍 In position" if in_position else "⏳ No position"
    _post_feed(
        f"☀️ **NOON UPDATE — Day {session_day}**\n"
        f" SPY: ${spy_price:.2f}\n"
        f" P&L today: ${daily_pnl:+.2f}\n"
        f" Trades: {trade_count}\n"
        f" Regime: {regime}\n"
        f" {position_line}"
    )


def check_milestones(eval_pnl: float):
    """Called after every trade close — fires any newly crossed eval milestones."""
    from worker.db import is_milestone_fired, mark_milestone_fired
    for key, (threshold, message) in MILESTONES.items():
        if threshold is None:
            continue
        if eval_pnl >= threshold and not is_milestone_fired(key):
            mark_milestone_fired(key)
            _post_feed(message)


def post_milestone(key: str):
    """Manually trigger a non-automatic milestone (funded, payout, 10k, all_5)."""
    from worker.db import is_milestone_fired, mark_milestone_fired
    entry = MILESTONES.get(key)
    if entry is None:
        logger.warning(f"Unknown milestone key: {key}")
        return
    _, message = entry
    if not is_milestone_fired(key):
        mark_milestone_fired(key)
    _post_feed(message)


def post_bihourly_update(
    label: str,
    daily_pnl: float,
    trade_count: int,
    regime: str,
    spy_price: float,
    in_position: bool,
    recent_trades: list,
    near_misses: list,
    assessment: str = "",
):
    """Bi-hourly Discord pulse: what happened in the last 2 hours and why."""
    # ── position line ─────────────────────────────────────────────────────────
    pos_icon = "📍 In position" if in_position else "⏳ Flat"
    pnl_icon = "▲" if daily_pnl >= 0 else "▼"

    # ── trades this window ────────────────────────────────────────────────────
    if recent_trades:
        trade_lines = []
        for t in recent_trades:
            pnl  = t.get("pnl_dollars") or t.get("pnl") or 0.0
            icon = "✅" if pnl > 0 else "❌"
            d    = t.get("direction", t.get("dir", "?"))
            gr   = t.get("grade", "?")
            px_e = t.get("entry_price") or t.get("entry") or 0.0
            px_x = t.get("exit_price")  or t.get("exit")  or 0.0
            rsn  = t.get("exit_reason") or t.get("reason") or ""
            tm   = t.get("exit_time")   or t.get("time")   or ""
            if hasattr(tm, "strftime"):
                tm = tm.strftime("%H:%M")
            trade_lines.append(
                f"  {icon} {d} {gr} ${px_e:.2f}→${px_x:.2f}  {pnl:+.0f}  {rsn}  {tm}"
            )
        trades_section = "**Trades this window:**\n" + "\n".join(trade_lines)
    else:
        trades_section = "**Trades this window:** none"

    # ── near-misses / no-trade reasons ───────────────────────────────────────
    reason_labels = {
        "risk_blocked":   "risk gate",
        "chop_blocked":   "choppy market",
        "volume_blocked": "low volume",
        "grade_blocked":  "below grade",
        "regime_blocked": "wrong regime",
        "no_signal":      "no setup",
    }
    if near_misses:
        seen: dict = {}
        for nm in near_misses:
            r = nm.get("blocked_reason") or "no_signal"
            seen[r] = seen.get(r, 0) + 1
        nm_parts = [f"{reason_labels.get(r, r)} ×{n}" for r, n in seen.items()]
        nm_section = "**Skipped because:** " + ", ".join(nm_parts)
    else:
        nm_section = "**Skipped because:** no setups scanned"

    body = (
        f"🕐 **{label} PULSE**\n"
        f" SPY: ${spy_price:.2f} | Regime: {regime}\n"
        f" P&L today: ${daily_pnl:+,.0f}  {pnl_icon} | Trades: {trade_count} | {pos_icon}\n"
        f"\n{trades_section}\n"
        f"{nm_section}"
    )
    if assessment:
        body += f"\n\n_{assessment}_"

    _post_feed(body)


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
            system=SABLE_SYSTEM,
            messages=[{
                "role": "user",
                "content": (
                    f"News: {headline}\n\n"
                    "Explain this in 3 sentences:\n"
                    "1. What happened in plain English\n"
                    "2. Why it matters (market-related or not)\n"
                    "3. One thing to keep an eye on\n"
                    "Be direct. No jargon."
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
