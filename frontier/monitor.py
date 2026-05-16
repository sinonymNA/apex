"""
frontier/monitor.py — Frontier Airlines Go Wild deal detector.

Three parallel monitors, all independent — if one source fails the others
continue running:

  1. Presale URL probe  — polls /gowildpresale every 5 min.
     HTTP 200 + GoWild keywords in body = sale just went live.
     Redirects or 404 = not active. 403 = can't determine, skip.

  2. Google News RSS    — polls a "frontier airlines go wild" news search
     every 10 min. Catches press releases and news articles from all
     sources within minutes of publication.

  3. PR Newswire RSS    — polls Frontier's official press release feed
     every 15 min as a second opinion. Seeds seen GUIDs on first run so
     old articles never trigger alerts.

All alerts go to the existing Discord webhook (DISCORD_MIRROR_WEBHOOK_URL
or DISCORD_WEBHOOK_URL). No new infrastructure needed.
"""
from __future__ import annotations

import asyncio
import os
import threading
import xml.etree.ElementTree as ET
from datetime import datetime, timezone

import httpx
from loguru import logger

# ── Config ─────────────────────────────────────────────────────────────────────

PRESALE_URL = "https://www.flyfrontier.com/gowildpresale"
DEALS_URL   = "https://www.flyfrontier.com/deals/gowild-pass/"

_GOOGLE_NEWS_RSS = (
    "https://news.google.com/rss/search"
    "?q=frontier+airlines+%22go+wild%22&hl=en-US&gl=US&ceid=US:en"
)
_PRNEWSWIRE_RSS = (
    "https://www.prnewswire.com/rss/news-releases-list.rss"
    "?company=frontier-airlines"
)

PRESALE_POLL  = int(os.getenv("FRONTIER_PRESALE_POLL",  "300"))   # 5 min
NEWS_POLL     = int(os.getenv("FRONTIER_NEWS_POLL",     "600"))   # 10 min
PR_POLL       = int(os.getenv("FRONTIER_PR_POLL",       "900"))   # 15 min

GOWILD_KEYWORDS = [
    "go wild", "gowild", "unlimited flights",
    "all-you-can-fly", "all you can fly",
]

_WEBHOOK_URL: str = (
    os.getenv("DISCORD_MIRROR_WEBHOOK_URL")
    or os.getenv("DISCORD_WEBHOOK_URL")
    or ""
)

# Browser-like headers to pass basic bot filters
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}


# ── Discord poster ─────────────────────────────────────────────────────────────

def _post_discord(content: str) -> None:
    if not _WEBHOOK_URL:
        logger.warning("Frontier monitor: no Discord webhook configured")
        return
    try:
        with httpx.Client(timeout=8.0) as client:
            r = client.post(_WEBHOOK_URL, json={"content": content})
            if r.status_code not in (200, 204):
                logger.warning(f"Discord webhook {r.status_code}: {r.text[:100]}")
    except Exception as e:
        logger.error(f"Frontier monitor Discord post failed: {e}")


# ── RSS parser ─────────────────────────────────────────────────────────────────

def _parse_rss(xml_text: str) -> list[dict]:
    items: list[dict] = []
    try:
        root = ET.fromstring(xml_text)
        channel = root.find("channel")
        if channel is None:
            return items
        for item in channel.findall("item"):
            def _text(tag: str) -> str:
                el = item.find(tag)
                return (el.text or "").strip() if el is not None else ""
            items.append({
                "guid":        _text("guid") or _text("link"),
                "title":       _text("title"),
                "link":        _text("link"),
                "description": _text("description"),
                "pubDate":     _text("pubDate"),
            })
    except ET.ParseError as e:
        logger.warning(f"RSS parse error: {e}")
    return items


def _has_gowild(item: dict) -> bool:
    text = (item["title"] + " " + item["description"]).lower()
    return any(kw in text for kw in GOWILD_KEYWORDS)


# ── Monitor 1: Presale URL probe ───────────────────────────────────────────────

class PresaleProbe:
    def __init__(self) -> None:
        self._was_live = False

    async def check(self) -> bool:
        try:
            async with httpx.AsyncClient(
                follow_redirects=False, timeout=15.0, headers=_HEADERS
            ) as client:
                r = await client.get(PRESALE_URL)

            if r.status_code == 200:
                body = r.text.lower()
                if any(kw in body for kw in GOWILD_KEYWORDS):
                    if not self._was_live:
                        self._was_live = True
                        return True   # newly live
                else:
                    self._was_live = False
            elif r.status_code in (301, 302, 303, 307, 308, 404):
                if self._was_live:
                    logger.info("Frontier presale URL is no longer active")
                self._was_live = False
            # 403 = blocked, can't determine — leave state unchanged

        except Exception as e:
            logger.debug(f"Frontier presale probe error: {e}")
        return False

    async def run_forever(self) -> None:
        logger.info(f"Frontier: presale probe started (every {PRESALE_POLL}s)")
        while True:
            if await self.check():
                _post_discord(
                    "🚨 @everyone **FRONTIER GO WILD DEAL IS LIVE RIGHT NOW!**\n"
                    f"The presale page just went active.\n"
                    f"➜ <{PRESALE_URL}>\n"
                    f"➜ <{DEALS_URL}>\n"
                    "**These sell out fast — check now.**"
                )
                logger.info("Frontier: presale LIVE alert sent to Discord")
            await asyncio.sleep(PRESALE_POLL)


# ── Monitor 2: Google News RSS ─────────────────────────────────────────────────

class GoogleNewsMonitor:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._seeded = False

    async def check(self) -> list[dict]:
        new_items: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
                r = await client.get(_GOOGLE_NEWS_RSS)
            if r.status_code != 200:
                logger.debug(f"Google News RSS returned {r.status_code}")
                return []

            items = _parse_rss(r.text)
            for item in items:
                guid = item["guid"]
                if not guid:
                    continue
                if guid not in self._seen:
                    self._seen.add(guid)
                    if self._seeded and _has_gowild(item):
                        new_items.append(item)

            if not self._seeded:
                logger.info(f"Frontier Google News RSS seeded ({len(self._seen)} items)")
                self._seeded = True

        except Exception as e:
            logger.debug(f"Frontier Google News RSS error: {e}")
        return new_items

    async def run_forever(self) -> None:
        logger.info(f"Frontier: Google News RSS monitor started (every {NEWS_POLL}s)")
        while True:
            for item in await self.check():
                _post_discord(
                    f"📰 **Frontier Go Wild — news alert:**\n"
                    f"**{item['title']}**\n"
                    f"<{item['link']}>\n"
                    f"_{item['pubDate']}_"
                )
                logger.info(f"Frontier Google News alert: {item['title']}")
            await asyncio.sleep(NEWS_POLL)


# ── Monitor 3: PR Newswire RSS ─────────────────────────────────────────────────

class PRNewswireMonitor:
    def __init__(self) -> None:
        self._seen: set[str] = set()
        self._seeded = False

    async def check(self) -> list[dict]:
        new_items: list[dict] = []
        try:
            async with httpx.AsyncClient(timeout=20.0, headers=_HEADERS) as client:
                r = await client.get(_PRNEWSWIRE_RSS)
            if r.status_code != 200:
                logger.debug(f"PR Newswire RSS returned {r.status_code}")
                return []

            items = _parse_rss(r.text)
            for item in items:
                guid = item["guid"]
                if not guid:
                    continue
                if guid not in self._seen:
                    self._seen.add(guid)
                    if self._seeded and _has_gowild(item):
                        new_items.append(item)

            if not self._seeded:
                logger.info(f"Frontier PR Newswire RSS seeded ({len(self._seen)} items)")
                self._seeded = True

        except Exception as e:
            logger.debug(f"Frontier PR Newswire RSS error: {e}")
        return new_items

    async def run_forever(self) -> None:
        logger.info(f"Frontier: PR Newswire RSS monitor started (every {PR_POLL}s)")
        while True:
            for item in await self.check():
                _post_discord(
                    f"📢 **Frontier Go Wild — official press release:**\n"
                    f"**{item['title']}**\n"
                    f"<{item['link']}>\n"
                    f"_{item['pubDate']}_"
                )
                logger.info(f"Frontier PR Newswire alert: {item['title']}")
            await asyncio.sleep(PR_POLL)


# ── Entry point ────────────────────────────────────────────────────────────────

async def _run_all() -> None:
    await asyncio.gather(
        PresaleProbe().run_forever(),
        GoogleNewsMonitor().run_forever(),
        PRNewswireMonitor().run_forever(),
    )


def start_background() -> None:
    """Start all three monitors in a daemon thread with its own event loop."""
    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(_run_all())
        except Exception as e:
            logger.error(f"Frontier monitor crashed: {e}")
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True, name="frontier-monitor")
    t.start()
    logger.info("Frontier Go Wild monitor started in background")
