"""
worker/news.py — Fetch and analyse market news headlines for Sable Stocks emails.

Primary:  NewsAPI (requires NEWSAPI_KEY env var)
Fallback: MarketWatch RSS (no API key required)
Analysis: Claude claude-sonnet-4-6 (requires ANTHROPIC_API_KEY)
"""
import os
import xml.etree.ElementTree as ET
from datetime import date

import requests
from loguru import logger

_NEWSAPI_URL = "https://newsapi.org/v2/everything"
_RSS_FEEDS = [
    "https://feeds.marketwatch.com/marketwatch/topstories/",
    "https://feeds.finance.yahoo.com/rss/2.0/headline?s=SPY&region=US&lang=en-US",
]


def fetch_news(max_items: int = 5) -> list:
    """
    Fetch top market news headlines.
    Returns list of {title, source, url} dicts. Empty list on complete failure.
    """
    newsapi_key = os.getenv("NEWSAPI_KEY", "")
    if newsapi_key:
        result = _fetch_newsapi(newsapi_key, max_items)
        if result:
            return result
    return _fetch_rss(max_items)


def _fetch_newsapi(api_key: str, max_items: int) -> list:
    try:
        resp = requests.get(
            _NEWSAPI_URL,
            params={
                "q": 'SPY OR "S&P 500" OR "stock market" OR "Federal Reserve"',
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": max_items,
                "apiKey": api_key,
            },
            timeout=10,
        )
        if resp.status_code != 200:
            logger.warning(f"NewsAPI returned {resp.status_code}: {resp.text[:100]}")
            return []
        articles = resp.json().get("articles", [])
        return [
            {
                "title":  a.get("title", "").split(" - ")[0].strip(),
                "source": a.get("source", {}).get("name", "Unknown"),
                "url":    a.get("url", ""),
            }
            for a in articles[:max_items]
            if a.get("title") and "[Removed]" not in a.get("title", "")
        ]
    except Exception as e:
        logger.warning(f"NewsAPI fetch failed (non-fatal): {e}")
        return []


def _fetch_rss(max_items: int) -> list:
    for feed_url in _RSS_FEEDS:
        try:
            resp = requests.get(
                feed_url, timeout=10,
                headers={"User-Agent": "Mozilla/5.0 (compatible; SableStocks/1.0)"},
            )
            if resp.status_code != 200:
                continue
            root = ET.fromstring(resp.content)
            items = root.findall("./channel/item")
            result = []
            for item in items[:max_items]:
                title = (item.findtext("title") or "").strip()
                link  = (item.findtext("link") or "").strip()
                if title:
                    result.append({"title": title, "source": "MarketWatch", "url": link})
            if result:
                return result
        except Exception as e:
            logger.warning(f"RSS fetch from {feed_url} failed (non-fatal): {e}")
    return []


def analyze_with_claude(headlines: list) -> str:
    """
    Return a 2-3 sentence trading-relevant analysis of the headlines.
    Returns empty string if no API key or call fails.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key or not headlines:
        return ""
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        lines = "\n".join(f"- {h['title']} ({h['source']})" for h in headlines)
        today = date.today().strftime("%B %-d, %Y")
        prompt = (
            f"Today is {today}. You are a concise market analyst for an automated SPY "
            f"day-trading system that uses momentum breakout signals.\n"
            f"Here are today's top market headlines:\n{lines}\n\n"
            f"In 2-3 sentences, explain what these headlines mean for SPY price action and "
            f"breakout probability today. Be specific about bullish/bearish bias if clear. "
            f"Plain prose only — no bullet points, no headers."
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text.strip()
    except Exception as e:
        logger.warning(f"Claude news analysis failed (non-fatal): {e}")
        return ""
