"""
frontier/destinations.py — User destination watchlist for deal filtering.

Stores destinations as IATA airport codes in frontier/watchlist.json.
Uses Claude Haiku to convert natural language ("Huntington Beach", "near Chicago")
into nearby airport codes. When a watchlist is set, deal alerts are filtered
to only show deals to/from watched airports.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import anthropic
from loguru import logger

_WATCHLIST_PATH = Path(__file__).parent / "watchlist.json"
_client: Optional[anthropic.Anthropic] = None


def _claude() -> anthropic.Anthropic:
    global _client
    if _client is None:
        _client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY", ""))
    return _client


def _load() -> dict:
    if _WATCHLIST_PATH.exists():
        try:
            return json.loads(_WATCHLIST_PATH.read_text())
        except Exception:
            pass
    return {"destinations": [], "airports": []}


def _save(data: dict) -> None:
    _WATCHLIST_PATH.parent.mkdir(parents=True, exist_ok=True)
    _WATCHLIST_PATH.write_text(json.dumps(data, indent=2))


def get_watched_airports() -> list[str]:
    return _load().get("airports", [])


def get_watched_destinations() -> list[str]:
    return _load().get("destinations", [])


def lookup_airports(destination: str) -> list[str]:
    """
    Ask Claude Haiku for IATA codes of all commercial airports within ~100 miles
    of the given destination string. Returns list like ["LAX", "SNA", "LGB"].
    """
    try:
        resp = _claude().messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=150,
            messages=[{
                "role": "user",
                "content": (
                    f"List IATA codes for all commercial airports within 100 miles of "
                    f"'{destination}', USA. Include every airport people realistically "
                    f"fly from (regional and major). "
                    f"Return ONLY a JSON array, e.g. [\"LAX\",\"SNA\",\"LGB\"]. "
                    f"No explanation."
                ),
            }],
        )
        text = resp.content[0].text.strip()
        start, end = text.find("["), text.rfind("]") + 1
        if start >= 0 and end > start:
            codes = json.loads(text[start:end])
            return [c.upper().strip() for c in codes if isinstance(c, str) and 2 <= len(c) <= 4]
    except Exception as e:
        logger.error(f"Airport lookup failed for '{destination}': {e}")
    return []


def add_destination(destination_text: str) -> tuple[list[str], list[str]]:
    """
    Add a destination to the watchlist.
    Returns (airports_for_this_destination, full_airport_list).
    """
    airports = lookup_airports(destination_text)
    if not airports:
        return [], get_watched_airports()

    data = _load()
    if destination_text not in data["destinations"]:
        data["destinations"].append(destination_text)

    existing = set(data["airports"])
    data["airports"] = sorted(existing | set(airports))
    _save(data)

    logger.info(f"Watchlist: '{destination_text}' → {airports}")
    return airports, data["airports"]


def clear() -> None:
    _save({"destinations": [], "airports": []})
    logger.info("Watchlist cleared")


def matches_watchlist(title: str, body: str = "") -> bool:
    """
    Return True if the deal text mentions any watched airport code.
    If the watchlist is empty, returns True (show all deals).
    """
    airports = get_watched_airports()
    if not airports:
        return True
    text = (title + " " + body).upper()
    return any(code in text for code in airports)
