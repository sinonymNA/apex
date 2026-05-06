"""
mirror/agent.py — Mirror Agent v1 main module.

Connects to Tradovate market data WebSocket, buffers 1-minute candles,
runs MirrorStrategy scoring on each closed bar, fires Discord alerts,
and tracks paper simulation results.

ALERT-ONLY: No live trades, no orders, no broker interaction whatsoever.
Data sources (in priority order):
  1. Tradovate WebSocket — real CME futures bars (requires registered API app + CID/SEC)
  2. Alpaca 1-min SPY bars — free proxy via existing Alpaca keys (auto-used when
     TRADOVATE_USERNAME is blank or Tradovate auth fails permanently)
"""
from __future__ import annotations

import asyncio
import json
import os
import threading
import time as _time_module
from datetime import datetime, timedelta, timezone
from typing import Callable, Optional

import httpx
import websockets
from loguru import logger

from mirror.discord_alerts import (
    send_alert,
    send_error,
    send_paper_result,
    send_startup,
)
from mirror.logger import log_alert, log_error
from mirror.paper_sim import PaperSim
from mirror.strategy import MirrorStrategy, SCORE_THRESHOLD

# ── Config ─────────────────────────────────────────────────────────────────────

TRADOVATE_ENV: str = os.getenv("TRADOVATE_ENV", "demo")
TRADOVATE_USERNAME: str = os.getenv("TRADOVATE_USERNAME", "")
TRADOVATE_PASSWORD: str = os.getenv("TRADOVATE_PASSWORD", "")
TRADOVATE_APP_ID: str = os.getenv("TRADOVATE_APP_ID", "Mirror-Agent")
TRADOVATE_APP_VERSION: str = os.getenv("TRADOVATE_APP_VERSION", "1.0")
TRADOVATE_CID: str = os.getenv("TRADOVATE_CID", "")
TRADOVATE_SEC: str = os.getenv("TRADOVATE_SEC", "")

SYMBOLS: list[str] = [
    s.strip().upper()
    for s in os.getenv("SYMBOLS", "MESM6,MNQM6").split(",")
    if s.strip()
]

# Stop/target in native instrument points
RISK_PARAMS: dict[str, dict] = {
    "MES": {
        "stop": float(os.getenv("MES_STOP_POINTS", "5")),
        "target": float(os.getenv("MES_TARGET_POINTS", "6")),
    },
    "MNQ": {
        "stop": float(os.getenv("MNQ_STOP_POINTS", "30")),
        "target": float(os.getenv("MNQ_TARGET_POINTS", "40")),
    },
}

_TV_AUTH_URL: dict[str, str] = {
    "live": "https://live.tradovateapi.com/v1/auth/accesstokenrequest",
    "demo": "https://demo.tradovateapi.com/v1/auth/accesstokenrequest",
}
_TV_WS_URL: dict[str, str] = {
    "live": "wss://md.tradovateapi.com/v1/websocket",
    "demo": "wss://md-demo.tradovateapi.com/v1/websocket",
}

_NO_DATA_MSG = (
    "Real futures data unavailable. Configure Tradovate/CME market data "
    "permissions or connect another real futures data source."
)

# ── Alpaca fallback config ─────────────────────────────────────────────────────

ALPACA_API_KEY: str = os.getenv("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY: str = os.getenv("ALPACA_SECRET_KEY", "")
ALPACA_BASE_URL: str = os.getenv("ALPACA_BASE_URL", "https://paper-api.alpaca.markets")
_ALPACA_DATA_URL = "https://data.alpaca.markets"

# Symbol used when Tradovate is unavailable — SPY ×10 ≈ ES/MES price proxy
_ALPACA_PROXY_SYMBOL = "SPY"

# Whether to use Alpaca as the data source
# Auto-detected: True if TRADOVATE_USERNAME is blank
_use_alpaca_fallback: bool = not bool(TRADOVATE_USERNAME)

# ── Shared state (read by FastAPI health/stats endpoints) ──────────────────────

_agent_state: dict = {
    "status": "starting",
    "symbols": SYMBOLS,
    "open_paper_trades": 0,
    "last_candle_time": None,
    "last_alert_time": None,
    "alerts_today": 0,
    "data_source": (
        f"Alpaca 1-min SPY (proxy)"
        if not TRADOVATE_USERNAME
        else f"Tradovate {TRADOVATE_ENV} WebSocket"
    ),
    "error": None,
}


# ── SockJS frame parsing ───────────────────────────────────────────────────────

def _parse_sockjs(raw: str) -> list[dict]:
    """Decode a SockJS frame into a list of message dicts."""
    if not raw:
        return []
    frame_type = raw[0]
    if frame_type in ('o', 'h', 'c'):
        return []
    if frame_type == 'a':
        try:
            items = json.loads(raw[1:])
            return [json.loads(item) for item in items if isinstance(item, str)]
        except (json.JSONDecodeError, ValueError):
            return []
    return []


def _bar_to_candle(bar: dict) -> dict:
    return {
        "open": float(bar["open"]),
        "high": float(bar["high"]),
        "low": float(bar["low"]),
        "close": float(bar["close"]),
        "timestamp": str(bar.get("timestamp", "")),
    }


# ── Tradovate WebSocket client ─────────────────────────────────────────────────

class TradovateMarketData:
    """
    Connects to the Tradovate market data WebSocket, authenticates,
    subscribes to 1-minute chart data per symbol, and fires `on_bar_closed`
    with the full closed-bar list each time a new bar opens.
    """

    def __init__(
        self,
        on_bar_closed: Callable[[str, list[dict]], None],
        symbols: list[str],
    ):
        self._on_bar_closed = on_bar_closed
        self._symbols = symbols

        self._token: str = ""
        self._msg_id: int = 0

        # Subscriptions are mapped by their sequential index (Tradovate assigns
        # chart IDs 0, 1, 2... in the order subscriptions are made).
        self._sub_order: list[str] = []           # index → symbol
        self._pending_sub_ids: set[int] = set()   # msg_ids awaiting server ack

        # Per-chart-id candle state
        self._buffers: dict[int, list[dict]] = {}  # chart_id → ordered candle list
        self._current_ts: dict[int, str] = {}      # chart_id → current open-bar timestamp

    def _next_id(self) -> int:
        self._msg_id += 1
        return self._msg_id

    async def authenticate(self) -> bool:
        """Obtain Tradovate access token via REST."""
        if not TRADOVATE_USERNAME or not TRADOVATE_PASSWORD:
            logger.error(
                "Tradovate credentials not configured — "
                "set TRADOVATE_USERNAME and TRADOVATE_PASSWORD"
            )
            return False

        url = _TV_AUTH_URL.get(TRADOVATE_ENV, _TV_AUTH_URL["demo"])
        payload: dict = {
            "name": TRADOVATE_USERNAME,
            "password": TRADOVATE_PASSWORD,
            "appId": TRADOVATE_APP_ID,
            "appVersion": TRADOVATE_APP_VERSION,
        }
        if TRADOVATE_CID:
            payload["cid"] = int(TRADOVATE_CID)
        if TRADOVATE_SEC:
            payload["sec"] = TRADOVATE_SEC

        try:
            async with httpx.AsyncClient(timeout=15.0) as client:
                r = await client.post(url, json=payload)
            data = r.json()
        except Exception as e:
            logger.error(f"Tradovate auth HTTP error: {e}")
            return False

        if "accessToken" in data:
            self._token = data["accessToken"]
            logger.info(
                f"Tradovate authenticated — env={TRADOVATE_ENV} "
                f"userId={data.get('userId')} name={data.get('name')}"
            )
            return True

        # Surface the full Tradovate error so it appears in Railway logs
        err = data.get("p-ticket") or data.get("message") or str(data)[:200]
        logger.error(
            f"Tradovate auth failed — HTTP {r.status_code} — "
            f"payload name={TRADOVATE_USERNAME!r} appId={TRADOVATE_APP_ID!r} — "
            f"response: {err}"
        )
        return False

    # Give up after this many consecutive auth failures and let MirrorAgent
    # fall back to the Alpaca proxy if one is available.
    _MAX_AUTH_FAILURES = 5

    async def run_forever(self) -> None:
        """Main reconnect loop — authenticates then runs the WebSocket session."""
        ws_url = _TV_WS_URL.get(TRADOVATE_ENV, _TV_WS_URL["demo"])
        backoff = 5
        _auth_error_notified = False  # send Discord error only once per agent start
        _consecutive_auth_failures = 0

        while True:
            try:
                if not self._token:
                    ok = await self.authenticate()
                    if not ok:
                        _consecutive_auth_failures += 1
                        log_error(_NO_DATA_MSG)
                        _agent_state["status"] = "error"
                        _agent_state["error"] = "Authentication failed — check Tradovate credentials"
                        if not _auth_error_notified:
                            send_error(_NO_DATA_MSG)
                            _auth_error_notified = True
                        if _consecutive_auth_failures >= self._MAX_AUTH_FAILURES and ALPACA_API_KEY:
                            raise RuntimeError(
                                f"Tradovate auth failed {_consecutive_auth_failures} times in a row — "
                                "giving up and switching to Alpaca fallback"
                            )
                        await asyncio.sleep(backoff)
                        backoff = min(backoff * 2, 300)
                        continue
                    _consecutive_auth_failures = 0  # reset on success

                _auth_error_notified = False  # reset on successful connect

                logger.info(f"Connecting to Tradovate MD WebSocket ({TRADOVATE_ENV}): {ws_url}")
                async with websockets.connect(
                    ws_url,
                    ping_interval=20,
                    ping_timeout=30,
                    extra_headers={"User-Agent": f"MirrorAgent/{TRADOVATE_APP_VERSION}"},
                ) as ws:
                    backoff = 5
                    _agent_state["status"] = "connected"
                    _agent_state["error"] = None

                    # SockJS sends 'o' (open) immediately on connect
                    raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                    if not raw.startswith('o'):
                        logger.warning(f"Unexpected first WS frame: {raw[:60]!r}")

                    # Authorize over WebSocket
                    auth_id = self._next_id()
                    await ws.send(f"authorize\n{auth_id}\n\n{self._token}")

                    authorized = False
                    for _ in range(10):
                        raw = await asyncio.wait_for(ws.recv(), timeout=15.0)
                        for msg in _parse_sockjs(raw):
                            if msg.get("i") == auth_id and msg.get("s") == 200:
                                authorized = True
                                logger.info("Tradovate WS authorized")
                        if authorized:
                            break

                    if not authorized:
                        raise ConnectionError("WS authorization not confirmed — token may be expired")

                    # Reset subscription state for this session
                    self._sub_order.clear()
                    self._pending_sub_ids.clear()
                    self._buffers.clear()
                    self._current_ts.clear()

                    # Subscribe to each symbol's 1-minute chart
                    for symbol in self._symbols:
                        sub_id = self._next_id()
                        self._pending_sub_ids.add(sub_id)
                        self._sub_order.append(symbol)
                        body = json.dumps({
                            "symbol": symbol,
                            "chartDescription": {
                                "underlyingType": "MinuteBar",
                                "elementSize": 1,
                                "elementSizeUnit": "UnderlyingUnits",
                                "withHistogram": False,
                            },
                            "timeRange": {"asMuchAsElements": 60},
                        })
                        await ws.send(f"md/subscribeTVChart\n{sub_id}\n\n{body}")
                        logger.info(f"Subscribing to {symbol} 1-min chart (msg_id={sub_id})")

                    # Main message loop
                    async for raw in ws:
                        await self._handle_raw(raw)

            except asyncio.TimeoutError:
                logger.warning("Tradovate WS timed out — reconnecting")
                _agent_state["status"] = "reconnecting"
                self._token = ""

            except (websockets.ConnectionClosed, ConnectionError) as e:
                logger.warning(f"Tradovate WS closed ({e}) — retrying in {backoff}s")
                _agent_state["status"] = "reconnecting"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)
                self._token = ""

            except Exception as e:
                err_str = str(e)
                logger.error(f"Tradovate WS unexpected error: {e}")
                log_error(f"Tradovate WS error: {err_str}")
                _agent_state["status"] = "error"
                _agent_state["error"] = err_str[:120]

                if any(x in err_str for x in ("403", "401", "permission", "Forbidden")):
                    send_error(_NO_DATA_MSG)

                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)
                self._token = ""

    async def _handle_raw(self, raw: str) -> None:
        for msg in _parse_sockjs(raw):
            await self._route(msg)

    async def _route(self, msg: dict) -> None:
        i = msg.get("i")
        e = msg.get("e")

        # Subscription acknowledgement
        if i is not None and i in self._pending_sub_ids:
            self._pending_sub_ids.discard(i)
            if msg.get("s") == 200:
                logger.debug(f"Chart subscription ack'd (msg_id={i})")
            else:
                err = f"Chart subscription rejected (msg_id={i} s={msg.get('s')}): {msg.get('d')}"
                logger.error(err)
                send_error(f"Mirror: {err}")
            return

        if e == "chart":
            await self._handle_chart(msg.get("d", {}))

    async def _handle_chart(self, d: dict) -> None:
        for chart in d.get("charts", []):
            chart_id: int = chart.get("id", -1)
            bars: list[dict] = chart.get("bars", [])
            if not bars:
                continue
            if chart_id < 0 or chart_id >= len(self._sub_order):
                logger.debug(f"Chart event for unknown chart_id={chart_id} (have {len(self._sub_order)} subs)")
                continue

            symbol = self._sub_order[chart_id]
            new_candles = [_bar_to_candle(b) for b in bars]

            if chart_id not in self._buffers:
                # Initial snapshot — treat all bars as historical
                self._buffers[chart_id] = list(new_candles)
                self._current_ts[chart_id] = new_candles[-1]["timestamp"]
                logger.info(f"{symbol}: loaded {len(new_candles)} initial bars (chart_id={chart_id})")
                return

            buf = self._buffers[chart_id]
            prev_ts = self._current_ts[chart_id]
            latest_ts = new_candles[-1]["timestamp"]

            if latest_ts > prev_ts:
                # A new bar has opened — the previous bar is now closed.
                # Update the stored last bar with its final OHLC values.
                for nc in new_candles:
                    if nc["timestamp"] == prev_ts and buf:
                        buf[-1] = nc
                        break

                # Append any bars with timestamps not yet in buffer
                existing_ts = {c["timestamp"] for c in buf}
                for nc in new_candles:
                    if nc["timestamp"] not in existing_ts:
                        buf.append(nc)
                        existing_ts.add(nc["timestamp"])

                # Keep buffer bounded
                if len(buf) > 100:
                    self._buffers[chart_id] = buf[-100:]
                    buf = self._buffers[chart_id]

                self._current_ts[chart_id] = buf[-1]["timestamp"]
                _agent_state["last_candle_time"] = datetime.now(timezone.utc).isoformat()

                # Fire callback with all closed bars (everything except the current open bar)
                closed = list(buf[:-1])
                self._on_bar_closed(symbol, closed)

            else:
                # Same bar still building — update the last candle in place
                if buf and new_candles:
                    buf[-1] = new_candles[-1]


# ── Alpaca 1-min bar poller (SPY proxy fallback) ───────────────────────────────

class AlpacaMarketData:
    """
    Polls Alpaca's REST API for 1-minute SPY bars every 60 seconds.
    Fires `on_bar_closed` with the symbol "SPY" and the last 100 closed bars
    each time a new bar has appeared.

    Used automatically when TRADOVATE_USERNAME is not configured, giving the
    Mirror Agent a free data source via the existing Alpaca paper-trading keys.
    """

    _BARS_URL = f"{_ALPACA_DATA_URL}/v2/stocks/{{symbol}}/bars"

    def __init__(
        self,
        on_bar_closed: Callable[[str, list[dict]], None],
    ):
        self._on_bar_closed = on_bar_closed
        self._last_ts: Optional[str] = None
        self._buffer: list[dict] = []

    async def _fetch_bars(self) -> list[dict]:
        """Fetch the last 100 1-minute bars for SPY from Alpaca."""
        if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")

        url = self._BARS_URL.format(symbol=_ALPACA_PROXY_SYMBOL)
        params = {
            "timeframe": "1Min",
            "limit": 100,
            "feed": "iex",
            "sort": "asc",
        }
        headers = {
            "APCA-API-KEY-ID": ALPACA_API_KEY,
            "APCA-API-SECRET-KEY": ALPACA_SECRET_KEY,
        }
        async with httpx.AsyncClient(timeout=15.0) as client:
            r = await client.get(url, params=params, headers=headers)
        r.raise_for_status()
        data = r.json()
        bars = data.get("bars") or []
        return [
            {
                "open": float(b["o"]),
                "high": float(b["h"]),
                "low": float(b["l"]),
                "close": float(b["c"]),
                "timestamp": b["t"],
            }
            for b in bars
        ]

    async def run_forever(self) -> None:
        """Poll Alpaca every ~60 seconds and fire on_bar_closed for each new bar."""
        backoff = 10
        no_key_notified = False

        while True:
            try:
                if not ALPACA_API_KEY or not ALPACA_SECRET_KEY:
                    if not no_key_notified:
                        msg = (
                            "Mirror Agent: no data source configured. "
                            "Set TRADOVATE_USERNAME (futures) or ensure ALPACA_API_KEY is set (SPY proxy)."
                        )
                        log_error(msg)
                        send_error(msg)
                        no_key_notified = True
                        _agent_state["status"] = "error"
                        _agent_state["error"] = "No data source configured"
                    await asyncio.sleep(60)
                    continue

                no_key_notified = False
                bars = await self._fetch_bars()

                if not bars:
                    await asyncio.sleep(30)
                    continue

                latest_ts = bars[-1]["timestamp"]
                _agent_state["status"] = "connected"
                _agent_state["error"] = None
                backoff = 10

                if latest_ts != self._last_ts:
                    self._buffer = bars
                    self._last_ts = latest_ts
                    _agent_state["last_candle_time"] = datetime.now(timezone.utc).isoformat()

                    # Pass all bars except the most recent (still building) as closed
                    closed = list(self._buffer[:-1])
                    if closed:
                        self._on_bar_closed(_ALPACA_PROXY_SYMBOL, closed)

                # Wait until roughly the next minute boundary (+5s buffer)
                now = datetime.now(timezone.utc)
                seconds_to_next = 65 - (now.second + now.microsecond / 1e6)
                await asyncio.sleep(max(seconds_to_next, 5))

            except httpx.HTTPStatusError as e:
                logger.warning(f"Alpaca bars HTTP {e.response.status_code}: {e}")
                _agent_state["status"] = "reconnecting"
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)

            except Exception as e:
                logger.error(f"Alpaca market data error: {e}")
                log_error(f"Alpaca market data error: {e}")
                _agent_state["status"] = "error"
                _agent_state["error"] = str(e)[:120]
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, 300)


# ── Mirror Agent ───────────────────────────────────────────────────────────────

class MirrorAgent:
    """
    Orchestrates Mirror Agent v1: runs strategy on closed bars, fires Discord
    alerts for qualifying setups, and tracks paper simulation results.
    """

    def __init__(self):
        self._strategy = MirrorStrategy()
        self._paper = PaperSim()
        # Choose data source: Tradovate if credentials present, else Alpaca SPY proxy
        if TRADOVATE_USERNAME:
            self._md: TradovateMarketData | AlpacaMarketData = TradovateMarketData(
                on_bar_closed=self._on_bar_closed,
                symbols=SYMBOLS,
            )
        else:
            self._md = AlpacaMarketData(on_bar_closed=self._on_bar_closed)
            _agent_state["data_source"] = "Alpaca 1-min SPY (proxy)"
            logger.info("Mirror Agent: no Tradovate credentials — using Alpaca SPY proxy")
        self._alerts_today: int = 0

    def _on_bar_closed(self, symbol: str, candles: list[dict]) -> None:
        """Called synchronously each time a 1-minute bar closes."""
        if not candles:
            return

        latest = candles[-1]

        # Parse candle timestamp for paper sim
        try:
            candle_time = datetime.fromisoformat(
                latest["timestamp"].replace("Z", "+00:00")
            )
        except (ValueError, KeyError):
            candle_time = datetime.now(timezone.utc)

        # Update open paper trades against this candle's range
        try:
            closed_trades = self._paper.update(
                candle_high=latest["high"],
                candle_low=latest["low"],
                candle_time=candle_time,
            )
            _agent_state["open_paper_trades"] = self._paper.open_count

            for t in closed_trades:
                send_paper_result(
                    symbol=t["symbol"],
                    direction=t["direction"],
                    outcome=t["outcome"],
                    points=t["points"],
                    duration_minutes=t["duration_minutes"],
                    entry=t["entry"],
                    exit_price=t["exit_price"],
                )
        except Exception as e:
            logger.error(f"Paper sim update error ({symbol}): {e}")

        # Run Mirror System v2 scoring
        try:
            setups = self._strategy.analyze(candles)
        except Exception as e:
            logger.error(f"Strategy error ({symbol}): {e}")
            log_error(f"Strategy error ({symbol}): {e}")
            return

        for setup in setups:
            if setup.score < SCORE_THRESHOLD:
                continue

            price = latest["close"]
            rp = self._get_risk_params(symbol)
            stop_pts = rp["stop"]
            tgt_pts = rp["target"]

            if setup.direction == "LONG":
                stop = round(price - stop_pts, 2)
                target = round(price + tgt_pts, 2)
            else:
                stop = round(price + stop_pts, 2)
                target = round(price - tgt_pts, 2)

            sent = send_alert(
                symbol=symbol,
                direction=setup.direction,
                score=setup.score,
                price=price,
                stop=stop,
                target=target,
                trend_read=setup.trend_read,
                pullback_count=setup.pullback_count,
                confirm_vs_avg=setup.confirm_vs_avg_body,
                extension_risk=setup.extension_risk,
                setup_summary=setup.setup_summary,
                ema9=setup.ema9,
                ema21=setup.ema21,
            )

            if not sent:
                continue  # suppressed by cooldown

            self._alerts_today += 1
            now_iso = datetime.now(timezone.utc).isoformat()
            _agent_state["alerts_today"] = self._alerts_today
            _agent_state["last_alert_time"] = now_iso

            log_alert({
                "timestamp": now_iso,
                "symbol": symbol,
                "direction": setup.direction,
                "score": setup.score,
                "price": price,
                "stop": stop,
                "target": target,
                "trend_score": setup.trend_score,
                "pullback_score": setup.pullback_score,
                "confirm_score": setup.confirm_score,
                "location_score": setup.location_score,
                "pullback_count": setup.pullback_count,
                "extension_risk": setup.extension_risk,
                "ema9": setup.ema9,
                "ema21": setup.ema21,
            })

            self._paper.open_trade(
                symbol=symbol,
                direction=setup.direction,
                entry=price,
                stop=stop,
                target=target,
                score=setup.score,
                summary=setup.setup_summary,
            )
            _agent_state["open_paper_trades"] = self._paper.open_count

    def _get_risk_params(self, symbol: str) -> dict:
        for prefix in ("MNQ", "MES"):
            if symbol.startswith(prefix):
                return RISK_PARAMS.get(prefix, RISK_PARAMS["MES"])
        return RISK_PARAMS["MES"]

    def get_stats(self) -> dict:
        stats = self._paper.get_stats()
        return {
            "alerts_today": self._alerts_today,
            "paper_wins": stats["wins"],
            "paper_losses": stats["losses"],
            "paper_win_rate": stats["win_rate"],
            "net_paper_points_by_symbol": stats["net_points_by_symbol"],
            "open_paper_trades": self._paper.open_count,
        }

    async def run(self) -> None:
        send_startup(SYMBOLS)
        try:
            await self._md.run_forever()
        except Exception as e:
            # If Tradovate fails permanently, fall back to Alpaca SPY proxy
            if isinstance(self._md, TradovateMarketData) and ALPACA_API_KEY:
                logger.warning(
                    f"Tradovate data source failed ({e}) — switching to Alpaca SPY proxy"
                )
                fallback_msg = (
                    "Tradovate connection failed. Switching to Alpaca SPY proxy for Mirror Agent alerts."
                )
                send_error(fallback_msg)
                _agent_state["data_source"] = "Alpaca 1-min SPY (proxy, Tradovate fallback)"
                self._md = AlpacaMarketData(on_bar_closed=self._on_bar_closed)
                await self._md.run_forever()
            else:
                raise


# ── Singleton and background launcher ─────────────────────────────────────────

_agent: Optional[MirrorAgent] = None
_agent_lock = threading.Lock()


def get_or_create_agent() -> MirrorAgent:
    global _agent
    with _agent_lock:
        if _agent is None:
            _agent = MirrorAgent()
    return _agent


def start_background() -> None:
    """Launch Mirror Agent in a daemon thread with its own asyncio event loop."""
    agent = get_or_create_agent()

    def _run() -> None:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(agent.run())
        except Exception as e:
            logger.error(f"Mirror Agent crashed: {e}")
            log_error(f"Mirror Agent crashed: {e}")
            _agent_state["status"] = "crashed"
            _agent_state["error"] = str(e)[:120]
        finally:
            loop.close()

    t = threading.Thread(target=_run, daemon=True, name="mirror-agent")
    t.start()
    logger.info(f"Mirror Agent started in background (env={TRADOVATE_ENV} symbols={SYMBOLS})")
