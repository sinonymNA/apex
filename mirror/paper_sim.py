"""
mirror/paper_sim.py — Paper simulation tracker for Mirror Agent.

Creates simulated planned trades when alerts fire.
Monitors open trades candle-by-candle and closes them at stop, target, or timeout.

ALERT-ONLY: No broker interaction. No real orders. Entirely in-memory + JSONL.
"""
import os
import uuid
from datetime import datetime, timezone, timedelta
from typing import Optional

from loguru import logger
from mirror.logger import log_paper_trade

PAPER_MAX_MINUTES: int = int(os.getenv("PAPER_MAX_MINUTES", "45"))


class PaperSim:
    """Thread-safe-ish paper trade tracker (single-threaded asyncio use only)."""

    def __init__(self, max_minutes: int = PAPER_MAX_MINUTES):
        self.max_minutes = max_minutes
        self._open: dict[str, dict] = {}    # trade_id → trade
        self._closed: list[dict] = []

    # ── Opening ────────────────────────────────────────────────────────────────

    def open_trade(
        self,
        *,
        symbol: str,
        direction: str,
        entry: float,
        stop: float,
        target: float,
        score: int,
        summary: str,
    ) -> str:
        trade_id = str(uuid.uuid4())[:8]
        now = datetime.now(timezone.utc)
        trade = {
            "id": trade_id,
            "opened_at_iso": now.isoformat(),
            "symbol": symbol,
            "direction": direction,
            "entry": entry,
            "stop": stop,
            "target": target,
            "score": score,
            "summary": summary,
            "status": "OPEN",
            # internal only — deleted before logging
            "_opened_at": now,
        }
        self._open[trade_id] = trade
        log_paper_trade({k: v for k, v in trade.items() if not k.startswith("_")})
        logger.info(
            f"Paper trade OPEN: {symbol} {direction} @ {entry:.2f} "
            f"| stop={stop:.2f} target={target:.2f} | id={trade_id}"
        )
        return trade_id

    # ── Per-candle update ──────────────────────────────────────────────────────

    def update(
        self,
        candle_high: float,
        candle_low: float,
        candle_time: Optional[datetime] = None,
    ) -> list[dict]:
        """
        Check all open trades against the latest closed candle.
        Returns list of newly closed trade dicts (for Discord notification).

        Priority: target > stop > timeout (within the same candle, target wins
        unless both hit — then it's AMBIGUOUS with conservative LOSS outcome).
        """
        candle_time = candle_time or datetime.now(timezone.utc)
        newly_closed = []

        for tid in list(self._open):
            trade = self._open[tid]
            result = self._evaluate(trade, candle_high, candle_low, candle_time)
            if result is None:
                continue

            opened_at = trade.pop("_opened_at")
            elapsed = (candle_time - opened_at).total_seconds() / 60

            trade.update({
                "status": "CLOSED",
                "outcome": result["outcome"],
                "exit_price": result["exit_price"],
                "points": result["points"],
                "closed_at_iso": candle_time.isoformat(),
                "duration_minutes": round(elapsed, 1),
            })

            del self._open[tid]
            self._closed.append(trade)
            log_paper_trade({k: v for k, v in trade.items() if not k.startswith("_")})
            newly_closed.append(trade)
            logger.info(
                f"Paper trade CLOSED: {trade['symbol']} {trade['direction']} "
                f"| {trade['outcome']} {'+' if trade['points'] >= 0 else ''}{trade['points']:.2f}pts "
                f"| {trade['duration_minutes']:.0f}m | id={tid}"
            )

        return newly_closed

    # ── Evaluation helpers ─────────────────────────────────────────────────────

    def _evaluate(
        self,
        trade: dict,
        candle_high: float,
        candle_low: float,
        candle_time: datetime,
    ) -> Optional[dict]:
        direction = trade["direction"]
        entry = trade["entry"]
        stop = trade["stop"]
        target = trade["target"]
        opened_at = trade["_opened_at"]

        # Timeout check first
        elapsed_min = (candle_time - opened_at).total_seconds() / 60
        if elapsed_min >= self.max_minutes:
            mid = (candle_high + candle_low) / 2
            pts = (mid - entry) if direction == "LONG" else (entry - mid)
            return {"outcome": "TIMEOUT", "exit_price": round(mid, 2), "points": round(pts, 2)}

        if direction == "LONG":
            target_hit = candle_high >= target
            stop_hit = candle_low <= stop
        else:
            target_hit = candle_low <= target
            stop_hit = candle_high >= stop

        if target_hit and stop_hit:
            # Same-candle ambiguity — conservative assumption is LOSS
            pts = (stop - entry) if direction == "LONG" else (entry - stop)
            return {"outcome": "AMBIGUOUS", "exit_price": stop, "points": round(pts, 2)}

        if target_hit:
            pts = (target - entry) if direction == "LONG" else (entry - target)
            return {"outcome": "WIN", "exit_price": target, "points": round(pts, 2)}

        if stop_hit:
            pts = (stop - entry) if direction == "LONG" else (entry - stop)
            return {"outcome": "LOSS", "exit_price": stop, "points": round(pts, 2)}

        return None

    # ── Stats ──────────────────────────────────────────────────────────────────

    @property
    def open_count(self) -> int:
        return len(self._open)

    def get_stats(self) -> dict:
        wins = [t for t in self._closed if t.get("outcome") == "WIN"]
        losses = [t for t in self._closed if t.get("outcome") in ("LOSS", "AMBIGUOUS")]
        total = len(wins) + len(losses)
        return {
            "total_closed": len(self._closed),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": round(len(wins) / total, 3) if total > 0 else 0.0,
            "net_points_by_symbol": self._net_by_symbol(),
        }

    def _net_by_symbol(self) -> dict:
        net: dict[str, float] = {}
        for t in self._closed:
            sym = t.get("symbol", "?")
            net[sym] = round(net.get(sym, 0.0) + t.get("points", 0.0), 2)
        return net
