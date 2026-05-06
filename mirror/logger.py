"""
mirror/logger.py — File logging for Mirror Agent.

Writes to logs/ directory:
  mirror_alerts.jsonl        — every setup alert fired
  mirror_paper_trades.jsonl  — every paper trade opened/closed
  mirror_errors.log          — errors and warnings
"""
import json
import logging
import os
from pathlib import Path

LOGS_DIR = Path(os.getenv("LOGS_DIR", "logs"))
LOGS_DIR.mkdir(exist_ok=True)

_ALERTS_PATH = LOGS_DIR / "mirror_alerts.jsonl"
_TRADES_PATH = LOGS_DIR / "mirror_paper_trades.jsonl"
_ERRORS_PATH = LOGS_DIR / "mirror_errors.log"

# Dedicated file logger for errors
_err_log = logging.getLogger("mirror.errors")
_err_log.setLevel(logging.ERROR)
if not _err_log.handlers:
    _fh = logging.FileHandler(_ERRORS_PATH)
    _fh.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    _err_log.addHandler(_fh)


def log_alert(alert: dict) -> None:
    """Append an alert to mirror_alerts.jsonl."""
    with open(_ALERTS_PATH, "a") as f:
        f.write(json.dumps(alert) + "\n")


def log_paper_trade(trade: dict) -> None:
    """Append a paper trade open or close to mirror_paper_trades.jsonl."""
    with open(_TRADES_PATH, "a") as f:
        f.write(json.dumps(trade) + "\n")


def log_error(msg: str) -> None:
    """Append an error line to mirror_errors.log."""
    _err_log.error(msg)
