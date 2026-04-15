"""
diagnostics/analyzer.py — Anomaly analysis for Apex Trading System.

Uses claude-sonnet-4-6 for structured diagnosis when ANTHROPIC_API_KEY is set.
Falls back to static maps gracefully if the key is missing or the API call fails.
"""
import json
import os

from loguru import logger

from worker.db import log_anomaly


def _call_claude(event_type: str, context: dict) -> dict | None:
    """Call claude-sonnet-4-6 for structured anomaly diagnosis.
    Returns parsed dict on success, None on any failure.
    """
    api_key = os.getenv("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        prompt = (
            f"You are a risk analyst for an automated SPY paper trading system.\n"
            f"A '{event_type}' anomaly occurred. Context:\n"
            f"{json.dumps(context, indent=2, default=str)}\n\n"
            f"Respond ONLY with valid JSON (no markdown fences):\n"
            f'{{"severity":"CRITICAL|WARNING|INFO","diagnosis":"...","recommendation":"...","estimated_cause":"..."}}'
        )
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=400,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()
        # Strip any accidental markdown code fences
        if "```" in text:
            text = text.split("```")[1].lstrip("json").strip()
        return json.loads(text)
    except Exception as e:
        logger.warning(f"Claude diagnostic call failed (non-fatal): {e}")
        return None


def analyze_anomaly(event_type: str, context: dict) -> dict:
    """
    Analyze a trading anomaly and log it to the database.

    Args:
        event_type: Type of anomaly (e.g., "kill_switch", "slippage_spike",
                    "regime_shift", "drawdown_warning")
        context: Dictionary with relevant data for the event

    Returns:
        dict with keys: severity, diagnosis, recommendation, estimated_cause
    """
    # ── Severity classification ───────────────────────────────────────────────
    severity_map = {
        "kill_switch": "CRITICAL",
        "drawdown_emergency": "CRITICAL",
        "drawdown_warning": "WARNING",
        "slippage_spike": "WARNING",
        "regime_shift": "INFO",
        "max_loss_hit": "CRITICAL",
        "connection_error": "WARNING",
    }
    severity = severity_map.get(event_type, "INFO")

    # ── Try Claude first; fall back to static maps ────────────────────────────
    claude_result = _call_claude(event_type, context)

    if claude_result:
        diagnosis       = claude_result.get("diagnosis", "")
        recommendation  = claude_result.get("recommendation", "")
        estimated_cause = claude_result.get("estimated_cause", "")
        severity        = claude_result.get("severity", severity)
    else:
        diagnosis_map = {
            "kill_switch": (
                "Kill switch triggered due to consecutive losses or drawdown breach. "
                "Trading is paused for remainder of session."
            ),
            "drawdown_emergency": (
                "Daily loss limit reached. Portfolio has exceeded the maximum acceptable "
                "daily drawdown threshold."
            ),
            "drawdown_warning": (
                "Drawdown approaching critical level. Monitor closely and consider "
                "reducing position size on next entry."
            ),
            "slippage_spike": (
                "Execution slippage exceeded normal parameters. May indicate low liquidity "
                "or rapid price movement at time of order."
            ),
            "regime_shift": (
                "Market regime has changed. Strategy performance may differ from backtest "
                "expectations in current regime."
            ),
            "max_loss_hit": (
                "Maximum daily loss reached. All trading suspended for the session."
            ),
            "connection_error": (
                "Connection to broker API interrupted. Orders may not have executed as expected."
            ),
        }

        recommendation_map = {
            "kill_switch": "Review trades in daily email. Resume tomorrow after checking regime.",
            "drawdown_emergency": "Stop trading for the day. Investigate entries in morning review.",
            "drawdown_warning": "Continue with caution. Reduce to 1 contract if another loss occurs.",
            "slippage_spike": "Check broker connectivity. Avoid entries during high-impact news windows.",
            "regime_shift": "Verify regime classifier output. Consider skipping entries in Range-Bound regime.",
            "max_loss_hit": "No further action today. Review P&L attribution tomorrow morning.",
            "connection_error": "Check API keys and network. Verify all open positions manually.",
        }

        estimated_cause_map = {
            "kill_switch": "Consecutive losing trades exceeded the configured threshold.",
            "drawdown_emergency": "Cumulative intraday losses breached the maximum daily loss floor.",
            "drawdown_warning": "Intraday equity drawdown approaching the configured warning level.",
            "slippage_spike": "Market microstructure or news event caused unusual bid-ask spread.",
            "regime_shift": "HMM + RF classifier detected a transition between market states.",
            "max_loss_hit": "Total session P&L hit the hard stop defined in risk constants.",
            "connection_error": "Network interruption or expired API credentials.",
        }

        diagnosis       = diagnosis_map.get(event_type, f"Anomaly of type '{event_type}' detected. Manual review recommended.")
        recommendation  = recommendation_map.get(event_type, "Review system logs and verify all positions.")
        estimated_cause = estimated_cause_map.get(event_type, "Unknown — see context.")

    result = {
        "severity":        severity,
        "diagnosis":       diagnosis,
        "recommendation":  recommendation,
        "estimated_cause": estimated_cause,
    }

    # Log to database (estimated_cause stored in context_json)
    log_anomaly({
        "event_type":     event_type,
        "severity":       severity,
        "diagnosis":      diagnosis,
        "recommendation": recommendation,
        "context_json":   json.dumps({**context, "estimated_cause": estimated_cause}, default=str),
    })

    logger.warning(f"[{severity}] Anomaly: {event_type} — {diagnosis[:80]}")
    return result
