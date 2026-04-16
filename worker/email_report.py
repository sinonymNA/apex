"""
worker/email_report.py — Email reports for Apex Trading System.

Three emails per trading day:
  9:25 AM  — Morning brief: regime, breakout level, entry plan
  12:00 PM — Noon update: morning recap, P&L, near-misses
  4:05 PM  — EOD summary: full trade log, gates, anomalies

Sends via Gmail SMTP using GMAIL_USER + GMAIL_APP_PASSWORD.
Graceful degradation: logs a warning and returns if env vars are missing.
"""
import os
import smtplib
from datetime import date
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from loguru import logger

from worker.db import (
    get_today_summary,
    get_recent_trades,
    get_gate_status,
    get_risk_log,
    get_recent_anomalies,
    get_latest_status,
    get_today_near_misses,
)

_SHARED_CSS = """
  body { font-family: 'Courier New', monospace; background: #0d0d0d; color: #e0e0e0; margin: 0; padding: 20px; }
  h2 { color: #00e676; border-bottom: 1px solid #333; padding-bottom: 8px; }
  h3 { color: #82b1ff; margin-top: 20px; }
  table { width: 100%; border-collapse: collapse; margin: 8px 0; }
  th { background: #1a1a2e; color: #82b1ff; padding: 8px 10px; text-align: left; font-size: 12px; }
  tr:nth-child(even) { background: #111; }
  .stat { display: inline-block; background: #1a1a1a; border: 1px solid #333; padding: 8px 16px; margin: 4px; border-radius: 4px; }
  .stat-val { font-size: 20px; font-weight: bold; }
  .stat-label { font-size: 11px; color: #888; }
"""


def _get_email_creds():
    """Return (gmail_user, gmail_pass, notify_email) or (None, None, None)."""
    u = os.getenv("GMAIL_USER")
    p = os.getenv("GMAIL_APP_PASSWORD")
    n = os.getenv("NOTIFY_EMAIL", u)
    if not u or not p:
        logger.warning("GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping email")
        return None, None, None
    return u, p, n


def _smtp_send(gmail_user: str, gmail_pass: str, to: str, subject: str, html: str):
    """Send an HTML email via Gmail SMTP. Logs errors, never raises."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"]    = gmail_user
    msg["To"]      = to
    msg.attach(MIMEText(html, "html"))
    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_pass)
            server.sendmail(gmail_user, to, msg.as_string())
        logger.info(f"Email sent → {to} | {subject}")
    except Exception as e:
        logger.error(f"Email failed: {e}")


def send_morning_brief(session_day: int, regime: str, spy_price: float,
                       breakout_level: float, atr: float):
    """Send 9:25 AM morning briefing email."""
    u, p, to = _get_email_creds()
    if not u:
        return

    today_str = date.today().isoformat()
    subject   = f"ATS Morning Brief | {today_str} | Day {session_day}/20 | {regime}"

    pct_gap = (breakout_level - spy_price) / spy_price * 100 if spy_price > 0 else 0
    stop_lvl   = round(breakout_level - atr, 2)
    target_lvl = round(breakout_level + 2 * atr, 2)

    trading_blocked = regime in ("Range-Bound", "Extreme Volatility")

    regime_color = {
        "Strong Trend":    "#00c853",
        "Weak Trend":      "#82b1ff",
        "Range-Bound":     "#ff6d00",
        "High Volatility": "#ff6d00",
        "Extreme Volatility": "#d50000",
    }.get(regime, "#888")

    regime_note = {
        "Strong Trend":    "Strong trending conditions — breakout setups have higher probability.",
        "Weak Trend":      "Moderate conditions — standard breakout rules apply.",
        "Range-Bound":     "RANGE-BOUND — all entries blocked until regime changes.",
        "High Volatility": "Elevated volatility — ATR stops will be wider than usual.",
        "Extreme Volatility": "EXTREME VOLATILITY — all entries blocked until regime changes.",
    }.get(regime, "Unknown regime — defaulting to standard rules.")

    blocked_banner = (
        '<div style="background:#2a0000;border:1px solid #d50000;color:#ff6d6d;'
        'padding:10px 16px;border-radius:4px;margin:12px 0">'
        '<strong>TRADING BLOCKED</strong> — regime filter will reject all entries. '
        'Monitor dashboard; entries resume if regime shifts.</div>'
    ) if trading_blocked else ''

    levels_section = "" if trading_blocked else f"""
  <h3>IF BREAKOUT FIRES</h3>
  <div>
    <div class="stat"><div class="stat-val" style="color:#00e676">${breakout_level:.2f}</div><div class="stat-label">Entry (approx)</div></div>
    <div class="stat"><div class="stat-val" style="color:#d50000">${stop_lvl:.2f}</div><div class="stat-label">Stop Loss (1× ATR)</div></div>
    <div class="stat"><div class="stat-val" style="color:#00c853">${target_lvl:.2f}</div><div class="stat-label">Target (2× ATR)</div></div>
    <div class="stat"><div class="stat-val">${atr:.2f}</div><div class="stat-label">ATR-14</div></div>
  </div>"""

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>{_SHARED_CSS}</style></head><body>
  <h2>▲ APEX — Morning Brief</h2>
  <p style="color:#888">Day <strong style="color:#e0e0e0">{session_day}/20</strong> &nbsp;|&nbsp; {today_str}
  &nbsp;|&nbsp; Trading window: <strong style="color:#00e676">9:30 AM – 3:30 PM ET</strong></p>

  <h3>TODAY'S REGIME</h3>
  <div style="background:#1a1a1a;border-left:4px solid {regime_color};padding:10px 16px;border-radius:4px;margin:8px 0">
    <strong style="color:{regime_color};font-size:16px">{regime}</strong><br>
    <span style="color:#ccc;font-size:13px">{regime_note}</span>
  </div>
  {blocked_banner}

  <h3>KEY LEVELS</h3>
  <div>
    <div class="stat"><div class="stat-val">${spy_price:.2f}</div><div class="stat-label">SPY Last Close</div></div>
    <div class="stat"><div class="stat-val" style="color:#00e676">${breakout_level:.2f}</div><div class="stat-label">Breakout Level (20-bar high)</div></div>
    <div class="stat"><div class="stat-val" style="color:{'#ff6d00' if pct_gap > 0.5 else '#00c853'}">{pct_gap:+.2f}%</div><div class="stat-label">Gap to Breakout</div></div>
  </div>
  {levels_section}

  <h3>ENTRY CONDITIONS CHECKLIST</h3>
  <ul style="color:#ccc;line-height:2">
    <li>SPY closes <strong style="color:#00e676">above ${breakout_level:.2f}</strong> on any 5-min bar</li>
    <li>Volume <strong>≥ 1.5×</strong> the 20-bar average</li>
    <li>Time between <strong>9:30 AM – 3:30 PM ET</strong></li>
    <li>Regime is <strong>not</strong> Range-Bound or Extreme Volatility</li>
  </ul>

  <hr style="border-color:#333;margin-top:20px">
  <p style="font-size:10px;color:#555">Noon update at 12:00 PM ET · EOD summary at 4:05 PM ET</p>
</body></html>"""

    _smtp_send(u, p, to, subject, html)


def send_noon_update(session_day: int, daily_pnl: float, trade_count: int,
                     regime: str, spy_price: float, in_position: bool,
                     near_misses_am: list):
    """Send 12:00 PM midday update email."""
    u, p, to = _get_email_creds()
    if not u:
        return

    today_str  = date.today().isoformat()
    pnl_color  = "#00c853" if daily_pnl >= 0 else "#d50000"
    trades_rem = max(0, 3 - trade_count)
    subject    = f"ATS Noon Update | {today_str} | P&L: ${daily_pnl:+.0f} | {trade_count} trade{'s' if trade_count != 1 else ''}"

    # Morning near-miss narrative
    nm_count = len(near_misses_am)
    if nm_count == 0:
        nm_html = "<p style='color:#888'>No bars evaluated yet this morning — either market was quiet or system just started.</p>"
    else:
        pcts    = [abs(r.get("percent_to_breakout") or 999) for r in near_misses_am]
        vols    = [r.get("volume_ratio") or 0.0 for r in near_misses_am]
        closest = min(pcts)
        best_vol = max(vols)
        if closest < 0.1:
            narrative = f"Very close! Price came within {closest:.2f}% of the breakout level (best volume: {best_vol:.2f}×)."
        elif closest < 0.3:
            narrative = f"Near approach — price reached within {closest:.2f}% of trigger. Best volume ratio: {best_vol:.2f}×."
        else:
            narrative = f"No meaningful breakout attempt. Closest approach: {closest:.2f}% away. Volume: {best_vol:.2f}×."
        nm_html = f"<p style='color:#ccc'>{nm_count} bars evaluated. {narrative}</p>"

    pos_color = "#00e676" if in_position else "#888"
    pos_text  = "IN POSITION — monitoring stop/target" if in_position else "No open position"

    html = f"""<!DOCTYPE html><html><head><meta charset="UTF-8">
<style>{_SHARED_CSS}</style></head><body>
  <h2>▲ APEX — Noon Update</h2>
  <p style="color:#888">Day <strong style="color:#e0e0e0">{session_day}/20</strong> &nbsp;|&nbsp; {today_str}
  &nbsp;|&nbsp; <strong style="color:#ff6d00">3h 30m remaining</strong> in session</p>

  <div>
    <div class="stat"><div class="stat-val" style="color:{pnl_color}">${daily_pnl:+.0f}</div><div class="stat-label">P&amp;L So Far</div></div>
    <div class="stat"><div class="stat-val">{trade_count}</div><div class="stat-label">Trades This AM</div></div>
    <div class="stat"><div class="stat-val">{trades_rem}</div><div class="stat-label">Trades Remaining</div></div>
    <div class="stat"><div class="stat-val">${spy_price:.2f}</div><div class="stat-label">SPY Now</div></div>
    <div class="stat"><div class="stat-val" style="color:#82b1ff">{regime}</div><div class="stat-label">Regime</div></div>
  </div>

  <h3>MORNING SESSION RECAP</h3>
  {nm_html}
  <p style="color:{pos_color}"><strong>{pos_text}</strong></p>

  <h3>AFTERNOON OUTLOOK</h3>
  <p style="color:#ccc">Session closes at <strong>3:30 PM ET</strong>.
  {"Max trades reached — no more entries today." if trade_count >= 3 else f"{trades_rem} trade slot{'s' if trades_rem != 1 else ''} remaining."}</p>

  <hr style="border-color:#333;margin-top:20px">
  <p style="font-size:10px;color:#555">EOD summary will be sent at 4:05 PM ET</p>
</body></html>"""

    _smtp_send(u, p, to, subject, html)


def send_daily_report(trade_day_n: int = 1):
    """
    Compile today's data and send the EOD email report (4:05 PM ET).

    Args:
        trade_day_n: The current day number in the 20-day evaluation period.
    """
    u, p, to = _get_email_creds()
    if not u:
        return

    summary       = get_today_summary()
    trades        = get_recent_trades(n=20)
    gates         = get_gate_status()
    risk_log      = get_risk_log(n=30)
    anomalies     = get_recent_anomalies(n=10)
    system_status = get_latest_status()
    near_misses   = get_today_near_misses()

    pnl       = summary.get("gross_pnl", 0.0) if summary else 0.0
    today_str = date.today().isoformat()
    subject   = f"ATS Daily | {today_str} | P&L: ${pnl:+.0f} | Day {trade_day_n}/20"

    html_body = _build_html(summary, trades, gates, risk_log, anomalies, system_status, trade_day_n, near_misses)
    _smtp_send(u, p, to, subject, html_body)


def _build_signal_readiness_section(near_misses: list) -> str:
    """Build the SIGNAL READINESS HTML block for the daily email."""
    if not near_misses:
        return """
    <h3>SIGNAL READINESS</h3>
    <p style="color:#888">No near-miss data recorded today.
    Either no bars were close to triggering, or the pipeline did not run.
    Check the System Check panel on the dashboard.</p>"""

    # Compute summary stats
    pcts          = [abs(r.get("percent_to_breakout") or 999) for r in near_misses]
    vols          = [r.get("volume_ratio") or 0.0 for r in near_misses]
    regime        = near_misses[0].get("regime", "Unknown")
    count         = len(near_misses)
    closest_pct   = min(pcts)
    strongest_vol = max(vols)
    reasons       = [r.get("blocked_reason", "") for r in near_misses]

    # Narrative
    if "regime_blocked" in reasons:
        narrative = f"A potential setup formed, but the regime filter ({regime}) blocked all entries."
    elif closest_pct < 0.05:
        narrative = "Price cleared breakout levels but volume confirmation was insufficient."
    elif closest_pct < 0.15:
        narrative = f"Price came very close to breakout ({closest_pct:.2f}% away). Best vol ratio: {strongest_vol:.2f}x."
    elif closest_pct < 0.3:
        narrative = f"No valid breakout formed today. Best setup came within {closest_pct:.2f}% of trigger."
    else:
        narrative = f"No meaningful breakout attempt observed. Pipeline checked {count} bars."

    return f"""
    <h3>SIGNAL READINESS</h3>
    <div>
      <div class="stat"><div class="stat-val">{closest_pct:.2f}%</div><div class="stat-label">Closest breakout distance</div></div>
      <div class="stat"><div class="stat-val">{strongest_vol:.2f}x</div><div class="stat-label">Strongest volume ratio</div></div>
      <div class="stat"><div class="stat-val">{regime}</div><div class="stat-label">Last regime</div></div>
      <div class="stat"><div class="stat-val">{count}</div><div class="stat-label">Near-miss bars</div></div>
    </div>
    <p style="color:#ccc;margin-top:8px">{narrative}</p>"""


def _build_html(summary, trades, gates, risk_log, anomalies, system_status, trade_day_n, near_misses=None) -> str:
    today_str = date.today().isoformat()
    pnl = summary.get("gross_pnl", 0.0) if summary else 0.0
    pnl_color = "#00c853" if pnl >= 0 else "#d50000"
    total_trades = summary.get("total_trades", 0) if summary else 0
    win_rate = summary.get("win_rate", 0.0) if summary else 0.0
    avg_r = summary.get("avg_r", 0.0) if summary else 0.0

    # ── Gate status section ───────────────────────────────────────────────────
    gate_rows = ""
    if gates:
        gate_defs = [
            ("Gate 1", "Cumulative Return", gates.get("gate1_return"), "> 0%", lambda v: v is not None and v > 0),
            ("Gate 2", "Rule Violations", gates.get("gate2_violations"), "< 3", lambda v: v is not None and v < 3),
            ("Gate 3", "Max Drawdown", gates.get("gate3_drawdown"), "> -$2800", lambda v: v is not None and v > -2800),
            ("Gate 4", "Win Rate", gates.get("gate4_winrate"), "> 38%", lambda v: v is not None and v > 0.38),
            ("Gate 5", "Avg Slippage", gates.get("gate5_slippage"), "< $0.05", lambda v: v is not None and v < 0.05),
        ]
        for name, label, value, target, check_fn in gate_defs:
            passed = check_fn(value)
            badge_color = "#00c853" if passed else "#d50000"
            badge_text = "PASS" if passed else "FAIL"
            val_str = f"{value:.4f}" if isinstance(value, float) else str(value or "—")
            gate_rows += f"""
            <tr>
                <td style="padding:6px 10px">{name}</td>
                <td style="padding:6px 10px">{label}</td>
                <td style="padding:6px 10px">{val_str}</td>
                <td style="padding:6px 10px">{target}</td>
                <td style="padding:6px 10px">
                    <span style="background:{badge_color};color:#fff;padding:2px 8px;border-radius:3px;font-size:11px">{badge_text}</span>
                </td>
            </tr>"""
    else:
        gate_rows = '<tr><td colspan="5" style="padding:6px 10px;color:#888">No gate data yet</td></tr>'

    # ── Trades section ────────────────────────────────────────────────────────
    trade_rows = ""
    today_str_date = today_str
    for t in trades[:10]:
        pnl_d = t.get("pnl_dollars") or 0
        r_val = t.get("pnl_r") or 0
        t_color = "#00c853" if pnl_d >= 0 else "#d50000"
        trade_rows += f"""
        <tr>
            <td style="padding:4px 8px;font-size:11px">{str(t.get("entry_time") or "")[:16]}</td>
            <td style="padding:4px 8px;font-size:11px">{t.get("symbol","SPY")}</td>
            <td style="padding:4px 8px;font-size:11px">{t.get("entry_price","")}</td>
            <td style="padding:4px 8px;font-size:11px">{t.get("exit_price","")}</td>
            <td style="padding:4px 8px;font-size:11px;color:{t_color}">${pnl_d:+.0f}</td>
            <td style="padding:4px 8px;font-size:11px">{r_val:.2f}R</td>
            <td style="padding:4px 8px;font-size:11px">{t.get("regime","—")}</td>
            <td style="padding:4px 8px;font-size:11px">{t.get("exit_reason","—")}</td>
        </tr>"""
    if not trade_rows:
        trade_rows = '<tr><td colspan="8" style="padding:6px;color:#888">No trades today</td></tr>'

    # ── Risk section ──────────────────────────────────────────────────────────
    approvals = sum(1 for r in risk_log if r.get("result") == "APPROVED")
    blocks = sum(1 for r in risk_log if r.get("result") == "BLOCKED")
    block_reasons = [r.get("reason", "") for r in risk_log if r.get("result") == "BLOCKED"]
    unique_reasons = list(dict.fromkeys(block_reasons))[:5]
    risk_reasons_html = "".join(f"<li style='color:#ff6d00'>{r}</li>" for r in unique_reasons) or "<li>None</li>"

    # ── Anomalies section ─────────────────────────────────────────────────────
    anomaly_rows = ""
    for a in anomalies:
        sev = a.get("severity", "INFO")
        sev_color = {"CRITICAL": "#d50000", "WARNING": "#ff6d00", "INFO": "#888"}.get(sev, "#888")
        anomaly_rows += f"""
        <tr>
            <td style="padding:4px 8px;font-size:11px">{str(a.get("detected_at",""))[:16]}</td>
            <td style="padding:4px 8px;font-size:11px;color:{sev_color}">{sev}</td>
            <td style="padding:4px 8px;font-size:11px">{a.get("event_type","—")}</td>
            <td style="padding:4px 8px;font-size:11px">{a.get("diagnosis","—")}</td>
        </tr>"""
    if not anomaly_rows:
        anomaly_rows = '<tr><td colspan="4" style="padding:6px;color:#888">No anomalies</td></tr>'

    # ── System section ────────────────────────────────────────────────────────
    status_str = system_status.get("status", "UNKNOWN") if system_status else "UNKNOWN"
    kill_active = system_status.get("kill_switch_active", False) if system_status else False
    regime_str = system_status.get("regime", "Unknown") if system_status else "Unknown"

    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="UTF-8">
  <style>{_SHARED_CSS}</style>
</head>
<body>
  <h2>APEX TRADING SYSTEM — Daily Report</h2>
  <p style="color:#888">Day <strong style="color:#e0e0e0">{trade_day_n}/20</strong> &nbsp;|&nbsp; {today_str}</p>

  <div>
    <div class="stat">
      <div class="stat-val" style="color:{pnl_color}">${pnl:+.0f}</div>
      <div class="stat-label">Today P&L</div>
    </div>
    <div class="stat">
      <div class="stat-val">{total_trades}</div>
      <div class="stat-label">Trades</div>
    </div>
    <div class="stat">
      <div class="stat-val">{win_rate:.1%}</div>
      <div class="stat-label">Win Rate</div>
    </div>
    <div class="stat">
      <div class="stat-val">{avg_r:.2f}R</div>
      <div class="stat-label">Avg R</div>
    </div>
  </div>

  <h3>GATE STATUS</h3>
  <table>
    <tr><th>Gate</th><th>Criteria</th><th>Current</th><th>Target</th><th>Status</th></tr>
    {gate_rows}
  </table>

  <h3>TODAY'S TRADES</h3>
  <table>
    <tr><th>Time</th><th>Symbol</th><th>Entry</th><th>Exit</th><th>P&L</th><th>R</th><th>Regime</th><th>Exit Reason</th></tr>
    {trade_rows}
  </table>

  <h3>RISK ENGINE</h3>
  <p>Approvals: <strong style="color:#00c853">{approvals}</strong> &nbsp;|&nbsp; Blocks: <strong style="color:#d50000">{blocks}</strong></p>
  <p style="color:#888;font-size:12px">Top block reasons:</p>
  <ul style="font-size:12px">{risk_reasons_html}</ul>

  {_build_signal_readiness_section(near_misses or [])}

  <h3>ANOMALIES</h3>
  <table>
    <tr><th>Time</th><th>Severity</th><th>Type</th><th>Diagnosis</th></tr>
    {anomaly_rows}
  </table>

  <h3>SYSTEM</h3>
  <p>Status: <strong>{status_str}</strong> &nbsp;|&nbsp; Kill Switch: <strong style="color:{'#d50000' if kill_active else '#00c853'}">{'ACTIVE' if kill_active else 'INACTIVE'}</strong></p>
  <p>Regime: <strong>{regime_str}</strong></p>
  <hr style="border-color:#333">
  <p style="font-size:10px;color:#555">Apex Trading System — automated paper trading report</p>
</body>
</html>"""
