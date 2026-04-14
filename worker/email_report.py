"""
worker/email_report.py — Daily email report for Apex Trading System.

Sends via Gmail SMTP using GMAIL_USER + GMAIL_APP_PASSWORD.
Graceful degradation: logs a warning and returns if env vars are missing.

Subject: ATS Daily | {date} | P&L: ${pnl} | Day {n}/20
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
)


def send_daily_report(trade_day_n: int = 1):
    """
    Compile today's data and send the daily email report.

    Args:
        trade_day_n: The current day number in the 20-day evaluation period.
    """
    gmail_user = os.getenv("GMAIL_USER")
    gmail_password = os.getenv("GMAIL_APP_PASSWORD")
    notify_email = os.getenv("NOTIFY_EMAIL", gmail_user)

    if not gmail_user or not gmail_password:
        logger.warning("GMAIL_USER or GMAIL_APP_PASSWORD not set — skipping email report")
        return

    # Gather data
    summary = get_today_summary()
    trades = get_recent_trades(n=20)
    gates = get_gate_status()
    risk_log = get_risk_log(n=30)
    anomalies = get_recent_anomalies(n=10)
    system_status = get_latest_status()

    pnl = summary.get("gross_pnl", 0.0) if summary else 0.0
    today_str = date.today().isoformat()

    subject = f"ATS Daily | {today_str} | P&L: ${pnl:+.0f} | Day {trade_day_n}/20"

    html_body = _build_html(summary, trades, gates, risk_log, anomalies, system_status, trade_day_n)

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_user
    msg["To"] = notify_email
    msg.attach(MIMEText(html_body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
            server.ehlo()
            server.starttls()
            server.login(gmail_user, gmail_password)
            server.sendmail(gmail_user, notify_email, msg.as_string())
        logger.info(f"Daily report sent to {notify_email} — {subject}")
    except Exception as e:
        logger.error(f"Failed to send daily email: {e}")


def _build_html(summary, trades, gates, risk_log, anomalies, system_status, trade_day_n) -> str:
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
  <style>
    body {{ font-family: 'Courier New', monospace; background: #0d0d0d; color: #e0e0e0; margin: 0; padding: 20px; }}
    h2 {{ color: #00e676; border-bottom: 1px solid #333; padding-bottom: 8px; }}
    h3 {{ color: #82b1ff; margin-top: 24px; }}
    table {{ width: 100%; border-collapse: collapse; margin: 8px 0; }}
    th {{ background: #1a1a2e; color: #82b1ff; padding: 8px 10px; text-align: left; font-size: 12px; }}
    tr:nth-child(even) {{ background: #111; }}
    .stat {{ display: inline-block; background: #1a1a1a; border: 1px solid #333; padding: 8px 16px; margin: 4px; border-radius: 4px; }}
    .stat-val {{ font-size: 20px; font-weight: bold; }}
    .stat-label {{ font-size: 11px; color: #888; }}
  </style>
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
