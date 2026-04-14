import React, { useState, useEffect, useCallback } from 'react';

// ── Config ────────────────────────────────────────────────────────────────────
// Empty string = same origin (API and dashboard served from the same Railway service).
// Override only for local dev: set REACT_APP_API_URL=http://localhost:8000 in .env
const API_URL =
  process.env.REACT_APP_API_URL ||
  localStorage.getItem('apex_api_url') ||
  '';

const REFRESH_INTERVAL = 30_000; // 30 seconds

// ── Sub-components ────────────────────────────────────────────────────────────
function Badge({ text, color }) {
  const colors = {
    green: 'bg-green-500 text-black',
    red: 'bg-red-600 text-white',
    yellow: 'bg-yellow-500 text-black',
    gray: 'bg-gray-600 text-white',
    blue: 'bg-blue-600 text-white',
  };
  return (
    <span className={`text-xs font-bold px-2 py-0.5 rounded ${colors[color] || colors.gray}`}>
      {text}
    </span>
  );
}

function StatusBar({ status, lastRefresh, onRefresh }) {
  const sysStatus = status?.status || 'UNKNOWN';
  const statusColor = sysStatus === 'RUNNING' ? 'green' : sysStatus === 'PAUSED' ? 'yellow' : 'red';
  const killActive = status?.kill_switch_active;

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 mb-3 flex flex-wrap items-center gap-2 justify-between">
      <div className="flex items-center gap-3">
        <span className="text-green-400 font-bold text-sm tracking-widest">APEX</span>
        <Badge text={sysStatus} color={statusColor} />
        {killActive && <Badge text="KILL SWITCH" color="red" />}
      </div>
      <div className="flex items-center gap-3 text-xs text-gray-400">
        <span>Last: {lastRefresh || '—'}</span>
        <button
          onClick={onRefresh}
          className="text-gray-500 hover:text-green-400 transition-colors border border-gray-700 px-2 py-0.5 rounded"
        >
          ↻ REFRESH
        </button>
      </div>
    </div>
  );
}

function StatCard({ title, value, sub, color }) {
  const textColor = color === 'green' ? 'text-green-400'
    : color === 'red' ? 'text-red-400'
    : color === 'yellow' ? 'text-yellow-400'
    : 'text-gray-100';

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <div className={`text-lg font-bold ${textColor} font-mono`}>{value ?? '—'}</div>
      <div className="text-xs text-gray-500 mt-1">{title}</div>
      {sub && <div className="text-xs text-gray-600 mt-0.5">{sub}</div>}
    </div>
  );
}

function GateTracker({ gates, status }) {
  const dayNum = gates?.day_number ?? 1;
  const gateDefs = [
    { key: 'gate1_return',    label: 'Return',      target: '> 0%',     fmt: v => v != null ? `${(v * 100).toFixed(1)}%` : '—', pass: v => v > 0 },
    { key: 'gate2_violations',label: 'Violations',  target: '< 3',      fmt: v => v != null ? String(v) : '—',                  pass: v => v < 3 },
    { key: 'gate3_drawdown',  label: 'Drawdown',    target: '> -$2800', fmt: v => v != null ? `$${v.toFixed(0)}` : '—',         pass: v => v > -2800 },
    { key: 'gate4_winrate',   label: 'Win Rate',    target: '> 38%',    fmt: v => v != null ? `${(v * 100).toFixed(1)}%` : '—', pass: v => v > 0.38 },
    { key: 'gate5_slippage',  label: 'Slippage',    target: '< $0.05',  fmt: v => v != null ? `$${v.toFixed(3)}` : '—',         pass: v => v < 0.05 },
  ];

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 mb-3">
      <div className="flex justify-between items-center mb-2">
        <span className="text-xs text-gray-400 font-bold tracking-wider">GATE TRACKER</span>
        <span className="text-xs text-blue-400">Day {dayNum} of 20</span>
      </div>
      <div className="space-y-1.5">
        {gateDefs.map(({ key, label, target, fmt, pass }) => {
          const val = gates?.[key];
          const passed = val != null ? pass(val) : null;
          return (
            <div key={key} className="flex items-center justify-between text-xs">
              <span className="text-gray-400 w-24">{label}</span>
              <span className="text-gray-300 w-16 text-right font-mono">{fmt(val)}</span>
              <span className="text-gray-600 w-20 text-right">{target}</span>
              <span className="w-12 text-right">
                {passed === null ? (
                  <span className="text-gray-600">—</span>
                ) : passed ? (
                  <span className="text-green-400 font-bold">PASS</span>
                ) : (
                  <span className="text-red-400 font-bold">FAIL</span>
                )}
              </span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function TradesTable({ trades }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <div className="text-xs text-gray-400 font-bold tracking-wider mb-2">RECENT TRADES</div>
      <div className="overflow-x-auto">
        <table className="w-full text-xs font-mono">
          <thead>
            <tr className="text-gray-500 border-b border-gray-800">
              <th className="text-left pb-1 pr-2">Time</th>
              <th className="text-right pb-1 pr-2">Entry</th>
              <th className="text-right pb-1 pr-2">Exit</th>
              <th className="text-right pb-1 pr-2">P&L</th>
              <th className="text-right pb-1 pr-2">R</th>
              <th className="text-left pb-1">Reason</th>
            </tr>
          </thead>
          <tbody>
            {trades && trades.length > 0 ? trades.slice(0, 15).map((t, i) => {
              const pnl = t.pnl_dollars ?? 0;
              const pnlColor = pnl >= 0 ? 'text-green-400' : 'text-red-400';
              const timeStr = t.exit_time ? String(t.exit_time).slice(11, 16) : '—';
              return (
                <tr key={i} className="border-b border-gray-800/50 hover:bg-gray-800/30">
                  <td className="py-1 pr-2 text-gray-500">{timeStr}</td>
                  <td className="py-1 pr-2 text-right">{t.entry_price?.toFixed(2) ?? '—'}</td>
                  <td className="py-1 pr-2 text-right">{t.exit_price?.toFixed(2) ?? '—'}</td>
                  <td className={`py-1 pr-2 text-right font-bold ${pnlColor}`}>
                    {pnl >= 0 ? '+' : ''}{pnl.toFixed(0)}
                  </td>
                  <td className="py-1 pr-2 text-right text-gray-400">{(t.pnl_r ?? 0).toFixed(2)}R</td>
                  <td className="py-1 text-gray-500 truncate max-w-24">{t.exit_reason ?? '—'}</td>
                </tr>
              );
            }) : (
              <tr>
                <td colSpan={6} className="py-3 text-center text-gray-600">No trades yet</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

function RiskLogTable({ riskLog }) {
  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3">
      <div className="text-xs text-gray-400 font-bold tracking-wider mb-2">RISK LOG</div>
      <div className="overflow-y-auto max-h-48 space-y-1">
        {riskLog && riskLog.length > 0 ? riskLog.slice(0, 15).map((r, i) => {
          const approved = r.result === 'APPROVED';
          return (
            <div key={i} className="flex items-start gap-2 text-xs border-b border-gray-800/40 pb-1">
              <span className={`shrink-0 font-bold ${approved ? 'text-green-400' : 'text-red-400'}`}>
                {approved ? 'OK' : 'BLK'}
              </span>
              <span className="text-gray-500 truncate">{r.reason ?? r.check_type}</span>
            </div>
          );
        }) : (
          <div className="text-gray-600 text-xs py-2">No risk checks yet</div>
        )}
      </div>
    </div>
  );
}

function AnomaliesFeed({ anomalies }) {
  const severityColor = sev => ({
    CRITICAL: 'text-red-400 border-red-900',
    WARNING: 'text-yellow-400 border-yellow-900',
    INFO: 'text-gray-400 border-gray-800',
  }[sev] || 'text-gray-400 border-gray-800');

  return (
    <div className="bg-gray-900 border border-gray-700 rounded p-3 mb-3">
      <div className="text-xs text-gray-400 font-bold tracking-wider mb-2">ANOMALIES</div>
      <div className="space-y-2">
        {anomalies && anomalies.length > 0 ? anomalies.slice(0, 5).map((a, i) => (
          <div key={i} className={`border-l-2 pl-2 ${severityColor(a.severity)}`}>
            <div className="flex gap-2 items-center text-xs">
              <span className="font-bold">{a.severity}</span>
              <span className="text-gray-500">{String(a.detected_at || '').slice(0, 16)}</span>
            </div>
            <div className="text-xs text-gray-400">{a.event_type}</div>
            <div className="text-xs text-gray-600 truncate">{a.diagnosis}</div>
          </div>
        )) : (
          <div className="text-gray-600 text-xs">No anomalies detected</div>
        )}
      </div>
    </div>
  );
}

function RawStatusLog({ status }) {
  return (
    <details className="mt-3">
      <summary className="text-xs text-gray-600 cursor-pointer hover:text-gray-400 select-none">
        ▶ Raw system status
      </summary>
      <pre className="mt-2 bg-gray-950 border border-gray-800 rounded p-2 text-xs text-gray-500 overflow-auto max-h-40 font-mono">
        {JSON.stringify(status, null, 2)}
      </pre>
    </details>
  );
}

// ── Login screen ──────────────────────────────────────────────────────────────
function LoginScreen({ onLogin }) {
  const [secret, setSecret] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    localStorage.setItem('apex_secret', secret);
    onLogin(secret);
  };

  return (
    <div className="min-h-screen bg-gray-950 flex items-center justify-center p-4">
      <div className="bg-gray-900 border border-gray-700 rounded-lg p-8 w-full max-w-sm">
        <h1 className="text-green-400 font-bold text-xl font-mono tracking-widest mb-1">APEX</h1>
        <p className="text-gray-500 text-xs mb-6">Trading System Dashboard</p>
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="text-xs text-gray-400 block mb-1">Dashboard Password</label>
            <input
              type="password"
              value={secret}
              onChange={e => setSecret(e.target.value)}
              className="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm text-gray-200 font-mono focus:outline-none focus:border-green-500"
              placeholder="Enter your DASHBOARD_SECRET..."
              autoFocus
            />
          </div>
          <button
            type="submit"
            className="w-full bg-green-600 hover:bg-green-500 text-black font-bold py-2 rounded text-sm transition-colors"
          >
            ACCESS DASHBOARD
          </button>
        </form>
      </div>
    </div>
  );
}

// ── Main App ──────────────────────────────────────────────────────────────────
export default function App() {
  const [secret, setSecret] = useState(() => localStorage.getItem('apex_secret') || '');
  const [status, setStatus] = useState(null);
  const [summary, setSummary] = useState(null);
  const [trades, setTrades] = useState([]);
  const [gates, setGates] = useState(null);
  const [riskLog, setRiskLog] = useState([]);
  const [anomalies, setAnomalies] = useState([]);
  const [lastRefresh, setLastRefresh] = useState(null);
  const [error, setError] = useState(null);

  const effectiveApiUrl =
    localStorage.getItem('apex_api_url') || process.env.REACT_APP_API_URL || 'http://localhost:8000';

  const fetchAll = useCallback(async () => {
    if (!secret) return;
    const headers = { 'X-Dashboard-Secret': secret };

    const safe = async (url) => {
      try {
        const res = await fetch(url, { headers });
        if (!res.ok) return null;
        return await res.json();
      } catch {
        return null;
      }
    };

    const [s, sum, t, g, r, a] = await Promise.all([
      safe(`${effectiveApiUrl}/api/status`),
      safe(`${effectiveApiUrl}/api/summary`),
      safe(`${effectiveApiUrl}/api/trades?limit=50`),
      safe(`${effectiveApiUrl}/api/gates`),
      safe(`${effectiveApiUrl}/api/risk-log?limit=20`),
      safe(`${effectiveApiUrl}/api/anomalies?limit=10`),
    ]);

    if (s) setStatus(s);
    if (sum) setSummary(sum);
    if (t) setTrades(Array.isArray(t) ? t : []);
    if (g) setGates(g);
    if (r) setRiskLog(Array.isArray(r) ? r : []);
    if (a) setAnomalies(Array.isArray(a) ? a : []);

    setLastRefresh(new Date().toLocaleTimeString('en-US', { timeZone: 'America/New_York', hour12: false }));
    setError(null);
  }, [secret, effectiveApiUrl]);

  useEffect(() => {
    if (!secret) return;
    fetchAll();
    const timer = setInterval(fetchAll, REFRESH_INTERVAL);
    return () => clearInterval(timer);
  }, [secret, fetchAll]);

  if (!secret) {
    return <LoginScreen onLogin={s => setSecret(s)} />;
  }

  // ── Derived values ──────────────────────────────────────────────────────────
  const pnl = summary?.gross_pnl ?? 0;
  const pnlColor = pnl >= 0 ? 'green' : 'red';
  const pnlStr = `${pnl >= 0 ? '+' : ''}$${Math.abs(pnl).toFixed(0)}`;

  const dailyLossLimit = 1500;
  const ddPct = status?.daily_pnl != null
    ? Math.min(100, Math.abs(Math.min(0, status.daily_pnl)) / dailyLossLimit * 100).toFixed(0)
    : '0';
  const ddStr = `$${Math.abs(Math.min(0, status?.daily_pnl ?? 0)).toFixed(0)} of $${dailyLossLimit}`;

  const winRate = summary?.win_rate;
  const winRateStr = winRate != null ? `${(winRate * 100).toFixed(1)}%` : '—';
  const winRateColor = winRate != null ? (winRate > 0.38 ? 'green' : 'red') : 'gray';

  const regime = status?.regime || 'Unknown';
  const regimeColor = regime === 'Strong Trend' ? 'green'
    : regime === 'Extreme Volatility' ? 'red'
    : regime === 'Range-Bound' ? 'yellow'
    : 'gray';

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100 font-mono p-2 md:p-4 max-w-2xl mx-auto">

      {/* Status bar */}
      <StatusBar status={status} lastRefresh={lastRefresh} onRefresh={fetchAll} />

      {/* 4 Stat cards */}
      <div className="grid grid-cols-2 gap-2 mb-3">
        <StatCard
          title="Today P&L"
          value={pnlStr}
          color={pnlColor}
          sub={`${summary?.total_trades ?? 0} trades`}
        />
        <StatCard
          title="Drawdown"
          value={ddStr}
          color={Number(ddPct) > 75 ? 'red' : Number(ddPct) > 50 ? 'yellow' : 'green'}
          sub={`${ddPct}% of limit`}
        />
        <StatCard
          title="Win Rate"
          value={winRateStr}
          color={winRateColor}
          sub={`Baseline: 38%`}
        />
        <StatCard
          title="Regime"
          value={regime}
          color={regimeColor}
          sub={status?.status || '—'}
        />
      </div>

      {/* Gate tracker */}
      <GateTracker gates={gates} status={status} />

      {/* Trades + Risk log (stacked on mobile) */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-2 mb-3">
        <TradesTable trades={trades} />
        <RiskLogTable riskLog={riskLog} />
      </div>

      {/* Anomalies */}
      <AnomaliesFeed anomalies={anomalies} />

      {/* Raw status log (collapsible) */}
      <RawStatusLog status={status} />

      {/* Logout */}
      <div className="mt-4 text-center">
        <button
          onClick={() => { localStorage.removeItem('apex_secret'); setSecret(''); }}
          className="text-xs text-gray-700 hover:text-gray-500 transition-colors"
        >
          sign out
        </button>
      </div>
    </div>
  );
}
