#!/usr/bin/env python3
"""
7-day MES/SPY backtest simulation.

Generates calibrated synthetic SPY 1-min bars and runs them through the
actual MultiSessionStrategy + risk engine.

Real SPY characteristics used for calibration:
  - Price: ~$580
  - 1-min σ: 0.04-0.06% per bar  (annual 15% / √(252×390) ≈ 0.048%)
  - Strong trend day: +1.5% = +$8.70 net drift over 390 bars
  - Average daily range: $4-8  (~0.7-1.5%)
"""
import math
import sys
import pytz
import numpy as np
import pandas as pd
from datetime import datetime, time, timedelta

sys.path.insert(0, ".")
from worker.strategy import MultiSessionStrategy
from worker import risk

ET      = pytz.timezone("America/New_York")
BASE    = 580.0   # SPY base price ($)
MES_PV  = 5.0     # $ per MES point
SPY_X   = 10.0    # SPY × 10 ≈ ES points


# ── synthetic data ─────────────────────────────────────────────────────────────

def make_day(scenario: str, base_price: float = BASE, seed: int = 0) -> pd.DataFrame:
    """
    One trading day of realistic 1-min SPY OHLCV.

    Calibration:
      vol   = 1-min log-return σ  (0.0005 ≈ real SPY ~15% annual)
      drift = per-bar log-return mean
              trend day  +1.5% / 390 bars ≈ +0.000038/bar
              flat day   ≈ 0
    """
    rng = np.random.default_rng(seed)
    n   = 390  # 9:30–16:00

    cfgs = {
        # scenario:         drift       vol      notes
        "orb_long":      (+0.000025, 0.00035, True,   0.0),
        "bull_trend":    (+0.000038, 0.00048, False,  0.0),
        "choppy":        (+0.000002, 0.00065, False,  0.0),
        "bear_trend":    (-0.000038, 0.00048, False,  0.0),
        "slow_grind_up": (+0.000008, 0.00035, False,  0.000060),
        "volatile":      (-0.000010, 0.00095, False,  0.0),
        "orb_short":     (-0.000025, 0.00035, True,   0.0),
    }
    drift, vol, or_tight, pm_boost = cfgs[scenario]

    r = rng.normal(drift, vol, n)

    if scenario == "choppy":
        # mean-reversion — price oscillates around VWAP
        cumr = np.cumsum(r)
        r -= 0.12 * np.concatenate([[0], cumr[:-1]])

    if scenario == "slow_grind_up":
        r[:190] = rng.normal(0.000002, vol * 0.5, 190)   # flat morning
        r[190:] += pm_boost                               # afternoon push

    if or_tight:
        r[:15] = rng.normal(0, vol * 0.25, 15)           # tight OR
        r[15]  = drift * 35 if drift > 0 else drift * 35 # decisive breakout

    if scenario == "volatile":
        shocks = rng.choice(range(20, n), size=8, replace=False)
        r[shocks] += rng.choice([-1, 1], size=8) * vol * 3.5

    closes  = base_price * np.exp(np.cumsum(r))
    volumes = _vol_profile(n, rng)

    rows = []
    for i, c in enumerate(closes):
        bar_σ = abs(c) * vol * rng.uniform(0.5, 2.0)
        o     = closes[i - 1] if i > 0 else c
        h     = max(o, c) + bar_σ * rng.uniform(0.05, 0.45)
        lo    = min(o, c) - bar_σ * rng.uniform(0.05, 0.45)
        rows.append((o, h, lo, c, volumes[i]))

    idx = pd.date_range("2026-05-12 09:30:00", periods=n, freq="1min", tz=ET)
    return pd.DataFrame(rows, index=idx,
                        columns=["Open", "High", "Low", "Close", "Volume"])


def _vol_profile(n: int, rng) -> np.ndarray:
    t       = np.linspace(0, 1, n)
    profile = 0.5 + 1.5 * np.exp(-20 * t) + 0.8 * np.exp(-15 * (1 - t))
    return (600_000 * profile * rng.uniform(0.6, 1.4, n)).astype(float)


# ── position sim ──────────────────────────────────────────────────────────────

def simulate_day(df, scenario, equity, peak_equity, eval_pnl):
    strategy = MultiSessionStrategy()
    day_pnl  = 0.0
    n_trades = 0
    consec_L = 0
    position = None
    log      = []

    for i in range(15, len(df)):
        bar_time = df.index[i]
        t        = bar_time.time()
        if t > time(16, 0):
            break

        df_w = df.iloc[:i + 1]
        try:
            df_ind = strategy.compute_indicators(df_w)
        except Exception:
            continue

        price = float(df_ind["Close"].iloc[-1])

        # ── manage open position ─────────────────────────────────────────────
        if position is not None:
            d       = position["direction"]
            reason  = None
            exit_px = price     # default: fill at close

            if d == "LONG":
                if price <= position["stop"]:
                    reason  = "stop"
                    exit_px = position["stop"]   # fill at stop, not worse
                elif price >= position["target"]:
                    reason  = "target"
                    exit_px = position["target"] # fill at target, cap upside realistically
            else:
                if price >= position["stop"]:
                    reason  = "stop"
                    exit_px = position["stop"]
                elif price <= position["target"]:
                    reason  = "target"
                    exit_px = position["target"]

            if t >= time(15, 55) and reason is None:
                reason = "EOD"

            elapsed = (bar_time - position["entry_time"]).total_seconds() / 60
            if elapsed >= 90 and reason is None:
                reason  = "timeout"

            # multi-stage trail (uses initial dist; 2R lock-in for A+ 3R targets)
            if reason is None:
                entry    = position["entry"]
                D        = position["stop_dist"]
                target_r = position.get("target_r", 2.0)
                if D > 0:
                    if d == "LONG":
                        r = (price - entry) / D
                        lock_15R = round(entry + 1.5 * D, 2)
                        lock_05R = round(entry + 0.5 * D, 2)
                        if target_r >= 3.0 and r >= 2.0 and position["stop"] < lock_15R:
                            position["stop"] = lock_15R
                        elif r >= 1.5 and position["stop"] < lock_05R:
                            position["stop"] = lock_05R
                        elif r >= 1.0 and position["stop"] < entry:
                            position["stop"] = round(entry + 0.02, 2)
                    else:
                        r = (entry - price) / D
                        lock_15R = round(entry - 1.5 * D, 2)
                        lock_05R = round(entry - 0.5 * D, 2)
                        if target_r >= 3.0 and r >= 2.0 and position["stop"] > lock_15R:
                            position["stop"] = lock_15R
                        elif r >= 1.5 and position["stop"] > lock_05R:
                            position["stop"] = lock_05R
                        elif r >= 1.0 and position["stop"] > entry:
                            position["stop"] = round(entry - 0.02, 2)

            if reason is not None:
                c  = position["contracts"]
                D  = position["stop_dist"]
                if d == "LONG":
                    pnl = (exit_px - position["entry"]) * SPY_X * MES_PV * c
                else:
                    pnl = (position["entry"] - exit_px) * SPY_X * MES_PV * c

                day_pnl     += pnl
                equity      += pnl
                peak_equity  = max(peak_equity, equity)
                n_trades    += 1
                consec_L     = 0 if pnl > 0 else consec_L + 1

                r_mult = round(pnl / (D * SPY_X * MES_PV * c), 2) \
                         if D > 0 and c > 0 else 0.0
                log.append({
                    "time":      bar_time.strftime("%H:%M"),
                    "dir":       d[0],
                    "strategy":  position["strategy"],
                    "grade":     position.get("grade", "?"),
                    "entry":     position["entry"],
                    "exit":      exit_px,
                    "contracts": c,
                    "pnl":       pnl,
                    "r":         r_mult,
                    "reason":    reason,
                })
                position = None
            continue

        # ── seek new signal ──────────────────────────────────────────────────
        rc = risk.pre_trade_check(
            daily_pnl=day_pnl,
            trade_count=n_trades,
            time_et=bar_time,
            consecutive_losses=consec_L,
            eval_pnl=eval_pnl + day_pnl,
        )
        if not rc["approved"]:
            continue

        try:
            sig = strategy.generate_signals(
                df_ind, bar_time, n_trades,
                daily_pnl=day_pnl,
                current_equity=equity,
                peak_equity=peak_equity,
            )
        except Exception:
            continue

        if sig is None:
            continue

        position = {
            "direction":  sig["direction"],
            "entry":      sig["price"],
            "stop":       sig["stop"],
            "target":     sig["target"],
            "contracts":  sig["contracts"],
            "stop_dist":  sig["stop_distance"],
            "entry_time": bar_time,
            "strategy":   sig.get("strategy", "?"),
            "grade":      sig.get("grade", "?"),
            "target_r":   sig.get("target_r", 2.0),
            "score":      sig.get("score", 0),
        }

    # force-close leftover
    if position is not None:
        exit_px = float(df["Close"].iloc[-1])
        d  = position["direction"]
        c  = position["contracts"]
        D  = position["stop_dist"]
        pnl = (exit_px - position["entry"]) * SPY_X * MES_PV * c if d == "LONG" \
              else (position["entry"] - exit_px) * SPY_X * MES_PV * c
        day_pnl += pnl
        equity  += pnl
        n_trades += 1
        r_mult = round(pnl / (D * SPY_X * MES_PV * c), 2) if D > 0 and c > 0 else 0.0
        log.append({
            "time": "16:00", "dir": d[0], "strategy": position["strategy"],
            "grade": position.get("grade", "?"),
            "entry": position["entry"], "exit": exit_px, "contracts": c,
            "pnl": pnl, "r": r_mult, "reason": "EOD",
        })

    return {"scenario": scenario, "day_pnl": day_pnl, "equity": equity,
            "peak": peak_equity, "trades": log, "n_trades": n_trades}


# ── main ───────────────────────────────────────────────────────────────────────

WEEK = [
    ("Mon May-11", "orb_long",      10),
    ("Tue May-12", "bull_trend",    11),
    ("Wed May-13", "choppy",        12),
    ("Thu May-14", "bear_trend",    13),
    ("Fri May-15", "slow_grind_up", 14),
    ("Mon May-18", "volatile",      15),
    ("Tue May-19", "orb_short",     16),
]
LABELS = {
    "orb_long":      "ORB long break",
    "bull_trend":    "Bull trend day",
    "choppy":        "Choppy/chop",
    "bear_trend":    "Bear trend day",
    "slow_grind_up": "Flat AM→PM rally",
    "volatile":      "Volatile/whipsaw",
    "orb_short":     "ORB short break",
}

equity      = 10_000.0
peak_equity = equity
cumulative  = 0.0
all_trades  = []
daily_pnls  = []

W = 80
print("═" * W)
print(f"{'  7-DAY MES SIMULATION  —  Calibrated Synthetic SPY':^{W}}")
print(f"{'  $10K account  |  1 MES = $5/pt  |  SPY×10=ES pts  |  shorts enabled':^{W}}")
print("═" * W)

for label, scenario, seed in WEEK:
    # Use previous day close as base price (accounts carry over)
    day_base = equity / 10_000.0 * BASE if equity > 0 else BASE
    df  = make_day(scenario, base_price=day_base, seed=seed)
    res = simulate_day(df, scenario, equity, peak_equity, equity - 10_000.0)

    equity      = res["equity"]
    peak_equity = res["peak"]
    cumulative += res["day_pnl"]
    daily_pnls.append(res["day_pnl"])
    all_trades.extend(res["trades"])

    sign_ch = "▲" if res["day_pnl"] >= 0 else "▼"
    print(f"\n{'─' * W}")
    print(f"  {label:<14}  {LABELS[scenario]:<20}  "
          f"P&L: {'+' if res['day_pnl'] >= 0 else ''}{res['day_pnl']:>7.0f}   "
          f"Equity: ${equity:>9,.0f}  {sign_ch}")
    print(f"{'─' * W}")

    if not res["trades"]:
        print("    (no trades — filtered by trend/grade/RSI/volume/chop/risk)")
    else:
        print(f"  {'Time':>5}  {'D':1}  {'Grade':>3}  {'Strategy':<14} {'Entry':>7} {'Exit':>7} "
              f"{'Ctrs':>4} {'P&L':>9} {'R':>5}  Exit reason")
        for t in res["trades"]:
            row = (f"  {t['time']:>5}  {t['dir']:1}  {t.get('grade','?'):>3}  "
                   f"{t['strategy']:<14} "
                   f"{t['entry']:>7.2f} {t['exit']:>7.2f} "
                   f"{t['contracts']:>4} ${t['pnl']:>+8.0f} "
                   f"{t['r']:>+5.2f}R  {t['reason']}")
            print(row)


# ── summary ────────────────────────────────────────────────────────────────────
wins   = [t for t in all_trades if t["pnl"] > 0]
losses = [t for t in all_trades if t["pnl"] <= 0]
total  = len(all_trades)

best_day  = max(daily_pnls) if daily_pnls else 0
worst_day = min(daily_pnls) if daily_pnls else 0

print(f"\n{'═' * W}")
print(f"{'  SIMULATION SUMMARY':^{W}}")
print(f"{'═' * W}")
print(f"  Starting equity      : $10,000")
print(f"  Ending equity        : ${equity:>10,.2f}")
print(f"  7-day net P&L        : ${cumulative:>+10.2f}")
print(f"  Best day             : ${best_day:>+.0f}")
print(f"  Worst day            : ${worst_day:>+.0f}")

if total:
    wr       = 100 * len(wins) / total
    avg_win  = float(np.mean([t["pnl"] for t in wins]))  if wins   else 0.0
    avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
    avg_r    = float(np.mean([t["r"]   for t in all_trades]))
    pf       = abs(len(wins) * avg_win / (len(losses) * avg_loss)) \
               if losses and avg_loss != 0 else float("inf")

    print(f"  Total trades         : {total}  ({total/7:.1f}/day avg)")
    print(f"  Win rate             : {len(wins)}/{total}  ({wr:.0f}%)")
    print(f"  Avg winner           : ${avg_win:>+.0f}")
    print(f"  Avg loser            : ${avg_loss:>+.0f}")
    print(f"  Profit factor        : {pf:.2f}")
    print(f"  Avg R per trade      : {avg_r:>+.2f}R")

    # by strategy
    by_s: dict = {}
    for t in all_trades:
        s = t["strategy"]
        by_s.setdefault(s, {"n": 0, "pnl": 0.0, "w": 0})
        by_s[s]["n"]   += 1
        by_s[s]["pnl"] += t["pnl"]
        by_s[s]["w"]   += 1 if t["pnl"] > 0 else 0
    print(f"\n  ── By strategy ──")
    for s, d in sorted(by_s.items(), key=lambda x: -x[1]["pnl"]):
        swr = 100 * d["w"] / d["n"]
        print(f"    {s:<16}  {d['n']:>2} trades  "
              f"{d['w']}/{d['n']} wins ({swr:.0f}%)  ${d['pnl']:>+.0f}")

    # by grade
    by_g: dict = {}
    for t in all_trades:
        g = t.get("grade", "?")
        by_g.setdefault(g, {"n": 0, "pnl": 0.0, "w": 0})
        by_g[g]["n"]   += 1
        by_g[g]["pnl"] += t["pnl"]
        by_g[g]["w"]   += 1 if t["pnl"] > 0 else 0
    print(f"\n  ── By grade ──")
    for g in ("A+", "A", "B"):
        if g in by_g:
            d = by_g[g]
            gwr = 100 * d["w"] / d["n"]
            print(f"    {g:<3}              {d['n']:>2} trades  "
                  f"{d['w']}/{d['n']} wins ({gwr:.0f}%)  ${d['pnl']:>+.0f}")
else:
    print("  Total trades         : 0  (no signals generated)")

ret_7d    = (equity / 10_000.0 - 1) * 100
ret_month = ret_7d * (21.0 / 7.0)
ret_year  = ((1 + ret_month / 100) ** 12 - 1) * 100

print(f"\n  ── Projections (linear scale-out) ──")
print(f"    7-day return       : {ret_7d:>+.2f}%")
print(f"    Monthly (×3)       : {ret_month:>+.1f}%  on $10K = ${10_000 * ret_month / 100:>+.0f}/mo")
print(f"    Annual (compounded): {ret_year:>+.1f}%")
print("═" * W)
print()
print("  NOTE: Simulation uses calibrated synthetic data (σ≈0.05%/bar, matching")
print("  real SPY 1-min vol). Entry fills at signal close; stops and targets fill")
print("  at the trigger price. No transaction costs. Slippage not modeled.")
print("═" * W)
