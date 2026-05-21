#!/usr/bin/env python3
"""
21-trading-day (1 month) MES/SPY backtest using the actual MultiSessionStrategy
+ risk engine.

What it answers:
  - If I deploy today and let it run for a month, what do I see?
  - When (if ever) does the $3,000 funded eval target get hit?
  - What's the worst day? Worst drawdown? Any rule violations?
  - How many days hit profit-lock? How many hit the daily loss cap?
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
from simulate import make_day, simulate_day, BASE, SPY_X, MES_PV

ET = pytz.timezone("America/New_York")

# Realistic month mix — ~21 trading days
# Distribution roughly mirrors actual market regimes:
#   ~30% trend (bull/bear), ~25% choppy, ~15% volatile, ~15% ORB break,
#   ~15% slow grind. This is intentionally NOT cherry-picked.
MONTH = [
    # week 1
    ("D01 Mon", "orb_long",      101),
    ("D02 Tue", "bull_trend",    102),
    ("D03 Wed", "choppy",        103),
    ("D04 Thu", "slow_grind_up", 104),
    ("D05 Fri", "bear_trend",    105),
    # week 2
    ("D06 Mon", "volatile",      106),
    ("D07 Tue", "orb_short",     107),
    ("D08 Wed", "bull_trend",    108),
    ("D09 Thu", "choppy",        109),
    ("D10 Fri", "slow_grind_up", 110),
    # week 3
    ("D11 Mon", "bear_trend",    111),
    ("D12 Tue", "orb_long",      112),
    ("D13 Wed", "volatile",      113),
    ("D14 Thu", "bull_trend",    114),
    ("D15 Fri", "choppy",        115),
    # week 4
    ("D16 Mon", "slow_grind_up", 116),
    ("D17 Tue", "orb_short",     117),
    ("D18 Wed", "bear_trend",    118),
    ("D19 Thu", "bull_trend",    119),
    ("D20 Fri", "choppy",        120),
    # week 5 (partial)
    ("D21 Mon", "orb_long",      121),
]

LABELS = {
    "orb_long":      "ORB long",
    "bull_trend":    "Bull trend",
    "choppy":        "Choppy",
    "bear_trend":    "Bear trend",
    "slow_grind_up": "Flat→PM rally",
    "volatile":      "Volatile",
    "orb_short":     "ORB short",
}


def fmt_money(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}${x:,.0f}"


def main():
    equity      = 50_000.0
    peak_equity = equity
    eval_pnl    = 0.0     # cumulative for funded account tracker
    all_trades  = []
    daily       = []      # list of (label, scenario, day_pnl, equity, eval_pnl)

    target_hit_day = None
    blow_day       = None

    W = 92
    print("═" * W)
    print(f"{'  21-DAY (1 MONTH) FUNDED ACCOUNT SIMULATION':^{W}}")
    print(f"{'  Target: +$3,000  |  Trailing DD limit: -$2,500  |  Consistency cap: $570/day':^{W}}")
    print(f"{'  $50K paper equity  |  MES = $5/pt  |  SPY×10 proxy  |  5× sizing  |  shorts enabled':^{W}}")
    print("═" * W)
    print()
    print(f"  {'Day':<8} {'Scenario':<16} {'Trades':>6} {'Day P&L':>10} {'Eval P&L':>10} {'Equity':>10}  Status")
    print(f"  {'-'*8} {'-'*16} {'-'*6} {'-'*10} {'-'*10} {'-'*10}  {'-'*40}")

    for label, scenario, seed in MONTH:
        # If eval target already hit, bot stops trading (risk check 0)
        df  = make_day(scenario, base_price=BASE, seed=seed)
        res = simulate_day(df, scenario, equity, peak_equity, eval_pnl)

        day_pnl  = res["day_pnl"]
        equity   = res["equity"]
        peak_equity = res["peak"]
        eval_pnl += day_pnl

        # status flags
        status = []
        if eval_pnl >= 3_000 and target_hit_day is None:
            target_hit_day = label
            status.append("TARGET HIT!")
        if equity - peak_equity <= -2_500 and blow_day is None:
            blow_day = label
            status.append("DD BREACH")
        if day_pnl >= 570:
            status.append("consistency cap")
        if day_pnl <= -500:
            status.append("daily loss cap")
        status_str = " ".join(status) if status else ""

        sign = "▲" if day_pnl >= 0 else ("▼" if day_pnl < 0 else "·")

        print(f"  {label:<8} {LABELS[scenario]:<16} {res['n_trades']:>6} "
              f"{fmt_money(day_pnl):>10} {fmt_money(eval_pnl):>10} ${equity:>9,.0f}  {sign} {status_str}")

        all_trades.extend(res["trades"])
        daily.append((label, scenario, day_pnl, equity, eval_pnl, res["n_trades"]))

    # ── analysis ─────────────────────────────────────────────────────────────
    daily_pnls = [d[2] for d in daily]
    eval_curve = [d[4] for d in daily]
    eq_curve   = [d[3] for d in daily]

    wins   = [t for t in all_trades if t["pnl"] > 0]
    losses = [t for t in all_trades if t["pnl"] <= 0]
    total  = len(all_trades)

    # max drawdown from running peak in eval P&L
    running_peak = 0.0
    max_dd       = 0.0
    for ep in eval_curve:
        running_peak = max(running_peak, ep)
        dd = ep - running_peak
        max_dd = min(max_dd, dd)

    best_day  = max(daily_pnls) if daily_pnls else 0
    worst_day = min(daily_pnls) if daily_pnls else 0
    pos_days  = sum(1 for p in daily_pnls if p > 0)
    neg_days  = sum(1 for p in daily_pnls if p < 0)
    flat_days = sum(1 for p in daily_pnls if p == 0)

    print()
    print("═" * W)
    print(f"{'  MONTH SUMMARY':^{W}}")
    print("═" * W)
    print(f"  Starting equity         : $50,000")
    print(f"  Ending equity           : ${equity:>10,.2f}")
    print(f"  Net P&L (21 days)       : {fmt_money(eval_pnl):>12}")
    print(f"  Max drawdown (peak→tro) : {fmt_money(max_dd):>12}  (funded limit: -$2,500)")
    print(f"  Best day                : {fmt_money(best_day):>12}")
    print(f"  Worst day               : {fmt_money(worst_day):>12}")
    print(f"  Up days                 : {pos_days:>3}  |  Down days: {neg_days:>3}  |  Flat: {flat_days:>3}")
    if target_hit_day:
        print(f"  $3,000 TARGET HIT       : {target_hit_day}  ⇒ eval should PASS")
    else:
        gap = 3_000 - eval_pnl
        print(f"  Target progress         : {fmt_money(eval_pnl)} / $3,000  ({eval_pnl/3000*100:>4.0f}%)  gap: {fmt_money(gap)}")
    if blow_day:
        print(f"  ⚠ DD BREACH             : {blow_day}  (would have blown account)")

    # consistency check — biggest day must be < 20% of total
    if eval_pnl > 0:
        biggest = max(daily_pnls)
        if biggest > 0:
            consistency_pct = biggest / eval_pnl * 100
            print(f"  Consistency (best/tot)  : {consistency_pct:>4.1f}%  (must be <20% at eval end)")
            if consistency_pct >= 20:
                print(f"    ⚠ FAILS 20% rule — biggest day too large vs total")

    if total:
        wr = 100 * len(wins) / total
        avg_win  = float(np.mean([t["pnl"] for t in wins]))  if wins   else 0.0
        avg_loss = float(np.mean([t["pnl"] for t in losses])) if losses else 0.0
        avg_r    = float(np.mean([t["r"]   for t in all_trades]))
        pf       = abs(len(wins) * avg_win / (len(losses) * avg_loss)) \
                   if losses and avg_loss != 0 else float("inf")

        print()
        print(f"  Total trades            : {total}  ({total/21:.1f}/day avg)")
        print(f"  Win rate                : {len(wins)}/{total}  ({wr:.0f}%)")
        print(f"  Avg winner              : {fmt_money(avg_win)}")
        print(f"  Avg loser               : {fmt_money(avg_loss)}")
        print(f"  Profit factor           : {pf:.2f}")
        print(f"  Avg R per trade         : {avg_r:>+.2f}R")

        # by strategy
        by_s: dict = {}
        for t in all_trades:
            s = t.get("strategy", "?")
            by_s.setdefault(s, {"n": 0, "pnl": 0.0, "w": 0})
            by_s[s]["n"]   += 1
            by_s[s]["pnl"] += t["pnl"]
            by_s[s]["w"]   += 1 if t["pnl"] > 0 else 0
        print()
        print(f"  ── By strategy ──")
        for s, d in sorted(by_s.items(), key=lambda x: -x[1]["pnl"]):
            swr = 100 * d["w"] / d["n"]
            print(f"    {s:<16}  {d['n']:>2} trades  {d['w']}/{d['n']} ({swr:>3.0f}% WR)  {fmt_money(d['pnl'])}")

        # by grade
        by_g: dict = {}
        for t in all_trades:
            g = t.get("grade", "?")
            by_g.setdefault(g, {"n": 0, "pnl": 0.0, "w": 0})
            by_g[g]["n"]   += 1
            by_g[g]["pnl"] += t["pnl"]
            by_g[g]["w"]   += 1 if t["pnl"] > 0 else 0
        print()
        print(f"  ── By grade ──")
        for g in ("A+", "A", "B"):
            if g in by_g:
                d = by_g[g]
                gwr = 100 * d["w"] / d["n"]
                print(f"    {g:<3}              {d['n']:>2} trades  {d['w']}/{d['n']} ({gwr:>3.0f}% WR)  {fmt_money(d['pnl'])}")

    # ── verdict ──────────────────────────────────────────────────────────────
    print()
    print("═" * W)
    print(f"{'  VERDICT':^{W}}")
    print("═" * W)
    if target_hit_day and not blow_day:
        # Did consistency rule pass?
        biggest = max(daily_pnls) if daily_pnls else 0
        consistent = (biggest / eval_pnl * 100 < 20) if eval_pnl > 0 else False
        if consistent:
            print(f"  ✓ EVAL PASSES — hit $3,000 on {target_hit_day}, no DD breach, consistency OK")
            print(f"    Expected funded-account payout cycle: ~once every {21 / (MONTH.index([d for d in MONTH if d[0] == target_hit_day][0]) + 1):.1f} months")
        else:
            print(f"  ⚠ TARGET HIT but consistency rule FAILED — biggest day too large vs total")
    elif blow_day:
        print(f"  ✗ ACCOUNT WOULD HAVE BLOWN on {blow_day} — DD exceeded $1,700")
    else:
        print(f"  ◷ MONTH ENDED MID-EVAL — {fmt_money(eval_pnl)} / $3,000 ({eval_pnl/3000*100:.0f}%)")
        days_to_target = 21 * 3000 / eval_pnl if eval_pnl > 0 else float("inf")
        if eval_pnl > 0:
            print(f"    At this pace, ~{days_to_target:.0f} trading days to pass eval (~{days_to_target/21:.1f} months)")

    print()
    print("  NOTES:")
    print("  • Synthetic data calibrated to real SPY 1-min stats (σ≈0.05%/bar)")
    print("  • Real markets have macro events, gaps, halts not modeled here")
    print("  • Fills assume stop/target trigger price (no slippage)")
    print("  • This is the BEST CASE — real fills will be ~$5-15 worse per trade")
    print("═" * W)


if __name__ == "__main__":
    main()
