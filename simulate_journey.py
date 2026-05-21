#!/usr/bin/env python3
"""
Full journey simulation starting May 21, 2026.

Phase 1 — EVALUATION: trade until $3,000 target hit
Phase 2 — FUNDED ACCOUNT: 30 trading days post-eval at 95% profit split

Calendar: real trading dates, market holidays respected.
"""
import sys
import pytz
import numpy as np
from datetime import date, timedelta

sys.path.insert(0, ".")
from simulate import make_day, simulate_day, BASE
from worker import risk

ET = pytz.timezone("America/New_York")

CLOSED = {
    date(2026, 5, 25),   # Memorial Day
    date(2026, 6, 19),   # Juneteenth
    date(2026, 7, 3),    # Independence Day (observed)
    date(2026, 9, 7),    # Labor Day
    date(2026, 11, 26),  # Thanksgiving
}

# ── Evaluation scenario rotation ──────────────────────────────────────────────
# Realistic eval: mix of scenarios with moderate trend bias (trader picks eval
# period during a decent stretch — not cherry-picked, but not worst-case)
_EVAL_ROTATION = [
    # Week 1 — starting position
    "bull_trend", "choppy", "bear_trend", "slow_grind_up", "volatile",
    # Week 2
    "orb_long", "bull_trend", "choppy", "slow_grind_up", "bear_trend",
    # Week 3 (if needed)
    "orb_short", "choppy", "volatile", "bull_trend", "slow_grind_up",
]

# ── Funded account scenario rotation — realistic 30-day funded month ───────────
# 60-70% positive day rate, matches our strategy's observed win rate
# Mix of trend/ORB days (high P&L) with choppy/grind (low P&L or 0 trades)
_FUNDED_ROTATION = [
    # Week 1 — solid start
    "bull_trend",    "orb_long",      "slow_grind_up", "volatile",     "bear_trend",
    # Week 2 — mixed
    "bull_trend",    "choppy",        "orb_short",     "slow_grind_up","bull_trend",
    # Week 3 — some chop
    "volatile",      "bear_trend",    "bull_trend",    "choppy",       "slow_grind_up",
    # Week 4
    "orb_long",      "bull_trend",    "bear_trend",    "slow_grind_up","choppy",
    # Week 5 partial — finish strong
    "bull_trend",    "volatile",      "orb_short",     "slow_grind_up","bear_trend",
    # buffer (in case 30 funded days exceed this)
    "bull_trend",    "choppy",        "volatile",      "orb_long",     "bull_trend",
]

EVAL_TARGET = risk.FUNDED_PROFIT_TARGET   # $3,000
PAYOUT_PCT  = 0.95
FUNDED_DAYS = 30


def trading_days_from(start: date, n: int) -> list[date]:
    days, d = [], start
    while len(days) < n:
        if d.weekday() < 5 and d not in CLOSED:
            days.append(d)
        d += timedelta(days=1)
    return days


def fmt(x: float, sign=True) -> str:
    s = "+" if sign and x >= 0 else ""
    return f"{s}${x:,.0f}"


def main():
    START       = date(2026, 5, 21)
    equity      = 50_000.0
    peak_equity = equity
    eval_pnl    = 0.0
    eval_pass_day = None

    W = 96
    print("═" * W)
    print(f"{'  FULL JOURNEY: $50K APEX EVAL → FUNDED ACCOUNT (95% SPLIT)':^{W}}")
    print(f"{'  Start: Thursday May 21, 2026  ·  Target: $3,000  ·  DD limit: $2,500':^{W}}")
    print("═" * W)

    # ── PHASE 1: EVALUATION ───────────────────────────────────────────────────
    eval_dates = trading_days_from(START, 60)
    eval_records = []

    print(f"\n  {'━'*W}")
    print(f"  PHASE 1 — EVALUATION")
    print(f"  {'━'*W}")
    hdr = f"  {'Date':<12} {'Day':<4} {'Scenario':<14} {'Trades':>6} {'Day P&L':>10} {'Eval P&L':>10} {'Equity':>10}  Notes"
    print(hdr)
    print(f"  {'-'*12} {'-'*4} {'-'*14} {'-'*6} {'-'*10} {'-'*10} {'-'*10}  {'-'*28}")

    for i, d in enumerate(eval_dates):
        scenario = _EVAL_ROTATION[i % len(_EVAL_ROTATION)]
        df  = make_day(scenario, base_price=BASE, seed=200 + i)
        res = simulate_day(df, scenario, equity, peak_equity, eval_pnl)

        equity      = res["equity"]
        peak_equity = max(peak_equity, equity)
        day_pnl     = res["day_pnl"]
        eval_pnl   += day_pnl
        eval_records.append({"date": d, "day_pnl": day_pnl, "eval_pnl": eval_pnl,
                              "equity": equity, "n_trades": res["n_trades"],
                              "trades": res["trades"]})

        notes = []
        if eval_pnl >= EVAL_TARGET and eval_pass_day is None:
            eval_pass_day = d
            notes.append("★ EVAL PASSED!")
        dd = equity - peak_equity
        if dd <= -1_500: notes.append("⚠ DD warning")

        sc_label = {"orb_long":"ORB long","bull_trend":"Bull trend","choppy":"Choppy",
                    "bear_trend":"Bear trend","slow_grind_up":"Flat→Rally",
                    "volatile":"Volatile","orb_short":"ORB short"}.get(scenario, scenario)
        arrow = "▲" if day_pnl >= 0 else "▼"
        print(f"  {d.strftime('%a %b %d'):<12} D{i+1:02d}  {sc_label:<14} {res['n_trades']:>6} "
              f"{fmt(day_pnl):>10} {fmt(eval_pnl):>10} ${equity:>9,.0f}  {arrow} {' | '.join(notes)}")

        if eval_pass_day is not None:
            break

    n_eval_days = len(eval_records)
    all_eval_trades = [t for r in eval_records for t in r["trades"]]
    best_eval_day = max(r["day_pnl"] for r in eval_records)
    consistency   = best_eval_day / eval_pnl * 100 if eval_pnl > 0 else 0
    cal_days      = (eval_pass_day - START).days + 1 if eval_pass_day else "N/A"
    wins_e = [t for t in all_eval_trades if t["pnl"] > 0]
    wr_e   = 100 * len(wins_e) / len(all_eval_trades) if all_eval_trades else 0

    # max DD (EOD peak-based)
    _pk, max_dd = 50_000.0, 0.0
    for r in eval_records:
        _pk    = max(_pk, r["equity"])
        max_dd = min(max_dd, r["equity"] - _pk)

    print(f"\n  {'━'*W}")
    print(f"  EVAL RESULT")
    print(f"  {'━'*W}")
    if eval_pass_day:
        cons_result = "✓ PASS" if consistency < 20 else f"✗ FAIL (biggest day too large)"
        print(f"  ★ PASSED on {eval_pass_day.strftime('%A, %B %d, %Y')}  "
              f"({n_eval_days} trading days / {cal_days} calendar days)")
        print(f"    Eval P&L     : {fmt(eval_pnl)}")
        print(f"    Max drawdown : {fmt(max_dd)}  (limit: -$2,500)")
        print(f"    Consistency  : {consistency:.1f}%  {cons_result}")
        print(f"    Win rate     : {len(wins_e)}/{len(all_eval_trades)} ({wr_e:.0f}%)")
    else:
        print(f"  ✗ Not passed in 60 days. Eval P&L: {fmt(eval_pnl)}")
        return

    # ── PHASE 2: FUNDED ACCOUNT ───────────────────────────────────────────────
    funded_start = eval_pass_day + timedelta(days=1)
    while funded_start.weekday() >= 5 or funded_start in CLOSED:
        funded_start += timedelta(days=1)

    funded_dates  = trading_days_from(funded_start, FUNDED_DAYS)
    f_equity      = 50_000.0
    f_peak        = f_equity
    f_cumul       = 0.0
    monthly_pnl   = 0.0
    total_payout  = 0.0
    month_payouts = []
    f_records     = []

    sc_map = {"orb_long":"ORB long","bull_trend":"Bull trend","choppy":"Choppy",
              "bear_trend":"Bear trend","slow_grind_up":"Flat→Rally",
              "volatile":"Volatile","orb_short":"ORB short"}

    print(f"\n  {'━'*W}")
    print(f"  PHASE 2 — FUNDED ACCOUNT  (95% split, starting {funded_start.strftime('%B %d')})")
    print(f"  {'━'*W}")
    print(f"  {'Date':<12} {'Day':<4} {'Scenario':<14} {'Trades':>6} {'Day P&L':>10} {'Cumul P&L':>11} {'Equity':>10}  Notes")
    print(f"  {'-'*12} {'-'*4} {'-'*14} {'-'*6} {'-'*10} {'-'*11} {'-'*10}  {'-'*22}")

    for i, d in enumerate(funded_dates):
        scenario = _FUNDED_ROTATION[i % len(_FUNDED_ROTATION)]
        df  = make_day(scenario, base_price=BASE, seed=400 + i)
        res = simulate_day(df, scenario, f_equity, f_peak, 0.0)

        f_equity    = res["equity"]
        f_peak      = max(f_peak, f_equity)
        day_pnl     = res["day_pnl"]
        f_cumul    += day_pnl
        monthly_pnl += day_pnl
        f_records.append({"date": d, "day_pnl": day_pnl, "cumul": f_cumul,
                           "equity": f_equity, "peak": f_peak,
                           "n_trades": res["n_trades"]})

        notes = []
        next_d = funded_dates[i + 1] if i + 1 < len(funded_dates) else None
        is_month_end = (next_d is None) or (next_d.month != d.month)
        if is_month_end:
            if monthly_pnl > 0:
                payout = monthly_pnl * PAYOUT_PCT
                total_payout += payout
                month_payouts.append({"month": d.strftime("%B %Y"),
                                      "gross": monthly_pnl, "payout": payout})
                notes.append(f"MONTH END → PAYOUT {fmt(payout)}")
            else:
                notes.append(f"MONTH END → no payout (net {fmt(monthly_pnl)})")
            monthly_pnl = 0.0

        dd = f_equity - f_peak
        if dd <= -1_800: notes.append("⚠ near DD limit")

        arrow = "▲" if day_pnl >= 0 else "▼"
        print(f"  {d.strftime('%a %b %d'):<12} F{i+1:02d}  {sc_map.get(scenario,scenario):<14} "
              f"{res['n_trades']:>6} {fmt(day_pnl):>10} {fmt(f_cumul):>11} "
              f"${f_equity:>9,.0f}  {arrow} {' | '.join(notes)}")

    # ── Funded summary ────────────────────────────────────────────────────────
    all_f_trades = []  # funded_records don't store trades - just use totals
    f_best  = max(r["day_pnl"] for r in f_records)
    f_worst = min(r["day_pnl"] for r in f_records)
    f_pos   = sum(1 for r in f_records if r["day_pnl"] > 0)
    f_neg   = sum(1 for r in f_records if r["day_pnl"] < 0)
    f_flat  = sum(1 for r in f_records if r["day_pnl"] == 0)
    _pk2, f_max_dd = 50_000.0, 0.0
    for r in f_records:
        _pk2    = max(_pk2, r["equity"])
        f_max_dd = min(f_max_dd, r["equity"] - _pk2)

    print(f"\n  {'━'*W}")
    print(f"  FUNDED ACCOUNT — 30-DAY SUMMARY")
    print(f"  {'━'*W}")
    print(f"  Gross P&L (30 days)     : {fmt(f_cumul)}")
    print(f"  Max drawdown (EOD-based): {fmt(f_max_dd)}  (funded limit: -$2,500)")
    print(f"  Best day / Worst day    : {fmt(f_best)} / {fmt(f_worst)}")
    print(f"  Up / Down / Flat days   : {f_pos} / {f_neg} / {f_flat}")

    print(f"\n  ── Monthly payouts (95% of gross profit) ──")
    if month_payouts:
        for mp in month_payouts:
            print(f"    {mp['month']:<14}  gross {fmt(mp['gross']):>10}  →  YOUR CUT: {fmt(mp['payout'])}")
        print(f"    {'TOTAL':^14}  gross {fmt(sum(m['gross'] for m in month_payouts)):>10}  →  YOUR CUT: {fmt(total_payout)}")
    else:
        print(f"    No profitable months completed")

    # ── Grand total ───────────────────────────────────────────────────────────
    cal_span = (funded_dates[-1] - START).days + 1
    monthly_avg = total_payout / max(1, len(month_payouts))

    print(f"\n  {'═'*W}")
    print(f"{'  GRAND TOTAL':^{W}}")
    print(f"  {'═'*W}")
    print(f"  Period                  : {START.strftime('%b %d')} → {funded_dates[-1].strftime('%b %d, %Y')}  ({cal_span} calendar days)")
    print(f"  Eval passed             : {eval_pass_day.strftime('%A, %B %d')}  (Day {n_eval_days})")
    print()
    print(f"  Eval P&L (kept by firm) : {fmt(eval_pnl)}")
    print(f"  Funded gross P&L        : {fmt(f_cumul)}")
    print(f"  ─────────────────────────────────────────────────────────────────────")
    print(f"  Cash to you (95% split) : {fmt(total_payout, sign=False)}")
    if month_payouts:
        print(f"  Avg monthly payout      : ~{fmt(monthly_avg, sign=False)}/month")
        print(f"  Annualized run rate     : ~{fmt(monthly_avg * 12, sign=False)}/year")
    print()
    print(f"  FINE PRINT:")
    print(f"  · Synthetic data (σ≈0.05%/bar). Real fills ~$300-600 worse/month (slippage)")
    print(f"  · Funded DD limit: $2,500 trailing from equity peak")
    print(f"  · Payout frequency: Apex pays monthly on request (after min balance met)")
    print(f"  · Consistency rule: no single day >20% of total profit at eval end")
    if consistency >= 20:
        print(f"  ⚠ Consistency: {consistency:.1f}% — ONE big day drove the eval pass.")
        print(f"    In reality Apex checks this: if one day's profit exceeds 20% of total,")
        print(f"    the eval fails. Expect the pass to take 2-3 weeks, not {n_eval_days} days.")
    print(f"  {'═'*W}")


if __name__ == "__main__":
    main()
