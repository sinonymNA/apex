"""
backtest/run.py — Backtesting and regime model training for Apex Trading System.

Steps:
  1. Fetch 5yr daily SPY + 60day 5min SPY via yfinance
  2. Train regime classifier on daily data and save pkl
  3. Run MomentumBreakout on 5min data (walk-forward: pre-2024 / 2024+)
  4. Compute and print side-by-side metrics table
  5. Evaluate against gate criteria (GATE PASSED / GATE FAILED)
  6. Save results to logs/backtest_report.json

Run: python backtest/run.py
"""
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import pytz
import yfinance as yf
from loguru import logger

from worker.strategy import MomentumBreakout
from models.regime_classifier import RegimeClassifier


# ── Constants ─────────────────────────────────────────────────────────────────
SYMBOL = "SPY"
TRAIN_CUTOFF = "2024-01-01"

# Gate thresholds
GATE_WIN_RATE = 0.38
GATE_PROFIT_FACTOR = 1.1
GATE_MAX_DRAWDOWN = -8000  # dollars


# ── Synthetic data generator (fallback when yfinance is unavailable) ──────────
def _make_synthetic_ohlcv(n: int, freq: str, base: float = 480.0, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic-looking SPY OHLCV data for offline backtesting.
    Used when the Yahoo Finance connection is unavailable (e.g., CI/sandbox).
    """
    rng = np.random.default_rng(seed)
    prices = base * np.cumprod(1 + rng.normal(0.0002, 0.003, n))
    noise = rng.uniform(0.001, 0.004, n)
    close = prices
    high = prices * (1 + noise)
    low = prices * (1 - noise)
    open_ = prices * (1 + rng.uniform(-0.002, 0.002, n))
    volume = rng.integers(500_000, 8_000_000, n).astype(float)
    # Inject occasional volume spikes to trigger strategy entries
    spike_idx = rng.choice(n, size=n // 15, replace=False)
    volume[spike_idx] *= 3.5
    idx = pd.date_range("2019-01-02", periods=n, freq=freq, tz="America/New_York")
    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=idx,
    )


# ── Data fetching ─────────────────────────────────────────────────────────────
def fetch_data() -> tuple:
    """
    Fetch daily (5yr) and 5-minute (60-day) SPY data from yfinance.
    Falls back to synthetic data if network is unavailable.
    Returns (daily_df, intraday_df).
    """
    logger.info("Fetching 5yr daily SPY data...")
    daily = pd.DataFrame()
    try:
        daily = yf.download(
            SYMBOL,
            period="5y",
            interval="1d",
            auto_adjust=True,
            progress=False,
        )
        if hasattr(daily.columns, "levels"):
            daily.columns = daily.columns.get_level_values(0)
    except Exception as e:
        logger.warning(f"yfinance daily download failed: {e}")

    if daily.empty:
        logger.warning("Using synthetic daily data (yfinance unavailable)")
        daily = _make_synthetic_ohlcv(n=1260, freq="B", base=400.0, seed=1)
    else:
        logger.info(f"Daily bars: {len(daily)} rows ({daily.index[0].date()} to {daily.index[-1].date()})")

    logger.info("Fetching 60-day 5min SPY data...")
    intraday = pd.DataFrame()
    try:
        intraday = yf.download(
            SYMBOL,
            period="60d",
            interval="5m",
            auto_adjust=True,
            progress=False,
        )
        if hasattr(intraday.columns, "levels"):
            intraday.columns = intraday.columns.get_level_values(0)
    except Exception as e:
        logger.warning(f"yfinance intraday download failed: {e}")

    if intraday.empty:
        logger.warning("Using synthetic 5-min data (yfinance unavailable)")
        # Build a proper trading-hours index: 9:30-15:55 ET, Mon-Fri, 2022-2025
        ET_tz = pytz.timezone("America/New_York")
        trading_bars = []
        start_date = pd.Timestamp("2022-01-03", tz=ET_tz)
        end_date = pd.Timestamp("2025-12-31", tz=ET_tz)
        cur = start_date
        while cur <= end_date:
            if cur.weekday() < 5:  # Mon-Fri only
                bar_time = cur.replace(hour=9, minute=30)
                close_time = cur.replace(hour=16, minute=0)
                while bar_time < close_time:
                    trading_bars.append(bar_time)
                    bar_time += pd.Timedelta(minutes=5)
            cur += pd.Timedelta(days=1)
        trade_idx = pd.DatetimeIndex(trading_bars)
        n_bars = len(trade_idx)
        logger.info(f"Generated {n_bars} synthetic trading bars (2022-2025)")
        rng = np.random.default_rng(2)
        base_prices = 480.0 * np.cumprod(1 + rng.normal(0.00004, 0.0015, n_bars))
        noise = rng.uniform(0.001, 0.004, n_bars)
        volume_base = rng.integers(500_000, 5_000_000, n_bars).astype(float)
        spike_idx = rng.choice(n_bars, size=n_bars // 15, replace=False)
        volume_base[spike_idx] *= 3.5
        intraday = pd.DataFrame({
            "Open":   base_prices * (1 + rng.uniform(-0.001, 0.001, n_bars)),
            "High":   base_prices * (1 + noise),
            "Low":    base_prices * (1 - noise),
            "Close":  base_prices,
            "Volume": volume_base,
        }, index=trade_idx)
    else:
        logger.info(f"5min bars: {len(intraday)} rows ({intraday.index[0].date()} to {intraday.index[-1].date()})")

    return daily, intraday


# ── Strategy simulation ───────────────────────────────────────────────────────
def run_backtest_on(df_5min: pd.DataFrame, label: str) -> tuple[list, dict]:
    """
    Simulate the MomentumBreakout strategy on 5-min OHLCV data.

    Uses a rolling window; no forward-looking data.
    Returns (trades_list, metrics_dict).
    """
    strategy = MomentumBreakout()
    trades = []
    position = None  # {entry, stop, target, entry_time, entry_price, atr}

    import pytz
    ET = pytz.timezone("America/New_York")

    # Compute indicators on the full dataset first
    df = strategy.compute_indicators(df_5min.copy())
    valid_df = df.dropna(subset=["high_20", "volume_avg", "atr14"])

    logger.info(f"[{label}] Simulating on {len(valid_df)} valid bars...")

    for idx in range(len(valid_df)):
        bar = valid_df.iloc[idx]
        bar_time = valid_df.index[idx]

        # Convert to ET
        if hasattr(bar_time, "tzinfo") and bar_time.tzinfo is not None:
            et_time = bar_time.astimezone(ET)
        else:
            et_time = ET.localize(bar_time.replace(tzinfo=None))

        current_price = float(bar["Close"])

        # Monitor open position
        if position is not None:
            elapsed_min = (bar_time - position["entry_time"]).total_seconds() / 60

            exit_price = None
            exit_reason = None

            if current_price <= position["stop"]:
                exit_price = position["stop"]
                exit_reason = "stop_hit"
            elif current_price >= position["target"]:
                exit_price = position["target"]
                exit_reason = "target_hit"
            elif elapsed_min >= strategy.MAX_HOLD_MINUTES:
                exit_price = current_price
                exit_reason = "max_hold_exceeded"

            if exit_price is not None:
                entry_price = position["entry_price"]
                shares = 2  # MAX_CONTRACTS
                pnl = (exit_price - entry_price) * shares
                risk_amt = abs(entry_price - position["stop"]) * shares
                r_val = pnl / risk_amt if risk_amt > 0 else 0.0

                trades.append({
                    "entry_time": position["entry_time"],
                    "exit_time": bar_time,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "pnl": pnl,
                    "r": r_val,
                    "exit_reason": exit_reason,
                })
                position = None
            continue  # still in position, no new entry

        # Signal generation
        et_hour = et_time.hour
        et_minute = et_time.minute
        from datetime import time as dtime
        trade_count = len(trades)  # simplified — doesn't reset per day

        signal = strategy.generate_signals(
            valid_df.iloc[: idx + 1],
            et_time,
            len([t for t in trades if t["entry_time"].date() == et_time.date()]),
        )

        if signal is not None:
            levels = strategy.get_levels(signal["price"], signal["atr"])
            position = {
                "entry_time": bar_time,
                "entry_price": levels["entry"],
                "stop": levels["stop"],
                "target": levels["target"],
                "atr": signal["atr"],
            }

    # Close any remaining open position at last bar price
    if position is not None and not valid_df.empty:
        exit_price = float(valid_df["Close"].iloc[-1])
        entry_price = position["entry_price"]
        shares = 2
        pnl = (exit_price - entry_price) * shares
        risk_amt = abs(entry_price - position["stop"]) * shares
        r_val = pnl / risk_amt if risk_amt > 0 else 0.0
        trades.append({
            "entry_time": position["entry_time"],
            "exit_time": valid_df.index[-1],
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "r": r_val,
            "exit_reason": "end_of_data",
        })

    metrics = compute_metrics(trades, label)
    return trades, metrics


def compute_metrics(trades: list, label: str = "") -> dict:
    """Compute performance metrics from a list of trade dicts."""
    if not trades:
        return {
            "label": label,
            "total_trades": 0,
            "win_rate": 0.0,
            "avg_r": 0.0,
            "max_drawdown": 0.0,
            "gross_pnl": 0.0,
            "profit_factor": 0.0,
            "sharpe": 0.0,
            "best_month": 0.0,
            "worst_month": 0.0,
            "max_consecutive_losses": 0,
        }

    pnls = [t["pnl"] for t in trades]
    rs = [t["r"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    # Equity curve
    equity = np.cumsum(pnls)
    peak = np.maximum.accumulate(equity)
    drawdown = equity - peak
    max_drawdown = float(np.min(drawdown))

    # Profit factor
    gross_wins = sum(wins) if wins else 0
    gross_losses = abs(sum(losses)) if losses else 1e-9
    profit_factor = gross_wins / gross_losses

    # Sharpe (daily grouping)
    daily_pnl = {}
    for t in trades:
        d = t["entry_time"].date() if hasattr(t["entry_time"], "date") else str(t["entry_time"])[:10]
        daily_pnl[d] = daily_pnl.get(d, 0) + t["pnl"]
    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252)
    else:
        sharpe = 0.0

    # Monthly P&L
    monthly_pnl = {}
    for t in trades:
        entry = t["entry_time"]
        month_key = f"{entry.year}-{entry.month:02d}" if hasattr(entry, "year") else str(entry)[:7]
        monthly_pnl[month_key] = monthly_pnl.get(month_key, 0) + t["pnl"]
    month_vals = list(monthly_pnl.values()) if monthly_pnl else [0]
    best_month = max(month_vals)
    worst_month = min(month_vals)

    # Max consecutive losses
    max_consec = 0
    current_consec = 0
    for p in pnls:
        if p <= 0:
            current_consec += 1
            max_consec = max(max_consec, current_consec)
        else:
            current_consec = 0

    return {
        "label": label,
        "total_trades": len(trades),
        "win_rate": len(wins) / len(trades),
        "avg_r": float(np.mean(rs)) if rs else 0.0,
        "max_drawdown": max_drawdown,
        "gross_pnl": float(np.sum(pnls)),
        "profit_factor": profit_factor,
        "sharpe": float(sharpe),
        "best_month": best_month,
        "worst_month": worst_month,
        "max_consecutive_losses": max_consec,
    }


# ── Display ───────────────────────────────────────────────────────────────────
def print_table(train_m: dict, test_m: dict):
    """Print a side-by-side comparison table."""
    rows = [
        ("Total Trades",         f"{train_m['total_trades']}",                 f"{test_m['total_trades']}"),
        ("Win Rate",             f"{train_m['win_rate']:.1%}",                 f"{test_m['win_rate']:.1%}"),
        ("Avg R",                f"{train_m['avg_r']:.2f}R",                   f"{test_m['avg_r']:.2f}R"),
        ("Gross P&L",            f"${train_m['gross_pnl']:+,.0f}",             f"${test_m['gross_pnl']:+,.0f}"),
        ("Max Drawdown",         f"${train_m['max_drawdown']:,.0f}",           f"${test_m['max_drawdown']:,.0f}"),
        ("Profit Factor",        f"{train_m['profit_factor']:.2f}",            f"{test_m['profit_factor']:.2f}"),
        ("Sharpe Ratio",         f"{train_m['sharpe']:.2f}",                   f"{test_m['sharpe']:.2f}"),
        ("Best Month",           f"${train_m['best_month']:+,.0f}",            f"${test_m['best_month']:+,.0f}"),
        ("Worst Month",          f"${train_m['worst_month']:+,.0f}",           f"${test_m['worst_month']:+,.0f}"),
        ("Max Consec. Losses",   f"{train_m['max_consecutive_losses']}",       f"{test_m['max_consecutive_losses']}"),
    ]

    col_w = 22
    header_l = f"Train (pre-{TRAIN_CUTOFF[:4]})"
    header_r = f"Test ({TRAIN_CUTOFF[:4]}–present)"

    sep = "+" + "-" * 26 + "+" + "-" * col_w + "+" + "-" * col_w + "+"
    print("\n" + sep)
    print(f"| {'Metric':<24} | {header_l:<{col_w}} | {header_r:<{col_w}} |")
    print(sep)
    for name, lval, rval in rows:
        print(f"| {name:<24} | {lval:<{col_w}} | {rval:<{col_w}} |")
    print(sep)


def gate_check(test_m: dict) -> bool:
    """Evaluate walk-forward test results against gate criteria."""
    results = {
        "win_rate":      (test_m["win_rate"] > GATE_WIN_RATE,       f"{test_m['win_rate']:.1%} > {GATE_WIN_RATE:.0%}"),
        "profit_factor": (test_m["profit_factor"] > GATE_PROFIT_FACTOR, f"{test_m['profit_factor']:.2f} > {GATE_PROFIT_FACTOR}"),
        "max_drawdown":  (test_m["max_drawdown"] > GATE_MAX_DRAWDOWN,   f"${test_m['max_drawdown']:,.0f} > ${GATE_MAX_DRAWDOWN:,}"),
    }

    print("\n── Gate Evaluation ─────────────────────────────────")
    all_passed = True
    for name, (passed, detail) in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {name:<20} {detail:<30} [{status}]")
        if not passed:
            all_passed = False

    print()
    if all_passed:
        print("  ✓ GATE PASSED — Strategy meets minimum thresholds")
    else:
        print("  ✗ GATE FAILED — Review strategy before going live")
    print("─" * 50)

    return all_passed


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    logger.info("=== Apex Trading System Backtest ===")

    # 1. Fetch data
    daily_df, intraday_df = fetch_data()

    # 2. Train regime classifier on daily data and save pkl
    logger.info("Training regime classifier...")
    rc = RegimeClassifier()
    try:
        rc.train_and_save(daily_df)
        logger.info("Regime classifier trained and saved to models/regime_rf.pkl")
    except Exception as e:
        logger.error(f"Regime training failed: {e}")
        logger.warning("Proceeding with backtest using default 'Weak Trend' regime")

    # 3. Walk-forward split on 5-min data
    cutoff = pd.Timestamp(TRAIN_CUTOFF)
    if intraday_df.index.tz is not None:
        cutoff = cutoff.tz_localize(intraday_df.index.tz)

    train_df = intraday_df[intraday_df.index < cutoff]
    test_df = intraday_df[intraday_df.index >= cutoff]

    # Note: yfinance 60-day limit means test_df may be empty if running pre-2024
    # In that case, treat all data as test
    if train_df.empty or len(train_df) < 50:
        logger.warning(
            f"Train split has only {len(train_df)} bars (yfinance 60-day limit). "
            "Using all data as test period."
        )
        train_df = intraday_df.copy()
        test_df = intraday_df.copy()

    logger.info(f"Train bars: {len(train_df)} | Test bars: {len(test_df)}")

    # 4. Run backtests
    train_trades, train_metrics = run_backtest_on(train_df, f"Train (pre-{TRAIN_CUTOFF[:4]})")
    test_trades, test_metrics = run_backtest_on(test_df, f"Test ({TRAIN_CUTOFF[:4]}+)")

    # 5. Print table
    print_table(train_metrics, test_metrics)

    # 6. Gate check
    gate_passed = gate_check(test_metrics)

    # 7. Save report
    report = {
        "generated_at": datetime.now().isoformat(),
        "symbol": SYMBOL,
        "train_cutoff": TRAIN_CUTOFF,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "gate_passed": gate_passed,
        "gate_criteria": {
            "win_rate_min": GATE_WIN_RATE,
            "profit_factor_min": GATE_PROFIT_FACTOR,
            "max_drawdown_min": GATE_MAX_DRAWDOWN,
        },
        "train_trades": len(train_trades),
        "test_trades": len(test_trades),
    }

    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    report_path = log_dir / "backtest_report.json"
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    logger.info(f"Report saved to {report_path}")
    return report


if __name__ == "__main__":
    main()
