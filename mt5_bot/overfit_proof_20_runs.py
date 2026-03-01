import argparse
import json
import math
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest_improved as bi


def _binomial_tail_probability_at_least_k(k: int, n: int, p: float = 0.5) -> float:
    if n <= 0:
        return 1.0
    if k <= 0:
        return 1.0
    if k > n:
        return 0.0

    total = 0.0
    for i in range(k, n + 1):
        comb = math.comb(n, i)
        total += comb * (p ** i) * ((1 - p) ** (n - i))
    return float(total)


def _safe_float(value, default=0.0):
    try:
        return float(value)
    except Exception:
        return float(default)


def _build_day_windows(available_days: float, count: int = 20):
    base = [20, 30, 40, 50, 60, 75, 90, 110, 130, 150, 170, 190, 210, 230, 250, 270, 300, 330, 365, 420]
    max_days = max(20, int(available_days * 0.85))
    windows = sorted({d for d in base if d <= max_days})

    if len(windows) >= count:
        return windows[:count]

    min_days = 20
    extra = np.linspace(min_days, max_days, num=count).astype(int).tolist()
    merged = sorted({*windows, *extra})

    if len(merged) < count:
        step = max(1, (max_days - min_days) // max(1, count - len(merged)))
        x = min_days
        while len(merged) < count and x <= max_days:
            merged.append(int(x))
            x += step
        merged = sorted(set(merged))

    if len(merged) > count:
        idx = np.linspace(0, len(merged) - 1, num=count).astype(int)
        merged = [merged[i] for i in idx]

    return merged


def _plot_overlay_equity(equity_runs, out_path):
    fig = plt.figure(figsize=(11, 5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)

    for idx, eq in enumerate(equity_runs):
        if not eq or len(eq) < 2:
            continue
        x = np.linspace(0, 1, num=len(eq))
        ax.plot(x, eq, alpha=0.25, linewidth=1)

    if equity_runs:
        max_len = max(len(eq) for eq in equity_runs if eq)
        if max_len >= 2:
            aligned = []
            for eq in equity_runs:
                if not eq or len(eq) < 2:
                    continue
                src_x = np.linspace(0, 1, num=len(eq))
                dst_x = np.linspace(0, 1, num=max_len)
                aligned.append(np.interp(dst_x, src_x, np.array(eq, dtype=float)))
            if aligned:
                mean_eq = np.mean(np.array(aligned), axis=0)
                ax.plot(np.linspace(0, 1, num=max_len), mean_eq, linewidth=2.8, label="Mean equity")
                ax.legend(loc="best")

    ax.set_title("20 Historical Windows - Equity Curves Overlay")
    ax.set_xlabel("Normalized time in window")
    ax.set_ylabel("Balance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def _plot_return_hist(returns, out_path):
    fig = plt.figure(figsize=(9, 4.5), dpi=140)
    ax = fig.add_subplot(1, 1, 1)
    ax.hist(returns, bins=10, alpha=0.8)
    ax.axvline(np.mean(returns), linestyle="--", linewidth=2, label=f"Mean {np.mean(returns):.2f}%")
    ax.axvline(np.median(returns), linestyle=":", linewidth=2, label=f"Median {np.median(returns):.2f}%")
    ax.set_title("Distribution of Return % Across 20 Backtests")
    ax.set_xlabel("Return %")
    ax.set_ylabel("Count")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def run_overfit_proof(symbol: str, bars: int, risk_pct: float, out_dir: Path):
    df = bi.fetch_data(symbol, bars=bars)
    if df is None or len(df) < 1200:
        raise RuntimeError(f"Not enough history for {symbol}. Need at least 1200 bars.")

    df = df.sort_values("Time").reset_index(drop=True)
    n = len(df)

    interval_minutes = 15
    if len(df) > 1:
        diffs = pd.to_datetime(df["Time"]).diff().dropna()
        if not diffs.empty:
            interval_minutes = max(1, int(diffs.median().total_seconds() // 60))

    available_days = (n * interval_minutes) / (60 * 24)
    windows_days = _build_day_windows(available_days=available_days, count=20)

    runs = []
    equities = []

    for idx, days in enumerate(windows_days):
        bars_window = max(200, int((days * 24 * 60) / interval_minutes))
        bars_window = min(bars_window, n - 20)

        max_start = max(0, n - bars_window - 1)
        start_idx = int((max_start * idx) / max(1, len(windows_days) - 1))
        end_idx = min(n, start_idx + bars_window)

        sub = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
        if len(sub) < 120:
            continue

        result = bi.run_backtest_no_lookahead(sub, symbol, risk_pct=risk_pct)
        metrics = result.get("metrics", {})
        eq = result.get("equity_curve", [])
        equities.append(eq)

        runs.append({
            "run_id": idx + 1,
            "requested_days": int(days),
            "actual_bars": int(len(sub)),
            "start_time": str(pd.Timestamp(sub["Time"].iloc[0])),
            "end_time": str(pd.Timestamp(sub["Time"].iloc[-1])),
            "metrics": {
                "total_trades": int(metrics.get("total_trades", 0) or 0),
                "win_rate": _safe_float(metrics.get("win_rate", 0)),
                "profit_factor": _safe_float(metrics.get("profit_factor", 0)),
                "return_pct": _safe_float(metrics.get("return_pct", 0)),
                "total_profit": _safe_float(metrics.get("total_profit", 0)),
                "max_drawdown": _safe_float(metrics.get("max_drawdown", 0)),
            },
        })

    if len(runs) < 10:
        raise RuntimeError(f"Only {len(runs)} valid runs for {symbol}; need at least 10.")

    returns = np.array([r["metrics"]["return_pct"] for r in runs], dtype=float)
    pfs = np.array([r["metrics"]["profit_factor"] for r in runs], dtype=float)
    dds = np.array([r["metrics"]["max_drawdown"] for r in runs], dtype=float)
    wins = int(np.sum(returns > 0))
    n_runs = int(len(runs))

    luck_p_value = _binomial_tail_probability_at_least_k(wins, n_runs, 0.5)
    positive_profit_factor_runs = int(np.sum(pfs >= 1.0))

    profits = np.array([r["metrics"]["total_profit"] for r in runs], dtype=float)
    positive_total = float(np.sum(np.clip(profits, 0, None)))
    concentration = float(np.max(np.clip(profits, 0, None)) / positive_total) if positive_total > 0 else 1.0

    summary = {
        "symbol": symbol,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "input": {
            "bars": bars,
            "risk_pct": risk_pct,
            "interval_minutes": interval_minutes,
            "available_days_estimate": round(available_days, 2),
            "runs": n_runs,
        },
        "consistency": {
            "profitable_runs": wins,
            "runs_total": n_runs,
            "consistency_pct": round((wins / n_runs) * 100.0, 2),
            "profit_factor_ge_1_runs": positive_profit_factor_runs,
        },
        "distribution": {
            "mean_return_pct": round(float(np.mean(returns)), 4),
            "median_return_pct": round(float(np.median(returns)), 4),
            "std_return_pct": round(float(np.std(returns)), 4),
            "p10_return_pct": round(float(np.percentile(returns, 10)), 4),
            "p90_return_pct": round(float(np.percentile(returns, 90)), 4),
            "mean_pf": round(float(np.mean(pfs)), 4),
            "median_pf": round(float(np.median(pfs)), 4),
            "mean_drawdown": round(float(np.mean(dds)), 4),
        },
        "anti_luck": {
            "binomial_p_value": round(float(luck_p_value), 6),
            "interpretation": "strong_evidence_not_luck" if luck_p_value < 0.05 else "insufficient_statistical_evidence",
        },
        "overfitting_risk": {
            "profit_concentration_ratio": round(float(concentration), 4),
            "interpretation": "lower_risk" if concentration <= 0.35 else "higher_risk",
        },
        "verdict": "robust" if (wins >= int(math.ceil(n_runs * 0.6)) and float(np.median(returns)) > 0 and luck_p_value < 0.05) else "needs_improvement",
        "runs": runs,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    file_tag = f"{symbol}_bars{bars}_{stamp}"
    json_path = out_dir / f"overfit_proof_20_{file_tag}.json"
    csv_path = out_dir / f"overfit_proof_20_{file_tag}.csv"
    overlay_png = out_dir / f"overfit_proof_20_{file_tag}_equity_overlay.png"
    hist_png = out_dir / f"overfit_proof_20_{file_tag}_returns_hist.png"

    json_path.write_text(json.dumps(summary, indent=2, default=str))

    rows = []
    for r in runs:
        m = r["metrics"]
        rows.append({
            "run_id": r["run_id"],
            "requested_days": r["requested_days"],
            "actual_bars": r["actual_bars"],
            "start_time": r["start_time"],
            "end_time": r["end_time"],
            "trades": m["total_trades"],
            "win_rate": m["win_rate"],
            "profit_factor": m["profit_factor"],
            "return_pct": m["return_pct"],
            "total_profit": m["total_profit"],
            "max_drawdown": m["max_drawdown"],
        })
    pd.DataFrame(rows).to_csv(csv_path, index=False)

    _plot_overlay_equity(equities, overlay_png)
    _plot_return_hist(returns, hist_png)

    print(f"\n=== OVERFIT PROOF ({symbol}) ===")
    print(f"Runs: {n_runs}")
    print(f"Profitable runs: {wins}/{n_runs} ({summary['consistency']['consistency_pct']}%)")
    print(f"Median return: {summary['distribution']['median_return_pct']:+.2f}%")
    print(f"Median PF: {summary['distribution']['median_pf']:.2f}")
    print(f"Binomial p-value: {summary['anti_luck']['binomial_p_value']}")
    print(f"Verdict: {summary['verdict']}")
    print(f"Saved JSON: {json_path}")
    print(f"Saved CSV: {csv_path}")
    print(f"Saved equity overlay: {overlay_png}")
    print(f"Saved returns histogram: {hist_png}")


def main():
    ap = argparse.ArgumentParser(description="20-run anti-overfitting proof backtest")
    ap.add_argument("--symbol", default="XAUUSD", help="Trading symbol")
    ap.add_argument("--bars", type=int, default=60000, help="Bars to fetch from history")
    ap.add_argument("--risk_pct", type=float, default=1.0, help="Risk per trade in percent")
    ap.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "backtest_results" / "overfit_proof_20"),
        help="Output folder",
    )
    args = ap.parse_args()

    symbol = str(args.symbol or "XAUUSD").upper()
    bars = max(10000, int(args.bars))
    risk_pct = max(0.1, min(10.0, float(args.risk_pct)))
    out_dir = Path(args.output_dir)

    run_overfit_proof(symbol=symbol, bars=bars, risk_pct=risk_pct, out_dir=out_dir)


if __name__ == "__main__":
    main()
