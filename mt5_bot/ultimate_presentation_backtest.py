import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import backtest_improved as bi


def plot_equity(points, title, out_path):
    if not points or len(points) < 2:
        return
    x = list(range(1, len(points) + 1))
    fig = plt.figure(figsize=(10, 4), dpi=130)
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, points, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Step")
    ax.set_ylabel("Balance")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def build_verdict(std_metrics, robust):
    ret = float(std_metrics.get("return_pct", 0.0) or 0.0)
    pf = float(std_metrics.get("profit_factor", 0.0) or 0.0)
    max_dd = float(std_metrics.get("max_drawdown", 0.0) or 0.0)
    consistency = float(robust.get("consistency", 0.0) or 0.0)

    checks = {
        "profitable_standard": ret > 0,
        "profit_factor_ok": pf >= 1.1,
        "drawdown_ok": max_dd <= 2000.0,
        "robustness_ok": consistency >= 55.0,
    }
    score = round(sum(1 for v in checks.values() if v) / len(checks) * 100.0, 2)
    if score >= 75:
        verdict = "presentation_ready"
    elif score >= 50:
        verdict = "promising_but_risky"
    else:
        verdict = "needs_more_tuning"

    return checks, score, verdict


def run_ultimate(args):
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbols = sorted(bi.INSTRUMENTS.keys()) if args.symbol == "ALL" else [args.symbol]

    summary_rows = []
    robustness_rows = []
    package = {
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "engine": "backtest_improved.py",
        "mode": "ultimate_presentation_backtest",
        "settings": {
            "risk_pct": args.risk_pct,
            "periods": args.periods,
            "bars": args.bars,
        },
        "symbols": {},
    }

    for sym in symbols:
        print(f"\n=== {sym} ===", flush=True)
        df = bi.fetch_data(sym, bars=args.bars)
        if df is None or len(df) < 350:
            print("  SKIP: not enough data", flush=True)
            package["symbols"][sym] = {"error": "not_enough_data"}
            continue

        std = bi.run_backtest_no_lookahead(df, sym, risk_pct=args.risk_pct)
        robust = bi.run_robustness_20_periods(df, sym, periods=args.periods, risk_pct=args.risk_pct)

        std_m = std.get("metrics", {})
        checks, score, verdict = build_verdict(std_m, robust)

        std_eq = std.get("equity_curve", [])
        rob_eq = robust.get("combined_equity_curve", [])

        plot_equity(std_eq, f"{sym} Standard Equity", out_dir / f"equity_standard_{sym}.png")
        plot_equity(rob_eq, f"{sym} Robustness ({args.periods} periods)", out_dir / f"equity_robust_{sym}.png")

        symbol_payload = {
            "standard": std,
            "robustness_20": robust,
            "checks": checks,
            "presentation_score": score,
            "verdict": verdict,
        }
        package["symbols"][sym] = symbol_payload

        summary_rows.append({
            "symbol": sym,
            "trades": int(std_m.get("total_trades", 0) or 0),
            "win_rate": float(std_m.get("win_rate", 0) or 0),
            "profit_factor": float(std_m.get("profit_factor", 0) or 0),
            "return_pct": float(std_m.get("return_pct", 0) or 0),
            "max_drawdown": float(std_m.get("max_drawdown", 0) or 0),
            "robustness_consistency": float(robust.get("consistency", 0) or 0),
            "robust_periods_run": int(robust.get("periods_run", 0) or 0),
            "presentation_score": score,
            "verdict": verdict,
        })

        for p in robust.get("period_results", []):
            m = p.get("metrics", {})
            robustness_rows.append({
                "symbol": sym,
                "period_index": int(p.get("period_index", 0) or 0),
                "start_time": p.get("start_time"),
                "end_time": p.get("end_time"),
                "trades": int(m.get("total_trades", 0) or 0),
                "win_rate": float(m.get("win_rate", 0) or 0),
                "return_pct": float(m.get("return_pct", 0) or 0),
                "profit_factor": float(m.get("profit_factor", 0) or 0),
                "max_drawdown": float(m.get("max_drawdown", 0) or 0),
            })

        print(
            f"  Standard: {std_m.get('total_trades', 0)}t "
            f"WR={std_m.get('win_rate', 0):.1f}% PF={std_m.get('profit_factor', 0):.2f} "
            f"Ret={std_m.get('return_pct', 0):+.2f}%"
        )
        print(
            f"  Robustness: periods={robust.get('periods_run', 0)} "
            f"consistency={robust.get('consistency', 0):.1f}% "
            f"score={score:.1f} verdict={verdict}"
        )

    summary_df = pd.DataFrame(summary_rows)
    robust_df = pd.DataFrame(robustness_rows)

    if not summary_df.empty:
        summary_df = summary_df.sort_values(["presentation_score", "return_pct"], ascending=[False, False])
        summary_df.to_csv(out_dir / "ultimate_summary_table.csv", index=False)

    if not robust_df.empty:
        robust_df = robust_df.sort_values(["symbol", "period_index"])
        robust_df.to_csv(out_dir / "ultimate_robustness_periods.csv", index=False)

    package["portfolio"] = {
        "symbols_tested": int(len(summary_df)),
        "presentation_ready_count": int((summary_df["verdict"] == "presentation_ready").sum()) if not summary_df.empty else 0,
        "avg_return_pct": float(summary_df["return_pct"].mean()) if not summary_df.empty else 0.0,
        "avg_pf": float(summary_df["profit_factor"].mean()) if not summary_df.empty else 0.0,
        "avg_robustness": float(summary_df["robustness_consistency"].mean()) if not summary_df.empty else 0.0,
    }

    (out_dir / "ultimate_presentation_package.json").write_text(json.dumps(package, indent=2, default=str))

    print("\nSaved:")
    print(f"  - {out_dir / 'ultimate_presentation_package.json'}")
    if not summary_df.empty:
        print(f"  - {out_dir / 'ultimate_summary_table.csv'}")
    if not robust_df.empty:
        print(f"  - {out_dir / 'ultimate_robustness_periods.csv'}")


def main():
    ap = argparse.ArgumentParser(description="Ultimate backtest runner for presentation")
    ap.add_argument("--symbol", default="ALL", help="Symbol or ALL")
    ap.add_argument("--risk_pct", type=float, default=1.0, help="Risk % per trade")
    ap.add_argument("--periods", type=int, default=20, help="Robustness periods")
    ap.add_argument("--bars", type=int, default=12000, help="Bars to fetch")
    ap.add_argument(
        "--output_dir",
        default=str(Path(__file__).parent / "backtest_results" / "presentation_ultimate"),
        help="Output directory",
    )
    args = ap.parse_args()

    args.symbol = str(args.symbol or "ALL").upper()
    args.risk_pct = max(0.1, min(10.0, float(args.risk_pct)))
    args.periods = max(5, min(40, int(args.periods)))
    args.bars = max(3000, int(args.bars))

    run_ultimate(args)


if __name__ == "__main__":
    main()
