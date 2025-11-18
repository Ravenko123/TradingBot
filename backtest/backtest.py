"""CLI helper for running the ICT SMC backtest engine."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine, run_backtest_sync  # type: ignore  # noqa: E402
from backtest.metrics import (  # type: ignore  # noqa: E402
    export_equity_curve,
    export_report_json,
    export_trades_csv,
    plot_equity_curve,
)
from core.logger import configure_logging, get_logger  # type: ignore  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run ICT SMC backtests")
    parser.add_argument("--data", required=True, help="Path to CSV data feed")
    parser.add_argument("--instrument", default="EURUSD", help="Instrument symbol")
    parser.add_argument("--strategy", default="ict_smc", help="Strategy name registered in the factory")
    parser.add_argument("--export", help="Optional CSV export path for equity curve")
    parser.add_argument("--plot", help="Optional PNG path for equity curve visualization")
    parser.add_argument("--spread", type=float, help="Override broker spread (in price units)")
    parser.add_argument("--slippage", type=float, help="Override broker slippage (in price units)")
    parser.add_argument("--trades-csv", help="Optional trade blotter CSV export path")
    parser.add_argument("--report-json", help="Optional JSON file for performance + session stats")
    return parser


def main() -> None:
    configure_logging()
    system_logger = get_logger("system")
    error_logger = get_logger("errors")
    args = build_parser().parse_args()
    try:
        engine = BacktestEngine(
            data_path=args.data,
            instrument=args.instrument,
            strategy_name=args.strategy,
            spread=args.spread,
            slippage=args.slippage,
        )
        result = run_backtest_sync(engine)
        system_logger.info("Backtest CLI finished | instrument=%s strategy=%s", args.instrument, args.strategy)
        print("Performance Report:")
        for key, value in result.report.as_dict().items():
            print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
        if result.session_stats:
            print("Session Stats:")
            for session, stats in result.session_stats.items():
                win_rate = (stats.wins / stats.trades * 100) if stats.trades else 0.0
                avg_r = stats.total_r / stats.trades if stats.trades else 0.0
                print(
                    f"  {session}: trades={stats.trades} win_rate={win_rate:.2f}% avg_R={avg_r:.2f} pnl={stats.pnl:.2f}"
                )
        if args.export:
            export_equity_curve(Path(args.export), result.timestamps, result.equity_curve)
            print(f"Equity curve exported to {args.export}")
        if args.plot:
            plot_equity_curve(Path(args.plot), result.timestamps, result.equity_curve)
            print(f"Equity chart saved to {args.plot}")
        if args.trades_csv:
            export_trades_csv(Path(args.trades_csv), result.trades)
            if result.trades:
                print(f"Trades exported to {args.trades_csv}")
            else:
                print(f"Trades CSV created at {args.trades_csv} (no executions this run)")
        if args.report_json:
            export_report_json(Path(args.report_json), result.report, result.session_stats)
            print(f"Report JSON saved to {args.report_json}")
    except Exception:
        error_logger.exception("Backtest CLI failed")
        raise


if __name__ == "__main__":  # pragma: no cover
    main()