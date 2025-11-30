"""Entry point for running the ICT SMC trading bot."""

from __future__ import annotations

import argparse
import asyncio
from pathlib import Path
from typing import Optional

from backtest.engine import BacktestEngine, run_backtest_sync
from backtest.metrics import export_equity_curve, export_report_json, export_trades_csv, plot_equity_curve
from config.settings import SETTINGS
from core.broker import SimulatedBroker
from core.datafeed import CSVDataFeed
from core.execution import ExecutionEngine
from core.logger import configure_logging, get_logger
from strategies import StrategyContext, StrategyRegistry


system_logger = get_logger("system")
error_logger = get_logger("errors")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICT SMC Trading Bot")
    parser.add_argument("--data", default="data/sample_data.csv", help="CSV file containing 1m candles")
    parser.add_argument("--instrument", default="EURUSD+", help="Instrument symbol")
    parser.add_argument("--strategy", default="ict_smc", help="Strategy name to run")
    parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="Execution mode")
    parser.add_argument("--speed", type=float, default=0.0, help="Seconds to await between candles in live mode")
    parser.add_argument("--loop-data", action="store_true", help="Loop the data feed endlessly")
    parser.add_argument("--max-candles", type=int, help="Optional cap on number of candles to process in live mode")
    parser.add_argument("--export", help="Optional equity curve export when using backtest mode")
    parser.add_argument("--plot", help="Optional PNG path for equity curve visualization (backtest mode)")
    parser.add_argument("--spread", type=float, help="Override broker spread (price units)")
    parser.add_argument("--slippage", type=float, help="Override broker slippage (price units)")
    parser.add_argument("--trades-csv", help="Export executed trades to CSV in backtest mode")
    parser.add_argument("--report-json", help="Export performance and session stats to JSON in backtest mode")
    return parser


async def run_live(
    data_path: str,
    instrument_key: str,
    strategy_name: str,
    speed: float,
    loop_data: bool,
    max_candles: Optional[int],
    spread: Optional[float],
    slippage: Optional[float],
) -> None:
    instrument = SETTINGS.allowed_instruments[instrument_key]
    broker = SimulatedBroker(instrument, spread=spread, slippage=slippage)
    engine = ExecutionEngine(instrument, broker=broker)
    context = StrategyContext(
        instrument=instrument.symbol,
        risk_per_trade=SETTINGS.risk.risk_per_trade,
        session_windows=SETTINGS.sessions,
    )
    strategy = StrategyRegistry.create(strategy_name, context)

    feed = CSVDataFeed(
        path=data_path,
        speed=speed,
        loop=loop_data,
        max_candles=max_candles,
    )

    async for candle in feed:
        signal = strategy.on_candle(candle)
        if signal:
            await engine.handle_signal(signal, candle)
        engine.update_position(candle)

    system_logger.info("Live session completed | balance=%.2f", engine.balance)


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    Path(args.data).expanduser()
    try:
        if args.mode == "backtest":
            engine = BacktestEngine(
                data_path=args.data,
                instrument=args.instrument,
                strategy_name=args.strategy,
                spread=args.spread,
                slippage=args.slippage,
            )
            result = run_backtest_sync(engine)
            system_logger.info("Backtest mode complete | instrument=%s strategy=%s", args.instrument, args.strategy)
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
                export_equity_curve(args.export, result.timestamps, result.equity_curve)
                print(f"Equity curve exported to {args.export}")
            if args.plot:
                plot_equity_curve(args.plot, result.timestamps, result.equity_curve)
                print(f"Equity chart saved to {args.plot}")
            if args.trades_csv:
                export_trades_csv(args.trades_csv, result.trades)
                if result.trades:
                    print(f"Trades exported to {args.trades_csv}")
                else:
                    print(f"Trades CSV created at {args.trades_csv} (no executions)")
            if args.report_json:
                export_report_json(args.report_json, result.report, result.session_stats)
                print(f"JSON report saved to {args.report_json}")
        else:
            asyncio.run(
                run_live(
                    data_path=args.data,
                    instrument_key=args.instrument,
                    strategy_name=args.strategy,
                    speed=args.speed,
                    loop_data=args.loop_data,
                    max_candles=args.max_candles,
                    spread=args.spread,
                    slippage=args.slippage,
                )
            )
    except Exception:
        error_logger.exception("Fatal error while running mode=%s", args.mode)
        raise


if __name__ == "__main__":  # pragma: no cover
    main()