"""Sequential candle-by-candle backtest engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional

from config.settings import SETTINGS
from core.broker import SimulatedBroker
from core.execution import ExecutionEngine, SessionStats, TradeRecord
from core.logger import get_logger
from core.utils import load_candles
from strategies import StrategyContext, StrategyRegistry

from .metrics import PerformanceReport, compute_performance


system_logger = get_logger("system")


@dataclass
class BacktestResult:
    """Container for engine outputs."""

    report: PerformanceReport
    equity_curve: List[float]
    timestamps: List[str]
    session_stats: Dict[str, SessionStats]
    trades: List[TradeRecord]


class BacktestEngine:
    """Simulates trade lifecycle across historical candles."""

    def __init__(
        self,
        data_path: str,
        instrument: str,
        strategy_name: str = "ict_smc",
        spread: Optional[float] = None,
        slippage: Optional[float] = None,
    ) -> None:
        self.data_path = data_path
        self.instrument_key = instrument
        self.strategy_name = strategy_name
        self.spread = spread
        self.slippage = slippage

    async def run(self) -> BacktestResult:
        candles = load_candles(self.data_path)
        instrument = SETTINGS.allowed_instruments[self.instrument_key]
        broker = SimulatedBroker(spread=self.spread, slippage=self.slippage)
        engine = ExecutionEngine(broker=broker, instrument=instrument)
        context = StrategyContext(
            instrument=instrument.symbol,
            risk_per_trade=SETTINGS.risk_per_trade,
            session_windows=SETTINGS.sessions,
        )
        strategy = StrategyRegistry.create(self.strategy_name, context)

        equity_curve: List[float] = [engine.balance]
        timestamps: List[str] = []

        for candle in candles:
            timestamps.append(candle["timestamp"].isoformat())
            signal = strategy.on_candle(candle)
            if signal:
                await engine.handle_signal(signal, candle)
            engine.update_position(candle)
            equity_curve.append(engine.equity)

        report = compute_performance(equity_curve, engine.trade_history)
        system_logger.info("Backtest complete | %s", report.as_dict())
        return BacktestResult(
            report=report,
            equity_curve=equity_curve,
            timestamps=timestamps,
            session_stats=engine.session_stats,
            trades=engine.trade_history,
        )


def run_backtest_sync(engine: BacktestEngine) -> BacktestResult:
    """Helper for running the async engine from sync contexts."""

    return asyncio.run(engine.run())


__all__ = ["BacktestEngine", "BacktestResult", "run_backtest_sync"]