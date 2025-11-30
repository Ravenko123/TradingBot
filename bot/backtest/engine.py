"""Sequential candle-by-candle backtest engine."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from config.settings import SETTINGS
from core.broker import SimulatedBroker
from core.execution import ExecutionEngine, SessionStats, TradeRecord
from core.logger import get_logger
from core.utils import Candle, load_candles
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
    data_start: datetime
    data_end: datetime
    candle_count: int


class BacktestEngine:
    """Simulates trade lifecycle across historical candles."""

    def __init__(
        self,
        instrument: str,
        data_path: Optional[str] = None,
        strategy_name: str = "ict_smc",
        spread: Optional[float] = None,
        slippage: Optional[float] = None,
        *,
        initial_balance: Optional[float] = None,
        strategy_params: Optional[Dict[str, Any]] = None,
        start_timestamp: Optional[datetime] = None,
        end_timestamp: Optional[datetime] = None,
        risk_per_trade: Optional[float] = None,
        lookback_days: Optional[int] = None,
        show_progress: bool = False,
        candles: Optional[List[Candle]] = None,
    ) -> None:
        self.data_path = data_path
        self.instrument_key = instrument
        self.strategy_name = strategy_name
        self.spread = spread
        self.slippage = slippage
        self.initial_balance = initial_balance or SETTINGS.initial_balance
        self.strategy_params = strategy_params or {}
        self.start_timestamp = start_timestamp
        self.end_timestamp = end_timestamp
        self.risk_per_trade = risk_per_trade
        self.lookback_days = lookback_days
        self.show_progress = show_progress
        self._candles_override = candles

    async def run(self) -> BacktestResult:
        candles = self._slice_candles(self._load_candles())
        if not candles:
            raise RuntimeError("No candles available after applying the selected lookback window.")
        instrument = SETTINGS.allowed_instruments[self.instrument_key]
        broker = SimulatedBroker(instrument, spread=self.spread, slippage=self.slippage)
        engine = ExecutionEngine(
            instrument,
            broker=broker,
            initial_balance=self.initial_balance,
            risk_per_trade=self.risk_per_trade,
        )
        context = StrategyContext(
            instrument=instrument.symbol,
            risk_per_trade=self.risk_per_trade or SETTINGS.risk.risk_per_trade,
            session_windows=SETTINGS.sessions,
            parameters=self.strategy_params,
        )
        strategy = StrategyRegistry.create(self.strategy_name, context)

        equity_curve: List[float] = [engine.balance]
        timestamps: List[str] = []

        for candle in self._iterate_with_progress(candles):
            timestamps.append(candle.timestamp.isoformat())
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
            data_start=candles[0].timestamp,
            data_end=candles[-1].timestamp,
            candle_count=len(candles),
        )

    def _slice_candles(self, candles: List[Candle]) -> List[Candle]:
        if not candles:
            return []
        start = self.start_timestamp
        end = self.end_timestamp
        if self.lookback_days and len(candles) > 0:
            cutoff = candles[-1].timestamp - timedelta(days=self.lookback_days)
            start = cutoff if start is None or cutoff > start else start
        if not start and not end:
            return candles
        filtered: List[Candle] = []
        for candle in candles:
            if start and candle.timestamp < start:
                continue
            if end and candle.timestamp > end:
                continue
            filtered.append(candle)
        return filtered

    def _load_candles(self) -> List[Candle]:
        if self._candles_override is not None:
            return list(self._candles_override)
        if not self.data_path:
            raise RuntimeError("BacktestEngine requires either candles or a data_path.")
        return load_candles(self.data_path)

    def _iterate_with_progress(self, candles: List[Candle]):
        if not self.show_progress or not candles:
            for candle in candles:
                yield candle
            return
        total = len(candles)
        try:
            from tqdm import tqdm  # type: ignore

            with tqdm(total=total, desc=f"{self.instrument_key} candles", unit="bar") as bar:
                for candle in candles:
                    yield candle
                    bar.update(1)
        except Exception:
            step = max(1, total // 50)
            for idx, candle in enumerate(candles, start=1):
                yield candle
                if idx % step == 0 or idx == total:
                    percent = idx / total * 100
                    print(
                        f"\rProcessing {self.instrument_key}: {percent:5.1f}% ({idx}/{total})",
                        end="",
                        flush=True,
                    )
            print()


def run_backtest_sync(engine: BacktestEngine) -> BacktestResult:
    """Helper for running the async engine from sync contexts."""

    return asyncio.run(engine.run())


__all__ = ["BacktestEngine", "BacktestResult", "run_backtest_sync"]