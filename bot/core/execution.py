"""Risk-aware execution engine with IOC routing."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import Instrument, SETTINGS
from core.broker import SimulatedBroker
from core.logger import get_logger
from core.order_types import IOCOrder, OrderSide, StrategySignal
from core.risk import RiskManager
from core.utils import Candle


trade_logger = get_logger("trades")
system_logger = get_logger("system")


@dataclass(slots=True)
class Position:
    """Represents an open trade with SL/TP, including breakeven state."""

    instrument: Instrument
    side: OrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    session: str
    break_even_moved: bool = False


@dataclass(slots=True)
class TradeRecord:
    """Completed trade used for reporting and analytics."""

    instrument: str
    side: OrderSide
    quantity: float
    entry_price: float
    exit_price: float
    opened_at: datetime
    closed_at: datetime
    pnl: float
    r_multiple: float
    session: str


@dataclass(slots=True)
class SessionStats:
    """Aggregated stats per session window."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    total_r: float = 0.0


class ExecutionEngine:
    """Sizes signals, routes IOC orders, and tracks trade lifecycle."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        broker: Optional[SimulatedBroker] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_balance: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
    ) -> None:
        self.instrument = instrument
        self.broker = broker or SimulatedBroker(instrument)
        starting_balance = initial_balance or SETTINGS.initial_balance
        self.balance = starting_balance
        self.equity = starting_balance
        self.risk = risk_manager or RiskManager(starting_balance, risk_per_trade=risk_per_trade)
        self.risk.set_balance(self.balance)
        self.position: Optional[Position] = None
        self.trade_history: List[TradeRecord] = []
        self.session_stats: Dict[str, SessionStats] = {}

    async def handle_signal(self, signal: StrategySignal, candle: Candle) -> None:
        """Validate risk limits, size, and submit an IOC order for the signal."""

        if signal.instrument != self.instrument.symbol or self.position is not None:
            return

        allowed, reason = self.risk.evaluate(signal.timestamp, signal.session)
        if not allowed:
            system_logger.info("Signal blocked: %s", reason)
            return

        quantity = self.risk.size_position(self.instrument, signal.entry, signal.stop_loss)
        if quantity <= 0:
            return

        latency = random.uniform(*SETTINGS.execution.latency_ms) / 1000
        await asyncio.sleep(latency)

        order = IOCOrder(
            instrument=signal.instrument,
            side=signal.side,
            quantity=quantity,
            timestamp=signal.timestamp,
            limit_price=signal.entry if SETTINGS.execution.enforce_entry_limits else None,
        )
        fill = await self.broker.execute_ioc(order, candle)
        if fill.status != "filled" or fill.price is None:
            system_logger.info("IOC cancelled: %s", fill.reason)
            return

        self.position = Position(
            instrument=self.instrument,
            side=signal.side,
            quantity=fill.filled_quantity,
            entry_price=fill.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=candle.timestamp,
            session=signal.session,
        )
        trade_logger.info(
            "OPEN | %s | qty=%.2f entry=%.5f sl=%.5f tp=%.5f",
            signal.side.value.upper(),
            fill.filled_quantity,
            fill.price,
            signal.stop_loss,
            signal.take_profit,
        )

    def update_position(self, candle: Candle) -> Optional[TradeRecord]:
        """Advance trailing stops, detect exits, and book closed trades."""

        if self.position is None:
            return None

        pos = self.position
        exit_price: Optional[float] = None
        if pos.side is OrderSide.BUY:
            if candle.low <= pos.stop_loss:
                exit_price = pos.stop_loss
            elif candle.high >= pos.take_profit:
                exit_price = pos.take_profit
            elif not pos.break_even_moved and candle.high - pos.entry_price >= pos.entry_price - pos.stop_loss:
                pos.stop_loss = pos.entry_price
                pos.break_even_moved = True
        else:
            if candle.high >= pos.stop_loss:
                exit_price = pos.stop_loss
            elif candle.low <= pos.take_profit:
                exit_price = pos.take_profit
            elif not pos.break_even_moved and pos.entry_price - candle.low >= pos.stop_loss - pos.entry_price:
                pos.stop_loss = pos.entry_price
                pos.break_even_moved = True

        if exit_price is None:
            return None

        pnl = self._pnl(pos, exit_price)
        r_multiple = self._r_multiple(pos, pnl)
        self.balance += pnl
        self.equity = self.balance
        self.risk.register_trade(pnl, pos.session)
        self.risk.set_balance(self.balance)

        trade = TradeRecord(
            instrument=pos.instrument.symbol,
            side=pos.side,
            quantity=pos.quantity,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            opened_at=pos.opened_at,
            closed_at=candle.timestamp,
            pnl=pnl,
            r_multiple=r_multiple,
            session=pos.session,
        )
        self.trade_history.append(trade)
        self._update_session_stats(trade)
        trade_logger.info(
            "CLOSE | %s | qty=%.2f exit=%.5f pnl=%.2f R=%.2f",
            pos.side.value.upper(),
            pos.quantity,
            exit_price,
            pnl,
            r_multiple,
        )
        self.position = None
        return trade

    def _pnl(self, pos: Position, exit_price: float) -> float:
        direction = 1 if pos.side is OrderSide.BUY else -1
        ticks = (exit_price - pos.entry_price) * direction / self.instrument.tick_size
        return ticks * self.instrument.pip_value * (pos.quantity / self.instrument.contract_size)

    def _r_multiple(self, pos: Position, pnl: float) -> float:
        risk_ticks = abs(pos.entry_price - pos.stop_loss) / self.instrument.tick_size
        risk = risk_ticks * self.instrument.pip_value * (pos.quantity / self.instrument.contract_size)
        return pnl / risk if risk else 0.0

    def _update_session_stats(self, trade: TradeRecord) -> None:
        stats = self.session_stats.setdefault(trade.session, SessionStats())
        stats.trades += 1
        stats.pnl += trade.pnl
        stats.total_r += trade.r_multiple
        if trade.pnl > 0:
            stats.wins += 1
        else:
            stats.losses += 1


__all__ = ["ExecutionEngine", "TradeRecord", "SessionStats", "Position"]