"""Risk management and trade lifecycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional

from config.settings import Instrument, SETTINGS
from core.broker import FillResult, Order, OrderSide, SimulatedBroker
from core.logger import get_logger
from core.utils import Candle, pip_distance


trade_logger = get_logger("trades")
system_logger = get_logger("system")


@dataclass
class StrategySignal:
    """Instruction emitted by a strategy after processing a candle."""

    instrument: str
    side: OrderSide
    entry: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    confidence: float
    session_name: Optional[str] = None


@dataclass
class Position:
    """Represents an open position."""

    instrument: str
    side: OrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    break_even_moved: bool = False
    session_name: Optional[str] = None


@dataclass
class TradeRecord:
    """Historical trade used for analytics."""

    position: Position
    closed_at: datetime
    exit_price: float
    pnl: float
    r_multiple: float
    session_name: Optional[str]


@dataclass
class SessionStats:
    """Aggregated performance per session."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    total_r: float = 0.0
    pnl: float = 0.0


@dataclass
class ExecutionEngine:
    """Handles sizing, broker routing, and position monitoring."""

    broker: SimulatedBroker
    instrument: Instrument
    balance: float = field(default=SETTINGS.initial_balance)
    equity: float = field(init=False)
    open_position: Optional[Position] = field(default=None, init=False)
    trade_history: List[TradeRecord] = field(default_factory=list, init=False)
    daily_loss: float = field(default=0.0, init=False)
    last_signal_minute: Optional[int] = field(default=None, init=False)
    session_stats: Dict[str, SessionStats] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.equity = self.balance

    async def handle_signal(self, signal: StrategySignal, candle: Candle) -> Optional[TradeRecord]:
        """Validate limits and route an IOC order if permissible."""

        if signal.instrument not in SETTINGS.allowed_instruments:
            system_logger.warning("Instrument %s not allowed", signal.instrument)
            return None

        minute = signal.timestamp.minute
        if self.last_signal_minute == minute:
            return None  # enforce 1-minute entries
        self.last_signal_minute = minute

        if self._risk_limits_breached():
            system_logger.warning("Risk limits reached; skipping new trades")
            return None

        qty = self._position_size(signal)
        if qty <= 0:
            return None

        order = Order(instrument=signal.instrument, side=signal.side, quantity=qty, timestamp=signal.timestamp)
        fill = await self.broker.submit_order(order, candle)
        if not fill.filled or fill.fill_price is None:
            system_logger.warning("Order rejected: %s", fill.reason)
            return None

        self.open_position = Position(
            instrument=signal.instrument,
            side=signal.side,
            quantity=qty,
            entry_price=fill.fill_price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=signal.timestamp,
            session_name=signal.session_name,
        )
        trade_logger.info(
            "OPEN | %s | qty=%.2f entry=%.5f sl=%.5f tp=%.5f",
            signal.side.value.upper(),
            qty,
            fill.fill_price,
            signal.stop_loss,
            signal.take_profit,
        )
        return None

    def update_position(self, candle: Candle) -> Optional[TradeRecord]:
        """Evaluate open trade for stops, targets, or break-even logic."""

        if not self.open_position:
            return None

        pos = self.open_position
        exit_price: Optional[float] = None

        if pos.side == OrderSide.BUY:
            if candle["low"] <= pos.stop_loss:
                exit_price = pos.stop_loss
            elif candle["high"] >= pos.take_profit:
                exit_price = pos.take_profit
            elif not pos.break_even_moved and candle["high"] - pos.entry_price >= (pos.entry_price - pos.stop_loss):
                pos.stop_loss = pos.entry_price
                pos.break_even_moved = True
        else:
            if candle["high"] >= pos.stop_loss:
                exit_price = pos.stop_loss
            elif candle["low"] <= pos.take_profit:
                exit_price = pos.take_profit
            elif not pos.break_even_moved and pos.entry_price - candle["low"] >= (pos.stop_loss - pos.entry_price):
                pos.stop_loss = pos.entry_price
                pos.break_even_moved = True

        if exit_price is None:
            return None

        pnl = self._calculate_pnl(pos, exit_price)
        r_multiple = self._calculate_r_multiple(pos, pnl)
        self.balance += pnl
        self.equity = self.balance
        self.daily_loss = min(self.daily_loss, 0) + (pnl if pnl < 0 else 0)
        record = TradeRecord(
            position=pos,
            closed_at=candle["timestamp"],
            exit_price=exit_price,
            pnl=pnl,
            r_multiple=r_multiple,
            session_name=pos.session_name,
        )
        self.trade_history.append(record)
        self._update_session_stats(record)
        trade_logger.info(
            "CLOSE | %s | qty=%.2f exit=%.5f pnl=%.2f R=%.2f session=%s",
            pos.side.value.upper(),
            pos.quantity,
            exit_price,
            pnl,
            r_multiple,
            pos.session_name or "-",
        )
        self.open_position = None
        return record

    def _position_size(self, signal: StrategySignal) -> float:
        """Size a trade based on account balance and stop distance."""

        risk_capital = self.balance * SETTINGS.risk_per_trade
        stop_distance = abs(signal.entry - signal.stop_loss)
        if stop_distance <= 0:
            return 0.0
        pip_size = self.instrument.min_tick
        pip_count = pip_distance(signal.entry, signal.stop_loss, pip_size)
        risk_per_lot = pip_count * self.instrument.pip_value
        if risk_per_lot == 0:
            return 0.0
        contracts = risk_capital / risk_per_lot
        quantity = contracts * self.instrument.contract_size
        return max(quantity, 0.0)

    def _risk_limits_breached(self) -> bool:
        """Check if any high-level risk control has been breached."""

        daily_limit = -SETTINGS.max_daily_loss_pct * SETTINGS.initial_balance
        if self.daily_loss <= daily_limit:
            return True
        equity_floor = SETTINGS.initial_balance * (1 - SETTINGS.equity_depletion_stop_pct)
        return self.equity <= equity_floor

    def _calculate_pnl(self, position: Position, exit_price: float) -> float:
        """Calculate PnL for the trade."""

        direction = 1 if position.side == OrderSide.BUY else -1
        price_diff = (exit_price - position.entry_price) * direction
        return (price_diff / self.instrument.min_tick) * self.instrument.pip_value * (position.quantity / self.instrument.contract_size)

    def _risk_value(self, position: Position) -> float:
        """Monetary risk based on entry and stop distance."""

        stop_distance = abs(position.entry_price - position.stop_loss)
        if stop_distance <= 0:
            return 0.0
        return (stop_distance / self.instrument.min_tick) * self.instrument.pip_value * (
            position.quantity / self.instrument.contract_size
        )

    def _calculate_r_multiple(self, position: Position, pnl: float) -> float:
        """Return trade result expressed in R multiples."""

        risk = self._risk_value(position)
        return pnl / risk if risk else 0.0

    def _update_session_stats(self, record: TradeRecord) -> None:
        """Accumulate stats per trading session."""

        session = record.session_name or "Unspecified"
        stats = self.session_stats.setdefault(session, SessionStats())
        stats.trades += 1
        stats.pnl += record.pnl
        stats.total_r += record.r_multiple
        if record.pnl > 0:
            stats.wins += 1
        else:
            stats.losses += 1


__all__ = ["ExecutionEngine", "StrategySignal", "TradeRecord", "Position", "SessionStats"]