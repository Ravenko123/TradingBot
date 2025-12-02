"""Risk-aware execution engine with IOC routing."""

from __future__ import annotations

import asyncio
import random
from dataclasses import dataclass, field
from datetime import datetime, time
from typing import Dict, List, Optional

from config.settings import Instrument, SETTINGS
from core.broker import SimulatedBroker, MT5Broker
from core.logger import get_logger
from core.order_types import IOCOrder, OrderSide, StrategySignal
from core.risk import RiskManager
from core.utils import Candle


trade_logger = get_logger("trades")
system_logger = get_logger("system")


# Time filters - DISABLED for now (was blocking valid trades)
# These need to be adjusted to the proper timezone
AVOID_TIMES = [
    # DISABLED - causing issues with MT5 server time vs NY time
    # (time(11, 30), time(13, 0)),  # NY lunch
    # (time(20, 0), time(21, 0)),
    # (time(0, 0), time(6, 0)),
]


@dataclass(slots=True)
class Position:
    """Represents an open trade with advanced trailing stop management."""

    instrument: Instrument
    side: OrderSide
    quantity: float
    entry_price: float
    stop_loss: float
    take_profit: float
    opened_at: datetime
    session: str
    
    # MT5 ticket for live trading
    ticket: int = 0
    
    # Advanced trailing stop stages
    initial_stop: float = 0.0  # Original stop
    break_even_moved: bool = False
    partial_closed: bool = False
    trail_active: bool = False
    
    # Trade quality grade (A+, A, B, C)
    grade: str = "B"
    confluence_score: int = 0
    
    def __post_init__(self):
        if self.initial_stop == 0.0:
            self.initial_stop = self.stop_loss


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
    grade: str = "B"
    is_partial: bool = False


@dataclass(slots=True)
class SessionStats:
    """Aggregated stats per session window."""

    trades: int = 0
    wins: int = 0
    losses: int = 0
    pnl: float = 0.0
    total_r: float = 0.0


class ExecutionEngine:
    """Sizes signals, routes IOC orders, and tracks trade lifecycle.
    
    Advanced features:
    - Multi-stage trailing stop (BE at 1R, trail at 2R)
    - Partial profit taking (50% at 1R)
    - Kelly Criterion position sizing
    - Volatility-based filtering
    - Time-based filtering
    - Trade grading (A+/A/B/C)
    """

    def __init__(
        self,
        instrument: Instrument,
        *,
        broker: Optional[SimulatedBroker | MT5Broker] = None,
        risk_manager: Optional[RiskManager] = None,
        initial_balance: Optional[float] = None,
        risk_per_trade: Optional[float] = None,
        use_kelly: bool = False,
        use_trailing: bool = True,   # ENABLED - trail SL on MT5
        use_partials: bool = True,   # ENABLED - partial closes on MT5
        use_time_filter: bool = True,
    ) -> None:
        self.instrument = instrument
        self.broker = broker or SimulatedBroker(instrument)
        starting_balance = initial_balance or SETTINGS.initial_balance
        self.balance = starting_balance
        self.peak_balance = starting_balance
        self.equity = starting_balance
        self.risk = risk_manager or RiskManager(starting_balance, risk_per_trade=risk_per_trade)
        self.risk.set_balance(self.balance)
        self.position: Optional[Position] = None
        self.trade_history: List[TradeRecord] = []
        self.session_stats: Dict[str, SessionStats] = {}
        
        # Advanced features
        self.use_kelly = use_kelly
        self.use_trailing = use_trailing
        self.use_partials = use_partials
        self.use_time_filter = use_time_filter
        
        # Performance tracking for Kelly
        self._win_count = 0
        self._loss_count = 0
        self._total_win_r = 0.0
        self._total_loss_r = 0.0

    def _is_bad_time(self, timestamp: datetime) -> bool:
        """Check if current time is in a low-quality window."""
        if not self.use_time_filter:
            return False
        current_time = timestamp.time()
        for start, end in AVOID_TIMES:
            if start <= current_time <= end:
                return True
        return False

    def _grade_trade(self, signal: StrategySignal) -> str:
        """Grade trade quality based on confluence score.
        
        A+ (10+): Perfect setup, all factors aligned
        A (7-9): Very strong setup
        B (4-6): Good setup
        C (1-3): Marginal setup, consider skipping
        """
        score = getattr(signal, 'confluence_score', 5)  # Default to B
        confidence = getattr(signal, 'confidence', 0.5)
        
        # Adjust based on confidence
        adjusted_score = score + (confidence - 0.5) * 2
        
        if adjusted_score >= 10:
            return "A+"
        elif adjusted_score >= 7:
            return "A"
        elif adjusted_score >= 4:
            return "B"
        else:
            return "C"

    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion optimal bet fraction.
        
        Kelly % = W - [(1-W) / R]
        Where:
            W = Win probability
            R = Win/Loss ratio (average win / average loss)
        
        We use fractional Kelly (25-50%) to reduce variance.
        """
        total_trades = self._win_count + self._loss_count
        if total_trades < 20:  # Not enough data
            return 1.0  # Use full position size
        
        win_prob = self._win_count / total_trades
        if self._loss_count == 0 or self._win_count == 0:
            return 1.0
        
        avg_win_r = self._total_win_r / self._win_count if self._win_count > 0 else 0
        avg_loss_r = abs(self._total_loss_r / self._loss_count) if self._loss_count > 0 else 1
        
        if avg_loss_r == 0:
            return 1.0
        
        win_loss_ratio = avg_win_r / avg_loss_r
        kelly = win_prob - ((1 - win_prob) / win_loss_ratio)
        
        # Use quarter Kelly for safety (reduces variance)
        kelly = kelly * 0.25
        
        # Cap between 0.1 and 1.0
        return max(0.1, min(1.0, kelly))

    async def handle_signal(self, signal: StrategySignal, candle: Candle) -> None:
        """Validate risk limits, size, and submit an IOC order for the signal."""

        if signal.instrument != self.instrument.symbol or self.position is not None:
            print(f"      [EXEC] Skipped: instrument mismatch or position exists")
            return
        
        # Double-check MT5 for existing position (failsafe)
        from core.broker import has_open_position
        if has_open_position(signal.instrument):
            print(f"      [EXEC] Skipped: MT5 already has position")
            system_logger.debug("Signal blocked: MT5 already has position for %s", signal.instrument)
            return

        # Time filter
        if self._is_bad_time(candle.timestamp):
            print(f"      [EXEC] Skipped: bad time window")
            system_logger.debug("Signal blocked: bad time window")
            return

        # Grade the trade
        grade = self._grade_trade(signal)
        
        # Skip C-grade trades
        if grade == "C":
            print(f"      [EXEC] Skipped: C-grade trade")
            system_logger.debug("Signal blocked: C-grade trade skipped")
            return

        allowed, reason = self.risk.evaluate(signal.timestamp, signal.session)
        if not allowed:
            print(f"      [EXEC] Skipped: risk blocked - {reason}")
            system_logger.info("Signal blocked: %s", reason)
            return

        quantity = self.risk.size_position(self.instrument, signal.entry, signal.stop_loss)
        if quantity <= 0:
            print(f"      [EXEC] Skipped: invalid quantity {quantity}")
            return

        # Apply Kelly Criterion if enabled
        if self.use_kelly:
            kelly_fraction = self._calculate_kelly_fraction()
            quantity *= kelly_fraction

        latency = random.uniform(*SETTINGS.execution.latency_ms) / 1000
        await asyncio.sleep(latency)

        order = IOCOrder(
            instrument=signal.instrument,
            side=signal.side,
            quantity=quantity,
            timestamp=signal.timestamp,
            limit_price=signal.entry if SETTINGS.execution.enforce_entry_limits else None,
            entry_price=signal.entry,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
        )
        fill = await self.broker.execute_ioc(order, candle)
        if fill.status != "filled" or fill.price is None:
            print(f"      [EXEC] IOC not filled: {fill.status} - {fill.reason}")
            system_logger.info("IOC cancelled: %s", fill.reason)
            return
        
        print(f"      [EXEC] Fill successful! Price={fill.price}, Qty={fill.filled_quantity}")

        confluence_score = int(getattr(signal, 'confidence', 0.5) * 10)
        
        self.position = Position(
            instrument=self.instrument,
            side=signal.side,
            quantity=fill.filled_quantity,
            entry_price=fill.price,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            opened_at=candle.timestamp,
            session=signal.session,
            ticket=fill.ticket,  # Store MT5 ticket for trailing/partials
            initial_stop=signal.stop_loss,
            grade=grade,
            confluence_score=confluence_score,
        )
        trade_logger.info(
            "OPEN | %s | qty=%.2f entry=%.5f sl=%.5f tp=%.5f grade=%s",
            signal.side.value.upper(),
            fill.filled_quantity,
            fill.price,
            signal.stop_loss,
            signal.take_profit,
            grade,
        )

    def update_position(self, candle: Candle) -> Optional[TradeRecord]:
        """Advance trailing stops, handle partials, detect exits, and book closed trades."""

        if self.position is None:
            return None

        pos = self.position
        risk = abs(pos.entry_price - pos.initial_stop)
        exit_price: Optional[float] = None
        partial_trade: Optional[TradeRecord] = None
        
        if pos.side is OrderSide.BUY:
            current_profit = candle.high - pos.entry_price
            
            # Check stop loss hit
            if candle.low <= pos.stop_loss:
                exit_price = pos.stop_loss
            # Check take profit hit
            elif candle.high >= pos.take_profit:
                exit_price = pos.take_profit
            else:
                # Advanced trailing stop stages (not aggressive - give room to run)
                if self.use_trailing:
                    old_stop = pos.stop_loss
                    
                    # Stage 1: Breakeven at 2R (give trade room to breathe)
                    if not pos.break_even_moved and current_profit >= risk * 2.0:
                        pos.stop_loss = pos.entry_price + (risk * 0.3)  # Lock in 0.3R profit
                        pos.break_even_moved = True
                        trade_logger.info("TRAIL | BE at 2R, stop moved to %.5f (+0.3R)", pos.stop_loss)
                    
                    # Stage 2: Trail at 3R (lock in gains but give room)
                    elif pos.break_even_moved and current_profit >= risk * 3.0:
                        pos.trail_active = True
                        # Trail 1R behind current high (generous room)
                        new_stop = candle.high - (risk * 1.0)
                        if new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                            trade_logger.info("TRAIL | 3R+ trail, stop at %.5f", pos.stop_loss)
                    
                    # Stage 3: Tighter trailing at 5R+ (big winner)
                    elif pos.trail_active and current_profit >= risk * 5.0:
                        # Trail 0.75R behind at higher profits
                        new_stop = candle.high - (risk * 0.75)
                        if new_stop > pos.stop_loss:
                            pos.stop_loss = new_stop
                    
                    # If SL changed, update on MT5
                    if pos.stop_loss != old_stop and pos.ticket > 0:
                        if hasattr(self.broker, 'modify_position_sl'):
                            self.broker.modify_position_sl(pos.ticket, pos.stop_loss)
                
                # Partial profit taking at 2R (lock in some profit, let rest run)
                # Only do partials if position is large enough (need at least 0.02 to close 0.01 and keep 0.01)
                if self.use_partials and not pos.partial_closed and current_profit >= risk * 2.0 and pos.quantity >= 0.02:
                    if pos.ticket > 0 and hasattr(self.broker, 'partial_close'):
                        # Calculate 30% of position, round down to 0.01, minimum 0.01
                        lots_to_close = max(0.01, round(pos.quantity * 0.3, 2))
                        # Never close more than 50% - keep runners
                        lots_to_close = min(lots_to_close, round(pos.quantity * 0.5, 2))
                        # Make sure we leave at least 0.01 lots
                        lots_to_close = min(lots_to_close, round(pos.quantity - 0.01, 2))
                        
                        if lots_to_close >= 0.01 and self.broker.partial_close(pos.ticket, lots_to_close):
                            pos.quantity -= lots_to_close  # Update our tracked quantity
                            pos.partial_closed = True
                            trade_logger.info("PARTIAL | Closed %.2f lots at 2R, remaining %.2f", lots_to_close, pos.quantity)
                    else:
                        # Simulation mode
                        partial_qty = pos.quantity * 0.3  # Only 30% - keep 70% for runners
                        partial_pnl = self._pnl_for_qty(pos, pos.entry_price + risk * 2.0, partial_qty)
                        pos.quantity -= partial_qty
                        pos.partial_closed = True
                        self.balance += partial_pnl
                        
                        partial_trade = TradeRecord(
                            instrument=pos.instrument.symbol,
                            side=pos.side,
                            quantity=partial_qty,
                            entry_price=pos.entry_price,
                            exit_price=pos.entry_price + risk * 2.0,
                            opened_at=pos.opened_at,
                            closed_at=candle.timestamp,
                            pnl=partial_pnl,
                            r_multiple=2.0,
                            session=pos.session,
                            grade=pos.grade,
                            is_partial=True,
                        )
                        trade_logger.info("PARTIAL | 30%% closed at 2R, pnl=%.2f", partial_pnl)
        else:  # SELL
            current_profit = pos.entry_price - candle.low
            
            if candle.high >= pos.stop_loss:
                exit_price = pos.stop_loss
            elif candle.low <= pos.take_profit:
                exit_price = pos.take_profit
            else:
                if self.use_trailing:
                    old_stop = pos.stop_loss
                    
                    # Stage 1: Breakeven at 2R (give trade room to breathe)
                    if not pos.break_even_moved and current_profit >= risk * 2.0:
                        pos.stop_loss = pos.entry_price - (risk * 0.3)  # Lock in 0.3R profit
                        pos.break_even_moved = True
                        trade_logger.info("TRAIL | BE at 2R, stop moved to %.5f (+0.3R)", pos.stop_loss)
                    
                    # Stage 2: Trail at 3R (lock in gains but give room)
                    elif pos.break_even_moved and current_profit >= risk * 3.0:
                        pos.trail_active = True
                        # Trail 1R behind current low (generous room)
                        new_stop = candle.low + (risk * 1.0)
                        if new_stop < pos.stop_loss:
                            pos.stop_loss = new_stop
                            trade_logger.info("TRAIL | 3R+ trail, stop at %.5f", pos.stop_loss)
                    
                    # Stage 3: Tighter trailing at 5R+ (big winner)
                    elif pos.trail_active and current_profit >= risk * 5.0:
                        # Trail 0.75R behind at higher profits
                        new_stop = candle.low + (risk * 0.75)
                        if new_stop < pos.stop_loss:
                            pos.stop_loss = new_stop
                    
                    # If SL changed, update on MT5
                    if pos.stop_loss != old_stop and pos.ticket > 0:
                        if hasattr(self.broker, 'modify_position_sl'):
                            self.broker.modify_position_sl(pos.ticket, pos.stop_loss)
                
                # Partial at 2R - only if position is large enough
                if self.use_partials and not pos.partial_closed and current_profit >= risk * 2.0 and pos.quantity >= 0.02:
                    if pos.ticket > 0 and hasattr(self.broker, 'partial_close'):
                        # Calculate 30% of position, round down to 0.01, minimum 0.01
                        lots_to_close = max(0.01, round(pos.quantity * 0.3, 2))
                        # Never close more than 50% - keep runners
                        lots_to_close = min(lots_to_close, round(pos.quantity * 0.5, 2))
                        # Make sure we leave at least 0.01 lots
                        lots_to_close = min(lots_to_close, round(pos.quantity - 0.01, 2))
                        
                        if lots_to_close >= 0.01 and self.broker.partial_close(pos.ticket, lots_to_close):
                            pos.quantity -= lots_to_close  # Update our tracked quantity
                            pos.partial_closed = True
                            trade_logger.info("PARTIAL | Closed %.2f lots at 2R, remaining %.2f", lots_to_close, pos.quantity)
                    else:
                        partial_qty = pos.quantity * 0.3
                        partial_pnl = self._pnl_for_qty(pos, pos.entry_price - risk * 2.0, partial_qty)
                        pos.quantity -= partial_qty
                        pos.partial_closed = True
                        self.balance += partial_pnl
                        
                        partial_trade = TradeRecord(
                            instrument=pos.instrument.symbol,
                            side=pos.side,
                            quantity=partial_qty,
                            entry_price=pos.entry_price,
                            exit_price=pos.entry_price - risk * 2.0,
                            opened_at=pos.opened_at,
                            closed_at=candle.timestamp,
                            pnl=partial_pnl,
                            r_multiple=2.0,
                            session=pos.session,
                            grade=pos.grade,
                            is_partial=True,
                        )
                        trade_logger.info("PARTIAL | 30%% closed at 2R, pnl=%.2f", partial_pnl)

        # Record partial if it occurred
        if partial_trade:
            self.trade_history.append(partial_trade)
            self._update_kelly_stats(2.0)  # 2R win for partial

        if exit_price is None:
            return None

        pnl = self._pnl(pos, exit_price)
        r_multiple = self._r_multiple(pos, pnl)
        self.balance += pnl
        self.equity = self.balance
        self.peak_balance = max(self.peak_balance, self.balance)
        self.risk.register_trade(pnl, pos.session)
        self.risk.set_balance(self.balance)
        
        # Update Kelly stats
        self._update_kelly_stats(r_multiple)

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
            grade=pos.grade,
        )
        self.trade_history.append(trade)
        self._update_session_stats(trade)
        trade_logger.info(
            "CLOSE | %s | qty=%.2f exit=%.5f pnl=%.2f R=%.2f grade=%s",
            pos.side.value.upper(),
            pos.quantity,
            exit_price,
            pnl,
            r_multiple,
            pos.grade,
        )
        self.position = None
        return trade

    def _update_kelly_stats(self, r_multiple: float) -> None:
        """Update win/loss stats for Kelly calculation."""
        if r_multiple > 0:
            self._win_count += 1
            self._total_win_r += r_multiple
        else:
            self._loss_count += 1
            self._total_loss_r += r_multiple

    def _pnl_for_qty(self, pos: Position, exit_price: float, quantity: float) -> float:
        """Calculate PnL for a specific quantity."""
        direction = 1 if pos.side is OrderSide.BUY else -1
        ticks = (exit_price - pos.entry_price) * direction / self.instrument.tick_size
        return ticks * self.instrument.pip_value * (quantity / self.instrument.contract_size)

    def _pnl(self, pos: Position, exit_price: float) -> float:
        direction = 1 if pos.side is OrderSide.BUY else -1
        ticks = (exit_price - pos.entry_price) * direction / self.instrument.tick_size
        return ticks * self.instrument.pip_value * (pos.quantity / self.instrument.contract_size)

    def _r_multiple(self, pos: Position, pnl: float) -> float:
        risk_ticks = abs(pos.entry_price - pos.initial_stop) / self.instrument.tick_size
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