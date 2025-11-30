"""Risk management helpers enforcing firm-wide limits."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Dict, Optional, Tuple

from config.settings import Instrument, SETTINGS


@dataclass(slots=True)
class RiskState:
    """Mutable counters tracked over the trading day."""

    trading_day: date = date.min
    daily_loss: float = 0.0
    consecutive_losses: int = 0
    session_trades: Dict[str, int] = field(default_factory=dict)


class RiskViolation(Exception):
    """Raised when an order would breach configured risk limits."""


class RiskManager:
    """Applies money-management limits before sizing or routing trades."""

    def __init__(self, initial_balance: float, *, risk_per_trade: Optional[float] = None) -> None:
        self._balance = initial_balance
        self._state = RiskState()
        self._risk_per_trade = risk_per_trade if risk_per_trade is not None else SETTINGS.risk.risk_per_trade

    def set_balance(self, balance: float) -> None:
        self._balance = balance

    def evaluate(self, now: datetime, session: str) -> Tuple[bool, str]:
        """Return (allowed, reason) prior to accepting a new trade."""

        self._roll_state(now)
        risk_cfg = SETTINGS.risk
        equity_floor = SETTINGS.initial_balance * risk_cfg.equity_floor_pct
        if self._balance <= equity_floor:
            return False, "equity_floor"
        max_daily_loss = -SETTINGS.initial_balance * risk_cfg.max_daily_loss_pct
        if self._state.daily_loss <= max_daily_loss:
            return False, "max_daily_loss"
        if self._state.consecutive_losses >= risk_cfg.max_consecutive_losses:
            return False, "consecutive_losses"
        if self._state.session_trades.get(session, 0) >= risk_cfg.max_trades_per_session:
            return False, "session_limit"
        return True, "ok"

    def register_trade(self, pnl: float, session: str) -> None:
        """Update counters after a trade closes."""

        self._state.session_trades[session] = self._state.session_trades.get(session, 0) + 1
        if pnl < 0:
            self._state.daily_loss += pnl
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0
        self._balance += pnl

    def size_position(self, instrument: Instrument, entry: float, stop_loss: float) -> float:
        """Compute contract quantity based on configured risk per trade."""

        risk_capital = self._balance * self._risk_per_trade
        stop_distance = abs(entry - stop_loss)
        if stop_distance <= 0:
            return 0.0
        ticks = stop_distance / instrument.tick_size
        risk_per_contract = ticks * instrument.pip_value
        if risk_per_contract <= 0:
            return 0.0
        contracts = risk_capital / risk_per_contract
        return max(contracts * instrument.contract_size, 0.0)

    def _roll_state(self, now: datetime) -> None:
        if self._state.trading_day == now.date():
            return
        self._state = RiskState(trading_day=now.date())


__all__ = ["RiskManager", "RiskViolation"]
