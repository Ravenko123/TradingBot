"""Shared order and signal primitives used across the bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional


class OrderSide(str, Enum):
    """Supported trade directions."""

    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class StrategySignal:
    """Canonical strategy instruction consumed by execution."""

    instrument: str
    side: OrderSide
    entry: float
    stop_loss: float
    take_profit: float
    timestamp: datetime
    session: str
    confidence: float


@dataclass(slots=True)
class IOCOrder:
    """Immediate-or-cancel order representation."""

    instrument: str
    side: OrderSide
    quantity: float
    timestamp: datetime
    limit_price: Optional[float] = None


@dataclass(slots=True)
class Fill:
    """Result returned by the broker after attempting to execute an IOC order."""

    filled_quantity: float
    price: Optional[float]
    status: str
    reason: Optional[str]


@dataclass(slots=True)
class LiquiditySnapshot:
    """Per-candle liquidity information used for execution modelling."""

    available_volume: float
    spread: float
    slippage: float


__all__ = [
    "OrderSide",
    "StrategySignal",
    "IOCOrder",
    "Fill",
    "LiquiditySnapshot",
]
