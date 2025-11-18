"""Broker abstraction and IOC execution simulation."""

from __future__ import annotations

import enum
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

from config.settings import SETTINGS
from core.utils import Candle


class OrderSide(str, enum.Enum):
    """Direction for an order."""

    BUY = "buy"
    SELL = "sell"


@dataclass
class Order:
    """Represents a market IOC order."""

    instrument: str
    side: OrderSide
    quantity: float
    timestamp: datetime


@dataclass
class FillResult:
    """Outcome of submitting an IOC order."""

    filled: bool
    fill_price: Optional[float]
    reason: Optional[str] = None


class SimulatedBroker:
    """Simulate IOC execution using the current candle's liquidity."""

    def __init__(self, spread: float | None = None, slippage: float | None = None) -> None:
        self.spread = SETTINGS.default_spread if spread is None else spread
        self.slippage = SETTINGS.slippage if slippage is None else slippage

    async def submit_order(self, order: Order, candle: Candle) -> FillResult:
        """Attempt to fill the order with IOC semantics."""

        available_liquidity = candle["volume"] * SETTINGS.liquidity_buffer
        if order.quantity > available_liquidity:
            return FillResult(filled=False, fill_price=None, reason="insufficient_liquidity")

        mid_price = (candle["high"] + candle["low"]) / 2
        spread_half = self.spread / 2
        slip = self.slippage

        if order.side == OrderSide.BUY:
            execution_price = min(candle["high"], mid_price + spread_half + slip)
        else:
            execution_price = max(candle["low"], mid_price - spread_half - slip)

        return FillResult(filled=True, fill_price=execution_price)


__all__ = ["Order", "OrderSide", "FillResult", "SimulatedBroker"]