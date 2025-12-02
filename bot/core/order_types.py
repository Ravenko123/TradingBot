"""Shared order and signal primitives used across the bot."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import List, Optional, Tuple


class OrderSide(str, Enum):
    """Supported trade directions."""

    BUY = "buy"
    SELL = "sell"


@dataclass(slots=True)
class TakeProfitLevel:
    """Single take-profit level with position percentage."""
    
    price: float  # Take profit price level
    percent: float  # Percentage of position to close (0.0 - 1.0)
    rr_ratio: float  # R:R ratio this level represents


@dataclass
class StrategySignal:
    """Canonical strategy instruction consumed by execution.
    
    Supports both single TP (legacy) and tiered partial TPs.
    """

    instrument: str
    side: OrderSide
    entry: float
    stop_loss: float
    take_profit: float  # Primary/final take profit level
    timestamp: datetime
    session: str
    confidence: float
    
    # Optional tiered take profit levels for partial exits
    # Format: List of (price, percent_to_close, rr_ratio)
    # If empty, single TP is used
    partial_tps: List[TakeProfitLevel] = field(default_factory=list)
    
    def get_all_tp_levels(self) -> List[TakeProfitLevel]:
        """Get all TP levels including the primary TP.
        
        Returns sorted list of TP levels from closest to furthest.
        """
        if not self.partial_tps:
            # Single TP mode - return just the main TP
            return [TakeProfitLevel(self.take_profit, 1.0, self._calculate_rr())]
        
        return sorted(self.partial_tps, key=lambda tp: abs(tp.price - self.entry))
    
    def _calculate_rr(self) -> float:
        """Calculate the R:R ratio for the main take profit."""
        risk = abs(self.entry - self.stop_loss)
        if risk <= 0:
            return 0.0
        reward = abs(self.take_profit - self.entry)
        return reward / risk


@dataclass(slots=True)
class IOCOrder:
    """Immediate-or-cancel order representation."""

    instrument: str
    side: OrderSide
    quantity: float
    timestamp: datetime
    limit_price: Optional[float] = None
    entry_price: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None


@dataclass(slots=True)
class Fill:
    """Result returned by the broker after attempting to execute an IOC order."""

    filled_quantity: float
    price: Optional[float]
    status: str
    reason: Optional[str]
    ticket: int = 0  # MT5 order ticket


@dataclass(slots=True)
class LiquiditySnapshot:
    """Per-candle liquidity information used for execution modelling."""

    available_volume: float
    spread: float
    slippage: float


__all__ = [
    "OrderSide",
    "StrategySignal",
    "TakeProfitLevel",
    "IOCOrder",
    "Fill",
    "LiquiditySnapshot",
]
