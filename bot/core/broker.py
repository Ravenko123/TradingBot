"""Simulated broker that enforces IOC semantics with spread/slippage."""

from __future__ import annotations

import random

from config.settings import Instrument, SETTINGS
from core.order_types import Fill, IOCOrder
from core.utils import Candle


class SimulatedBroker:
    """Simple market execution model used by both live and backtest flows."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        spread: float | None = None,
        slippage: float | None = None,
    ) -> None:
        self.instrument = instrument
        self.exec_cfg = SETTINGS.execution
        self._spread = spread if spread is not None else self.exec_cfg.default_spread
        self._slippage = slippage if slippage is not None else self.exec_cfg.default_slippage

    async def execute_ioc(self, order: IOCOrder, candle: Candle) -> Fill:
        """Attempt to fill an IOC order, simulating liquidity and slippage."""

        volume_units = candle.volume
        if volume_units <= 0:
            volume_units = self.instrument.liquidity_per_min
        elif volume_units < self.instrument.contract_size:
            volume_units *= self.instrument.contract_size

        available = min(self.instrument.liquidity_per_min, volume_units)
        available *= self.exec_cfg.min_liquidity_ratio
        min_floor = self.instrument.contract_size * self.exec_cfg.min_liquidity_ratio
        available = max(available, min_floor)
        if available <= 0:
            return Fill(filled_quantity=0.0, price=None, status="rejected", reason="no_liquidity")

        fill_qty = min(order.quantity, available)
        fill_ratio = fill_qty / order.quantity if order.quantity else 0.0
        if fill_ratio < self.exec_cfg.partial_fill_threshold:
            return Fill(filled_quantity=0.0, price=None, status="cancelled", reason="insufficient_liquidity")

        spread_half = self._spread / 2
        slip = self._slippage * random.uniform(0.5, 1.5)
        mid = candle.mid
        if order.side.value == "buy":
            price = min(candle.high, mid + spread_half + slip)
        else:
            price = max(candle.low, mid - spread_half - slip)

        if order.limit_price is not None:
            if order.side.value == "buy" and price > order.limit_price:
                return Fill(0.0, None, "cancelled", "limit_price_exceeded")
            if order.side.value == "sell" and price < order.limit_price:
                return Fill(0.0, None, "cancelled", "limit_price_exceeded")

        return Fill(filled_quantity=fill_qty, price=price, status="filled", reason=None)


__all__ = ["SimulatedBroker"]