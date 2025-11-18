"""ICT Smart Money Concepts strategy implementation."""

from __future__ import annotations

from collections import deque
from datetime import datetime
from typing import Deque, Dict, Optional

from core.execution import StrategySignal
from core.broker import OrderSide
from core.utils import (
    Candle,
    active_session,
    calculate_atr,
    calculate_displacement,
    detect_liquidity_sweep,
)
from strategies import BaseStrategy, StrategyRegistry


class ICTSMCStrategy(BaseStrategy):
    """Detects liquidity grabs and displacement for 1-minute entries."""

    name = "ict_smc"

    def __init__(self, context) -> None:
        super().__init__(context)
        self._candles: Deque[Candle] = deque(maxlen=250)

    def on_candle(self, candle: Dict[str, float]) -> Optional[StrategySignal]:
        structured: Candle = Candle(
            timestamp=candle["timestamp"],
            open=candle["open"],
            high=candle["high"],
            low=candle["low"],
            close=candle["close"],
            volume=candle["volume"],
        )
        self._candles.append(structured)
        if len(self._candles) < 60:
            return None

        session_name = active_session(structured["timestamp"], self.context.session_windows)
        if session_name not in {"London Open", "New York Open"}:
            return None

        sweep = detect_liquidity_sweep(list(self._candles), lookback=40)
        if not sweep:
            return None

        displacement = calculate_displacement(list(self._candles), lookback=10)
        if displacement < 1.2:
            return None

        atr = calculate_atr(list(self._candles), period=14)
        buffer = atr * 0.25
        entry = structured["close"]

        if sweep.direction == "sell":
            stop_loss = max(structured["high"], sweep.swept_level) + buffer
            take_profit = entry - 2 * (stop_loss - entry)
            side = OrderSide.SELL
        else:
            stop_loss = min(structured["low"], sweep.swept_level) - buffer
            take_profit = entry + 2 * (entry - stop_loss)
            side = OrderSide.BUY

        if take_profit <= 0 or stop_loss <= 0:
            return None

        return StrategySignal(
            instrument=self.context.instrument,
            side=side,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=structured["timestamp"],
            confidence=displacement,
            session_name=session_name,
        )


StrategyRegistry.register(ICTSMCStrategy)

__all__ = ["ICTSMCStrategy"]