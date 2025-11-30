"""ICT Smart Money Concepts strategy implementation."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Deque, Iterable, Optional

from core.order_types import OrderSide, StrategySignal
from core.utils import (
    Candle,
    FairValueGap,
    LiquiditySweep,
    OrderBlock,
    SwingPoint,
    active_session,
    calculate_atr,
    calculate_displacement,
    detect_bos,
    detect_fair_value_gaps,
    detect_liquidity_sweep,
    detect_order_blocks,
    detect_mss,
    recent_structure,
    swing_points,
)
from strategies import BaseStrategy, StrategyRegistry


@dataclass
class MarketContext:
    """Lightweight container describing the current market state."""

    direction: OrderSide
    sweep_level: float
    displacement: float
    atr: float
    session: str
    gap: Optional[FairValueGap]
    block: Optional[OrderBlock]


class ICTSMCStrategy(BaseStrategy):
    """Detects BOS/MSS + liquidity sweep confluence for 1-minute entries."""

    name = "ict_smc"
    DEFAULT_PARAMS = {
        "min_candles": 20,
        "sessions": ("London", "NewYork"),
        "rr_target": 2.0,
        "atr_period": 14,
        "displacement_lookback": 4,
        "min_displacement": 0.05,
        "sweep_lookback": 15,
        "gap_tolerance_mult": 2.0,
        "atr_buffer_mult": 0.15,
        "min_volume": 0.0,
        "bias_mode": "auto",
        "require_bias_alignment": False,
        "require_gap": False,
        "require_block": False,
        "sweep_reentry_buffer_mult": 0.35,
        "sweep_reentry_bars": 4,
        "allow_synthetic_sweeps": True,
        "demo_mode": False,
        "demo_stride": 5,
    }

    def __init__(self, context) -> None:
        super().__init__(context)
        self._candles: Deque[Candle] = deque(maxlen=400)
        self.params = {**self.DEFAULT_PARAMS, **(self.context.parameters or {})}

    def on_candle(self, candle: Candle) -> Optional[StrategySignal]:
        self._candles.append(candle)
        if len(self._candles) < int(self.params["min_candles"]):
            self._debug(f"warming up: {len(self._candles)}/{self.params['min_candles']}")
            return self._demo_signal(candle)

        min_vol = float(self.params.get("min_volume", 0.0))
        if min_vol and candle.volume < min_vol:
            return self._demo_signal(candle)

        session_name = active_session(candle.timestamp, self.context.session_windows)
        session_whitelist = set(self.params.get("sessions", self.DEFAULT_PARAMS["sessions"]))
        if session_name not in session_whitelist:
            self._debug(f"session filtered: {session_name}")
            return self._demo_signal(candle)

        market_ctx = self._build_market_context(session_name)
        if market_ctx is None:
            self._debug("market context rejected")
            return self._demo_signal(candle)

        if bool(self.params.get("require_gap")) and market_ctx.gap is None:
            self._debug("gap required but missing")
            return self._demo_signal(candle)
        if bool(self.params.get("require_block")) and market_ctx.block is None:
            self._debug("order block required but missing")
            return self._demo_signal(candle)

        entry = self._determine_entry(candle, market_ctx)
        if entry is None:
            self._debug("entry calculation failed")
            return self._demo_signal(candle)

        stop_loss = self._stop_loss(candle, market_ctx)
        if stop_loss is None:
            self._debug("stop calculation failed")
            return self._demo_signal(candle)

        risk = abs(entry - stop_loss)
        if risk <= 0:
            return self._demo_signal(candle)

        rr = float(self.params["rr_target"])
        if market_ctx.direction is OrderSide.BUY:
            take_profit = entry + rr * risk
        else:
            take_profit = entry - rr * risk

        if (market_ctx.direction is OrderSide.BUY and take_profit <= entry) or (
            market_ctx.direction is OrderSide.SELL and take_profit >= entry
        ):
            self._debug("invalid take profit")
            return self._demo_signal(candle)

        signal = StrategySignal(
            instrument=self.context.instrument,
            side=market_ctx.direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=self._candles[-1].timestamp,
            confidence=market_ctx.displacement,
            session=market_ctx.session,
        )
        self._debug(
            "signal generated | side=%s entry=%.5f stop=%.5f tp=%.5f",
            market_ctx.direction.value,
            entry,
            stop_loss,
            take_profit,
        )
        return signal

    def _build_market_context(self, session_name: str) -> Optional[MarketContext]:
        candles = list(self._candles)
        atr = calculate_atr(candles, period=int(self.params["atr_period"]))
        buffer_mult = float(self.params.get("sweep_reentry_buffer_mult", 0.2))
        sweep_buffer = atr * max(buffer_mult, 0.0)
        reentry_bars = max(int(self.params.get("sweep_reentry_bars", 3)), 1)
        sweep = detect_liquidity_sweep(
            candles,
            lookback=int(self.params["sweep_lookback"]),
            reentry_buffer=sweep_buffer,
            reentry_bars=reentry_bars,
        )
        swings = recent_structure(swing_points(candles, depth=3), max_age_minutes=240)
        if not sweep and bool(self.params.get("allow_synthetic_sweeps", True)):
            sweep = self._synthetic_sweep(candles)
        if not sweep:
            self._debug("no sweep or fallback context")
            return None
        bias = self._directional_bias(swings)
        if bias is None:
            bias = sweep.direction
        if bool(self.params.get("require_bias_alignment", False)) and bias != sweep.direction:
            self._debug("bias misaligned")
            return None

        direction = OrderSide.BUY if sweep.direction == "buy" else OrderSide.SELL
        displacement = calculate_displacement(candles, lookback=int(self.params["displacement_lookback"]))
        if displacement < float(self.params["min_displacement"]):
            self._debug(f"displacement too low: {displacement:.2f}")
            return None

        gaps = detect_fair_value_gaps(candles)
        blocks = detect_order_blocks(swings, candles)
        gap_tolerance = atr * float(self.params["gap_tolerance_mult"])
        gap = self._select_gap(gaps, direction, candles[-1].close, gap_tolerance)
        block = self._select_block(blocks, direction)

        return MarketContext(
            direction=direction,
            sweep_level=sweep.swept_level,
            displacement=displacement,
            atr=atr,
            session=session_name,
            gap=gap,
            block=block,
        )

    def _directional_bias(self, swings: Iterable[SwingPoint]) -> Optional[str]:
        swings_list = list(swings)
        if len(swings_list) < 4:
            return None
        mode = str(self.params.get("bias_mode", "auto")).lower()
        if mode == "none":
            return None
        if mode in {"auto", "bos_only"}:
            bos = self._bos_direction(swings_list)
            if bos:
                return bos
            if mode == "bos_only":
                return None
        if mode in {"auto", "mss_only"}:
            return self._mss_direction(swings_list)
        return None

    @staticmethod
    def _synthetic_sweep(candles: Iterable[Candle]) -> Optional[LiquiditySweep]:
        candles_list = list(candles)
        if len(candles_list) < 3:
            return None
        last = candles_list[-1]
        recent = candles_list[-5:]
        direction = "buy" if last.close >= last.open else "sell"
        if direction == "buy":
            level = min(c.low for c in recent)
        else:
            level = max(c.high for c in recent)
        return LiquiditySweep(direction=direction, swept_level=level, timestamp=last.timestamp)

    def _debug(self, message: str, *args) -> None:
        if bool(self.params.get("debug", False)):
            text = message % args if args else message
            print(f"[ICT-SMC] {text}")

    @staticmethod
    def _bos_direction(swings: Iterable[SwingPoint]) -> Optional[str]:
        bos = detect_bos(list(swings))
        if bos is None:
            return None
        if bos.kind == "high":
            return "buy"
        if bos.kind == "low":
            return "sell"
        return None

    @staticmethod
    def _mss_direction(swings: Iterable[SwingPoint]) -> Optional[str]:
        mss = detect_mss(list(swings))
        if mss is None:
            return None
        if mss.kind == "high":
            return "sell"
        if mss.kind == "low":
            return "buy"
        return None

    @staticmethod
    def _select_gap(
        gaps: Iterable[FairValueGap],
        direction: OrderSide,
        price: float,
        tolerance: float,
    ) -> Optional[FairValueGap]:
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        for gap in reversed(list(gaps)):
            if gap.direction != desired:
                continue
            if gap.contains(price):
                return gap
            if tolerance and (
                abs(price - gap.upper) <= tolerance
                or abs(price - gap.lower) <= tolerance
            ):
                return gap
        return None

    @staticmethod
    def _select_block(blocks: Iterable[OrderBlock], direction: OrderSide) -> Optional[OrderBlock]:
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        for block in reversed(list(blocks)):
            if block.direction == desired:
                return block
        return None

    def _determine_entry(self, candle: Candle, context: MarketContext) -> Optional[float]:
        entry = candle.close
        if context.gap:
            midpoint = (context.gap.upper + context.gap.lower) / 2
            if context.direction is OrderSide.BUY:
                entry = max(entry, midpoint)
            else:
                entry = min(entry, midpoint)
        elif context.block:
            midpoint = (context.block.high + context.block.low) / 2
            if context.direction is OrderSide.BUY:
                entry = max(entry, midpoint)
            else:
                entry = min(entry, midpoint)
        return entry if entry > 0 else None

    def _stop_loss(self, candle: Candle, context: MarketContext) -> Optional[float]:
        buffer = context.atr * float(self.params.get("atr_buffer_mult", 0.3))
        if context.direction is OrderSide.BUY:
            reference = min(candle.low, context.sweep_level)
            if context.block:
                reference = min(reference, context.block.low)
            stop = reference - buffer
        else:
            reference = max(candle.high, context.sweep_level)
            if context.block:
                reference = max(reference, context.block.high)
            stop = reference + buffer
        return stop if stop > 0 else None

    def _demo_signal(self, candle: Candle) -> Optional[StrategySignal]:
        if not bool(self.params.get("demo_mode")):
            return None
        stride = max(int(self.params.get("demo_stride", 5)), 1)
        if len(self._candles) < stride or len(self._candles) % stride:
            return None
        direction = OrderSide.BUY if candle.close >= candle.open else OrderSide.SELL
        entry = candle.close
        buffer = max(candle.range, 1e-5)
        rr = float(self.params.get("rr_target", 2.0))
        if direction is OrderSide.BUY:
            stop_loss = entry - buffer
            take_profit = entry + rr * buffer
        else:
            stop_loss = entry + buffer
            take_profit = entry - rr * buffer
        if stop_loss <= 0 or take_profit <= 0:
            return None
        session_name = active_session(candle.timestamp, self.context.session_windows) or "Demo"
        return StrategySignal(
            instrument=self.context.instrument,
            side=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=candle.timestamp,
            confidence=1.0,
            session=session_name,
        )


StrategyRegistry.register(ICTSMCStrategy)

__all__ = ["ICTSMCStrategy"]