"""Utility helpers shared across trading and backtesting flows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from statistics import mean, pstdev
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from config.settings import SessionWindow


@dataclass(slots=True)
class Candle:
    """Normalized OHLCV candle structure."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass(slots=True)
class SwingPoint:
    """Represents a swing high or low used for structure analysis."""

    index: int
    price: float
    timestamp: datetime
    kind: str  # "high" or "low"


@dataclass(slots=True)
class FairValueGap:
    """Simplified 3-candle imbalance."""

    start_index: int
    end_index: int
    upper: float
    lower: float
    direction: str  # "bullish" or "bearish"

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass(slots=True)
class OrderBlock:
    """Lightweight order-block representation derived from swings."""

    index: int
    high: float
    low: float
    direction: str
    origin_time: datetime


@dataclass(slots=True)
class LiquiditySweep:
    """Represents a simple buy/sell-side liquidity sweep."""

    direction: str  # "buy" or "sell"
    swept_level: float
    timestamp: datetime


def load_candles(path: str) -> List[Candle]:
    """Load OHLCV data from CSV."""

    candles: List[Candle] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candles.append(
                Candle(
                    timestamp=datetime.fromisoformat(str(row["timestamp"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


def active_session(ts: datetime, sessions: Iterable[SessionWindow]) -> Optional[str]:
    """Return the configured session name covering the supplied timestamp."""

    current = ts.time()
    for session in sessions:
        if _within_window(current, session.start, session.end):
            return session.name
    return None


def _within_window(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def floor_to_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def rolling_atr(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        raise ValueError("Not enough candles for ATR")
    tr_values: List[float] = []
    for idx in range(1, period + 1):
        cur = candles[-idx]
        prev = candles[-idx - 1]
        tr_values.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    return sum(tr_values) / len(tr_values)


def displacement_score(candles: Sequence[Candle], lookback: int = 5) -> float:
    if len(candles) < lookback:
        return 0.0
    window = candles[-lookback:]
    avg_range = mean(c.range for c in window)
    last = candles[-1]
    body = abs(last.close - last.open)
    return body / avg_range if avg_range else 0.0


def swing_points(candles: Sequence[Candle], depth: int = 3) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    for idx in range(depth, len(candles) - depth):
        window = candles[idx - depth : idx + depth + 1]
        pivot = candles[idx]
        highs = [c.high for c in window if c is not pivot]
        lows = [c.low for c in window if c is not pivot]
        if pivot.high > max(highs):
            swings.append(SwingPoint(index=idx, price=pivot.high, timestamp=pivot.timestamp, kind="high"))
        elif pivot.low < min(lows):
            swings.append(SwingPoint(index=idx, price=pivot.low, timestamp=pivot.timestamp, kind="low"))
    return swings


def recent_structure(swings: Sequence[SwingPoint], max_age_minutes: int = 180) -> List[SwingPoint]:
    if not swings:
        return []
    cutoff = swings[-1].timestamp - timedelta(minutes=max_age_minutes)
    return [s for s in swings if s.timestamp >= cutoff]


def detect_bos(swings: Sequence[SwingPoint]) -> Optional[SwingPoint]:
    if len(swings) < 4:
        return None
    a, b, c, d = swings[-4:]
    if a.kind == "low" and c.kind == "low" and b.kind == "high" and d.kind == "high":
        if c.price > a.price and d.price > b.price:
            return d
    if a.kind == "high" and c.kind == "high" and b.kind == "low" and d.kind == "low":
        if c.price < a.price and d.price < b.price:
            return d
    return None


def detect_mss(swings: Sequence[SwingPoint]) -> Optional[SwingPoint]:
    if len(swings) < 5:
        return None
    last_swings = swings[-5:]
    highs = [s for s in last_swings if s.kind == "high"]
    lows = [s for s in last_swings if s.kind == "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            return highs[-1]
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            return lows[-1]
    return None


def detect_fair_value_gaps(candles: Sequence[Candle]) -> List[FairValueGap]:
    gaps: List[FairValueGap] = []
    for idx in range(2, len(candles)):
        c0, c1, c2 = candles[idx - 2], candles[idx - 1], candles[idx]
        if c0.high < c2.low:
            gaps.append(
                FairValueGap(
                    start_index=idx - 2,
                    end_index=idx,
                    upper=c1.low,
                    lower=c0.high,
                    direction="bullish",
                )
            )
        elif c0.low > c2.high:
            gaps.append(
                FairValueGap(
                    start_index=idx - 2,
                    end_index=idx,
                    upper=c0.low,
                    lower=c1.high,
                    direction="bearish",
                )
            )
    return gaps


def detect_order_blocks(swings: Sequence[SwingPoint], candles: Sequence[Candle]) -> List[OrderBlock]:
    blocks: List[OrderBlock] = []
    for swing in swings[-8:]:
        candle = candles[swing.index]
        if swing.kind == "high":
            blocks.append(
                OrderBlock(index=swing.index, high=candle.high, low=candle.open, direction="bearish", origin_time=candle.timestamp)
            )
        else:
            blocks.append(
                OrderBlock(index=swing.index, high=candle.open, low=candle.low, direction="bullish", origin_time=candle.timestamp)
            )
    return blocks


def equity_curve_stats(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] != 0]
    if not returns:
        return 0.0, 0.0
    return mean(returns), pstdev(returns)


def microstructure_path(candle: Candle, slices: int) -> Iterator[float]:
    if slices <= 0:
        raise ValueError("slices must be positive")
    half = max(slices // 2, 1)
    up_step = (candle.high - candle.open) / half if half else 0.0
    down_step = (candle.open - candle.low) / max(slices - half, 1)
    price = candle.open
    for _ in range(half):
        price += up_step
        yield min(price, candle.high)
    for _ in range(slices - half):
        price -= down_step
        yield max(price, candle.low)


def pip_distance(price_a: float, price_b: float, tick_size: float) -> float:
    return abs(price_a - price_b) / tick_size


def calculate_atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Backward-compatible ATR helper wrapping rolling_atr."""

    return rolling_atr(candles, period)


def calculate_displacement(candles: Sequence[Candle], lookback: int = 5) -> float:
    """Legacy wrapper for displacement score."""

    return displacement_score(candles, lookback)


def detect_liquidity_sweep(
    candles: Sequence[Candle],
    lookback: int = 40,
    reentry_buffer: float = 0.0,
    reentry_bars: int = 3,
) -> Optional[LiquiditySweep]:
    """Detect a liquidity grab that breaks a prior extreme and returns inside within a few candles."""

    total = len(candles)
    if total < 3:
        return None
    window = list(candles)[-min(max(lookback, reentry_bars + 2), total):]
    buffer = max(reentry_buffer, 0.0)
    recent = min(reentry_bars, len(window) - 1)
    if recent <= 0:
        return None
    start_idx = len(window) - recent
    for idx in range(start_idx, len(window)):
        candidate = window[idx]
        prior_slice = window[:idx]
        if not prior_slice:
            continue
        high = max(c.high for c in prior_slice)
        low = min(c.low for c in prior_slice)
        body_low = min(candidate.open, candidate.close)
        body_high = max(candidate.open, candidate.close)
        if candidate.high > high and body_low <= high + buffer:
            return LiquiditySweep(direction="sell", swept_level=high, timestamp=candidate.timestamp)
        if candidate.low < low and body_high >= low - buffer:
            return LiquiditySweep(direction="buy", swept_level=low, timestamp=candidate.timestamp)
    return None


__all__ = [
    "Candle",
    "SwingPoint",
    "FairValueGap",
    "OrderBlock",
    "LiquiditySweep",
    "load_candles",
    "active_session",
    "rolling_atr",
    "displacement_score",
    "calculate_atr",
    "calculate_displacement",
    "detect_liquidity_sweep",
    "swing_points",
    "recent_structure",
    "detect_bos",
    "detect_mss",
    "detect_fair_value_gaps",
    "detect_order_blocks",
    "equity_curve_stats",
    "microstructure_path",
    "floor_to_minute",
    "pip_distance",
]