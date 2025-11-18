"""Technical analysis helpers and shared data structures."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time
from statistics import mean
from typing import Iterable, List, Optional, Sequence, TypedDict

import pandas as pd

from config.settings import SessionWindow


class Candle(TypedDict):
    """Canonical candle representation used by the bot."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class LiquiditySweep:
    """Represents a detected liquidity sweep event."""

    direction: str
    swept_level: float
    candle_time: datetime


def load_candles(path: str) -> List[Candle]:
    """Load candles from CSV into the canonical structure."""

    frame = pd.read_csv(path, parse_dates=["timestamp"])
    candles: List[Candle] = []
    for row in frame.itertuples(index=False):
        candles.append(
            Candle(
                timestamp=row.timestamp.to_pydatetime(),
                open=float(row.open),
                high=float(row.high),
                low=float(row.low),
                close=float(row.close),
                volume=float(row.volume),
            )
        )
    return candles


def calculate_atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Compute Average True Range for the supplied candles."""

    if len(candles) < period + 1:
        raise ValueError("Not enough candles to compute ATR")
    trs: List[float] = []
    for idx in range(1, period + 1):
        current = candles[-idx]
        previous = candles[-idx - 1]
        true_range = max(
            current["high"] - current["low"],
            abs(current["high"] - previous["close"]),
            abs(current["low"] - previous["close"]),
        )
        trs.append(true_range)
    return sum(trs) / len(trs)


def calculate_displacement(candles: Sequence[Candle], lookback: int = 5) -> float:
    """Measure displacement as candle body relative to recent average range."""

    if len(candles) < lookback:
        raise ValueError("Not enough candles for displacement")
    recent = candles[-lookback:]
    avg_range = mean(c["high"] - c["low"] for c in recent)
    last = candles[-1]
    body = abs(last["close"] - last["open"])
    return body / avg_range if avg_range else 0.0


def detect_liquidity_sweep(
    candles: Sequence[Candle],
    lookback: int = 20,
) -> Optional[LiquiditySweep]:
    """Detect sweeps of recent highs or lows indicative of stop hunts."""

    if len(candles) < lookback:
        return None
    window = candles[-lookback:]
    highs = [c["high"] for c in window[:-1]]
    lows = [c["low"] for c in window[:-1]]
    current = window[-1]

    prev_high = max(highs)
    prev_low = min(lows)

    if current["high"] > prev_high and current["close"] < current["open"]:
        return LiquiditySweep(direction="sell", swept_level=prev_high, candle_time=current["timestamp"])
    if current["low"] < prev_low and current["close"] > current["open"]:
        return LiquiditySweep(direction="buy", swept_level=prev_low, candle_time=current["timestamp"])
    return None


def active_session(timestamp: datetime, sessions: Iterable[SessionWindow]) -> Optional[str]:
    """Return the name of the active session for the timestamp, if any."""

    ts_time = timestamp.time()
    for window in sessions:
        if _is_within(ts_time, window.start, window.end):
            return window.name
    return None


def _is_within(value: time, start: time, end: time) -> bool:
    """Handle session windows that may wrap past midnight."""

    if start <= end:
        return start <= value <= end
    return value >= start or value <= end


def pip_distance(price_a: float, price_b: float, pip_size: float) -> float:
    """Return the absolute pip distance between two price points."""

    return abs(price_a - price_b) / pip_size


__all__ = [
    "Candle",
    "LiquiditySweep",
    "active_session",
    "calculate_atr",
    "calculate_displacement",
    "detect_liquidity_sweep",
    "load_candles",
    "pip_distance",
]