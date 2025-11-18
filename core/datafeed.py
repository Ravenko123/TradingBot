"""Reusable candle data feed abstractions."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import AsyncIterator, Iterable, List, Optional

from core.utils import Candle, load_candles


class DataFeedError(Exception):
    """Raised when the data feed encounters an unrecoverable issue."""


@dataclass
class BaseDataFeed:
    """Base class for candle feeds."""

    speed: float = 0.0
    loop: bool = False
    max_candles: Optional[int] = None

    async def stream(self) -> AsyncIterator[Candle]:  # pragma: no cover - abstract
        raise NotImplementedError

    def __aiter__(self) -> AsyncIterator[Candle]:
        return self.stream()


class CSVDataFeed(BaseDataFeed):
    """Streams candles from a CSV file sequentially."""

    def __init__(
        self,
        path: str,
        *,
        speed: float = 0.0,
        loop: bool = False,
        max_candles: Optional[int] = None,
    ) -> None:
        super().__init__(speed=speed, loop=loop, max_candles=max_candles)
        candles = load_candles(path)
        if not candles:
            raise DataFeedError(f"No candles found in {path}")
        self._candles: List[Candle] = candles

    async def stream(self) -> AsyncIterator[Candle]:
        emitted = 0
        while True:
            for candle in self._candles:
                yield candle
                emitted += 1
                if self.speed:
                    await asyncio.sleep(self.speed)
                if self.max_candles and emitted >= self.max_candles:
                    return
            if not self.loop:
                break


class MemoryDataFeed(BaseDataFeed):
    """Simple feed backed by in-memory candles for testing."""

    def __init__(
        self,
        candles: Iterable[Candle],
        *,
        speed: float = 0.0,
        loop: bool = False,
        max_candles: Optional[int] = None,
    ) -> None:
        super().__init__(speed=speed, loop=loop, max_candles=max_candles)
        self._candles = list(candles)
        if not self._candles:
            raise DataFeedError("MemoryDataFeed requires at least one candle")

    async def stream(self) -> AsyncIterator[Candle]:
        emitted = 0
        while True:
            for candle in self._candles:
                yield candle
                emitted += 1
                if self.speed:
                    await asyncio.sleep(self.speed)
                if self.max_candles and emitted >= self.max_candles:
                    return
            if not self.loop:
                break


__all__ = ["BaseDataFeed", "CSVDataFeed", "MemoryDataFeed", "DataFeedError"]