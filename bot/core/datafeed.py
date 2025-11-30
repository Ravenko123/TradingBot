"""Data feeds for CSV, in-memory, and websocket streams."""

from __future__ import annotations

import asyncio
import json
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Iterable, List, Optional

import websockets

from core.utils import Candle, load_candles


class DataFeedError(Exception):
    """Raised when the data feed encounters an unrecoverable issue."""


class BaseDataFeed(ABC):
    """Common contract for all candle feeds."""

    def __init__(self, *, speed: float = 0.0, loop: bool = False) -> None:
        self.speed = speed
        self.loop = loop

    def __aiter__(self) -> AsyncIterator[Candle]:
        return self.stream()

    @abstractmethod
    async def stream(self) -> AsyncIterator[Candle]:
        """Yield candles sequentially."""


class CSVDataFeed(BaseDataFeed):
    """Sequential feed backed by CSV data."""

    def __init__(
        self,
        path: str | Path,
        *,
        speed: float = 0.0,
        loop: bool = False,
        max_candles: Optional[int] = None,
    ) -> None:
        super().__init__(speed=speed, loop=loop)
        self._candles: List[Candle] = load_candles(str(path))
        if not self._candles:
            raise DataFeedError(f"No candles found at {path}")
        self._max = max_candles

    async def stream(self) -> AsyncIterator[Candle]:
        emitted = 0
        while True:
            for candle in self._candles:
                yield candle
                emitted += 1
                if self.speed:
                    await asyncio.sleep(self.speed)
                if self._max is not None and emitted >= self._max:
                    return
            if not self.loop:
                break


class MemoryDataFeed(BaseDataFeed):
    """Simple feed backed by in-memory candles for tests."""

    def __init__(self, candles: Iterable[Candle], *, speed: float = 0.0, loop: bool = False) -> None:
        super().__init__(speed=speed, loop=loop)
        self._candles = list(candles)
        if not self._candles:
            raise DataFeedError("MemoryDataFeed requires at least one candle")

    async def stream(self) -> AsyncIterator<Candle>:
        while True:
            for candle in self._candles:
                yield candle
                if self.speed:
                    await asyncio.sleep(self.speed)
            if not self.loop:
                break


class WebsocketFeed(BaseDataFeed):
    """Consume candles from a websocket stream (JSON payloads)."""

    def __init__(self, url: str, *, speed: float = 0.0) -> None:
        super().__init__(speed=speed, loop=True)
        self.url = url

    async def stream(self) -> AsyncIterator[Candle]:
        try:
            async with websockets.connect(self.url, ping_interval=None) as ws:  # type: ignore[arg-type]
                async for message in ws:
                    payload = json.loads(message)
                    candle = Candle(
                        timestamp=_parse_timestamp(payload["timestamp"]),
                        open=float(payload["open"]),
                        high=float(payload["high"]),
                        low=float(payload["low"]),
                        close=float(payload["close"]),
                        volume=float(payload.get("volume", 0.0)),
                    )
                    yield candle
                    if self.speed:
                        await asyncio.sleep(self.speed)
        except Exception as exc:  # pragma: no cover
            raise DataFeedError(f"Websocket feed error: {exc}") from exc


class SimulatedWebsocketFeed(BaseDataFeed):
    """Replay CSV candles with websocket-like pacing."""

    def __init__(self, path: str | Path, *, speed: float = 0.05) -> None:
        super().__init__(speed=speed, loop=True)
        self._candles = load_candles(str(path))
        if not self._candles:
            raise DataFeedError("Simulated feed requires candles")

    async def stream(self) -> AsyncIterator[Candle]:
        while True:
            for candle in self._candles:
                yield candle
                await asyncio.sleep(self.speed)


def _parse_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


__all__ = [
    "BaseDataFeed",
    "CSVDataFeed",
    "MemoryDataFeed",
    "WebsocketFeed",
    "SimulatedWebsocketFeed",
    "DataFeedError",
]