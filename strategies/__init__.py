"""Strategy factory and base classes for the ICT SMC bot."""

from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Dict, List, Optional, Type

from core.execution import StrategySignal
from config.settings import SessionWindow

class StrategyError(Exception):
    """Raised when strategy creation fails."""


@dataclass
class StrategyContext:
    """Runtime context available to all strategies."""

    instrument: str
    risk_per_trade: float
    session_windows: List[SessionWindow]


class BaseStrategy:
    """Base class all strategies must derive from."""

    name: str = "base"

    def __init__(self, context: StrategyContext) -> None:
        self.context = context

    def on_candle(self, candle: Dict[str, float]) -> Optional[StrategySignal]:
        """Process the latest candle and optionally return a strategy signal."""
        raise NotImplementedError


class StrategyRegistry:
    """Registry for mapping strategy names to implementations."""

    _registry: Dict[str, Type[BaseStrategy]] = {}

    @classmethod
    def register(cls, strategy_cls: Type[BaseStrategy]) -> None:
        cls._registry[strategy_cls.name] = strategy_cls

    @classmethod
    def create(cls, name: str, context: StrategyContext) -> BaseStrategy:
        if name not in cls._registry:
            try:
                importlib.import_module(f"strategies.{name}.strategy")
            except ModuleNotFoundError as exc:  # pragma: no cover
                raise StrategyError(f"Unknown strategy: {name}") from exc
        if name not in cls._registry:
            raise StrategyError(f"Unknown strategy: {name}")
        return cls._registry[name](context)
