"""Centralized runtime configuration for the ICT SMC bot."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import time
from pathlib import Path
from typing import Dict, List


@dataclass(frozen=True)
class SessionWindow:
    """Represents a trading session window in UTC."""

    name: str
    start: time
    end: time


@dataclass(frozen=True)
class Instrument:
    """Instrument-level configuration useful for routing and sizing."""

    symbol: str
    pip_value: float
    min_tick: float
    contract_size: float


@dataclass
class Settings:
    """Top-level application settings."""

    initial_balance: float
    risk_per_trade: float
    max_daily_loss_pct: float
    allowed_instruments: Dict[str, Instrument]
    default_spread: float
    slippage: float
    sessions: List[SessionWindow]
    data_timezone: str
    liquidity_buffer: float
    equity_depletion_stop_pct: float
    logs_dir: Path


LOGS_DIR = Path("logs")
LOGS_DIR.mkdir(exist_ok=True)

SETTINGS = Settings(
    initial_balance=100_000.0,
    risk_per_trade=0.01,
    max_daily_loss_pct=0.03,
    allowed_instruments={
        "EURUSD": Instrument(symbol="EURUSD", pip_value=10.0, min_tick=0.0001, contract_size=100_000),
        "GBPUSD": Instrument(symbol="GBPUSD", pip_value=10.0, min_tick=0.0001, contract_size=100_000),
    },
    default_spread=0.00012,
    slippage=0.00005,
    sessions=[
        SessionWindow(name="London Open", start=time(hour=7), end=time(hour=10)),
        SessionWindow(name="New York Open", start=time(hour=12), end=time(hour=16)),
    ],
    data_timezone="UTC",
    liquidity_buffer=0.25,
    equity_depletion_stop_pct=0.4,
    logs_dir=LOGS_DIR,
)
