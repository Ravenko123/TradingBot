"""Centralized configuration shared by the live bot and backtester."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import time
from pathlib import Path
from typing import Dict, Tuple


@dataclass(frozen=True)
class SessionWindow:
    """Represents a trading session window in UTC."""

    name: str
    start: time
    end: time


@dataclass(frozen=True)
class Instrument:
    """Instrument metadata required for sizing and routing."""

    symbol: str
    pip_value: float
    tick_size: float
    contract_size: float
    base_currency: str
    liquidity_per_min: float
    margin_rate: float


@dataclass(frozen=True)
class RiskSettings:
    """Firm-wide risk controls enforced before every trade."""

    risk_per_trade: float
    max_daily_loss_pct: float
    max_consecutive_losses: int
    max_trades_per_session: int
    equity_floor_pct: float


@dataclass(frozen=True)
class ExecutionSettings:
    """Execution model toggles shared by live/backtest engines."""

    default_spread: float
    default_slippage: float
    latency_ms: Tuple[int, int]
    partial_fill_threshold: float
    min_liquidity_ratio: float
    enforce_entry_limits: bool


@dataclass(frozen=True)
class BacktestSettings:
    """Backtest-wide switches for sequential simulations."""

    microstructure_slices: int
    export_dir: Path


@dataclass(frozen=True)
class DataSettings:
    """File-system locations and defaults for data feeds."""

    data_dir: Path
    default_csv: Path


@dataclass(frozen=True)
class Settings:
    """Bundle of all configuration groups."""

    initial_balance: float
    timezone: str
    sessions: Tuple[SessionWindow, ...]
    allowed_instruments: Dict[str, Instrument]
    risk: RiskSettings
    execution: ExecutionSettings
    backtest: BacktestSettings
    data: DataSettings
    logs_dir: Path = field(default=Path("logs"))


BASE_DIR = Path(__file__).resolve().parents[2]
LOGS_DIR = BASE_DIR / "logs"
DATA_DIR = BASE_DIR / "data"
EXPORT_DIR = BASE_DIR / "results"
for directory in (LOGS_DIR, DATA_DIR, EXPORT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

SESSIONS: Tuple[SessionWindow, ...] = (
    SessionWindow(name="London", start=time(hour=7), end=time(hour=10)),
    SessionWindow(name="NewYork", start=time(hour=12), end=time(hour=16)),
)

ALLOWED_INSTRUMENTS: Dict[str, Instrument] = {
    "XAUUSD+": Instrument(
        symbol="XAUUSD+",
        pip_value=1.0,
        tick_size=0.1,
        contract_size=100.0,
        base_currency="USD",
        liquidity_per_min=5_000_000,
        margin_rate=0.02,
    ),
    "EURUSD+": Instrument(
        symbol="EURUSD+",
        pip_value=10.0,
        tick_size=0.0001,
        contract_size=100_000.0,
        base_currency="USD",
        liquidity_per_min=25_000_000,
        margin_rate=0.02,
    ),
    "GBPUSD+": Instrument(
        symbol="GBPUSD+",
        pip_value=10.0,
        tick_size=0.0001,
        contract_size=100_000.0,
        base_currency="USD",
        liquidity_per_min=18_000_000,
        margin_rate=0.02,
    ),
    "GBPJPY+": Instrument(
        symbol="GBPJPY+",
        pip_value=9.0,
        tick_size=0.01,
        contract_size=100_000.0,
        base_currency="JPY",
        liquidity_per_min=12_000_000,
        margin_rate=0.03,
    ),
    "USDJPY+": Instrument(
        symbol="USDJPY+",
        pip_value=9.0,
        tick_size=0.01,
        contract_size=100_000.0,
        base_currency="JPY",
        liquidity_per_min=15_000_000,
        margin_rate=0.02,
    ),
    "BTCUSD": Instrument(
        symbol="BTCUSD",
        pip_value=5.0,
        tick_size=0.5,
        contract_size=1.0,
        base_currency="USD",
        liquidity_per_min=2_000_000,
        margin_rate=0.1,
    ),
}

SETTINGS = Settings(
    initial_balance=200_000.0,
    timezone="UTC",
    sessions=SESSIONS,
    allowed_instruments=ALLOWED_INSTRUMENTS,
    risk=RiskSettings(
        risk_per_trade=0.0075,
        max_daily_loss_pct=0.03,
        max_consecutive_losses=4,
        max_trades_per_session=3,
        equity_floor_pct=0.7,
    ),
    execution=ExecutionSettings(
        default_spread=0.00012,
        default_slippage=0.00008,
        latency_ms=(50, 180),
        partial_fill_threshold=0.2,
        min_liquidity_ratio=0.85,
        enforce_entry_limits=False,
    ),
    backtest=BacktestSettings(
        microstructure_slices=6,
        export_dir=EXPORT_DIR,
    ),
    data=DataSettings(
        data_dir=DATA_DIR,
        default_csv=DATA_DIR / "sample_data.csv",
    ),
    logs_dir=LOGS_DIR,
)


__all__ = [
    "Instrument",
    "SessionWindow",
    "RiskSettings",
    "ExecutionSettings",
    "BacktestSettings",
    "DataSettings",
    "Settings",
    "SETTINGS",
]
