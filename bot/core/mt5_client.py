"""Lightweight MetaTrader 5 candle downloader."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional

from core.utils import Candle

try:  # pragma: no cover - module import varies by environment
    import MetaTrader5 as mt5  # type: ignore
except Exception:  # pragma: no cover - handled dynamically
    mt5 = None  # type: ignore

_INITIALIZED = False


class MT5NotAvailable(RuntimeError):
    """Raised when MetaTrader 5 isn't available locally."""


# =============================================================================
# SYMBOL MAPPING: Internal name -> MT5 broker symbol
# BTCUSD has no suffix, all others have "+" suffix
# =============================================================================
SYMBOL_MAP: Dict[str, str] = {
    "BTCUSD": "BTCUSD",      # No suffix
    "USDJPY": "USDJPY+",
    "XAUUSD": "XAUUSD+",
    "GBPJPY": "GBPJPY+",
    "EURUSD": "EURUSD+",
    "GBPUSD": "GBPUSD+",
}


def get_mt5_symbol(internal_symbol: str) -> str:
    """Convert internal symbol name to MT5 broker symbol."""
    normalized = internal_symbol.upper().strip()
    if normalized in SYMBOL_MAP:
        return SYMBOL_MAP[normalized]
    # Default: add + suffix for unknown symbols (except BTC pairs)
    if "BTC" in normalized:
        return normalized
    return normalized + "+"


def ensure_initialized(path: Optional[str] = None) -> None:
    """Start an MT5 session if one is not already active."""

    global _INITIALIZED
    if _INITIALIZED:
        return
    if mt5 is None:
        raise MT5NotAvailable(
            "MetaTrader5 Python package is missing. Install it via 'pip install MetaTrader5'."
        )
    initialized = False
    if path:
        initialized = mt5.initialize(path=path)
        if not initialized:
            code, message = mt5.last_error()
            if code == -2:  # invalid path, fall back to default discovery
                initialized = mt5.initialize()
            else:
                raise RuntimeError(f"Failed to initialize MetaTrader5 ({code}): {message}")
    else:
        initialized = mt5.initialize()
    if not initialized:  # pragma: no cover - depends on terminal availability
        code, message = mt5.last_error()
        raise RuntimeError(f"Failed to initialize MetaTrader5 ({code}): {message}")
    _INITIALIZED = True


def shutdown() -> None:
    """Gracefully close the MT5 connection."""

    global _INITIALIZED
    if not _INITIALIZED or mt5 is None:
        return
    mt5.shutdown()
    _INITIALIZED = False


def fetch_candles(
    symbol: str,
    timeframe: str,
    days: int,
    *,
    terminal_path: Optional[str] = None,
    buffer_days: int = 3,
) -> List[Candle]:
    """Download at least ``days`` worth of candles for ``symbol``."""

    if days <= 0:
        raise ValueError("days must be positive")
    ensure_initialized(path=terminal_path)
    tf_code = _timeframe_constant(timeframe)
    utc = timezone.utc
    end = datetime.now(tz=utc)
    start = end - timedelta(days=days + buffer_days)
    
    # Convert internal symbol to MT5 broker symbol
    mt5_symbol = get_mt5_symbol(symbol)
    
    rates = mt5.copy_rates_range(mt5_symbol, tf_code, start, end)
    if rates is None or len(rates) == 0:
        code, message = mt5.last_error()
        raise RuntimeError(f"MetaTrader5 returned no data for {mt5_symbol} ({code}): {message}")
    volume_field = _volume_field_name(rates)
    candles = [
        Candle(
            timestamp=datetime.fromtimestamp(int(row["time"]), tz=utc).replace(tzinfo=None),
            open=float(row["open"]),
            high=float(row["high"]),
            low=float(row["low"]),
            close=float(row["close"]),
            volume=float(row[volume_field]) if volume_field else 0.0,
        )
        for row in rates
    ]
    return candles


def _timeframe_constant(name: str) -> int:
    if mt5 is None:
        raise MT5NotAvailable(
            "MetaTrader5 Python package is missing. Install it via 'pip install MetaTrader5'."
        )
    normalized = name.upper().strip()
    mapping: Dict[str, int] = {
        "M1": mt5.TIMEFRAME_M1,
        "M2": mt5.TIMEFRAME_M2,
        "M3": mt5.TIMEFRAME_M3,
        "M4": mt5.TIMEFRAME_M4,
        "M5": mt5.TIMEFRAME_M5,
        "M6": mt5.TIMEFRAME_M6,
        "M10": mt5.TIMEFRAME_M10,
        "M12": mt5.TIMEFRAME_M12,
        "M15": mt5.TIMEFRAME_M15,
        "M20": mt5.TIMEFRAME_M20,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H2": mt5.TIMEFRAME_H2,
        "H3": mt5.TIMEFRAME_H3,
        "H4": mt5.TIMEFRAME_H4,
        "H6": mt5.TIMEFRAME_H6,
        "H8": mt5.TIMEFRAME_H8,
        "H12": mt5.TIMEFRAME_H12,
        "D1": mt5.TIMEFRAME_D1,
        "W1": mt5.TIMEFRAME_W1,
        "MN1": mt5.TIMEFRAME_MN1,
    }
    if normalized not in mapping:
        raise ValueError(f"Unsupported timeframe '{name}'.")
    return mapping[normalized]


def _volume_field_name(rates) -> Optional[str]:
    names = getattr(rates, "dtype", None)
    if names and names.names:
        if "real_volume" in names.names:
            return "real_volume"
        if "tick_volume" in names.names:
            return "tick_volume"
    return None


__all__ = ["fetch_candles", "ensure_initialized", "shutdown", "MT5NotAvailable", "get_mt5_symbol", "SYMBOL_MAP"]
