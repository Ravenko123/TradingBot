"""Strategy settings loader.
Loads persisted best settings saved by backtest_improved.py for use in live/forward code.
"""
import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_SETTINGS = {}
SETTINGS_FILE = Path(__file__).parent / 'best_settings.json'


def _normalize_instrument_settings(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Map backtest/export keys to live engine canonical keys."""
    if not raw:
        return {}

    # Slow EMA in historical files is often stored as "EMA"
    ema_slow = int(raw.get('EMA_Slow', raw.get('EMA', 21)))
    ema_fast = int(raw.get('EMA_Fast', max(5, ema_slow // 3)))
    adx = float(raw.get('ADX', 25.0))
    atr_mult = float(raw.get('ATR_Mult', raw.get('SL_Mult', 1.5)))
    rr = float(raw.get('RR', 2.0))

    # Keep values in sane trading ranges
    ema_fast = max(3, min(ema_fast, 80))
    ema_slow = max(ema_fast + 1, min(ema_slow, 240))
    adx = max(10.0, min(adx, 60.0))
    atr_mult = max(0.3, min(atr_mult, 6.0))
    rr = max(0.5, min(rr, 6.0))

    out = dict(raw)
    out.update({
        'EMA_Fast': ema_fast,
        'EMA_Slow': ema_slow,
        'ADX': adx,
        'ATR_Mult': atr_mult,
        'RR': rr,
    })
    return out


def load_best_settings(path: Path = SETTINGS_FILE) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data.get('instruments', {})
    except Exception:
        return DEFAULT_SETTINGS


def get_instrument_settings(symbol: str, path: Path = SETTINGS_FILE) -> Dict[str, Any]:
    settings = load_best_settings(path)
    raw = settings.get(symbol, {})
    return _normalize_instrument_settings(raw)
