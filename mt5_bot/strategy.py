"""Strategy settings loader.
Loads persisted best settings saved by backtest_improved.py for use in live/forward code.
"""
import json
from pathlib import Path
from typing import Dict, Any

DEFAULT_SETTINGS = {}
SETTINGS_FILE = Path(__file__).parent / 'best_settings.json'


def load_best_settings(path: Path = SETTINGS_FILE) -> Dict[str, Any]:
    try:
        data = json.loads(path.read_text())
        return data.get('instruments', {})
    except Exception:
        return DEFAULT_SETTINGS


def get_instrument_settings(symbol: str, path: Path = SETTINGS_FILE) -> Dict[str, Any]:
    settings = load_best_settings(path)
    return settings.get(symbol, {})
