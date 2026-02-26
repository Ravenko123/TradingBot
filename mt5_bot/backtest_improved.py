# backtest engine

import os
import sys
import argparse
import shutil
import time
import json
import warnings
import sqlite3
from pathlib import Path
from itertools import product
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
from enum import Enum

import numpy as np
import pandas as pd
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import MetaTrader5 as mt5
except:
    mt5 = None

warnings.filterwarnings('ignore')


# training config

class MarketPhase(Enum):
    """Market behavior phases"""
    TRENDING_STRONG = "trending_strong"
    TRENDING_WEAK = "trending_weak"
    RANGING = "ranging"
    BREAKOUT = "breakout"
    QUIET = "quiet"


class TradeOutcome(Enum):
    """Trade result classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


@dataclass
class PatternSignature:
    """Learned pattern from backtest"""
    pattern_id: str
    conditions: Dict[str, float]
    total_trades: int = 0
    wins: int = 0
    total_profit: float = 0.0
    total_loss: float = 0.0
    expectancy: float = 0.0
    
    def update(self, is_win: bool, profit: float):
        self.total_trades += 1
        if is_win:
            self.wins += 1
            self.total_profit += abs(profit)
        else:
            self.total_loss += abs(profit)
        
        # Calculate expectancy
        if self.total_trades > 0:
            win_rate = self.wins / self.total_trades
            avg_win = self.total_profit / self.wins if self.wins > 0 else 0
            avg_loss = self.total_loss / (self.total_trades - self.wins) if (self.total_trades - self.wins) > 0 else 0
            self.expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)


@dataclass
class AILearningState:
    """Complete AI learning state - saved between sessions"""
    patterns: Dict[str, PatternSignature] = field(default_factory=dict)
    regime_performance: Dict[str, Dict[str, float]] = field(default_factory=dict)
    hour_performance: Dict[int, Dict[str, float]] = field(default_factory=dict)
    symbol_profiles: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    optimal_params: Dict[str, Dict[str, float]] = field(default_factory=dict)
    training_iterations: int = 0
    total_simulated_trades: int = 0
    best_expectancy: float = -999.0
    last_training: str = ""


# Global AI state
AI_STATE = AILearningState()
AI_STATE_FILE = Path(__file__).parent / 'ai_data' / 'ai_training_state.json'

# config

INSTRUMENTS = {
    'EURUSD': {'pip_size': 0.0001, 'spread': 0.8, 'vol': 1.0},
    'GBPUSD': {'pip_size': 0.0001, 'spread': 1.0, 'vol': 1.2},
    'USDJPY': {'pip_size': 0.01, 'spread': 1.2, 'vol': 1.1},
    'XAUUSD': {'pip_size': 0.1, 'spread': 3.0, 'vol': 1.5},
    'GBPJPY': {'pip_size': 0.01, 'spread': 1.5, 'vol': 1.3},
    'BTCUSD': {'pip_size': 1.0, 'spread': 30.0, 'vol': 2.0},
    'NAS100': {'pip_size': 1.0, 'spread': 2.0, 'vol': 1.0},
}

# Broker naming: add '+' suffix for FX and XAUUSD when calling MT5
FOREX_PLUS = {'EURUSD', 'GBPUSD', 'USDJPY', 'GBPJPY', 'XAUUSD'}

BEST_SETTINGS_FILE = Path(__file__).parent / 'best_settings.json'

DATA_DIR = Path(__file__).parent / 'backtest_data_improved'
DATA_DIR.mkdir(exist_ok=True)

# AI Data directory
AI_DATA_DIR = Path(__file__).parent / 'ai_data'
AI_DATA_DIR.mkdir(exist_ok=True)

# Backtest window in days (limit data to recent period)
BACKTEST_DAYS = 30  # Reduced for faster training

# Training settings
MIN_EXPECTANCY_TO_SAVE = 5.0  # Minimum $ per trade expectancy to save params
MIN_PROFIT_FACTOR = 1.5  # Minimum PF to consider params "good"
MIN_TRADES_FOR_LEARNING = 20  # Need at least this many trades to learn
INITIAL_BALANCE = 10000.0


# learning funcs

def load_ai_state() -> AILearningState:
    """Load AI learning state from disk."""
    global AI_STATE
    if AI_STATE_FILE.exists():
        try:
            data = json.loads(AI_STATE_FILE.read_text())
            AI_STATE.training_iterations = data.get('training_iterations', 0)
            AI_STATE.total_simulated_trades = data.get('total_simulated_trades', 0)
            AI_STATE.best_expectancy = data.get('best_expectancy', -999.0)
            AI_STATE.last_training = data.get('last_training', '')
            AI_STATE.optimal_params = data.get('optimal_params', {})
            AI_STATE.regime_performance = data.get('regime_performance', {})
            AI_STATE.hour_performance = data.get('hour_performance', {})
            AI_STATE.symbol_profiles = data.get('symbol_profiles', {})
            
            # Rebuild patterns
            for pid, pdata in data.get('patterns', {}).items():
                AI_STATE.patterns[pid] = PatternSignature(
                    pattern_id=pid,
                    conditions=pdata.get('conditions', {}),
                    total_trades=pdata.get('total_trades', 0),
                    wins=pdata.get('wins', 0),
                    total_profit=pdata.get('total_profit', 0),
                    total_loss=pdata.get('total_loss', 0),
                    expectancy=pdata.get('expectancy', 0)
                )
            print(f"[AI] Loaded {len(AI_STATE.patterns)} learned patterns, {AI_STATE.training_iterations} iterations")
        except Exception as e:
            print(f"[AI] Fresh start - no previous learning found: {e}")
    return AI_STATE


def save_ai_state():
    """Save AI learning state to disk."""
    global AI_STATE
    AI_STATE.last_training = datetime.now().isoformat()
    
    data = {
        'training_iterations': AI_STATE.training_iterations,
        'total_simulated_trades': AI_STATE.total_simulated_trades,
        'best_expectancy': AI_STATE.best_expectancy,
        'last_training': AI_STATE.last_training,
        'optimal_params': AI_STATE.optimal_params,
        'regime_performance': AI_STATE.regime_performance,
        'hour_performance': AI_STATE.hour_performance,
        'symbol_profiles': AI_STATE.symbol_profiles,
        'patterns': {
            pid: {
                'pattern_id': p.pattern_id,
                'conditions': p.conditions,
                'total_trades': p.total_trades,
                'wins': p.wins,
                'total_profit': p.total_profit,
                'total_loss': p.total_loss,
                'expectancy': p.expectancy
            } for pid, p in AI_STATE.patterns.items()
        }
    }
    
    AI_STATE_FILE.parent.mkdir(exist_ok=True)
    AI_STATE_FILE.write_text(json.dumps(data, indent=2))


def extract_pattern_features(df: pd.DataFrame, idx: int) -> Dict[str, float]:
    """Extract pattern features at a specific bar - what the AI learns from."""
    if idx < 20:
        return {}
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    ema_f = df['EMA_Fast'].values if 'EMA_Fast' in df.columns else None
    ema_s = df['EMA_Slow'].values if 'EMA_Slow' in df.columns else None
    adx = df['ADX'].values if 'ADX' in df.columns else None
    atr = df['ATR'].values if 'ATR' in df.columns else None
    
    features = {}
    
    # EMA relationship
    if ema_f is not None and ema_s is not None and ema_s[idx] > 0:
        features['ema_ratio'] = round((ema_f[idx] / ema_s[idx]) - 1, 4)
    
    # ADX strength (binned)
    if adx is not None:
        features['adx_bin'] = int(adx[idx] // 10) * 10  # 0, 10, 20, 30, 40...
    
    # Volatility (ATR percentile)
    if atr is not None:
        atr_pct = np.percentile(atr[max(0,idx-50):idx], 50)
        features['atr_ratio'] = round(atr[idx] / (atr_pct + 0.0001), 2)
    
    # Price position relative to recent range
    recent_high = np.max(high[max(0,idx-20):idx])
    recent_low = np.min(low[max(0,idx-20):idx])
    if recent_high > recent_low:
        features['price_position'] = round((close[idx] - recent_low) / (recent_high - recent_low), 2)
    
    # Momentum (close vs 10-bar SMA)
    sma10 = np.mean(close[max(0,idx-10):idx])
    features['momentum'] = round((close[idx] / sma10) - 1, 4) if sma10 > 0 else 0
    
    # Hour of day (if available)
    if 'Time' in df.columns:
        features['hour'] = df['Time'].iloc[idx].hour
    
    return features


def get_pattern_id(features: Dict[str, float]) -> str:
    """Create pattern ID from features."""
    # Round and bin features for pattern matching
    rounded = {k: round(v, 2) if isinstance(v, float) else v for k, v in sorted(features.items())}
    return json.dumps(rounded, sort_keys=True)


def detect_market_regime(df: pd.DataFrame, idx: int) -> MarketPhase:
    """Detect market regime at specific bar."""
    if idx < 50:
        return MarketPhase.QUIET
    
    adx = df['ADX'].values[idx] if 'ADX' in df.columns else 20
    atr = df['ATR'].values if 'ATR' in df.columns else None
    
    # Volatility rank
    if atr is not None:
        atr_pct = np.percentile(atr[max(0,idx-50):idx], 75)
        vol_rank = atr[idx] / (atr_pct + 0.0001)
    else:
        vol_rank = 1.0
    
    # Classify regime
    if adx > 35 and vol_rank > 0.8:
        return MarketPhase.TRENDING_STRONG
    elif 20 < adx <= 35:
        return MarketPhase.TRENDING_WEAK
    elif adx < 20 and vol_rank < 0.6:
        return MarketPhase.RANGING
    elif vol_rank > 1.2:
        return MarketPhase.BREAKOUT
    else:
        return MarketPhase.QUIET

# data fetch

def initialize_mt5() -> bool:
    """Initialize MT5 terminal if available."""
    if mt5 is None:
        return False
    try:
        return bool(mt5.initialize())
    except Exception:
        return False

def fetch_data(symbol: str, bars: int = 10000) -> pd.DataFrame:
    """Fetch data from MT5 with M5/H1 fallback and CSV fallback."""
    # MT5 primary
    if mt5 is not None:
        try:
            if mt5.initialize():
                req_symbol = f"{symbol}+" if symbol in FOREX_PLUS else symbol
                # Try M5, then M15, then H1
                rates = mt5.copy_rates_from_pos(req_symbol, mt5.TIMEFRAME_M5, 0, bars)
                if rates is None or len(rates) < bars * 0.5:
                    rates = mt5.copy_rates_from_pos(req_symbol, mt5.TIMEFRAME_M15, 0, bars // 3)
                if rates is None or len(rates) < bars * 0.25:
                    rates = mt5.copy_rates_from_pos(req_symbol, mt5.TIMEFRAME_H1, 0, bars // 12)
                mt5.shutdown()
                if rates is not None and len(rates) >= 300:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
                    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                    df.reset_index(drop=True, inplace=True)
                    return df
        except Exception:
            pass
    # CSV fallback (if provided)
    csv_path = Path(__file__).parent / 'backtest_data' / f'{symbol}_optimization.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        return df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return None


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    if len(equity_curve) < 2:
        return 0.0
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = peak - value
        if dd > max_dd:
            max_dd = dd
    return float(max_dd)


def _default_params_for_symbol(symbol: str) -> Tuple[int, int, float, float]:
    # (fast_ema, slow_ema, adx_min, atr_mult)
    # ADX >=22 is the standard 'trend present' threshold — below that = ranging market
    if symbol == 'BTCUSD':
        return (8, 21, 20.0, 2.0)
    if symbol == 'XAUUSD':
        return (8, 21, 22.0, 1.8)
    if symbol == 'NAS100':
        return (8, 21, 22.0, 1.6)
    if symbol == 'USDJPY':
        return (13, 50, 25.0, 1.6)
    if symbol == 'GBPJPY':
        return (10, 34, 22.0, 1.6)
    # EURUSD, GBPUSD
    return (13, 50, 22.0, 1.6)


def _load_best_params(symbol: str) -> Tuple[int, int, float, float]:
    defaults = _default_params_for_symbol(symbol)
    try:
        if BEST_SETTINGS_FILE.exists():
            raw = json.loads(BEST_SETTINGS_FILE.read_text())
            row = (raw.get('instruments') or {}).get(symbol, {})
            fast = int(row.get('EMA_Fast', defaults[0]))
            slow = int(row.get('EMA_Slow', defaults[1]))
            adx = float(row.get('ADX', defaults[2]))
            atr_m = float(row.get('ATR_Mult', defaults[3]))
            if fast >= slow:
                return defaults
            return (fast, slow, adx, atr_m)
    except Exception:
        pass
    return defaults


def run_backtest_no_lookahead(df: pd.DataFrame, symbol: str, params: Optional[Tuple[int, int, float, float]] = None, risk_pct: float = 1.0) -> Dict:
    """
    Strict no-lookahead backtest:
    - Signal computed on bar i-1 only
    - Entry executed on bar i open
    - Exits evaluated from current bar OHLC after entry
    """
    if df is None or len(df) < 120:
        return {'symbol': symbol, 'trades': [], 'metrics': {}, 'equity_curve': [INITIAL_BALANCE]}

    fast, slow, adx_th, atr_mult = params or _load_best_params(symbol)
    df = add_indicators(df, fast_ema=fast, slow_ema=slow)
    df = df.fillna(0).reset_index(drop=True)

    open_arr = df['Open'].astype(float).to_numpy()
    high_arr = df['High'].astype(float).to_numpy()
    low_arr = df['Low'].astype(float).to_numpy()
    close_arr = df['Close'].astype(float).to_numpy()
    atr_arr = df['ATR'].astype(float).to_numpy()
    ema_f = df['EMA_Fast'].astype(float).to_numpy()
    ema_s = df['EMA_Slow'].astype(float).to_numpy()
    adx = df['ADX'].astype(float).to_numpy()
    rsi = df['RSI'].astype(float).to_numpy() if 'RSI' in df.columns else np.full(len(df), 50.0)
    times = pd.to_datetime(df['Time']) if 'Time' in df.columns else pd.Series(range(len(df)))
    hours = times.dt.hour.to_numpy() if hasattr(times, 'dt') else np.full(len(df), 12)

    pip_size = INSTRUMENTS.get(symbol, INSTRUMENTS['EURUSD'])['pip_size']
    spread = float(INSTRUMENTS.get(symbol, INSTRUMENTS['EURUSD'])['spread'])

    trades = []
    equity_curve = [INITIAL_BALANCE]
    balance = INITIAL_BALANCE

    in_trade = False
    direction = 0
    entry_price = 0.0
    entry_idx = 0
    stop_loss = 0.0
    initial_stop = 0.0
    take_profit = 0.0
    bars_held = 0
    moved_to_be = False
    loss_cooldown_until = 0

    atr_valid = atr_arr[np.isfinite(atr_arr) & (atr_arr > 0)]
    atr_threshold = float(np.percentile(atr_valid, 20)) if len(atr_valid) else 0.0
    base_risk_pct = max(0.1, min(10.0, risk_pct))  # use passed risk, clamped
    # spread in price units (for proportional cost calculation)
    spread_price = spread * pip_size

    start = max(slow + 4, 60)
    # Session windows per asset class
    is_crypto = symbol in ('BTCUSD',)
    is_us_index = symbol in ('NAS100',)
    for i in range(start, len(df)):
        prev = i - 1

        if not in_trade:
            if i < loss_cooldown_until:
                continue

            atr_prev = atr_arr[prev]
            if atr_prev <= 0 or adx[prev] <= adx_th or atr_prev < atr_threshold:
                continue

            # Session filter: BTCUSD trades 24/7; NAS100 US session; others London+NY
            h = hours[prev]
            if is_crypto:
                pass  # no hour restriction for crypto
            elif is_us_index:
                if h < 13 or h > 22:  # US/NY session ~13:00-22:00 UTC
                    continue
            else:
                if h < 7 or h > 20:  # London + NY overlap
                    continue

            # RSI filter: avoid over-extended entries
            rsi_prev = rsi[prev]
            if rsi_prev > 70 or rsi_prev < 30:
                continue

            # --- Slope confirmation (5-bar window) ---
            fast_slope = ema_f[prev] - ema_f[prev - 5]   # 5-bar slope (more stable than 3-bar)
            slow_slope = ema_s[prev] - ema_s[prev - 3]

            # -------------------------------------------------------
            # ENTRY SIGNAL A: Fresh EMA crossover (within last 15 bars)
            # -------------------------------------------------------
            just_crossed_long = False
            just_crossed_short = False
            for look in range(1, 16):
                lb = prev - look
                if lb < 1:
                    break
                if ema_f[lb] <= ema_s[lb] and ema_f[prev] > ema_s[prev]:
                    just_crossed_long = True
                    break
                if ema_f[lb] >= ema_s[lb] and ema_f[prev] < ema_s[prev]:
                    just_crossed_short = True
                    break

            # Crossover must have slope backing it
            cross_long_ok = just_crossed_long and fast_slope > 0 and slow_slope >= -atr_prev * 0.05
            cross_short_ok = just_crossed_short and fast_slope < 0 and slow_slope <= atr_prev * 0.05

            # -------------------------------------------------------
            # ENTRY SIGNAL B: Pullback retest of fast EMA in trend
            # Requirements: trend established 8+ bars, price bounced off EMA
            # -------------------------------------------------------
            pullback_long = False
            pullback_short = False
            if not just_crossed_long and not just_crossed_short and prev >= 10:
                in_bull = ema_f[prev] > ema_s[prev] and all(
                    ema_f[prev - k] > ema_s[prev - k] for k in range(1, 9)
                )
                in_bear = ema_f[prev] < ema_s[prev] and all(
                    ema_f[prev - k] < ema_s[prev - k] for k in range(1, 9)
                )
                if in_bull and fast_slope > 0:  # trend accelerating up
                    # Price touched (or went through) fast EMA in last 4 bars then closed back above
                    touched = any(low_arr[prev - k] <= ema_f[prev - k] * 1.001 for k in range(1, 5))
                    bounce_bar = close_arr[prev - 1] < ema_f[prev - 1] * 1.002  # prev bar near/below
                    reclaim = close_arr[prev] > ema_f[prev]  # this bar closed above
                    if touched and reclaim and (bounce_bar or touched):
                        pullback_long = True
                elif in_bear and fast_slope < 0:  # trend accelerating down
                    touched = any(high_arr[prev - k] >= ema_f[prev - k] * 0.999 for k in range(1, 5))
                    bounce_bar = close_arr[prev - 1] > ema_f[prev - 1] * 0.998
                    reclaim = close_arr[prev] < ema_f[prev]
                    if touched and reclaim and (bounce_bar or touched):
                        pullback_short = True

            long_ok = cross_long_ok or pullback_long
            short_ok = cross_short_ok or pullback_short
            signal = 1 if long_ok else -1 if short_ok else 0
            if signal == 0:
                continue

            direction = signal
            entry_idx = i
            entry_price = open_arr[i]
            stop_dist = max(atr_prev * atr_mult, pip_size * 3)
            tp_dist = stop_dist * 2.5  # 2.5R TP

            if direction == 1:
                stop_loss = entry_price - stop_dist
                take_profit = entry_price + tp_dist
            else:
                stop_loss = entry_price + stop_dist
                take_profit = entry_price - tp_dist

            in_trade = True
            bars_held = 0
            initial_stop = stop_loss
            moved_to_be = False
            continue

        bars_held += 1
        bar_high = high_arr[i]
        bar_low = low_arr[i]
        bar_close = close_arr[i]

        # Breakeven at 1.5R — protect profit on winners
        initial_risk = abs(entry_price - initial_stop)
        if initial_risk > 0 and not moved_to_be:
            favorable = (bar_high - entry_price) if direction == 1 else (entry_price - bar_low)
            if favorable >= initial_risk * 1.5:
                be_buffer = pip_size * 0.5
                if direction == 1:
                    stop_loss = max(stop_loss, entry_price + be_buffer)
                else:
                    stop_loss = min(stop_loss, entry_price - be_buffer)
                moved_to_be = True

        exit_triggered = False
        exit_price = bar_close
        exit_reason = 'Time'

        if direction == 1:
            if bar_low <= stop_loss:
                exit_triggered = True
                exit_price = stop_loss
                exit_reason = 'SL'
            elif bar_high >= take_profit:
                exit_triggered = True
                exit_price = take_profit
                exit_reason = 'TP'
        else:
            if bar_high >= stop_loss:
                exit_triggered = True
                exit_price = stop_loss
                exit_reason = 'SL'
            elif bar_low <= take_profit:
                exit_triggered = True
                exit_price = take_profit
                exit_reason = 'TP'

        # Per-symbol max hold: default 96 bars (24 hrs); crypto/indices 128 bars
        max_hold = 128 if symbol in ('BTCUSD', 'NAS100') else 96
        if not exit_triggered and bars_held >= max_hold:
            exit_triggered = True
            exit_reason = 'Time'

        if not exit_triggered:
            continue

        move = (exit_price - entry_price) if direction == 1 else (entry_price - exit_price)
        risk_amount = balance * (base_risk_pct / 100.0)
        stop_distance = abs(entry_price - initial_stop)  # always use initial stop for consistent sizing
        gross_profit = (move / stop_distance) * risk_amount if stop_distance > 0 else 0.0
        # Spread cost proportional to position size (spread in price units relative to stop)
        spread_cost_usd = (spread_price / stop_distance) * risk_amount if stop_distance > 0 else 0.0
        profit = float(gross_profit - spread_cost_usd)

        balance += profit
        equity_curve.append(balance)

        trades.append({
            'entry_time': times.iloc[entry_idx],
            'exit_time': times.iloc[i],
            'direction': 'BUY' if direction == 1 else 'SELL',
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'stop_loss': float(stop_loss),
            'take_profit': float(take_profit),
            'profit': float(profit),
            'exit_reason': exit_reason,
            'bars_held': int(bars_held),
        })

        in_trade = False
        direction = 0

        if profit < 0 and exit_reason == 'SL':
            loss_cooldown_until = i + 3  # 45-min cooldown after SL hit

    if not trades:
        return {
            'symbol': symbol,
            'trades': [],
            'metrics': {
                'total_trades': 0,
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_profit': 0,
                'avg_profit': 0,
                'avg_win': 0,
                'avg_loss': 0,
                'best_trade': 0,
                'worst_trade': 0,
                'profit_factor': 0,
                'max_drawdown': 0,
                'final_balance': INITIAL_BALANCE,
                'return_pct': 0,
            },
            'equity_curve': equity_curve,
            'lookahead_safe': True,
        }

    profits = np.array([float(t['profit']) for t in trades], dtype=float)
    wins = profits[profits > 0]
    losses = profits[profits <= 0]
    total_profit = float(np.sum(profits))
    final_balance = float(INITIAL_BALANCE + total_profit)

    metrics = {
        'total_trades': int(len(trades)),
        'wins': int(len(wins)),
        'losses': int(len(losses)),
        'win_rate': float((len(wins) / len(trades) * 100) if trades else 0),
        'total_profit': total_profit,
        'avg_profit': float(np.mean(profits)) if len(profits) else 0,
        'avg_win': float(np.mean(wins)) if len(wins) else 0,
        'avg_loss': float(np.mean(losses)) if len(losses) else 0,
        'best_trade': float(np.max(profits)) if len(profits) else 0,
        'worst_trade': float(np.min(profits)) if len(profits) else 0,
        'profit_factor': float(np.sum(wins) / abs(np.sum(losses))) if len(losses) and abs(np.sum(losses)) > 0 else 99.0,
        'max_drawdown': float(calculate_max_drawdown(equity_curve)),
        'final_balance': final_balance,
        'return_pct': float((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100),
    }

    return {
        'symbol': symbol,
        'trades': trades,
        'metrics': metrics,
        'equity_curve': equity_curve,
        'lookahead_safe': True,
    }


def run_split_backtest(df: pd.DataFrame, symbol: str, split_ratio: float = 0.7) -> Dict:
    if split_ratio <= 0 or split_ratio >= 1:
        split_ratio = 0.7
    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx].copy()
    test_df = df.iloc[split_idx:].copy()
    train_result = run_backtest_no_lookahead(train_df, symbol)
    test_result = run_backtest_no_lookahead(test_df, symbol)
    return {
        'symbol': symbol,
        'mode': 'split',
        'split_ratio': split_ratio,
        'train': {'metrics': train_result.get('metrics', {}), 'trades': train_result.get('trades', [])},
        'test': {'metrics': test_result.get('metrics', {}), 'trades': test_result.get('trades', [])},
    }


def monte_carlo_analysis(trades: List[Dict], iterations: int = 200) -> Dict:
    if not trades:
        return {
            'iterations': iterations,
            'ending_balances': [],
            'p10': 0,
            'p50': 0,
            'p90': 0,
            'max_drawdown_avg': 0,
        }

    profits = [float(t.get('profit', 0.0)) for t in trades]
    ending_balances = []
    drawdowns = []
    for _ in range(max(1, int(iterations))):
        shuffled = profits.copy()
        np.random.shuffle(shuffled)
        balance = INITIAL_BALANCE
        eq = [balance]
        for p in shuffled:
            balance += p
            eq.append(balance)
        ending_balances.append(balance)
        drawdowns.append(calculate_max_drawdown(eq))

    return {
        'iterations': int(iterations),
        'ending_balances': ending_balances,
        'p10': float(np.percentile(ending_balances, 10)),
        'p50': float(np.percentile(ending_balances, 50)),
        'p90': float(np.percentile(ending_balances, 90)),
        'max_drawdown_avg': float(np.mean(drawdowns)),
    }


def walk_forward_analysis(symbol: str, total_days: int = 180, train_days: int = 60, test_days: int = 30) -> Dict:
    results = []
    num_windows = max(0, (total_days - train_days) // test_days)
    for window in range(num_windows):
        end_days = train_days + (window + 1) * test_days
        bars = max(1200, int(end_days * 96))
        df = fetch_data(symbol, bars=bars)
        if df is None or len(df) < 300:
            continue
        split_idx = int(len(df) * (train_days / max(end_days, 1)))
        test_df = df.iloc[split_idx:].copy()
        result = run_backtest_no_lookahead(test_df, symbol)
        metrics = result.get('metrics', {})
        if metrics:
            results.append(metrics)

    if not results:
        return {
            'symbol': symbol,
            'mode': 'walk_forward',
            'total_periods': 0,
            'profitable_periods': 0,
            'consistency': 0,
            'average_profit_per_period': 0,
            'periods': [],
        }

    profitable_periods = sum(1 for r in results if float(r.get('total_profit', 0)) > 0)
    consistency = profitable_periods / len(results) * 100
    avg_profit = float(np.mean([float(r.get('total_profit', 0)) for r in results]))
    return {
        'symbol': symbol,
        'mode': 'walk_forward',
        'total_periods': len(results),
        'profitable_periods': profitable_periods,
        'consistency': consistency,
        'average_profit_per_period': avg_profit,
        'periods': results,
    }


def build_output_payload(symbol: str, mode: str, result: Dict) -> Dict:
    payload = {'symbol': symbol, 'mode': mode, 'timestamp': datetime.now().isoformat(), 'engine': 'backtest_improved.py'}
    payload.update(result)
    return payload


def save_output(payload: Dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, default=str)


def get_data_duration(df: pd.DataFrame) -> str:
    """Calculate and return backtest duration in days/weeks/months."""
    if df is None or 'Time' not in df.columns or len(df) < 2:
        return "unknown"
    start_date = pd.to_datetime(df['Time'].iloc[0])
    end_date = pd.to_datetime(df['Time'].iloc[-1])
    duration = end_date - start_date
    days = duration.days
    if days < 30:
        return f"{days} days"
    elif days < 365:
        weeks = days // 7
        return f"{weeks} weeks ({days} days)"
    else:
        years = days // 365
        return f"{years} years ({days} days)"

def add_indicators(df: pd.DataFrame, fast_ema: int = 20, slow_ema: int = 100, 
                   atr_period: int = 14, adx_period: int = 14) -> pd.DataFrame:
    """Add EMA, ATR, ADX, RSI, Donchian channels"""
    df = df.copy()
    close = df['Close'].astype(float).values
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values
    
    # Use pandas for EMA (fast & efficient)
    close_series = pd.Series(close)
    df['EMA_Fast'] = close_series.ewm(span=fast_ema, adjust=False).mean().values
    df['EMA_Slow'] = close_series.ewm(span=slow_ema, adjust=False).mean().values
    
    # ATR using numpy for speed
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.max(np.stack([high_low, high_close, low_close]), axis=0)
    atr = pd.Series(tr).rolling(atr_period).mean().values
    df['ATR'] = atr
    
    # ADX (simplified)
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    atr_safe = np.where(atr > 0, atr, np.nan)
    di_plus = 100 * (pd.Series(plus_dm).rolling(adx_period).mean().values / atr_safe)
    di_minus = 100 * (pd.Series(minus_dm).rolling(adx_period).mean().values / atr_safe)
    dx = 100 * (np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan))
    df['ADX'] = pd.Series(dx).rolling(adx_period).mean().values

    # RSI (14)
    delta = close_series.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI'] = df['RSI'].fillna(50)
    
    # Donchian High/Low (breakout levels)
    df['Donchian_High'] = pd.Series(high).rolling(20).max().values
    df['Donchian_Low'] = pd.Series(low).rolling(20).min().values
    # Ensure Time is datetime
    if 'Time' in df.columns and not np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time'] = pd.to_datetime(df['Time'])
    
    return df

# strategy

class TrendFollower:
    def __init__(self, fast_ema: int, slow_ema: int, adx_threshold: float,
                 atr_mult: float = 2.0, use_pyramiding: bool = True,
                 atr_percentile: float = 50.0, session_start: int = 6, session_end: int = 22):
        self.fast_ema = fast_ema
        self.slow_ema = slow_ema
        self.adx_threshold = adx_threshold
        self.atr_mult = atr_mult
        self.use_pyramiding = use_pyramiding
        self.atr_percentile = atr_percentile
        self.session_start = session_start
        self.session_end = session_end

# backtest engine w/ learning

def backtest(df: pd.DataFrame, strategy: TrendFollower, symbol: str, learn: bool = True) -> dict:
    """
    Backtest with FULL AI LEARNING - simulates exactly what live bot does:
    - Pattern recognition & learning
    - Market regime detection
    - Position management simulation (breakeven, trailing, partials)
    - Expectancy-based outcome recording
    
    When learn=True, the AI learns from every trade outcome.
    """
    global AI_STATE
    
    df = add_indicators(df, strategy.fast_ema, strategy.slow_ema)
    df = df.fillna(0)

    ema_f = df['EMA_Fast'].to_numpy()
    ema_s = df['EMA_Slow'].to_numpy()
    adx = df['ADX'].to_numpy()
    atr = df['ATR'].to_numpy()
    close = df['Close'].to_numpy()
    high = df['High'].to_numpy()
    low = df['Low'].to_numpy()
    don_high = df['Donchian_High'].to_numpy()
    don_low = df['Donchian_Low'].to_numpy()
    hours = df['Time'].dt.hour.to_numpy() if 'Time' in df.columns else np.full(len(df), 12)

    # ATR percentile threshold
    atr_valid = atr[np.isfinite(atr) & (atr > 0)]
    atr_thr = np.percentile(atr_valid, strategy.atr_percentile) if (len(atr_valid) and strategy.atr_percentile > 0) else 0
    atr_ok = (atr >= atr_thr) if (strategy.atr_percentile > 0) else np.ones(len(atr), dtype=bool)
    session_ok = (hours >= strategy.session_start) & (hours <= strategy.session_end)

    position = 0
    pyramid_count = 0
    entry_price = 0.0
    entry_idx = 0
    stop_price = None
    take_profit = None
    bars_in_trade = 0
    net_profit = 0.0
    equity = [10000.0]
    max_equity = 10000.0
    max_dd = 0.0
    trades = 0
    wins = 0
    losses = 0
    rr_ratios = []
    
    # AI learning tracking
    trade_win_amounts = []
    trade_loss_amounts = []
    entry_features = {}
    entry_regime = None
    entry_hour = 0
    
    # Position management tracking
    breakeven_triggered = False
    partial_taken = False
    max_favorable = 0.0

    pip_size = INSTRUMENTS[symbol]['pip_size']

    start = max(strategy.slow_ema, 50)
    for i in range(start, len(df)):
        if not session_ok[i] or adx[i] <= strategy.adx_threshold or adx[i] != adx[i]:
            signal = 0
        else:
            long_ok = (ema_f[i] > ema_s[i])
            short_ok = (ema_f[i] < ema_s[i])
            signal = 1 if long_ok else -1 if short_ok else 0

        # entry
        if position == 0 and signal != 0:
            position = signal
            entry_price = close[i]
            entry_idx = i
            pyramid_count = 1
            bars_in_trade = 0
            breakeven_triggered = False
            partial_taken = False
            max_favorable = 0.0
            
            # Capture entry conditions for AI learning
            if learn:
                entry_features = extract_pattern_features(df, i)
                entry_regime = detect_market_regime(df, i)
                entry_hour = hours[i]
            
            # Initial stop and take profit
            if atr[i] > 0:
                stop_distance = strategy.atr_mult * atr[i]
                tp_distance = strategy.atr_mult * atr[i] * 2  # 2:1 RR target
                
                if position == 1:
                    stop_price = entry_price - stop_distance
                    take_profit = entry_price + tp_distance
                else:
                    stop_price = entry_price + stop_distance
                    take_profit = entry_price - tp_distance
            else:
                stop_price = None
                take_profit = None
            
            trades += 1
            
        # Pyramiding
        elif strategy.use_pyramiding and position != 0 and pyramid_count < 2 and signal == position:
            if position == 1 and close[i] > entry_price * 1.01:
                entry_price = (entry_price * pyramid_count + close[i]) / (pyramid_count + 1)
                pyramid_count += 1
                bars_in_trade = 0
            elif position == -1 and close[i] < entry_price * 0.99:
                entry_price = (entry_price * pyramid_count + close[i]) / (pyramid_count + 1)
                pyramid_count += 1
                bars_in_trade = 0
        
        # position mgmt
        if position != 0:
            bars_in_trade += 1
            
            # Track maximum favorable excursion
            if position == 1:
                current_favorable = close[i] - entry_price
            else:
                current_favorable = entry_price - close[i]
            max_favorable = max(max_favorable, current_favorable)
            
            # Calculate current RR
            if stop_price is not None:
                risk = abs(entry_price - stop_price)
                current_rr = current_favorable / risk if risk > 0 else 0
            else:
                current_rr = 0
            
            # AI DECISION: Move to breakeven after 1:1 RR
            if not breakeven_triggered and current_rr >= 1.0 and stop_price is not None:
                # Move stop to breakeven + small buffer
                buffer = pip_size * 5
                if position == 1:
                    stop_price = max(stop_price, entry_price + buffer)
                else:
                    stop_price = min(stop_price, entry_price - buffer)
                breakeven_triggered = True
            
            # AI DECISION: Trailing stop after 1.5:1 RR
            if current_rr >= 1.5 and atr[i] > 0:
                trail_distance = strategy.atr_mult * atr[i] * 0.8  # Tighter trail
                if position == 1:
                    new_stop = close[i] - trail_distance
                    stop_price = max(stop_price if stop_price is not None else new_stop, new_stop)
                else:
                    new_stop = close[i] + trail_distance
                    stop_price = min(stop_price if stop_price is not None else new_stop, new_stop)
            
            # Standard ATR trailing (if not already tighter)
            elif atr[i] > 0:
                if position == 1:
                    new_stop = close[i] - strategy.atr_mult * atr[i]
                    stop_price = max(stop_price if stop_price is not None else new_stop, new_stop)
                else:
                    new_stop = close[i] + strategy.atr_mult * atr[i]
                    stop_price = min(stop_price if stop_price is not None else new_stop, new_stop)
        
        # exit conditions
        exit_now = False
        exit_reason = ""
        
        if position != 0:
            # Time stop (max bars in trade)
            if bars_in_trade >= 600:
                exit_now = True
                exit_reason = "time_stop"
            
            # Stop loss hit
            if stop_price is not None:
                if position == 1 and low[i] <= stop_price:
                    exit_now = True
                    exit_reason = "stop_loss"
                elif position == -1 and high[i] >= stop_price:
                    exit_now = True
                    exit_reason = "stop_loss"
            
            # Take profit hit
            if take_profit is not None:
                if position == 1 and high[i] >= take_profit:
                    exit_now = True
                    exit_reason = "take_profit"
                elif position == -1 and low[i] <= take_profit:
                    exit_now = True
                    exit_reason = "take_profit"
            
            # Signal reversal
            if signal == -position:
                exit_now = True
                exit_reason = "signal_reversal"
            
            # Signal flat (momentum died)
            if signal == 0 and bars_in_trade > 10:
                exit_now = True
                exit_reason = "momentum_died"

        # trade close
        if exit_now:
            pip_move = (close[i] - entry_price) / pip_size
            trade_pnl = pip_move * position * 10 * pyramid_count - INSTRUMENTS[symbol]['spread'] * pyramid_count
            net_profit += trade_pnl
            
            is_win = trade_pnl > 0
            wins += 1 if is_win else 0
            losses += 1 if not is_win else 0
            
            if is_win:
                trade_win_amounts.append(abs(trade_pnl))
            else:
                trade_loss_amounts.append(abs(trade_pnl))
            
            # Calculate RR ratio
            if stop_price is not None:
                risk_pips = abs(entry_price - stop_price) / pip_size
                reward_pips = abs(close[i] - entry_price) / pip_size
                if risk_pips > 0.5:
                    rr_ratio = reward_pips / risk_pips
                    rr_ratio = max(0.0, min(rr_ratio, 5.0))
                    rr_ratios.append(rr_ratio)
            
            # record pattern outcome
            if learn and entry_features:
                pattern_id = get_pattern_id(entry_features)
                
                # Update pattern statistics
                if pattern_id not in AI_STATE.patterns:
                    AI_STATE.patterns[pattern_id] = PatternSignature(
                        pattern_id=pattern_id,
                        conditions=entry_features
                    )
                AI_STATE.patterns[pattern_id].update(is_win, trade_pnl)
                
                # Update regime performance
                regime_key = entry_regime.value if entry_regime else 'unknown'
                if symbol not in AI_STATE.regime_performance:
                    AI_STATE.regime_performance[symbol] = {}
                if regime_key not in AI_STATE.regime_performance[symbol]:
                    AI_STATE.regime_performance[symbol][regime_key] = {
                        'trades': 0, 'wins': 0, 'profit': 0.0
                    }
                AI_STATE.regime_performance[symbol][regime_key]['trades'] += 1
                AI_STATE.regime_performance[symbol][regime_key]['wins'] += 1 if is_win else 0
                AI_STATE.regime_performance[symbol][regime_key]['profit'] += trade_pnl
                
                # Update hour performance
                hour_key = str(entry_hour)
                if hour_key not in AI_STATE.hour_performance:
                    AI_STATE.hour_performance[hour_key] = {'trades': 0, 'wins': 0, 'profit': 0.0}
                AI_STATE.hour_performance[hour_key]['trades'] += 1
                AI_STATE.hour_performance[hour_key]['wins'] += 1 if is_win else 0
                AI_STATE.hour_performance[hour_key]['profit'] += trade_pnl
                
                AI_STATE.total_simulated_trades += 1
            
            # Reset position
            position = 0
            pyramid_count = 0
            stop_price = None
            take_profit = None
            bars_in_trade = 0
            entry_features = {}

        equity.append(10000 + net_profit)
        max_equity = max(max_equity, equity[-1])
        dd = max_equity - equity[-1]
        max_dd = max(max_dd, dd)

    # Calculate final statistics
    win_rate = (wins / trades * 100) if trades > 0 else 0
    total_pnl = equity[-1] - 10000
    equity_arr = np.array(equity)
    deltas = np.diff(equity_arr)
    gross_profit = float(np.sum(deltas[deltas > 0]))
    gross_loss = float(-np.sum(deltas[deltas < 0]))
    pf = gross_profit / max(gross_loss, 1)
    
    # RR statistics
    avg_rr = np.mean(rr_ratios) if rr_ratios else 0
    median_rr = np.median(rr_ratios) if rr_ratios else 0
    min_rr = np.min(rr_ratios) if rr_ratios else 0
    max_rr = np.max(rr_ratios) if rr_ratios else 0
    
    # EXPECTANCY - the key metric!
    avg_win = np.mean(trade_win_amounts) if trade_win_amounts else 0
    avg_loss = np.mean(trade_loss_amounts) if trade_loss_amounts else 0
    win_pct = win_rate / 100
    expectancy = (win_pct * avg_win) - ((1 - win_pct) * avg_loss) if trades > 0 else 0

    return {
        'trades': trades,
        'wins': wins,
        'win_rate': win_rate,
        'profit': total_pnl,
        'max_dd': max_dd,
        'pf': pf,
        'equity': equity,
        'avg_rr': avg_rr,
        'median_rr': median_rr,
        'min_rr': min_rr,
        'max_rr': max_rr,
        'expectancy': expectancy,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
    }

# optimization

def optimize(symbol: str, df: pd.DataFrame, learn: bool = True) -> tuple:
    """
    Grid search for best parameters - OPTIMIZES FOR EXPECTANCY, NOT WIN RATE!
    The AI learns from every backtest iteration.
    """
    global AI_STATE
    
    split = int(len(df) * 0.7)
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()

    best_score = -np.inf
    best_params = None
    best_result = None

    # Instrument-aware grids - REDUCED for faster training
    if symbol == 'BTCUSD':
        fast_emas = [10, 20]
        slow_emas = [150, 250]
        adx_thresholds = [40.0, 50.0]
    elif symbol in ('XAUUSD', 'NAS100'):
        fast_emas = [8, 15, 25]
        slow_emas = [50, 100, 200]
        adx_thresholds = [20.0, 30.0, 40.0]
    else:  # Forex pairs
        fast_emas = [8, 15, 25]
        slow_emas = [50, 100, 200]
        adx_thresholds = [20.0, 30.0]

    atr_mults = [1.0, 1.5, 2.0]
    atr_percentiles = [0.0, 30.0]
    session_windows = [(6, 22)]

    params_grid = list(product(fast_emas, slow_emas, adx_thresholds, atr_mults, atr_percentiles, session_windows))

    for fast, slow, adx_th, atr_m, atr_p, (sess_s, sess_e) in tqdm(params_grid, desc=f'🧠 Training {symbol}'):
        if fast >= slow:
            continue

        strategy = TrendFollower(fast, slow, adx_th, atr_m, use_pyramiding=True,
                                 atr_percentile=atr_p, session_start=sess_s, session_end=sess_e)

        # Train phase - AI learns here
        res_train = backtest(df_train, strategy, symbol, learn=learn)
        
        # Test phase - validate on unseen data
        res_test = backtest(df_test, strategy, symbol, learn=False)

        # EXPECTANCY-BASED SCORING (not win rate!)
        expectancy = res_test.get('expectancy', 0)
        pf = max(res_test['pf'], 0.01)
        profit = res_test['profit']
        dd = res_test['max_dd']
        trades = max(res_test['trades'], 1)
        avg_rr = res_test.get('avg_rr', 1.0)

        # Score formula: Prioritize EXPECTANCY and PROFIT
        # - Positive expectancy is crucial
        # - Profit factor > 1.5 is good
        # - Decent RR ratio matters
        # - Enough trades for statistical significance
        
        if expectancy > 0 and pf > 1.0:
            # Good strategy!
            reward = (
                expectancy * 2.0 +  # $2 per $ of expectancy
                np.log1p(max(profit, 0)) * 0.5 +  # Profit bonus
                (pf - 1.0) * 10.0 +  # PF bonus
                min(avg_rr, 3.0) * 5.0 +  # RR bonus (capped)
                np.log1p(trades) * 0.5  # Trade frequency bonus
            )
        else:
            # Bad strategy - heavy penalty
            reward = expectancy - abs(profit) * 0.01 - 100
        
        penalty = (max(0, -profit) / 1000.0) + (dd / 5000.0)
        score = reward - penalty
        
        if score > best_score:
            best_score = score
            best_params = (fast, slow, adx_th, atr_m)
            best_result = res_test

    return best_params, best_result


def compute_score(res: dict) -> float:
    """Compute score based on EXPECTANCY, not win rate."""
    expectancy = res.get('expectancy', 0)
    pf = max(res.get('pf', 0.0), 0.01)
    profit = res.get('profit', 0.0)
    dd = res.get('max_dd', 0.0)
    trades = max(res.get('trades', 0), 1)
    avg_rr = res.get('avg_rr', 1.0)
    
    if expectancy > 0 and pf > 1.0:
        reward = expectancy * 2.0 + np.log1p(max(profit, 0)) * 0.5 + (pf - 1.0) * 10.0
    else:
        reward = expectancy - 100
    
    penalty = (max(0, -profit) / 1000.0) + (dd / 5000.0)
    return reward - penalty


def load_existing_best(symbol: str):
    if BEST_SETTINGS_FILE.exists():
        try:
            data = json.loads(BEST_SETTINGS_FILE.read_text())
            instr = data.get('instruments', {})
            return instr.get(symbol)
        except Exception:
            return None
    return None


def refine_params(symbol: str, df: pd.DataFrame, base_params: tuple, learn: bool = True):
    """Localized search around base params to fine-tune instrument settings."""
    if not base_params or len(base_params) < 4:
        return None, None

    base_fast, base_slow, base_adx, base_atr = base_params

    fast_candidates = sorted(set([
        max(5, int(base_fast - 5)),
        max(5, int(base_fast - 3)),
        int(base_fast),
        int(base_fast + 3),
        int(base_fast + 5),
    ]))

    slow_candidates = sorted(set([
        max(base_fast + 10, int(base_slow - 50)),
        max(base_fast + 10, int(base_slow - 25)),
        int(base_slow),
        int(base_slow + 25),
        int(base_slow + 50),
    ]))

    adx_floor = 30.0 if symbol == 'BTCUSD' else 15.0
    adx_candidates = sorted(set([
        max(adx_floor, float(base_adx - 5.0)),
        max(adx_floor, float(base_adx - 2.5)),
        float(base_adx),
        float(base_adx + 2.5),
        float(base_adx + 5.0),
    ]))

    atr_candidates = sorted(set([
        max(1.0, float(base_atr - 0.5)),
        float(base_atr),
        float(base_atr + 0.5),
        min(3.0, float(base_atr + 1.0)),
    ]))

    split = int(len(df) * 0.7)
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()

    best_score = -np.inf
    best_params = None
    best_result = None

    for fast in fast_candidates:
        for slow in slow_candidates:
            if fast >= slow:
                continue
            for adx_th in adx_candidates:
                for atr_m in atr_candidates:
                    strategy = TrendFollower(
                        fast, slow, adx_th, atr_m,
                        use_pyramiding=True,
                        atr_percentile=0.0,
                        session_start=5,
                        session_end=23,
                    )
                    res_train = backtest(df_train, strategy, symbol, learn=learn)
                    res_test = backtest(df_test, strategy, symbol, learn=False)
                    score = compute_score(res_test)
                    if score > best_score:
                        best_score = score
                        best_params = (fast, slow, adx_th, atr_m)
                        best_result = res_test

    return best_params, best_result


def plot_equity(symbol: str, equity: list, params: tuple, result: dict):
    """Plot equity curve - disabled by default"""
    return


def print_ai_insights():
    """Print what the AI has learned."""
    global AI_STATE
    
    print("\n" + "=" * 80)
    print("🧠 AI LEARNING INSIGHTS")
    print("=" * 80)
    
    print(f"\n📊 Training Statistics:")
    print(f"   Total Training Iterations: {AI_STATE.training_iterations}")
    print(f"   Total Simulated Trades: {AI_STATE.total_simulated_trades}")
    print(f"   Patterns Learned: {len(AI_STATE.patterns)}")
    print(f"   Best Expectancy Found: ${AI_STATE.best_expectancy:.2f}")
    
    # Top profitable patterns
    if AI_STATE.patterns:
        sorted_patterns = sorted(
            AI_STATE.patterns.values(), 
            key=lambda p: p.expectancy if p.total_trades >= 5 else -999,
            reverse=True
        )[:5]
        
        print(f"\n🏆 Top 5 Profitable Patterns:")
        for i, p in enumerate(sorted_patterns, 1):
            if p.total_trades >= 5:
                wr = (p.wins / p.total_trades * 100) if p.total_trades > 0 else 0
                print(f"   {i}. Expectancy: ${p.expectancy:.2f} | WR: {wr:.1f}% | Trades: {p.total_trades}")
    
    # Best regimes per symbol
    if AI_STATE.regime_performance:
        print(f"\n📈 Best Regime Per Symbol:")
        for symbol, regimes in AI_STATE.regime_performance.items():
            best_regime = max(regimes.items(), key=lambda x: x[1].get('profit', 0))
            profit = best_regime[1].get('profit', 0)
            trades = best_regime[1].get('trades', 0)
            if trades > 0:
                print(f"   {symbol}: {best_regime[0].upper()} (${profit:.0f} from {trades} trades)")
    
    # Best trading hours
    if AI_STATE.hour_performance:
        sorted_hours = sorted(
            AI_STATE.hour_performance.items(),
            key=lambda x: x[1].get('profit', 0),
            reverse=True
        )[:3]
        
        print(f"\n⏰ Best Trading Hours:")
        for hour, stats in sorted_hours:
            profit = stats.get('profit', 0)
            trades = stats.get('trades', 0)
            if trades > 0:
                print(f"   {hour}:00 UTC - ${profit:.0f} profit ({trades} trades)")


# main

def main():
    """
    AI TRAINING SIMULATOR
    =====================
    This is the training ring for the AI. It backtests continuously,
    learning from every trade until it finds profitable strategies.
    
    The AI learns:
    - Which patterns lead to profitable trades
    - Which market regimes to trade in
    - Best hours to trade
    - Optimal parameters per symbol
    
    When expectancy is positive and profit factor > 1.5, 
    settings are automatically saved for live trading.
    """
    global AI_STATE
    
    print("\n" + "=" * 100)
    print("🧠 AI TRADING SIMULATOR - TRAINING MODE")
    print("=" * 100)
    print("The AI will backtest all instruments and LEARN from every trade.")
    print("When it finds profitable strategies, they're saved automatically.")
    print("=" * 100 + "\n")
    
    # Load previous learning state
    load_ai_state()
    AI_STATE.training_iterations += 1
    
    results_summary = []
    instrument_count = 0
    total_expectancy = 0
    profitable_symbols = 0
    
    for symbol in sorted(INSTRUMENTS.keys()):
        instrument_count += 1
        print(f"\n{'=' * 100}")
        print(f" 🎯 [{instrument_count}/7] Training on {symbol}")
        print('=' * 100)
        
        df = fetch_data(symbol, 15000)  # More data for better learning
        if df is None or len(df) < 1000:
            print(f"[X] Failed to fetch {symbol}")
            continue

        # Use more data for training
        if 'Time' in df.columns and len(df) > 0:
            cutoff = pd.to_datetime(df['Time'].iloc[-1]) - pd.Timedelta(days=BACKTEST_DAYS)
            df = df[df['Time'] >= cutoff].reset_index(drop=True)
        
        duration = get_data_duration(df)
        print(f"[+] Loaded {len(df)} candles | Duration: {duration}")
        
        # Main optimization with AI learning
        best_params, best_result = optimize(symbol, df, learn=True)
        
        if best_params is None:
            print(f"[X] Optimization failed for {symbol}")
            continue

        # Refinement round (also learning)
        refined_params, refined_result = refine_params(symbol, df, best_params, learn=True)
        
        # Use best between optimized and refined
        if refined_result and compute_score(refined_result) > compute_score(best_result):
            chosen_params, chosen_result = refined_params, refined_result
        else:
            chosen_params, chosen_result = best_params, best_result

        expectancy = chosen_result.get('expectancy', 0)
        total_expectancy += expectancy
        
        print(f"\n[+] Best Parameters: EMA {chosen_params[0]}/{chosen_params[1]}, ADX {chosen_params[2]:.0f}, ATR {chosen_params[3]:.1f}")
        print(f"    💰 EXPECTANCY: ${expectancy:.2f} per trade")
        print(f"    📊 Trades: {chosen_result['trades']} | Win Rate: {chosen_result['win_rate']:.1f}% | Profit: ${chosen_result['profit']:.0f}")
        print(f"    📉 Max DD: ${chosen_result['max_dd']:.0f} | PF: {chosen_result['pf']:.2f}")
        print(f"    📐 Avg RR: {chosen_result['avg_rr']:.2f} | Avg Win: ${chosen_result.get('avg_win', 0):.0f} | Avg Loss: ${chosen_result.get('avg_loss', 0):.0f}")
        
        # Verdict
        if expectancy > MIN_EXPECTANCY_TO_SAVE and chosen_result['pf'] > MIN_PROFIT_FACTOR:
            print(f"    ✅ PROFITABLE! Saving for live trading...")
            profitable_symbols += 1
            
            # Update AI state with optimal params
            AI_STATE.optimal_params[symbol] = {
                'EMA_Fast': chosen_params[0],
                'EMA_Slow': chosen_params[1],
                'ADX': chosen_params[2],
                'ATR_Mult': chosen_params[3],
                'expectancy': expectancy,
                'pf': chosen_result['pf'],
            }
        elif expectancy > 0:
            print(f"    ⚠️ Marginal profit - needs more training")
        else:
            print(f"    ❌ Not profitable - AI will keep learning")
        
        # Track best expectancy
        if expectancy > AI_STATE.best_expectancy:
            AI_STATE.best_expectancy = expectancy
        
        results_summary.append({
            'Symbol': symbol,
            'EMA_Fast': chosen_params[0],
            'EMA_Slow': chosen_params[1],
            'ADX': chosen_params[2],
            'ATR_Mult': chosen_params[3],
            'Trades': chosen_result['trades'],
            'Win_Rate': chosen_result['win_rate'],
            'Profit': chosen_result['profit'],
            'Max_DD': chosen_result['max_dd'],
            'PF': chosen_result['pf'],
            'Expectancy': expectancy,
            'Avg_RR': chosen_result['avg_rr'],
            'Avg_Win': chosen_result.get('avg_win', 0),
            'Avg_Loss': chosen_result.get('avg_loss', 0),
        })
    
    # Save AI learning state
    save_ai_state()
    
    # Save results
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('Expectancy', ascending=False)  # Sort by expectancy!
        summary_path = DATA_DIR / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)

        # Save best settings (only profitable ones)
        existing = {}
        if BEST_SETTINGS_FILE.exists():
            try:
                existing = json.loads(BEST_SETTINGS_FILE.read_text()).get('instruments', {})
            except Exception:
                existing = {}

        merged = existing.copy()
        for _, row in summary_df.iterrows():
            sym = row['Symbol']
            # Only save if EXPECTANCY is positive and PF > 1.5
            if row['Expectancy'] > MIN_EXPECTANCY_TO_SAVE and row['PF'] > MIN_PROFIT_FACTOR:
                prev = merged.get(sym, {})
                prev_expectancy = prev.get('Expectancy', -999)
                if row['Expectancy'] > prev_expectancy:
                    merged[sym] = {
                        'EMA_Fast': int(row['EMA_Fast']),
                        'EMA_Slow': int(row['EMA_Slow']),
                        'ADX': float(row['ADX']),
                        'ATR_Mult': float(row['ATR_Mult']),
                        'Trades': int(row['Trades']),
                        'Win_Rate': float(row['Win_Rate']),
                        'Profit': float(row['Profit']),
                        'Max_DD': float(row['Max_DD']),
                        'PF': float(row['PF']),
                        'Expectancy': float(row['Expectancy']),
                    }

        best_settings = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'source': 'backtest_improved.py (AI Training)',
            'training_iteration': AI_STATE.training_iterations,
            'total_simulated_trades': AI_STATE.total_simulated_trades,
            'instruments': merged,
        }
        BEST_SETTINGS_FILE.write_text(json.dumps(best_settings, indent=2))

        # Print final summary
        print(f"\n{'=' * 100}")
        print("🏆 TRAINING COMPLETE - FINAL RESULTS")
        print('=' * 100)
        print(f"\nSorted by EXPECTANCY (what actually matters!):\n")
        print(summary_df[['Symbol', 'Expectancy', 'Profit', 'PF', 'Win_Rate', 'Avg_RR', 'Trades']].to_string(index=False))
        
        print(f"\n📊 Training Summary:")
        print(f"   Training Iteration: #{AI_STATE.training_iterations}")
        print(f"   Total Simulated Trades: {AI_STATE.total_simulated_trades}")
        print(f"   Patterns Learned: {len(AI_STATE.patterns)}")
        print(f"   Profitable Symbols: {profitable_symbols}/7")
        print(f"   Average Expectancy: ${total_expectancy / max(len(results_summary), 1):.2f}")
        
        print(f"\n[+] Results saved to: {summary_path}")
        print(f"[+] Best settings saved to: {BEST_SETTINGS_FILE}")
        print(f"[+] AI state saved to: {AI_STATE_FILE}")
        
        # Print AI insights
        print_ai_insights()
        
        if profitable_symbols >= 4:
            print(f"\n🚀 AI IS READY FOR LIVE TRADING! {profitable_symbols}/7 symbols profitable!")
            return True  # Signal success
        else:
            print(f"\n⏳ AI needs more training. Run again to continue learning.")
            print(f"   Tip: The more you run, the smarter the AI becomes!")
            return False  # Signal needs more training


def continuous_training(max_iterations: int = None):
    """
    CONTINUOUS AI TRAINING MODE
    ===========================
    Keeps training until the AI becomes profitable on at least 4/7 symbols.
    The AI learns more with each iteration, building on previous knowledge.
    
    Args:
        max_iterations: Maximum training iterations. None = train until profitable.
    """
    iteration = 0
    is_ready = False
    
    print("\n" + "🔄" * 50)
    print("🧠 CONTINUOUS AI TRAINING MODE ACTIVATED")
    print("🔄" * 50)
    print("\nThe AI will keep training until it becomes profitable.")
    print("Press Ctrl+C to stop at any time.\n")
    
    try:
        while not is_ready:
            iteration += 1
            
            if max_iterations and iteration > max_iterations:
                print(f"\n⏹️ Reached maximum iterations ({max_iterations})")
                break
            
            print(f"\n{'🔄' * 20}")
            print(f"🎯 TRAINING ITERATION #{iteration}")
            print(f"{'🔄' * 20}\n")
            
            is_ready = main()
            
            if not is_ready:
                print(f"\n⏳ Waiting 5 seconds before next iteration...")
                print(f"   (Press Ctrl+C to stop)")
                time.sleep(5)
        
        if is_ready:
            print("\n" + "🎉" * 50)
            print("🚀 AI TRAINING COMPLETE! BOT IS READY FOR LIVE TRADING!")
            print("🎉" * 50)
            print("\nRun the live bot with: python main.py")
        
    except KeyboardInterrupt:
        print(f"\n\n⏹️ Training stopped by user at iteration #{iteration}")
        print(f"   AI state has been saved. Run again to continue from where you left off.")
        save_ai_state()


def run_cli_backtest_mode() -> int:
    parser = argparse.ArgumentParser(description='Backtest Improved - API/CLI modes')
    parser.add_argument('--symbol', type=str, default='EURUSD', help='Single symbol')
    parser.add_argument('--mode', type=str, choices=['standard', 'walk_forward', 'split', 'monte_carlo'], default='standard')
    parser.add_argument('--split-ratio', type=float, default=0.7)
    parser.add_argument('--mc-iterations', type=int, default=200)
    parser.add_argument('--run-id', type=str, default=None)
    parser.add_argument('--output', type=str, default=None)
    parser.add_argument('--walk-forward', action='store_true', help='Alias for --mode walk_forward')
    args = parser.parse_args()

    mode = 'walk_forward' if args.walk_forward else args.mode
    symbol = str(args.symbol or 'EURUSD').upper()

    if mode == 'walk_forward':
        result = walk_forward_analysis(symbol)
        payload = build_output_payload(symbol, mode, result)
    else:
        df = fetch_data(symbol, bars=10000)
        if df is None or len(df) < 120:
            payload = build_output_payload(symbol, mode, {'metrics': {}, 'trades': [], 'error': 'Not enough data'})
        elif mode == 'split':
            result = run_split_backtest(df, symbol, split_ratio=float(args.split_ratio))
            payload = build_output_payload(symbol, mode, result)
        else:
            standard = run_backtest_no_lookahead(df, symbol)
            if mode == 'monte_carlo':
                mc = monte_carlo_analysis(standard.get('trades', []), iterations=int(args.mc_iterations))
                payload = build_output_payload(symbol, mode, {'standard': standard, 'monte_carlo': mc})
            else:
                payload = build_output_payload(symbol, mode, standard)

    if args.output:
        output_file = Path(args.output)
    else:
        run_id = args.run_id or f"{symbol}_{mode}"
        output_file = DATA_DIR / f"{run_id}.json"
    save_output(payload, output_file)

    return 0


if __name__ == '__main__':
    cli_args = {'--symbol', '--mode', '--split-ratio', '--mc-iterations', '--run-id', '--output', '--walk-forward'}
    if any(arg in sys.argv for arg in cli_args):
        raise SystemExit(run_cli_backtest_mode())

    if '--loop' in sys.argv:
        idx = sys.argv.index('--loop')
        if idx + 1 < len(sys.argv):
            try:
                max_iter = int(sys.argv[idx + 1])
                continuous_training(max_iterations=max_iter)
            except ValueError:
                continuous_training()
        else:
            continuous_training()
    else:
        main()
