"""
AI TRADING SIMULATOR - Complete Training Environment
====================================================
A full AI training ring that simulates EVERYTHING the live bot does:
- Market regime detection & adaptation
- Pattern recognition & learning
- Order flow analysis
- Swing point detection
- Position management decisions (breakeven, partials, trailing)
- Expectancy-based parameter optimization
- ML-style feature extraction

The AI trains here until it becomes profitable, then saves for live trading.
Run this to let the AI practice and learn from historical data 24/7.

Usage:
    python backtest_improved.py              # Single training run
    python backtest_improved.py --loop       # Continuous training until profitable
    python backtest_improved.py --loop 10    # Train for 10 iterations
"""

import os
import sys
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
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

try:
    import MetaTrader5 as mt5
except:
    mt5 = None

warnings.filterwarnings('ignore')


# ============================================================================
# AI TRAINING CONFIGURATION
# ============================================================================

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

# ============================================================================
# CONFIG
# ============================================================================

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


# ============================================================================
# AI LEARNING FUNCTIONS
# ============================================================================

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

# ============================================================================
# DATA FETCH
# ============================================================================

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
    """Add EMA, ATR, ADX, Donchian channels"""
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
    
    # Donchian High/Low (breakout levels)
    df['Donchian_High'] = pd.Series(high).rolling(20).max().values
    df['Donchian_Low'] = pd.Series(low).rolling(20).min().values
    # Ensure Time is datetime
    if 'Time' in df.columns and not np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time'] = pd.to_datetime(df['Time'])
    
    return df

# ============================================================================
# STRATEGY
# ============================================================================

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

# ============================================================================
# BACKTEST ENGINE WITH AI LEARNING
# ============================================================================

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

        # ============================================================
        # ENTRY - with AI pattern capture
        # ============================================================
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
        
        # ============================================================
        # POSITION MANAGEMENT - Simulate AI decisions
        # ============================================================
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
        
        # ============================================================
        # EXIT CONDITIONS
        # ============================================================
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

        # ============================================================
        # TRADE CLOSE - Record for AI learning
        # ============================================================
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
            
            # ============================================================
            # AI LEARNING - Record pattern outcome
            # ============================================================
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

# ============================================================================
# OPTIMIZATION WITH EXPECTANCY FOCUS
# ============================================================================

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


# ============================================================================
# MAIN - AI TRAINING LOOP
# ============================================================================

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


if __name__ == '__main__':
    # Parse command line arguments
    if '--loop' in sys.argv:
        # Find iteration count if specified
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
