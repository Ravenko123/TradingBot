# live trading bot — ICT / SMC strategy engine
# Documentation: Automatizovaný obchodný systém s Python + MetaTrader 5
# State machine: IDLE → SCANNING → ZONING → MONITORING → EXECUTION → MANAGEMENT

import argparse
import csv
import json
import math
import os
import sys
import time
import threading
from enum import Enum
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import MetaTrader5 as mt5

# .env support (MT5_LOGIN, TELEGRAM_BOT_TOKEN, RISK_PER_TRADE, …)
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except ImportError:
    pass  # python-dotenv optional; falls back to json configs

from strategy import get_instrument_settings
from telegram_bot import send_telegram_message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

# SQLite database (trades, order_blocks, logs)
from db import (
    init_trading_db, insert_trade, close_trade, get_daily_pnl,
    insert_order_block, mitigate_order_block, get_active_order_blocks,
    log_event, get_trades,
)


# ─── State Machine ──────────────────────────────────────────────
class BotState(Enum):
    """Trading engine finite-state machine (6 states per documentation)."""
    IDLE       = "IDLE"        # Waiting for session / outside kill-zone
    SCANNING   = "SCANNING"    # Analyzing market data & indicators
    ZONING     = "ZONING"      # Mapping OB / FVG / BOS zones
    MONITORING = "MONITORING"  # Watching for pullback into OB zone
    EXECUTION  = "EXECUTION"   # Sending order to MT5
    MANAGEMENT = "MANAGEMENT"  # Managing open position (BE / trail / close)


# Global state tracker
_current_state = BotState.IDLE


def set_state(new_state: BotState):
    """Transition the bot FSM and log the change."""
    global _current_state
    if new_state != _current_state:
        _current_state = new_state


def get_state() -> BotState:
    return _current_state


def _configure_safe_console_output():
    """Avoid crashes when terminal encoding cannot print emoji/unicode symbols."""
    for stream_name in ("stdout", "stderr"):
        stream = getattr(sys, stream_name, None)
        if stream is None:
            continue
        try:
            stream.reconfigure(errors='replace')
        except Exception:
            pass


_configure_safe_console_output()

# Optional AI layers (bot must still run without these files)
try:
    from ai_brain import get_brain, TradingIntelligenceBrain
except ImportError:
    class TradingIntelligenceBrain:
        pass

    class _FallbackBrain:
        def initialize_symbol(self, *args, **kwargs):
            return None
        def process_market_data(self, *args, **kwargs):
            return None
        def manage_open_position(self, *args, **kwargs):
            return None
        def on_trade_opened(self, *args, **kwargs):
            return None
        def on_trade_closed(self, *args, **kwargs):
            return None
        def get_daily_summary(self):
            return "AI brain unavailable"
        def get_symbol_report(self, symbol):
            return f"No AI report for {symbol}"
        def cleanup(self):
            return None

    def get_brain():
        print("[WARN] ai_brain.py not found - running with fallback brain")
        return _FallbackBrain()

try:
    from advanced_ai_brain import get_advanced_brain, TradeOutcome
except ImportError:
    class TradeOutcome:
        WIN = "WIN"
        LOSS = "LOSS"
        BREAKEVEN = "BREAKEVEN"

    class _FallbackAdvancedBrain:
        def __init__(self):
            self.patterns = type('Patterns', (), {'patterns': {}})()
            self.profiler = type('Profiler', (), {'profiles': {}})()
            self.market_structure = type('MS', (), {
                'order_flow': type('OF', (), {'history': []})()
            })()
        def analyze_signal(self, symbol, df_ai, signal, bid, ask, spread):
            if not signal:
                return None
            signal.setdefault('confidence', 0.6)
            signal.setdefault('ai_approved', True)
            signal.setdefault('reason', 'fallback_advanced_ai')
            return signal
        def update_after_trade(self, *args, **kwargs):
            return None

    _ADVANCED_FALLBACK_INSTANCE = _FallbackAdvancedBrain()

    def get_advanced_brain():
        return _ADVANCED_FALLBACK_INSTANCE

# Import Neural AI (real machine learning)
try:
    from neural_ai import get_neural_ai, NeuralTradingAI
    NEURAL_AI_AVAILABLE = True
except ImportError:
    NEURAL_AI_AVAILABLE = False
    print("[WARN] Neural AI not available - install sklearn for ML features")

# config

# Symbols to trade - MT5 provides real tick size, spread, digits
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'BTCUSD', 'NAS100']

# Broker uses '+' suffix for these symbols
BROKER_SUFFIX = {'EURUSD': '+', 'GBPUSD': '+', 'USDJPY': '+', 'GBPJPY': '+', 'XAUUSD': '+'}

# Strategy constants (same as backtested)
# Session kill zones: London 07-10 UTC, New York 13-16 UTC
SESSION_LONDON_START = 7
SESSION_LONDON_END   = 10
SESSION_NY_START     = 13
SESSION_NY_END       = 16
ATR_PERIOD = 14
ADX_PERIOD = 14
ADX_THRESHOLD = 25       # Doc: ADX < 25 = consolidation → skip
LOOKBACK_BARS = 500      # enough bars for indicator calculation
TP_RR_RATIO = 2.0        # Take Profit at 2:1 risk-reward (TP = 2× SL distance)

# ICT / SMC constants (ported from validated backtest engine)
BOS_VALIDITY = 50        # Order Block zone valid for 50 bars
OB_SCAN = 20             # Look back up to 20 bars for OB candle
SWING_LOOKBACK = 5       # Swing-point detection window
MIN_CONFLUENCE = 3       # Minimum confluence score to enter
COOLDOWN_BARS = 3        # Minimum bars between signals

# Safety mechanisms — circuit breakers
MAX_DAILY_DRAWDOWN = float(os.getenv('MAX_DAILY_DRAWDOWN', '5.0'))   # 5 % daily max loss → deactivate
MAX_MARGIN_USAGE   = float(os.getenv('MAX_MARGIN_USAGE', '20.0'))    # Block orders if > 20 % margin used

# Runtime config persistence
CONFIG_FILE = Path(__file__).parent / 'runtime_config.json'
DEFAULT_RISK = float(os.getenv('RISK_PER_TRADE', '2.0'))  # Doc: 2 % per trade

# Track last signal to avoid spam
last_signals = {}

# Track open positions to detect closures
tracked_positions = {}

# Order retry settings
MAX_ORDER_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Trade logging (only closed trades)
TRADE_LOG_FILE = 'liverun/live_trades.csv'
RUNTIME_STATUS_FILE = Path(__file__).parent / 'liverun' / 'runtime_status.json'

# Learned parameters (auto-adjusted by optimizer)
LEARNED_PARAMS_FILE = 'liverun/learned_params.json'

# Optimizer settings
OPTIMIZER_INTERVAL = 300  # Run optimizer every 5 minutes
MIN_TRADES_FOR_LEARNING = 5  # Min closed trades before adjusting params

# param optimizer

class ParameterOptimizer:
    """Analyzes closed trades and auto-adjusts strategy parameters in real-time."""
    
    def __init__(self):
        self.learned = self.load_learned_params()
        self.last_optimize = time.time()
    
    def load_learned_params(self) -> dict:
        """Load previously learned parameter improvements."""
        if Path(LEARNED_PARAMS_FILE).exists():
            try:
                return json.loads(Path(LEARNED_PARAMS_FILE).read_text())
            except:
                pass
        return {}
    
    def save_learned_params(self):
        """Persist learned parameter improvements."""
        try:
            Path(LEARNED_PARAMS_FILE).parent.mkdir(parents=True, exist_ok=True)
            Path(LEARNED_PARAMS_FILE).write_text(json.dumps(self.learned, indent=2))
        except:
            pass
    
    def analyze_trades(self, symbol: str, baseline_params: dict) -> dict | None:
        """Analyze recent closed trades for a symbol and suggest parameter tweaks."""
        if not Path(TRADE_LOG_FILE).exists():
            return None
        
        try:
            df = pd.read_csv(TRADE_LOG_FILE)
        except:
            return None
        
        # Get closed trades for this symbol (last 30 trades max)
        closed = df[(df['symbol'] == symbol) & (df['status'] == 'CLOSED')].tail(30)
        
        if len(closed) < MIN_TRADES_FOR_LEARNING:
            return None  # Not enough data yet
        
        # Calculate metrics
        wins = len(closed[closed['profit'].astype(float) > 0])
        losses = len(closed[closed['profit'].astype(float) <= 0])
        win_rate = wins / len(closed) if len(closed) > 0 else 0
        total_profit = closed['profit'].astype(float).sum()
        avg_profit = total_profit / len(closed) if len(closed) > 0 else 0
        max_loss = closed['profit'].astype(float).min()
        profit_factor = (closed[closed['profit'].astype(float) > 0]['profit'].astype(float).sum() / 
                        abs(closed[closed['profit'].astype(float) <= 0]['profit'].astype(float).sum())) if losses > 0 else 99
        
        # Suggested adjustments based on performance
        adjustments = {}
        adx_threshold = float(baseline_params.get('ADX', 20))
        atr_mult = float(baseline_params.get('ATR_Mult', 1.5))
        ema_fast = int(baseline_params.get('EMA_Fast', 10))
        ema_slow = int(baseline_params.get('EMA_Slow', 50))
        
        # Heuristic 1: If win rate is very low, increase ADX threshold (stricter filter)
        if win_rate < 0.35:
            adx_threshold = min(adx_threshold + 2, 30)
            adjustments['ADX'] = adx_threshold
        # Heuristic 2: If win rate is high, we could relax slightly (more trades)
        elif win_rate > 0.65:
            adx_threshold = max(adx_threshold - 1, 15)
            adjustments['ADX'] = adx_threshold
        
        # Heuristic 3: If profit factor is very low, reduce ATR multiplier (tighter SL/TP)
        if profit_factor < 1.0:
            atr_mult = max(atr_mult - 0.1, 0.8)
            adjustments['ATR_Mult'] = round(atr_mult, 1)
        # Heuristic 4: If profit factor is high, we could widen a bit (more profit per trade)
        elif profit_factor > 2.5:
            atr_mult = min(atr_mult + 0.1, 2.5)
            adjustments['ATR_Mult'] = round(atr_mult, 1)
        
        # Heuristic 5: If max loss is severe, tighten ATR mult further
        if max_loss < -50:
            atr_mult = max(atr_mult - 0.15, 0.7)
            adjustments['ATR_Mult'] = round(atr_mult, 1)
        
        return {
            'symbol': symbol,
            'trades': len(closed),
            'win_rate': round(win_rate, 3),
            'profit_factor': round(profit_factor, 2),
            'total_profit': round(total_profit, 2),
            'avg_profit': round(avg_profit, 2),
            'max_loss': round(max_loss, 2),
            'adjustments': adjustments
        }
    
    def optimize_all(self, baseline: dict) -> dict:
        """Analyze all symbols and apply learned adjustments."""
        results = {}
        updated = False
        
        for symbol in SYMBOLS:
            analysis = self.analyze_trades(symbol, baseline.get(symbol, {}))
            if analysis and analysis['adjustments']:
                results[symbol] = analysis
                # Apply adjustments to learned params
                if symbol not in self.learned:
                    self.learned[symbol] = baseline.get(symbol, {}).copy()
                for key, val in analysis['adjustments'].items():
                    self.learned[symbol][key] = val
                updated = True
        
        if updated:
            self.save_learned_params()
            print(f"\n[🤖 LEARNING] Updated {len(results)} symbols based on trade analysis")
            for sym, analysis in results.items():
                print(f"  {sym}: {analysis['trades']} trades, {analysis['win_rate']*100:.0f}% win, PF {analysis['profit_factor']}")
        
        return results
    
    def get_active_params(self, symbol: str, baseline: dict) -> dict:
        """Get current params: learned if available, else baseline."""
        if symbol in self.learned:
            return self.learned[symbol]
        return baseline.get(symbol, {})


optimizer = ParameterOptimizer()

# Initialize AI Brain
brain = get_brain()


def update_runtime_status(**fields):
    """Persist lightweight runtime heartbeat for dashboard/API diagnostics."""
    try:
        payload = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            **fields,
        }
        RUNTIME_STATUS_FILE.parent.mkdir(parents=True, exist_ok=True)
        RUNTIME_STATUS_FILE.write_text(json.dumps(payload, indent=2))
    except Exception:
        pass

# trade logging

def log_trade(trade_data: dict):
    """Log trade to CSV file for live testing records."""
    file_exists = Path(TRADE_LOG_FILE).exists()
    
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        fieldnames = ['timestamp', 'symbol', 'direction', 'entry_price', 'stop_loss', 'take_profit',
                     'lot_size', 'risk_percent', 'status', 'exit_time', 'exit_price', 'profit']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade_data)


# mt5 connection

def init_mt5() -> bool:
    """Initialize MT5 connection."""
    if not mt5.initialize():
        print(f"[!] MT5 init failed: {mt5.last_error()}")
        update_runtime_status(state='error', message=f"MT5 init failed: {mt5.last_error()}")
        return False
    update_runtime_status(state='mt5_connected', message='MT5 initialized successfully')
    return True


def check_mt5_connection() -> bool:
    """Check if MT5 is still connected."""
    try:
        account_info = mt5.account_info()
        if account_info is None:
            return False
        return True
    except Exception:
        return False


def reconnect_mt5() -> bool:
    """Attempt to reconnect to MT5."""
    print("[!] MT5 connection lost. Attempting to reconnect...")
    try:
        mt5.shutdown()
        time.sleep(2)
        if mt5.initialize():
            account_info = mt5.account_info()
            if account_info:
                print(f"[✓] MT5 reconnected: Account {account_info.login}")
                return True
    except Exception as e:
        print(f"[!] Reconnection failed: {e}")
    return False


def shutdown_mt5():
    """Shutdown MT5 connection."""
    mt5.shutdown()


def get_broker_symbol(symbol: str) -> str:
    """Get the broker-specific symbol name."""
    suffix = BROKER_SUFFIX.get(symbol, '')
    return f"{symbol}{suffix}"


def get_symbol_info(symbol: str) -> dict | None:
    """Get real symbol info from MT5 (tick size, digits, spread, etc.)."""
    # Check connection first
    if not check_mt5_connection():
        if not reconnect_mt5():
            return None
    
    broker_sym = get_broker_symbol(symbol)
    info = mt5.symbol_info(broker_sym)
    if info is None:
        return None
    return {
        'symbol': symbol,
        'broker_symbol': broker_sym,
        'tick_size': info.trade_tick_size,
        'tick_value': info.trade_tick_value,
        'digits': info.digits,
        'spread': info.spread,
        'bid': info.bid,
        'ask': info.ask,
        'volume_min': info.volume_min,
        'volume_max': info.volume_max,
        'volume_step': info.volume_step,
    }


# position mgmt

def get_open_positions() -> list:
    """Get all open positions from MT5."""
    positions = mt5.positions_get()
    if positions is None:
        return []
    return list(positions)


def get_position_for_symbol(symbol: str) -> dict | None:
    """Check if we have an open position for this symbol."""
    broker_sym = get_broker_symbol(symbol)
    positions = mt5.positions_get(symbol=broker_sym)
    if positions is None or len(positions) == 0:
        return None
    
    pos = positions[0]
    return {
        'ticket': pos.ticket,
        'symbol': symbol,
        'broker_symbol': broker_sym,
        'direction': 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL',
        'volume': pos.volume,
        'open_price': pos.price_open,
        'current_price': pos.price_current,
        'sl': pos.sl,
        'tp': pos.tp,
        'profit': pos.profit,
        'open_time': datetime.fromtimestamp(pos.time, tz=timezone.utc),
    }


def calculate_lot_size(symbol: str, risk_percent: float, stop_distance: float, sym_info: dict) -> float:
    """Calculate lot size based on risk percentage and stop distance."""
    account = mt5.account_info()
    if account is None:
        return sym_info['volume_min']
    
    balance = account.balance
    risk_amount = balance * (risk_percent / 100.0)
    
    # Get tick value (profit per 1 lot per 1 tick movement)
    tick_value = sym_info['tick_value']
    tick_size = sym_info['tick_size']
    
    if tick_value <= 0 or tick_size <= 0 or stop_distance <= 0:
        return sym_info['volume_min']
    
    # Calculate how many ticks in our stop distance
    ticks_in_stop = stop_distance / tick_size
    
    # Calculate lot size: risk_amount / (ticks * tick_value)
    lot_size = risk_amount / (ticks_in_stop * tick_value)
    
    # Round to volume step and clamp to min/max
    vol_step = sym_info['volume_step']
    lot_size = round(lot_size / vol_step) * vol_step
    lot_size = max(sym_info['volume_min'], min(lot_size, sym_info['volume_max']))
    
    return round(lot_size, 2)


def place_order(signal: dict, sym_info: dict, lot_size: float) -> dict:
    """Place a market order with SL. Returns result dict."""
    broker_sym = signal['broker_symbol']
    
    # Determine order type
    if signal['direction'] == 'BUY':
        order_type = mt5.ORDER_TYPE_BUY
        price = sym_info['ask']  # fresh ask
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = sym_info['bid']  # fresh bid
    
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': broker_sym,
        'volume': lot_size,
        'type': order_type,
        'price': price,
        'sl': signal['stop'],
        'tp': signal['tp'],
        'deviation': 20,  # slippage in points
        'magic': 123456,  # EA magic number
        'comment': 'TrendBot',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    return result


def open_position_with_retry(signal: dict, sym_info: dict, risk_percent: float) -> bool:
    """Attempt to open position with retry logic. Returns True if successful."""
    stop_distance = abs(signal['entry'] - signal['stop'])
    lot_size = calculate_lot_size(signal['symbol'], risk_percent, stop_distance, sym_info)
    
    for attempt in range(1, MAX_ORDER_RETRIES + 1):
        # Refresh symbol info for latest prices
        fresh_info = get_symbol_info(signal['symbol'])
        if fresh_info:
            sym_info = fresh_info
        
        result = place_order(signal, sym_info, lot_size)
        
        if result is None:
            print(f"[!] Order send failed: {mt5.last_error()}")
            time.sleep(RETRY_DELAY)
            continue
        
        if result.retcode == mt5.TRADE_RETCODE_DONE:
            print(f"[✓] Order filled: {signal['direction']} {lot_size} {signal['symbol']} @ {result.price}")
            signal['lot_size'] = lot_size  # Store for logging
            return True
        
        # Check if it's a retriable error
        retriable_codes = [
            mt5.TRADE_RETCODE_REQUOTE,
            mt5.TRADE_RETCODE_PRICE_CHANGED,
            mt5.TRADE_RETCODE_PRICE_OFF,
            mt5.TRADE_RETCODE_TIMEOUT,
            mt5.TRADE_RETCODE_CONNECTION,
        ]
        
        if result.retcode in retriable_codes and attempt < MAX_ORDER_RETRIES:
            print(f"[!] Order attempt {attempt} failed (code {result.retcode}), retrying...")
            time.sleep(RETRY_DELAY)
        else:
            print(f"[!] Order failed: code={result.retcode}, comment={result.comment}")
            return False
    
    return False


def close_position(position: dict) -> bool:
    """Close an existing position."""
    broker_sym = position['broker_symbol']
    ticket = position['ticket']
    volume = position['volume']
    
    # Determine close direction (opposite of position)
    if position['direction'] == 'BUY':
        order_type = mt5.ORDER_TYPE_SELL
        price = mt5.symbol_info_tick(broker_sym).bid
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price = mt5.symbol_info_tick(broker_sym).ask
    
    request = {
        'action': mt5.TRADE_ACTION_DEAL,
        'symbol': broker_sym,
        'volume': volume,
        'type': order_type,
        'position': ticket,
        'price': price,
        'deviation': 20,
        'magic': 123456,
        'comment': 'TrendBot Close',
        'type_time': mt5.ORDER_TIME_GTC,
        'type_filling': mt5.ORDER_FILLING_IOC,
    }
    
    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        print(f"[✓] Position closed: {position['direction']} {volume} {position['symbol']}")
        return True
    
    print(f"[!] Close failed: {result.retcode if result else mt5.last_error()}")
    return False


def modify_position_sl_tp(position: dict, new_sl: float | None = None, new_tp: float | None = None) -> bool:
    """Modify SL/TP for an open position. Returns True if successful."""
    if new_sl is None and new_tp is None:
        return False

    broker_sym = position['broker_symbol']
    digits = get_symbol_info(position['symbol']).get('digits', 5) if get_symbol_info(position['symbol']) else 5
    request = {
        'action': mt5.TRADE_ACTION_SLTP,
        'symbol': broker_sym,
        'position': position['ticket'],
        'sl': round(new_sl, digits) if new_sl is not None else position.get('sl', 0),
        'tp': round(new_tp, digits) if new_tp is not None else position.get('tp', 0),
        'magic': 123456,
        'comment': 'TrendBot SLTP'
    }

    result = mt5.order_send(request)
    if result and result.retcode == mt5.TRADE_RETCODE_DONE:
        return True
    print(f"[!] SL/TP modify failed: {result.retcode if result else mt5.last_error()}")
    return False


def fetch_live_candles(symbol: str, timeframe=mt5.TIMEFRAME_M15, bars: int = LOOKBACK_BARS) -> pd.DataFrame | None:
    """Fetch live candles from MT5 (M15 timeframe per documentation)."""
    # Check connection first
    if not check_mt5_connection():
        if not reconnect_mt5():
            return None
    
    broker_sym = get_broker_symbol(symbol)
    
    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(broker_sym, True):
        # Try once more after small delay
        time.sleep(0.5)
        if not mt5.symbol_select(broker_sym, True):
            print(f"[!] Failed to select {broker_sym}")
            return None
    
    rates = mt5.copy_rates_from_pos(broker_sym, timeframe, 0, bars)
    if rates is None or len(rates) < 100:
        return None
    
    df = pd.DataFrame(rates)
    df['Time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['Time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    return df


# multi-tf context

def get_htf_bias(symbol: str, params: dict, timeframe = mt5.TIMEFRAME_H1) -> str:
    """Determine higher timeframe bias using EMAs and ADX.
    Returns one of: 'BULL', 'BEAR', 'FLAT', 'UNKNOWN'
    """
    try:
        df_htf = fetch_live_candles(symbol, timeframe=timeframe, bars=400)
        if df_htf is None or len(df_htf) < 100:
            return 'UNKNOWN'

        fast = int(params.get('EMA_Fast', 10))
        slow = int(params.get('EMA_Slow', 50))
        adx_th = float(params.get('ADX', 20))

        df_i = add_indicators(df_htf, fast, slow).fillna(0)
        i = len(df_i) - 1
        ema_f = float(df_i['EMA_Fast'].iat[i])
        ema_s = float(df_i['EMA_Slow'].iat[i])
        adx = float(df_i['ADX'].iat[i])

        # Slightly relaxed ADX threshold for HTF to get bias more often
        htf_adx_th = max(15.0, adx_th * 0.8)

        if not np.isfinite(adx):
            return 'UNKNOWN'

        if adx > htf_adx_th:
            if ema_f > ema_s:
                return 'BULL'
            elif ema_f < ema_s:
                return 'BEAR'
        return 'FLAT'
    except Exception:
        return 'UNKNOWN'


# position sizing

def adjust_risk_percent(base_risk: float, signal: dict, htf_bias: str) -> float:
    """Scale risk % by AI confidence, market phase, and HTF alignment.
    Applies safety clamps to keep risk within sane bounds.
    """
    # Confidence multiplier
    confidence = float(signal.get('confidence', 0.6))
    if confidence < 0.6:
        m_conf = 0.8
    elif confidence < 0.7:
        m_conf = 0.9
    elif confidence < 0.8:
        m_conf = 1.0
    elif confidence < 0.9:
        m_conf = 1.2
    else:
        m_conf = 1.35

    # Market phase multiplier
    phase = str(signal.get('market_phase', '') or '').upper()
    phase_map = {
        'TRENDING_STRONG': 1.25,
        'TRENDING_WEAK': 1.1,
        'BREAKOUT': 1.2,
        'RANGING': 0.9,
        'REVERSAL': 0.8,
        'QUIET': 0.7,
    }
    m_phase = phase_map.get(phase, 1.0)

    # HTF alignment multiplier
    direction = signal.get('direction', '')
    aligned = (htf_bias == 'BULL' and direction == 'BUY') or (htf_bias == 'BEAR' and direction == 'SELL')
    if htf_bias in ('UNKNOWN', 'FLAT'):
        m_htf = 0.9
    elif aligned:
        m_htf = 1.0
    else:
        m_htf = 0.6

    # Combine with clamps
    scaled = base_risk * m_conf * m_phase * m_htf
    scaled = max(0.2, min(3.0, round(scaled, 2)))  # keep between 0.2% and 3%
    return scaled


# indicators

def add_indicators(df: pd.DataFrame, fast_ema: int, slow_ema: int) -> pd.DataFrame:
    """Add EMA, ATR, ADX indicators."""
    df = df.copy()
    close = df['Close'].astype(float).values
    high = df['High'].astype(float).values
    low = df['Low'].astype(float).values

    # EMAs
    close_series = pd.Series(close)
    df['EMA_Fast'] = close_series.ewm(span=fast_ema, adjust=False).mean().values
    df['EMA_Slow'] = close_series.ewm(span=slow_ema, adjust=False).mean().values

    # ATR
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    tr = np.max(np.stack([high_low, high_close, low_close]), axis=0)
    df['ATR'] = pd.Series(tr).rolling(ATR_PERIOD).mean().values

    # ADX
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    atr = df['ATR'].to_numpy()
    atr_safe = np.where(atr > 0, atr, np.nan)
    di_plus = 100 * (pd.Series(plus_dm).rolling(ADX_PERIOD).mean().values / atr_safe)
    di_minus = 100 * (pd.Series(minus_dm).rolling(ADX_PERIOD).mean().values / atr_safe)
    dx = 100 * (np.abs(di_plus - di_minus) / np.where((di_plus + di_minus) != 0, (di_plus + di_minus), np.nan))
    df['ADX'] = pd.Series(dx).rolling(ADX_PERIOD).mean().values

    return df


def _find_swing_points(highs, lows, lookback=SWING_LOOKBACK):
    """Detect swing highs and swing lows using lookback window (no future leak)."""
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    for i in range(lookback, n - lookback):
        wh = highs[i - lookback: i + lookback + 1]
        wl = lows[i - lookback: i + lookback + 1]
        if highs[i] == np.max(wh):
            swing_highs[i] = highs[i]
        if lows[i] == np.min(wl):
            swing_lows[i] = lows[i]
    return swing_highs, swing_lows


def in_session_kill_zone(hour: int) -> bool:
    """Return True if current hour falls inside London or NY kill zone."""
    return (SESSION_LONDON_START <= hour <= SESSION_LONDON_END) or \
           (SESSION_NY_START <= hour <= SESSION_NY_END)


def generate_signal(df: pd.DataFrame, params: dict, symbol: str, sym_info: dict) -> dict | None:
    """Generate ICT / SMC trading signal with confluence scoring.

    HARD requirements (all must be true):
      • EMA trend alignment (EMA 9 > EMA 21 for BUY, vice-versa)
      • ADX ≥ threshold (skip consolidation)
      • Valid Order Block zone from a BOS event
      • Price pulls back into the OB zone
      • Confirmation candle (close in trade direction)

    CONFLUENCE SCORING (soft — need total ≥ MIN_CONFLUENCE):
      +1  Displacement BOS (candle body ≥ 150 % of avg last 10)
      +1  FVG present in the impulse
      +1  Liquidity sweep before the move
      +1  In session kill zone (London 07-10 / NY 13-16)
      +1  Premium/Discount alignment
      +1  Rejection wick on entry candle
      +1  Strong OB candle (body > avg)
    """
    set_state(BotState.SCANNING)

    if not params:
        return None

    fast = int(params.get('EMA_Fast', 9))
    slow = int(params.get('EMA_Slow', 21))
    adx_th = float(params.get('ADX', ADX_THRESHOLD))
    atr_mult = float(params.get('ATR_Mult', 1.5))

    # Brain param adaptation (AI layer)
    try:
        brain_analysis = brain.process_market_data(symbol, df, sym_info, params)
        adapted_params = brain_analysis.get('params', {}) if brain_analysis else {}
        if adapted_params:
            adx_th = float(adapted_params.get('ADX', adx_th))
            atr_mult = float(adapted_params.get('ATR_Mult', atr_mult))
    except Exception:
        pass

    df = add_indicators(df, fast, slow).fillna(0)
    n = len(df)
    if n < max(slow, 60):
        return None

    closes = df['Close'].to_numpy().astype(float)
    opens  = df['Open'].to_numpy().astype(float)
    highs  = df['High'].to_numpy().astype(float)
    lows   = df['Low'].to_numpy().astype(float)
    ema_f_arr = df['EMA_Fast'].to_numpy().astype(float)
    ema_s_arr = df['EMA_Slow'].to_numpy().astype(float)
    adx_arr   = df['ADX'].to_numpy().astype(float)
    atr_arr   = df['ATR'].to_numpy().astype(float)
    times     = pd.to_datetime(df['Time'])
    hours     = times.dt.hour.to_numpy()

    # Pre-compute helpers
    swing_highs, swing_lows = _find_swing_points(highs, lows, SWING_LOOKBACK)
    bodies = np.abs(closes - opens)

    # Avg body of last 10 candles (per documentation: 150 % displacement)
    avg_body_10 = np.zeros(n)
    for j in range(10, n):
        avg_body_10[j] = np.mean(bodies[j - 10:j])

    # Avg body of last 20 for OB strength
    avg_body_20 = np.zeros(n)
    for j in range(20, n):
        avg_body_20[j] = np.mean(bodies[j - 20:j])

    set_state(BotState.ZONING)

    # ── Scan for BOS events and build OB zones (last ~100 bars) ──
    bull_obs = []  # {hi, lo, bar, score, strong_ob}
    bear_obs = []
    start = max(slow, 2 * SWING_LOOKBACK + 5, 42)
    scan_from = max(start, n - 120)  # Only scan recent bars for live signals

    for i in range(scan_from, n):
        if not np.isfinite(adx_arr[i]) or adx_arr[i] < adx_th:
            continue

        bullish_trend = ema_f_arr[i] > ema_s_arr[i]
        bearish_trend = ema_f_arr[i] < ema_s_arr[i]

        # Find most recent swing high / swing low
        recent_sh = np.nan
        recent_sl_val = np.nan
        for k in range(i - SWING_LOOKBACK - 1, max(start - 1, SWING_LOOKBACK) - 1, -1):
            if np.isnan(recent_sh) and not np.isnan(swing_highs[k]):
                recent_sh = swing_highs[k]
            if np.isnan(recent_sl_val) and not np.isnan(swing_lows[k]):
                recent_sl_val = swing_lows[k]
            if not np.isnan(recent_sh) and not np.isnan(recent_sl_val):
                break

        # ── Bullish BOS ──
        if bullish_trend and not np.isnan(recent_sh):
            for k in range(max(start, i - SWING_LOOKBACK), i):
                if closes[k] > recent_sh:
                    # Displacement: BOS candle body ≥ 150 % of avg last 10
                    bos_disp = bodies[k] > avg_body_10[k] * 1.5 if avg_body_10[k] > 0 else False

                    # Find OB: last bearish candle before the impulse
                    ob_hi, ob_lo, ob_strong = np.nan, np.nan, False
                    for ob_idx in range(k, max(k - OB_SCAN, 0), -1):
                        if closes[ob_idx] < opens[ob_idx]:
                            ob_hi = highs[ob_idx]
                            ob_lo = min(opens[ob_idx], closes[ob_idx])
                            ob_strong = bodies[ob_idx] > avg_body_20[ob_idx] * 0.7 if avg_body_20[ob_idx] > 0 else False
                            break
                    if np.isnan(ob_hi):
                        break

                    score = 0
                    if bos_disp:
                        score += 1
                    # FVG check
                    for f in range(max(start, k - 8), k + 1):
                        if f >= 2 and lows[f] > highs[f - 2]:
                            score += 1
                            break
                    # Liquidity sweep
                    if k >= 2 and lows[k - 1] < lows[k - 2] and closes[k - 1] > lows[k - 2]:
                        score += 1
                    if ob_strong:
                        score += 1

                    bull_obs.append({'hi': ob_hi, 'lo': ob_lo, 'bar': i, 'score': score})
                    if len(bull_obs) > 3:
                        bull_obs.pop(0)

                    # Store OB in database
                    try:
                        insert_order_block(symbol=symbol, price_high=ob_hi, price_low=ob_lo, direction='BULL')
                    except Exception:
                        pass
                    break

        # ── Bearish BOS ──
        if bearish_trend and not np.isnan(recent_sl_val):
            for k in range(max(start, i - SWING_LOOKBACK), i):
                if closes[k] < recent_sl_val:
                    bos_disp = bodies[k] > avg_body_10[k] * 1.5 if avg_body_10[k] > 0 else False

                    ob_hi, ob_lo, ob_strong = np.nan, np.nan, False
                    for ob_idx in range(k, max(k - OB_SCAN, 0), -1):
                        if closes[ob_idx] > opens[ob_idx]:
                            ob_lo = lows[ob_idx]
                            ob_hi = max(opens[ob_idx], closes[ob_idx])
                            ob_strong = bodies[ob_idx] > avg_body_20[ob_idx] * 0.7 if avg_body_20[ob_idx] > 0 else False
                            break
                    if np.isnan(ob_lo):
                        break

                    score = 0
                    if bos_disp:
                        score += 1
                    for f in range(max(start, k - 8), k + 1):
                        if f >= 2 and highs[f] < lows[f - 2]:
                            score += 1
                            break
                    if k >= 2 and highs[k - 1] > highs[k - 2] and closes[k - 1] < highs[k - 2]:
                        score += 1
                    if ob_strong:
                        score += 1

                    bear_obs.append({'hi': ob_hi, 'lo': ob_lo, 'bar': i, 'score': score})
                    if len(bear_obs) > 3:
                        bear_obs.pop(0)

                    try:
                        insert_order_block(symbol=symbol, price_high=ob_hi, price_low=ob_lo, direction='BEAR')
                    except Exception:
                        pass
                    break

    # Expire old OB zones
    bull_obs = [z for z in bull_obs if (n - 1 - z['bar']) <= BOS_VALIDITY]
    bear_obs = [z for z in bear_obs if (n - 1 - z['bar']) <= BOS_VALIDITY]

    set_state(BotState.MONITORING)

    # ── Check pullback entry on the LATEST bar ──
    i = n - 1
    bar_time = df['Time'].iat[i]
    hour = hours[i]
    adx_val = adx_arr[i]
    ema_f = ema_f_arr[i]
    ema_s = ema_s_arr[i]
    atr_val = atr_arr[i]

    # Use LIVE bid/ask from MT5
    bid = sym_info['bid']
    ask = sym_info['ask']
    spread_points = sym_info['spread']
    digits = sym_info['digits']

    # Session kill zone filter
    if not in_session_kill_zone(hour):
        set_state(BotState.IDLE)
        return None

    # ADX filter — doc: ADX < 25 = consolidation → skip
    if not np.isfinite(adx_val) or adx_val < adx_th:
        return None

    bullish_trend = ema_f > ema_s
    bearish_trend = ema_f < ema_s
    if not bullish_trend and not bearish_trend:
        return None

    # Entry-level confluence helpers
    range_lb = min(40, n - 1)
    range_high = np.max(highs[i - range_lb:i]) if range_lb > 0 else highs[i]
    range_low  = np.min(lows[i - range_lb:i]) if range_lb > 0 else lows[i]
    range_mid  = (range_high + range_low) / 2.0
    body_i = abs(closes[i] - opens[i])

    signal_result = None

    # ── BULLISH ENTRY ──
    if bullish_trend:
        for z_idx, z in enumerate(bull_obs):
            if (i - z['bar']) < 1:
                continue
            touched = lows[i] <= z['hi'] and closes[i] >= z['lo']
            bull_candle = closes[i] > opens[i]
            if not (touched and bull_candle):
                continue

            entry_score = z['score']
            if in_session_kill_zone(hour):
                entry_score += 1
            if closes[i] < range_mid:  # Discount zone
                entry_score += 1
            lower_wick = min(opens[i], closes[i]) - lows[i]
            if body_i > 0 and lower_wick > body_i * 0.3:
                entry_score += 1  # Rejection wick

            if entry_score >= MIN_CONFLUENCE:
                entry = ask
                stop = entry - (atr_mult * atr_val)
                tp = entry + (atr_mult * atr_val * TP_RR_RATIO)
                signal_result = {
                    'symbol': symbol,
                    'broker_symbol': sym_info['broker_symbol'],
                    'direction': 'BUY',
                    'entry': round(entry, digits),
                    'stop': round(stop, digits),
                    'tp': round(tp, digits),
                    'bid': bid, 'ask': ask,
                    'spread': spread_points,
                    'adx': round(adx_val, 2),
                    'atr': round(atr_val, digits),
                    'atr_mult': atr_mult,
                    'ema_fast': round(ema_f, digits),
                    'ema_slow': round(ema_s, digits),
                    'timestamp': bar_time,
                    'confluence_score': entry_score,
                    'params': f"ICT EMA{fast}/{slow} ADX>{adx_th} ATR×{atr_mult} Score:{entry_score}",
                }
                # Mark OB as mitigated
                bull_obs.pop(z_idx)
                break

    # ── BEARISH ENTRY ──
    if bearish_trend and signal_result is None:
        for z_idx, z in enumerate(bear_obs):
            if (i - z['bar']) < 1:
                continue
            touched = highs[i] >= z['lo'] and closes[i] <= z['hi']
            bear_candle = closes[i] < opens[i]
            if not (touched and bear_candle):
                continue

            entry_score = z['score']
            if in_session_kill_zone(hour):
                entry_score += 1
            if closes[i] > range_mid:  # Premium zone
                entry_score += 1
            upper_wick = highs[i] - max(opens[i], closes[i])
            if body_i > 0 and upper_wick > body_i * 0.3:
                entry_score += 1  # Rejection wick

            if entry_score >= MIN_CONFLUENCE:
                entry = bid
                stop = entry + (atr_mult * atr_val)
                tp = entry - (atr_mult * atr_val * TP_RR_RATIO)
                signal_result = {
                    'symbol': symbol,
                    'broker_symbol': sym_info['broker_symbol'],
                    'direction': 'SELL',
                    'entry': round(entry, digits),
                    'stop': round(stop, digits),
                    'tp': round(tp, digits),
                    'bid': bid, 'ask': ask,
                    'spread': spread_points,
                    'adx': round(adx_val, 2),
                    'atr': round(atr_val, digits),
                    'atr_mult': atr_mult,
                    'ema_fast': round(ema_f, digits),
                    'ema_slow': round(ema_s, digits),
                    'timestamp': bar_time,
                    'confluence_score': entry_score,
                    'params': f"ICT EMA{fast}/{slow} ADX>{adx_th} ATR×{atr_mult} Score:{entry_score}",
                }
                bear_obs.pop(z_idx)
                break

    return signal_result


# runtime config

def load_config() -> dict:
    """Load runtime config from file."""
    cfg = {
        'risk_percent': DEFAULT_RISK,
        'enabled_symbols': SYMBOLS.copy(),
    }
    if CONFIG_FILE.exists():
        try:
            data = json.loads(CONFIG_FILE.read_text())
            cfg.update(data)
        except Exception:
            pass
    cfg['risk_percent'] = max(0.01, min(100.0, float(cfg.get('risk_percent', DEFAULT_RISK))))
    cfg['enabled_symbols'] = [s for s in cfg.get('enabled_symbols', SYMBOLS) if s in SYMBOLS]
    return cfg


def save_config(cfg: dict):
    """Save runtime config to file."""
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


# telegram

class TelegramBot:
    def __init__(self):
        self.token = os.getenv('TELEGRAM_BOT_TOKEN', TELEGRAM_BOT_TOKEN)
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', TELEGRAM_CHAT_ID)
        self.offset = None

    def is_configured(self) -> bool:
        return (self.token not in (None, '', 'YOUR_BOT_TOKEN_HERE') and 
                self.chat_id not in (None, '', 'YOUR_CHAT_ID_HERE'))

    def poll_commands(self, cfg: dict) -> dict:
        """Poll Telegram for commands and update config."""
        if not self.is_configured():
            return cfg
        try:
            params = {'timeout': 0}
            if self.offset:
                params['offset'] = self.offset
            resp = requests.get(f"https://api.telegram.org/bot{self.token}/getUpdates", 
                              params=params, timeout=3)
            data = resp.json()
            if not data.get('ok'):
                return cfg
            
            changed = False
            for upd in data.get('result', []):
                self.offset = upd['update_id'] + 1
                msg = upd.get('message', {})
                if str(msg.get('chat', {}).get('id', '')) != str(self.chat_id):
                    continue
                
                text = (msg.get('text') or '').strip().lower()
                
                # /start or /help command
                if text in ['/start', '/help']:
                    help_msg = (
                        "🤖 Ultima Trading Bot\n"
                        "--------------------------------\n"
                        "📊 Trading Commands\n"
                        "/risk [0.01-100] - Set risk % per trade\n"
                        "/positions - View open positions\n"
                        "/status - Bot status & settings\n"
                        "/ping - Check bot connectivity\n\n"
                        "🧠 AI Intelligence\n"
                        "/brain - AI insights & daily summary\n"
                        "/brain <sym> - Symbol intelligence report\n"
                        "/learn - Show learned parameters\n\n"
                        "🔍 Debugging\n"
                        "/debug <symbol> - Detailed indicators\n"
                        "/why <symbol> - Quick verdict\n\n"
                        "🔧 Symbol Toggles\n"
                        "/eurusd /gbpusd /usdjpy /xauusd /gbpjpy /btcusd /nas100\n\n"
                        "📈 Scans every 30s with AI adaptation"
                    )
                    sent = send_telegram_message(help_msg, silent=False)
                    if not sent:
                        print("[!] Failed to send /help message to Telegram")
                
                # /debug <symbol> — detailed indicators and reason
                elif text.startswith('/debug '):
                    parts = text.split()
                    if len(parts) >= 2:
                        sym = parts[1].upper()
                        if sym in SYMBOLS:
                            sym_info = get_symbol_info(sym)
                            df = fetch_live_candles(sym)
                            params = get_instrument_settings(sym)
                            if not sym_info or df is None or params is None or len(df) < 100:
                                send_telegram_message(f"❌ Not enough data for {sym}")
                            else:
                                fast = int(params.get('EMA_Fast', 10))
                                slow = int(params.get('EMA_Slow', 50))
                                adx_th = float(params.get('ADX', 20))
                                atr_mult = float(params.get('ATR_Mult', 1.5))
                                df_i = add_indicators(df, fast, slow).fillna(0)
                                i = len(df_i) - 1
                                ema_f = float(df_i['EMA_Fast'].iat[i])
                                ema_s = float(df_i['EMA_Slow'].iat[i])
                                adx = float(df_i['ADX'].iat[i])
                                atr = float(df_i['ATR'].iat[i])
                                hour = int(df_i['Time'].iat[i].hour)
                                bid = sym_info['bid']; ask = sym_info['ask']
                                spread = sym_info['spread']
                                direction = 'BUY' if ema_f > ema_s else ('SELL' if ema_f < ema_s else 'FLAT')
                                reason = []
                                if not in_session_kill_zone(hour):
                                    reason.append(f"Outside kill zone ({hour} UTC)")
                                if not np.isfinite(adx) or adx <= adx_th:
                                    reason.append(f"ADX {adx:.1f} ≤ {adx_th}")
                                if direction == 'FLAT':
                                    reason.append("EMAs equal / no trend")
                                # Would we signal?
                                sig = generate_signal(df, params, sym, sym_info)
                                would = 'YES' if sig else 'NO'
                                because = 'OK' if would == 'YES' else (', '.join(reason) or 'No condition met')
                                msg_lines = [
                                    f"🔎 <b>DEBUG {sym}</b>",
                                    "━━━━━━━━━━━━━━━━",
                                    f"EMA Fast/Slow: {ema_f:.5f} / {ema_s:.5f}",
                                    f"ADX: {adx:.1f}  | ATR: {atr:.5f}",
                                    f"Bid/Ask: {bid} / {ask}  | Spread: {spread} pts",
                                    f"Direction: {direction}",
                                    f"Session OK: {'YES' if in_session_kill_zone(hour) else 'NO'}",
                                    f"Would Signal Now: {would}",
                                    f"Reason: {because}"
                                ]
                                send_telegram_message('\n'.join(msg_lines))
                        else:
                            send_telegram_message(f"❌ Unknown symbol: {sym}")
                    else:
                        send_telegram_message("Usage: /debug <symbol>  e.g., /debug xauusd")

                # /why <symbol> — one-line verdict
                elif text.startswith('/why '):
                    parts = text.split()
                    if len(parts) >= 2:
                        sym = parts[1].upper()
                        if sym in SYMBOLS:
                            sym_info = get_symbol_info(sym)
                            df = fetch_live_candles(sym)
                            params = get_instrument_settings(sym)
                            if not sym_info or df is None or params is None or len(df) < 100:
                                send_telegram_message(f"❌ {sym}: not enough data")
                            else:
                                fast = int(params.get('EMA_Fast', 10))
                                slow = int(params.get('EMA_Slow', 50))
                                adx_th = float(params.get('ADX', 20))
                                df_i = add_indicators(df, fast, slow).fillna(0)
                                i = len(df_i) - 1
                                ema_f = float(df_i['EMA_Fast'].iat[i])
                                ema_s = float(df_i['EMA_Slow'].iat[i])
                                adx = float(df_i['ADX'].iat[i])
                                hour = int(df_i['Time'].iat[i].hour)
                                reason = None
                                if not in_session_kill_zone(hour):
                                    reason = f"outside kill zone (UTC {hour})"
                                elif not np.isfinite(adx) or adx <= adx_th:
                                    reason = f"adx {adx:.1f} ≤ {adx_th}"
                                elif abs(ema_f - ema_s) < 1e-12:
                                    reason = "emas equal/no trend"
                                sig = generate_signal(df, params, sym, sym_info)
                                if sig and not reason:
                                    send_telegram_message(f"✅ {sym}: signal ready ({sig['direction']})")
                                else:
                                    send_telegram_message(f"⏸️ {sym}: no signal — {reason or 'no condition met'}")
                        else:
                            send_telegram_message(f"❌ Unknown symbol: {sym}")
                    else:
                        send_telegram_message("Usage: /why <symbol>  e.g., /why xauusd")
                
                # /learn command — see what bot has learned
                elif text == '/learn':
                    if not optimizer.learned:
                        send_telegram_message("📚 <b>No Learning Yet</b>\n\nBot learns from closed trades.\nNeed at least 5 closed trades per symbol.")
                    else:
                        lines = ["📚 <b>Bot Learned Parameters</b>\n━━━━━━━━━━━━━━━━"]
                        for sym, params_adj in optimizer.learned.items():
                            lines.append(f"\n<b>{sym}</b>:")
                            if 'ADX' in params_adj:
                                lines.append(f"   ADX threshold: {params_adj['ADX']}")
                            if 'ATR_Mult' in params_adj:
                                lines.append(f"   ATR multiplier: {params_adj['ATR_Mult']}")
                            if 'EMA_Fast' in params_adj:
                                lines.append(f"   EMA Fast: {params_adj['EMA_Fast']}")
                        lines.append(f"\n━━━━━━━━━━━━━━━━\n✅ Parameters auto-save to: learned_params.json")
                        send_telegram_message('\n'.join(lines))
                
                # /brain command — AI brain insights
                elif text == '/brain':
                    try:
                        summary = brain.get_daily_summary()
                        send_telegram_message(summary)
                    except Exception as e:
                        send_telegram_message(f"🧠 AI Brain initializing...\nRun bot for a while to collect data.")
                
                # /brain <symbol> — detailed symbol intelligence
                elif text.startswith('/brain '):
                    parts = text.split()
                    if len(parts) >= 2:
                        sym = parts[1].upper()
                        if sym in SYMBOLS:
                            try:
                                report = brain.get_symbol_report(sym)
                                send_telegram_message(report)
                            except Exception as e:
                                send_telegram_message(f"🧠 No data for {sym} yet.\nBot needs to run and collect market data.")
                        else:
                            send_telegram_message(f"❌ Unknown symbol: {sym}")
                    else:
                        send_telegram_message("Usage: /brain <symbol>  e.g., /brain xauusd")
                
                # /neural command — Neural AI (Machine Learning) status
                elif text == '/neural':
                    if NEURAL_AI_AVAILABLE:
                        try:
                            neural_ai = get_neural_ai()
                            stats = neural_ai.get_stats()
                            
                            lines = ["🧠 <b>NEURAL TRADING AI</b>\n━━━━━━━━━━━━━━━━"]
                            lines.append(f"\n📊 <b>Training Stats:</b>")
                            lines.append(f"   • Trades Learned: {stats['total_trades']}")
                            lines.append(f"   • Win Rate: {stats['win_rate']:.1%}")
                            lines.append(f"   • Total Profit: ${stats['total_profit']:.2f}")
                            lines.append(f"   • Memory Size: {stats['memory_size']} experiences")
                            
                            lines.append(f"\n🎯 <b>Model Status:</b>")
                            lines.append(f"   • Exploration Rate (ε): {stats['epsilon']:.3f}")
                            lines.append(f"   • Models Fitted: {'✅' if stats['models_fitted'] else '❌'}")
                            
                            if stats['symbol_stats']:
                                lines.append(f"\n📈 <b>Per-Symbol Performance:</b>")
                                for sym, sym_stats in list(stats['symbol_stats'].items())[:5]:
                                    total = sym_stats['wins'] + sym_stats['losses']
                                    wr = sym_stats['wins'] / total if total > 0 else 0
                                    lines.append(f"   • {sym}: {wr:.0%} WR, ${sym_stats['profit']:.0f}")
                            
                            lines.append(f"\n━━━━━━━━━━━━━━━━\n🤖 Real ML is ACTIVE!")
                            
                            send_telegram_message('\n'.join(lines))
                        except Exception as e:
                            send_telegram_message(f"🧠 Neural AI Error: {str(e)[:100]}")
                    else:
                        send_telegram_message("🧠 Neural AI not available.\nInstall sklearn: pip install scikit-learn")
                
                # /ai command — advanced AI statistics
                elif text == '/ai':
                    try:
                        advanced_brain = get_advanced_brain()
                        
                        # Build report
                        lines = ["🤖 <b>ADVANCED AI SYSTEM</b>\n━━━━━━━━━━━━━━━━"]
                        
                        # Patterns learned
                        if advanced_brain.patterns.patterns:
                            lines.append(f"\n📊 <b>Patterns Learned:</b> {len(advanced_brain.patterns.patterns)}")
                            # Top 3 patterns
                            top_patterns = sorted(
                                advanced_brain.patterns.patterns.values(),
                                key=lambda p: p.win_rate * p.occurrences,
                                reverse=True
                            )[:3]
                            for p in top_patterns:
                                wr = p.win_rate * 100
                                lines.append(f"   • WR {wr:.1f}% ({p.occurrences} trades)")
                        
                        # Symbol profiles
                        if advanced_brain.profiler.profiles:
                            lines.append(f"\n🎯 <b>Symbol Profiles:</b> {len(advanced_brain.profiler.profiles)}")
                            hot_symbols = [p for p in advanced_brain.profiler.profiles.values() if p.is_hot]
                            cold_symbols = [p for p in advanced_brain.profiler.profiles.values() if p.is_cold]
                            if hot_symbols:
                                lines.append(f"   🔥 Hot (winning streak): {', '.join(p.symbol for p in hot_symbols)}")
                            if cold_symbols:
                                lines.append(f"   ❄️ Cold (losing streak): {', '.join(p.symbol for p in cold_symbols)}")
                        
                        # Market phases
                        lines.append(f"\n📈 <b>Market Structure:</b> Detection Active")
                        
                        # Order flow
                        lines.append(f"💧 <b>Order Flow:</b> Monitoring ({len(advanced_brain.market_structure.order_flow.history)} samples)")
                        
                        lines.append(f"\n━━━━━━━━━━━━━━━━\n✅ AI is learning in real-time")
                        
                        send_telegram_message('\n'.join(lines))
                    except Exception as e:
                        send_telegram_message(f"🤖 Advanced AI: Initializing...\n{str(e)[:100]}")

                # /ping command
                elif text == '/ping':
                    send_telegram_message("✅ Bot is online and connected.")

                # /risk command
                elif text.startswith('/risk'):
                    parts = text.split()
                    if len(parts) >= 2:
                        try:
                            val = float(parts[1])
                            if 0.01 <= val <= 100.0:
                                cfg['risk_percent'] = round(val, 2)
                                send_telegram_message(f"✅ <b>Risk Updated</b>\n{cfg['risk_percent']}% per trade")
                                changed = True
                            else:
                                send_telegram_message("❌ Risk must be between 0.01-100%\nExample: /risk 2.5")
                        except ValueError:
                            send_telegram_message("❌ Invalid format\nExample: /risk 2.5")
                    else:
                        send_telegram_message(f"📊 <b>Current Risk</b>\n{cfg['risk_percent']}% per trade\n\nTo change: /risk [value]")
                
                # /status command
                elif text == '/status':
                    enabled = ', '.join(cfg['enabled_symbols'])
                    positions = get_open_positions()
                    pos_count = len(positions)
                    total_pl = sum(p.profit for p in positions)
                    daily_pnl = get_daily_pnl()
                    state_name = get_state().value
                    
                    status_msg = (
                        "📊 <b>Bot Status</b>\n"
                        "━━━━━━━━━━━━━━━━\n"
                        f"🔄 State: {state_name}\n"
                        f"⚠️ Risk: {cfg['risk_percent']}% per trade\n"
                        f"✅ Active Symbols: {len(cfg['enabled_symbols'])}/{len(SYMBOLS)}\n"
                        f"   {enabled}\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"📈 Open Positions: {pos_count}\n"
                    )
                    if pos_count > 0:
                        status_msg += f"💰 Open P/L: ${total_pl:.2f}\n"
                    status_msg += f"📅 Daily P&amp;L: ${daily_pnl:.2f}"
                    send_telegram_message(status_msg)
                
                # /positions command
                elif text == '/positions':
                    positions = get_open_positions()
                    if not positions:
                        send_telegram_message("📈 <b>No Open Positions</b>\n\nWaiting for signals...")
                    else:
                        lines = [f"📈 <b>Open Positions ({len(positions)})</b>\n━━━━━━━━━━━━━━━━"]
                        total_pl = 0
                        for pos in positions:
                            direction = 'BUY 🟢' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL 🔴'
                            pl_emoji = '✅' if pos.profit >= 0 else '❌'
                            lines.append(
                                f"\n<b>{pos.symbol}</b> {direction}\n"
                                f"   Lot: {pos.volume} @ {pos.price_open}\n"
                                f"   {pl_emoji} P/L: ${pos.profit:.2f}"
                            )
                            total_pl += pos.profit
                        lines.append(f"\n━━━━━━━━━━━━━━━━\n💰 <b>Total: ${total_pl:.2f}</b>")
                        send_telegram_message('\n'.join(lines))
                
                # Symbol toggle commands
                elif text.startswith('/'):
                    sym = text[1:].upper()
                    if sym in SYMBOLS:
                        enabled = set(cfg['enabled_symbols'])
                        if sym in enabled:
                            enabled.remove(sym)
                            remaining = len(enabled)
                            send_telegram_message(
                                f"🔴 <b>Disabled {sym}</b>\n"
                                f"Active symbols: {remaining}/{len(SYMBOLS)}"
                            )
                        else:
                            enabled.add(sym)
                            send_telegram_message(
                                f"🟢 <b>Enabled {sym}</b>\n"
                                f"Active symbols: {len(enabled)}/{len(SYMBOLS)}"
                            )
                        cfg['enabled_symbols'] = sorted(enabled)
                        changed = True
                    else:
                        send_telegram_message(
                            f"❌ Unknown command: {text}\n\n"
                            f"Type /help for available commands"
                        )
            
            if changed:
                save_config(cfg)
            return cfg
        except Exception:
            return cfg


def log_trade(trade_data: dict):
    """Log trade to CSV file."""
    file_exists = Path(TRADE_LOG_FILE).exists()
    
    with open(TRADE_LOG_FILE, 'a', newline='') as f:
        fieldnames = ['timestamp', 'symbol', 'direction', 'entry_price', 'stop_loss', 'take_profit',
                     'lot_size', 'risk_percent', 'status', 'exit_time', 'exit_price', 'profit']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(trade_data)


def send_signal_alert(signal: dict, risk: float, quiet: bool = False):
    """Send signal alert to Telegram."""
    ts = signal['timestamp'].strftime('%Y-%m-%d %H:%M:%S')
    direction_emoji = "🟢" if signal['direction'] == 'BUY' else "🔴"
    
    msg = (
        f"{direction_emoji} <b>{signal['direction']} {signal['symbol']}</b>\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"📍 Entry: {signal['entry']}\n"
        f"🎯 TP: {signal['tp']}\n"
        f"🛑 SL: {signal['stop']}\n"
        f"📊 Spread: {signal['spread']} pts\n"
        f"━━━━━━━━━━━━━━━━\n"
        f"ADX: {signal['adx']} | ATR: {signal['atr']}\n"
        f"EMA: {signal['ema_fast']} / {signal['ema_slow']}\n"
        f"⚠️ Risk: {risk}%\n"
        f"🕐 {ts}\n"
        f"<i>{signal['params']}</i>"
    )
    
    try:
        send_telegram_message(msg)
    except Exception:
        if not quiet:
            print(f"[!] Telegram send failed")


# main loop

def show_open_positions():
    """Display all open positions on startup."""
    positions = get_open_positions()
    if not positions:
        print("[i] No open positions")
        return
    
    print(f"[i] Open positions: {len(positions)}")
    for pos in positions:
        direction = 'BUY' if pos.type == mt5.ORDER_TYPE_BUY else 'SELL'
        print(f"    {pos.symbol}: {direction} {pos.volume} @ {pos.price_open} (P/L: ${pos.profit:.2f})")


def process_single_symbol(symbol: str, enabled: set, risk: float) -> tuple:
    """Process a single symbol and return its status and signal."""
    global last_signals, tracked_positions
    
    if symbol not in enabled:
        return (symbol, None, None)
    
    # Get REAL symbol info from MT5
    sym_info = get_symbol_info(symbol)
    if not sym_info:
        return (symbol, f"{symbol}:ERR", None)
    
    # Check if we already have an open position for this symbol
    existing_pos = get_position_for_symbol(symbol)
    
    # Add entry_regime from tracked_positions if available
    if existing_pos and symbol in tracked_positions:
        existing_pos['entry_regime'] = tracked_positions[symbol].get('entry_regime', 'unknown')
    
    # Get strategy params from best_settings.json (baseline)
    params = get_instrument_settings(symbol)
    if not params:
        return (symbol, f"{symbol}:NOCFG", None)
    
    # Get learned params if available (bot has learned improvements)
    params = optimizer.get_active_params(symbol, {symbol: params})
    if symbol in params:
        params = params[symbol]
    
    # Fetch LIVE candles
    df = fetch_live_candles(symbol)
    if df is None or len(df) < 100:
        return (symbol, f"{symbol}:NODATA", None)
    
    # Generate basic signal
    signal = generate_signal(df, params, symbol, sym_info)
    
    # ADVANCED AI ANALYSIS - Validate and enhance signal
    if signal:
        try:
            advanced_brain = get_advanced_brain()
            # Add indicators for the AI (keep original column case: ATR/ADX/EMA_*) and add lowercase OHLC for compatibility
            df_ai = add_indicators(df, int(params.get('EMA_Fast', 10)), int(params.get('EMA_Slow', 50))).fillna(0)
            # Ensure lowercase aliases expected by advanced_ai_brain internals
            for upper, lower in [('High', 'high'), ('Low', 'low'), ('Close', 'close'), ('Open', 'open'), ('Time', 'time'), ('Volume', 'volume')]:
                if upper in df_ai.columns:
                    df_ai[lower] = df_ai[upper]
            signal = advanced_brain.analyze_signal(
                symbol, df_ai, signal,
                sym_info['bid'], sym_info['ask'], sym_info['spread']
            )
            
            # 🧠 NEURAL AI - REAL Machine Learning Analysis
            if NEURAL_AI_AVAILABLE and signal.get('ai_approved', False):
                try:
                    neural_ai = get_neural_ai()
                    direction = signal.get('direction', 'BUY')
                    current_price = sym_info['bid'] if direction == 'SELL' else sym_info['ask']
                    atr_value = df_ai['ATR'].iloc[-1] if 'ATR' in df_ai.columns else 0.001
                    
                    neural_result = neural_ai.analyze_entry(
                        symbol, df_ai, direction, current_price, atr_value
                    )
                    
                    # Neural AI can boost or reduce confidence
                    neural_conf = neural_result.get('confidence', 0.5)
                    original_conf = signal.get('confidence', 0.6)
                    
                    # Blend confidences: 40% original, 60% neural
                    blended_conf = original_conf * 0.4 + neural_conf * 0.6
                    signal['confidence'] = blended_conf
                    signal['neural_confidence'] = neural_conf
                    signal['neural_approved'] = neural_result.get('should_trade', True)
                    
                    # Store features for learning later
                    signal['neural_features'] = neural_result.get('features', [])
                    
                    if neural_conf < 0.35:
                        print(f"🧠 {symbol}: Neural AI LOW confidence ({neural_conf:.1%}) - cautious")
                    elif neural_conf > 0.65:
                        print(f"🧠 {symbol}: Neural AI HIGH confidence ({neural_conf:.1%}) - aggressive")
                    
                except Exception as e:
                    print(f"[!] Neural AI error on {symbol}: {e}")
            
            # If advanced AI disapproves, reject signal
            if not signal.get('ai_approved', False):
                print(f"⚠️  {symbol}: Signal rejected - {signal.get('reason', 'Low confidence')}")
                signal = None
            else:
                # Use smart SL/TP if available
                if 'smart_sl' in signal and 'smart_tp' in signal:
                    signal['stop'] = signal['smart_sl']
                    signal['tp'] = signal['smart_tp']
                
                print(f"✅ {symbol}: AI approved (confidence {signal.get('confidence', 0):.1%})")
        except Exception as e:
            print(f"[!] Advanced AI error on {symbol}: {e}")
            # Continue with original signal if advanced AI fails but tag a safe confidence
            signal['confidence'] = max(signal.get('confidence', 0.6), 0.6)
            signal['ai_approved'] = True
        
        # Ensure confidence defaults so HTF gate doesn't see 0%
        signal.setdefault('confidence', 0.6)
        signal.setdefault('ai_approved', True)
    
    # If we have a position → MANAGEMENT state
    if existing_pos:
        set_state(BotState.MANAGEMENT)
        pl = existing_pos['profit']
        
        # ── BREAK-EVEN & TRAILING STOP (per documentation) ──────────
        # Rule 1: At 1R profit → SL moves to entry price (break-even)
        # Rule 2: At 50 % of target profit → SL moves to 25 % profit level
        tracked = tracked_positions.get(symbol, {})
        entry_price = tracked.get('entry_price') or existing_pos.get('open_price', 0)
        original_sl = tracked.get('original_sl') or existing_pos.get('sl', 0)
        original_tp = tracked.get('original_tp') or existing_pos.get('tp', 0)
        current_sl = existing_pos.get('sl', original_sl)
        be_stage = tracked.get('be_stage', 0)  # 0=none, 1=BE done, 2=trail done

        if entry_price and original_sl and original_tp:
            initial_risk = abs(entry_price - original_sl)
            target_profit = abs(original_tp - entry_price)
            current_price = sym_info['bid'] if existing_pos['direction'] == 'BUY' else sym_info['ask']
            digits = sym_info.get('digits', 5)

            if existing_pos['direction'] == 'BUY':
                unrealized = current_price - entry_price
            else:
                unrealized = entry_price - current_price

            # Stage 1: At 1R profit → break-even (SL = entry)
            if be_stage < 1 and initial_risk > 0 and unrealized >= initial_risk:
                new_sl = round(entry_price, digits)
                if (existing_pos['direction'] == 'BUY' and new_sl > current_sl) or \
                   (existing_pos['direction'] == 'SELL' and new_sl < current_sl):
                    if modify_position_sl_tp(existing_pos, new_sl=new_sl):
                        tracked_positions.setdefault(symbol, {})['be_stage'] = 1
                        tracked_positions[symbol]['sl'] = new_sl
                        log_event(f"{symbol}: Break-even SL moved to {new_sl}", "INFO")
                        send_telegram_message(
                            f"🛡️ <b>Break-Even</b>\n"
                            f"{symbol}: SL → {new_sl} (entry price)\n"
                            f"Unrealized: ${unrealized:.2f}"
                        )
                        be_stage = 1

            # Stage 2: At 50 % of target → SL moves to 25 % profit level
            if be_stage < 2 and target_profit > 0 and unrealized >= target_profit * 0.5:
                if existing_pos['direction'] == 'BUY':
                    new_sl = round(entry_price + target_profit * 0.25, digits)
                else:
                    new_sl = round(entry_price - target_profit * 0.25, digits)
                if (existing_pos['direction'] == 'BUY' and new_sl > current_sl) or \
                   (existing_pos['direction'] == 'SELL' and new_sl < current_sl):
                    if modify_position_sl_tp(existing_pos, new_sl=new_sl):
                        tracked_positions.setdefault(symbol, {})['be_stage'] = 2
                        tracked_positions[symbol]['sl'] = new_sl
                        log_event(f"{symbol}: Trailing SL moved to {new_sl} (25% profit)", "INFO")
                        send_telegram_message(
                            f"📈 <b>Trailing Stop</b>\n"
                            f"{symbol}: SL → {new_sl} (25% profit lock)\n"
                            f"Unrealized: ${unrealized:.2f}"
                        )

        # AI position manager (supplementary)
        try:
            existing_pos['current_price'] = sym_info['bid'] if existing_pos['direction'] == 'BUY' else sym_info['ask']
            existing_pos['bid'] = sym_info['bid']
            existing_pos['ask'] = sym_info['ask']
            
            decision = brain.manage_open_position(symbol, existing_pos, df)
            
            if decision and decision.action != 'hold':
                if decision.action == 'close_full':
                    print(f"\n🧠 AI: {symbol} close_full - {decision.reason}")
                    if close_position(existing_pos):
                        return (symbol, f"{symbol}:AI_CLOSED", None)
                elif decision.action == 'trail_sl' and decision.new_sl:
                    # Only allow AI to tighten SL further, never loosen
                    cur_sl = tracked_positions.get(symbol, {}).get('sl', existing_pos.get('sl', 0))
                    if existing_pos['direction'] == 'BUY' and decision.new_sl > cur_sl:
                        if modify_position_sl_tp(existing_pos, new_sl=decision.new_sl):
                            tracked_positions.setdefault(symbol, {})['sl'] = decision.new_sl
                    elif existing_pos['direction'] == 'SELL' and decision.new_sl < cur_sl:
                        if modify_position_sl_tp(existing_pos, new_sl=decision.new_sl):
                            tracked_positions.setdefault(symbol, {})['sl'] = decision.new_sl
        except Exception:
            pass
        
        # Check traditional reversal signal (only close if in profit to avoid cutting losers)
        if signal and signal['direction'] != existing_pos['direction']:
            if pl > 0:
                print(f"\n>>> {symbol}: REVERSAL {existing_pos['direction']} -> {signal['direction']} | Closing...")
                send_telegram_message(
                    f"🔄 <b>Closing {existing_pos['direction']} {symbol}</b>\n"
                    f"Signal reversed to {signal['direction']}\n"
                    f"P/L: ${pl:.2f}"
                )
                if close_position(existing_pos):
                    return (symbol, f"{symbol}:CLOSED", None)
                else:
                    return (symbol, f"{symbol}:CLOSEFAIL", None)
            else:
                # Ignore reversal if losing; keep holding
                pass

        # Holding position
        dir_char = '▲' if existing_pos['direction'] == 'BUY' else '▼'
        pl_str = f"+${pl:.0f}" if pl >= 0 else f"-${abs(pl):.0f}"
        return (symbol, f"{symbol}:{dir_char}{pl_str}", None)
    
    # No existing position - check for new signal
    if signal:
        sig_key = f"{symbol}_{signal['direction']}"
        last_time = last_signals.get(sig_key)
        if last_time and (datetime.now(timezone.utc) - last_time).seconds < 60:
            return (symbol, f"{symbol}:WAIT", None)
        
        # Return signal for execution
        return (symbol, None, signal)
    else:
        return (symbol, f"{symbol}:-", None)


def scan_markets(cfg: dict, verbose: bool = False):
    """Scan all enabled symbols for signals and execute trades (PARALLEL)."""
    global last_signals, tracked_positions
    
    # Connection health check
    if not check_mt5_connection():
        print("[!] MT5 connection lost during scan")
        if not reconnect_mt5():
            print("[!] Could not reconnect to MT5. Skipping this scan cycle.")
            update_runtime_status(state='degraded', message='MT5 reconnect failed', enabled_symbols=cfg.get('enabled_symbols', []))
            return
    
    enabled = set(cfg.get('enabled_symbols', SYMBOLS))
    risk = cfg.get('risk_percent', DEFAULT_RISK)
    
    # First, check for closed positions (SL/TP hit) - keep this sequential
    current_positions = {pos.symbol: pos for pos in get_open_positions()}
    
    for symbol, prev_pos in list(tracked_positions.items()):
        if symbol not in current_positions:
            # Position was closed (SL or TP hit)
            from_date = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0)
            to_date = datetime.now(timezone.utc)
            
            deals = mt5.history_deals_get(from_date, to_date)
            if deals:
                for deal in reversed(deals):
                    if deal.symbol == symbol and deal.position_id == prev_pos.get('ticket'):
                        exit_price = deal.price
                        profit = deal.profit
                        
                        # Log closed trade (CSV — legacy)
                        log_trade({
                            'timestamp': prev_pos['open_time'],
                            'symbol': symbol,
                            'direction': prev_pos['direction'],
                            'entry_price': prev_pos['entry_price'],
                            'stop_loss': prev_pos.get('sl', ''),
                            'take_profit': prev_pos.get('tp', ''),
                            'lot_size': prev_pos['lot_size'],
                            'risk_percent': prev_pos.get('risk_percent', risk),
                            'status': 'CLOSED',
                            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'exit_price': exit_price,
                            'profit': f"{profit:.2f}"
                        })
                        
                        # Log closed trade (SQLite database)
                        try:
                            close_trade(
                                prev_pos.get('ticket', 0),
                                close_price=exit_price,
                                profit=profit,
                                exit_reason='SL/TP',
                            )
                            log_event(f"Trade closed: {prev_pos['direction']} {symbol} profit=${profit:.2f}", "INFO")
                        except Exception:
                            pass
                        
                        # ADVANCED AI LEARNING - Learn from closed trades
                        try:
                            advanced_brain = get_advanced_brain()
                            
                            # Determine outcome
                            if profit > 0:
                                outcome = TradeOutcome.WIN
                            elif profit < 0:
                                outcome = TradeOutcome.LOSS
                            else:
                                outcome = TradeOutcome.BREAKEVEN
                            
                            # Record for learning
                            advanced_brain.update_after_trade(
                                symbol=symbol,
                                features={},  # Would extract from trade data
                                outcome=outcome,
                                pl=profit,
                                hour=datetime.now().hour
                            )
                        except Exception as e:
                            print(f"[!] Advanced AI learning error: {e}")
                        
                        # 🧠 NEURAL AI LEARNING - Learn from closed trade
                        if NEURAL_AI_AVAILABLE:
                            try:
                                neural_ai = get_neural_ai()
                                
                                # Get stored features from entry (if available)
                                entry_features = prev_pos.get('neural_features', None)
                                if entry_features is not None:
                                    import numpy as np
                                    entry_features = np.array(entry_features)
                                    
                                    neural_ai.record_trade_result(
                                        symbol=symbol,
                                        entry_features=entry_features,
                                        exit_features=entry_features,  # Use same for now
                                        profit=profit,
                                        direction=prev_pos['direction']
                                    )
                                    print(f"🧠 Neural AI learned from {symbol} trade: ${profit:.2f}")
                            except Exception as e:
                                print(f"[!] Neural AI learning error: {e}")
                        
                        # Notify old AI brain of trade close
                        try:
                            brain.on_trade_closed(symbol, exit_price, profit)
                        except Exception:
                            pass
                        
                        result_emoji = "✅" if profit > 0 else "❌"
                        send_telegram_message(
                            f"{result_emoji} <b>Trade Closed</b>\\n"
                            f"{prev_pos['direction']} {symbol}\\n"
                            f"Entry: {prev_pos['entry_price']}\\n"
                            f"Exit: {exit_price}\\n"
                            f"Profit: ${profit:.2f}"
                        )
                        break
            
            del tracked_positions[symbol]
    
    # PARALLEL SCAN: Process all symbols concurrently
    status = []
    signals_to_execute = []
    
    with ThreadPoolExecutor(max_workers=len(SYMBOLS)) as executor:
        # Submit all symbol processing tasks
        future_to_symbol = {executor.submit(process_single_symbol, symbol, enabled, risk): symbol 
                           for symbol in SYMBOLS}
        
        # Collect results as they complete
        for future in as_completed(future_to_symbol):
            symbol, status_str, signal = future.result()
            
            if status_str:
                status.append(status_str)
            
            if signal:
                signals_to_execute.append((symbol, signal))
    
    # Execute any new signals (sequential for safety)
    for symbol, signal in signals_to_execute:
        set_state(BotState.EXECUTION)

        # ── SAFETY: Daily Drawdown Circuit Breaker (5 %) ────────────
        # Doc: "ak denný drawdown presiahne 5 %, bot automaticky deaktivuje obchodovanie"
        try:
            account = mt5.account_info()
            if account:
                daily_pnl = get_daily_pnl()
                if account.balance > 0 and abs(daily_pnl) / account.balance * 100 >= MAX_DAILY_DRAWDOWN and daily_pnl < 0:
                    print(f"[!] CIRCUIT BREAKER: daily loss ${daily_pnl:.2f} exceeds {MAX_DAILY_DRAWDOWN}% — blocking new trades")
                    log_event(f"Circuit breaker triggered: daily PnL ${daily_pnl:.2f}", "WARN")
                    send_telegram_message(
                        f"🚨 <b>Circuit Breaker</b>\n"
                        f"Daily loss ${daily_pnl:.2f} exceeds {MAX_DAILY_DRAWDOWN}%\n"
                        f"New trades blocked until tomorrow"
                    )
                    status.append(f"{symbol}:BLOCKED")
                    continue
        except Exception:
            pass

        # ── SAFETY: Margin Protection (20 %) ────────────────────────
        # Doc: "ak je viac ako 20 % kapitálu viazaného v marži, nový príkaz sa zablokuje"
        try:
            account = mt5.account_info()
            if account and account.balance > 0:
                margin_used_pct = (account.balance - account.margin_free) / account.balance * 100
                if margin_used_pct > MAX_MARGIN_USAGE:
                    print(f"[!] MARGIN LIMIT: {margin_used_pct:.1f}% margin used > {MAX_MARGIN_USAGE}% — blocking new order")
                    log_event(f"Margin protection: {margin_used_pct:.1f}% used", "WARN")
                    status.append(f"{symbol}:MARGIN")
                    continue
        except Exception:
            pass

        # NEW SIGNAL - this is important, print it
        sym_info = get_symbol_info(symbol)
        dir_char = '▲' if signal['direction'] == 'BUY' else '▼'
        print(f"\n>>> NEW SIGNAL: {dir_char} {signal['direction']} {symbol} @ {signal['entry']} | TP:{signal['tp']} SL:{signal['stop']}")
        
        # Multi-timeframe confirmation (H1)
        params_for_htf = get_instrument_settings(symbol)
        htf_bias = get_htf_bias(symbol, params_for_htf, timeframe=mt5.TIMEFRAME_H1)

        # Gate disabled for learning phase unless confidence is extremely low
        conf = float(signal.get('confidence', 0.6))
        opposes = (htf_bias == 'BULL' and signal['direction'] == 'SELL') or (htf_bias == 'BEAR' and signal['direction'] == 'BUY')
        if htf_bias in ('BULL', 'BEAR') and opposes and conf < 0.25:
            print(f"⚠️  {symbol}: Rejected by HTF ({htf_bias}) with ultra-low confidence {conf:.0%}")
            status.append(f"{symbol}:MTF")
            continue

        # Adjust risk based on AI + HTF
        risk_used = adjust_risk_percent(risk, signal, htf_bias)

        # Don't send signal alert - will send when position actually opens
        success = open_position_with_retry(signal, sym_info, risk_used)
        
        if success:
            sig_key = f"{symbol}_{signal['direction']}"
            last_signals[sig_key] = datetime.now(timezone.utc)
            status.append(f"{symbol}:OPENED")
            send_telegram_message(
                f"✅ <b>Position Opened</b>\n"
                f"{signal['direction']} {symbol}\n"
                f"Entry: {signal['entry']}\n"
                f"🎯 TP: {signal['tp']}\n"
                f"🛑 SL: {signal['stop']}\n"
                f"⚠️ Risk: {risk_used}%  | HTF: {htf_bias}"
            )
            
            # Track this position for closure detection
            pos = get_position_for_symbol(symbol)
            if pos:
                # Get current regime for entry tracking
                try:
                    sym_info_temp = get_symbol_info(symbol)
                    df_temp = fetch_live_candles(symbol)
                    entry_regime = 'unknown'
                    if sym_info_temp and df_temp is not None:
                        params_temp = get_instrument_settings(symbol)
                        brain_data = brain.process_market_data(symbol, df_temp, sym_info_temp, params_temp)
                        if brain_data and 'regime' in brain_data:
                            entry_regime = brain_data['regime'].regime
                except Exception:
                    entry_regime = 'unknown'
                
                tracked_positions[symbol] = {
                    'ticket': pos.get('ticket'),
                    'direction': signal['direction'],
                    'entry_price': signal['entry'],
                    'original_sl': signal['stop'],
                    'original_tp': signal['tp'],
                    'sl': signal['stop'],
                    'tp': signal['tp'],
                    'lot_size': signal.get('lot_size', 0),
                    'open_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_regime': entry_regime,
                    'risk_percent': risk_used,
                    'htf_bias': htf_bias,
                    'be_stage': 0,  # 0=none, 1=BE done, 2=trail done
                }
                
                # ── SQLite: record trade open ──
                try:
                    insert_trade(
                        ticket=pos.get('ticket', 0),
                        symbol=symbol,
                        trade_type=signal['direction'],
                        open_price=signal['entry'],
                        sl=signal['stop'],
                        tp=signal['tp'],
                        lot_size=signal.get('lot_size', 0),
                        risk_percent=risk_used,
                    )
                    log_event(f"Trade opened: {signal['direction']} {symbol} @ {signal['entry']}", "INFO")
                except Exception:
                    pass
                
                # Notify AI brain of trade open
                try:
                    params = get_instrument_settings(symbol)
                    brain.on_trade_opened(
                        symbol, signal['direction'], signal['entry'],
                        signal['stop'], signal['tp'], pos.get('lot_size', 0.01), params
                    )
                except Exception:
                    pass
        else:
            status.append(f"{symbol}:FAIL")
            # Don't spam Telegram on failed orders
    
    # Single compact status line (sorted by symbol order in SYMBOLS)
    status_dict = {s.split(':')[0]: s for s in status}
    sorted_status = [status_dict.get(sym, f"{sym}:-") for sym in SYMBOLS if sym in enabled]
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {' | '.join(sorted_status)}")

    opened_count = sum(1 for s in status if s.endswith(':OPENED'))
    fail_count = sum(1 for s in status if s.endswith(':FAIL'))
    update_runtime_status(
        state='running',
        message='Scan cycle completed',
        enabled_symbols=sorted(enabled),
        open_positions=len(get_open_positions()),
        signals_detected=len(signals_to_execute),
        positions_opened=opened_count,
        failed_orders=fail_count,
        status_line=' | '.join(sorted_status),
    )


def main():
    parser = argparse.ArgumentParser(description="LIVE Trading Bot")
    parser.add_argument('--once', action='store_true', help='Single scan and exit')
    parser.add_argument('--loop', type=int, default=10, help='Seconds between scans (default: 10)')
    args = parser.parse_args()

    print("=" * 60)
    print("  ULTIMA TRADING BOT — ICT / SMC Engine")
    print("=" * 60)

    # Initialize SQLite database (trades, order_blocks, logs)
    init_trading_db()
    log_event("Bot process starting", "INFO")

    # Initialize MT5
    if not init_mt5():
        print("[!] Cannot start without MT5 connection")
        update_runtime_status(state='error', message='Cannot start without MT5 connection')
        return
    
    print(f"[✓] MT5 connected: {mt5.terminal_info().name}")
    print(f"[✓] Account: {mt5.account_info().login}")
    update_runtime_status(state='starting', message='Bot process initialized')
    
    # Load config
    cfg = load_config()
    print(f"[✓] Risk: {cfg['risk_percent']}%")
    print(f"[✓] Symbols: {', '.join(cfg['enabled_symbols'])}")
    
    # Load baseline strategy params
    baseline_params = {}
    for sym in SYMBOLS:
        params = get_instrument_settings(sym)
        if params:
            baseline_params[sym] = params
    
    # Load optimizer (learns from past trades)
    if Path(LEARNED_PARAMS_FILE).exists():
        print("[🤖] Loading learned parameters from past trades...")
    
    # Telegram bot for commands
    tg = TelegramBot()
    if tg.is_configured():
        print("[✓] Telegram connected")
    else:
        print("[!] Telegram not configured")
    
    # Initialize AI brain with base parameters for each symbol
    print("[🧠] Initializing AI Trading Brain...")
    for sym in SYMBOLS:
        base_params = get_instrument_settings(sym)
        if base_params:
            brain.initialize_symbol(sym, base_params)
    print("[✓] AI Brain online - learning from market data")
    
    # Show existing open positions
    show_open_positions()
    
    print("=" * 60)

    try:
        if args.once:
            cfg = load_config()
            scan_markets(cfg)
        else:
            interval = max(2, args.loop)
            print(f"[+] Scanning every {interval}s (Ctrl+C to stop)")
            print(f"[🤖] Self-learning enabled - optimizing every {OPTIMIZER_INTERVAL}s\n")
            
            # Start Telegram polling in separate thread for instant response
            telegram_active = threading.Event()
            telegram_active.set()
            
            def telegram_loop():
                nonlocal cfg
                while telegram_active.is_set():
                    cfg = tg.poll_commands(cfg)
                    time.sleep(1)
            
            # Start optimizer thread for continuous learning
            optimizer_active = threading.Event()
            optimizer_active.set()
            
            def optimizer_loop():
                while optimizer_active.is_set():
                    try:
                        elapsed_since_opt = time.time() - optimizer.last_optimize
                        if elapsed_since_opt >= OPTIMIZER_INTERVAL:
                            optimizer.optimize_all(baseline_params)
                            optimizer.last_optimize = time.time()
                    except Exception as e:
                        print(f"[!] Optimizer error: {e}")
                    time.sleep(10)  # Check every 10s if time to optimize
            
            # Start AI brain analysis thread for real-time intelligence
            brain_active = threading.Event()
            brain_active.set()
            brain_analysis_interval = 20  # Analyze every 20 seconds for faster reactions
            last_brain_analysis = time.time()
            
            def brain_analysis_loop():
                nonlocal last_brain_analysis
                while brain_active.is_set():
                    try:
                        elapsed = time.time() - last_brain_analysis
                        if elapsed >= brain_analysis_interval:
                            # Run strategy evolution for all symbols
                            for sym in SYMBOLS:
                                try:
                                    sym_info = get_symbol_info(sym)
                                    df = fetch_live_candles(sym)
                                    if sym_info and df is not None and len(df) >= 50:
                                        params = get_instrument_settings(sym)
                                        brain.process_market_data(sym, df, sym_info, params)
                                except Exception:
                                    pass
                            
                            # Cleanup old data weekly
                            if datetime.now().weekday() == 0 and datetime.now().hour == 3:
                                brain.cleanup()
                            
                            last_brain_analysis = time.time()
                    except Exception as e:
                        print(f"[!] Brain analysis error: {e}")
                    time.sleep(15)  # Check every 15s
            
            telegram_thread = threading.Thread(target=telegram_loop, daemon=True)
            optimizer_thread = threading.Thread(target=optimizer_loop, daemon=True)
            brain_thread = threading.Thread(target=brain_analysis_loop, daemon=True)
            telegram_thread.start()
            optimizer_thread.start()
            brain_thread.start()
            
            print(f"[🧠] AI Brain analyzing every {brain_analysis_interval}s")
            
            # Market scanning loop
            while True:
                start = time.time()
                try:
                    cfg = load_config()
                    scan_markets(cfg)
                except Exception as e:
                    print(f"[!] Scan loop error: {e}")
                    update_runtime_status(state='error', message=f"Scan loop error: {e}")
                elapsed = time.time() - start
                time.sleep(max(1, interval - elapsed))
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
        update_runtime_status(state='stopped', message='Stopped by user')
        if not args.once:
            telegram_active.clear()
            optimizer_active.clear()
            brain_active.clear()
    finally:
        shutdown_mt5()
        update_runtime_status(state='stopped', message='MT5 disconnected')
        print("[✓] MT5 disconnected")


if __name__ == '__main__':
    main()
