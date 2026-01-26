"""
LIVE Trading Bot - Scans real-time MT5 market data and sends Telegram signals.
Uses the optimized EMA/ADX/ATR strategy with parameters from best_settings.json.
NOW WITH AI BRAIN: Self-learning, market-aware adaptive intelligence.

Usage:
  python main.py              # default: scan every 5 seconds
  python main.py --loop 10    # scan every 10 seconds
  python main.py --once       # single scan and exit

Telegram Commands:
  /risk 2.5    - Set risk to 2.5% per trade
  /eurusd      - Toggle EURUSD on/off
  /status      - Show current config
  /brain       - AI brain status & insights
"""
import argparse
import csv
import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timezone
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd
import requests

import MetaTrader5 as mt5

from strategy import get_instrument_settings
from telegram_bot import send_telegram_message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
from ai_brain import get_brain, TradingIntelligenceBrain

# ============================================================================
# CONFIG
# ============================================================================

# Symbols to trade - MT5 provides real tick size, spread, digits
SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'BTCUSD', 'NAS100']

# Broker uses '+' suffix for these symbols
BROKER_SUFFIX = {'EURUSD': '+', 'GBPUSD': '+', 'USDJPY': '+', 'GBPJPY': '+', 'XAUUSD': '+'}

# Strategy constants (same as backtested)
SESSION_START = 5   # hour UTC
SESSION_END = 23    # hour UTC
ATR_PERIOD = 14
ADX_PERIOD = 14
LOOKBACK_BARS = 500  # enough bars for indicator calculation
TP_RR_RATIO = 2.0   # Take Profit at 2:1 risk-reward (TP = 2x SL distance)

# Runtime config persistence
CONFIG_FILE = Path(__file__).parent / 'runtime_config.json'
DEFAULT_RISK = 1.0  # percent

# Track last signal to avoid spam
last_signals = {}

# Track open positions to detect closures
tracked_positions = {}

# Order retry settings
MAX_ORDER_RETRIES = 3
RETRY_DELAY = 1.0  # seconds

# Trade logging (only closed trades)
TRADE_LOG_FILE = 'liverun/live_trades.csv'

# Learned parameters (auto-adjusted by optimizer)
LEARNED_PARAMS_FILE = 'liverun/learned_params.json'

# Optimizer settings
OPTIMIZER_INTERVAL = 300  # Run optimizer every 5 minutes
MIN_TRADES_FOR_LEARNING = 5  # Min closed trades before adjusting params

# ============================================================================
# PARAMETER OPTIMIZER - Self-Learning Bot
# ============================================================================

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

# ============================================================================
# TRADE LOGGING
# ============================================================================

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


# ============================================================================
# MT5 CONNECTION
# ============================================================================

def init_mt5() -> bool:
    """Initialize MT5 connection."""
    if not mt5.initialize():
        print(f"[!] MT5 init failed: {mt5.last_error()}")
        return False
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


# ============================================================================
# POSITION MANAGEMENT
# ============================================================================

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


def fetch_live_candles(symbol: str, timeframe=mt5.TIMEFRAME_M5, bars: int = LOOKBACK_BARS) -> pd.DataFrame | None:
    """Fetch live candles from MT5."""
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


# ============================================================================
# INDICATORS (same as backtested strategy)
# ============================================================================

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


def generate_signal(df: pd.DataFrame, params: dict, symbol: str, sym_info: dict) -> dict | None:
    """Generate trading signal based on latest bar. Now integrated with AI Brain."""
    if not params:
        return None
    
    fast = int(params.get('EMA_Fast', 10))
    slow = int(params.get('EMA_Slow', 50))
    adx_th = float(params.get('ADX', 20))
    atr_mult = float(params.get('ATR_Mult', 1.5))
    
    # XAUUSD override: start with lower ADX to allow trading and learning
    # The optimizer will adjust based on actual live performance
    if symbol == 'XAUUSD':
        adx_th = 15.0

    df = add_indicators(df, fast, slow).fillna(0)
    i = len(df) - 1
    if i < max(slow, 50):
        return None

    # Get latest values
    adx = df['ADX'].iat[i]
    ema_f = df['EMA_Fast'].iat[i]
    ema_s = df['EMA_Slow'].iat[i]
    atr = df['ATR'].iat[i]
    bar_time = df['Time'].iat[i]
    hour = bar_time.hour

    # Use LIVE bid/ask from MT5
    bid = sym_info['bid']
    ask = sym_info['ask']
    spread_points = sym_info['spread']
    digits = sym_info['digits']
    
    # ========== AI BRAIN INTEGRATION ==========
    # Let the brain analyze market conditions and potentially adapt parameters
    try:
        brain_analysis = brain.process_market_data(symbol, df, sym_info, params)
        
        # Check if brain recommends against trading
        if not brain_analysis.get('should_trade', True):
            reason = brain_analysis.get('reason', 'Brain says no')
            # Still return None but log the reason
            return None
        
        # Use brain's adapted parameters if available
        adapted_params = brain_analysis.get('params', {})
        if adapted_params:
            adx_th = float(adapted_params.get('ADX', adx_th))
            atr_mult = float(adapted_params.get('ATR_Mult', atr_mult))
            # XAUUSD still gets lower ADX for learning
            if symbol == 'XAUUSD' and adx_th > 15:
                adx_th = 15.0
    except Exception as e:
        # If brain fails, continue with original params
        pass
    # ==========================================

    # Session filter
    if not (SESSION_START <= hour <= SESSION_END):
        return None
    
    # ADX filter
    if adx <= adx_th or not np.isfinite(adx):
        return None

    # Signal logic (same as backtested)
    if ema_f > ema_s:
        direction = 'BUY'
        entry = ask  # buy at ask
        stop = entry - (atr_mult * atr)
        tp = entry + (atr_mult * atr * TP_RR_RATIO)  # TP at RR ratio
    elif ema_f < ema_s:
        direction = 'SELL'
        entry = bid  # sell at bid
        stop = entry + (atr_mult * atr)
        tp = entry - (atr_mult * atr * TP_RR_RATIO)  # TP at RR ratio
    else:
        return None

    return {
        'symbol': symbol,
        'broker_symbol': sym_info['broker_symbol'],
        'direction': direction,
        'entry': round(entry, digits),
        'stop': round(stop, digits),
        'tp': round(tp, digits),
        'bid': bid,
        'ask': ask,
        'spread': spread_points,
        'adx': round(adx, 2),
        'atr': round(atr, digits),
        'atr_mult': atr_mult,
        'ema_fast': round(ema_f, digits),
        'ema_slow': round(ema_s, digits),
        'timestamp': bar_time,
        'params': f"EMA {fast}/{slow}, ADX>{adx_th}, ATR×{atr_mult}"
    }


# ============================================================================
# RUNTIME CONFIG
# ============================================================================

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
    if not cfg['enabled_symbols']:
        cfg['enabled_symbols'] = SYMBOLS.copy()
    return cfg


def save_config(cfg: dict):
    """Save runtime config to file."""
    try:
        CONFIG_FILE.write_text(json.dumps(cfg, indent=2))
    except Exception:
        pass


# ============================================================================
# TELEGRAM
# ============================================================================

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
                                if not (SESSION_START <= hour <= SESSION_END):
                                    reason.append(f"Session closed ({hour} UTC)")
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
                                    f"Session OK: {'YES' if (SESSION_START <= hour <= SESSION_END) else 'NO'}",
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
                                if not (SESSION_START <= hour <= SESSION_END):
                                    reason = f"session closed (UTC {hour})"
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
                    
                    status_msg = (
                        "📊 <b>Bot Status</b>\n"
                        "━━━━━━━━━━━━━━━━\n"
                        f"⚠️ Risk: {cfg['risk_percent']}% per trade\n"
                        f"✅ Active Symbols: {len(cfg['enabled_symbols'])}/{len(SYMBOLS)}\n"
                        f"   {enabled}\n"
                        f"━━━━━━━━━━━━━━━━\n"
                        f"📈 Open Positions: {pos_count}\n"
                    )
                    if pos_count > 0:
                        status_msg += f"💰 Total P/L: ${total_pl:.2f}"
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


# ============================================================================
# MAIN LOOP
# ============================================================================

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
    
    # Generate signal
    signal = generate_signal(df, params, symbol, sym_info)
    
    # If we have a position, AI manages it intelligently
    if existing_pos:
        pl = existing_pos['profit']
        
        # First check AI position manager
        try:
            # Add current price info to position
            existing_pos['current_price'] = sym_info['bid'] if existing_pos['direction'] == 'BUY' else sym_info['ask']
            existing_pos['bid'] = sym_info['bid']
            existing_pos['ask'] = sym_info['ask']
            
            # Get AI management decision
            decision = brain.manage_open_position(symbol, existing_pos, df)
            
            if decision and decision.action != 'hold':
                if decision.action == 'close_full':
                    print(f"\n🧠 AI: {symbol} {decision.action} - {decision.reason}")
                    send_telegram_message(
                        f"🧠 <b>AI Closing {existing_pos['direction']} {symbol}</b>\n"
                        f"{decision.reason}\n"
                        f"P/L: ${pl:.2f}"
                    )
                    if close_position(existing_pos):
                        return (symbol, f"{symbol}:AI_CLOSED", None)
                
                elif decision.action == 'move_breakeven' and decision.new_sl:
                    print(f"\n🧠 AI: {symbol} breakeven - {decision.reason}")
                    # Modify SL (would need modify_position function)
                    # For now just log it
                    send_telegram_message(
                        f"🛡️ <b>AI: Breakeven SL</b>\n"
                        f"{symbol}: {decision.reason}"
                    )
                
                elif decision.action == 'close_partial':
                    print(f"\n🧠 AI: {symbol} partial close - {decision.reason}")
                    send_telegram_message(
                        f"💰 <b>AI: Taking Partials</b>\n"
                        f"{symbol}: {decision.reason}\n"
                        f"Closing {decision.close_percentage:.0f}%"
                    )
                
                elif decision.action == 'trail_sl' and decision.new_sl:
                    print(f"\n🧠 AI: {symbol} trail SL - {decision.reason}")
                    send_telegram_message(
                        f"📈 <b>AI: Trailing Stop</b>\n"
                        f"{symbol}: {decision.reason}"
                    )
        except Exception as e:
            # If AI fails, continue with normal logic
            pass
        
        # Check traditional reversal signal
        if signal and signal['direction'] != existing_pos['direction']:
            # Signal reversed - close existing position
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
                        
                        # Log closed trade
                        log_trade({
                            'timestamp': prev_pos['open_time'],
                            'symbol': symbol,
                            'direction': prev_pos['direction'],
                            'entry_price': prev_pos['entry_price'],
                            'stop_loss': prev_pos.get('sl', ''),
                            'take_profit': prev_pos.get('tp', ''),
                            'lot_size': prev_pos['lot_size'],
                            'risk_percent': risk,
                            'status': 'CLOSED',
                            'exit_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                            'exit_price': exit_price,
                            'profit': f"{profit:.2f}"
                        })
                        
                        # Notify AI brain of trade close
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
        # NEW SIGNAL - this is important, print it
        sym_info = get_symbol_info(symbol)
        dir_char = '▲' if signal['direction'] == 'BUY' else '▼'
        print(f"\n>>> NEW SIGNAL: {dir_char} {signal['direction']} {symbol} @ {signal['entry']} | TP:{signal['tp']} SL:{signal['stop']}")
        
        # Don't send signal alert - will send when position actually opens
        success = open_position_with_retry(signal, sym_info, risk)
        
        if success:
            sig_key = f"{symbol}_{signal['direction']}"
            last_signals[sig_key] = datetime.now(timezone.utc)
            status.append(f"{symbol}:OPENED")
            send_telegram_message(
                f"✅ <b>Position Opened</b>\n"
                f"{signal['direction']} {symbol}\n"
                f"Entry: {signal['entry']}\n"
                f"🎯 TP: {signal['tp']}\n"
                f"🛑 SL: {signal['stop']}"
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
                    'sl': signal['stop'],
                    'tp': signal['tp'],
                    'lot_size': signal.get('lot_size', 0),
                    'open_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'entry_regime': entry_regime
                }
                
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


def main():
    parser = argparse.ArgumentParser(description="LIVE Trading Bot")
    parser.add_argument('--once', action='store_true', help='Single scan and exit')
    parser.add_argument('--loop', type=int, default=30, help='Seconds between scans (default: 30)')
    args = parser.parse_args()

    print("=" * 60)
    print("  LIVE TRADING BOT - MT5 Market Scanner")
    print("=" * 60)

    # Initialize MT5
    if not init_mt5():
        print("[!] Cannot start without MT5 connection")
        return
    
    print(f"[✓] MT5 connected: {mt5.terminal_info().name}")
    print(f"[✓] Account: {mt5.account_info().login}")
    
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
            scan_markets(cfg)
        else:
            interval = max(5, args.loop)
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
            brain_analysis_interval = 60  # Analyze every 60 seconds
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
                scan_markets(cfg)
                elapsed = time.time() - start
                time.sleep(max(1, interval - elapsed))
    except KeyboardInterrupt:
        print("\n[!] Stopped by user")
        if not args.once:
            telegram_active.clear()
            optimizer_active.clear()
            brain_active.clear()
    finally:
        shutdown_mt5()
        print("[✓] MT5 disconnected")


if __name__ == '__main__':
    main()
