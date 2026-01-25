"""
LIVE Trading Bot - Scans real-time MT5 market data and sends Telegram signals.
Uses the optimized EMA/ADX/ATR strategy with parameters from best_settings.json.

Usage:
  python main.py              # default: scan every 5 seconds
  python main.py --loop 10    # scan every 10 seconds
  python main.py --once       # single scan and exit

Telegram Commands:
  /risk 2.5    - Set risk to 2.5% per trade
  /eurusd      - Toggle EURUSD on/off
  /status      - Show current config
"""
import argparse
import csv
import json
import os
import time
import threading
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

import MetaTrader5 as mt5

from strategy import get_instrument_settings
from telegram_bot import send_telegram_message, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID

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


def shutdown_mt5():
    """Shutdown MT5 connection."""
    mt5.shutdown()


def get_broker_symbol(symbol: str) -> str:
    """Get the broker-specific symbol name."""
    suffix = BROKER_SUFFIX.get(symbol, '')
    return f"{symbol}{suffix}"


def get_symbol_info(symbol: str) -> dict | None:
    """Get real symbol info from MT5 (tick size, digits, spread, etc.)."""
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
    broker_sym = get_broker_symbol(symbol)
    
    # Ensure symbol is visible in Market Watch
    if not mt5.symbol_select(broker_sym, True):
        print(f"[!] Failed to select {broker_sym}")
        return None
    
    rates = mt5.copy_rates_from_pos(broker_sym, timeframe, 0, bars)
    if rates is None or len(rates) < 100:
        print(f"[!] Failed to fetch candles for {broker_sym}: {mt5.last_error()}")
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
    """Generate trading signal based on latest bar."""
    if not params:
        return None
    
    fast = int(params.get('EMA_Fast', 10))
    slow = int(params.get('EMA_Slow', 50))
    adx_th = float(params.get('ADX', 20))
    atr_mult = float(params.get('ATR_Mult', 1.5))

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
                        "🤖 <b>Ultima Trading Bot</b>\n"
                        "━━━━━━━━━━━━━━━━\n\n"
                        "<b>📊 Trading Commands</b>\n"
                        "/risk [0.01-100] - Set risk % per trade\n"
                        "   Example: /risk 2.5\n"
                        "/positions - View open positions\n"
                        "/status - Bot status & settings\n\n"
                        "<b>🔧 Symbol Controls</b>\n"
                        "/eurusd - Toggle EUR/USD\n"
                        "/gbpusd - Toggle GBP/USD\n"
                        "/usdjpy - Toggle USD/JPY\n"
                        "/xauusd - Toggle XAU/USD (Gold)\n"
                        "/gbpjpy - Toggle GBP/JPY\n"
                        "/btcusd - Toggle BTC/USD\n"
                        "/nas100 - Toggle NAS100\n\n"
                        "<b>ℹ️ Other</b>\n"
                        "/help - Show this message\n\n"
                        "━━━━━━━━━━━━━━━━\n"
                        "💡 Bot scans markets every 30s\n"
                        "📊 Uses EMA/ADX/ATR strategy\n"
                        "🎯 2:1 Risk-Reward ratio"
                    )
                    send_telegram_message(help_msg)
                
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


def scan_markets(cfg: dict, verbose: bool = False):
    """Scan all enabled symbols for signals and execute trades."""
    global last_signals, tracked_positions
    
    enabled = set(cfg.get('enabled_symbols', SYMBOLS))
    risk = cfg.get('risk_percent', DEFAULT_RISK)
    
    # First, check for closed positions (SL/TP hit)
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
    
    status = []  # Collect status for single-line output
    
    for symbol in SYMBOLS:
        if symbol not in enabled:
            continue
        
        # Get REAL symbol info from MT5
        sym_info = get_symbol_info(symbol)
        if not sym_info:
            status.append(f"{symbol}:ERR")
            continue
        
        # Check if we already have an open position for this symbol
        existing_pos = get_position_for_symbol(symbol)
        
        # Get strategy params from best_settings.json
        params = get_instrument_settings(symbol)
        if not params:
            status.append(f"{symbol}:NOCFG")
            continue
        
        # Fetch LIVE candles
        df = fetch_live_candles(symbol)
        if df is None or len(df) < 100:
            status.append(f"{symbol}:NODATA")
            continue
        
        # Generate signal
        signal = generate_signal(df, params, symbol, sym_info)
        
        # If we have a position, check if signal reversed (close condition)
        if existing_pos:
            pl = existing_pos['profit']
            if signal and signal['direction'] != existing_pos['direction']:
                # Signal reversed - close existing position
                print(f"\n>>> {symbol}: REVERSAL {existing_pos['direction']} -> {signal['direction']} | Closing...")
                send_telegram_message(
                    f"🔄 <b>Closing {existing_pos['direction']} {symbol}</b>\n"
                    f"Signal reversed to {signal['direction']}\n"
                    f"P/L: ${pl:.2f}"
                )
                if close_position(existing_pos):
                    existing_pos = None
                else:
                    status.append(f"{symbol}:CLOSEFAIL")
                    continue
            else:
                # Holding position
                dir_char = '▲' if existing_pos['direction'] == 'BUY' else '▼'
                pl_str = f"+${pl:.0f}" if pl >= 0 else f"-${abs(pl):.0f}"
                status.append(f"{symbol}:{dir_char}{pl_str}")
                continue
        
        # No existing position - check for new signal
        if signal:
            sig_key = f"{symbol}_{signal['direction']}"
            last_time = last_signals.get(sig_key)
            if last_time and (datetime.now(timezone.utc) - last_time).seconds < 60:
                status.append(f"{symbol}:WAIT")
                continue
            
            # NEW SIGNAL - this is important, print it
            dir_char = '▲' if signal['direction'] == 'BUY' else '▼'
            print(f"\n>>> NEW SIGNAL: {dir_char} {signal['direction']} {symbol} @ {signal['entry']} | TP:{signal['tp']} SL:{signal['stop']}")
            
            # Don't send signal alert - will send when position actually opens
            success = open_position_with_retry(signal, sym_info, risk)
            
            if success:
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
                    tracked_positions[symbol] = {
                        'ticket': pos.get('ticket'),
                        'direction': signal['direction'],
                        'entry_price': signal['entry'],
                        'sl': signal['stop'],
                        'tp': signal['tp'],
                        'lot_size': signal.get('lot_size', 0),
                        'open_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    }
            else:
                status.append(f"{symbol}:FAIL")
                # Don't spam Telegram on failed orders
        else:
            status.append(f"{symbol}:-")
    
    # Single compact status line
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {' | '.join(status)}")


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
    
    # Telegram bot for commands
    tg = TelegramBot()
    if tg.is_configured():
        print("[✓] Telegram connected")
    else:
        print("[!] Telegram not configured")
    
    # Show existing open positions
    show_open_positions()
    
    print("=" * 60)

    try:
        if args.once:
            scan_markets(cfg)
        else:
            interval = max(5, args.loop)
            print(f"[+] Scanning every {interval}s (Ctrl+C to stop)\n")
            
            # Start Telegram polling in separate thread for instant response
            telegram_active = threading.Event()
            telegram_active.set()
            
            def telegram_loop():
                nonlocal cfg
                while telegram_active.is_set():
                    cfg = tg.poll_commands(cfg)
                    time.sleep(1)  # Check Telegram every second
            
            telegram_thread = threading.Thread(target=telegram_loop, daemon=True)
            telegram_thread.start()
            
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
    finally:
        shutdown_mt5()
        print("[✓] MT5 disconnected")


if __name__ == '__main__':
    main()
