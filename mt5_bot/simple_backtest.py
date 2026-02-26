# simple backtest

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import argparse
import sys
from typing import Dict, List, Tuple

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

# config

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'BTCUSD', 'NAS100']
BROKER_SUFFIX = {'EURUSD': '+', 'GBPUSD': '+', 'USDJPY': '+', 'GBPJPY': '+', 'XAUUSD': '+'}

# Strategy parameters (ICT / SMC tuned)
EMA_FAST = 9       # Fast EMA period — responsive trend
EMA_SLOW = 21      # Slow EMA period — structure trend
ADX_PERIOD = 14    # ADX calculation period
ADX_THRESHOLD = 25 # Minimum ADX to trade (Doc: <25 = consolidation)
ATR_PERIOD = 14    # ATR calculation period
ATR_MULTIPLIER = 1.0  # Stop loss = ATR × multiplier (tight OB-based SL)
RISK_REWARD = 2.0  # Take profit = Stop loss × risk-reward (ICT 2R target)

# Backtest settings
BACKTEST_DAYS = 180  # 6 months of data
INITIAL_BALANCE = 10000  # Starting with $10,000
RISK_PERCENT = 2.0  # Risk 2% per trade
MAX_SPREAD = 3.0  # Maximum spread in pips (skip if wider)

# Results directory
RESULTS_DIR = Path(__file__).parent / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)


# helpers

def get_broker_symbol(symbol: str) -> str:
    """Add broker suffix to symbol name"""
    suffix = BROKER_SUFFIX.get(symbol, '')
    return f"{symbol}{suffix}"


def resolve_broker_symbol(symbol: str) -> str | None:
    """Resolve symbol name across broker naming variants (suffix/prefix variants)."""
    base = symbol.upper().strip()
    candidates = []

    def add_candidate(name: str):
        if name and name not in candidates:
            candidates.append(name)

    add_candidate(get_broker_symbol(base))
    add_candidate(base)

    for pattern in (f"{base}*", f"*{base}*", f"{base}.*"):
        try:
            matches = mt5.symbols_get(pattern) or []
        except Exception:
            matches = []
        for sym in matches:
            add_candidate(getattr(sym, "name", ""))

    for cand in candidates:
        try:
            if mt5.symbol_select(cand, True):
                info = mt5.symbol_info(cand)
                if info is not None and getattr(info, "visible", True):
                    return cand
        except Exception:
            continue

    return None


def initialize_mt5() -> bool:
    """Connect to MT5"""
    if not mt5.initialize():
        print(f"❌ MT5 initialization failed: {mt5.last_error()}")
        return False
    print(f"✅ MT5 connected: {mt5.terminal_info().name}")
    return True


def fetch_data(symbol: str, days: int = BACKTEST_DAYS) -> pd.DataFrame:
    """
    Download historical price data from MT5
    
    Returns DataFrame with columns: Time, Open, High, Low, Close, Volume
    """
    broker_sym = resolve_broker_symbol(symbol)

    if not broker_sym:
        print(f"⚠️ Could not resolve broker symbol for {symbol}")
        return pd.DataFrame()
    
    # Download data
    utc_now = datetime.now(timezone.utc)
    utc_from = utc_now - timedelta(days=days)
    
    rates = mt5.copy_rates_range(broker_sym, mt5.TIMEFRAME_M15, utc_from, utc_now)
    
    if rates is None or len(rates) == 0:
        print(f"⚠️ No data for {symbol}")
        return pd.DataFrame()
    
    # Convert to DataFrame
    df = pd.DataFrame(rates)
    df['Time'] = pd.to_datetime(df['time'], unit='s')
    df = df[['Time', 'open', 'high', 'low', 'close', 'tick_volume']]
    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
    
    print(f"📊 {symbol}: Downloaded {len(df)} bars")
    return df


# indicators

def calculate_ema(prices: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Exponential Moving Average
    
    EMA gives more weight to recent prices:
    EMA = Price(today) × k + EMA(yesterday) × (1 - k)
    where k = 2 / (period + 1)
    """
    ema = np.zeros_like(prices, dtype=float)
    ema[0] = prices[0]
    multiplier = 2 / (period + 1)
    
    for i in range(1, len(prices)):
        ema[i] = (prices[i] - ema[i-1]) * multiplier + ema[i-1]
    
    return ema


def calculate_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Average True Range
    
    ATR measures volatility:
    - True Range = max(High-Low, |High-PrevClose|, |Low-PrevClose|)
    - ATR = EMA of True Range
    """
    # True Range calculation
    high_low = high - low
    high_close = np.abs(high - np.roll(close, 1))
    low_close = np.abs(low - np.roll(close, 1))
    
    true_range = np.maximum(high_low, np.maximum(high_close, low_close))
    true_range[0] = high_low[0]  # First value
    
    # ATR is EMA of True Range
    atr = calculate_ema(true_range, period)
    return atr


def calculate_adx(high: np.ndarray, low: np.ndarray, close: np.ndarray, period: int) -> np.ndarray:
    """
    Calculate Average Directional Index
    
    ADX measures trend strength (0-100):
    - < 20: Weak/no trend
    - 20-30: Emerging trend
    - > 30: Strong trend
    """
    # Directional movements
    plus_dm = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    
    # Keep only the larger move
    plus_dm = np.where(plus_dm > minus_dm, plus_dm, 0)
    minus_dm = np.where(minus_dm > plus_dm, minus_dm, 0)
    
    # ATR for normalization
    atr = calculate_atr(high, low, close, period)
    atr_safe = np.where(atr > 0, atr, 1e-8)
    
    # Directional indicators
    plus_di = 100 * calculate_ema(plus_dm, period) / atr_safe
    minus_di = 100 * calculate_ema(minus_dm, period) / atr_safe
    
    # DX (Directional Index)
    dx = 100 * np.abs(plus_di - minus_di) / np.where((plus_di + minus_di) != 0, (plus_di + minus_di), 1e-8)
    
    # ADX is EMA of DX
    adx = calculate_ema(dx, period)
    
    return adx


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add all technical indicators to the dataframe
    
    Adds: EMA_Fast, EMA_Slow, ATR, ADX
    """
    df = df.copy()
    
    close = df['Close'].values
    high = df['High'].values
    low = df['Low'].values
    
    # Calculate indicators
    df['EMA_Fast'] = calculate_ema(close, EMA_FAST)
    df['EMA_Slow'] = calculate_ema(close, EMA_SLOW)
    df['ATR'] = calculate_atr(high, low, close, ATR_PERIOD)
    df['ADX'] = calculate_adx(high, low, close, ADX_PERIOD)
    
    return df


# strategy

def generate_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate BUY/SELL signals based on strategy
    
    Rules:
    - BUY: Fast EMA > Slow EMA AND ADX > threshold
    - SELL: Fast EMA < Slow EMA AND ADX > threshold
    - No signal: ADX too low (choppy market)
    """
    df = df.copy()
    
    # Initialize signal column
    df['Signal'] = 0  # 0 = no signal, 1 = BUY, -1 = SELL
    
    for i in range(EMA_SLOW, len(df)):
        ema_fast = df['EMA_Fast'].iloc[i]
        ema_slow = df['EMA_Slow'].iloc[i]
        adx = df['ADX'].iloc[i]
        
        # Check if ADX is strong enough
        if adx < ADX_THRESHOLD:
            continue  # Skip weak trends
        
        # Check for crossover
        if ema_fast > ema_slow:
            df.loc[df.index[i], 'Signal'] = 1  # BUY
        elif ema_fast < ema_slow:
            df.loc[df.index[i], 'Signal'] = -1  # SELL
    
    return df


def _find_swing_points(highs, lows, lookback=5):
    """Detect swing highs and swing lows using lookback window.

    A swing high at i means highs[i] is the max of highs[i-lookback : i+lookback+1].
    Uses only completed bars (no future leak — we lag by *lookback* bars).
    """
    n = len(highs)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)

    for i in range(lookback, n - lookback):
        window_h = highs[i - lookback: i + lookback + 1]
        window_l = lows[i - lookback: i + lookback + 1]
        if highs[i] == np.max(window_h):
            swing_highs[i] = highs[i]
        if lows[i] == np.min(window_l):
            swing_lows[i] = lows[i]

    return swing_highs, swing_lows


def generate_signals_ict_smc(df: pd.DataFrame) -> pd.DataFrame:
    """ICT / SMC strategy — multi-bar state-machine with confluence scoring.

    HARD requirements (must ALL be true):
      • EMA trend alignment
      • ADX ≥ threshold
      • Valid OB zone from a BOS event
      • Price pulls back into the OB zone
      • Confirmation candle (close in trade direction)

    CONFLUENCE SCORING (soft — need total ≥ 3):
      +1  Displacement BOS (strong BOS candle)
      +1  FVG present in the impulse
      +1  Liquidity sweep before the move
      +1  In session kill zone (London 07-10 / NY 13-16)
      +1  Premium/Discount alignment
      +1  Rejection wick on entry candle
      +1  Strong OB candle (body > avg)
    """
    df = df.copy()
    df['Signal'] = 0

    closes = df['Close'].to_numpy()
    opens  = df['Open'].to_numpy()
    highs  = df['High'].to_numpy()
    lows   = df['Low'].to_numpy()
    ema_fast = df['EMA_Fast'].to_numpy()
    ema_slow = df['EMA_Slow'].to_numpy()
    adx = df['ADX'].to_numpy()

    times = pd.to_datetime(df['Time'])
    hours = times.dt.hour.to_numpy()

    n = len(df)
    SWING_LB     = 5
    OB_SCAN      = 20
    BOS_VALIDITY  = 50     # OB zone valid for 50 bars
    ADX_MIN      = ADX_THRESHOLD  # Use global threshold (Doc: 25)
    COOLDOWN     = 3
    RANGE_LB     = 40      # lookback for premium/discount
    MIN_SCORE    = 3       # minimum confluence score to enter

    # --- pre-compute helpers ------------------------------------
    swing_highs, swing_lows = _find_swing_points(highs, lows, SWING_LB)

    bodies = np.abs(closes - opens)
    avg_body = np.zeros(n)
    for i in range(20, n):
        avg_body[i] = np.mean(bodies[i - 20:i])

    # --- state: tracked OB zones --------------------------------
    # We can track multiple OB zones (up to 3 per direction)
    bull_obs = []  # list of dicts: {hi, lo, bar, score, strong_ob}
    bear_obs = []

    last_signal_bar = -999
    start = max(EMA_SLOW, 2 * SWING_LB + 5, RANGE_LB + 2)

    for i in range(start, n):
        # ---- 0. Basic filter ------------------------------------
        if not np.isfinite(adx[i]) or adx[i] < ADX_MIN:
            continue

        bullish_trend = ema_fast[i] > ema_slow[i]
        bearish_trend = ema_fast[i] < ema_slow[i]

        # ---- 1. Detect BOS events --------------------------------
        recent_sh = np.nan
        recent_sl = np.nan
        for k in range(i - SWING_LB - 1, max(start - 1, SWING_LB) - 1, -1):
            if np.isnan(recent_sh) and not np.isnan(swing_highs[k]):
                recent_sh = swing_highs[k]
            if np.isnan(recent_sl) and not np.isnan(swing_lows[k]):
                recent_sl = swing_lows[k]
            if not np.isnan(recent_sh) and not np.isnan(recent_sl):
                break

        # --- Bullish BOS ---
        if bullish_trend and not np.isnan(recent_sh):
            for k in range(max(start, i - SWING_LB), i):
                if closes[k] > recent_sh:
                    # Displacement: BOS candle body >= 150% of avg (Doc: 150%)
                    bos_disp = bodies[k] > avg_body[k] * 1.5 if avg_body[k] > 0 else False

                    # Find OB: last bearish candle before the impulse
                    ob_hi = np.nan
                    ob_lo = np.nan
                    ob_strong = False
                    for ob_idx in range(k, max(k - OB_SCAN, 0), -1):
                        if closes[ob_idx] < opens[ob_idx]:  # bearish
                            ob_hi = highs[ob_idx]
                            ob_lo = min(opens[ob_idx], closes[ob_idx])
                            ob_strong = bodies[ob_idx] > avg_body[ob_idx] * 0.7 if avg_body[ob_idx] > 0 else False
                            break

                    if np.isnan(ob_hi):
                        break

                    # BOS-level confluences
                    score = 0
                    if bos_disp:
                        score += 1
                    # FVG
                    for f in range(max(start, k - 8), k + 1):
                        if f >= 2 and lows[f] > highs[f - 2]:
                            score += 1
                            break
                    # Sweep before move
                    if k >= 2 and lows[k - 1] < lows[k - 2] and closes[k - 1] > lows[k - 2]:
                        score += 1
                    if ob_strong:
                        score += 1

                    bull_obs.append({'hi': ob_hi, 'lo': ob_lo, 'bar': i,
                                     'score': score, 'strong_ob': ob_strong})
                    # Keep max 3 active zones
                    if len(bull_obs) > 3:
                        bull_obs.pop(0)
                    break

        # --- Bearish BOS ---
        if bearish_trend and not np.isnan(recent_sl):
            for k in range(max(start, i - SWING_LB), i):
                if closes[k] < recent_sl:
                    bos_disp = bodies[k] > avg_body[k] * 1.5 if avg_body[k] > 0 else False

                    ob_hi = np.nan
                    ob_lo = np.nan
                    ob_strong = False
                    for ob_idx in range(k, max(k - OB_SCAN, 0), -1):
                        if closes[ob_idx] > opens[ob_idx]:  # bullish
                            ob_lo = lows[ob_idx]
                            ob_hi = max(opens[ob_idx], closes[ob_idx])
                            ob_strong = bodies[ob_idx] > avg_body[ob_idx] * 0.7 if avg_body[ob_idx] > 0 else False
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

                    bear_obs.append({'hi': ob_hi, 'lo': ob_lo, 'bar': i,
                                     'score': score, 'strong_ob': ob_strong})
                    if len(bear_obs) > 3:
                        bear_obs.pop(0)
                    break

        # ---- 2. Expire old zones ---------------------------------
        bull_obs = [z for z in bull_obs if (i - z['bar']) <= BOS_VALIDITY]
        bear_obs = [z for z in bear_obs if (i - z['bar']) <= BOS_VALIDITY]

        # ---- 3. Pullback entry -----------------------------------
        if i - last_signal_bar < COOLDOWN:
            continue

        # Entry-level confluence (computed once)
        h = hours[i]
        in_session = (7 <= h <= 10) or (13 <= h <= 16)

        range_high = np.max(highs[i - RANGE_LB:i])
        range_low  = np.min(lows[i - RANGE_LB:i])
        range_mid  = (range_high + range_low) / 2.0

        body_i = abs(closes[i] - opens[i])

        # --- BULLISH ENTRY ---
        if bullish_trend:
            for z_idx, z in enumerate(bull_obs):
                if (i - z['bar']) < 1:
                    continue
                touched = lows[i] <= z['hi'] and closes[i] >= z['lo']
                bull_candle = closes[i] > opens[i]
                if not (touched and bull_candle):
                    continue

                # Build entry score on top of BOS score
                entry_score = z['score']
                if in_session:
                    entry_score += 1
                if closes[i] < range_mid:   # discount
                    entry_score += 1
                # Rejection wick
                lower_wick = min(opens[i], closes[i]) - lows[i]
                if body_i > 0 and lower_wick > body_i * 0.3:
                    entry_score += 1

                if entry_score >= MIN_SCORE:
                    df.at[df.index[i], 'Signal'] = 1
                    last_signal_bar = i
                    bull_obs.pop(z_idx)
                    break

        # --- BEARISH ENTRY ---
        if bearish_trend and last_signal_bar != i:
            for z_idx, z in enumerate(bear_obs):
                if (i - z['bar']) < 1:
                    continue
                touched = highs[i] >= z['lo'] and closes[i] <= z['hi']
                bear_candle = closes[i] < opens[i]
                if not (touched and bear_candle):
                    continue

                entry_score = z['score']
                if in_session:
                    entry_score += 1
                if closes[i] > range_mid:   # premium
                    entry_score += 1
                upper_wick = highs[i] - max(opens[i], closes[i])
                if body_i > 0 and upper_wick > body_i * 0.3:
                    entry_score += 1

                if entry_score >= MIN_SCORE:
                    df.at[df.index[i], 'Signal'] = -1
                    last_signal_bar = i
                    bear_obs.pop(z_idx)
                    break

    return df


# backtest engine

def run_backtest(df: pd.DataFrame, symbol: str) -> Dict:
    """
    Simulate trading on historical data
    
    For each signal:
    1. Enter trade at current close price
    2. Set Stop Loss = Entry ± ATR × multiplier
    3. Set Take Profit = Entry ± ATR × multiplier × 2
    4. Simulate until SL or TP hit
    5. Record profit/loss
    
    Returns dictionary with trades and metrics
    """
    df = add_indicators(df)
    df = generate_signals_ict_smc(df)
    
    trades = []
    balance = INITIAL_BALANCE
    equity_curve = [INITIAL_BALANCE]
    
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    direction = 0
    entry_idx = 0
    initial_risk = 0  # distance from entry to original SL
    trailing_stage = 0  # 0=none, 1=BE, 2=trail-1R

    for i in range(len(df) - 1):
        row = df.iloc[i]

        if not in_trade:
            # Look for entry signal
            if row['Signal'] != 0:
                direction = row['Signal']
                entry_price = df.iloc[i + 1]['Open']
                atr = row['ATR']

                if direction == 1:  # BUY
                    stop_loss = entry_price - (atr * ATR_MULTIPLIER)
                    take_profit = entry_price + (atr * ATR_MULTIPLIER * RISK_REWARD)
                else:  # SELL
                    stop_loss = entry_price + (atr * ATR_MULTIPLIER)
                    take_profit = entry_price - (atr * ATR_MULTIPLIER * RISK_REWARD)

                initial_risk = abs(entry_price - stop_loss)
                trailing_stage = 0
                in_trade = True
                entry_idx = i + 1

        else:
            # Check for exit
            current_high = row['High']
            current_low = row['Low']
            current_close = row['Close']

            # --- Break-even & trailing SL (per documentation) ---
            # Rule 1: At 1R profit → SL moves to entry price (break-even)
            # Rule 2: At 50% of target profit → SL moves to 25% profit level
            if initial_risk > 0:
                target_profit = abs(take_profit - entry_price)
                if direction == 1:  # BUY
                    unrealized = current_high - entry_price
                    if trailing_stage == 0 and unrealized >= initial_risk:
                        # Reached 1R: move SL to entry price (break-even)
                        stop_loss = entry_price
                        trailing_stage = 1
                    if trailing_stage == 1 and target_profit > 0 and unrealized >= target_profit * 0.5:
                        # Reached 50% of target: move SL to 25% profit level
                        stop_loss = entry_price + target_profit * 0.25
                        trailing_stage = 2
                else:  # SELL
                    unrealized = entry_price - current_low
                    if trailing_stage == 0 and unrealized >= initial_risk:
                        stop_loss = entry_price
                        trailing_stage = 1
                    if trailing_stage == 1 and target_profit > 0 and unrealized >= target_profit * 0.5:
                        stop_loss = entry_price - target_profit * 0.25
                        trailing_stage = 2

            exit_triggered = False
            exit_price = 0
            exit_reason = ""

            if direction == 1:  # BUY position
                if current_low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL" if trailing_stage == 0 else f"BE/Trail"
                    exit_triggered = True
                elif current_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_triggered = True

            else:  # SELL position
                if current_high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL" if trailing_stage == 0 else f"BE/Trail"
                    exit_triggered = True
                elif current_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_triggered = True

            # Time-based exit (max 80 bars for ICT — multi-session holds)
            if not exit_triggered and (i - entry_idx) > 80:
                exit_price = current_close
                exit_reason = "Time"
                exit_triggered = True
            
            if exit_triggered:
                # Calculate profit
                if direction == 1:
                    pips = exit_price - entry_price
                else:
                    pips = entry_price - exit_price
                
                # Estimate profit in dollars (simplified)
                # Actual calculation would depend on lot size and pip value
                risk_amount = balance * (RISK_PERCENT / 100)
                stop_distance = abs(entry_price - stop_loss)
                profit = (pips / stop_distance) * risk_amount if stop_distance > 0 else 0
                
                balance += profit
                equity_curve.append(balance)
                
                # Record trade
                trades.append({
                    'entry_time': df.iloc[entry_idx]['Time'],
                    'exit_time': row['Time'],
                    'direction': 'BUY' if direction == 1 else 'SELL',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'stop_loss': stop_loss,
                    'take_profit': take_profit,
                    'profit': profit,
                    'exit_reason': exit_reason,
                    'bars_held': i - entry_idx
                })
                
                in_trade = False
    
    # Calculate metrics
    if len(trades) == 0:
        return {'symbol': symbol, 'trades': [], 'metrics': {}}
    
    profits = [t['profit'] for t in trades]
    wins = [p for p in profits if p > 0]
    losses = [p for p in profits if p <= 0]
    
    metrics = {
        'total_trades': len(trades),
        'wins': len(wins),
        'losses': len(losses),
        'win_rate': len(wins) / len(trades) * 100,
        'total_profit': sum(profits),
        'avg_profit': np.mean(profits),
        'avg_win': np.mean(wins) if wins else 0,
        'avg_loss': np.mean(losses) if losses else 0,
        'best_trade': max(profits),
        'worst_trade': min(profits),
        'profit_factor': sum(wins) / abs(sum(losses)) if losses and sum(losses) != 0 else 99,
        'max_drawdown': calculate_max_drawdown(equity_curve),
        'final_balance': balance,
        'return_pct': (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    }
    
    return {
        'symbol': symbol,
        'trades': trades,
        'metrics': metrics,
        'equity_curve': equity_curve
    }


def calculate_max_drawdown(equity_curve: List[float]) -> float:
    """
    Calculate maximum drawdown
    
    Drawdown = largest peak-to-valley drop in equity
    """
    if len(equity_curve) < 2:
        return 0
    
    peak = equity_curve[0]
    max_dd = 0
    
    for value in equity_curve:
        if value > peak:
            peak = value
        dd = peak - value
        if dd > max_dd:
            max_dd = dd
    
    return max_dd


def run_split_backtest(df: pd.DataFrame, symbol: str, split_ratio: float = 0.7) -> Dict:
    """
    In-sample / out-of-sample split backtest.
    Uses the first portion as "train" and the last portion as "test".
    """
    if split_ratio <= 0 or split_ratio >= 1:
        split_ratio = 0.7

    split_idx = int(len(df) * split_ratio)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_result = run_backtest(train_df, symbol)
    test_result = run_backtest(test_df, symbol)

    return {
        "symbol": symbol,
        "mode": "split",
        "split_ratio": split_ratio,
        "train": {
            "metrics": train_result.get("metrics", {}),
            "trades": train_result.get("trades", [])
        },
        "test": {
            "metrics": test_result.get("metrics", {}),
            "trades": test_result.get("trades", [])
        }
    }


def monte_carlo_analysis(trades: List[Dict], iterations: int = 200) -> Dict:
    """
    Monte Carlo analysis by shuffling trade outcomes.
    Returns distribution of ending balances.
    """
    if not trades:
        return {
            "iterations": iterations,
            "ending_balances": [],
            "p10": 0,
            "p50": 0,
            "p90": 0,
            "max_drawdown_avg": 0
        }

    profits = [t.get("profit", 0) for t in trades]
    ending_balances = []
    drawdowns = []

    for _ in range(iterations):
        shuffled = profits.copy()
        np.random.shuffle(shuffled)

        balance = INITIAL_BALANCE
        equity_curve = [balance]
        for p in shuffled:
            balance += p
            equity_curve.append(balance)

        ending_balances.append(balance)
        drawdowns.append(calculate_max_drawdown(equity_curve))

    return {
        "iterations": iterations,
        "ending_balances": ending_balances,
        "p10": float(np.percentile(ending_balances, 10)),
        "p50": float(np.percentile(ending_balances, 50)),
        "p90": float(np.percentile(ending_balances, 90)),
        "max_drawdown_avg": float(np.mean(drawdowns))
    }


def build_output_payload(symbol: str, mode: str, result: Dict) -> Dict:
    """Normalize output payload for saving."""
    payload = {
        "symbol": symbol,
        "mode": mode,
        "timestamp": datetime.now().isoformat()
    }
    payload.update(result)
    return payload


def save_output(payload: Dict, output_path: Path) -> None:
    """Save output JSON to file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(payload, f, indent=2, default=str)


# walk-forward analysis

def walk_forward_analysis(symbol: str, total_days: int = 180, train_days: int = 60, test_days: int = 30) -> Dict:
    """
    Walk-forward analysis to detect overfitting
    
    Process:
    1. Train on first 60 days → Test on next 30 days
    2. Train on next 60 days → Test on next 30 days
    3. Repeat until data exhausted
    
    If strategy works across ALL test periods = NOT OVERFITTED
    """
    print(f"\n{'='*60}")
    print(f"WALK-FORWARD ANALYSIS: {symbol}")
    print(f"{'='*60}")
    
    results = []
    
    # Calculate number of windows
    num_windows = (total_days - train_days) // test_days
    
    for window in range(num_windows):
        start_day = window * test_days
        train_end = start_day + train_days
        test_end = train_end + test_days
        
        print(f"\n📊 Window {window + 1}/{num_windows}")
        print(f"   Training: Day {start_day} to {train_end}")
        print(f"   Testing: Day {train_end} to {test_end}")
        
        # Fetch data for this window
        df = fetch_data(symbol, days=test_end)
        if len(df) < 100:
            continue
        
        # Split into train/test
        train_df = df.iloc[:int(len(df) * train_end / test_end)]
        test_df = df.iloc[int(len(df) * train_end / test_end):]
        
        # Run backtest on test period
        result = run_backtest(test_df, symbol)
        
        if result['metrics']:
            metrics = result['metrics']
            print(f"   ✅ Trades: {metrics['total_trades']}, WR: {metrics['win_rate']:.1f}%, Profit: ${metrics['total_profit']:.2f}")
            results.append(metrics)
        else:
            print(f"   ⚠️ No trades in this period")
    
    # Analyze consistency
    if results:
        avg_profit = np.mean([r['total_profit'] for r in results])
        profitable_periods = sum(1 for r in results if r['total_profit'] > 0)
        consistency = profitable_periods / len(results) * 100
        
        print(f"\n{'='*60}")
        print(f"WALK-FORWARD SUMMARY:")
        print(f"   Total Periods: {len(results)}")
        print(f"   Profitable Periods: {profitable_periods} ({consistency:.0f}%)")
        print(f"   Average Profit per Period: ${avg_profit:.2f}")
        
        if consistency >= 70:
            print(f"   ✅ ROBUST STRATEGY (not overfitted)")
        elif consistency >= 50:
            print(f"   ⚠️ MODERATELY ROBUST (some periods struggle)")
        else:
            print(f"   ❌ LIKELY OVERFITTED (inconsistent across periods)")
        
        print(f"{'='*60}\n")
        
        return {
            "symbol": symbol,
            "mode": "walk_forward",
            "total_periods": len(results),
            "profitable_periods": profitable_periods,
            "consistency": consistency,
            "average_profit_per_period": avg_profit,
            "periods": results
        }
    
    return {
        "symbol": symbol,
        "mode": "walk_forward",
        "total_periods": 0,
        "profitable_periods": 0,
        "consistency": 0,
        "average_profit_per_period": 0,
        "periods": []
    }


# main

def main():
    parser = argparse.ArgumentParser(description='Simple Backtest - ICT/SMC Strategy')
    parser.add_argument('--symbol', type=str, help='Test single symbol')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    parser.add_argument('--mode', type=str, choices=['standard', 'walk_forward', 'split', 'monte_carlo'], default='standard')
    parser.add_argument('--split-ratio', type=float, default=0.7, help='Train/Test split ratio for split mode')
    parser.add_argument('--mc-iterations', type=int, default=200, help='Monte Carlo iterations')
    parser.add_argument('--run-id', type=str, default=None, help='Optional run ID for output file')
    parser.add_argument('--output', type=str, default=None, help='Output JSON file path')
    args = parser.parse_args()
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    # Determine which symbols to test
    test_symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    
    all_results = {}
    requested_mode = 'walk_forward' if args.walk_forward else args.mode
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'='*60}")
        
        mode = requested_mode

        if mode == 'walk_forward':
            wf_result = walk_forward_analysis(symbol)
            all_results[symbol] = wf_result
            payload = build_output_payload(symbol, mode, wf_result)
        else:
            df = fetch_data(symbol)
            if len(df) < 100:
                continue

            if mode == 'split':
                split_result = run_split_backtest(df, symbol, split_ratio=args.split_ratio)
                all_results[symbol] = split_result
                payload = build_output_payload(symbol, mode, split_result)
            else:
                standard_result = run_backtest(df, symbol)
                all_results[symbol] = standard_result

                if mode == 'monte_carlo':
                    mc = monte_carlo_analysis(standard_result.get('trades', []), iterations=args.mc_iterations)
                    payload = build_output_payload(symbol, mode, {"standard": standard_result, "monte_carlo": mc})
                else:
                    payload = build_output_payload(symbol, mode, standard_result)

            # Display results for standard backtest
            if mode == 'standard' and standard_result.get('metrics'):
                m = standard_result['metrics']
                print(f"\n📊 RESULTS:")
                print(f"   Total Trades: {m['total_trades']}")
                print(f"   Wins: {m['wins']} ({m['win_rate']:.1f}%)")
                print(f"   Losses: {m['losses']}")
                print(f"   Total Profit: ${m['total_profit']:.2f}")
                print(f"   Avg Profit/Trade: ${m['avg_profit']:.2f}")
                print(f"   Best Trade: ${m['best_trade']:.2f}")
                print(f"   Worst Trade: ${m['worst_trade']:.2f}")
                print(f"   Profit Factor: {m['profit_factor']:.2f}")
                print(f"   Max Drawdown: ${m['max_drawdown']:.2f}")
                print(f"   Final Balance: ${m['final_balance']:.2f}")
                print(f"   Return: {m['return_pct']:.2f}%")

        # Save output file
        if args.output:
            output_file = Path(args.output)
        else:
            run_id = args.run_id or f"{symbol}_{mode}"
            output_file = RESULTS_DIR / f"{run_id}.json"

        save_output(payload, output_file)
        print(f"\n💾 Results saved to: {output_file}")
    
    # Summary table
    if all_results and (requested_mode == 'standard'):
        print(f"\n{'='*60}")
        print("SUMMARY TABLE")
        print(f"{'='*60}")
        print(f"{'Symbol':<10} {'Trades':<8} {'Win %':<8} {'Profit':<12} {'PF':<6} {'Status'}")
        print(f"{'-'*60}")
        
        for symbol, result in all_results.items():
            if result['metrics']:
                m = result['metrics']
                status = "✅" if m['total_profit'] > 0 else "❌"
                print(f"{symbol:<10} {m['total_trades']:<8} {m['win_rate']:<7.1f}% ${m['total_profit']:<11.2f} {m['profit_factor']:<5.2f} {status}")
        
        print(f"{'='*60}\n")
    
    mt5.shutdown()
    print("✅ Backtest complete!")


if __name__ == "__main__":
    main()
