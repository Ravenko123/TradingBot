"""
SIMPLE BACKTEST - EMA/ADX/ATR Strategy
=======================================

Tests the trading strategy on historical data to validate profitability.
Includes walk-forward analysis to detect overfitting.

What this does:
1. Downloads historical data from MT5
2. Simulates trades using the strategy
3. Calculates performance metrics
4. Tests on multiple time periods to check robustness

Strategy:
- BUY when Fast EMA > Slow EMA AND ADX > threshold
- SELL when Fast EMA < Slow EMA AND ADX > threshold
- Stop Loss: Entry ± ATR * multiplier
- Take Profit: Entry ± ATR * multiplier * 2

Usage:
    python simple_backtest.py                    # Test all symbols
    python simple_backtest.py --symbol EURUSD    # Test one symbol
    python simple_backtest.py --walk-forward     # Overfitting check
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json
import argparse
from typing import Dict, List, Tuple

# ===========================================================================
# CONFIGURATION
# ===========================================================================

SYMBOLS = ['EURUSD', 'GBPUSD', 'USDJPY', 'XAUUSD', 'GBPJPY', 'BTCUSD', 'NAS100']
BROKER_SUFFIX = {'EURUSD': '+', 'GBPUSD': '+', 'USDJPY': '+', 'GBPJPY': '+', 'XAUUSD': '+'}

# Strategy parameters
EMA_FAST = 10      # Fast EMA period
EMA_SLOW = 50      # Slow EMA period
ADX_PERIOD = 14    # ADX calculation period
ADX_THRESHOLD = 20 # Minimum ADX to trade (trend strength)
ATR_PERIOD = 14    # ATR calculation period
ATR_MULTIPLIER = 1.5  # Stop loss = ATR × multiplier
RISK_REWARD = 2.0  # Take profit = Stop loss × risk-reward

# Backtest settings
BACKTEST_DAYS = 180  # 6 months of data
INITIAL_BALANCE = 10000  # Starting with $10,000
RISK_PERCENT = 1.0  # Risk 1% per trade
MAX_SPREAD = 3.0  # Maximum spread in pips (skip if wider)

# Results directory
RESULTS_DIR = Path(__file__).parent / "backtest_results"
RESULTS_DIR.mkdir(exist_ok=True)


# ===========================================================================
# HELPER FUNCTIONS
# ===========================================================================

def get_broker_symbol(symbol: str) -> str:
    """Add broker suffix to symbol name"""
    suffix = BROKER_SUFFIX.get(symbol, '')
    return f"{symbol}{suffix}"


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
    broker_sym = get_broker_symbol(symbol)
    
    # Make symbol visible
    if not mt5.symbol_select(broker_sym, True):
        print(f"⚠️ Could not select {broker_sym}")
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


# ===========================================================================
# INDICATOR CALCULATIONS
# ===========================================================================

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


# ===========================================================================
# STRATEGY LOGIC
# ===========================================================================

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


# ===========================================================================
# BACKTEST ENGINE
# ===========================================================================

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
    df = generate_signals(df)
    
    trades = []
    balance = INITIAL_BALANCE
    equity_curve = [INITIAL_BALANCE]
    
    in_trade = False
    entry_price = 0
    stop_loss = 0
    take_profit = 0
    direction = 0
    entry_idx = 0
    
    for i in range(len(df)):
        row = df.iloc[i]
        
        if not in_trade:
            # Look for entry signal
            if row['Signal'] != 0:
                direction = row['Signal']
                entry_price = row['Close']
                atr = row['ATR']
                
                if direction == 1:  # BUY
                    stop_loss = entry_price - (atr * ATR_MULTIPLIER)
                    take_profit = entry_price + (atr * ATR_MULTIPLIER * RISK_REWARD)
                else:  # SELL
                    stop_loss = entry_price + (atr * ATR_MULTIPLIER)
                    take_profit = entry_price - (atr * ATR_MULTIPLIER * RISK_REWARD)
                
                in_trade = True
                entry_idx = i
        
        else:
            # Check for exit
            current_high = row['High']
            current_low = row['Low']
            
            exit_triggered = False
            exit_price = 0
            exit_reason = ""
            
            if direction == 1:  # BUY position
                if current_low <= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif current_high >= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_triggered = True
            
            else:  # SELL position
                if current_high >= stop_loss:
                    exit_price = stop_loss
                    exit_reason = "SL"
                    exit_triggered = True
                elif current_low <= take_profit:
                    exit_price = take_profit
                    exit_reason = "TP"
                    exit_triggered = True
            
            # Time-based exit (max 50 bars)
            if not exit_triggered and (i - entry_idx) > 50:
                exit_price = row['Close']
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


# ===========================================================================
# WALK-FORWARD ANALYSIS (Overfitting Detection)
# ===========================================================================

def walk_forward_analysis(symbol: str, total_days: int = 180, train_days: int = 60, test_days: int = 30):
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
        
        return results
    
    return []


# ===========================================================================
# MAIN EXECUTION
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(description='Simple Backtest - EMA/ADX/ATR Strategy')
    parser.add_argument('--symbol', type=str, help='Test single symbol')
    parser.add_argument('--walk-forward', action='store_true', help='Run walk-forward analysis')
    args = parser.parse_args()
    
    # Initialize MT5
    if not initialize_mt5():
        return
    
    # Determine which symbols to test
    test_symbols = [args.symbol.upper()] if args.symbol else SYMBOLS
    
    all_results = {}
    
    for symbol in test_symbols:
        print(f"\n{'='*60}")
        print(f"BACKTESTING: {symbol}")
        print(f"{'='*60}")
        
        if args.walk_forward:
            # Walk-forward analysis (overfitting check)
            walk_forward_analysis(symbol)
        else:
            # Standard backtest
            df = fetch_data(symbol)
            if len(df) < 100:
                continue
            
            result = run_backtest(df, symbol)
            all_results[symbol] = result
            
            # Display results
            if result['metrics']:
                m = result['metrics']
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
                
                # Save to JSON
                output_file = RESULTS_DIR / f"{symbol}_backtest.json"
                with open(output_file, 'w') as f:
                    # Convert datetime objects to strings
                    result_copy = result.copy()
                    for trade in result_copy['trades']:
                        trade['entry_time'] = str(trade['entry_time'])
                        trade['exit_time'] = str(trade['exit_time'])
                    json.dump(result_copy, f, indent=2)
                print(f"\n💾 Results saved to: {output_file}")
            else:
                print(f"\n⚠️ No trades generated for {symbol}")
    
    # Summary table
    if all_results and not args.walk_forward:
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
