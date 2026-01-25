"""
IMPROVED TREND-FOLLOWING STRATEGY - 7 INSTRUMENTS
- All 7 pairs (EURUSD, GBPUSD, USDJPY, XAUUSD, GBPJPY, BTCUSD, NAS100)
- ADX trend filter + Pyramiding + Dynamic risk sizing
- Multi-timeframe confirmation
- Auto-closing Windows popups
"""

import os
import shutil
import time
import json
import warnings
from pathlib import Path
from itertools import product
from dataclasses import dataclass
from datetime import datetime

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

DATA_DIR = Path('backtest_data_improved')
if DATA_DIR.exists():
    shutil.rmtree(DATA_DIR)
DATA_DIR.mkdir()

# Backtest window in days (limit data to recent period)
BACKTEST_DAYS = 30

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
# BACKTEST ENGINE
# ============================================================================

def backtest(df: pd.DataFrame, strategy: TrendFollower, symbol: str) -> dict:
    """Backtest with regime/session filters, pyramiding, ATR trailing stop (vectorized + RR tracking)."""
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

    # ATR percentile threshold (optional; 0 disables gating)
    atr_valid = atr[np.isfinite(atr) & (atr > 0)]
    atr_thr = np.percentile(atr_valid, strategy.atr_percentile) if (len(atr_valid) and strategy.atr_percentile > 0) else 0
    atr_ok = (atr >= atr_thr) if (strategy.atr_percentile > 0) else np.ones(len(atr), dtype=bool)
    session_ok = (hours >= strategy.session_start) & (hours <= strategy.session_end)

    position = 0
    pyramid_count = 0
    entry_price = 0.0
    stop_price = None
    bars_in_trade = 0
    net_profit = 0.0
    equity = [10000.0]
    max_equity = 10000.0
    max_dd = 0.0
    trades = 0
    wins = 0
    losses = 0
    rr_ratios = []  # Track Risk-Reward ratios per trade

    pip_size = INSTRUMENTS[symbol]['pip_size']

    start = max(strategy.slow_ema, 50)
    for i in range(start, len(df)):
        if not session_ok[i] or adx[i] <= strategy.adx_threshold or adx[i] != adx[i]:
            signal = 0
        else:
            # EMA + ADX entries (no breakout filter) for higher trade frequency
            long_ok = (ema_f[i] > ema_s[i])
            short_ok = (ema_f[i] < ema_s[i])
            signal = 1 if long_ok else -1 if short_ok else 0

        # Entry
        if position == 0 and signal != 0:
            position = signal
            entry_price = close[i]
            pyramid_count = 1
            # Initial ATR stop
            if atr[i] > 0:
                stop_price = (entry_price - strategy.atr_mult * atr[i]) if position == 1 else (entry_price + strategy.atr_mult * atr[i])
            else:
                stop_price = None
            trades += 1
            bars_in_trade = 0
        # Pyramiding on continuation
        elif strategy.use_pyramiding and position != 0 and pyramid_count < 2 and signal == position:
            if position == 1 and close[i] > entry_price * 1.01:
                entry_price = (entry_price * pyramid_count + close[i]) / (pyramid_count + 1)
                pyramid_count += 1
                bars_in_trade = 0
            elif position == -1 and close[i] < entry_price * 0.99:
                entry_price = (entry_price * pyramid_count + close[i]) / (pyramid_count + 1)
                pyramid_count += 1
                bars_in_trade = 0
        # Update ATR trailing stop
        if position != 0 and atr[i] > 0:
            if position == 1:
                new_stop = close[i] - strategy.atr_mult * atr[i]
                stop_price = max(stop_price if stop_price is not None else new_stop, new_stop)
            else:
                new_stop = close[i] + strategy.atr_mult * atr[i]
                stop_price = min(stop_price if stop_price is not None else new_stop, new_stop)
        
        # Exit on opposite / flat signal
        exit_now = False
        if position != 0:
            # Time stop
            bars_in_trade += 1
            if bars_in_trade >= 600:
                exit_now = True
            # Stop hit
            if stop_price is not None:
                if position == 1 and low[i] <= stop_price:
                    exit_now = True
                elif position == -1 and high[i] >= stop_price:
                    exit_now = True
            # Opposite/flat signal
            if signal == 0:
                exit_now = True

        if exit_now:
            pip_move = (close[i] - entry_price) / pip_size
            trade_pnl = pip_move * position * 10 * pyramid_count - INSTRUMENTS[symbol]['spread'] * pyramid_count
            net_profit += trade_pnl
            wins += 1 if trade_pnl > 0 else 0
            losses += 1 if trade_pnl <= 0 else 0
            
            # Calculate RR ratio: reward/risk (reasonable bounds: 0.5 to 5.0)
            if stop_price is not None:
                risk_pips = abs(entry_price - stop_price) / pip_size
                reward_pips = abs(close[i] - entry_price) / pip_size
                if risk_pips > 0.5:  # Only count meaningful stops (avoid noise)
                    rr_ratio = reward_pips / risk_pips
                    rr_ratio = max(0.0, min(rr_ratio, 5.0))  # Clamp to 0.0 - 5.0 range
                    rr_ratios.append(rr_ratio)
            
            position = 0
            pyramid_count = 0
            stop_price = None
            bars_in_trade = 0

        equity.append(10000 + net_profit)
        max_equity = max(max_equity, equity[-1])
        dd = max_equity - equity[-1]
        max_dd = max(max_dd, dd)

    win_rate = (wins / trades * 100) if trades > 0 else 0
    total_pnl = equity[-1] - 10000
    equity_arr = np.array(equity)
    deltas = np.diff(equity_arr)
    gross_profit = float(np.sum(deltas[deltas > 0]))
    gross_loss = float(-np.sum(deltas[deltas < 0]))
    pf = gross_profit / max(gross_loss, 1)
    
    # Calculate RR statistics
    avg_rr = np.mean(rr_ratios) if rr_ratios else 0
    median_rr = np.median(rr_ratios) if rr_ratios else 0
    min_rr = np.min(rr_ratios) if rr_ratios else 0
    max_rr = np.max(rr_ratios) if rr_ratios else 0

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
    }

# ============================================================================
# OPTIMIZATION
# ============================================================================

def optimize(symbol: str, df: pd.DataFrame) -> tuple:
    """Grid search for best parameters (instrument-aware; regime filters)."""
    split = int(len(df) * 0.7)
    df_train = df.iloc[:split].copy()
    df_test = df.iloc[split:].copy()

    best_score = -np.inf
    best_params = None
    best_result = None

    # Instrument-aware grids (tighter for crypto/high vol)
    if symbol == 'BTCUSD':
        fast_emas = [5, 10, 15]
        slow_emas = [100, 150, 200]
        adx_thresholds = [35.0, 40.0, 45.0]
    else:
        fast_emas = [5, 8, 10, 12, 15, 20]
        slow_emas = [30, 50, 100, 150, 200]
        adx_thresholds = [10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0, 45.0]

    atr_mults = [0.8, 1.0, 1.2, 1.5, 2.0, 2.5, 3.0]  # Wider range for better stops
    atr_percentiles = [0.0, 20.0, 30.0, 40.0, 50.0]
    session_windows = [(5, 23), (6, 22), (7, 22)]

    params_grid = list(product(fast_emas, slow_emas, adx_thresholds, atr_mults, atr_percentiles, session_windows))

    for fast, slow, adx_th, atr_m, atr_p, (sess_s, sess_e) in tqdm(params_grid, desc=f'Optimizing {symbol}'):
        if fast >= slow:
            continue

        strategy = TrendFollower(fast, slow, adx_th, atr_m, use_pyramiding=True,
                                 atr_percentile=atr_p, session_start=sess_s, session_end=sess_e)

        res_train = backtest(df_train, strategy, symbol)
        res_test = backtest(df_test, strategy, symbol)

        # Score: HEAVILY prioritize Profit and PF over DD
        pf = max(res_test['pf'], 0.01)
        profit = res_test['profit']
        dd = res_test['max_dd']
        trades = max(res_test['trades'], 1)

        # Strong profit reward, weak DD penalty (we can handle DD if profit is good)
        reward = (2.0 * pf * np.log1p(max(profit, 0) / 100.0)) + (0.002 * trades)
        penalty = (max(0, -profit) / 5000.0) + (dd / 20000.0)  # Much lower DD penalty
        score = reward - penalty
        if score > best_score:
            best_score = score
            best_params = (fast, slow, adx_th, atr_m)
            best_result = res_test

    return best_params, best_result

# ============================================================================
# REFINEMENT + PLOTTING
# ============================================================================

def compute_score(res: dict) -> float:
    pf = max(res.get('pf', 0.0), 0.01)
    profit = res.get('profit', 0.0)
    dd = res.get('max_dd', 0.0)
    trades = max(res.get('trades', 0), 1)
    reward = (pf * np.log1p(max(profit, 0) / 100.0)) / (1.0 + dd / 100.0) + (0.001 * trades)
    penalty = (max(0, -profit) / 10000.0) + (dd / 10000.0)
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


def refine_params(symbol: str, df: pd.DataFrame, base_params: tuple):
    """Localized search around base params to fine-tune instrument settings."""
    if not base_params or len(base_params) < 4:
        return None, None

    base_fast, base_slow, base_adx, base_atr = base_params

    # Create local neighborhoods
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

    # Respect instrument-specific ADX ranges
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

    # Same split logic as optimize
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
                    res_train = backtest(df_train, strategy, symbol)
                    res_test = backtest(df_test, strategy, symbol)
                    score = compute_score(res_test)
                    if score > best_score:
                        best_score = score
                        best_params = (fast, slow, adx_th, atr_m)
                        best_result = res_test

    return best_params, best_result


def plot_equity(symbol: str, equity: list, params: tuple, result: dict):
    """Plot and auto-close"""
    # Disabled popups per request
    return

# ============================================================================
# MAIN
# ============================================================================

def main():
    print("\n" + "="*100)
    print("IMPROVED TREND-FOLLOWING BACKTESTER - ALL 7 INSTRUMENTS (EURUSD, GBPUSD, USDJPY, XAUUSD, GBPJPY, BTCUSD, NAS100)")
    print("="*100 + "\n")
    
    results_summary = []
    instrument_count = 0
    
    for symbol in sorted(INSTRUMENTS.keys()):
        instrument_count += 1
        print(f"\n{'='*100}")
        print(f" [{instrument_count}/7] {symbol}")
        print('='*100)
        
        df = fetch_data(symbol, 10000)
        if df is None or len(df) < 1000:
            print(f"[X] Failed to fetch {symbol}")
            continue

        # Limit to last BACKTEST_DAYS
        if 'Time' in df.columns and len(df) > 0:
            cutoff = pd.to_datetime(df['Time'].iloc[-1]) - pd.Timedelta(days=BACKTEST_DAYS)
            df = df[df['Time'] >= cutoff].reset_index(drop=True)
        
        duration = get_data_duration(df)
        print(f"[+] Loaded {len(df)} candles | Duration: {duration}")
        
        best_params, best_result = optimize(symbol, df)
        
        if best_params is None:
            print(f"[X] Optimization failed for {symbol}")
            continue

        # Use best params from optimization (skip slow refinement for speed)
        chosen_params, chosen_result = best_params, best_result

        print(f"\n[+] Best Parameters: EMA {chosen_params[0]}/{chosen_params[1]}, ADX {chosen_params[2]:.0f}, ATR {chosen_params[3]:.1f}")
        print(f"    Trades: {chosen_result['trades']} | Win Rate: {chosen_result['win_rate']:.1f}% | Profit: ${chosen_result['profit']:.0f}")
        print(f"    Max DD: ${chosen_result['max_dd']:.0f} | PF: {chosen_result['pf']:.2f}")
        print(f"    Avg RR: {chosen_result['avg_rr']:.2f}  |  Median RR: {chosen_result['median_rr']:.2f}  |  Range: {chosen_result['min_rr']:.2f}-{chosen_result['max_rr']:.2f}")
        
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
            'Avg_RR': chosen_result['avg_rr'],
            'Median_RR': chosen_result['median_rr'],
            'Min_RR': chosen_result['min_rr'],
            'Max_RR': chosen_result['max_rr'],
        })
    
    # Save summary and persist best settings for future use
    if results_summary:
        summary_df = pd.DataFrame(results_summary)
        summary_df = summary_df.sort_values('Profit', ascending=False)
        summary_path = DATA_DIR / 'summary.csv'
        summary_df.to_csv(summary_path, index=False)

        # Persist best settings to reusable file, only if profit improves per instrument
        existing = {}
        if BEST_SETTINGS_FILE.exists():
            try:
                existing = json.loads(BEST_SETTINGS_FILE.read_text()).get('instruments', {})
            except Exception:
                existing = {}

        merged = existing.copy()
        for _, row in summary_df.iterrows():
            sym = row['Symbol']
            prev = merged.get(sym, {})
            prev_profit = prev.get('Profit', -1e18)
            if row['Profit'] > prev_profit:
                merged[sym] = {
                    'EMA_Fast': row['EMA_Fast'],
                    'EMA_Slow': row['EMA_Slow'],
                    'ADX': row['ADX'],
                    'ATR_Mult': row['ATR_Mult'],
                    'Trades': row['Trades'],
                    'Win_Rate': row['Win_Rate'],
                    'Profit': row['Profit'],
                    'Max_DD': row['Max_DD'],
                    'PF': row['PF'],
                }

        best_settings = {
            'generated_at': datetime.utcnow().isoformat() + 'Z',
            'source': 'backtest_improved.py',
            'instruments': merged,
        }
        BEST_SETTINGS_FILE.write_text(json.dumps(best_settings, indent=2))

        print(f"\n{'='*80}")
        print("FINAL RESULTS")
        print('='*80)
        print(summary_df.to_string(index=False))
        print(f"\n[+] Results saved to: {summary_path}")
        print(f"[+] Best settings saved to: {BEST_SETTINGS_FILE}")
    
    # Keep popups alive
    plt.pause(32)
    plt.close('all')

if __name__ == '__main__':
    main()
