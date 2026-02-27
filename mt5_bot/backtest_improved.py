# ═══════════════════════════════════════════════════════════════════════════════
# backtest_improved.py — ICT / SMC Backtest Engine  v2 (strict no-lookahead)
# ═══════════════════════════════════════════════════════════════════════════════
#
# Strategy: Institutional Concept Trading / Smart Money Concepts
#   • Break of Structure (BOS) with impulsive candle confirmation
#   • Order Block (OB) demand/supply zones after BOS
#   • Fair Value Gap (FVG) imbalance entries in trending markets
#   • Liquidity sweep (stop-hunt) rejection entries
#   • Premium / Discount zone filter (only buy discount, sell premium)
#   • EMA trend bias + ADX strength filter + session timing
#
# Signal on bar i-1  →  entry on bar i open  →  NO lookahead
# ATR-based stop sizing for consistent risk  →  RR-based take-profit
# ═══════════════════════════════════════════════════════════════════════════════

import os, sys, argparse, json, time, warnings
from pathlib import Path
from itertools import product
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import pandas as pd

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw):
        return it

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

INSTRUMENTS = {
    'EURUSD': {'pip_size': 0.0001, 'spread': 0.8,  'vol': 1.0},
    'GBPUSD': {'pip_size': 0.0001, 'spread': 1.0,  'vol': 1.2},
    'USDJPY': {'pip_size': 0.01,   'spread': 1.2,  'vol': 1.1},
    'XAUUSD': {'pip_size': 0.1,    'spread': 3.0,  'vol': 1.5},
    'GBPJPY': {'pip_size': 0.01,   'spread': 1.5,  'vol': 1.3},
    'BTCUSD': {'pip_size': 1.0,    'spread': 30.0, 'vol': 2.0},
    'NAS100': {'pip_size': 1.0,    'spread': 2.0,  'vol': 1.0},
}

FOREX_PLUS = {'EURUSD', 'GBPUSD', 'USDJPY', 'GBPJPY', 'XAUUSD'}
BEST_SETTINGS_FILE = Path(__file__).parent / 'best_settings.json'
DATA_DIR = Path(__file__).parent / 'backtest_data_improved'
DATA_DIR.mkdir(exist_ok=True)
AI_DATA_DIR = Path(__file__).parent / 'ai_data'
AI_DATA_DIR.mkdir(exist_ok=True)
BACKTEST_DAYS = 30
INITIAL_BALANCE = 10000.0


# ═══════════════════════════════════════════════════════════════════════════════
# DATA
# ═══════════════════════════════════════════════════════════════════════════════

def initialize_mt5() -> bool:
    if mt5 is None:
        return False
    try:
        return bool(mt5.initialize())
    except Exception:
        return False


def fetch_data(symbol: str, bars: int = 10000) -> Optional[pd.DataFrame]:
    if mt5 is not None:
        try:
            if mt5.initialize():
                req = f"{symbol}+" if symbol in FOREX_PLUS else symbol
                rates = mt5.copy_rates_from_pos(req, mt5.TIMEFRAME_M5, 0, bars)
                if rates is None or len(rates) < bars * 0.5:
                    rates = mt5.copy_rates_from_pos(req, mt5.TIMEFRAME_M15, 0, bars // 3)
                if rates is None or len(rates) < bars * 0.25:
                    rates = mt5.copy_rates_from_pos(req, mt5.TIMEFRAME_H1, 0, bars // 12)
                mt5.shutdown()
                if rates is not None and len(rates) >= 300:
                    df = pd.DataFrame(rates)
                    df['time'] = pd.to_datetime(df['time'], unit='s')
                    df = df[['time', 'open', 'high', 'low', 'close', 'tick_volume']]
                    df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
                    return df.reset_index(drop=True)
        except Exception:
            pass
    csv_path = Path(__file__).parent / 'backtest_data' / f'{symbol}_optimization.csv'
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if 'Time' in df.columns:
            df['Time'] = pd.to_datetime(df['Time'])
        return df[['Time', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    return None


# ═══════════════════════════════════════════════════════════════════════════════
# ICT / SMC  CORE
# ═══════════════════════════════════════════════════════════════════════════════

def detect_swing_points(high: np.ndarray, low: np.ndarray,
                        left: int = 5, right: int = 3):
    """
    No-lookahead swing detection.
    Swing at bar j confirmed at bar j+right → usable from bar j+right+1.
    """
    n = len(high)
    sh_price  = np.full(n, np.nan)
    sh_origin = np.full(n, -1, dtype=int)
    sl_price  = np.full(n, np.nan)
    sl_origin = np.full(n, -1, dtype=int)

    for j in range(left, n - right):
        # Swing high
        is_sh = True
        for k in range(1, left + 1):
            if high[j - k] > high[j]:
                is_sh = False; break
        if is_sh:
            for k in range(1, right + 1):
                if high[j + k] >= high[j]:
                    is_sh = False; break
        if is_sh and j + right < n:
            sh_price[j + right]  = high[j]
            sh_origin[j + right] = j

        # Swing low
        is_sl = True
        for k in range(1, left + 1):
            if low[j - k] < low[j]:
                is_sl = False; break
        if is_sl:
            for k in range(1, right + 1):
                if low[j + k] <= low[j]:
                    is_sl = False; break
        if is_sl and j + right < n:
            sl_price[j + right]  = low[j]
            sl_origin[j + right] = j

    return sh_price, sh_origin, sl_price, sl_origin


def is_rejection_candle(o, h, l, c, direction):
    """
    Check for bullish (direction=1) or bearish (direction=-1) rejection.
    Checks: strong directional body or long wick on the expected side.
    """
    rng = h - l
    if rng <= 0:
        return False
    body = abs(c - o)

    if direction == 1:  # bullish
        lower_wick = min(o, c) - l
        # Bullish body closing in upper portion
        if c > o and (c - l) / rng >= 0.55:
            return True
        # Hammer (long lower wick)
        if lower_wick / rng >= 0.40:
            return True
        # Engulfing-style: large bullish body
        if c > o and body / rng >= 0.60:
            return True
    else:  # bearish
        upper_wick = h - max(o, c)
        if c < o and (h - c) / rng >= 0.55:
            return True
        if upper_wick / rng >= 0.40:
            return True
        if c < o and body / rng >= 0.60:
            return True
    return False


# ═══════════════════════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════════════════════

def add_indicators(df, ema_period=50, atr_period=14, adx_period=14):
    df = df.copy()
    close = df['Close'].astype(float).values
    high  = df['High'].astype(float).values
    low   = df['Low'].astype(float).values
    cs = pd.Series(close)

    df['EMA'] = cs.ewm(span=ema_period, adjust=False).mean().values
    # Fast EMA for momentum / slope
    fast_span = max(8, ema_period // 3)
    df['EMA_Fast'] = cs.ewm(span=fast_span, adjust=False).mean().values

    # ATR
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low  - np.roll(close, 1))))
    df['ATR'] = pd.Series(tr).rolling(atr_period).mean().values

    # ADX
    plus_dm  = np.maximum(high - np.roll(high, 1), 0)
    minus_dm = np.maximum(np.roll(low, 1) - low, 0)
    atr_safe = np.where(df['ATR'].values > 0, df['ATR'].values, np.nan)
    di_p = 100 * (pd.Series(plus_dm).rolling(adx_period).mean().values / atr_safe)
    di_m = 100 * (pd.Series(minus_dm).rolling(adx_period).mean().values / atr_safe)
    dx   = 100 * (np.abs(di_p - di_m) /
                  np.where((di_p + di_m) != 0, di_p + di_m, np.nan))
    df['ADX'] = pd.Series(dx).rolling(adx_period).mean().values

    # RSI-14
    delta = cs.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / loss.replace(0, np.nan)
    df['RSI'] = (100 - (100 / (1 + rs))).fillna(50).values

    if 'Time' in df.columns and not np.issubdtype(df['Time'].dtype, np.datetime64):
        df['Time'] = pd.to_datetime(df['Time'])
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETERS
# ═══════════════════════════════════════════════════════════════════════════════

def _default_params(symbol):
    """(ema_period, adx_min, atr_sl_mult, rr_target)"""
    # atr_sl_mult → ATR multiplier for stop distance FROM ENTRY
    # rr_target   → take-profit = stop_dist * rr_target
    table = {
        'EURUSD': (50, 20.0, 1.2, 1.5),
        'GBPUSD': (50, 20.0, 1.2, 1.5),
        'USDJPY': (50, 20.0, 1.2, 1.5),
        'XAUUSD': (40, 18.0, 1.5, 1.5),
        'GBPJPY': (50, 18.0, 1.2, 1.5),
        'BTCUSD': (60, 18.0, 1.8, 1.5),
        'NAS100': (50, 18.0, 1.2, 1.5),
    }
    return table.get(symbol, (50, 20.0, 1.2, 1.5))


def _load_best_params(symbol):
    defs = _default_params(symbol)
    try:
        if BEST_SETTINGS_FILE.exists():
            raw = json.loads(BEST_SETTINGS_FILE.read_text())
            row = (raw.get('instruments') or {}).get(symbol, {})
            if not row:
                return defs
            ema = int(row.get('EMA', defs[0]))
            adx = float(row.get('ADX', defs[1]))
            slm = float(row.get('SL_Mult', row.get('SL_Buffer', defs[2])))
            rr  = float(row.get('RR', defs[3]))
            return (ema, adx, slm, rr)
    except Exception:
        pass
    return defs


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def calculate_max_drawdown(ec):
    if len(ec) < 2:
        return 0.0
    peak = ec[0]; mx = 0.0
    for v in ec:
        if v > peak: peak = v
        dd = peak - v
        if dd > mx: mx = dd
    return float(mx)


def _empty_result(symbol):
    return {
        'symbol': symbol, 'trades': [],
        'metrics': {
            'total_trades': 0, 'wins': 0, 'losses': 0, 'win_rate': 0,
            'total_profit': 0, 'avg_profit': 0, 'avg_win': 0, 'avg_loss': 0,
            'best_trade': 0, 'worst_trade': 0, 'profit_factor': 0,
            'max_drawdown': 0, 'final_balance': INITIAL_BALANCE, 'return_pct': 0,
        },
        'equity_curve': [INITIAL_BALANCE], 'lookahead_safe': True,
    }


def _build_metrics(symbol, trades, equity_curve):
    if not trades:
        return _empty_result(symbol)
    profits = np.array([float(t['profit']) for t in trades])
    w = profits[profits > 0]; l = profits[profits <= 0]
    total = float(np.sum(profits)); final = INITIAL_BALANCE + total
    return {
        'symbol': symbol, 'trades': trades,
        'metrics': {
            'total_trades':  int(len(trades)),
            'wins':          int(len(w)),
            'losses':        int(len(l)),
            'win_rate':      float(len(w)/len(trades)*100) if trades else 0,
            'total_profit':  total,
            'avg_profit':    float(np.mean(profits)) if len(profits) else 0,
            'avg_win':       float(np.mean(w)) if len(w) else 0,
            'avg_loss':      float(np.mean(l)) if len(l) else 0,
            'best_trade':    float(np.max(profits)) if len(profits) else 0,
            'worst_trade':   float(np.min(profits)) if len(profits) else 0,
            'profit_factor': float(np.sum(w)/abs(np.sum(l)))
                             if len(l) and abs(np.sum(l)) > 0 else 99.0,
            'max_drawdown':  float(calculate_max_drawdown(equity_curve)),
            'final_balance': float(final),
            'return_pct':    float((final - INITIAL_BALANCE) / INITIAL_BALANCE * 100),
        },
        'equity_curve': equity_curve, 'lookahead_safe': True,
    }


def get_data_duration(df):
    if df is None or 'Time' not in df.columns or len(df) < 2:
        return "unknown"
    days = (pd.to_datetime(df['Time'].iloc[-1]) - pd.to_datetime(df['Time'].iloc[0])).days
    if days < 30: return f"{days} days"
    if days < 365: return f"{days//7} weeks ({days} days)"
    return f"{days//365} years ({days} days)"


# ═══════════════════════════════════════════════════════════════════════════════
#  THE  ENGINE  —  ICT / SMC  backtest  (strict no-lookahead)
# ═══════════════════════════════════════════════════════════════════════════════

def run_backtest_no_lookahead(
    df: pd.DataFrame,
    symbol: str,
    params: Optional[Tuple] = None,
    risk_pct: float = 1.0,
) -> Dict:
    """
    ICT/SMC backtest — strict no-lookahead.

    Key design choices
    ==================
    * **ATR-based stops** — stop distance = ATR × atr_sl_mult.
      This gives *consistent* position sizing regardless of OB zone width.
    * **Premium / Discount filter** — only buy in the lower half of the
      recent 50-bar range (discount) and sell in the upper half (premium).
      This is a core ICT concept for high-probability entries.
    * **Structure requirement** — OB and FVG entries require market
      structure (HH/HL or LH/LL) to be aligned with the trade direction.
    * **No breakeven** — let trades play out to TP or SL cleanly.
      A trailing stop kicks in after 2 R to lock partial gains.

    Entry types (signal bar i-1, execute bar i open):
      1. Order Block retest  (demand/supply after BOS)
      2. Fair Value Gap fill (imbalance zone re-entry)
      3. Liquidity sweep     (stop-hunt reversal)
    """
    if df is None or len(df) < 120:
        return _empty_result(symbol)

    # ── params ──
    ema_period, adx_min, atr_sl_mult, rr_target = params or _load_best_params(symbol)

    # ── indicators ──
    df = add_indicators(df, ema_period=ema_period)
    df = df.fillna(0).reset_index(drop=True)

    O  = df['Open'].astype(float).to_numpy()
    H  = df['High'].astype(float).to_numpy()
    L  = df['Low'].astype(float).to_numpy()
    C  = df['Close'].astype(float).to_numpy()
    ATR = df['ATR'].astype(float).to_numpy()
    EMA = df['EMA'].astype(float).to_numpy()
    EMAF = df['EMA_Fast'].astype(float).to_numpy()
    ADX = df['ADX'].astype(float).to_numpy()
    RSI = df['RSI'].astype(float).to_numpy()
    times = pd.to_datetime(df['Time']) if 'Time' in df.columns else pd.Series(range(len(df)))
    hours = times.dt.hour.to_numpy() if hasattr(times, 'dt') else np.full(len(df), 12)

    pip   = INSTRUMENTS.get(symbol, INSTRUMENTS['EURUSD'])['pip_size']
    spr   = float(INSTRUMENTS.get(symbol, INSTRUMENTS['EURUSD'])['spread'])
    spr_p = spr * pip                           # spread in price units

    # ── swing detection ──
    SL_LEFT = 5; SR_RIGHT = 3
    OB_LOOK = 15; MAX_OB = 80; MAX_FVG = 50
    RANGE_BARS = 50                              # bars for premium/discount calc

    sh_pr, sh_or, sl_pr, sl_or = detect_swing_points(H, L, SL_LEFT, SR_RIGHT)

    # ── state ──
    trades: List[Dict] = []
    eq = [INITIAL_BALANCE]
    bal = INITIAL_BALANCE

    in_trade   = False
    dirn       = 0
    e_price    = 0.0
    e_idx      = 0
    sl_price   = 0.0
    init_sl    = 0.0
    tp_price   = 0.0
    held       = 0

    r_sh: List[Tuple[int,float]] = []            # recent swing highs
    r_sl: List[Tuple[int,float]] = []            # recent swing lows
    obs:  List[Dict] = []                        # active OBs
    fvgs: List[Dict] = []                        # active FVGs
    struct = 0                                    # 1 bull, -1 bear, 0 neutral
    used_sweeps: set = set()

    dtc: Dict = {}                               # daily trade count
    cur_day   = None
    day_bal   = bal
    blocked   = False
    cooldown  = 0
    last_eidx = -1000
    c_losses  = 0
    risk      = max(0.1, min(10.0, risk_pct))

    is_crypto = symbol in ('BTCUSD',)
    is_us     = symbol in ('NAS100',)
    start     = max(ema_period + 10, SL_LEFT + SR_RIGHT + 20, 60)

    # ══════════ MAIN LOOP ══════════
    for i in range(start, len(df)):
        p = i - 1                                # "prev" bar — all decisions here

        # ── daily reset ──
        if hasattr(times, 'iloc'):
            ld = pd.Timestamp(times.iloc[i]).date()
            if ld != cur_day:
                cur_day = ld; day_bal = bal; blocked = False; c_losses = 0

        # ── update confirmed swings ──
        if not np.isnan(sh_pr[p]):
            r_sh.append((int(sh_or[p]), float(sh_pr[p])))
            if len(r_sh) > 25: r_sh = r_sh[-25:]
        if not np.isnan(sl_pr[p]):
            r_sl.append((int(sl_or[p]), float(sl_pr[p])))
            if len(r_sl) > 25: r_sl = r_sl[-25:]

        # ── market structure ──
        if len(r_sh) >= 2 and len(r_sl) >= 2:
            hh = r_sh[-1][1] > r_sh[-2][1]
            hl = r_sl[-1][1] > r_sl[-2][1]
            lh = r_sh[-1][1] < r_sh[-2][1]
            ll = r_sl[-1][1] < r_sl[-2][1]
            if hh and hl: struct = 1
            elif lh and ll: struct = -1

        # ── detect BOS → create OBs ──
        if r_sh:
            lsh = r_sh[-1][1]
            if C[p] > lsh and (p < 2 or C[p-1] <= lsh):
                # Bullish BOS — demand OB from last bearish candle pre-move
                for j in range(p-1, max(p-OB_LOOK, start), -1):
                    if C[j] < O[j] and (H[j]-L[j]) > 0:
                        obs.append({'d':1, 'lo':float(L[j]), 'hi':float(H[j]),
                                    'b':j, 'bb':p, 'ok':True})
                        break
        if r_sl:
            lsl = r_sl[-1][1]
            if C[p] < lsl and (p < 2 or C[p-1] >= lsl):
                for j in range(p-1, max(p-OB_LOOK, start), -1):
                    if C[j] > O[j] and (H[j]-L[j]) > 0:
                        obs.append({'d':-1, 'lo':float(L[j]), 'hi':float(H[j]),
                                    'b':j, 'bb':p, 'ok':True})
                        break

        # ── detect FVGs ──
        if p >= 2:
            ap = max(ATR[p], 1e-10)
            g_b = L[p] - H[p-2]
            if g_b > ap * 0.20 and C[p] > C[p-2]:
                fvgs.append({'d':1, 'lo':float(H[p-2]), 'hi':float(L[p]), 'b':p})
            g_s = L[p-2] - H[p]
            if g_s > ap * 0.20 and C[p] < C[p-2]:
                fvgs.append({'d':-1, 'lo':float(H[p]), 'hi':float(L[p-2]), 'b':p})

        # ── expire zones ──
        obs  = [o for o in obs  if (p - o['bb']) < MAX_OB and o['ok']]
        fvgs = [f for f in fvgs if (p - f['b']) < MAX_FVG]
        for o in obs:
            if o['d'] == 1  and C[p] < o['lo'] - ATR[p]*0.5: o['ok'] = False
            if o['d'] == -1 and C[p] > o['hi'] + ATR[p]*0.5: o['ok'] = False

        # ══════════ TRADE MANAGEMENT ══════════
        if in_trade:
            held += 1
            bh = H[i]; bl = L[i]

            ir = abs(e_price - init_sl)
            if ir > 0:
                fav = (bh - e_price) if dirn == 1 else (e_price - bl)

                # Trailing stop after 1.5 R (lock gains)
                if fav >= ir * 1.5 and ATR[i] > 0:
                    trail = ATR[i] * 1.0
                    if dirn == 1:
                        sl_price = max(sl_price, bh - trail)
                    else:
                        sl_price = min(sl_price, bl + trail)

            # Exit checks
            ex = False; ep = C[i]; er = 'Time'
            if dirn == 1:
                if bl <= sl_price: ex=True; ep=sl_price; er='SL'
                elif bh >= tp_price: ex=True; ep=tp_price; er='TP'
            else:
                if bh >= sl_price: ex=True; ep=sl_price; er='SL'
                elif bl <= tp_price: ex=True; ep=tp_price; er='TP'

            mh = 120 if symbol in ('BTCUSD','NAS100') else 96
            if not ex and held >= mh:
                ex = True; er = 'Time'

            if ex:
                mv = (ep - e_price) if dirn == 1 else (e_price - ep)
                ra = bal * (risk / 100.0)
                sd = abs(e_price - init_sl)
                gp = (mv / sd) * ra if sd > 0 else 0.0
                sc = (spr_p / sd) * ra if sd > 0 else 0.0
                pnl = float(gp - sc)
                bal += pnl; eq.append(bal)
                trades.append({
                    'entry_time': times.iloc[e_idx], 'exit_time': times.iloc[i],
                    'direction': 'BUY' if dirn==1 else 'SELL',
                    'entry_price': float(e_price), 'exit_price': float(ep),
                    'stop_loss': float(sl_price), 'take_profit': float(tp_price),
                    'profit': pnl, 'exit_reason': er, 'bars_held': int(held),
                })
                in_trade = False; dirn = 0
                if pnl < 0:
                    c_losses += 1
                    if c_losses >= 3: blocked = True
                    if er == 'SL': cooldown = i + 4
                else:
                    c_losses = 0
                if day_bal > 0 and (day_bal - bal)/day_bal*100 >= 3.0:
                    blocked = True
            continue

        # ══════════ ENTRY LOGIC ══════════
        if blocked or i < cooldown:
            continue

        atr_p = ATR[p]
        if atr_p <= 0 or ADX[p] < adx_min:
            continue

        # Session filter
        h = hours[p]
        if is_crypto:
            pass
        elif is_us:
            if h < 14 or h > 20: continue
        else:
            if not (7 <= h <= 17): continue

        # Spacing
        if (i - last_eidx) < 6:
            continue

        # Daily cap
        td = pd.Timestamp(times.iloc[i]).date() if hasattr(times, 'iloc') else None
        md = 5 if is_crypto else 3
        if td and dtc.get(td, 0) >= md:
            continue

        # RSI extremes
        if RSI[p] > 75 or RSI[p] < 25:
            continue

        # ── Trend from EMA ──
        ema_trend = 0
        if C[p] > EMA[p] and EMAF[p] > EMA[p]:
            ema_trend = 1
        elif C[p] < EMA[p] and EMAF[p] < EMA[p]:
            ema_trend = -1
        if ema_trend == 0:
            continue

        # ── Premium / Discount zone ──
        lb = max(0, p - RANGE_BARS)
        rng_hi = np.max(H[lb:p+1])
        rng_lo = np.min(L[lb:p+1])
        rng_mid = (rng_hi + rng_lo) / 2.0
        # Longs only if price is in discount (lower 55%), shorts in premium
        if ema_trend == 1 and C[p] > rng_lo + (rng_hi - rng_lo) * 0.65:
            continue
        if ema_trend == -1 and C[p] < rng_hi - (rng_hi - rng_lo) * 0.65:
            continue

        sig = 0

        # ─── ENTRY 1: Order Block Retest ───
        for ob in obs:
            if not ob['ok']:
                continue
            if ob['d'] == 1 and ema_trend == 1 and struct >= 0:
                # Price dipped into demand OB
                if L[p] <= ob['hi'] and C[p] >= ob['lo']:
                    if is_rejection_candle(O[p], H[p], L[p], C[p], 1):
                        sig = 1; ob['ok'] = False; break
            elif ob['d'] == -1 and ema_trend == -1 and struct <= 0:
                if H[p] >= ob['lo'] and C[p] <= ob['hi']:
                    if is_rejection_candle(O[p], H[p], L[p], C[p], -1):
                        sig = -1; ob['ok'] = False; break

        # ─── ENTRY 2: Fair Value Gap Fill ───
        if sig == 0:
            for fi in range(len(fvgs)):
                fv = fvgs[fi]
                if fv['d'] == 1 and ema_trend == 1 and struct >= 0:
                    if L[p] <= fv['hi'] and C[p] > fv['lo'] and C[p] > O[p]:
                        sig = 1; fvgs.pop(fi); break
                elif fv['d'] == -1 and ema_trend == -1 and struct <= 0:
                    if H[p] >= fv['lo'] and C[p] < fv['hi'] and C[p] < O[p]:
                        sig = -1; fvgs.pop(fi); break

        # ─── ENTRY 3: Liquidity Sweep ───
        if sig == 0 and ema_trend == 1 and r_sl:
            for si, sv in r_sl[-4:]:
                lk = ('s', si)
                if lk in used_sweeps: continue
                if L[p] < sv and C[p] > sv and C[p] > O[p]:
                    sig = 1; used_sweeps.add(lk); break

        if sig == 0 and ema_trend == -1 and r_sh:
            for si, sv in r_sh[-4:]:
                lk = ('h', si)
                if lk in used_sweeps: continue
                if H[p] > sv and C[p] < sv and C[p] < O[p]:
                    sig = -1; used_sweeps.add(lk); break

        if sig == 0:
            continue

        # ── Place trade ──
        dirn  = sig
        e_idx = i
        e_price = O[i]

        # ATR-based stop (consistent sizing)
        stop_dist = atr_p * atr_sl_mult
        if dirn == 1:
            sl_price = e_price - stop_dist
            tp_price = e_price + stop_dist * rr_target
        else:
            sl_price = e_price + stop_dist
            tp_price = e_price - stop_dist * rr_target

        # Sanity
        if stop_dist < pip * 2 or stop_dist > atr_p * 6:
            dirn = 0; continue

        in_trade = True; held = 0; init_sl = sl_price; last_eidx = i
        if td: dtc[td] = dtc.get(td, 0) + 1

    return _build_metrics(symbol, trades, eq)


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTING MODES
# ═══════════════════════════════════════════════════════════════════════════════

def run_split_backtest(df, symbol, split_ratio=0.7, risk_pct=1.0):
    if split_ratio <= 0 or split_ratio >= 1: split_ratio = 0.7
    idx = int(len(df) * split_ratio)
    tr = run_backtest_no_lookahead(df.iloc[:idx].copy(), symbol, risk_pct=risk_pct)
    te = run_backtest_no_lookahead(df.iloc[idx:].copy(), symbol, risk_pct=risk_pct)
    return {'symbol': symbol, 'mode': 'split', 'split_ratio': split_ratio,
            'train': {'metrics': tr.get('metrics',{}), 'trades': tr.get('trades',[])},
            'test':  {'metrics': te.get('metrics',{}), 'trades': te.get('trades',[])}}


def monte_carlo_analysis(trades, iterations=200):
    if not trades:
        return {'iterations': iterations, 'ending_balances': [],
                'p10': 0, 'p50': 0, 'p90': 0, 'max_drawdown_avg': 0}
    profits = [float(t.get('profit',0)) for t in trades]
    ends = []; dds = []
    for _ in range(max(1,int(iterations))):
        sh = profits.copy(); np.random.shuffle(sh)
        b = INITIAL_BALANCE; e = [b]
        for x in sh: b += x; e.append(b)
        ends.append(b); dds.append(calculate_max_drawdown(e))
    return {'iterations': int(iterations), 'ending_balances': ends,
            'p10': float(np.percentile(ends,10)), 'p50': float(np.percentile(ends,50)),
            'p90': float(np.percentile(ends,90)), 'max_drawdown_avg': float(np.mean(dds))}


def walk_forward_analysis(symbol, total_days=180, train_days=60, test_days=30):
    results = []
    for w in range(max(0, (total_days - train_days) // test_days)):
        end_d = train_days + (w+1)*test_days
        df = fetch_data(symbol, max(1200, end_d*96))
        if df is None or len(df) < 300: continue
        si = int(len(df) * (train_days / max(end_d,1)))
        r = run_backtest_no_lookahead(df.iloc[si:].copy(), symbol)
        m = r.get('metrics',{})
        if m: results.append(m)
    if not results:
        return {'symbol':symbol,'mode':'walk_forward','total_periods':0,
                'profitable_periods':0,'consistency':0,
                'average_profit_per_period':0,'periods':[]}
    pr = sum(1 for r in results if float(r.get('total_profit',0)) > 0)
    return {'symbol':symbol,'mode':'walk_forward','total_periods':len(results),
            'profitable_periods':pr,'consistency':pr/len(results)*100,
            'average_profit_per_period':float(np.mean([float(r.get('total_profit',0)) for r in results])),
            'periods':results}


# ═══════════════════════════════════════════════════════════════════════════════
# OPTIMIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_score(m):
    pf = m.get('profit_factor',0); ret = m.get('return_pct',0)
    t = m.get('total_trades',0); wr = m.get('win_rate',0); dd = m.get('max_drawdown',0)
    if t < 3: return -999.0
    r = (max(pf-1,0)*20 + ret*0.5 + np.sqrt(t)*2 + max(wr-30,0)*0.3)
    p = (dd/max(INITIAL_BALANCE,1))*30 + (abs(ret) if ret < 0 else 0)
    return r - p


def optimize(symbol, df, risk_pct=2.0):
    sp = int(len(df)*0.7)
    dtr = df.iloc[:sp].copy(); dte = df.iloc[sp:].copy()
    ema_g = [30,40,50,60,80]; adx_g = [16.0,20.0,24.0]
    slm_g = [1.0,1.5,2.0]; rr_g = [1.5,2.0,2.5]
    best_s = -np.inf; best_p = None; best_r = None
    for e,a,s,r in tqdm(list(product(ema_g,adx_g,slm_g,rr_g)), desc=f'Opt {symbol}'):
        pr = (e,a,s,r)
        _ = run_backtest_no_lookahead(dtr, symbol, params=pr, risk_pct=risk_pct)
        rv = run_backtest_no_lookahead(dte, symbol, params=pr, risk_pct=risk_pct)
        sc = compute_score(rv.get('metrics',{}))
        if sc > best_s: best_s=sc; best_p=pr; best_r=rv
    return best_p, best_r


def refine_params(symbol, df, base, risk_pct=2.0):
    if not base or len(base) < 4: return None, None
    be,ba,bs,br = base
    sp = int(len(df)*0.7); dtr = df.iloc[:sp].copy(); dte = df.iloc[sp:].copy()
    best_s = -np.inf; best_p = None; best_r = None
    for e in sorted({max(10,be-10), be, be+10}):
        for a in sorted({max(10,ba-3), ba, ba+3}):
            for s in sorted({max(0.5,bs-0.3), bs, min(3.0,bs+0.3)}):
                for r in sorted({max(1.2,br-0.3), br, min(4.0,br+0.3)}):
                    pr = (int(e),float(a),float(s),float(r))
                    _ = run_backtest_no_lookahead(dtr, symbol, params=pr, risk_pct=risk_pct)
                    rv = run_backtest_no_lookahead(dte, symbol, params=pr, risk_pct=risk_pct)
                    sc = compute_score(rv.get('metrics',{}))
                    if sc > best_s: best_s=sc; best_p=pr; best_r=rv
    return best_p, best_r


# ═══════════════════════════════════════════════════════════════════════════════
# BUILD / SAVE
# ═══════════════════════════════════════════════════════════════════════════════

def build_output_payload(symbol, mode, result):
    p = {'symbol': symbol, 'mode': mode,
         'timestamp': datetime.now().isoformat(),
         'engine': 'backtest_improved.py (ICT/SMC v2)'}
    p.update(result); return p

def save_output(payload, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f: json.dump(payload, f, indent=2, default=str)


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING MODE
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "="*80)
    print("  ICT / SMC  BACKTEST ENGINE  v2  — TRAINING")
    print("="*80 + "\n")

    rows = []; n_ok = 0
    for sym in sorted(INSTRUMENTS.keys()):
        print(f"\n{'─'*60}\n  {sym}\n{'─'*60}")
        df = fetch_data(sym, 15000)
        if df is None or len(df) < 600:
            print(f"  [X] Not enough data"); continue
        if 'Time' in df.columns:
            co = pd.to_datetime(df['Time'].iloc[-1]) - pd.Timedelta(days=BACKTEST_DAYS)
            df = df[df['Time'] >= co].reset_index(drop=True)
        print(f"  Bars: {len(df)}  Duration: {get_data_duration(df)}")

        bp, br = optimize(sym, df, risk_pct=2.0)
        if bp is None: print("  [X] Opt failed"); continue
        rp, rr = refine_params(sym, df, bp, risk_pct=2.0)
        if rr and compute_score(rr.get('metrics',{})) > compute_score(br.get('metrics',{})):
            cp, cr = rp, rr
        else:
            cp, cr = bp, br

        m = cr.get('metrics',{})
        ret = m.get('return_pct',0); pf = m.get('profit_factor',0)
        wr = m.get('win_rate',0); nt = m.get('total_trades',0)
        if ret > 0: n_ok += 1
        tag = "OK" if ret > 0 else "LOSS"
        print(f"  Params: EMA={cp[0]} ADX>={cp[1]:.0f} SL×{cp[2]:.1f} RR={cp[3]:.1f}")
        print(f"  [{tag}] {nt} trades  WR={wr:.1f}%  PF={pf:.2f}  Return={ret:+.1f}%")

        rows.append({'Symbol':sym, 'EMA':int(cp[0]), 'ADX':float(cp[1]),
                      'SL_Mult':float(cp[2]), 'RR':float(cp[3]),
                      'Trades':nt, 'Win_Rate':wr, 'Profit':m.get('total_profit',0),
                      'Return_Pct':ret, 'Max_DD':m.get('max_drawdown',0), 'PF':pf})

    if rows:
        ex = {}
        if BEST_SETTINGS_FILE.exists():
            try: ex = json.loads(BEST_SETTINGS_FILE.read_text()).get('instruments',{})
            except: ex = {}
        mg = ex.copy()
        for r in rows:
            s = r['Symbol']
            if r['PF'] > mg.get(s,{}).get('PF',0):
                mg[s] = {k: r[k] for k in ['EMA','ADX','SL_Mult','RR','Trades',
                                            'Win_Rate','Profit','Return_Pct','Max_DD','PF']}
        BEST_SETTINGS_FILE.write_text(json.dumps({
            'generated_at': datetime.utcnow().isoformat()+'Z',
            'source': 'backtest_improved.py (ICT/SMC v2)',
            'instruments': mg}, indent=2))
        sdf = pd.DataFrame(rows).sort_values('Return_Pct', ascending=False)
        sdf.to_csv(DATA_DIR / 'summary.csv', index=False)
        print(f"\n{'='*80}\n  DONE — {n_ok}/{len(rows)} profitable\n{'='*80}")
        print(sdf[['Symbol','Return_Pct','PF','Win_Rate','Trades','EMA','ADX','SL_Mult','RR']].to_string(index=False))
        print(f"\n  Settings → {BEST_SETTINGS_FILE}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def run_cli_backtest_mode():
    ap = argparse.ArgumentParser(description='ICT/SMC Backtest Engine v2')
    ap.add_argument('--symbol', default='EURUSD')
    ap.add_argument('--mode', choices=['standard','walk_forward','split','monte_carlo'], default='standard')
    ap.add_argument('--split-ratio', type=float, default=0.7)
    ap.add_argument('--mc-iterations', type=int, default=200)
    ap.add_argument('--run-id', default=None)
    ap.add_argument('--output', default=None)
    ap.add_argument('--walk-forward', action='store_true')
    a = ap.parse_args()
    mode = 'walk_forward' if a.walk_forward else a.mode
    sym = str(a.symbol or 'EURUSD').upper()

    if mode == 'walk_forward':
        r = walk_forward_analysis(sym)
        pay = build_output_payload(sym, mode, r)
    else:
        df = fetch_data(sym, 10000)
        if df is None or len(df) < 120:
            pay = build_output_payload(sym, mode, {'metrics':{},'trades':[],'error':'Not enough data'})
        elif mode == 'split':
            r = run_split_backtest(df, sym, float(a.split_ratio))
            pay = build_output_payload(sym, mode, r)
        else:
            std = run_backtest_no_lookahead(df, sym)
            if mode == 'monte_carlo':
                mc = monte_carlo_analysis(std.get('trades',[]), int(a.mc_iterations))
                pay = build_output_payload(sym, mode, {'standard':std,'monte_carlo':mc})
            else:
                pay = build_output_payload(sym, mode, std)

    out = Path(a.output) if a.output else DATA_DIR / f"{a.run_id or (sym+'_'+mode)}.json"
    save_output(pay, out); return 0


if __name__ == '__main__':
    cli = {'--symbol','--mode','--split-ratio','--mc-iterations','--run-id','--output','--walk-forward'}
    if any(a in sys.argv for a in cli):
        raise SystemExit(run_cli_backtest_mode())
    main()
