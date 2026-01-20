"""
MT5 Strategy Backtester
Tests EMA 50/200 + RSI(14) strategy on historical EURUSD data
"""

import MetaTrader5 as mt5
import pandas as pd
import numpy as np
from datetime import datetime, timedelta, timezone
import sys


class StrategyBacktest:
    def __init__(self, symbol="EURUSD", timeframe=mt5.TIMEFRAME_M15, 
                 ema_fast=50, ema_slow=100, rsi_period=14,
                 sl_pips=15, tp_pips=60, lot_size=0.10,
                 rsi_buy_thresh=50, rsi_sell_thresh=50,
                 min_atr_pips=3, min_ema_gap_pips=2,
                 session_start=7, session_end=22,
                 atr_sl_mult=1.1, atr_tp_mult=3.5, use_atr_exits=True,
                 tp_sl_ratio=4.0,
                 risk_percent=0.01, min_lot=0.01, max_lot=1.0,
                 break_even_R=1.0, trail_atr_mult=1.5, enable_trailing=False,
                 min_body_ratio=0.0, trade_cooldown_bars=4,
                 min_adx=20, max_trades_per_day=3,
                 regime_adx_threshold=18, range_tp_sl_ratio=1.5,
                 range_rsi_low=40, range_rsi_high=60):
        
        self.symbol = symbol
        self.timeframe = timeframe
        self.ema_fast = ema_fast
        self.ema_slow = ema_slow
        self.rsi_period = rsi_period
        self.rsi_buy_thresh = rsi_buy_thresh
        self.rsi_sell_thresh = rsi_sell_thresh
        self.sl_pips = sl_pips
        self.tp_pips = tp_pips
        self.lot_size = lot_size
        self.pip_size = 0.0001  # will be updated from symbol info if available
        self.pip_value_dollars = 10 * self.lot_size  # $10 per pip per standard lot
        self.min_atr_pips = min_atr_pips
        self.min_ema_gap_pips = min_ema_gap_pips
        self.session_start = session_start
        self.session_end = session_end
        self.atr_sl_mult = atr_sl_mult
        self.atr_tp_mult = atr_tp_mult
        self.use_atr_exits = use_atr_exits
        self.tp_sl_ratio = tp_sl_ratio
        self.risk_percent = risk_percent
        self.min_lot = min_lot
        self.max_lot = max_lot
        self.break_even_R = break_even_R
        self.trail_atr_mult = trail_atr_mult
        self.enable_trailing = enable_trailing
        self.min_body_ratio = min_body_ratio
        self.trade_cooldown_bars = trade_cooldown_bars
        self.last_trade_index = -999  # Track last trade to enforce cooldown
        self.min_adx = min_adx
        self.max_trades_per_day = max_trades_per_day
        self.last_trade_day = None
        self.trades_today = 0
        self.regime_adx_threshold = regime_adx_threshold
        self.range_tp_sl_ratio = range_tp_sl_ratio
        self.range_rsi_low = range_rsi_low
        self.range_rsi_high = range_rsi_high
        
        self.trades = []
        self.equity_curve = []
        
    def calculate_ema(self, data, period):
        """Calculate Exponential Moving Average"""
        return data.ewm(span=period, adjust=False).mean()
    
    def calculate_rsi(self, data, period):
        """Calculate Relative Strength Index"""
        delta = data.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_adx(self, high, low, close, period=14):
        """Calculate Average Directional Index (ADX) using Wilder's smoothing approximation."""
        # True Range
        prev_close = close.shift(1)
        tr = pd.concat([
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs()
        ], axis=1).max(axis=1)

        # Directional Movement
        up_move = high.diff()
        down_move = -low.diff()
        dm_plus = ((up_move > down_move) & (up_move > 0)).astype(float) * up_move
        dm_minus = ((down_move > up_move) & (down_move > 0)).astype(float) * down_move

        # Wilder's smoothing via RMA (EWMA with alpha=1/period)
        alpha = 1.0 / period
        tr_rma = tr.ewm(alpha=alpha, adjust=False).mean()
        dm_plus_rma = dm_plus.ewm(alpha=alpha, adjust=False).mean()
        dm_minus_rma = dm_minus.ewm(alpha=alpha, adjust=False).mean()

        di_plus = 100.0 * (dm_plus_rma / tr_rma).replace([np.inf, -np.inf], np.nan)
        di_minus = 100.0 * (dm_minus_rma / tr_rma).replace([np.inf, -np.inf], np.nan)
        dx = (100.0 * (di_plus - di_minus).abs() / (di_plus + di_minus)).replace([np.inf, -np.inf], np.nan)
        adx = dx.ewm(alpha=alpha, adjust=False).mean()
        return adx
    
    def fetch_data(self, days=60):
        """Fetch historical data from MT5"""
        if not mt5.initialize():
            print(f"MT5 initialization failed: {mt5.last_error()}")
            return None

        account = mt5.account_info()
        if account is None:
            print("MT5 account not connected. Open MT5 terminal and log in first.")
            mt5.shutdown()
            return None

        info = mt5.symbol_info(self.symbol)
        if info is None:
            print(f"Symbol {self.symbol} not found in MT5.")
            mt5.shutdown()
            return None

        if not info.visible:
            mt5.symbol_select(self.symbol, True)

        self.pip_size = info.point * 10  # one pip = 10 points for FX majors

        count = days * 96  # 96 M15 candles per day
        rates = mt5.copy_rates_from_pos(self.symbol, self.timeframe, 0, count)

        if rates is None or len(rates) == 0:
            print(f"No data received from MT5. Last error: {mt5.last_error()}")
            mt5.shutdown()
            return None

        df = pd.DataFrame(rates)
        df['time'] = pd.to_datetime(df['time'], unit='s', utc=True)

        mt5.shutdown()

        start_ts = df['time'].iloc[0]
        end_ts = df['time'].iloc[-1]
        print(f"Loaded {len(df)} candles from {start_ts} to {end_ts}")

        # Sanity check: ensure we have recent data
        if datetime.now(timezone.utc) - end_ts.to_pydatetime() > timedelta(days=2):
            print("Warning: Latest candle is older than 2 days. Check MT5 connection or market hours.")
        
        return df
    
    def prepare_indicators(self, df):
        """Calculate all indicators"""
        df['ema_fast'] = self.calculate_ema(df['close'], self.ema_fast)
        df['ema_slow'] = self.calculate_ema(df['close'], self.ema_slow)
        df['rsi'] = self.calculate_rsi(df['close'], self.rsi_period)
        df['ema_fast_shift'] = df['ema_fast'].shift(1)
        df['close_prev'] = df['close'].shift(1)
        # ATR for volatility filter
        tr1 = df['high'] - df['low']
        tr2 = (df['high'] - df['close'].shift()).abs()
        tr3 = (df['low'] - df['close'].shift()).abs()
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['atr'] = tr.rolling(window=14).mean()
        # ADX for trend strength
        df['adx'] = self.calculate_adx(df['high'], df['low'], df['close'], period=14)
        
        # Drop NaN values
        df = df.dropna()
        
        return df

    def _session_ok(self, ts):
        hour = ts.hour
        return self.session_start <= hour < self.session_end
    
    def check_buy_signal(self, row):
        """Check if buy conditions are met"""
        if not self._session_ok(row['time']):
            return False
        vol_ok = row['atr'] >= (self.min_atr_pips * self.pip_size)
        # Use regime threshold to determine trending vs ranging for entries
        is_trending = (row.get('adx', 0) >= self.regime_adx_threshold)
        if is_trending:
            gap_ok = (row['ema_fast'] - row['ema_slow']) >= (self.min_ema_gap_pips * self.pip_size)
            slope_ok = row['ema_fast'] > row['ema_fast_shift']
            entry_ok = row['close'] > row['ema_fast']
            return (row['ema_fast'] > row['ema_slow'] and row['rsi'] > self.rsi_buy_thresh
                and gap_ok and slope_ok and vol_ok and entry_ok)
        else:
            # Range regime: buy low (RSI oversold)
            range_entry = (row['rsi'] <= self.range_rsi_low)
            return vol_ok and range_entry
    
    def check_sell_signal(self, row):
        """Check if sell conditions are met"""
        if not self._session_ok(row['time']):
            return False
        vol_ok = row['atr'] >= (self.min_atr_pips * self.pip_size)
        is_trending = (row.get('adx', 0) >= self.regime_adx_threshold)
        if is_trending:
            gap_ok = (row['ema_slow'] - row['ema_fast']) >= (self.min_ema_gap_pips * self.pip_size)
            slope_ok = row['ema_fast'] < row['ema_fast_shift']
            entry_ok = row['close'] < row['ema_fast']
            return (row['ema_fast'] < row['ema_slow'] and row['rsi'] < self.rsi_sell_thresh
                and gap_ok and slope_ok and vol_ok and entry_ok)
        else:
            # Range regime: sell high (RSI overbought)
            range_entry = (row['rsi'] >= self.range_rsi_high)
            return vol_ok and range_entry
    
    def simulate_trade(self, entry_price, trade_type, entry_time, atr_value=None, balance=10000, adx_value=None):
        """Simulate trade with SL/TP and balance-based sizing"""
        pip_value = self.pip_size

        # dynamic SL/TP in pips
        if self.use_atr_exits and atr_value is not None and not np.isnan(atr_value):
            sl_pips = atr_value / pip_value * self.atr_sl_mult
            rr = self.range_tp_sl_ratio if (adx_value is not None and adx_value < self.regime_adx_threshold) else self.tp_sl_ratio
            tp_pips = sl_pips * rr
        else:
            sl_pips = self.sl_pips
            rr = self.range_tp_sl_ratio if (adx_value is not None and adx_value < self.regime_adx_threshold) else self.tp_sl_ratio
            tp_pips = rr * sl_pips

        sl_pips = max(sl_pips, 1.0)
        tp_pips = max(tp_pips, sl_pips * 1.1)  # enforce positive RR

        # position sizing by risk % of balance
        dollars_at_risk = balance * self.risk_percent
        lot = dollars_at_risk / (sl_pips * 10.0)  # $10 per pip per 1.0 lot
        lot = max(self.min_lot, min(self.max_lot, lot))
        pip_value_dollars = 10.0 * lot

        if trade_type == "BUY":
            sl_price = entry_price - (sl_pips * pip_value)
            tp_price = entry_price + (tp_pips * pip_value)
        else:  # SELL
            sl_price = entry_price + (sl_pips * pip_value)
            tp_price = entry_price - (tp_pips * pip_value)
        
        return {
            'entry_time': entry_time,
            'entry_price': entry_price,
            'type': trade_type,
            'sl_price': sl_price,
            'current_sl_price': sl_price,
            'tp_price': tp_price,
            'sl_pips': sl_pips,
            'tp_pips': tp_pips,
            'lot': lot,
            'pip_value_dollars': pip_value_dollars,
            'status': 'open',
            'exit_price': None,
            'exit_time': None,
            'profit': 0,
            'breakeven_set': False
        }
    
    def check_trade_exit(self, trade, candle):
        """Check if trade hits SL or TP, with breakeven and ATR trailing"""
        pip_value = self.pip_size

        # Update breakeven and trailing stop if enabled
        if self.enable_trailing:
            if trade['type'] == "BUY":
                favorable_move_pips = max(0.0, (candle['high'] - trade['entry_price']) / pip_value)
                if not trade['breakeven_set'] and favorable_move_pips >= trade['sl_pips'] * self.break_even_R:
                    trade['current_sl_price'] = max(trade['current_sl_price'], trade['entry_price'])
                    trade['breakeven_set'] = True
                # Trail only when in profit area
                if candle['close'] > trade['entry_price'] and trade['breakeven_set']:
                    trail_price = candle['close'] - (self.trail_atr_mult * candle['atr'])
                    # Do not move SL downwards
                    trade['current_sl_price'] = max(trade['current_sl_price'], trail_price)
                    # Never trail beyond TP
                    trade['current_sl_price'] = min(trade['current_sl_price'], trade['tp_price'] - 0.5 * pip_value)
            else:  # SELL
                favorable_move_pips = max(0.0, (trade['entry_price'] - candle['low']) / pip_value)
                if not trade['breakeven_set'] and favorable_move_pips >= trade['sl_pips'] * self.break_even_R:
                    trade['current_sl_price'] = min(trade['current_sl_price'], trade['entry_price'])
                    trade['breakeven_set'] = True
                if candle['close'] < trade['entry_price'] and trade['breakeven_set']:
                    trail_price = candle['close'] + (self.trail_atr_mult * candle['atr'])
                    trade['current_sl_price'] = min(trade['current_sl_price'], trail_price)
                    trade['current_sl_price'] = max(trade['current_sl_price'], trade['tp_price'] + 0.5 * pip_value)

        # Check exits using current SL and static TP
        if trade['type'] == "BUY":
            if candle['high'] >= trade['tp_price']:
                trade['exit_price'] = trade['tp_price']
                trade['exit_time'] = candle['time']
                trade['status'] = 'win'
                trade['profit'] = (trade['tp_price'] - trade['entry_price']) / pip_value
                return True
            if candle['low'] <= trade.get('current_sl_price', trade['sl_price']):
                slp = trade.get('current_sl_price', trade['sl_price'])
                trade['exit_price'] = slp
                trade['exit_time'] = candle['time']
                trade['status'] = 'loss' if slp < trade['entry_price'] else 'win'
                trade['profit'] = (slp - trade['entry_price']) / pip_value
                return True
        else:  # SELL
            if candle['low'] <= trade['tp_price']:
                trade['exit_price'] = trade['tp_price']
                trade['exit_time'] = candle['time']
                trade['status'] = 'win'
                trade['profit'] = (trade['entry_price'] - trade['tp_price']) / pip_value
                return True
            if candle['high'] >= trade.get('current_sl_price', trade['sl_price']):
                slp = trade.get('current_sl_price', trade['sl_price'])
                trade['exit_price'] = slp
                trade['exit_time'] = candle['time']
                trade['status'] = 'loss' if slp > trade['entry_price'] else 'win'
                trade['profit'] = (trade['entry_price'] - slp) / pip_value
                return True

        return False
    
    def run_backtest(self, days=60, silent=False):
        """Run the backtest simulation"""
        if not silent:
            print(f"\n{'='*60}")
            print(f"MT5 STRATEGY BACKTEST")
            print(f"{'='*60}")
            print(f"Symbol: {self.symbol}")
            print(f"Timeframe: M15")
            print(f"Period: {days} days")
            print(f"Strategy: EMA {self.ema_fast}/{self.ema_slow} + RSI({self.rsi_period})")
            print(f"RSI thresholds: buy>{self.rsi_buy_thresh} sell<{self.rsi_sell_thresh}")
            print(f"Filters: ATR>={self.min_atr_pips}p | EMA gap>={self.min_ema_gap_pips}p | Session {self.session_start}:00-{self.session_end}:00 UTC")
            print(f"Risk: {self.sl_pips} pips SL | {self.tp_pips} pips TP | Lot: {self.lot_size}")
            print(f"{'='*60}\n")
            
            print("Fetching historical data...")
        df = self.fetch_data(days)
        
        if df is None:
            return None
        
        if not silent:
            print(f"Calculating indicators...")
        df = self.prepare_indicators(df)
        if not silent:
            print(f"Indicators ready ({len(df)} valid candles)")
            print("Running simulation...\n")
        
        # Simulation variables
        open_trade = None
        balance = 10000  # Starting balance
        self.trades = []
        self.equity_curve = []
        
        # Iterate through candles
        total_candles = len(df)
        progress_step = max(1, total_candles // 20)

        for idx, row in df.iterrows():
            
            # Check if we have an open trade
            if open_trade is not None:
                # Check if trade exits
                if self.check_trade_exit(open_trade, row):
                    # Trade closed
                    balance += open_trade['profit'] * open_trade['pip_value_dollars']
                    self.trades.append(open_trade.copy())
                    open_trade = None
            
            # Check for new signals (only if no open trade)
            if open_trade is None:
                # Reset daily trade counter if new day
                current_day = row['time'].date()
                if self.last_trade_day != current_day:
                    self.last_trade_day = current_day
                    self.trades_today = 0

                # Enforce cooldown: wait X bars after last trade and daily cap
                bars_since_last = idx - self.last_trade_index
                if bars_since_last >= self.trade_cooldown_bars and self.trades_today < self.max_trades_per_day:
                    if self.check_buy_signal(row):
                        open_trade = self.simulate_trade(row['close'], "BUY", row['time'], row['atr'], balance, adx_value=row.get('adx', None))
                        self.last_trade_index = idx  # Update cooldown tracker
                        self.trades_today += 1
                    elif self.check_sell_signal(row):
                        open_trade = self.simulate_trade(row['close'], "SELL", row['time'], row['atr'], balance, adx_value=row.get('adx', None))
                        self.last_trade_index = idx  # Update cooldown tracker
                        self.trades_today += 1
            
            # Track equity
            current_equity = balance
            if open_trade is not None:
                # Add unrealized P/L
                if open_trade['type'] == "BUY":
                    unrealized = (row['close'] - open_trade['entry_price']) / self.pip_size * open_trade['pip_value_dollars']
                else:
                    unrealized = (open_trade['entry_price'] - row['close']) / self.pip_size * open_trade['pip_value_dollars']
                current_equity += unrealized
            
            self.equity_curve.append(current_equity)

            if not silent and idx % progress_step == 0:
                pct = (idx + 1) / total_candles * 100
                print(f"Progress: {pct:5.1f}% ({idx+1}/{total_candles})", end='\r')

        if not silent:
            print(f"Progress: 100.0% ({total_candles}/{total_candles})")
        
        stats = self.build_stats(balance)
        if not silent:
            self.print_results(stats)
        return stats
    
    def build_stats(self, final_balance):
        """Compile stats dictionary"""
        if len(self.trades) == 0:
            return None
        pip_value_dollars = self.pip_value_dollars
        
        # Calculate statistics
        wins = [t for t in self.trades if t['status'] == 'win']
        losses = [t for t in self.trades if t['status'] == 'loss']
        
        total_trades = len(self.trades)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = (win_count / total_trades * 100) if total_trades > 0 else 0
        
        total_profit = sum([t['profit'] * pip_value_dollars for t in wins]) if wins else 0
        total_loss = sum([t['profit'] * pip_value_dollars for t in losses]) if losses else 0
        net_profit = total_profit + total_loss
        
        profit_factor = abs(total_profit / total_loss) if total_loss != 0 else 0
        
        avg_win = (total_profit / win_count) if win_count > 0 else 0
        avg_loss = (total_loss / loss_count) if loss_count > 0 else 0
        
        initial_balance = 10000
        roi = ((final_balance - initial_balance) / initial_balance) * 100

        # Max drawdown from equity curve
        dd = 0.0
        peak = -1e12
        for eq in self.equity_curve:
            if eq > peak:
                peak = eq
            dd = min(dd, (eq - peak) / peak * 100 if peak > 0 else 0)
        max_drawdown = abs(dd)

        return {
            "total_trades": total_trades,
            "wins": win_count,
            "losses": loss_count,
            "win_rate": win_rate,
            "gross_profit": total_profit,
            "gross_loss": total_loss,
            "net_profit": net_profit,
            "profit_factor": profit_factor,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "initial_balance": initial_balance,
            "final_balance": final_balance,
            "roi": roi,
            "trades": self.trades,
            "max_drawdown": max_drawdown,
        }

    def print_results(self, stats):
        """Print backtest results"""
        if stats is None:
            print("✗ No trades executed during backtest period")
            return

        pip_value_dollars = self.pip_value_dollars
        total_trades = stats["total_trades"]
        win_count = stats["wins"]
        loss_count = stats["losses"]
        win_rate = stats["win_rate"]
        total_profit = stats["gross_profit"]
        total_loss = stats["gross_loss"]
        net_profit = stats["net_profit"]
        profit_factor = stats["profit_factor"]
        avg_win = stats["avg_win"]
        avg_loss = stats["avg_loss"]
        initial_balance = stats["initial_balance"]
        final_balance = stats["final_balance"]
        roi = stats["roi"]

        print(f"{'='*60}")
        print(f"BACKTEST RESULTS")
        print(f"{'='*60}")
        print(f"\nTRADE STATISTICS:")
        print(f"  Total Trades: {total_trades}")
        print(f"  Wins: {win_count} ({win_rate:.1f}%)")
        print(f"  Losses: {loss_count} ({100-win_rate:.1f}%)")
        print(f"\nPROFIT/LOSS:")
        print(f"  Gross Profit: ${total_profit:.2f}")
        print(f"  Gross Loss: ${total_loss:.2f}")
        print(f"  Net Profit: ${net_profit:.2f}")
        print(f"  Profit Factor: {profit_factor:.2f}")
        print(f"\nAVERAGE RESULTS:")
        print(f"  Avg Win: ${avg_win:.2f}")
        print(f"  Avg Loss: ${avg_loss:.2f}")
        print(f"  Risk/Reward: {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Risk/Reward: N/A")
        print(f"\nACCOUNT:")
        print(f"  Initial Balance: ${initial_balance:.2f}")
        print(f"  Final Balance: ${final_balance:.2f}")
        print(f"  ROI: {roi:+.2f}%")
        print(f"\n{'='*60}")

        # Full trade log
        print(f"\nTRADE LOG (all trades):")
        print(f"{'-'*60}")
        print(f"#   Type Time(UTC)            Entry      Exit       SL         TP         Result  Pips    PnL$")
        print(f"{'-'*60}")
        for i, trade in enumerate(self.trades, 1):
            pips = trade['profit']
            pnl_dollars = trade['profit'] * pip_value_dollars
            print(f"{i:3} {trade['type']:4} {trade['entry_time']:%Y-%m-%d %H:%M}  "
                  f"{trade['entry_price']:.5f}  {trade['exit_price']:.5f}  "
                  f"{trade['sl_price']:.5f}  {trade['tp_price']:.5f}  "
                  f"{trade['status']:5}  {pips:+7.1f}  {pnl_dollars:+8.2f}")
        print(f"{'-'*60}")
        print(f"Total trades: {len(self.trades)} | Net PnL: ${net_profit:.2f} | Final balance: ${final_balance:.2f}")
        print(f"{'-'*60}\n")

    def optimize(self, days=90):
        """Quick grid search to find better params"""
        ema_fast_opts = [34, 50]
        ema_slow_opts = [100, 200]
        rsi_buy_opts = [50, 55]
        rsi_sell_opts = [50, 45]
        sltp_opts = [(15, 60), (20, 80)]
        ema_gap_opts = [0, 1, 2]  # Allow zero gap for permissive testing
        cooldown_opts = [4, 6]  # Bars to wait between trades (reduced from 4,8)
        break_even_R_opts = [1.5]
        trail_atr_mult_opts = [2.0]
        enable_trail_opts = [False]  # Trailing OFF for now
        atr_min_opts = [1, 2, 3]
        regime_adx_opts = [15, 18, 22]
        range_rsi_sets = [(35, 65), (40, 60), (45, 55)]

        results = []
        total_combos = (len(ema_fast_opts) * len(ema_slow_opts) * len(rsi_buy_opts) * 
                   len(sltp_opts) * len(ema_gap_opts) * len(cooldown_opts) * 
                   len(break_even_R_opts) * len(trail_atr_mult_opts) * len(enable_trail_opts) *
                   len(atr_min_opts) * len(regime_adx_opts) * len(range_rsi_sets))
        current_combo = 0
        print(f"Testing {total_combos} parameter combinations...\n")
        
        for ef in ema_fast_opts:
            for es in ema_slow_opts:
                if ef >= es:
                    continue
                for rb, rs in zip(rsi_buy_opts, rsi_sell_opts):
                    for sl, tp in sltp_opts:
                        for gap in ema_gap_opts:
                            for cd in cooldown_opts:
                                for br in break_even_R_opts:
                                    for tm in trail_atr_mult_opts:
                                        for et in enable_trail_opts:
                                            for atrm in atr_min_opts:
                                                for radx in regime_adx_opts:
                                                    for (rsi_low, rsi_high) in range_rsi_sets:
                                                        current_combo += 1
                                                        print(f"\rProgress: {current_combo}/{total_combos} ({current_combo/total_combos*100:.1f}%)", end='', flush=True)
                                                        self.ema_fast = ef
                                                        self.ema_slow = es
                                                        self.rsi_buy_thresh = rb
                                                        self.rsi_sell_thresh = rs
                                                        self.sl_pips = sl
                                                        self.tp_pips = tp
                                                        self.min_ema_gap_pips = gap
                                                        self.trade_cooldown_bars = cd
                                                        self.break_even_R = br
                                                        self.trail_atr_mult = tm
                                                        self.enable_trailing = et
                                                        self.min_atr_pips = atrm
                                                        self.regime_adx_threshold = radx
                                                        self.range_rsi_low = rsi_low
                                                        self.range_rsi_high = rsi_high
                                                        stats = self.run_backtest(days=days, silent=True)
                                if stats is None:
                                    continue
                                # Scoring: HEAVILY favor PF>1.2 and quality over quantity
                                pf = stats["profit_factor"]
                                pf_floor = 1.2
                                pf_bonus = (pf - pf_floor) * 50.0  # Increased from 35 to 50
                                roi_score = stats["roi"] * 0.8  # Increased from 0.6 to 0.8
                                dd_penalty = stats["max_drawdown"] * 0.6  # Increased from 0.5
                                trades = stats["total_trades"]
                                trade_penalty = 0.0
                                # Favor 20-100 trades (quality over quantity)
                                if trades < 20:
                                    trade_penalty += (20 - trades) * 0.5 + 10
                                elif trades > 100:
                                    trade_penalty += (trades - 100) * 0.15  # Heavier penalty
                                # Additional penalties
                                if pf < 1.1:
                                    trade_penalty += 25  # Increased from 10
                                if stats["win_rate"] < 20:
                                    trade_penalty += 8  # Increased from 3
                                score = roi_score + pf_bonus - dd_penalty - trade_penalty
                                if trades < 10:
                                    score -= 50  # Increased from 25
                                results.append({
                                    "ema_fast": ef,
                                    "ema_slow": es,
                                    "rsi_buy": rb,
                                    "rsi_sell": rs,
                                    "sl": sl,
                                    "tp": tp,
                                    "ema_gap": gap,
                                    "cooldown": cd,
                                    "break_even_R": br,
                                    "trail_atr_mult": tm,
                                    "enable_trailing": et,
                                    "atr_min": atrm,
                                    "regime_adx": radx,
                                    "range_rsi_low": rsi_low,
                                    "range_rsi_high": rsi_high,
                                    "roi": stats["roi"],
                                    "net": stats["net_profit"],
                                    "trades": stats["total_trades"],
                                    "win_rate": stats["win_rate"],
                                    "profit_factor": stats["profit_factor"],
                                    "dd": stats["max_drawdown"],
                                    "score": score,
                                })
        print("\n")  # New line after progress
        results = sorted(results, key=lambda x: x["score"], reverse=True)

        print("Optimization top 5 (score=ROI + PF term - DD):")
        print("ema_fast ema_slow rsi_buy rsi_sell SL TP gap cd atr regADX rL rH | ROI%  PF   DD%  Trades  Win%  Net$  Score")
        for r in results[:5]:
            print(f"{r['ema_fast']:7} {r['ema_slow']:8} {r['rsi_buy']:7} {r['rsi_sell']:8} "
                  f"{r['sl']:2} {r['tp']:2} {r['ema_gap']:3} {r['cooldown']:2} {r['atr_min']:3} {r['regime_adx']:6} {r['range_rsi_low']:2} {r['range_rsi_high']:2} | "
                  f"{r['roi']:6.2f} {r['profit_factor']:4.2f} {r['dd']:5.1f} "
                  f"{r['trades']:6} {r['win_rate']:6.1f} {r['net']:6.2f} {r['score']:6.2f}")

        if results:
            best = results[0]
            print("\nBest parameters:")
            print(best)
            # restore best to object
            self.ema_fast = best['ema_fast']
            self.ema_slow = best['ema_slow']
            self.rsi_buy_thresh = best['rsi_buy']
            self.rsi_sell_thresh = best['rsi_sell']
            self.sl_pips = best['sl']
            self.tp_pips = best['tp']
            self.min_ema_gap_pips = best['ema_gap']
            self.trade_cooldown_bars = best['cooldown']
            self.break_even_R = best['break_even_R']
            self.trail_atr_mult = best['trail_atr_mult']
            self.enable_trailing = best['enable_trailing']
            self.min_atr_pips = best['atr_min']
            self.regime_adx_threshold = best['regime_adx']
            self.range_rsi_low = best['range_rsi_low']
            self.range_rsi_high = best['range_rsi_high']
        else:
            print("No valid optimization results.")


def main():
    """Main execution"""
    backtest = StrategyBacktest()

    # Optional: run quick optimizer on 180 days, then test last 60 days
    print("Running quick optimization on last 180 days...")
    backtest.optimize(days=180)

    print("\nRunning backtest on last 60 days with optimized params...")
    backtest.run_backtest(days=60)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n✗ Backtest interrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Error: {e}")
        sys.exit(1)
