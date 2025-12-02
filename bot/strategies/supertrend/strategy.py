"""
SuperTrend + Hull MA Strategy
=============================
Proven trend-following strategy based on TradingView's Dark Vector system.

Key Concepts:
1. SuperTrend: ATR-based trailing stop that defines trend direction
2. Hull Moving Average (HMA): Fast, low-lag MA for trend confirmation
3. Choppiness Index: Filters out ranging/consolidation periods

Entry Rules:
- LONG: Price closes above SuperTrend line AND above HMA AND market not choppy
- SHORT: Price closes below SuperTrend line AND below HMA AND market not choppy

Exit Rules:
- Trend line flips direction (SuperTrend changes color)
- Trailing stop hit (the SuperTrend line itself acts as trailing stop)

Risk Management:
- Stop Loss: Beyond the SuperTrend line
- Take Profit: Trail with SuperTrend or use fixed R:R

Advantages:
- Works in TRENDING markets (current market condition)
- Volatility-adaptive (ATR-based)
- Filters choppy/ranging conditions
- Clear entry/exit rules
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add parent directory to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, List, Tuple
from datetime import datetime

from strategies import BaseStrategy, StrategyContext, StrategyRegistry
from core.order_types import StrategySignal
from core.utils import Candle


class SuperTrendStrategy(BaseStrategy):
    """
    SuperTrend + Hull MA Trend Following Strategy
    
    This is a professional-grade trend-following system that:
    1. Uses ATR-based SuperTrend for dynamic support/resistance
    2. Confirms trend with Hull Moving Average (faster than SMA/EMA)
    3. Avoids false signals with Choppiness Index filter
    """
    
    name: str = "supertrend"
    
    def __init__(self, context: StrategyContext):
        super().__init__(context)
        
        # Default parameters optimized for forex/crypto
        self.params = {
            # SuperTrend Settings
            'atr_period': 10,           # ATR period for volatility
            'atr_multiplier': 3.0,      # Multiplier for SuperTrend bands
            
            # Hull MA Settings  
            'hma_period': 50,           # Hull MA period for trend filter
            'use_hma_filter': True,     # Whether to require HMA alignment
            
            # Choppiness Index (Noise Gate)
            'chop_period': 14,          # Lookback for choppiness
            'chop_threshold': 61.8,     # Above this = choppy, avoid trading
            'use_chop_filter': True,    # Whether to use choppiness filter
            
            # Volume Filter
            'volume_filter': False,      # Forex has no real volume
            'volume_period': 20,
            'volume_threshold': 0.8,    # Volume must be at least 80% of average
            
            # Risk Management
            'sl_atr_multiplier': 1.5,   # SL = SuperTrend ± (1.5 * ATR)
            'tp_rr_ratio': 2.0,         # Risk:Reward ratio for TP
            'trailing_stop': True,      # Use SuperTrend as trailing stop
            
            # Position Sizing
            'risk_per_trade': 0.015,    # 1.5% risk per trade
            'max_positions': 1,         # Only 1 position at a time
            
            # Signal State (prevent repeated signals)
            'alternate_signals': True,   # Force alternating Long/Short
            
            # Minimum candles needed
            'min_candles': 100,
        }
        
        # Override with context parameters
        if context.parameters:
            self.params.update(context.parameters)
        
        # Candle buffer
        self.candles: List[Candle] = []
        
        # State tracking
        self.current_position = None
        self.last_signal = None
        self.entry_price = None
        self.stop_loss = None
        self.take_profit = None
        
    def calculate_atr(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """Calculate Average True Range"""
        if period is None:
            period = self.params['atr_period']
            
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()
        
        return atr
    
    def calculate_supertrend(self, df: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        Calculate SuperTrend indicator
        
        Returns:
            supertrend: The SuperTrend line values
            direction: 1 for bullish, -1 for bearish
        """
        atr = self.calculate_atr(df)
        multiplier = self.params['atr_multiplier']
        
        # Calculate basic bands
        hl2 = (df['high'] + df['low']) / 2
        upper_band = hl2 + (multiplier * atr)
        lower_band = hl2 - (multiplier * atr)
        
        # Initialize SuperTrend arrays
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        # Calculate SuperTrend
        for i in range(1, len(df)):
            # Upper band calculation (only moves down in uptrend)
            if lower_band.iloc[i] > lower_band.iloc[i-1] or df['close'].iloc[i-1] < lower_band.iloc[i-1]:
                lower_band.iloc[i] = lower_band.iloc[i]
            else:
                lower_band.iloc[i] = lower_band.iloc[i-1]
            
            # Lower band calculation (only moves up in downtrend)
            if upper_band.iloc[i] < upper_band.iloc[i-1] or df['close'].iloc[i-1] > upper_band.iloc[i-1]:
                upper_band.iloc[i] = upper_band.iloc[i]
            else:
                upper_band.iloc[i] = upper_band.iloc[i-1]
            
            # Determine trend direction
            if i == 1:
                direction.iloc[i] = 1 if df['close'].iloc[i] > upper_band.iloc[i] else -1
            else:
                if direction.iloc[i-1] == -1 and df['close'].iloc[i] > upper_band.iloc[i-1]:
                    direction.iloc[i] = 1
                elif direction.iloc[i-1] == 1 and df['close'].iloc[i] < lower_band.iloc[i-1]:
                    direction.iloc[i] = -1
                else:
                    direction.iloc[i] = direction.iloc[i-1]
            
            # Set SuperTrend value based on direction
            supertrend.iloc[i] = lower_band.iloc[i] if direction.iloc[i] == 1 else upper_band.iloc[i]
        
        return supertrend, direction
    
    def calculate_hma(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Hull Moving Average (HMA)
        HMA = WMA(2*WMA(n/2) - WMA(n), sqrt(n))
        
        Hull MA reduces lag significantly compared to SMA/EMA
        """
        if period is None:
            period = self.params['hma_period']
        
        close = df['close']
        half_period = int(period / 2)
        sqrt_period = int(np.sqrt(period))
        
        # Calculate WMAs
        wma_half = close.rolling(window=half_period).apply(
            lambda x: np.sum(x * np.arange(1, half_period + 1)) / np.sum(np.arange(1, half_period + 1))
        )
        wma_full = close.rolling(window=period).apply(
            lambda x: np.sum(x * np.arange(1, period + 1)) / np.sum(np.arange(1, period + 1))
        )
        
        # Raw HMA value
        raw_hma = 2 * wma_half - wma_full
        
        # Final HMA
        hma = raw_hma.rolling(window=sqrt_period).apply(
            lambda x: np.sum(x * np.arange(1, sqrt_period + 1)) / np.sum(np.arange(1, sqrt_period + 1))
        )
        
        return hma
    
    def calculate_choppiness_index(self, df: pd.DataFrame, period: int = None) -> pd.Series:
        """
        Calculate Choppiness Index
        Values above 61.8 indicate choppy/ranging market
        Values below 38.2 indicate strong trending market
        
        Formula: 100 * LOG10(SUM(ATR,n) / (Highest High - Lowest Low)) / LOG10(n)
        """
        if period is None:
            period = self.params['chop_period']
        
        atr = self.calculate_atr(df, period=1)  # 1-period ATR = True Range
        atr_sum = atr.rolling(window=period).sum()
        
        highest_high = df['high'].rolling(window=period).max()
        lowest_low = df['low'].rolling(window=period).min()
        
        # Avoid division by zero
        range_hl = highest_high - lowest_low
        range_hl = range_hl.replace(0, np.nan)
        
        chop = 100 * np.log10(atr_sum / range_hl) / np.log10(period)
        
        return chop
    
    def calculate_volume_filter(self, df: pd.DataFrame) -> pd.Series:
        """Check if current volume is above threshold of average"""
        if 'volume' not in df.columns or df['volume'].isna().all():
            return pd.Series(True, index=df.index)  # If no volume, always pass
        
        avg_volume = df['volume'].rolling(window=self.params['volume_period']).mean()
        threshold = avg_volume * self.params['volume_threshold']
        
        return df['volume'] >= threshold
    
    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on SuperTrend strategy
        
        Returns DataFrame with columns:
        - signal: 1 (long), -1 (short), 0 (no signal)
        - supertrend: SuperTrend line value
        - direction: Current trend direction
        - entry_price: Suggested entry price
        - stop_loss: Calculated stop loss
        - take_profit: Calculated take profit
        """
        df = df.copy()
        
        # Calculate indicators
        supertrend, direction = self.calculate_supertrend(df)
        df['supertrend'] = supertrend
        df['direction'] = direction
        df['atr'] = self.calculate_atr(df)
        
        # Hull MA filter
        if self.params['use_hma_filter']:
            df['hma'] = self.calculate_hma(df)
            df['hma_bullish'] = df['close'] > df['hma']
            df['hma_bearish'] = df['close'] < df['hma']
        else:
            df['hma_bullish'] = True
            df['hma_bearish'] = True
        
        # Choppiness filter
        if self.params['use_chop_filter']:
            df['choppiness'] = self.calculate_choppiness_index(df)
            df['not_choppy'] = df['choppiness'] < self.params['chop_threshold']
        else:
            df['not_choppy'] = True
        
        # Volume filter
        if self.params['volume_filter']:
            df['volume_ok'] = self.calculate_volume_filter(df)
        else:
            df['volume_ok'] = True
        
        # Detect SuperTrend direction changes
        df['direction_change'] = df['direction'] != df['direction'].shift(1)
        df['bullish_flip'] = (df['direction'] == 1) & df['direction_change']
        df['bearish_flip'] = (df['direction'] == -1) & df['direction_change']
        
        # Generate signals
        df['signal'] = 0
        
        # LONG signal conditions:
        # 1. SuperTrend flips bullish (price closes above SuperTrend)
        # 2. Price above HMA (if filter enabled)
        # 3. Market not choppy (if filter enabled)
        # 4. Volume sufficient (if filter enabled)
        long_condition = (
            df['bullish_flip'] & 
            df['hma_bullish'] & 
            df['not_choppy'] & 
            df['volume_ok']
        )
        
        # SHORT signal conditions (inverse)
        short_condition = (
            df['bearish_flip'] & 
            df['hma_bearish'] & 
            df['not_choppy'] & 
            df['volume_ok']
        )
        
        df.loc[long_condition, 'signal'] = 1
        df.loc[short_condition, 'signal'] = -1
        
        # If alternate_signals enabled, filter out repeated same-direction signals
        if self.params['alternate_signals']:
            last_signal = 0
            for i in range(len(df)):
                if df['signal'].iloc[i] != 0:
                    if df['signal'].iloc[i] == last_signal:
                        df.iloc[i, df.columns.get_loc('signal')] = 0
                    else:
                        last_signal = df['signal'].iloc[i]
        
        # Calculate entry, SL, TP
        df['entry_price'] = df['close']
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        
        for i in range(len(df)):
            if df['signal'].iloc[i] == 1:  # Long
                sl_distance = self.params['sl_atr_multiplier'] * df['atr'].iloc[i]
                df.iloc[i, df.columns.get_loc('stop_loss')] = df['supertrend'].iloc[i] - sl_distance
                tp_distance = sl_distance * self.params['tp_rr_ratio']
                df.iloc[i, df.columns.get_loc('take_profit')] = df['entry_price'].iloc[i] + tp_distance
                
            elif df['signal'].iloc[i] == -1:  # Short
                sl_distance = self.params['sl_atr_multiplier'] * df['atr'].iloc[i]
                df.iloc[i, df.columns.get_loc('stop_loss')] = df['supertrend'].iloc[i] + sl_distance
                tp_distance = sl_distance * self.params['tp_rr_ratio']
                df.iloc[i, df.columns.get_loc('take_profit')] = df['entry_price'].iloc[i] - tp_distance
        
        return df
    
    def get_current_signal(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get the current trading signal from latest data"""
        signals_df = self.generate_signals(df)
        
        if len(signals_df) == 0:
            return {'signal': 0, 'reason': 'No data'}
        
        latest = signals_df.iloc[-1]
        
        result = {
            'signal': int(latest['signal']),
            'direction': int(latest['direction']) if not pd.isna(latest['direction']) else 0,
            'supertrend': float(latest['supertrend']) if not pd.isna(latest['supertrend']) else None,
            'entry_price': float(latest['entry_price']) if latest['signal'] != 0 else None,
            'stop_loss': float(latest['stop_loss']) if not pd.isna(latest['stop_loss']) else None,
            'take_profit': float(latest['take_profit']) if not pd.isna(latest['take_profit']) else None,
            'atr': float(latest['atr']) if not pd.isna(latest['atr']) else None,
            'choppiness': float(latest.get('choppiness', 50)) if 'choppiness' in latest else None,
            'is_choppy': not latest['not_choppy'] if 'not_choppy' in latest else False,
        }
        
        # Add reason for signal
        if result['signal'] == 1:
            result['reason'] = 'SuperTrend bullish flip + HMA confirmation'
        elif result['signal'] == -1:
            result['reason'] = 'SuperTrend bearish flip + HMA confirmation'
        elif latest.get('is_choppy', False):
            result['reason'] = 'Market is choppy - no trade'
        else:
            result['reason'] = f"Trend is {'bullish' if result['direction'] == 1 else 'bearish'} - waiting for flip"
        
        return result
    
    def should_exit(self, df: pd.DataFrame, position_type: str, entry_supertrend: float) -> Tuple[bool, str]:
        """
        Check if we should exit current position
        
        Args:
            position_type: 'long' or 'short'
            entry_supertrend: SuperTrend value when we entered
            
        Returns:
            (should_exit, reason)
        """
        signals_df = self.generate_signals(df)
        latest = signals_df.iloc[-1]
        
        current_price = latest['close']
        current_supertrend = latest['supertrend']
        current_direction = latest['direction']
        
        if position_type == 'long':
            # Exit long if SuperTrend flips bearish
            if current_direction == -1:
                return True, "SuperTrend flipped bearish"
            # Or if price closes below SuperTrend (trailing stop)
            if current_price < current_supertrend and self.params['trailing_stop']:
                return True, "Trailing stop hit (price below SuperTrend)"
                
        elif position_type == 'short':
            # Exit short if SuperTrend flips bullish
            if current_direction == 1:
                return True, "SuperTrend flipped bullish"
            # Or if price closes above SuperTrend (trailing stop)
            if current_price > current_supertrend and self.params['trailing_stop']:
                return True, "Trailing stop hit (price above SuperTrend)"
        
        return False, ""
    
    def get_trailing_stop(self, df: pd.DataFrame, position_type: str) -> float:
        """Get current trailing stop level (the SuperTrend line)"""
        signals_df = self.generate_signals(df)
        return float(signals_df['supertrend'].iloc[-1])
    
    def _candles_to_dataframe(self) -> pd.DataFrame:
        """Convert candle buffer to DataFrame for indicator calculation."""
        if not self.candles:
            return pd.DataFrame()
        
        data = {
            'open': [c.open for c in self.candles],
            'high': [c.high for c in self.candles],
            'low': [c.low for c in self.candles],
            'close': [c.close for c in self.candles],
            'volume': [c.volume for c in self.candles],
            'timestamp': [c.timestamp for c in self.candles],
        }
        return pd.DataFrame(data)
    
    def on_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """
        Process the latest candle and return a strategy signal if conditions are met.
        
        This is the main entry point called by the backtest engine.
        """
        # Add candle to buffer
        self.candles.append(candle)
        
        # Keep only the candles we need (hma_period + some buffer)
        max_candles = max(200, self.params.get('hma_period', 50) * 3)
        if len(self.candles) > max_candles:
            self.candles = self.candles[-max_candles:]
        
        # Need minimum candles for indicator calculation
        min_candles = self.params.get('min_candles', 100)
        if len(self.candles) < min_candles:
            return None
        
        # Convert to DataFrame
        df = self._candles_to_dataframe()
        
        # Generate signals
        signals_df = self.generate_signals(df)
        
        if len(signals_df) == 0:
            return None
        
        latest = signals_df.iloc[-1]
        signal_value = int(latest['signal'])
        
        # No signal
        if signal_value == 0:
            return None
        
        # Build StrategySignal
        direction = "buy" if signal_value == 1 else "sell"
        entry_price = float(latest['close'])
        stop_loss = float(latest['stop_loss']) if not pd.isna(latest['stop_loss']) else None
        take_profit = float(latest['take_profit']) if not pd.isna(latest['take_profit']) else None
        
        # Calculate risk for position sizing
        if stop_loss is not None:
            sl_distance = abs(entry_price - stop_loss)
        else:
            atr = float(latest['atr']) if not pd.isna(latest['atr']) else 0.001
            sl_distance = atr * self.params['sl_atr_multiplier']
        
        # Build reason/notes
        choppiness = float(latest.get('choppiness', 0)) if 'choppiness' in latest else 0
        supertrend = float(latest['supertrend']) if not pd.isna(latest['supertrend']) else 0
        hma = float(latest.get('hma', 0)) if 'hma' in latest else 0
        
        notes = (
            f"SuperTrend flip | ST={supertrend:.5f} | "
            f"HMA={hma:.5f} | Chop={choppiness:.1f}% | "
            f"ATR={latest['atr']:.5f}"
        )
        
        return StrategySignal(
            direction=direction,
            entry=entry_price,
            sl=stop_loss,
            tp=take_profit,
            sl_distance=sl_distance,
            tp_distance=abs(entry_price - take_profit) if take_profit else None,
            score=80,  # Default high score for trend signals
            grade="A" if choppiness < 50 else "B",
            notes=notes,
        )


# Timeframe-specific parameter presets
SUPERTREND_PRESETS = {
    'default': {
        'atr_period': 14,
        'atr_multiplier': 3.0,
        'hma_period': 50,
        'chop_period': 14,
        'chop_threshold': 61.8,
    },
    'fast_scalping': {
        'atr_period': 7,
        'atr_multiplier': 2.0,
        'hma_period': 20,
        'chop_period': 10,
        'chop_threshold': 55,
        'tp_rr_ratio': 1.5,
    },
    'swing_trading': {
        'atr_period': 18,
        'atr_multiplier': 3.5,
        'hma_period': 100,
        'chop_period': 20,
        'chop_threshold': 65,
        'tp_rr_ratio': 3.0,
    },
    'conservative': {
        'atr_period': 20,
        'atr_multiplier': 4.0,
        'hma_period': 80,
        'chop_period': 21,
        'chop_threshold': 70,  # Only trade in very strong trends
        'tp_rr_ratio': 2.5,
    }
}


# Instrument-specific adjustments
INSTRUMENT_PRESETS = {
    'EURUSD': {
        'atr_multiplier': 2.5,
        'chop_threshold': 60,
    },
    'GBPUSD': {
        'atr_multiplier': 2.8,  # More volatile
        'chop_threshold': 58,
    },
    'XAUUSD': {
        'atr_multiplier': 3.5,  # Much more volatile
        'chop_threshold': 55,
        'sl_atr_multiplier': 2.0,
    },
    'BTCUSD': {
        'atr_multiplier': 4.0,  # Very volatile
        'chop_threshold': 50,
        'sl_atr_multiplier': 2.5,
        'tp_rr_ratio': 2.5,
    },
}

# Register strategy with the factory
StrategyRegistry.register(SuperTrendStrategy)
