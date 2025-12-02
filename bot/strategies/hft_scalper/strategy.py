"""HFT Scalping Strategy - High Frequency / Rapid Trade Execution.

This strategy is designed for traders who want:
- Many small trades instead of few large ones
- Quick entries and exits (seconds to minutes)
- Small pip targets (3-15 pips depending on instrument)
- Tight stop losses (equal to or less than TP)
- High win rate focus (65%+ target)

Key differences from ICT/SMC swing trading:
1. M1 timeframe (vs M5/M15)
2. 1:1 or 0.5:1 R:R (vs 2:1-3.5:1)
3. Momentum-based entries (vs liquidity sweep reversals)
4. Session-based liquidity targeting
5. Spread-aware entry logic
6. Quick profit taking

ADVANCED HFT TECHNIQUES IMPLEMENTED:
=============================================
1. 5-8-13 EMA Triple Crossover System (Fibonacci-based)
   - Fast momentum detection
   - Trend alignment confirmation
   - Pullback entry identification

2. RSI Momentum Filter (7-period fast RSI)
   - Overbought/Oversold zones for reversals
   - RSI divergence detection
   - Momentum confirmation

3. Stochastic Oscillator (5,3,3 fast settings)
   - K/D crossover signals
   - Oversold bounce / Overbought fade
   - Works with trending markets

4. VWAP Proxy (Volume-Weighted Average Price simulation)
   - Institutional reference level
   - Mean reversion plays
   - Session VWAP deviation trades

5. MACD Fast Scalping (6,13,5 settings)
   - Faster than standard 12,26,9
   - Crossover entry signals
   - Histogram momentum analysis
   - Divergence detection

6. Bollinger Bands (10-period, 2 std)
   - Squeeze detection (low volatility)
   - Band touch reversals
   - Breakout momentum entries
   - Mean reversion to middle band

7. Micro-Structure Analysis
   - Micro FVGs (1-2 candle imbalances)
   - Candle momentum patterns
   - Wick rejection signals

8. Session Liquidity Windows
   - London Open (08:00-10:00 GMT)
   - NY Open (13:30-16:00 GMT)
   - Overlap period optimization

9. Spread & Volatility Filters
   - Dynamic spread monitoring
   - ATR-based volatility gates
   - Choppy market detection

Best suited for:
- Highly liquid pairs (EUR/USD, GBP/USD)
- Active session times (London/NY overlap)
- Low spread conditions
- Instruments with tight spreads
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Deque, List, Optional, Tuple

from core.order_types import OrderSide, StrategySignal
from core.utils import (
    Candle,
    FairValueGap,
    Killzone,
    active_killzone,
    active_session,
    calculate_atr,
    detect_fair_value_gaps,
)
from strategies import BaseStrategy, StrategyRegistry


@dataclass
class ScalpContext:
    """Market context for scalping decisions."""
    
    # Direction
    direction: OrderSide
    
    # EMA Analysis (5-8-13 system)
    ema_fast: float      # 5-period
    ema_medium: float    # 8-period
    ema_slow: float      # 13-period
    ema_trend: str       # "bullish", "bearish", "neutral"
    ema_aligned: bool    # All EMAs properly stacked
    ema_spread: float    # Distance between fast and slow EMA
    
    # Momentum indicators
    momentum_strength: float  # 0.0 - 1.0
    rsi_value: float          # 0-100
    rsi_signal: str           # "overbought", "oversold", "neutral"
    stoch_k: float            # Stochastic %K
    
    # MACD (Fast scalping settings: 6,13,5)
    macd_line: float          # MACD line value
    macd_signal: float        # Signal line value
    macd_histogram: float     # Histogram (MACD - Signal)
    macd_crossover: str       # "bullish", "bearish", "none"
    macd_momentum: str        # "increasing", "decreasing", "neutral"
    
    # Bollinger Bands (Fast: 10-period, 2 std dev)
    bb_upper: float           # Upper band
    bb_middle: float          # Middle band (SMA)
    bb_lower: float           # Lower band
    bb_position: str          # "above_upper", "above_middle", "below_middle", "below_lower"
    bb_squeeze: bool          # Bollinger squeeze (low volatility)
    bb_width: float           # Band width (volatility measure)
    stoch_d: float            # Stochastic %D
    stoch_signal: str         # "buy", "sell", "neutral"
    
    # ADX (Average Directional Index - Trend Strength)
    adx_value: float          # 0-100 trend strength
    plus_di: float            # +DI (positive directional indicator)
    minus_di: float           # -DI (negative directional indicator)
    adx_trend_strength: str   # "strong", "moderate", "weak", "none"
    di_crossover: str         # "bullish", "bearish", "none"
    
    # VWAP proxy
    vwap_proxy: float         # Volume-weighted average estimate
    price_vs_vwap: str        # "above", "below", "at"
    vwap_deviation: float     # How far from VWAP
    
    # Price action
    atr: float
    spread_estimate: float
    volatility_ok: bool
    
    # Session
    session: str
    killzone: Optional[Killzone]
    is_high_liquidity: bool
    
    # Micro structure
    micro_fvg: Optional[FairValueGap]
    consecutive_bullish: int
    consecutive_bearish: int
    wick_rejection: bool      # Strong wick rejection signal
    engulfing_pattern: bool   # Bullish/bearish engulfing
    
    # Trade quality metrics
    trend_strength: float     # 0.0 - 1.0
    entry_quality: str        # "A", "B", "C"
    
    # Confluence
    confluence_score: int


class HFTScalperStrategy(BaseStrategy):
    """High Frequency Trading / Scalping Strategy.
    
    Entry Logic:
    1. Trade only during high-liquidity sessions (London open, NY overlap)
    2. Follow short-term momentum (5/8/13 EMA alignment)
    3. Enter on pullbacks to fast EMA in trending conditions
    4. Enter on micro-FVG fills for quick reversions
    5. Exit quickly at small pip targets
    
    Risk Management:
    - Small position sizes (1-2% risk)
    - Tight stops (ATR-based, usually 3-8 pips)
    - Quick profit taking (TP1 at 0.5R, TP2 at 1R)
    - Max trades per session to prevent overtrading
    
    Best Instruments:
    - EUR/USD, GBP/USD (tightest spreads)
    - USD/JPY, USD/CHF (high liquidity)
    - Avoid: XAUUSD, BTCUSD (high volatility, wide spreads for scalping)
    """

    name = "hft_scalper"
    DEFAULT_PARAMS = {
        # Core scalping settings
        "timeframe": "M1",  # 1-minute candles
        "min_candles": 50,  # Enough for indicator warmup
        
        # ====================================================
        # EMA SETTINGS (5-8-13 Fibonacci Scalping System)
        # ====================================================
        # Classic scalping EMA setup based on Fibonacci numbers
        # Used by professional day traders worldwide
        "ema_fast": 5,      # Fast trend detection
        "ema_medium": 8,    # Medium-term momentum
        "ema_slow": 13,     # Trend filter
        "require_ema_alignment": True,  # All 3 EMAs must be stacked
        "ema_alignment_threshold": 0.0001,  # Min distance between EMAs
        
        # ====================================================
        # RSI SETTINGS (Fast Momentum Filter)
        # ====================================================
        # Shorter period RSI for quick scalping decisions
        "use_rsi": True,
        "rsi_period": 7,           # Fast RSI (vs standard 14)
        "rsi_overbought": 70,      # Overbought level
        "rsi_oversold": 30,        # Oversold level
        "rsi_extreme_ob": 80,      # Extreme overbought (fade zone)
        "rsi_extreme_os": 20,      # Extreme oversold (bounce zone)
        "rsi_momentum_zone": (40, 60),  # Neutral zone
        
        # ====================================================
        # STOCHASTIC SETTINGS (Fast Oscillator)
        # ====================================================
        # 5,3,3 fast stochastic for quick reversals
        "use_stochastic": True,
        "stoch_k_period": 5,       # %K period
        "stoch_d_period": 3,       # %D smoothing
        "stoch_smooth": 3,         # Additional smoothing
        "stoch_overbought": 80,
        "stoch_oversold": 20,
        
        # ====================================================
        # VWAP PROXY SETTINGS (Institutional Level)
        # ====================================================
        # Simulated VWAP using price-volume weighted average
        "use_vwap": True,
        "vwap_period": 20,         # Lookback for VWAP calculation
        "vwap_deviation_mult": 1.5, # Deviation bands multiplier
        "vwap_mean_reversion": True, # Trade towards VWAP
        
        # ====================================================
        # TAKE PROFIT / STOP LOSS TARGETS
        # ====================================================
        # KEY: TP must be > SL for positive expectancy
        # With 40% win rate: need R:R of at least 1.5:1
        # Fixed pip targets (in PIPS, not price units)
        "tp_pips": 10.0,    # Take profit in pips (larger than SL!)
        "sl_pips": 5.0,     # Stop loss in pips (tight)
        "breakeven_pips": 4.0,  # Move SL to breakeven after this profit
        
        # ATR-based targets (more adaptive to volatility)
        "use_atr_targets": True,
        "atr_period": 10,   # Shorter period for M1
        "tp_atr_mult": 1.8,  # TP = 1.8 x ATR (MUST be > SL for positive expectancy!)
        "sl_atr_mult": 0.8,  # SL = 0.8 x ATR (tight stop)
        
        # ====================================================
        # CHOPPY MARKET FILTER (Critical!)
        # ====================================================
        # When EMAs are too close together, market is ranging/choppy
        # These conditions produce many false signals - AVOID
        "min_ema_separation": 0.15,  # Minimum EMA spread in ATR units
        
        # ====================================================
        # SESSION FILTERING (Critical for Scalping)
        # ====================================================
        # DATA SHOWS: NY session has best win rates for HFT
        # London session underperforms - focus on NY overlap
        "sessions": ("NewYork",),  # Focus on NY session for best results
        "use_killzones": True,
        "killzone_filter": ("NY_AM", "NY_Lunch"),
        "require_killzone": False,
        
        # High liquidity windows (server time - GMT+2)
        # NY overlap: 14:00-20:00 server time (best for scalping)
        "high_liquidity_hours": [(14, 20)],
        
        # ====================================================
        # MOMENTUM REQUIREMENTS
        # ====================================================
        "min_momentum_strength": 0.15,  # 0.0-1.0 scale (lowered more)
        "min_trend_strength": 0.2,      # Minimum trend clarity (lowered more)
        
        # Consecutive candle analysis
        "min_consecutive_candles": 2,   # Min same-direction candles
        "max_consecutive_candles": 6,   # Avoid chasing extended moves
        
        # ====================================================
        # MICRO STRUCTURE
        # ====================================================
        "use_micro_fvg": True,
        "micro_fvg_max_size": 0.0008,   # Max FVG size relative to price
        "use_wick_rejection": True,      # Wick rejection patterns
        "wick_rejection_ratio": 0.6,     # Wick must be 60% of candle
        "use_engulfing": True,           # Engulfing patterns
        
        # ====================================================
        # SPREAD & VOLATILITY PROTECTION
        # ====================================================
        "max_spread_pips": 2.0,
        "spread_check": True,
        "min_atr_pips": 2.0,    # Need some movement (lowered)
        "max_atr_pips": 30.0,   # Avoid extreme volatility (raised)
        
        # ====================================================
        # TRADE MANAGEMENT
        # ====================================================
        "max_trades_per_session": 15,   # Prevent overtrading
        "min_minutes_between_trades": 3, # Faster cooldown for HFT
        "max_daily_trades": 50,          # Daily limit
        
        # Quick exit settings
        "use_quick_exit": True,
        "quick_exit_bars": 5,    # Exit if no move in 5 bars
        "quick_exit_threshold": 0.0,
        
        # ====================================================
        # CONFLUENCE & QUALITY
        # ====================================================
        "min_confluence": 3,     # Minimum confluence score
        "min_entry_quality": "B", # A, B, or C
        
        # Risk per trade
        "risk_percent": 1.0,  # 1% per trade (lower due to frequency)
        
        # ====================================================
        # MACD SETTINGS (Fast Scalping MACD)
        # ====================================================
        # Faster MACD settings for M1 scalping (6,13,5 vs standard 12,26,9)
        "use_macd": True,
        "macd_fast": 6,           # Fast EMA period
        "macd_slow": 13,          # Slow EMA period  
        "macd_signal": 5,         # Signal line period
        "macd_crossover_entry": True,  # Enter on crossover
        "macd_histogram_entry": True,  # Enter on histogram flip
        "macd_divergence_check": True, # Check for divergence
        
        # ====================================================
        # BOLLINGER BAND SETTINGS (Fast Volatility Filter)
        # ====================================================
        # Faster Bollinger for scalping (10-period vs standard 20)
        "use_bollinger": True,
        "bb_period": 10,          # SMA period for middle band
        "bb_std_dev": 2.0,        # Standard deviations
        "bb_squeeze_threshold": 0.001,  # Squeeze detection threshold
        "bb_breakout_entry": True,      # Enter on band breakout
        "bb_mean_reversion": True,      # Enter on return to middle
        "bb_touch_entry": True,         # Enter on band touch reversal
        
        # ====================================================
        # ADX SETTINGS (Average Directional Index - Trend Strength)
        # ====================================================
        # ADX measures trend strength from 0-100
        # Below 20 = no trend/weak, 20-25 = emerging, 25-50 = strong, 50+ = very strong
        "use_adx": True,
        "adx_period": 10,         # Faster ADX for scalping (vs standard 14)
        "adx_strong_trend": 25,   # ADX value for strong trend
        "adx_weak_trend": 20,     # ADX value below which trend is weak
        "adx_filter_trades": True, # Only trade when ADX > weak_trend
        "use_di_crossover": True,  # Use +DI/-DI crossovers for signals
        
        # ====================================================
        # ADVANCED ENTRY MODES
        # ====================================================
        # Pullback Entry Mode
        "use_pullback_entry": True,
        "pullback_ema_touch": True,  # Enter when price touches EMA
        "max_pullback_distance": 0.5, # Max 0.5 ATR from EMA
        
        # Breakout Entry Mode
        "use_breakout_entry": True,
        "breakout_candle_mult": 1.5,  # Candle must be 1.5x average
        
        # Mean Reversion Entry Mode
        "use_mean_reversion": True,
        "mean_reversion_threshold": 2.0,  # 2 std dev from mean
        
        # Debug
        "debug": False,
    }

    # Pip values for different instruments
    PIP_VALUES = {
        "EURUSD": 0.0001,
        "GBPUSD": 0.0001,
        "USDJPY": 0.01,
        "USDCHF": 0.0001,
        "AUDUSD": 0.0001,
        "NZDUSD": 0.0001,
        "USDCAD": 0.0001,
        "GBPJPY": 0.01,
        "EURJPY": 0.01,
        "XAUUSD": 0.01,   # Gold uses 0.01 (1 pip = $0.01)
        "BTCUSD": 1.0,    # BTC uses whole dollar
    }

    def __init__(self, context) -> None:
        super().__init__(context)
        self._candles: Deque[Candle] = deque(maxlen=200)  # Less history needed for scalping
        self.params = {**self.DEFAULT_PARAMS, **(self.context.parameters or {})}
        
        # Session tracking
        self._trades_this_session = 0
        self._last_trade_time: Optional[datetime] = None
        self._current_session: Optional[str] = None

    def on_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """Process incoming candle and generate scalping signal."""
        
        # Handle duplicate/update of forming candle
        if self._candles and self._candles[-1].timestamp == candle.timestamp:
            self._candles[-1] = candle
        else:
            self._candles.append(candle)
        
        min_candles = int(self.params["min_candles"])
        if len(self._candles) < min_candles:
            self._debug(f"warming up: {len(self._candles)}/{min_candles}")
            return None

        # Session check
        session_name = active_session(candle.timestamp, self.context.session_windows)
        session_whitelist = set(self.params.get("sessions", self.DEFAULT_PARAMS["sessions"]))
        
        if session_name not in session_whitelist:
            self._debug(f"session filtered: {session_name}")
            return None
        
        # Track session changes
        if session_name != self._current_session:
            self._current_session = session_name
            self._trades_this_session = 0
            self._debug(f"new session: {session_name}")
        
        # Max trades per session check
        max_trades = int(self.params.get("max_trades_per_session", 10))
        if self._trades_this_session >= max_trades:
            self._debug(f"max trades reached: {self._trades_this_session}/{max_trades}")
            return None
        
        # Cooldown check
        min_minutes = int(self.params.get("min_minutes_between_trades", 5))
        if self._last_trade_time:
            elapsed = (candle.timestamp - self._last_trade_time).total_seconds() / 60
            if elapsed < min_minutes:
                self._debug(f"cooldown: {elapsed:.1f}/{min_minutes} minutes")
                return None
        
        # Killzone check
        killzone = active_killzone(candle.timestamp) if self.params.get("use_killzones") else None
        allowed_killzones = set(self.params.get("killzone_filter", ()))
        
        if self.params.get("require_killzone"):
            if not killzone or (allowed_killzones and killzone.name not in allowed_killzones):
                self._debug(f"killzone required, current: {killzone.name if killzone else 'None'}")
                return None
        
        # Build scalping context
        scalp_ctx = self._build_scalp_context(session_name, candle, killzone)
        if scalp_ctx is None:
            return None
        
        # Confluence check
        min_confluence = int(self.params.get("min_confluence", 2))
        if scalp_ctx.confluence_score < min_confluence:
            self._debug(f"insufficient confluence: {scalp_ctx.confluence_score}/{min_confluence}")
            return None
        
        # Calculate entry, stop, and target
        entry = candle.close
        pip_value = self._get_pip_value()
        
        if self.params.get("use_atr_targets"):
            # ATR-based targets
            tp_distance = scalp_ctx.atr * float(self.params["tp_atr_mult"])
            sl_distance = scalp_ctx.atr * float(self.params["sl_atr_mult"])
        else:
            # Fixed pip targets
            tp_distance = float(self.params["tp_pips"]) * pip_value
            sl_distance = float(self.params["sl_pips"]) * pip_value
        
        if scalp_ctx.direction is OrderSide.BUY:
            stop_loss = entry - sl_distance
            take_profit = entry + tp_distance
        else:
            stop_loss = entry + sl_distance
            take_profit = entry - tp_distance
        
        # Validate prices
        if stop_loss <= 0 or take_profit <= 0:
            self._debug("invalid stop/tp")
            return None
        
        if (scalp_ctx.direction is OrderSide.BUY and take_profit <= entry) or \
           (scalp_ctx.direction is OrderSide.SELL and take_profit >= entry):
            self._debug("invalid tp direction")
            return None
        
        # Calculate confidence
        confidence = min(scalp_ctx.confluence_score / 5.0, 1.0)
        
        # Build signal
        signal = StrategySignal(
            instrument=self.context.instrument,
            side=scalp_ctx.direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=candle.timestamp,
            confidence=confidence,
            session=scalp_ctx.session,
        )
        
        # Update tracking
        self._trades_this_session += 1
        self._last_trade_time = candle.timestamp
        
        self._debug(
            "SCALP SIGNAL | %s | entry=%.5f sl=%.5f tp=%.5f | confluence=%d | %s",
            scalp_ctx.direction.value.upper(),
            entry,
            stop_loss,
            take_profit,
            scalp_ctx.confluence_score,
            self._describe_setup(scalp_ctx),
        )
        
        return signal

    def _build_scalp_context(
        self,
        session_name: str,
        candle: Candle,
        killzone: Optional[Killzone],
    ) -> Optional[ScalpContext]:
        """Build scalping context with advanced momentum and micro-structure analysis.
        
        This method implements the full HFT analysis:
        1. 5-8-13 EMA alignment check
        2. RSI momentum filter
        3. Stochastic oscillator signals
        4. VWAP proxy calculation
        5. Micro-structure patterns
        6. Trend strength calculation
        """
        
        candles = list(self._candles)
        confluence_score = 0
        pip_value = self._get_pip_value()
        
        # ========================================
        # VOLATILITY & ATR ANALYSIS
        # ========================================
        atr_period = int(self.params.get("atr_period", 10))
        atr = calculate_atr(candles, period=atr_period)
        atr_pips = atr / pip_value if pip_value > 0 else 0
        
        min_atr = float(self.params.get("min_atr_pips", 3.0))
        max_atr = float(self.params.get("max_atr_pips", 25.0))
        volatility_ok = min_atr <= atr_pips <= max_atr
        
        if not volatility_ok:
            self._debug(f"volatility out of range: {atr_pips:.1f} pips (need {min_atr}-{max_atr})")
            return None
        
        confluence_score += 1  # Volatility OK
        
        # ========================================
        # SPREAD ESTIMATION
        # ========================================
        recent_ranges = [c.high - c.low for c in candles[-5:]]
        avg_range = sum(recent_ranges) / len(recent_ranges) if recent_ranges else atr
        spread_estimate = avg_range * 0.05  # Conservative estimate
        spread_pips = spread_estimate / pip_value if pip_value > 0 else 0
        
        max_spread = float(self.params.get("max_spread_pips", 2.0))
        if self.params.get("spread_check") and spread_pips > max_spread:
            self._debug(f"spread too wide: {spread_pips:.1f} pips")
        else:
            confluence_score += 1  # Spread OK
        
        # ========================================
        # 5-8-13 EMA CALCULATION
        # ========================================
        ema_fast = self._calculate_ema(candles, int(self.params["ema_fast"]))
        ema_medium = self._calculate_ema(candles, int(self.params["ema_medium"]))
        ema_slow = self._calculate_ema(candles, int(self.params["ema_slow"]))
        
        if ema_fast is None or ema_medium is None or ema_slow is None:
            self._debug("insufficient data for EMAs")
            return None
        
        # EMA alignment analysis
        ema_threshold = float(self.params.get("ema_alignment_threshold", 0.0001))
        ema_spread = abs(ema_fast - ema_slow)
        price = candle.close
        
        # Determine trend from EMA stack
        if ema_fast > ema_medium > ema_slow:
            ema_trend = "bullish"
            ema_aligned = True
            if ema_spread > ema_threshold:
                confluence_score += 2  # Strong bullish EMA alignment
        elif ema_fast < ema_medium < ema_slow:
            ema_trend = "bearish"
            ema_aligned = True
            if ema_spread > ema_threshold:
                confluence_score += 2  # Strong bearish EMA alignment
        else:
            ema_trend = "neutral"
            ema_aligned = False
            if self.params.get("require_ema_alignment"):
                self._debug("EMAs not aligned - skipping")
                return None
        
        # ========================================
        # RSI CALCULATION (Fast 7-period)
        # ========================================
        rsi_value = 50.0
        rsi_signal = "neutral"
        
        if self.params.get("use_rsi"):
            rsi_value = self._calculate_rsi(candles, int(self.params.get("rsi_period", 7)))
            ob = float(self.params.get("rsi_overbought", 70))
            os = float(self.params.get("rsi_oversold", 30))
            extreme_ob = float(self.params.get("rsi_extreme_ob", 80))
            extreme_os = float(self.params.get("rsi_extreme_os", 20))
            
            if rsi_value >= extreme_ob:
                rsi_signal = "extreme_overbought"
                confluence_score += 1 if ema_trend == "bearish" else 0  # Fade signal
            elif rsi_value >= ob:
                rsi_signal = "overbought"
            elif rsi_value <= extreme_os:
                rsi_signal = "extreme_oversold"
                confluence_score += 1 if ema_trend == "bullish" else 0  # Bounce signal
            elif rsi_value <= os:
                rsi_signal = "oversold"
            else:
                rsi_signal = "neutral"
            
            # RSI confirms trend
            if (ema_trend == "bullish" and rsi_value > 50) or \
               (ema_trend == "bearish" and rsi_value < 50):
                confluence_score += 1
        
        # ========================================
        # STOCHASTIC OSCILLATOR (5,3,3)
        # ========================================
        stoch_k = 50.0
        stoch_d = 50.0
        stoch_signal = "neutral"
        
        if self.params.get("use_stochastic"):
            stoch_k, stoch_d = self._calculate_stochastic(
                candles,
                k_period=int(self.params.get("stoch_k_period", 5)),
                d_period=int(self.params.get("stoch_d_period", 3)),
                smooth=int(self.params.get("stoch_smooth", 3)),
            )
            
            stoch_ob = float(self.params.get("stoch_overbought", 80))
            stoch_os = float(self.params.get("stoch_oversold", 20))
            
            # Stochastic crossover signals
            if stoch_k > stoch_d and stoch_k < stoch_ob:
                stoch_signal = "buy"
                if ema_trend == "bullish":
                    confluence_score += 1
            elif stoch_k < stoch_d and stoch_k > stoch_os:
                stoch_signal = "sell"
                if ema_trend == "bearish":
                    confluence_score += 1
            elif stoch_k >= stoch_ob:
                stoch_signal = "overbought"
            elif stoch_k <= stoch_os:
                stoch_signal = "oversold"
        
        # ========================================
        # VWAP PROXY CALCULATION
        # ========================================
        vwap_proxy = price
        price_vs_vwap = "at"
        vwap_deviation = 0.0
        
        if self.params.get("use_vwap"):
            vwap_period = int(self.params.get("vwap_period", 20))
            vwap_proxy = self._calculate_vwap_proxy(candles, vwap_period)
            
            vwap_diff = price - vwap_proxy
            vwap_deviation = abs(vwap_diff) / atr if atr > 0 else 0
            
            if vwap_diff > atr * 0.25:
                price_vs_vwap = "above"
            elif vwap_diff < -atr * 0.25:
                price_vs_vwap = "below"
            else:
                price_vs_vwap = "at"
            
            # VWAP confluence
            if (ema_trend == "bullish" and price_vs_vwap == "above") or \
               (ema_trend == "bearish" and price_vs_vwap == "below"):
                confluence_score += 1
        
        # ========================================
        # MOMENTUM STRENGTH
        # ========================================
        ema_distance = abs(price - ema_slow) / atr if atr > 0 else 0
        momentum_strength = min(ema_distance, 1.0)
        
        min_momentum = float(self.params.get("min_momentum_strength", 0.25))
        if momentum_strength < min_momentum:
            self._debug(f"momentum too weak: {momentum_strength:.2f}")
            return None
        
        if momentum_strength > 0.5:
            confluence_score += 1
        
        # ========================================
        # CONSECUTIVE CANDLE ANALYSIS
        # ========================================
        consecutive_bullish = 0
        consecutive_bearish = 0
        
        for c in reversed(candles[-10:]):
            if c.close > c.open:
                if consecutive_bearish > 0:
                    break
                consecutive_bullish += 1
            elif c.close < c.open:
                if consecutive_bullish > 0:
                    break
                consecutive_bearish += 1
            else:
                break
        
        min_consecutive = int(self.params.get("min_consecutive_candles", 2))
        max_consecutive = int(self.params.get("max_consecutive_candles", 6))
        
        # ========================================
        # WICK REJECTION PATTERN
        # ========================================
        wick_rejection = False
        if self.params.get("use_wick_rejection"):
            wick_ratio = float(self.params.get("wick_rejection_ratio", 0.6))
            body = abs(candle.close - candle.open)
            total_range = candle.high - candle.low
            
            if total_range > 0:
                upper_wick = candle.high - max(candle.close, candle.open)
                lower_wick = min(candle.close, candle.open) - candle.low
                
                # Bullish rejection (long lower wick)
                if lower_wick / total_range >= wick_ratio and ema_trend == "bullish":
                    wick_rejection = True
                    confluence_score += 1
                    self._debug("bullish wick rejection detected")
                # Bearish rejection (long upper wick)
                elif upper_wick / total_range >= wick_ratio and ema_trend == "bearish":
                    wick_rejection = True
                    confluence_score += 1
                    self._debug("bearish wick rejection detected")
        
        # ========================================
        # ENGULFING PATTERN
        # ========================================
        engulfing_pattern = False
        if self.params.get("use_engulfing") and len(candles) >= 2:
            prev = candles[-2]
            curr = candles[-1]
            
            # Bullish engulfing
            if prev.close < prev.open and curr.close > curr.open:
                if curr.open <= prev.close and curr.close >= prev.open:
                    if ema_trend == "bullish":
                        engulfing_pattern = True
                        confluence_score += 2
                        self._debug("bullish engulfing pattern")
            # Bearish engulfing
            elif prev.close > prev.open and curr.close < curr.open:
                if curr.open >= prev.close and curr.close <= prev.open:
                    if ema_trend == "bearish":
                        engulfing_pattern = True
                        confluence_score += 2
                        self._debug("bearish engulfing pattern")
        
        # ========================================
        # CHOPPY MARKET FILTER (Critical for HFT)
        # ========================================
        # Avoid trading when EMAs are compressed - sign of ranging/choppy market
        ema_range = max(ema_fast, ema_medium, ema_slow) - min(ema_fast, ema_medium, ema_slow)
        ema_range_atr = ema_range / atr if atr > 0 else 0
        
        min_ema_separation = float(self.params.get("min_ema_separation", 0.1))  # 0.1 ATR
        if ema_range_atr < min_ema_separation:
            self._debug(f"EMAs too compressed (choppy market): {ema_range_atr:.3f} ATR")
            return None
        
        # ========================================
        # DIRECTION DETERMINATION - PULLBACK BASED
        # ========================================
        # KEY INSIGHT: We want to enter on PULLBACKS to the EMA cluster,
        # NOT on breakouts or trend continuation. This is the HFT edge.
        direction = None
        pullback_quality = 0
        
        # Calculate price position relative to EMA cluster
        ema_cluster_mid = (ema_fast + ema_medium + ema_slow) / 3
        price_to_cluster = (price - ema_cluster_mid) / atr if atr > 0 else 0
        
        if ema_trend == "bullish":
            # In uptrend, we BUY when price pulls back TO the EMA cluster
            # Perfect entry: price just touched or slightly below fast EMA
            if price <= ema_fast and price >= ema_slow:
                # Price is within the EMA cluster - excellent pullback entry
                direction = OrderSide.BUY
                pullback_quality = 3
                confluence_score += 3
                self._debug(f"PULLBACK BUY: price in EMA cluster")
            elif price < ema_slow and (ema_slow - price) / atr < 0.5:
                # Price slightly below cluster - aggressive pullback entry
                direction = OrderSide.BUY
                pullback_quality = 2
                confluence_score += 2
                self._debug(f"PULLBACK BUY: price below cluster")
            elif price > ema_fast and (price - ema_fast) / atr < 0.3:
                # Price just above fast EMA - momentum continuation
                if stoch_signal == "buy" or rsi_signal == "oversold":
                    direction = OrderSide.BUY
                    pullback_quality = 1
                    confluence_score += 1
                    self._debug(f"MOMENTUM BUY: confirmed by oscillators")
                    
        elif ema_trend == "bearish":
            # In downtrend, we SELL when price pulls back TO the EMA cluster
            if price >= ema_fast and price <= ema_slow:
                # Price is within the EMA cluster - excellent pullback entry
                direction = OrderSide.SELL
                pullback_quality = 3
                confluence_score += 3
                self._debug(f"PULLBACK SELL: price in EMA cluster")
            elif price > ema_slow and (price - ema_slow) / atr < 0.5:
                # Price slightly above cluster - aggressive pullback entry
                direction = OrderSide.SELL
                pullback_quality = 2
                confluence_score += 2
                self._debug(f"PULLBACK SELL: price above cluster")
            elif price < ema_fast and (ema_fast - price) / atr < 0.3:
                # Price just below fast EMA - momentum continuation
                if stoch_signal == "sell" or rsi_signal == "overbought":
                    direction = OrderSide.SELL
                    pullback_quality = 1
                    confluence_score += 1
                    self._debug(f"MOMENTUM SELL: confirmed by oscillators")
        
        if direction is None:
            self._debug(f"no pullback entry: trend={ema_trend}, price_to_cluster={price_to_cluster:.2f}")
            return None
        
        # Bonus for high-quality pullbacks
        if pullback_quality >= 2:
            confluence_score += 1
        
        # ========================================
        # MICRO FVG DETECTION
        # ========================================
        micro_fvg = None
        if self.params.get("use_micro_fvg"):
            all_gaps = detect_fair_value_gaps(candles[-10:])
            max_size = float(self.params.get("micro_fvg_max_size", 0.0008))
            
            for gap in reversed(all_gaps):
                gap_size = (gap.upper - gap.lower) / candle.close
                if gap_size <= max_size:
                    if gap.lower <= candle.close <= gap.upper:
                        micro_fvg = gap
                        confluence_score += 1
                        self._debug(f"micro FVG: {gap.direction}")
                        break
        
        # ========================================
        # LIQUIDITY & SESSION ANALYSIS
        # ========================================
        is_high_liquidity = self._is_high_liquidity_time(candle.timestamp)
        if is_high_liquidity:
            confluence_score += 1
            self._debug("high liquidity window")
        
        # BONUS: Session overlap is BEST for scalping
        is_overlap = self._is_session_overlap(candle.timestamp)
        if is_overlap:
            confluence_score += 2  # Strong bonus for overlap period
            self._debug("London/NY overlap - PRIME scalping time")
        
        if killzone:
            allowed_killzones = set(self.params.get("killzone_filter", ()))
            if killzone.name in allowed_killzones:
                confluence_score += 1
        
        # ========================================
        # MACD CALCULATION (6,13,5 Fast Settings)
        # ========================================
        macd_line = 0.0
        macd_signal_line = 0.0
        macd_histogram = 0.0
        macd_crossover = "none"
        macd_momentum = "neutral"
        
        if self.params.get("use_macd"):
            macd_line, macd_signal_line, macd_histogram, macd_crossover, macd_momentum = \
                self._calculate_macd(
                    candles,
                    fast_period=int(self.params.get("macd_fast", 6)),
                    slow_period=int(self.params.get("macd_slow", 13)),
                    signal_period=int(self.params.get("macd_signal", 5)),
                )
            
            # MACD confluence
            if macd_crossover == "bullish" and ema_trend == "bullish":
                confluence_score += 2
                self._debug("MACD bullish crossover in bullish trend")
            elif macd_crossover == "bearish" and ema_trend == "bearish":
                confluence_score += 2
                self._debug("MACD bearish crossover in bearish trend")
            
            # MACD momentum confirmation
            if (ema_trend == "bullish" and macd_momentum == "increasing") or \
               (ema_trend == "bearish" and macd_momentum == "decreasing"):
                confluence_score += 1
        
        # ========================================
        # BOLLINGER BANDS (10-period, 2 std)
        # ========================================
        bb_upper = price
        bb_middle = price
        bb_lower = price
        bb_position = "at_middle"
        bb_squeeze = False
        bb_width = 0.0
        
        if self.params.get("use_bollinger"):
            bb_upper, bb_middle, bb_lower, bb_position, bb_squeeze, bb_width = \
                self._calculate_bollinger_bands(
                    candles,
                    period=int(self.params.get("bb_period", 10)),
                    std_dev=float(self.params.get("bb_std_dev", 2.0)),
                )
            
            # Bollinger Band confluence
            if self.params.get("bb_mean_reversion"):
                # Mean reversion: price at band extremes
                if bb_position == "below_lower" and ema_trend == "bullish":
                    confluence_score += 2
                    self._debug("BB mean reversion buy signal")
                elif bb_position == "above_upper" and ema_trend == "bearish":
                    confluence_score += 2
                    self._debug("BB mean reversion sell signal")
            
            if self.params.get("bb_touch_entry"):
                # Band touch with reversal
                if bb_position == "below_lower" and stoch_signal == "buy":
                    confluence_score += 1
                elif bb_position == "above_upper" and stoch_signal == "sell":
                    confluence_score += 1
            
            # Squeeze detection - potential breakout coming
            if bb_squeeze:
                self._debug(f"BB squeeze detected, width={bb_width:.6f}")
        
        # ========================================
        # ADX CALCULATION (Trend Strength Filter)
        # ========================================
        adx_value = 0.0
        plus_di = 50.0
        minus_di = 50.0
        adx_trend_strength = "none"
        di_crossover = "none"
        
        if self.params.get("use_adx"):
            adx_value, plus_di, minus_di, adx_trend_strength, di_crossover = \
                self._calculate_adx(
                    candles,
                    period=int(self.params.get("adx_period", 10)),
                )
            
            # ADX confluence
            if adx_trend_strength in ("strong", "very_strong"):
                confluence_score += 1
                self._debug(f"ADX strong trend: {adx_value:.1f}")
            
            # DI crossover confluence
            if self.params.get("use_di_crossover"):
                if di_crossover == "bullish" and ema_trend == "bullish":
                    confluence_score += 2
                    self._debug("ADX +DI bullish crossover")
                elif di_crossover == "bearish" and ema_trend == "bearish":
                    confluence_score += 2
                    self._debug("ADX -DI bearish crossover")
            
            # Filter out weak trend trades
            if self.params.get("adx_filter_trades"):
                weak_level = float(self.params.get("adx_weak_trend", 20))
                if adx_value < weak_level:
                    self._debug(f"ADX too weak: {adx_value:.1f} < {weak_level}")
                    # Don't return None - just reduce confluence
                    confluence_score -= 1
        
        # ========================================
        # TREND STRENGTH & ENTRY QUALITY
        # ========================================
        trend_strength = self._calculate_trend_strength(candles, atr)
        
        # Grade the entry
        if confluence_score >= 8:
            entry_quality = "A"
        elif confluence_score >= 5:
            entry_quality = "B"
        else:
            entry_quality = "C"
        
        min_quality = str(self.params.get("min_entry_quality", "B"))
        if entry_quality > min_quality:  # String comparison: "C" > "B" > "A"
            self._debug(f"entry quality too low: {entry_quality}")
            return None
        
        return ScalpContext(
            direction=direction,
            ema_fast=ema_fast,
            ema_medium=ema_medium,
            ema_slow=ema_slow,
            ema_trend=ema_trend,
            ema_aligned=ema_aligned,
            ema_spread=ema_spread,
            momentum_strength=momentum_strength,
            rsi_value=rsi_value,
            rsi_signal=rsi_signal,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_signal=stoch_signal,
            macd_line=macd_line,
            macd_signal=macd_signal_line,
            macd_histogram=macd_histogram,
            macd_crossover=macd_crossover,
            macd_momentum=macd_momentum,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_position=bb_position,
            bb_squeeze=bb_squeeze,
            bb_width=bb_width,
            adx_value=adx_value,
            plus_di=plus_di,
            minus_di=minus_di,
            adx_trend_strength=adx_trend_strength,
            di_crossover=di_crossover,
            vwap_proxy=vwap_proxy,
            price_vs_vwap=price_vs_vwap,
            vwap_deviation=vwap_deviation,
            atr=atr,
            spread_estimate=spread_estimate,
            volatility_ok=volatility_ok,
            session=session_name,
            killzone=killzone,
            is_high_liquidity=is_high_liquidity,
            micro_fvg=micro_fvg,
            consecutive_bullish=consecutive_bullish,
            consecutive_bearish=consecutive_bearish,
            wick_rejection=wick_rejection,
            engulfing_pattern=engulfing_pattern,
            trend_strength=trend_strength,
            entry_quality=entry_quality,
            confluence_score=confluence_score,
        )

    def _calculate_ema(self, candles: List[Candle], period: int) -> Optional[float]:
        """Calculate EMA from candle closes."""
        if len(candles) < period:
            return None
        
        closes = [c.close for c in candles]
        multiplier = 2 / (period + 1)
        ema = sum(closes[:period]) / period
        
        for price in closes[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    def _calculate_rsi(self, candles: List[Candle], period: int = 7) -> float:
        """Calculate Relative Strength Index.
        
        Fast RSI with configurable period (default 7 for scalping).
        """
        if len(candles) < period + 1:
            return 50.0
        
        closes = [c.close for c in candles]
        
        gains = []
        losses = []
        
        for i in range(1, len(closes)):
            change = closes[i] - closes[i-1]
            if change > 0:
                gains.append(change)
                losses.append(0)
            else:
                gains.append(0)
                losses.append(abs(change))
        
        if len(gains) < period:
            return 50.0
        
        # Calculate initial average gain/loss
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        # Smooth with Wilder's method
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi

    def _calculate_stochastic(
        self,
        candles: List[Candle],
        k_period: int = 5,
        d_period: int = 3,
        smooth: int = 3,
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator %K and %D.
        
        Fast stochastic (5,3,3) for quick scalping signals.
        Returns (stoch_k, stoch_d).
        """
        if len(candles) < k_period + d_period:
            return 50.0, 50.0
        
        # Calculate raw %K values
        raw_k_values = []
        
        for i in range(k_period - 1, len(candles)):
            window = candles[i - k_period + 1:i + 1]
            highest = max(c.high for c in window)
            lowest = min(c.low for c in window)
            current_close = candles[i].close
            
            if highest == lowest:
                raw_k_values.append(50.0)
            else:
                k = ((current_close - lowest) / (highest - lowest)) * 100
                raw_k_values.append(k)
        
        if len(raw_k_values) < smooth:
            return 50.0, 50.0
        
        # Smooth %K
        smoothed_k = []
        for i in range(smooth - 1, len(raw_k_values)):
            avg = sum(raw_k_values[i - smooth + 1:i + 1]) / smooth
            smoothed_k.append(avg)
        
        if len(smoothed_k) < d_period:
            return smoothed_k[-1] if smoothed_k else 50.0, 50.0
        
        # Calculate %D (SMA of smoothed %K)
        stoch_k = smoothed_k[-1]
        stoch_d = sum(smoothed_k[-d_period:]) / d_period
        
        return stoch_k, stoch_d

    def _calculate_vwap_proxy(self, candles: List[Candle], period: int = 20) -> float:
        """Calculate VWAP proxy using typical price weighted by range.
        
        Since we may not have tick volume, we use the candle range
        as a proxy for volume (larger range = more activity).
        """
        if len(candles) < period:
            period = len(candles)
        
        recent = candles[-period:]
        
        total_weighted_price = 0.0
        total_weight = 0.0
        
        for c in recent:
            typical_price = (c.high + c.low + c.close) / 3
            weight = c.high - c.low  # Range as volume proxy
            
            if weight <= 0:
                weight = 0.0001  # Minimum weight
            
            total_weighted_price += typical_price * weight
            total_weight += weight
        
        if total_weight == 0:
            return candles[-1].close
        
        return total_weighted_price / total_weight

    def _calculate_trend_strength(self, candles: List[Candle], atr: float) -> float:
        """Calculate trend strength from 0.0 to 1.0.
        
        Uses price movement consistency and momentum.
        """
        if len(candles) < 20 or atr <= 0:
            return 0.5
        
        recent = candles[-20:]
        
        # Count directional candles
        bullish = sum(1 for c in recent if c.close > c.open)
        bearish = sum(1 for c in recent if c.close < c.open)
        
        # Trend consistency
        consistency = abs(bullish - bearish) / len(recent)
        
        # Price progression
        start_price = recent[0].close
        end_price = recent[-1].close
        price_change = abs(end_price - start_price) / (atr * 20)  # Normalized by ATR
        
        # Combine metrics
        trend_strength = min((consistency + price_change) / 2, 1.0)
        
        return trend_strength

    def _calculate_macd(
        self,
        candles: List[Candle],
        fast_period: int = 6,
        slow_period: int = 13,
        signal_period: int = 5,
    ) -> Tuple[float, float, float, str, str]:
        """Calculate MACD (Moving Average Convergence/Divergence).
        
        Using faster settings for scalping (6,13,5 vs standard 12,26,9).
        
        Returns:
            Tuple of (macd_line, signal_line, histogram, crossover, momentum)
            - macd_line: MACD line value
            - signal_line: Signal line value
            - histogram: MACD - Signal
            - crossover: "bullish", "bearish", or "none"
            - momentum: "increasing", "decreasing", or "neutral"
        """
        if len(candles) < slow_period + signal_period:
            return 0.0, 0.0, 0.0, "none", "neutral"
        
        closes = [c.close for c in candles]
        
        # Calculate fast and slow EMAs
        fast_ema = self._ema_series(closes, fast_period)
        slow_ema = self._ema_series(closes, slow_period)
        
        if not fast_ema or not slow_ema:
            return 0.0, 0.0, 0.0, "none", "neutral"
        
        # Calculate MACD line (fast EMA - slow EMA)
        min_len = min(len(fast_ema), len(slow_ema))
        macd_line_series = [
            fast_ema[-(min_len - i)] - slow_ema[-(min_len - i)]
            for i in range(min_len)
        ]
        
        if len(macd_line_series) < signal_period:
            return 0.0, 0.0, 0.0, "none", "neutral"
        
        # Calculate signal line (EMA of MACD line)
        signal_line_series = self._ema_series(macd_line_series, signal_period)
        
        if not signal_line_series:
            return 0.0, 0.0, 0.0, "none", "neutral"
        
        macd_line = macd_line_series[-1]
        signal_line = signal_line_series[-1]
        histogram = macd_line - signal_line
        
        # Detect crossover
        crossover = "none"
        if len(macd_line_series) >= 2 and len(signal_line_series) >= 2:
            prev_macd = macd_line_series[-2]
            prev_signal = signal_line_series[-2]
            
            # Bullish crossover: MACD crosses above signal
            if prev_macd <= prev_signal and macd_line > signal_line:
                crossover = "bullish"
            # Bearish crossover: MACD crosses below signal
            elif prev_macd >= prev_signal and macd_line < signal_line:
                crossover = "bearish"
        
        # Detect momentum direction from histogram
        momentum = "neutral"
        if len(macd_line_series) >= 3 and len(signal_line_series) >= 3:
            hist_current = histogram
            hist_prev = macd_line_series[-2] - signal_line_series[-2]
            hist_prev2 = macd_line_series[-3] - signal_line_series[-3]
            
            if hist_current > hist_prev > hist_prev2:
                momentum = "increasing"
            elif hist_current < hist_prev < hist_prev2:
                momentum = "decreasing"
        
        return macd_line, signal_line, histogram, crossover, momentum

    def _ema_series(self, data: List[float], period: int) -> List[float]:
        """Calculate EMA series for given data."""
        if len(data) < period:
            return []
        
        multiplier = 2 / (period + 1)
        ema_values = [sum(data[:period]) / period]
        
        for price in data[period:]:
            ema = (price - ema_values[-1]) * multiplier + ema_values[-1]
            ema_values.append(ema)
        
        return ema_values

    def _calculate_bollinger_bands(
        self,
        candles: List[Candle],
        period: int = 10,
        std_dev: float = 2.0,
    ) -> Tuple[float, float, float, str, bool, float]:
        """Calculate Bollinger Bands.
        
        Using faster settings for scalping (10-period vs standard 20).
        
        Returns:
            Tuple of (upper, middle, lower, position, squeeze, width)
            - upper: Upper band
            - middle: Middle band (SMA)
            - lower: Lower band
            - position: Price position relative to bands
            - squeeze: Whether bands are in a squeeze (low volatility)
            - width: Band width (normalized by price)
        """
        if len(candles) < period:
            price = candles[-1].close if candles else 0
            return price, price, price, "at_middle", False, 0.0
        
        closes = [c.close for c in candles[-period:]]
        
        # Calculate SMA (middle band)
        middle = sum(closes) / period
        
        # Calculate standard deviation
        variance = sum((x - middle) ** 2 for x in closes) / period
        std = variance ** 0.5
        
        # Calculate upper and lower bands
        upper = middle + (std_dev * std)
        lower = middle - (std_dev * std)
        
        # Current price position
        price = candles[-1].close
        position = "at_middle"
        
        if price > upper:
            position = "above_upper"
        elif price > middle:
            position = "above_middle"
        elif price < lower:
            position = "below_lower"
        elif price < middle:
            position = "below_middle"
        
        # Band width (normalized by middle)
        width = (upper - lower) / middle if middle > 0 else 0
        
        # Squeeze detection (low volatility)
        squeeze_threshold = float(self.params.get("bb_squeeze_threshold", 0.001))
        squeeze = width < squeeze_threshold
        
        return upper, middle, lower, position, squeeze, width

    def _calculate_adx(
        self,
        candles: List[Candle],
        period: int = 10,
    ) -> Tuple[float, float, float, str, str]:
        """Calculate ADX (Average Directional Index) with +DI and -DI.
        
        ADX measures trend STRENGTH (not direction):
        - 0-20: Weak or no trend (range-bound market)
        - 20-25: Emerging trend
        - 25-50: Strong trend
        - 50-75: Very strong trend
        - 75-100: Extremely strong trend
        
        +DI and -DI measure trend DIRECTION:
        - +DI > -DI: Uptrend
        - -DI > +DI: Downtrend
        
        Returns:
            Tuple of (adx, plus_di, minus_di, trend_strength, di_crossover)
        """
        if len(candles) < period * 2:
            return 0.0, 50.0, 50.0, "none", "none"
        
        # Calculate True Range and Directional Movement
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            curr = candles[i]
            prev = candles[i - 1]
            
            # True Range
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev.close),
                abs(curr.low - prev.close)
            )
            tr_list.append(tr)
            
            # Directional Movement
            up_move = curr.high - prev.high
            down_move = prev.low - curr.low
            
            if up_move > down_move and up_move > 0:
                plus_dm_list.append(up_move)
            else:
                plus_dm_list.append(0.0)
            
            if down_move > up_move and down_move > 0:
                minus_dm_list.append(down_move)
            else:
                minus_dm_list.append(0.0)
        
        if len(tr_list) < period:
            return 0.0, 50.0, 50.0, "none", "none"
        
        # Smooth TR and DM using Wilder's smoothing
        def wilder_smooth(data: List[float], period: int) -> List[float]:
            if len(data) < period:
                return []
            result = [sum(data[:period])]
            for i in range(period, len(data)):
                smoothed = result[-1] - (result[-1] / period) + data[i]
                result.append(smoothed)
            return result
        
        smooth_tr = wilder_smooth(tr_list, period)
        smooth_plus_dm = wilder_smooth(plus_dm_list, period)
        smooth_minus_dm = wilder_smooth(minus_dm_list, period)
        
        if not smooth_tr or smooth_tr[-1] == 0:
            return 0.0, 50.0, 50.0, "none", "none"
        
        # Calculate +DI and -DI
        plus_di = (smooth_plus_dm[-1] / smooth_tr[-1]) * 100
        minus_di = (smooth_minus_dm[-1] / smooth_tr[-1]) * 100
        
        # Calculate DX
        di_sum = plus_di + minus_di
        if di_sum == 0:
            dx_list = [0.0]
        else:
            dx_list = []
            for i in range(len(smooth_tr)):
                pdi = (smooth_plus_dm[i] / smooth_tr[i]) * 100 if smooth_tr[i] > 0 else 0
                mdi = (smooth_minus_dm[i] / smooth_tr[i]) * 100 if smooth_tr[i] > 0 else 0
                di_s = pdi + mdi
                if di_s > 0:
                    dx = (abs(pdi - mdi) / di_s) * 100
                else:
                    dx = 0.0
                dx_list.append(dx)
        
        # Calculate ADX (smoothed DX)
        if len(dx_list) < period:
            adx = dx_list[-1] if dx_list else 0.0
        else:
            adx_values = wilder_smooth(dx_list, period)
            adx = adx_values[-1] / period if adx_values else 0.0
        
        # Determine trend strength
        strong_level = float(self.params.get("adx_strong_trend", 25))
        weak_level = float(self.params.get("adx_weak_trend", 20))
        
        if adx >= 50:
            trend_strength = "very_strong"
        elif adx >= strong_level:
            trend_strength = "strong"
        elif adx >= weak_level:
            trend_strength = "moderate"
        else:
            trend_strength = "weak"
        
        # Determine DI crossover
        di_crossover = "none"
        if len(smooth_plus_dm) >= 2 and len(smooth_minus_dm) >= 2:
            prev_plus_di = (smooth_plus_dm[-2] / smooth_tr[-2]) * 100 if smooth_tr[-2] > 0 else 0
            prev_minus_di = (smooth_minus_dm[-2] / smooth_tr[-2]) * 100 if smooth_tr[-2] > 0 else 0
            
            # Bullish crossover: +DI crosses above -DI
            if prev_plus_di <= prev_minus_di and plus_di > minus_di:
                di_crossover = "bullish"
            # Bearish crossover: -DI crosses above +DI
            elif prev_plus_di >= prev_minus_di and plus_di < minus_di:
                di_crossover = "bearish"
        
        return adx, plus_di, minus_di, trend_strength, di_crossover

    def _get_pip_value(self) -> float:
        """Get pip value for current instrument."""
        instrument = self.context.instrument
        return self.PIP_VALUES.get(instrument, 0.0001)

    def _is_high_liquidity_time(self, timestamp: datetime) -> bool:
        """Check if current time is in a high liquidity window.
        
        Best scalping windows (in UTC):
        - London Open: 07:00-09:00 UTC (max volatility)
        - London/NY Overlap: 12:00-16:00 UTC (BEST - highest volume)
        - NY Afternoon: 16:00-20:00 UTC (still good)
        """
        hour = timestamp.hour  # Assuming UTC/GMT
        
        high_hours = self.params.get("high_liquidity_hours", [(7, 9), (12, 20)])
        for start, end in high_hours:
            if start <= hour < end:
                return True
        return False

    def _is_session_overlap(self, timestamp: datetime) -> bool:
        """Check if we're in London/NY session overlap - BEST for scalping.
        
        The overlap period (12:00-16:00 UTC) has:
        - Highest trading volume
        - Tightest spreads
        - Best price action
        """
        hour = timestamp.hour
        # London/NY overlap: 12:00-16:00 UTC (adjust for server time if needed)
        # Using 14:00-20:00 for GMT+2 server time
        return 14 <= hour <= 20

    def _describe_setup(self, ctx: ScalpContext) -> str:
        """Create human-readable description of the HFT setup."""
        parts = []
        
        # EMA status
        if ctx.ema_aligned:
            parts.append(f"EMA:{ctx.ema_trend[:4].upper()}")
        
        # RSI signal
        if ctx.rsi_signal != "neutral":
            parts.append(f"RSI:{ctx.rsi_value:.0f}")
        
        # Stochastic signal
        if ctx.stoch_signal in ("buy", "sell"):
            parts.append(f"STOCH:{ctx.stoch_signal.upper()}")
        
        # MACD status
        if ctx.macd_crossover != "none":
            parts.append(f"MACD:{ctx.macd_crossover[:4].upper()}")
        elif ctx.macd_momentum != "neutral":
            parts.append(f"MACD_MOM:{ctx.macd_momentum[:3].upper()}")
        
        # Bollinger Bands
        if ctx.bb_squeeze:
            parts.append("BB_SQZ")
        elif ctx.bb_position in ("above_upper", "below_lower"):
            parts.append(f"BB:{ctx.bb_position[6:9].upper()}")
        
        # ADX Trend Strength
        if ctx.adx_trend_strength in ("strong", "very_strong"):
            parts.append(f"ADX:{ctx.adx_value:.0f}")
        if ctx.di_crossover != "none":
            parts.append(f"DI:{ctx.di_crossover[:4].upper()}")
        
        # VWAP position
        if ctx.price_vs_vwap != "at":
            parts.append(f"VWAP:{ctx.price_vs_vwap[:3].upper()}")
        
        # Momentum
        if ctx.momentum_strength > 0.5:
            parts.append(f"MOM:{ctx.momentum_strength:.1f}")
        
        # Candle patterns
        if ctx.wick_rejection:
            parts.append("WICK_REJ")
        if ctx.engulfing_pattern:
            parts.append("ENGULF")
        
        # Consecutive candles
        if ctx.direction is OrderSide.BUY and ctx.consecutive_bullish > 0:
            parts.append(f"BULL:{ctx.consecutive_bullish}")
        elif ctx.direction is OrderSide.SELL and ctx.consecutive_bearish > 0:
            parts.append(f"BEAR:{ctx.consecutive_bearish}")
        
        # Micro structure
        if ctx.micro_fvg:
            parts.append("uFVG")
        
        # Session
        if ctx.is_high_liquidity:
            parts.append("HI_LIQ")
        if ctx.killzone:
            parts.append(ctx.killzone.name[:5])
        
        # Quality grade
        parts.append(f"[{ctx.entry_quality}]")
        
        return "+".join(parts)

    def _debug(self, message: str, *args) -> None:
        """Print debug message if debug mode is enabled."""
        if bool(self.params.get("debug", False)):
            text = message % args if args else message
            print(f"[HFT-SCALP] {text}")


# Register the strategy
StrategyRegistry.register(HFTScalperStrategy)

__all__ = ["HFTScalperStrategy"]
