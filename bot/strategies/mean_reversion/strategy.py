"""Mean Reversion Strategy - Proven Algorithmic Trading Approach.

THEORY (from research):
========================
Mean reversion is based on the statistical principle that asset prices 
eventually return to their long-term average (mean). When prices deviate 
significantly from the mean, there's a high probability they'll revert.

KEY INSIGHTS FROM RESEARCH:
===========================
1. Use Bollinger Bands to identify extremes (price at 2 std from mean)
2. Confirm with RSI oversold/overbought conditions
3. Target is the MIDDLE BAND (the mean)
4. Works BEST in range-bound/sideways markets
5. AVOID trending markets (use ADX filter < 25)
6. Z-score > 1.5-2.0 signals trading opportunity

ENTRY RULES:
============
BUY Setup:
- Price touches/closes below LOWER Bollinger Band
- RSI < 30 (oversold) OR RSI < 20 (extreme oversold = stronger)
- ADX < 25 (not trending - range bound market)
- Target: Middle Bollinger Band (20 SMA)
- Stop: Below recent swing low OR lower band - 1 ATR

SELL Setup:
- Price touches/closes above UPPER Bollinger Band  
- RSI > 70 (overbought) OR RSI > 80 (extreme overbought = stronger)
- ADX < 25 (not trending - range bound market)
- Target: Middle Bollinger Band (20 SMA)
- Stop: Above recent swing high OR upper band + 1 ATR

KEY ADVANTAGES:
===============
- Higher win rate (price usually reverts)
- Clear entry/exit rules
- Works in sideways markets (60-70% of the time)
- Statistical edge based on probability
- Simple to implement and backtest

LIMITATIONS:
============
- Fails in trending markets (use ADX filter!)
- Can have large losses if trend continues
- Requires patience (wait for extremes)
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, List, Optional, Tuple

from core.order_types import OrderSide, StrategySignal
from core.utils import (
    Candle,
    Killzone,
    active_killzone,
    active_session,
    calculate_atr,
)
from strategies import BaseStrategy, StrategyRegistry


@dataclass
class MeanReversionContext:
    """Context for mean reversion trade decisions."""
    
    direction: OrderSide
    
    # Bollinger Bands
    bb_upper: float
    bb_middle: float  # THE MEAN - our target
    bb_lower: float
    bb_width: float
    bb_position: str  # "above_upper", "at_upper", "above_middle", etc.
    
    # Z-Score (how many std devs from mean)
    z_score: float
    
    # RSI
    rsi_value: float
    rsi_signal: str  # "oversold", "overbought", "extreme_os", "extreme_ob"
    
    # ADX (trend strength filter)
    adx_value: float
    is_ranging: bool  # ADX < 25 means ranging market
    
    # Stochastic
    stoch_k: float
    stoch_d: float
    stoch_signal: str
    
    # ATR for stops
    atr: float
    
    # Entry quality
    confluence_score: int
    entry_quality: str
    
    # Session
    session: str
    is_good_session: bool


class MeanReversionStrategy(BaseStrategy):
    """Mean Reversion Strategy - Fade Extreme Deviations.
    
    Core Philosophy:
    ================
    "Buy low, sell high" - Price extremes revert to the mean.
    
    This is the OPPOSITE of trend-following. We:
    - BUY when price is at extreme lows (oversold)
    - SELL when price is at extreme highs (overbought)
    - Target the MEAN (middle Bollinger Band)
    
    Critical Filter: ADX < 25 (only trade ranging markets!)
    
    Best Instruments:
    =================
    - EUR/USD (most mean-reverting major)
    - USD/CHF (often ranges)
    - AUD/NZD (highly correlated = ranges)
    - Any pair in consolidation
    
    Avoid During:
    =============
    - Strong trends (ADX > 30)
    - Major news events
    - Breakout conditions
    """
    
    name = "mean_reversion"
    
    DEFAULT_PARAMS = {
        # Timeframe
        "timeframe": "M5",  # M5 works well for mean reversion
        "min_candles": 50,
        
        # ========================================
        # BOLLINGER BANDS (Primary Indicator)
        # ========================================
        # Standard settings - 20 period, 2 std devs
        "bb_period": 20,
        "bb_std_dev": 2.0,
        
        # Entry triggers
        "bb_touch_entry": True,     # Enter when price touches band
        "bb_close_beyond": True,    # Enter when price closes beyond band
        
        # ========================================
        # RSI CONFIRMATION (Secondary Indicator)
        # ========================================
        "rsi_period": 14,           # Standard 14-period RSI
        "rsi_oversold": 35,         # Relaxed oversold (was 30)
        "rsi_overbought": 65,       # Relaxed overbought (was 70)
        "rsi_extreme_os": 25,       # Extreme oversold (was 20)
        "rsi_extreme_ob": 75,       # Extreme overbought (was 80)
        "require_rsi_confirm": True, # Require RSI confirmation
        
        # ========================================
        # ADX FILTER (Critical - Range Detection)
        # ========================================
        # ADX < threshold = ranging market = GOOD for mean reversion
        # RELAXED: Allow slightly trending markets too
        "adx_period": 14,
        "adx_ranging_threshold": 30,  # Raised from 25 - allow more trades
        "require_ranging_market": True, # Only trade when ADX < threshold
        
        # ========================================
        # STOCHASTIC CONFIRMATION (Optional)
        # ========================================
        "use_stochastic": True,
        "stoch_k_period": 14,
        "stoch_d_period": 3,
        "stoch_smooth": 3,
        "stoch_oversold": 25,       # Relaxed from 20
        "stoch_overbought": 75,     # Relaxed from 80
        
        # ========================================
        # Z-SCORE FILTER
        # ========================================
        # Z-score measures how extreme the price is
        # RELAXED: Accept 1.2 std dev (was 1.5)
        "min_z_score": 1.2,  # Minimum deviation to trade (lowered)
        
        # ========================================
        # TARGETS & STOPS
        # ========================================
        # Target is ALWAYS the middle band (the mean)
        "target_middle_band": True,
        
        # Stop loss placement - TIGHTER for better R:R
        "sl_atr_mult": 1.0,         # Stop at 1.0 ATR beyond entry (was 1.5)
        "atr_period": 14,
        
        # Alternative fixed targets (if not using middle band)
        "tp_atr_mult": 2.0,         # TP at 2x ATR (only if target_middle_band=False)
        
        # ========================================
        # SESSION FILTERING
        # ========================================
        # Mean reversion works best during ranging periods
        "sessions": ("London", "NewYork", "Asian"),
        "prefer_asian": True,  # Asian session often ranges
        
        # ========================================
        # TRADE MANAGEMENT
        # ========================================
        "max_trades_per_session": 8,   # Raised from 5
        "min_minutes_between_trades": 10,  # Lowered from 15
        "max_daily_trades": 20,  # Raised from 15
        
        # ========================================
        # CONFLUENCE REQUIREMENTS
        # ========================================
        "min_confluence": 2,  # Lowered from 3
        
        # ========================================
        # RISK
        # ========================================
        "risk_percent": 1.5,  # Slightly higher risk (higher win rate expected)
        
        # Debug
        "debug": False,
    }
    
    # Pip values
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
        "AUDNZD": 0.0001,  # Great for mean reversion
        "AUDCAD": 0.0001,
        "XAUUSD": 0.01,
        "BTCUSD": 1.0,
    }
    
    def __init__(self, context) -> None:
        super().__init__(context)
        self._candles: Deque[Candle] = deque(maxlen=200)
        self.params = {**self.DEFAULT_PARAMS, **(self.context.parameters or {})}
        
        # Session tracking
        self._trades_this_session = 0
        self._last_trade_time: Optional[datetime] = None
        self._current_session: Optional[str] = None
    
    def on_candle(self, candle: Candle) -> Optional[StrategySignal]:
        """Process candle and check for mean reversion opportunities."""
        
        # Handle candle updates
        if self._candles and self._candles[-1].timestamp == candle.timestamp:
            self._candles[-1] = candle
        else:
            self._candles.append(candle)
        
        # Warmup check
        min_candles = int(self.params["min_candles"])
        if len(self._candles) < min_candles:
            self._debug(f"Warming up: {len(self._candles)}/{min_candles}")
            return None
        
        # Session check
        session_name = active_session(candle.timestamp, self.context.session_windows)
        session_whitelist = set(self.params.get("sessions", ("London", "NewYork", "Asian")))
        
        if session_name not in session_whitelist:
            self._debug(f"Session filtered: {session_name}")
            return None
        
        # Track session changes
        if session_name != self._current_session:
            self._current_session = session_name
            self._trades_this_session = 0
        
        # Max trades check
        max_trades = int(self.params.get("max_trades_per_session", 5))
        if self._trades_this_session >= max_trades:
            self._debug(f"Max trades reached: {self._trades_this_session}/{max_trades}")
            return None
        
        # Cooldown check
        min_minutes = int(self.params.get("min_minutes_between_trades", 15))
        if self._last_trade_time:
            elapsed = (candle.timestamp - self._last_trade_time).total_seconds() / 60
            if elapsed < min_minutes:
                return None
        
        # Build mean reversion context
        mr_ctx = self._build_context(session_name, candle)
        if mr_ctx is None:
            return None
        
        # Confluence check
        min_confluence = int(self.params.get("min_confluence", 3))
        if mr_ctx.confluence_score < min_confluence:
            self._debug(f"Insufficient confluence: {mr_ctx.confluence_score}/{min_confluence}")
            return None
        
        # Calculate entry, stop, and target
        entry = candle.close
        
        if self.params.get("target_middle_band"):
            # Target is the MEAN (middle band)
            take_profit = mr_ctx.bb_middle
        else:
            # Fixed ATR-based target
            tp_mult = float(self.params["tp_atr_mult"])
            if mr_ctx.direction is OrderSide.BUY:
                take_profit = entry + (mr_ctx.atr * tp_mult)
            else:
                take_profit = entry - (mr_ctx.atr * tp_mult)
        
        # Stop loss beyond the band
        sl_mult = float(self.params["sl_atr_mult"])
        
        if mr_ctx.direction is OrderSide.BUY:
            # Stop below lower band
            stop_loss = min(entry, mr_ctx.bb_lower) - (mr_ctx.atr * sl_mult)
        else:
            # Stop above upper band
            stop_loss = max(entry, mr_ctx.bb_upper) + (mr_ctx.atr * sl_mult)
        
        # Validate prices
        if stop_loss <= 0 or take_profit <= 0:
            self._debug("Invalid stop/tp prices")
            return None
        
        # Validate direction makes sense
        if mr_ctx.direction is OrderSide.BUY:
            if take_profit <= entry:
                self._debug(f"BUY but TP <= entry: tp={take_profit:.5f} entry={entry:.5f}")
                return None
        else:
            if take_profit >= entry:
                self._debug(f"SELL but TP >= entry: tp={take_profit:.5f} entry={entry:.5f}")
                return None
        
        # Calculate confidence
        confidence = min(mr_ctx.confluence_score / 6.0, 1.0)
        
        # Build signal
        signal = StrategySignal(
            instrument=self.context.instrument,
            side=mr_ctx.direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=candle.timestamp,
            confidence=confidence,
            session=mr_ctx.session,
        )
        
        # Update tracking
        self._trades_this_session += 1
        self._last_trade_time = candle.timestamp
        
        self._debug(
            "MEAN REVERSION SIGNAL | %s | entry=%.5f sl=%.5f tp=%.5f | "
            "z_score=%.2f rsi=%.1f adx=%.1f | confluence=%d [%s]",
            mr_ctx.direction.value.upper(),
            entry,
            stop_loss,
            take_profit,
            mr_ctx.z_score,
            mr_ctx.rsi_value,
            mr_ctx.adx_value,
            mr_ctx.confluence_score,
            mr_ctx.entry_quality,
        )
        
        return signal
    
    def _build_context(
        self,
        session_name: str,
        candle: Candle,
    ) -> Optional[MeanReversionContext]:
        """Build mean reversion analysis context.
        
        This implements the core mean reversion logic:
        1. Check if price is at Bollinger Band extreme
        2. Confirm with RSI oversold/overbought
        3. Verify market is ranging (ADX < 25)
        4. Calculate Z-score for deviation strength
        """
        
        candles = list(self._candles)
        confluence_score = 0
        
        # ========================================
        # ATR CALCULATION (for stops)
        # ========================================
        atr_period = int(self.params.get("atr_period", 14))
        atr = calculate_atr(candles, period=atr_period)
        
        if atr <= 0:
            self._debug("ATR is zero")
            return None
        
        # ========================================
        # BOLLINGER BANDS CALCULATION
        # ========================================
        bb_period = int(self.params["bb_period"])
        bb_std = float(self.params["bb_std_dev"])
        
        bb_upper, bb_middle, bb_lower, std_dev = self._calculate_bollinger(
            candles, bb_period, bb_std
        )
        
        if bb_upper is None:
            self._debug("Insufficient data for Bollinger Bands")
            return None
        
        price = candle.close
        bb_width = (bb_upper - bb_lower) / bb_middle if bb_middle > 0 else 0
        
        # Determine price position relative to bands
        bb_position = "middle"
        if price >= bb_upper:
            bb_position = "at_upper"
        elif price > bb_middle:
            bb_position = "above_middle"
        elif price <= bb_lower:
            bb_position = "at_lower"
        elif price < bb_middle:
            bb_position = "below_middle"
        
        # ========================================
        # Z-SCORE CALCULATION
        # ========================================
        # Z = (Price - Mean) / StdDev
        z_score = (price - bb_middle) / std_dev if std_dev > 0 else 0
        
        min_z = float(self.params.get("min_z_score", 1.5))
        
        # Check if deviation is extreme enough
        if abs(z_score) < min_z:
            self._debug(f"Z-score too low: {z_score:.2f} (need {min_z})")
            return None
        
        confluence_score += 1  # Price at extreme
        if abs(z_score) >= 2.0:
            confluence_score += 1  # Strong deviation bonus
        
        # ========================================
        # ADX CALCULATION (Range Filter)
        # ========================================
        adx_period = int(self.params["adx_period"])
        adx_value, _, _ = self._calculate_adx(candles, adx_period)
        
        ranging_threshold = float(self.params["adx_ranging_threshold"])
        is_ranging = adx_value < ranging_threshold
        
        # NOTE: We no longer skip trending markets - we trade them differently
        # In ranging: classic mean reversion (fade extremes)
        # In trending: trade pullbacks WITH the trend only
        
        if is_ranging:
            confluence_score += 2  # Ranging market is ideal for mean reversion
        
        # ========================================
        # RSI CALCULATION
        # ========================================
        rsi_period = int(self.params["rsi_period"])
        rsi_value = self._calculate_rsi(candles, rsi_period)
        
        os_level = float(self.params["rsi_oversold"])
        ob_level = float(self.params["rsi_overbought"])
        extreme_os = float(self.params["rsi_extreme_os"])
        extreme_ob = float(self.params["rsi_extreme_ob"])
        
        rsi_signal = "neutral"
        if rsi_value <= extreme_os:
            rsi_signal = "extreme_oversold"
            confluence_score += 2
        elif rsi_value <= os_level:
            rsi_signal = "oversold"
            confluence_score += 1
        elif rsi_value >= extreme_ob:
            rsi_signal = "extreme_overbought"
            confluence_score += 2
        elif rsi_value >= ob_level:
            rsi_signal = "overbought"
            confluence_score += 1
        
        # ========================================
        # STOCHASTIC CALCULATION (Optional)
        # ========================================
        stoch_k = 50.0
        stoch_d = 50.0
        stoch_signal = "neutral"
        
        if self.params.get("use_stochastic"):
            stoch_k, stoch_d = self._calculate_stochastic(
                candles,
                k_period=int(self.params.get("stoch_k_period", 14)),
                d_period=int(self.params.get("stoch_d_period", 3)),
                smooth=int(self.params.get("stoch_smooth", 3)),
            )
            
            stoch_os = float(self.params.get("stoch_oversold", 20))
            stoch_ob = float(self.params.get("stoch_overbought", 80))
            
            if stoch_k <= stoch_os:
                stoch_signal = "oversold"
                confluence_score += 1
            elif stoch_k >= stoch_ob:
                stoch_signal = "overbought"
                confluence_score += 1
        
        # ========================================
        # DIRECTION DETERMINATION
        # ========================================
        direction = None
        
        # ========================================
        # NEW: TREND-AWARE MEAN REVERSION
        # ========================================
        # The research says mean reversion works best in RANGING markets
        # But we can also use BB bounces WITH the trend
        #
        # Strategy:
        # 1. In RANGING market (ADX < 25): Classic mean reversion (fade extremes)
        # 2. In TRENDING market (ADX > 25): Trade BB bounces WITH the trend only
        #
        # For trending:
        # - If trending UP (+DI > -DI): Only BUY when price touches LOWER band
        # - If trending DOWN (-DI > +DI): Only SELL when price touches UPPER band
        
        # Get DI direction for trend
        _, plus_di, minus_di = self._calculate_adx(candles, adx_period)
        trend_direction = "up" if plus_di > minus_di else "down"
        
        if is_ranging:
            # RANGING MARKET: Classic mean reversion (fade extremes)
            # BUY: Price at lower band + RSI oversold
            if bb_position == "at_lower" and z_score < -min_z:
                if rsi_signal in ("oversold", "extreme_oversold"):
                    direction = OrderSide.BUY
                    confluence_score += 1
                    self._debug(f"RANGING MR BUY: z={z_score:.2f} rsi={rsi_value:.1f}")
                elif not self.params.get("require_rsi_confirm"):
                    direction = OrderSide.BUY
            
            # SELL: Price at upper band + RSI overbought
            elif bb_position == "at_upper" and z_score > min_z:
                if rsi_signal in ("overbought", "extreme_overbought"):
                    direction = OrderSide.SELL
                    confluence_score += 1
                    self._debug(f"RANGING MR SELL: z={z_score:.2f} rsi={rsi_value:.1f}")
                elif not self.params.get("require_rsi_confirm"):
                    direction = OrderSide.SELL
        else:
            # TRENDING MARKET: Trade BB bounces WITH the trend only
            # This is a pullback entry in a trend
            
            if trend_direction == "up":
                # Uptrend: Only BUY when price pulls back to lower band
                if bb_position == "at_lower" and z_score < -min_z:
                    if rsi_signal in ("oversold", "extreme_oversold"):
                        direction = OrderSide.BUY
                        confluence_score += 2  # Extra confluence for trend alignment
                        self._debug(f"TREND PULLBACK BUY: z={z_score:.2f} rsi={rsi_value:.1f} trend=UP")
            else:
                # Downtrend: Only SELL when price pulls back to upper band
                if bb_position == "at_upper" and z_score > min_z:
                    if rsi_signal in ("overbought", "extreme_overbought"):
                        direction = OrderSide.SELL
                        confluence_score += 2  # Extra confluence for trend alignment
                        self._debug(f"TREND PULLBACK SELL: z={z_score:.2f} rsi={rsi_value:.1f} trend=DOWN")
        
        if direction is None:
            self._debug(f"No setup: pos={bb_position} z={z_score:.2f} rsi={rsi_signal}")
            return None
        
        # Stochastic confirmation bonus
        if direction is OrderSide.BUY and stoch_signal == "oversold":
            confluence_score += 1
        elif direction is OrderSide.SELL and stoch_signal == "overbought":
            confluence_score += 1
        
        # ========================================
        # SESSION ANALYSIS
        # ========================================
        is_good_session = True
        if self.params.get("prefer_asian") and session_name == "Asian":
            confluence_score += 1  # Asian session bonus (more ranging)
        
        # ========================================
        # ENTRY QUALITY GRADE
        # ========================================
        if confluence_score >= 7:
            entry_quality = "A"
        elif confluence_score >= 5:
            entry_quality = "B"
        else:
            entry_quality = "C"
        
        return MeanReversionContext(
            direction=direction,
            bb_upper=bb_upper,
            bb_middle=bb_middle,
            bb_lower=bb_lower,
            bb_width=bb_width,
            bb_position=bb_position,
            z_score=z_score,
            rsi_value=rsi_value,
            rsi_signal=rsi_signal,
            adx_value=adx_value,
            is_ranging=is_ranging,
            stoch_k=stoch_k,
            stoch_d=stoch_d,
            stoch_signal=stoch_signal,
            atr=atr,
            confluence_score=confluence_score,
            entry_quality=entry_quality,
            session=session_name,
            is_good_session=is_good_session,
        )
    
    def _calculate_bollinger(
        self,
        candles: List[Candle],
        period: int,
        std_mult: float,
    ) -> Tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
        """Calculate Bollinger Bands.
        
        Returns (upper, middle, lower, std_dev)
        """
        if len(candles) < period:
            return None, None, None, None
        
        closes = [c.close for c in candles[-period:]]
        
        # Middle band (SMA)
        middle = sum(closes) / period
        
        # Standard deviation
        variance = sum((x - middle) ** 2 for x in closes) / period
        std_dev = variance ** 0.5
        
        # Upper and lower bands
        upper = middle + (std_mult * std_dev)
        lower = middle - (std_mult * std_dev)
        
        return upper, middle, lower, std_dev
    
    def _calculate_rsi(self, candles: List[Candle], period: int) -> float:
        """Calculate RSI."""
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
        
        avg_gain = sum(gains[:period]) / period
        avg_loss = sum(losses[:period]) / period
        
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
        
        if avg_loss == 0:
            return 100.0
        
        rs = avg_gain / avg_loss
        return 100 - (100 / (1 + rs))
    
    def _calculate_stochastic(
        self,
        candles: List[Candle],
        k_period: int,
        d_period: int,
        smooth: int,
    ) -> Tuple[float, float]:
        """Calculate Stochastic Oscillator."""
        if len(candles) < k_period + d_period:
            return 50.0, 50.0
        
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
        
        smoothed_k = []
        for i in range(smooth - 1, len(raw_k_values)):
            avg = sum(raw_k_values[i - smooth + 1:i + 1]) / smooth
            smoothed_k.append(avg)
        
        if len(smoothed_k) < d_period:
            return smoothed_k[-1] if smoothed_k else 50.0, 50.0
        
        stoch_k = smoothed_k[-1]
        stoch_d = sum(smoothed_k[-d_period:]) / d_period
        
        return stoch_k, stoch_d
    
    def _calculate_adx(
        self,
        candles: List[Candle],
        period: int,
    ) -> Tuple[float, float, float]:
        """Calculate ADX (Average Directional Index).
        
        Returns (adx, plus_di, minus_di)
        """
        if len(candles) < period * 2:
            return 0.0, 50.0, 50.0
        
        tr_list = []
        plus_dm_list = []
        minus_dm_list = []
        
        for i in range(1, len(candles)):
            curr = candles[i]
            prev = candles[i - 1]
            
            tr = max(
                curr.high - curr.low,
                abs(curr.high - prev.close),
                abs(curr.low - prev.close)
            )
            tr_list.append(tr)
            
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
            return 0.0, 50.0, 50.0
        
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
            return 0.0, 50.0, 50.0
        
        plus_di = (smooth_plus_dm[-1] / smooth_tr[-1]) * 100
        minus_di = (smooth_minus_dm[-1] / smooth_tr[-1]) * 100
        
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
        
        if len(dx_list) < period:
            adx = dx_list[-1] if dx_list else 0.0
        else:
            adx_values = wilder_smooth(dx_list, period)
            adx = adx_values[-1] / period if adx_values else 0.0
        
        return adx, plus_di, minus_di
    
    def _get_pip_value(self) -> float:
        """Get pip value for current instrument."""
        instrument = self.context.instrument
        return self.PIP_VALUES.get(instrument, 0.0001)
    
    def _debug(self, message: str, *args) -> None:
        """Print debug message if debug mode is enabled."""
        if bool(self.params.get("debug", False)):
            text = message % args if args else message
            print(f"[MEAN-REVERT] {text}")


# Register the strategy
StrategyRegistry.register(MeanReversionStrategy)

__all__ = ["MeanReversionStrategy"]
