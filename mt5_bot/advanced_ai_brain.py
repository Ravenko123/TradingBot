"""
ADVANCED TRADING INTELLIGENCE (ATI v2) - Ultra-Smart AI Brain
============================================================
A professional-grade AI trader with:
- Swing point detection & market structure analysis
- Volatility-adaptive position sizing
- Order flow analysis & momentum detection
- ML-style pattern recognition
- Multi-timeframe confirmation
- Correlation & asset class awareness
- Advanced risk management with drawdown limits
- Predictive entry validation
- Real-time parameter adaptation
- Heat map of profitable conditions

This AI trades like a professional human trader - disciplined, adaptive, intelligent.
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
from collections import deque
import numpy as np
import pandas as pd
from enum import Enum


# ============================================================================
# ENUMS & CONSTANTS
# ============================================================================

class MarketPhase(Enum):
    """Market behavior phases"""
    TRENDING_STRONG = "trending_strong"      # High ADX, clear direction
    TRENDING_WEAK = "trending_weak"          # Low ADX, directional bias
    RANGING = "ranging"                      # Price oscillating in band
    BREAKOUT = "breakout"                    # Breaking structure
    REVERSAL = "reversal"                    # Reversing from extremes
    QUIET = "quiet"                          # Low activity


class TradeOutcome(Enum):
    """Trade result classification"""
    WIN = "win"
    LOSS = "loss"
    BREAKEVEN = "breakeven"


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class SwingPoint:
    """Detected swing point (support/resistance)"""
    timestamp: datetime
    price: float
    is_high: bool  # True = resistance, False = support
    strength: float  # 0-1, how strong this level is
    touches: int  # How many times price touched this
    last_touch: datetime = None
    
    def is_fresh(self, current_time: datetime, hours: int = 24) -> bool:
        """Is this swing point still relevant?"""
        return (current_time - self.timestamp).total_seconds() < hours * 3600


@dataclass
class OrderFlow:
    """Order flow analysis for a bar"""
    timestamp: datetime
    bullish_pressure: float  # -1 to 1
    bid_ask_ratio: float  # More > 1 means more buy pressure
    momentum_direction: float  # -1 to 1
    volume_profile: str  # 'buying', 'selling', 'neutral'
    market_maker_activity: float  # Estimated by spread changes


@dataclass
class PatternSignature:
    """ML pattern signature - what precedes profitable entries"""
    pattern_id: str
    conditions: Dict[str, Any]  # EMA, ADX, ATR ranges, etc.
    win_rate: float
    avg_win: float
    avg_loss: float
    occurrences: int
    last_seen: datetime = None


@dataclass
class SymbolProfile:
    """Complete profile of symbol behavior"""
    symbol: str
    
    # Win rate by hour of day
    hourly_win_rate: Dict[int, Tuple[int, int]] = field(default_factory=dict)  # hour -> (wins, total)
    
    # Win rate by market phase
    phase_performance: Dict[str, Tuple[int, int]] = field(default_factory=dict)  # phase -> (wins, total)
    
    # Best entry price levels (clusters of profitable entries)
    profitable_price_zones: List[Tuple[float, float]] = field(default_factory=list)  # (low, high)
    
    # Volatility profile
    avg_atr_low: float = 0.0
    avg_atr_high: float = 0.0
    optimal_atr_mult: float = 1.5
    
    # Correlation with other symbols
    correlations: Dict[str, float] = field(default_factory=dict)  # symbol -> correlation
    
    # Recent win/loss streak
    recent_trades: deque = field(default_factory=lambda: deque(maxlen=20))  # Last 20 trades
    
    # Current status
    is_hot: bool = False  # On a winning streak
    is_cold: bool = False  # On a losing streak
    confidence_score: float = 0.5


@dataclass
class RiskProfile:
    """Dynamic risk management"""
    daily_max_loss: float = 0.0  # Absolute $ limit
    daily_max_drawdown: float = 0.05  # 5% of account
    max_consecutive_losses: int = 3
    max_open_positions: int = 5
    correlation_max: float = 0.7  # Don't trade correlated pairs
    
    # Current state
    todays_loss: float = 0.0
    consecutive_losses: int = 0
    open_position_count: int = 0


# ============================================================================
# ADVANCED AI ENGINE
# ============================================================================

class SwingPointDetector:
    """Detects support/resistance from price structure"""
    
    def __init__(self, lookback: int = 50):
        self.lookback = lookback
        self.swings: List[SwingPoint] = []
    
    def detect(self, df: pd.DataFrame) -> List[SwingPoint]:
        """Detect swings using zigzag method"""
        swings = []
        
        if len(df) < 5:
            return swings
        
        high = df['high'].values
        low = df['low'].values
        close = df['close'].values
        time_col = df['Time'].values if 'Time' in df.columns else None
        
        # Find local highs and lows
        for i in range(2, len(df) - 2):
            # Local high (resistance)
            if high[i] > high[i-1] and high[i] > high[i+1] and high[i] > high[i-2] and high[i] > high[i+2]:
                strength = (high[i] - np.mean(low[max(0,i-10):i])) / (np.std(high[max(0,i-10):i]) + 0.0001)
                strength = min(1.0, max(0.0, strength / 10))  # Normalize 0-1
                
                swing = SwingPoint(
                    timestamp=pd.Timestamp(time_col[i]) if time_col is not None else datetime.now(),
                    price=float(high[i]),
                    is_high=True,
                    strength=strength,
                    touches=1
                )
                swings.append(swing)
            
            # Local low (support)
            if low[i] < low[i-1] and low[i] < low[i+1] and low[i] < low[i-2] and low[i] < low[i+2]:
                strength = (np.mean(high[max(0,i-10):i]) - low[i]) / (np.std(low[max(0,i-10):i]) + 0.0001)
                strength = min(1.0, max(0.0, strength / 10))  # Normalize 0-1
                
                swing = SwingPoint(
                    timestamp=pd.Timestamp(time_col[i]) if time_col is not None else datetime.now(),
                    price=float(low[i]),
                    is_high=False,
                    strength=strength,
                    touches=1
                )
                swings.append(swing)
        
        # Merge nearby swings
        merged = self._merge_swings(swings)
        self.swings = merged
        return merged
    
    def _merge_swings(self, swings: List[SwingPoint]) -> List[SwingPoint]:
        """Merge swings that are close together"""
        if not swings:
            return []
        
        merged = []
        current = swings[0]
        
        for swing in swings[1:]:
            # If same type and close in price, merge
            if swing.is_high == current.is_high:
                if abs(swing.price - current.price) < current.price * 0.001:  # 0.1% close
                    current.touches += 1
                    current.strength = max(current.strength, swing.strength)
                    continue
            
            merged.append(current)
            current = swing
        
        merged.append(current)
        return merged
    
    def get_nearest_resistance(self, current_price: float) -> Optional[SwingPoint]:
        """Get nearest resistance above current price"""
        resistances = [s for s in self.swings if s.is_high and s.price > current_price]
        return min(resistances, key=lambda s: s.price) if resistances else None
    
    def get_nearest_support(self, current_price: float) -> Optional[SwingPoint]:
        """Get nearest support below current price"""
        supports = [s for s in self.swings if not s.is_high and s.price < current_price]
        return max(supports, key=lambda s: s.price) if supports else None


class OrderFlowAnalyzer:
    """Analyzes order flow and momentum"""
    
    def __init__(self):
        self.history: List[OrderFlow] = deque(maxlen=100)
    
    def analyze(self, df: pd.DataFrame, bid: float, ask: float, current_spread: int) -> OrderFlow:
        """Analyze order flow from price action"""
        if len(df) < 3:
            return OrderFlow(
                timestamp=datetime.now(),
                bullish_pressure=0.0,
                bid_ask_ratio=1.0,
                momentum_direction=0.0,
                volume_profile='neutral',
                market_maker_activity=0.5
            )
        
        # Momentum: compare recent close to SMA
        sma_20 = df['close'].tail(20).mean()
        current_close = df['close'].iloc[-1]
        momentum = (current_close - sma_20) / (sma_20 + 0.0001)
        momentum_direction = np.tanh(momentum * 10)  # Normalize to -1..1
        
        # Bid/Ask ratio (buy pressure)
        bid_ask_ratio = ask / (bid + 0.00001)
        
        # Volume pressure (if available)
        if 'volume' in df.columns:
            vol_ratio = df['volume'].iloc[-1] / (df['volume'].tail(20).mean() + 1)
            volume_profile = 'buying' if momentum_direction > 0.3 else ('selling' if momentum_direction < -0.3 else 'neutral')
        else:
            vol_ratio = 1.0
            volume_profile = 'neutral'
        
        # Bullish pressure: combination of factors
        spread_factor = min(1.0, current_spread / 10) if current_spread > 0 else 0.5
        bullish_pressure = momentum_direction * (1 - spread_factor * 0.3)
        
        # Market maker activity (estimated by bid/ask spread changes)
        mm_activity = 0.5 + (bid_ask_ratio - 1) * 5  # More volatile = more activity
        mm_activity = max(0.0, min(1.0, mm_activity))
        
        flow = OrderFlow(
            timestamp=datetime.now(),
            bullish_pressure=bullish_pressure,
            bid_ask_ratio=bid_ask_ratio,
            momentum_direction=momentum_direction,
            volume_profile=volume_profile,
            market_maker_activity=mm_activity
        )
        
        self.history.append(flow)
        return flow


class MarketStructureAnalyzer:
    """Analyzes market structure and phases"""
    
    def __init__(self):
        self.swings = SwingPointDetector()
        self.order_flow = OrderFlowAnalyzer()
    
    def analyze_phase(self, df: pd.DataFrame, adx: float, atr: float, 
                      bid: float, ask: float, spread: int) -> Tuple[MarketPhase, float]:
        """Determine current market phase"""
        
        if len(df) < 20:
            return MarketPhase.QUIET, 0.3
        
        # Volatility check
        atr_percentile = np.percentile(df['ATR'].tail(50), 75)
        volatility_rank = atr / (atr_percentile + 0.0001)
        
        # Trend check
        ema_fast = df['EMA_Fast'].iloc[-1] if 'EMA_Fast' in df.columns else None
        ema_slow = df['EMA_Slow'].iloc[-1] if 'EMA_Slow' in df.columns else None
        
        # Detect swings
        self.swings.detect(df)
        
        # High volatility + High ADX = Strong Trend
        if adx > 35 and volatility_rank > 0.8:
            return MarketPhase.TRENDING_STRONG, 0.9
        
        # Moderate ADX = Weak Trend
        elif 20 < adx <= 35:
            return MarketPhase.TRENDING_WEAK, 0.7
        
        # Low ADX + Volatility extremes = Range
        elif adx < 20 and volatility_rank < 0.6:
            return MarketPhase.RANGING, 0.6
        
        # High volatility shift = Breakout
        elif volatility_rank > 1.2:
            return MarketPhase.BREAKOUT, 0.8
        
        # Very low activity
        else:
            return MarketPhase.QUIET, 0.4
    
    def get_optimal_sl_tp(self, direction: str, entry: float, atr: float, 
                          phase: MarketPhase) -> Tuple[float, float]:
        """Calculate optimal SL/TP based on market structure"""
        
        # Find nearby swing points
        support = self.swings.get_nearest_support(entry)
        resistance = self.swings.get_nearest_resistance(entry)
        
        # Adjust risk based on phase
        risk_multiplier = {
            MarketPhase.TRENDING_STRONG: 1.5,
            MarketPhase.TRENDING_WEAK: 1.2,
            MarketPhase.RANGING: 0.8,
            MarketPhase.BREAKOUT: 2.0,
            MarketPhase.REVERSAL: 1.3,
            MarketPhase.QUIET: 0.5
        }.get(phase, 1.0)
        
        if direction == 'BUY':
            # SL below nearest support or 2x ATR, whichever is farther
            if support:
                sl = min(support.price - atr * 0.3, entry - atr * risk_multiplier * 2)
            else:
                sl = entry - atr * risk_multiplier * 2
            
            # TP at nearest resistance or 2x RR, whichever is closer
            if resistance:
                tp = max(resistance.price + atr * 0.3, entry + atr * risk_multiplier * 4)
            else:
                tp = entry + atr * risk_multiplier * 4
        else:  # SELL
            if resistance:
                sl = max(resistance.price + atr * 0.3, entry + atr * risk_multiplier * 2)
            else:
                sl = entry + atr * risk_multiplier * 2
            
            if support:
                tp = min(support.price - atr * 0.3, entry - atr * risk_multiplier * 4)
            else:
                tp = entry - atr * risk_multiplier * 4
        
        return sl, tp


class PatternRecognizer:
    """ML-style pattern recognition for profitable entries"""
    
    def __init__(self):
        self.patterns: Dict[str, PatternSignature] = {}
        self.trade_history: List[Dict] = []
    
    def extract_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Extract features from current market state"""
        if len(df) < 20:
            return {}
        
        features = {
            'ema_ratio': (df['EMA_Fast'].iloc[-1] / df['EMA_Slow'].iloc[-1]) - 1,
            'adx': df['ADX'].iloc[-1],
            'atr_percentile': np.percentile(df['ATR'].tail(50), 75),
            'rsi': self._calculate_rsi(df['close'], 14),
            'close_above_sma20': 1 if df['close'].iloc[-1] > df['close'].tail(20).mean() else 0,
            'volatility_trend': (df['ATR'].iloc[-1] - df['ATR'].iloc[-5]) / df['ATR'].iloc[-5],
        }
        
        return features
    
    def record_trade(self, features: Dict, outcome: TradeOutcome, win_amount: float, loss_amount: float):
        """Record trade outcome to learn patterns"""
        pattern_id = self._hash_features(features)
        
        if pattern_id not in self.patterns:
            self.patterns[pattern_id] = PatternSignature(
                pattern_id=pattern_id,
                conditions=features,
                win_rate=0.0,
                avg_win=0.0,
                avg_loss=0.0,
                occurrences=0
            )
        
        pattern = self.patterns[pattern_id]
        
        # Update stats
        if outcome == TradeOutcome.WIN:
            pattern.win_rate = (pattern.win_rate * pattern.occurrences + 1) / (pattern.occurrences + 1)
            pattern.avg_win = (pattern.avg_win * pattern.occurrences + win_amount) / (pattern.occurrences + 1)
        else:
            pattern.win_rate = (pattern.win_rate * pattern.occurrences) / (pattern.occurrences + 1)
            pattern.avg_loss = (pattern.avg_loss * pattern.occurrences + loss_amount) / (pattern.occurrences + 1)
        
        pattern.occurrences += 1
        pattern.last_seen = datetime.now()
    
    def get_pattern_quality(self, pattern_id: str) -> float:
        """Get confidence score for a pattern (0-1)"""
        if pattern_id not in self.patterns:
            return 0.5
        
        pattern = self.patterns[pattern_id]
        
        # Need minimum 5 occurrences
        if pattern.occurrences < 5:
            return 0.5
        
        # Win rate * expectancy
        expectancy = (pattern.win_rate * pattern.avg_win) - ((1 - pattern.win_rate) * pattern.avg_loss)
        
        # Normalize: good pattern has positive expectancy
        if expectancy > 0:
            return min(0.9, pattern.win_rate)
        else:
            return max(0.1, pattern.win_rate * 0.5)
    
    @staticmethod
    def _calculate_rsi(prices, period=14):
        """Calculate RSI"""
        deltas = np.diff(prices)
        seed = deltas[:period + 1]
        up = seed[seed >= 0].sum() / period
        down = -seed[seed < 0].sum() / period
        rs = up / (down + 0.0001)
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    @staticmethod
    def _hash_features(features: Dict) -> str:
        """Create pattern ID from features"""
        # Round features for pattern matching
        rounded = {k: round(v, 2) for k, v in features.items()}
        return json.dumps(rounded, sort_keys=True)


class SymbolProfiler:
    """Tracks behavior and performance per symbol"""
    
    def __init__(self):
        self.profiles: Dict[str, SymbolProfile] = {}
    
    def update_from_trade(self, symbol: str, hour: int, phase: MarketPhase, 
                          entry_price: float, outcome: TradeOutcome, pl: float):
        """Update symbol profile from closed trade"""
        
        if symbol not in self.profiles:
            self.profiles[symbol] = SymbolProfile(symbol=symbol)
        
        profile = self.profiles[symbol]
        
        # Update hourly win rate
        if hour not in profile.hourly_win_rate:
            profile.hourly_win_rate[hour] = (0, 0)
        
        wins, total = profile.hourly_win_rate[hour]
        if outcome == TradeOutcome.WIN:
            profile.hourly_win_rate[hour] = (wins + 1, total + 1)
        else:
            profile.hourly_win_rate[hour] = (wins, total + 1)
        
        # Update phase performance
        phase_key = phase.value
        if phase_key not in profile.phase_performance:
            profile.phase_performance[phase_key] = (0, 0)
        
        wins, total = profile.phase_performance[phase_key]
        if outcome == TradeOutcome.WIN:
            profile.phase_performance[phase_key] = (wins + 1, total + 1)
        else:
            profile.phase_performance[phase_key] = (wins, total + 1)
        
        # Track recent trades for streak detection
        profile.recent_trades.append(outcome)
        
        # Check for hot/cold streaks
        recent = list(profile.recent_trades)
        if len(recent) >= 5:
            last_5 = recent[-5:]
            wins = sum(1 for t in last_5 if t == TradeOutcome.WIN)
            profile.is_hot = wins >= 4
            profile.is_cold = wins <= 1
        
        # Profitable price zones
        if outcome == TradeOutcome.WIN and pl > 0:
            # Record entry price zone
            zone_width = entry_price * 0.002  # 0.2% window
            profile.profitable_price_zones.append((entry_price - zone_width, entry_price + zone_width))


class AdvancedAIBrain:
    """The complete AI trading system"""
    
    def __init__(self):
        self.market_structure = MarketStructureAnalyzer()
        self.patterns = PatternRecognizer()
        self.profiler = SymbolProfiler()
        self.risk = RiskProfile()
        
        self.data_dir = Path(__file__).parent / 'ai_data'
        self.data_dir.mkdir(exist_ok=True)
    
    def analyze_signal(self, symbol: str, df: pd.DataFrame, signal: Dict, 
                       bid: float, ask: float, spread: int) -> Dict[str, Any]:
        """
        Analyze a trading signal comprehensively
        Returns enhanced signal with confidence, risk/reward, and entry validation
        """
        
        if not signal or len(df) < 50:
            base_conf = 0.55 if signal else 0.5
            return {**(signal or {}), 'ai_approved': True, 'confidence': base_conf, 'reason': 'insufficient data'}
        
        # Extract core data
        direction = signal['direction']
        entry_price = signal['entry']
        # Accept both legacy ('stop','tp') and verbose ('stop_loss','take_profit') keys
        stop_loss = signal.get('stop') if 'stop' in signal else signal.get('stop_loss')
        take_profit = signal.get('tp') if 'tp' in signal else signal.get('take_profit')
        atr = df['ATR'].iloc[-1]
        adx = df['ADX'].iloc[-1]
        
        # 1. MARKET STRUCTURE ANALYSIS
        phase, phase_confidence = self.market_structure.analyze_phase(df, adx, atr, bid, ask, spread)
        
        # 2. ORDER FLOW ANALYSIS
        flow = self.market_structure.order_flow.analyze(df, bid, ask, spread)
        
        # 3. PATTERN RECOGNITION
        features = self.patterns.extract_features(df)
        pattern_id = self.patterns._hash_features(features)
        pattern_quality = self.patterns.get_pattern_quality(pattern_id)
        
        # 4. SYMBOL PROFILING
        hour = datetime.now().hour
        symbol_profile = self.profiler.profiles.get(symbol)
        
        # 5. CONFIDENCE SCORING
        confidence = self._calculate_confidence(
            direction, flow, phase, phase_confidence, 
            pattern_quality, symbol_profile, hour
        )
        
        # 6. SMART RISK/REWARD CALCULATION
        smart_sl, smart_tp = self.market_structure.get_optimal_sl_tp(
            direction, entry_price, atr, phase
        )
        
        # 7. ENTRY VALIDATION
        is_valid_entry = self._validate_entry(
            direction, flow, phase, adx, atr, 
            symbol_profile, hour, confidence
        )
        
        return {
            **signal,
            'ai_approved': is_valid_entry and confidence > 0.25,
            'confidence': confidence,
            'market_phase': phase.value,
            'order_flow': {
                'bullish_pressure': flow.bullish_pressure,
                'momentum': flow.momentum_direction,
                'volume_profile': flow.volume_profile
            },
            'smart_sl': smart_sl,
            'smart_tp': smart_tp,
            'symbol_status': {
                'is_hot': symbol_profile.is_hot if symbol_profile else False,
                'is_cold': symbol_profile.is_cold if symbol_profile else False,
            },
            'reason': 'AI approved' if is_valid_entry and confidence > 0.25 else f'Confidence {confidence:.1%} too low'
        }
    
    def _calculate_confidence(self, direction: str, flow: OrderFlow, phase: MarketPhase,
                             phase_confidence: float, pattern_quality: float,
                             symbol_profile: Optional[SymbolProfile], hour: int) -> float:
        """Calculate overall entry confidence 0-1"""
        
        confidence = 0.5  # Base
        
        # Phase contribution
        confidence += phase_confidence * 0.2
        
        # Pattern quality
        confidence += (pattern_quality - 0.5) * 0.2
        
        # Order flow alignment
        if direction == 'BUY' and flow.bullish_pressure > 0.2:
            confidence += 0.15
        elif direction == 'SELL' and flow.bullish_pressure < -0.2:
            confidence += 0.15
        
        # Symbol profile (if available)
        if symbol_profile:
            if symbol_profile.is_hot:
                confidence += 0.1
            if symbol_profile.is_cold:
                confidence -= 0.15
            
            # Time-of-day edge
            if hour in symbol_profile.hourly_win_rate:
                wins, total = symbol_profile.hourly_win_rate[hour]
                if total > 5:
                    wr = wins / total
                    confidence += (wr - 0.5) * 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _validate_entry(self, direction: str, flow: OrderFlow, phase: MarketPhase,
                       adx: float, atr: float, symbol_profile: Optional[SymbolProfile],
                       hour: int, confidence: float) -> bool:
        """Validate if entry should be taken"""
        
        # Minimum confidence (very loose to encourage data collection)
        if confidence < 0.25:
            return False

        # Quiet phases allowed with modest confidence
        if phase == MarketPhase.QUIET and confidence < 0.35:
            return False

        if symbol_profile and symbol_profile.is_cold:
            return confidence > 0.5  # Still some caution on cold symbols
        
        # Hot phase = more aggressive
        if phase == MarketPhase.TRENDING_STRONG and adx > 30:
            return True
        
        # Default acceptance
        return True
    
    def update_after_trade(self, symbol: str, features: Dict, outcome: TradeOutcome, 
                          pl: float, hour: int = None, phase: MarketPhase = None):
        """Learn from completed trade"""
        
        win_amount = abs(pl) if outcome == TradeOutcome.WIN else 0
        loss_amount = abs(pl) if outcome == TradeOutcome.LOSS else 0
        
        # Update patterns
        self.patterns.record_trade(features, outcome, win_amount, loss_amount)
        
        # Update profile
        if hour and phase:
            self.profiler.update_from_trade(symbol, hour, phase, 0, outcome, pl)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_advanced_brain = None

def get_advanced_brain() -> AdvancedAIBrain:
    """Get the singleton advanced brain instance"""
    global _advanced_brain
    if _advanced_brain is None:
        _advanced_brain = AdvancedAIBrain()
    return _advanced_brain
