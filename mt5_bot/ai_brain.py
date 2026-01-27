"""
Adaptive Trading Intelligence (ATI) - AI Brain for MT5 Bot
============================================================
A self-learning, market-aware trading intelligence system.

Features:
- Market regime detection (trending/ranging/volatile)
- Performance tracking with full context
- Adaptive parameter optimization
- Inactivity detection & strategy adjustment
- Economic calendar awareness
- Daily intelligence reports

Author: Ultima Trading Bot
Date: 2026-01-26
"""

import json
import sqlite3
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict, field
import numpy as np
import pandas as pd
import requests

# ============================================================================
# CONFIGURATION
# ============================================================================

AI_DATA_DIR = Path(__file__).parent / 'ai_data'
AI_DATA_DIR.mkdir(exist_ok=True)

DATABASE_PATH = AI_DATA_DIR / 'trading_intelligence.db'
BRAIN_STATE_PATH = AI_DATA_DIR / 'brain_state.json'
STRATEGY_EVOLUTION_PATH = AI_DATA_DIR / 'strategy_evolution.json'

# Learning intervals
REGIME_UPDATE_INTERVAL = 60  # seconds - check market regime
PERFORMANCE_ANALYSIS_INTERVAL = 300  # seconds - analyze performance
STRATEGY_EVOLUTION_INTERVAL = 1800  # seconds - evolve strategy (30 min)
INACTIVITY_THRESHOLD_HOURS = 6  # hours without trade = inactive

# Parameter adjustment limits (safety bounds)
PARAM_BOUNDS = {
    'ADX': (5.0, 50.0),
    'ATR_Mult': (0.5, 5.0),
    'EMA_Fast': (5, 50),
    'EMA_Slow': (20, 200),
}


# ============================================================================
# DATA CLASSES
# ============================================================================

@dataclass
class MarketSnapshot:
    """Point-in-time market conditions."""
    timestamp: datetime
    symbol: str
    bid: float
    ask: float
    spread: int
    atr: float
    adx: float
    ema_fast: float
    ema_slow: float
    volatility_percentile: float  # 0-100, how volatile vs recent history
    trend_strength: float  # -1 to 1, negative=down, positive=up
    volume_ratio: float  # current volume vs average
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile'


@dataclass
class TradeRecord:
    """Complete record of a trade with full context."""
    id: str
    symbol: str
    direction: str
    entry_price: float
    exit_price: Optional[float]
    stop_loss: float
    take_profit: float
    lot_size: float
    entry_time: datetime
    exit_time: Optional[datetime]
    profit: Optional[float]
    status: str  # 'open', 'win', 'loss', 'breakeven'
    # Context at entry
    entry_regime: str
    entry_adx: float
    entry_atr: float
    entry_volatility_pct: float
    entry_trend_strength: float
    # Parameters used
    params_used: Dict[str, Any]


@dataclass
class SymbolIntelligence:
    """Intelligence gathered about a specific symbol."""
    symbol: str
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit: float = 0.0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    best_regime: str = 'unknown'
    worst_regime: str = 'unknown'
    last_trade_time: Optional[datetime] = None
    hours_since_trade: float = 999.0
    is_active: bool = True
    current_params: Dict[str, Any] = field(default_factory=dict)
    suggested_params: Dict[str, Any] = field(default_factory=dict)
    confidence_score: float = 0.5  # 0-1, how confident in current params


@dataclass
class MarketRegime:
    """Current market regime for a symbol."""
    symbol: str
    regime: str  # 'trending_up', 'trending_down', 'ranging', 'volatile', 'quiet'
    confidence: float  # 0-1
    adx: float
    atr_percentile: float
    trend_direction: float  # -1 to 1
    recommended_action: str  # 'trade', 'wait', 'reduce_size'
    reasoning: str


# ============================================================================
# DATABASE MANAGER
# ============================================================================

class DatabaseManager:
    """SQLite database for storing all trading intelligence."""
    
    def __init__(self, db_path: Path = DATABASE_PATH):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Market snapshots table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS market_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                bid REAL,
                ask REAL,
                spread INTEGER,
                atr REAL,
                adx REAL,
                ema_fast REAL,
                ema_slow REAL,
                volatility_percentile REAL,
                trend_strength REAL,
                volume_ratio REAL,
                regime TEXT
            )
        ''')
        
        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id TEXT PRIMARY KEY,
                symbol TEXT NOT NULL,
                direction TEXT,
                entry_price REAL,
                exit_price REAL,
                stop_loss REAL,
                take_profit REAL,
                lot_size REAL,
                entry_time TEXT,
                exit_time TEXT,
                profit REAL,
                status TEXT,
                entry_regime TEXT,
                entry_adx REAL,
                entry_atr REAL,
                entry_volatility_pct REAL,
                entry_trend_strength REAL,
                params_used TEXT
            )
        ''')
        
        # Strategy evolution log
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS strategy_evolution (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                symbol TEXT NOT NULL,
                old_params TEXT,
                new_params TEXT,
                reason TEXT,
                expected_improvement TEXT
            )
        ''')
        
        # Daily summaries
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                symbol TEXT NOT NULL,
                trades INTEGER,
                wins INTEGER,
                losses INTEGER,
                profit REAL,
                avg_regime TEXT,
                notes TEXT
            )
        ''')
        
        # Create indexes for performance
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_snapshots_symbol_time ON market_snapshots(symbol, timestamp)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_trades_time ON trades(entry_time)')
        
        conn.commit()
        conn.close()
    
    def store_snapshot(self, snapshot: MarketSnapshot):
        """Store a market snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO market_snapshots 
            (timestamp, symbol, bid, ask, spread, atr, adx, ema_fast, ema_slow,
             volatility_percentile, trend_strength, volume_ratio, regime)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            snapshot.timestamp.isoformat(),
            snapshot.symbol,
            snapshot.bid,
            snapshot.ask,
            snapshot.spread,
            snapshot.atr,
            snapshot.adx,
            snapshot.ema_fast,
            snapshot.ema_slow,
            snapshot.volatility_percentile,
            snapshot.trend_strength,
            snapshot.volume_ratio,
            snapshot.regime
        ))
        conn.commit()
        conn.close()
    
    def store_trade(self, trade: TradeRecord):
        """Store or update a trade record."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO trades
            (id, symbol, direction, entry_price, exit_price, stop_loss, take_profit,
             lot_size, entry_time, exit_time, profit, status, entry_regime, entry_adx,
             entry_atr, entry_volatility_pct, entry_trend_strength, params_used)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            trade.id,
            trade.symbol,
            trade.direction,
            trade.entry_price,
            trade.exit_price,
            trade.stop_loss,
            trade.take_profit,
            trade.lot_size,
            trade.entry_time.isoformat() if trade.entry_time else None,
            trade.exit_time.isoformat() if trade.exit_time else None,
            trade.profit,
            trade.status,
            trade.entry_regime,
            trade.entry_adx,
            trade.entry_atr,
            trade.entry_volatility_pct,
            trade.entry_trend_strength,
            json.dumps(trade.params_used)
        ))
        conn.commit()
        conn.close()
    
    def get_recent_trades(self, symbol: str = None, hours: int = 24) -> List[Dict]:
        """Get recent trades, optionally filtered by symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        if symbol:
            cursor.execute('''
                SELECT * FROM trades WHERE symbol = ? AND entry_time > ? ORDER BY entry_time DESC
            ''', (symbol, cutoff))
        else:
            cursor.execute('''
                SELECT * FROM trades WHERE entry_time > ? ORDER BY entry_time DESC
            ''', (cutoff,))
        
        columns = [desc[0] for desc in cursor.description]
        trades = [dict(zip(columns, row)) for row in cursor.fetchall()]
        conn.close()
        return trades
    
    def get_recent_snapshots(self, symbol: str, hours: int = 24) -> pd.DataFrame:
        """Get recent market snapshots for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        
        df = pd.read_sql_query('''
            SELECT * FROM market_snapshots 
            WHERE symbol = ? AND timestamp > ? 
            ORDER BY timestamp DESC
        ''', conn, params=(symbol, cutoff))
        
        conn.close()
        return df
    
    def get_symbol_stats(self, symbol: str, days: int = 7) -> Dict:
        """Get performance statistics for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_trades,
                SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(COALESCE(profit, 0)) as total_profit,
                AVG(CASE WHEN profit > 0 THEN profit END) as avg_win,
                AVG(CASE WHEN profit < 0 THEN profit END) as avg_loss,
                MAX(entry_time) as last_trade
            FROM trades 
            WHERE symbol = ? AND entry_time > ?
        ''', (symbol, cutoff))
        
        row = cursor.fetchone()
        conn.close()
        
        return {
            'total_trades': row[0] or 0,
            'wins': row[1] or 0,
            'losses': row[2] or 0,
            'total_profit': row[3] or 0.0,
            'avg_win': row[4] or 0.0,
            'avg_loss': row[5] or 0.0,
            'last_trade': row[6]
        }
    
    def log_strategy_change(self, symbol: str, old_params: Dict, new_params: Dict, reason: str):
        """Log a strategy parameter change."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO strategy_evolution (timestamp, symbol, old_params, new_params, reason, expected_improvement)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (
            datetime.now(timezone.utc).isoformat(),
            symbol,
            json.dumps(old_params),
            json.dumps(new_params),
            reason,
            ''
        ))
        conn.commit()
        conn.close()
    
    def cleanup_old_data(self, days_to_keep: int = 30):
        """Remove data older than specified days to prevent database bloat."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days_to_keep)).isoformat()
        
        cursor.execute('DELETE FROM market_snapshots WHERE timestamp < ?', (cutoff,))
        cursor.execute('VACUUM')  # Reclaim space
        
        conn.commit()
        conn.close()


# ============================================================================
# MARKET REGIME DETECTOR
# ============================================================================

class MarketRegimeDetector:
    """Detects current market regime for optimal strategy selection."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self._atr_history = {}  # symbol -> list of recent ATR values
        self._adx_history = {}  # symbol -> list of recent ADX values
    
    def update_history(self, symbol: str, atr: float, adx: float):
        """Update indicator history for percentile calculations."""
        if symbol not in self._atr_history:
            self._atr_history[symbol] = []
            self._adx_history[symbol] = []
        
        self._atr_history[symbol].append(atr)
        self._adx_history[symbol].append(adx)
        
        # Keep last 500 values (roughly 4 hours of 30s data)
        self._atr_history[symbol] = self._atr_history[symbol][-500:]
        self._adx_history[symbol] = self._adx_history[symbol][-500:]
    
    def get_volatility_percentile(self, symbol: str, current_atr: float) -> float:
        """Get current ATR percentile vs recent history."""
        if symbol not in self._atr_history or len(self._atr_history[symbol]) < 10:
            return 50.0  # Default to middle
        
        history = self._atr_history[symbol]
        percentile = (np.sum(np.array(history) < current_atr) / len(history)) * 100
        return percentile
    
    def detect_regime(self, symbol: str, df: pd.DataFrame, 
                      current_adx: float, current_atr: float,
                      ema_fast: float, ema_slow: float) -> MarketRegime:
        """Detect the current market regime for a symbol."""
        
        # Update history
        self.update_history(symbol, current_atr, current_adx)
        
        # Calculate metrics
        volatility_pct = self.get_volatility_percentile(symbol, current_atr)
        trend_direction = (ema_fast - ema_slow) / ema_slow if ema_slow > 0 else 0
        trend_direction = max(-1, min(1, trend_direction * 100))  # Normalize to -1 to 1
        
        # Determine regime
        if current_adx > 30:
            if trend_direction > 0.1:
                regime = 'trending_up'
                confidence = min(1.0, current_adx / 50)
                action = 'trade'
                reason = f"Strong uptrend (ADX={current_adx:.1f}, trend={trend_direction:.2f})"
            elif trend_direction < -0.1:
                regime = 'trending_down'
                confidence = min(1.0, current_adx / 50)
                action = 'trade'
                reason = f"Strong downtrend (ADX={current_adx:.1f}, trend={trend_direction:.2f})"
            else:
                regime = 'volatile'
                confidence = 0.6
                action = 'wait'
                reason = f"High ADX but no clear direction (ADX={current_adx:.1f})"
        elif current_adx > 20:
            if volatility_pct > 70:
                regime = 'volatile'
                confidence = 0.5
                action = 'reduce_size'
                reason = f"Moderate trend, high volatility (ATR pct={volatility_pct:.0f})"
            else:
                regime = 'ranging'
                confidence = 0.6
                action = 'wait'
                reason = f"Weak trend, moderate conditions (ADX={current_adx:.1f})"
        else:
            if volatility_pct < 30:
                regime = 'quiet'
                confidence = 0.7
                action = 'wait'
                reason = f"Low volatility, no trend (ADX={current_adx:.1f}, ATR pct={volatility_pct:.0f})"
            else:
                regime = 'ranging'
                confidence = 0.5
                action = 'wait'
                reason = f"Ranging market (ADX={current_adx:.1f})"
        
        return MarketRegime(
            symbol=symbol,
            regime=regime,
            confidence=confidence,
            adx=current_adx,
            atr_percentile=volatility_pct,
            trend_direction=trend_direction,
            recommended_action=action,
            reasoning=reason
        )


# ============================================================================
# PERFORMANCE TRACKER
# ============================================================================

class PerformanceTracker:
    """Tracks and analyzes trading performance with context."""
    
    def __init__(self, db: DatabaseManager):
        self.db = db
        self._active_trades = {}  # symbol -> TradeRecord
    
    def record_trade_open(self, symbol: str, direction: str, entry_price: float,
                          stop_loss: float, take_profit: float, lot_size: float,
                          regime: MarketRegime, params: Dict) -> str:
        """Record a new trade opening."""
        trade_id = f"{symbol}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"
        
        trade = TradeRecord(
            id=trade_id,
            symbol=symbol,
            direction=direction,
            entry_price=entry_price,
            exit_price=None,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            entry_time=datetime.now(timezone.utc),
            exit_time=None,
            profit=None,
            status='open',
            entry_regime=regime.regime,
            entry_adx=regime.adx,
            entry_atr=0,  # Will be filled
            entry_volatility_pct=regime.atr_percentile,
            entry_trend_strength=regime.trend_direction,
            params_used=params
        )
        
        self._active_trades[symbol] = trade
        self.db.store_trade(trade)
        
        return trade_id
    
    def record_trade_close(self, symbol: str, exit_price: float, profit: float):
        """Record a trade closing."""
        if symbol not in self._active_trades:
            return
        
        trade = self._active_trades[symbol]
        trade.exit_price = exit_price
        trade.exit_time = datetime.now(timezone.utc)
        trade.profit = profit
        trade.status = 'win' if profit > 0 else ('loss' if profit < 0 else 'breakeven')
        
        self.db.store_trade(trade)
        del self._active_trades[symbol]
    
    def get_symbol_intelligence(self, symbol: str) -> SymbolIntelligence:
        """Get comprehensive intelligence about a symbol's performance."""
        stats = self.db.get_symbol_stats(symbol, days=7)
        
        total = stats['total_trades']
        wins = stats['wins']
        losses = stats['losses']
        
        intel = SymbolIntelligence(
            symbol=symbol,
            total_trades=total,
            winning_trades=wins,
            losing_trades=losses,
            total_profit=stats['total_profit'],
            win_rate=(wins / total * 100) if total > 0 else 0,
            avg_win=stats['avg_win'] or 0,
            avg_loss=abs(stats['avg_loss'] or 0),
            profit_factor=(stats['avg_win'] * wins / (abs(stats['avg_loss']) * losses)) if losses > 0 and stats['avg_loss'] else 0,
            last_trade_time=datetime.fromisoformat(stats['last_trade']) if stats['last_trade'] else None,
        )
        
        # Calculate hours since last trade
        if intel.last_trade_time:
            delta = datetime.now(timezone.utc) - intel.last_trade_time.replace(tzinfo=timezone.utc)
            intel.hours_since_trade = delta.total_seconds() / 3600
        
        # Analyze which regime works best
        recent_trades = self.db.get_recent_trades(symbol, hours=168)  # 7 days
        if recent_trades:
            regime_performance = {}
            for t in recent_trades:
                regime = t.get('entry_regime', 'unknown')
                profit = t.get('profit', 0) or 0
                if regime not in regime_performance:
                    regime_performance[regime] = []
                regime_performance[regime].append(profit)
            
            # Find best and worst regimes
            regime_avg = {r: np.mean(profits) for r, profits in regime_performance.items() if profits}
            if regime_avg:
                intel.best_regime = max(regime_avg, key=regime_avg.get)
                intel.worst_regime = min(regime_avg, key=regime_avg.get)
        
        # Set confidence based on data quality
        if total >= 10:
            intel.confidence_score = 0.8
        elif total >= 5:
            intel.confidence_score = 0.6
        elif total >= 1:
            intel.confidence_score = 0.4
        else:
            intel.confidence_score = 0.2
        
        return intel


# ============================================================================
# ADAPTIVE STRATEGY ENGINE
# ============================================================================

class AdaptiveStrategyEngine:
    """Evolves strategy parameters based on performance and market conditions."""
    
    def __init__(self, db: DatabaseManager, performance_tracker: PerformanceTracker,
                 regime_detector: MarketRegimeDetector):
        self.db = db
        self.tracker = performance_tracker
        self.regime_detector = regime_detector
        self._current_params = {}  # symbol -> params
        self._base_params = {}  # symbol -> original params from best_settings.json
        self._load_state()
    
    def _load_state(self):
        """Load saved strategy state."""
        if STRATEGY_EVOLUTION_PATH.exists():
            try:
                data = json.loads(STRATEGY_EVOLUTION_PATH.read_text())
                self._current_params = data.get('current_params', {})
            except Exception:
                pass
    
    def _save_state(self):
        """Save strategy state."""
        try:
            STRATEGY_EVOLUTION_PATH.write_text(json.dumps({
                'current_params': self._current_params,
                'last_updated': datetime.now(timezone.utc).isoformat()
            }, indent=2))
        except Exception:
            pass
    
    def set_base_params(self, symbol: str, params: Dict):
        """Set the base parameters from best_settings.json."""
        self._base_params[symbol] = params.copy()
        if symbol not in self._current_params:
            self._current_params[symbol] = params.copy()
    
    def get_active_params(self, symbol: str) -> Dict:
        """Get currently active parameters for a symbol."""
        if symbol in self._current_params:
            return self._current_params[symbol]
        return self._base_params.get(symbol, {})
    
    def analyze_and_adapt(self, symbol: str, current_regime: MarketRegime) -> Tuple[Dict, str]:
        """Analyze performance and adapt parameters - OPTIMIZES FOR PROFIT, NOT WIN RATE!"""
        intel = self.tracker.get_symbol_intelligence(symbol)
        current = self.get_active_params(symbol)
        base = self._base_params.get(symbol, current)
        
        if not current:
            return current, "No parameters available"
        
        new_params = current.copy()
        changes = []
        
        # Calculate EXPECTANCY (the only metric that matters!)
        # Expectancy = (Win% × AvgWin) - (Loss% × AvgLoss)
        if intel.total_trades >= 5:
            win_pct = intel.win_rate / 100
            loss_pct = 1 - win_pct
            expectancy = (win_pct * intel.avg_win) - (loss_pct * intel.avg_loss)
            
            # Risk-Reward Ratio (actual achieved)
            if intel.avg_loss > 0:
                rr_ratio = intel.avg_win / intel.avg_loss
            else:
                rr_ratio = 999
        else:
            expectancy = 0
            rr_ratio = 1
        
        # CASE 1: Symbol inactive for too long
        if intel.hours_since_trade > INACTIVITY_THRESHOLD_HOURS:
            # Loosen parameters to get more signals
            if current.get('ADX', 30) > PARAM_BOUNDS['ADX'][0] + 5:
                old_adx = current.get('ADX', 30)
                new_adx = max(PARAM_BOUNDS['ADX'][0], old_adx - 5)
                new_params['ADX'] = new_adx
                changes.append(f"ADX {old_adx:.0f}→{new_adx:.0f} (inactive {intel.hours_since_trade:.1f}h)")
        
        # CASE 2: NEGATIVE EXPECTANCY - We're losing money!
        elif intel.total_trades >= 5 and expectancy < 0:
            # Tighten parameters - quality over quantity
            if current.get('ADX', 30) < PARAM_BOUNDS['ADX'][1] - 5:
                old_adx = current.get('ADX', 30)
                new_adx = min(PARAM_BOUNDS['ADX'][1], old_adx + 5)
                new_params['ADX'] = new_adx
                changes.append(f"ADX {old_adx:.0f}→{new_adx:.0f} (negative expectancy ${expectancy:.2f})")
        
        # CASE 3: LOW RR RATIO - Winning but tiny gains vs big losses
        elif intel.total_trades >= 5 and rr_ratio < 1.0 and intel.total_profit < 0:
            # Increase ATR multiplier for wider TP targets
            if current.get('ATR_Mult', 1.5) < PARAM_BOUNDS['ATR_Mult'][1] - 0.5:
                old_mult = current.get('ATR_Mult', 1.5)
                new_mult = min(PARAM_BOUNDS['ATR_Mult'][1], old_mult + 0.5)
                new_params['ATR_Mult'] = new_mult
                changes.append(f"ATR_Mult {old_mult:.1f}→{new_mult:.1f} (RR {rr_ratio:.2f} too low, need bigger wins!)")
        
        # CASE 4: HIGH WIN RATE BUT LOSING MONEY - Classic trap!
        elif intel.total_trades >= 5 and intel.win_rate > 60 and intel.total_profit < 0:
            # This means we're taking tiny profits and big losses
            # Increase ATR multiplier to let winners run
            if current.get('ATR_Mult', 1.5) < PARAM_BOUNDS['ATR_Mult'][1] - 0.5:
                old_mult = current.get('ATR_Mult', 1.5)
                new_mult = min(PARAM_BOUNDS['ATR_Mult'][1], old_mult + 0.5)
                new_params['ATR_Mult'] = new_mult
                changes.append(f"ATR_Mult {old_mult:.1f}→{new_mult:.1f} (HIGH WIN RATE BUT LOSING! Let winners run!)")
            
            # Also tighten ADX to avoid bad setups
            if current.get('ADX', 30) < PARAM_BOUNDS['ADX'][1] - 5:
                old_adx = current.get('ADX', 30)
                new_adx = min(PARAM_BOUNDS['ADX'][1], old_adx + 5)
                new_params['ADX'] = new_adx
                changes.append(f"ADX {old_adx:.0f}→{new_adx:.0f} (filter out weak setups)")
        
        # CASE 5: POSITIVE EXPECTANCY & PROFIT - Strategy working!
        elif intel.total_trades >= 10 and expectancy > 5 and intel.total_profit > 50:
            # Only make minor adjustments if needed
            # Check if we can slightly relax to get more trades
            if current.get('ADX', 30) > base.get('ADX', 30) + 5:
                old_adx = current.get('ADX', 30)
                new_adx = max(base.get('ADX', 30), old_adx - 2)
                new_params['ADX'] = new_adx
                changes.append(f"ADX {old_adx:.0f}→{new_adx:.0f} (strong expectancy ${expectancy:.2f}, can relax)")
        
        # CASE 6: PROFIT FACTOR check (gross profit / gross loss)
        elif intel.total_trades >= 5 and intel.profit_factor > 0 and intel.profit_factor < 1.2:
            # PF < 1.2 is marginal, tighten up
            if current.get('ADX', 30) < PARAM_BOUNDS['ADX'][1] - 5:
                old_adx = current.get('ADX', 30)
                new_adx = min(PARAM_BOUNDS['ADX'][1], old_adx + 3)
                new_params['ADX'] = new_adx
                changes.append(f"ADX {old_adx:.0f}→{new_adx:.0f} (PF {intel.profit_factor:.2f} too low)")
        
        # CASE 7: Regime-based adaptation (ONLY if we're already profitable!)
        if current_regime.regime in ['volatile'] and current_regime.atr_percentile > 80:
            # High volatility - increase ATR multiplier for wider stops
            if current.get('ATR_Mult', 1.5) < PARAM_BOUNDS['ATR_Mult'][1] - 0.5:
                old_mult = current.get('ATR_Mult', 1.5)
                new_mult = min(PARAM_BOUNDS['ATR_Mult'][1], old_mult + 0.5)
                new_params['ATR_Mult'] = new_mult
                changes.append(f"ATR_Mult {old_mult:.1f}→{new_mult:.1f} (high volatility)")
        elif current_regime.regime == 'quiet' and current_regime.atr_percentile < 20:
            # Low volatility - decrease ATR multiplier for tighter stops
            if current.get('ATR_Mult', 1.5) > PARAM_BOUNDS['ATR_Mult'][0] + 0.5:
                old_mult = current.get('ATR_Mult', 1.5)
                new_mult = max(PARAM_BOUNDS['ATR_Mult'][0], old_mult - 0.25)
                new_params['ATR_Mult'] = new_mult
                changes.append(f"ATR_Mult {old_mult:.1f}→{new_mult:.1f} (low volatility)")
        
        # Apply changes if any
        if changes:
            self.db.log_strategy_change(symbol, current, new_params, '; '.join(changes))
            self._current_params[symbol] = new_params
            self._save_state()
            return new_params, f"Adapted: {'; '.join(changes)}"
        
        return current, "No changes needed"


# ============================================================================
# ECONOMIC CALENDAR
# ============================================================================

class EconomicCalendar:
    """Fetches and tracks high-impact economic events."""
    
    def __init__(self):
        self._events = []
        self._last_fetch = None
        self._fetch_interval = 3600  # 1 hour
    
    def fetch_events(self) -> List[Dict]:
        """Fetch economic events from free API."""
        now = datetime.now(timezone.utc)
        
        # Only fetch periodically
        if self._last_fetch and (now - self._last_fetch).seconds < self._fetch_interval:
            return self._events
        
        try:
            # Using a free economic calendar API (Forex Factory alternative)
            # This is a simplified version - in production you'd use a proper API
            self._events = self._get_mock_events()  # Placeholder
            self._last_fetch = now
        except Exception:
            pass
        
        return self._events
    
    def _get_mock_events(self) -> List[Dict]:
        """Get important economic events. In production, fetch from real API."""
        # Key events that affect multiple pairs
        return [
            {
                'name': 'NFP',
                'time': 'First Friday of month 13:30 UTC',
                'impact': 'high',
                'currencies': ['USD', 'XAUUSD', 'EURUSD', 'GBPUSD']
            },
            {
                'name': 'FOMC',
                'time': 'Check calendar',
                'impact': 'high',
                'currencies': ['USD', 'XAUUSD', 'EURUSD', 'GBPUSD', 'USDJPY']
            }
        ]
    
    def is_high_impact_period(self, symbol: str) -> Tuple[bool, str]:
        """Check if we're in a high-impact news period for a symbol."""
        # In production, this would check actual event times
        # For now, avoid trading 30 min before/after major events
        now = datetime.now(timezone.utc)
        
        # Simple heuristic: avoid first Friday 13:00-14:30 UTC (NFP)
        if now.weekday() == 4 and now.day <= 7:  # First Friday
            if 13 <= now.hour <= 14:
                return True, "NFP release window - avoiding new trades"
        
        return False, ""


# ============================================================================
# INTELLIGENCE BRAIN - MAIN COORDINATOR
# ============================================================================

class TradingIntelligenceBrain:
    """Main AI brain that coordinates all intelligence modules."""
    
    def __init__(self):
        self.db = DatabaseManager()
        self.regime_detector = MarketRegimeDetector(self.db)
        self.performance_tracker = PerformanceTracker(self.db)
        self.strategy_engine = AdaptiveStrategyEngine(
            self.db, self.performance_tracker, self.regime_detector
        )
        self.calendar = EconomicCalendar()
        self.position_manager = AdvancedPositionManager(self.db, self.regime_detector)
        
        self._running = False
        self._thread = None
        self._current_regimes = {}  # symbol -> MarketRegime
        self._insights = []  # Recent insights for reporting
        
        self._load_state()
    
    def _load_state(self):
        """Load brain state from disk."""
        if BRAIN_STATE_PATH.exists():
            try:
                data = json.loads(BRAIN_STATE_PATH.read_text())
                # Restore any persistent state
            except Exception:
                pass
    
    def _save_state(self):
        """Save brain state to disk."""
        try:
            state = {
                'last_update': datetime.now(timezone.utc).isoformat(),
                'insights': self._insights[-20:],  # Keep last 20 insights
            }
            BRAIN_STATE_PATH.write_text(json.dumps(state, indent=2))
        except Exception:
            pass
    
    def initialize_symbol(self, symbol: str, base_params: Dict):
        """Initialize tracking for a symbol."""
        self.strategy_engine.set_base_params(symbol, base_params)
    
    def process_market_data(self, symbol: str, df: pd.DataFrame, 
                            sym_info: Dict, params: Dict) -> Dict:
        """Process market data and return enhanced signal info."""
        if df is None or len(df) < 50:
            return {'should_trade': False, 'reason': 'Insufficient data'}
        
        # Get current indicators
        adx = df['ADX'].iloc[-1] if 'ADX' in df.columns else 0
        atr = df['ATR'].iloc[-1] if 'ATR' in df.columns else 0
        ema_fast = df['EMA_Fast'].iloc[-1] if 'EMA_Fast' in df.columns else 0
        ema_slow = df['EMA_Slow'].iloc[-1] if 'EMA_Slow' in df.columns else 0
        
        # Detect regime
        regime = self.regime_detector.detect_regime(
            symbol, df, adx, atr, ema_fast, ema_slow
        )
        self._current_regimes[symbol] = regime
        
        # Store market snapshot
        snapshot = MarketSnapshot(
            timestamp=datetime.now(timezone.utc),
            symbol=symbol,
            bid=sym_info.get('bid', 0),
            ask=sym_info.get('ask', 0),
            spread=sym_info.get('spread', 0),
            atr=atr,
            adx=adx,
            ema_fast=ema_fast,
            ema_slow=ema_slow,
            volatility_percentile=regime.atr_percentile,
            trend_strength=regime.trend_direction,
            volume_ratio=1.0,  # Could add volume analysis
            regime=regime.regime
        )
        self.db.store_snapshot(snapshot)
        
        # Check economic calendar
        is_news, news_reason = self.calendar.is_high_impact_period(symbol)
        if is_news:
            return {
                'should_trade': False,
                'reason': news_reason,
                'regime': regime,
                'params': params
            }
        
        # Get adapted parameters
        adapted_params, adaptation_reason = self.strategy_engine.analyze_and_adapt(
            symbol, regime
        )
        
        if adaptation_reason != "No changes needed":
            self._insights.append({
                'time': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'type': 'adaptation',
                'message': adaptation_reason
            })
        
        # Determine if we should trade
        should_trade = regime.recommended_action == 'trade'
        
        return {
            'should_trade': should_trade,
            'reason': regime.reasoning,
            'regime': regime,
            'params': adapted_params,
            'adaptation': adaptation_reason
        }
    
    def on_trade_opened(self, symbol: str, direction: str, entry: float,
                        sl: float, tp: float, lots: float, params: Dict) -> str:
        """Called when a trade is opened."""
        regime = self._current_regimes.get(symbol, MarketRegime(
            symbol=symbol, regime='unknown', confidence=0.5, adx=0,
            atr_percentile=50, trend_direction=0, recommended_action='trade',
            reasoning='No regime data'
        ))
        
        trade_id = self.performance_tracker.record_trade_open(
            symbol, direction, entry, sl, tp, lots, regime, params
        )
        
        self._insights.append({
            'time': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'type': 'trade_open',
            'message': f"Opened {direction} in {regime.regime} regime (ADX={regime.adx:.1f})"
        })
        
        return trade_id
    
    def on_trade_closed(self, symbol: str, exit_price: float, profit: float):
        """Called when a trade is closed."""
        self.performance_tracker.record_trade_close(symbol, exit_price, profit)
        self.position_manager.clear_position_state(symbol)
        
        result = "WIN" if profit > 0 else "LOSS"
        self._insights.append({
            'time': datetime.now(timezone.utc).isoformat(),
            'symbol': symbol,
            'type': 'trade_close',
            'message': f"{result}: ${profit:.2f}"
        })
        
        self._save_state()
    
    def manage_open_position(self, symbol: str, position: Dict, df: pd.DataFrame) -> Optional['PositionManagementDecision']:
        """Analyze and decide how to manage an open position."""
        regime = self._current_regimes.get(symbol)
        if not regime:
            return None
        
        # Get entry regime from tracked positions or default to current
        entry_regime = position.get('entry_regime', regime.regime)
        
        # Get current price
        direction = position.get('direction', 'BUY')
        if 'current_price' in position:
            current_price = position['current_price']
        elif direction == 'BUY':
            current_price = position.get('bid', position.get('entry_price', 0))
        else:
            current_price = position.get('ask', position.get('entry_price', 0))
        
        # Get management decision
        decision = self.position_manager.analyze_position(
            symbol, position, current_price, df, regime, entry_regime
        )
        
        # Log if taking action
        if decision.action != 'hold':
            self._insights.append({
                'time': datetime.now(timezone.utc).isoformat(),
                'symbol': symbol,
                'type': 'position_management',
                'message': f"{decision.action}: {decision.reason}"
            })
        
        return decision
    
    def get_symbol_report(self, symbol: str) -> str:
        """Get a detailed report for a symbol - PROFIT FOCUSED!"""
        intel = self.performance_tracker.get_symbol_intelligence(symbol)
        regime = self._current_regimes.get(symbol)
        params = self.strategy_engine.get_active_params(symbol)
        
        # Calculate EXPECTANCY - the only metric that matters!
        if intel.total_trades >= 1:
            win_pct = intel.win_rate / 100
            loss_pct = 1 - win_pct
            expectancy = (win_pct * intel.avg_win) - (loss_pct * intel.avg_loss)
            
            # Risk-Reward Ratio (actual achieved)
            if intel.avg_loss > 0:
                rr_ratio = intel.avg_win / intel.avg_loss
            else:
                rr_ratio = 999
        else:
            expectancy = 0
            rr_ratio = 0
        
        report = [
            f"📊 {symbol} Intelligence Report",
            "═" * 30,
            f"💰 Performance (7 days):",
            f"   TOTAL PROFIT: ${intel.total_profit:.2f}",
            f"   EXPECTANCY: ${expectancy:.2f} per trade" if intel.total_trades >= 1 else "   EXPECTANCY: N/A (need more data)",
            f"   Profit Factor: {intel.profit_factor:.2f}" if intel.profit_factor else "   P/F: N/A",
            f"",
            f"📈 Trade Stats:",
            f"   Trades: {intel.total_trades} ({intel.winning_trades}W/{intel.losing_trades}L)",
            f"   Win Rate: {intel.win_rate:.1f}% (irrelevant if not profitable!)",
            f"   RR Ratio: {rr_ratio:.2f}" if intel.total_trades >= 1 else "   RR Ratio: N/A",
            f"   Avg Win: ${intel.avg_win:.2f} | Avg Loss: ${intel.avg_loss:.2f}",
            f"",
            f"⏰ Activity:",
            f"   Last Trade: {intel.hours_since_trade:.1f}h ago" if intel.last_trade_time else "   Last Trade: Never",
            f"   Status: {'⚠️ INACTIVE' if intel.hours_since_trade > INACTIVITY_THRESHOLD_HOURS else '✅ Active'}",
            f"",
        ]
        
        if regime:
            report.extend([
                f"🌡️ Current Regime: {regime.regime.upper()}",
                f"   ADX: {regime.adx:.1f} | Volatility: {regime.atr_percentile:.0f}%ile",
                f"   Trend: {'↑' if regime.trend_direction > 0 else '↓' if regime.trend_direction < 0 else '→'} ({regime.trend_direction:.2f})",
                f"   Action: {regime.recommended_action.upper()}",
                f"",
            ])
        
        if params:
            report.extend([
                f"⚙️ Active Parameters:",
                f"   EMA: {params.get('EMA_Fast')}/{params.get('EMA_Slow')}",
                f"   ADX: >{params.get('ADX')}",
                f"   ATR×: {params.get('ATR_Mult')}",
            ])
        
        if intel.best_regime != 'unknown':
            report.extend([
                f"",
                f"💡 Best regime: {intel.best_regime}",
                f"⚠️ Worst regime: {intel.worst_regime}",
            ])
        
        # AI Verdict
        if intel.total_trades >= 5:
            if expectancy > 5 and intel.total_profit > 50:
                verdict = "🚀 MONEY PRINTER MODE - Keep going!"
            elif expectancy > 0 and intel.total_profit > 0:
                verdict = "✅ Profitable - Needs more data"
            elif intel.win_rate > 60 and intel.total_profit < 0:
                verdict = "⚠️ HIGH WIN RATE BUT LOSING! Taking tiny profits, big losses!"
            elif expectancy < 0:
                verdict = "🔴 LOSING MONEY - AI adapting parameters..."
            else:
                verdict = "⏳ Learning phase - collecting data..."
        else:
            verdict = "📚 Early days - need at least 5 trades for analysis"
        
        report.extend([
            f"",
            f"🎯 AI VERDICT: {verdict}",
        ])
        
        return '\n'.join(report)
    
    def get_daily_summary(self) -> str:
        """Get a daily summary across all symbols."""
        lines = [
            "🧠 AI Trading Intelligence - Daily Summary",
            "═" * 40,
            f"📅 {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
        ]
        
        total_profit = 0
        total_trades = 0
        
        for symbol in self._current_regimes.keys():
            stats = self.db.get_symbol_stats(symbol, days=1)
            total_profit += stats['total_profit']
            total_trades += stats['total_trades']
            
            if stats['total_trades'] > 0:
                lines.append(f"  {symbol}: {stats['total_trades']} trades, ${stats['total_profit']:.2f}")
        
        lines.extend([
            "",
            f"📊 Total: {total_trades} trades, ${total_profit:.2f}",
            "",
            "🔄 Recent Adaptations:",
        ])
        
        recent_insights = [i for i in self._insights if i['type'] == 'adaptation'][-5:]
        for insight in recent_insights:
            lines.append(f"  • {insight['symbol']}: {insight['message']}")
        
        if not recent_insights:
            lines.append("  • No adaptations today")
        
        return '\n'.join(lines)
    
    def cleanup(self):
        """Cleanup old data."""
        self.db.cleanup_old_data(days_to_keep=30)


# ============================================================================
# ADVANCED POSITION MANAGER - Intelligent Trade Management
# ============================================================================

@dataclass
class PositionManagementDecision:
    """AI decision for managing an open position."""
    action: str  # 'hold', 'close_full', 'close_partial', 'trail_sl', 'move_breakeven', 'trail_tp'
    reason: str
    confidence: float  # 0-1
    new_sl: Optional[float] = None
    new_tp: Optional[float] = None
    close_percentage: float = 100.0  # For partial closes


class AdvancedPositionManager:
    """AI-powered position management - closes, trails, takes partials intelligently."""
    
    def __init__(self, db: DatabaseManager, regime_detector: MarketRegimeDetector):
        self.db = db
        self.regime_detector = regime_detector
        self._position_states = {}  # symbol -> state tracking
        self._management_stats = {}  # Track what management actions work
        self._load_management_prefs()
    
    def _load_management_prefs(self):
        """Load learned management preferences."""
        # These will be learned over time
        self.prefs = {
            'breakeven_trigger_rr': 1.5,  # Only move to BE after solid profit
            'partial_profit_rr': 2.0,  # Take 50% only at 2:1 RR (real profit)
            'trail_sl_distance_atr': 1.5,  # Trail with buffer
            'trail_trigger_rr': 2.5,  # Only trail at 2.5+ RR
            'trail_tp_on_momentum': False,  # Disabled during learning
            'close_on_bias_shift': False,  # Disabled - let trades run
            'close_on_weak_momentum': False,  # Disabled - too aggressive
            'aggressive_management': False,  # Conservative during learning
        }
    
    def analyze_position(self, symbol: str, position: Dict, current_price: float,
                         df: pd.DataFrame, current_regime: 'MarketRegime',
                         entry_regime: str) -> PositionManagementDecision:
        """Analyze an open position and decide how to manage it."""
        
        direction = position.get('direction', 'BUY')
        entry_price = position.get('entry_price', current_price)
        stop_loss = position.get('sl', entry_price)
        take_profit = position.get('tp', entry_price)
        profit = position.get('profit', 0)
        
        # Calculate current metrics
        if 'ATR' in df.columns and len(df) > 0:
            atr = df['ATR'].iloc[-1]
        else:
            atr = abs(take_profit - entry_price) / 2  # Rough estimate
        
        if 'ADX' in df.columns and len(df) > 0:
            adx = df['ADX'].iloc[-1]
        else:
            adx = 25  # Default
        
        # Calculate risk-reward achieved
        risk = abs(entry_price - stop_loss)
        if risk > 0:
            if direction == 'BUY':
                profit_pips = current_price - entry_price
            else:
                profit_pips = entry_price - current_price
            rr_achieved = profit_pips / risk
        else:
            rr_achieved = 0
        
        # Initialize position state if new
        if symbol not in self._position_states:
            self._position_states[symbol] = {
                'breakeven_set': False,
                'partial_taken': False,
                'sl_trailed': False,
                'entry_adx': current_regime.adx,
                'peak_profit': profit,
                'entry_regime': entry_regime
            }
        
        state = self._position_states[symbol]
        state['peak_profit'] = max(state['peak_profit'], profit)
        
        # ============ DECISION LOGIC ============
        
        # 1. BIAS SHIFT - Market regime reversed
        if self.prefs['close_on_bias_shift']:
            if entry_regime in ['trending_up', 'ranging'] and current_regime.regime == 'trending_down' and direction == 'BUY':
                return PositionManagementDecision(
                    action='close_full',
                    reason=f"Bias shifted: {entry_regime} → {current_regime.regime}",
                    confidence=0.8
                )
            elif entry_regime in ['trending_down', 'ranging'] and current_regime.regime == 'trending_up' and direction == 'SELL':
                return PositionManagementDecision(
                    action='close_full',
                    reason=f"Bias shifted: {entry_regime} → {current_regime.regime}",
                    confidence=0.8
                )
        
        # 2. MOMENTUM DEATH - ADX collapsed
        if self.prefs['close_on_weak_momentum']:
            if adx < 15 and state['entry_adx'] > 25 and profit > 0:
                return PositionManagementDecision(
                    action='close_full',
                    reason=f"Momentum died (ADX {adx:.1f}, was {state['entry_adx']:.1f})",
                    confidence=0.7
                )
        
        # 3. PROFIT RETRACEMENT - Disabled during learning to avoid commission death spiral
        # Only re-enable once we have 100+ profitable trades
        # if state['peak_profit'] > 50 and profit < state['peak_profit'] * 0.3 and profit > 10:
        #     return PositionManagementDecision(
        #         action='close_full',
        #         reason=f"Retraced 70% of peak (${state['peak_profit']:.0f} → ${profit:.0f})",
        #         confidence=0.65
        #     )
        
        # 4. BREAKEVEN STOP - Reached target RR (now earlier)
        if not state['breakeven_set'] and rr_achieved >= self.prefs['breakeven_trigger_rr']:
            state['breakeven_set'] = True
            new_sl = entry_price + (0.2 * atr if direction == 'BUY' else -0.2 * atr)  # Slight buffer
            return PositionManagementDecision(
                action='move_breakeven',
                reason=f"Reached {rr_achieved:.1f}:1 RR → BE+buffer",
                confidence=0.9,
                new_sl=new_sl
            )
        
        # 5. PARTIAL PROFIT - Take 50% a bit sooner
        if not state['partial_taken'] and rr_achieved >= self.prefs['partial_profit_rr']:
            state['partial_taken'] = True
            return PositionManagementDecision(
                action='close_partial',
                reason=f"Reached {rr_achieved:.1f}:1 RR → Take 50%",
                confidence=0.85,
                close_percentage=50.0
            )
        
        # 6. TRAILING STOP - Strong momentum, lock in profits (earlier and closer)
        if rr_achieved >= self.prefs.get('trail_trigger_rr', 2.0) and current_regime.adx > 25:
            trail_distance = self.prefs['trail_sl_distance_atr'] * atr
            if direction == 'BUY':
                new_sl = current_price - trail_distance
                if new_sl > stop_loss:  # Only trail up
                    return PositionManagementDecision(
                        action='trail_sl',
                        reason=f"Strong momentum (ADX {current_regime.adx:.1f}), trail SL",
                        confidence=0.8,
                        new_sl=new_sl
                    )
            else:  # SELL
                new_sl = current_price + trail_distance
                if new_sl < stop_loss:  # Only trail down
                    return PositionManagementDecision(
                        action='trail_sl',
                        reason=f"Strong momentum (ADX {current_regime.adx:.1f}), trail SL",
                        confidence=0.8,
                        new_sl=new_sl
                    )
        
        # 7. TRAILING TP - Extreme momentum, let it run
        if self.prefs['trail_tp_on_momentum'] and rr_achieved > 3.0 and current_regime.adx > 40:
            extension = 1.5 * atr
            if direction == 'BUY':
                new_tp = take_profit + extension
            else:
                new_tp = take_profit - extension
            
            return PositionManagementDecision(
                action='trail_tp',
                reason=f"Extreme momentum (ADX {current_regime.adx:.1f}), extend TP",
                confidence=0.7,
                new_tp=new_tp
            )
        
        # Default: Hold
        return PositionManagementDecision(
            action='hold',
            reason="All conditions favorable, holding position",
            confidence=0.6
        )
    
    def clear_position_state(self, symbol: str):
        """Clear state when position closes."""
        if symbol in self._position_states:
            del self._position_states[symbol]
    
    def record_management_outcome(self, symbol: str, action: str, profit: float):
        """Learn which management actions work - OPTIMIZES FOR EXPECTANCY!"""
        if symbol not in self._management_stats:
            self._management_stats[symbol] = {}
        
        if action not in self._management_stats[symbol]:
            self._management_stats[symbol][action] = {
                'count': 0, 
                'total_profit': 0,
                'wins': 0,
                'losses': 0,
                'total_win_amount': 0,
                'total_loss_amount': 0
            }
        
        stats = self._management_stats[symbol][action]
        stats['count'] += 1
        stats['total_profit'] += profit
        
        if profit > 0:
            stats['wins'] += 1
            stats['total_win_amount'] += profit
        else:
            stats['losses'] += 1
            stats['total_loss_amount'] += abs(profit)
        
        # Calculate EXPECTANCY for this management action
        total_managed = sum(s['count'] for s in self._management_stats.get(symbol, {}).values())
        if total_managed >= 10:
            # Calculate expectancy for this action
            win_rate = stats['wins'] / stats['count'] if stats['count'] > 0 else 0
            loss_rate = 1 - win_rate
            avg_win = stats['total_win_amount'] / stats['wins'] if stats['wins'] > 0 else 0
            avg_loss = stats['total_loss_amount'] / stats['losses'] if stats['losses'] > 0 else 0
            expectancy = (win_rate * avg_win) - (loss_rate * avg_loss)
            
            # Adjust strategy based on EXPECTANCY, not just profit count
            if expectancy > 10 and stats['total_profit'] > 100:
                # This management style is PRINTING MONEY
                self.prefs['aggressive_management'] = True
                self.prefs['partial_profit_rr'] = 1.2  # Take profits earlier (working!)
            elif expectancy < 0 and stats['total_profit'] < -50:
                # This management style is LOSING MONEY
                self.prefs['aggressive_management'] = False
                self.prefs['partial_profit_rr'] = 2.5  # Let trades run more (current approach failing)


# ============================================================================
# SINGLETON INSTANCE
# ============================================================================

_brain_instance = None

def get_brain() -> TradingIntelligenceBrain:
    """Get the singleton brain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = TradingIntelligenceBrain()
    return _brain_instance
