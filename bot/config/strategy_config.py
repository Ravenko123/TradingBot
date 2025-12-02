"""Shared strategy configuration used by both live trading and backtesting.

This ensures main.py and backtest.py use IDENTICAL settings.
Edit this file to change strategy behavior across all modes.

AVAILABLE STRATEGIES:
1. ict_smc - ICT Smart Money Concepts (swing trading, M5/M15)
2. hft_scalper - High Frequency Trading / Scalping (M1, small pips)
3. mean_reversion - Mean Reversion Strategy (proven algo, M5/M15)
"""

from typing import Any, Dict

# ---------------------------------------------------------------------------
# MASTER STRATEGY CONFIGURATION
# Both main.py and backtest.py import from here to stay in sync.
# ---------------------------------------------------------------------------

# Change this to switch between strategies:
# - "ict_smc" for ICT Smart Money swing trading
# - "hft_scalper" for High Frequency scalping
# - "mean_reversion" for Mean Reversion (proven algorithmic strategy)
STRATEGY_NAME = "ict_smc"  # ICT/SMC is the BEST performer on XAUUSD/BTCUSD!

# Base params (used as defaults, then merged with instrument-specific)
STRATEGY_PARAMS: Dict[str, Any] = {
    # Core settings
    "min_candles": 50,
    "rr_target": 1.5,  # LOWERED: Smaller targets = more frequent wins
    "atr_period": 14,
    
    # Session filtering - ALL sessions for more trades
    "sessions": ("London", "NewYork", "Asian"),  # Added Asian for more trades
    "use_killzones": True,
    "killzone_filter": ("London_Open", "NY_AM", "NY_Lunch", "NY_PM", "Silver_Bullet_NY", "Silver_Bullet_London"),  # More killzones
    
    # Displacement & momentum
    "displacement_lookback": 4,
    "min_displacement": 0.25,
    "require_institutional_candle": False,
    
    # Liquidity sweep
    "sweep_lookback": 25,
    "sweep_reentry_buffer_mult": 0.25,
    "sweep_reentry_bars": 5,
    "allow_synthetic_sweeps": True,
    
    # Market structure
    "bias_mode": "auto",
    "require_bias_alignment": False,  # DISABLED: Allow counter-trend for more trades
    
    # HTF Trend Filter
    "use_htf_filter": True,
    "htf_ema_fast": 20,
    "htf_ema_slow": 50,
    "htf_trend_bars": 100,
    
    # Points of Interest (POI)
    "gap_tolerance_mult": 1.5,
    "require_gap": False,
    "require_block": False,
    "use_breaker_blocks": True,
    "use_mitigation_blocks": True,
    "prefer_unfilled_fvg": True,
    
    # OTE (Optimal Trade Entry)
    "use_ote": True,
    "require_ote": False,
    
    # AMD Pattern
    "use_amd": True,
    
    # Silver Bullet
    "silver_bullet_mode": True,
    
    # Premium/Discount Zones (ICT concept)
    "use_premium_discount": True,
    "require_optimal_zone": False,
    "premium_discount_bonus": 2,
    
    # Asian Range (key liquidity levels)
    "use_asian_range": True,
    "asian_sweep_bonus": 2,
    
    # Power of 3 Pattern
    "use_power_of_3": True,
    "po3_bonus": 2,
    
    # Time-Based Entry Quality
    "use_time_quality": True,
    "min_time_quality": 0.5,
    
    # Liquidity Pools
    "use_liquidity_pools": True,
    "liquidity_pool_bonus": 1,
    
    # NEW: CRT (Candle Range Theory) - High probability reversal pattern
    "use_crt": True,
    "crt_bonus": 3,
    "crt_tolerance": 0.002,
    
    # NEW: CISD (Change in State of Delivery) - Structure shift confirmation
    "use_cisd": True,
    "cisd_bonus": 3,
    "cisd_min_tolerance_ratio": 0.5,
    
    # NEW: ATR Displacement Classification
    "use_atr_displacement": True,
    "reject_judas_swing": False,  # Careful with this - may reduce signals
    "displacement_bonus": 2,
    
    # NEW: Momentum-Weighted FVGs
    "use_momentum_fvg": True,
    "momentum_fvg_bonus": 2,
    "momentum_rsi_period": 14,
    "momentum_strong_threshold": 0.7,
    
    # NEW: Session Range Tracking (London/AM/PM ranges)
    "use_session_ranges": True,
    "session_sweep_bonus": 2,
    
    # NEW: Gap Mitigation
    "use_gap_mitigation": True,
    "gap_mitigation_bonus": 1,
    
    # NEW: Tiered Take Profit System
    # OPTIMIZED FOR BIGGER WINS: Hold longer for more profit
    "use_tiered_tp": True,
    "tp1_rr": 1.0,      # First TP at 1:1 R:R (secure profit)
    "tp1_percent": 0.40,  # Close 40% to lock profit
    "tp2_rr": 1.5,      # Second TP at 1.5:1 R:R
    "tp2_percent": 0.30,  # Close another 30%
    "tp3_rr": 2.0,      # Third TP at 2:1 R:R (full target)
    "tp3_percent": 0.30,  # Let 30% run for max profit
    
    # Risk management
    "atr_buffer_mult": 0.20,  # Slightly wider for volatile pairs
    "min_confluence": 3,  # RAISED: Higher quality trades only
    
    # Volatility filter - DISABLED (was blocking 99% of trades)
    "use_volatility_filter": False,
    "min_atr_percentile": 10,
    "max_atr_percentile": 99,
    
    # Session-specific R:R
    "london_rr_bonus": 0.5,
    "ny_rr_bonus": 0.0,
    
    # Trade grading
    "min_grade": "B",  # RAISED: Only B-grade or better for quality
    
    # Debug & demo
    "demo_mode": False,
    "debug": False,
}

# ---------------------------------------------------------------------------
# PER-INSTRUMENT OPTIMIZATIONS
# Based on ICT/SMC research + 30-day backtests (2025-12-02)
#
# KEY INSIGHTS FROM ENHANCED ICT FEATURES:
# - Premium/Discount zones significantly improve EURUSD (+37% from +24%)
# - Power of 3 pattern helps with trend alignment
# - Asian range sweeps provide key entry triggers
# - Time-based quality helps filter low-probability trades
#
# VOLATILE PAIRS (XAUUSD, BTCUSD, GBPJPY):
# - Need HIGH R:R (3:1+) to capture full moves
# - Benefit from all enhanced ICT features
# - Premium/discount zones very effective
#
# MAJOR FOREX PAIRS (EURUSD, GBPUSD):
# - Work better on M15 (less noise)
# - EURUSD: Enhanced features work well
# - GBPUSD: Currently in difficult market conditions
#
# JPY PAIRS (USDJPY, GBPJPY):
# - Respond to US/Japan sessions differently
# - NY session tends to outperform
# ---------------------------------------------------------------------------

INSTRUMENT_PARAMS: Dict[str, Dict[str, Any]] = {
    # ========================================================================
    # XAUUSD (Gold) - OPTIMAL v8
    # R:R 4.0 = +42.48% (BEST) | R:R 5.0 = +22.3%
    # KEEP 4.0
    # ========================================================================
    "XAUUSD": {
        "rr_target": 4.0,  # OPTIMAL
        "min_confluence": 3,
        "require_bias_alignment": False,
        "use_premium_discount": True,
        "use_power_of_3": True,
        "use_asian_range": True,
        "use_crt": True,
        "use_cisd": True,
        "use_session_ranges": True,
        "use_momentum_fvg": True,
        "use_ote": True,
        "use_breaker_blocks": True,
        "use_atr_displacement": True,
        "sessions": ("London",),
    },
    
    # ========================================================================
    # BTCUSD - OPTIMIZED v4 (BEST CONFIG)
    # R:R 2.5 + conf 3 = +40.2% | R:R 2.2 + conf 4 = +69.96%
    # KEEP THE WINNING CONFIG
    # ========================================================================
    "BTCUSD": {
        "rr_target": 2.2,  # Original winning R:R
        "min_confluence": 4,  # Original winning confluence
        "require_bias_alignment": True,
        "use_premium_discount": True,
        "use_power_of_3": True,
        "use_crt": True,
        "use_atr_displacement": True,
        "displacement_bonus": 3,
        "use_momentum_fvg": True,
        "use_cisd": True,
        "use_ote": True,
        "use_breaker_blocks": True,
        "silver_bullet_mode": True,
        "silver_bullet_bonus": 2,
        "sessions": ("London", "NewYork"),
    },
    
    # ========================================================================
    # EURUSD - OPTIMAL CONFIG v7
    # R:R 1.5, conf 2, bias aligned = +13.55% (BEST)
    # ========================================================================
    "EURUSD": {
        "rr_target": 1.5,
        "min_confluence": 2,
        "require_bias_alignment": True,
        "use_premium_discount": True,
        "use_power_of_3": True,
        "use_asian_range": False,
        "use_cisd": True,
        "use_session_ranges": True,
        "use_gap_mitigation": True,
        "use_momentum_fvg": True,
        "use_ote": False,
        "use_breaker_blocks": False,
        "silver_bullet_mode": False,
        "sessions": ("London",),
    },
    
    # ========================================================================
    # USDJPY - OPTIMAL v8
    # R:R 1.7 = +43.91% (BEST!) | 41 trades | 51.2% WR
    # ========================================================================
    "USDJPY": {
        "rr_target": 1.7,  # OPTIMAL
        "min_confluence": 1,
        "require_bias_alignment": False,
        "use_premium_discount": True,
        "use_power_of_3": True,
        "use_atr_displacement": True,
        "use_cisd": True,
        "use_momentum_fvg": True,
        "use_ote": False,
        "use_breaker_blocks": False,
        "use_crt": True,
        "sessions": ("Asian", "London"),
    },
    
    # ========================================================================
    # GBPJPY - BEST CONFIG v7
    # ========================================================================
    # GBPJPY - OPTIMAL v18
    # M15 timeframe + NY only = +33.81% (BEST!) | 15 trades | 53% WR
    # KEY DISCOVERY: M15 NY >> M5 Asian
    # ========================================================================
    "GBPJPY": {
        "rr_target": 3.0,  # OPTIMAL
        "min_confluence": 1,
        "require_bias_alignment": True,  # CRITICAL
        "use_premium_discount": True,
        "use_crt": True,
        "use_cisd": True,
        "use_atr_displacement": True,
        "use_momentum_fvg": True,
        "use_ote": True,
        "use_breaker_blocks": True,
        "silver_bullet_mode": True,
        "silver_bullet_bonus": 2,
        "sessions": ("NewYork",),  # M15 NY ONLY
    },
    
    # ========================================================================
    # GBPUSD - OPTIMAL v17
    # R:R 1.7 = +22.47% (BEST!) | 27 trades | 52% WR
    # KEY: Asian + NY sessions (London is -10k loss!)
    # ========================================================================
    "GBPUSD": {
        "rr_target": 1.7,  # OPTIMAL
        "min_confluence": 2,
        "require_bias_alignment": True,
        "use_premium_discount": True,
        "use_power_of_3": True,
        "use_momentum_fvg": True,
        "use_session_ranges": True,
        "use_cisd": True,
        "use_crt": False,
        "use_ote": False,
        "use_atr_displacement": True,
        "use_breaker_blocks": False,
        "silver_bullet_mode": False,
        "sessions": ("Asian", "NewYork"),  # CRITICAL: No London!
    },
}

# Instrument-specific timeframes
# Research shows major FX pairs (EURUSD, GBPUSD) work better on M15
# GBPJPY also discovered to work better on M15 (NY session +33.8%)
INSTRUMENT_TIMEFRAMES: Dict[str, str] = {
    "XAUUSD": "M5",   # Gold - M5 captures intraday swings
    "BTCUSD": "M5",   # BTC - M5 for volatile crypto
    "GBPJPY": "M15",  # NOW M15! NY session +33.8% vs M5 +11%
    "USDJPY": "M5",   # JPY pairs - quick moves
    "EURUSD": "M15",  # Major pair - M15 reduces noise
    "GBPUSD": "M15",  # Major pair - M15 cleaner structure
}


def get_params_for_instrument(instrument: str) -> Dict[str, Any]:
    """Get merged params for a specific instrument."""
    params = dict(STRATEGY_PARAMS)  # Copy base params
    if instrument in INSTRUMENT_PARAMS:
        params.update(INSTRUMENT_PARAMS[instrument])  # Override with instrument-specific
    return params


def get_timeframe_for_instrument(instrument: str) -> str:
    """Get optimal timeframe for instrument."""
    return INSTRUMENT_TIMEFRAMES.get(instrument, "M5")


# ============================================================================
# HFT SCALPER CONFIGURATION
# High Frequency Trading / Scalping Strategy Settings
# ============================================================================

HFT_STRATEGY_PARAMS: Dict[str, Any] = {
    # Core settings
    "timeframe": "M1",
    "min_candles": 50,
    
    # 5-8-13 EMA System
    "ema_fast": 5,
    "ema_medium": 8,
    "ema_slow": 13,
    "require_ema_alignment": True,
    "ema_alignment_threshold": 0.0001,
    
    # RSI Settings
    "use_rsi": True,
    "rsi_period": 7,
    "rsi_overbought": 70,
    "rsi_oversold": 30,
    "rsi_extreme_ob": 80,
    "rsi_extreme_os": 20,
    
    # Stochastic Settings
    "use_stochastic": True,
    "stoch_k_period": 5,
    "stoch_d_period": 3,
    "stoch_smooth": 3,
    "stoch_overbought": 80,
    "stoch_oversold": 20,
    
    # MACD Settings (Fast scalping: 6,13,5)
    "use_macd": True,
    "macd_fast": 6,
    "macd_slow": 13,
    "macd_signal": 5,
    "macd_crossover_entry": True,
    "macd_histogram_entry": True,
    "macd_divergence_check": True,
    
    # Bollinger Bands (Fast: 10-period)
    "use_bollinger": True,
    "bb_period": 10,
    "bb_std_dev": 2.0,
    "bb_squeeze_threshold": 0.001,
    "bb_breakout_entry": True,
    "bb_mean_reversion": True,
    "bb_touch_entry": True,
    
    # ADX Settings (Trend Strength)
    "use_adx": True,
    "adx_period": 10,
    "adx_strong_trend": 25,
    "adx_weak_trend": 20,
    "adx_filter_trades": True,
    "use_di_crossover": True,
    
    # VWAP Proxy
    "use_vwap": True,
    "vwap_period": 20,
    "vwap_deviation_mult": 1.5,
    "vwap_mean_reversion": True,
    
    # ATR-based targets
    "use_atr_targets": True,
    "atr_period": 10,
    "tp_atr_mult": 1.2,
    "sl_atr_mult": 1.0,
    
    # Session filtering
    "sessions": ("London", "NewYork"),
    "use_killzones": True,
    "killzone_filter": ("London_Open", "NY_AM", "NY_Lunch"),
    "require_killzone": True,
    "high_liquidity_hours": [(8, 10), (13, 16)],
    
    # Momentum
    "min_momentum_strength": 0.25,
    "min_trend_strength": 0.3,
    "min_consecutive_candles": 2,
    "max_consecutive_candles": 6,
    
    # Micro structure
    "use_micro_fvg": True,
    "micro_fvg_max_size": 0.0008,
    "use_wick_rejection": True,
    "wick_rejection_ratio": 0.6,
    "use_engulfing": True,
    
    # Spread & Volatility
    "max_spread_pips": 2.0,
    "spread_check": True,
    "min_atr_pips": 3.0,
    "max_atr_pips": 25.0,
    
    # Trade management
    "max_trades_per_session": 15,
    "min_minutes_between_trades": 3,
    "max_daily_trades": 50,
    
    # Entry modes
    "use_pullback_entry": True,
    "use_breakout_entry": True,
    "use_mean_reversion": True,
    
    # Confluence
    "min_confluence": 3,
    "min_entry_quality": "B",
    
    # Risk
    "risk_percent": 1.0,
    
    # Debug
    "debug": False,
}

# Per-instrument HFT optimizations with ADX and indicator fine-tuning
HFT_INSTRUMENT_PARAMS: Dict[str, Dict[str, Any]] = {
    # EURUSD - BEST for scalping (tightest spreads, most liquid)
    # Characteristics: Low volatility, tight range, mean-reverting
    "EURUSD": {
        # Risk/Reward
        "tp_atr_mult": 1.0,
        "sl_atr_mult": 0.8,
        "max_spread_pips": 1.5,
        "min_atr_pips": 2.0,
        "max_atr_pips": 15.0,
        "max_trades_per_session": 20,
        # Confluence
        "min_confluence": 3,
        # ADX - Lower threshold (EURUSD often ranges)
        "adx_strong": 22,
        "adx_trending": 18,
        # Bollinger - Slightly wider for mean reversion
        "bb_period": 12,
        "bb_std": 2.0,
        # RSI - Tighter thresholds
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        # Stochastic
        "stoch_oversold": 18,
        "stoch_overbought": 82,
        # Strategy preference
        "prefer_mean_reversion": True,
        "prefer_momentum": False,
    },
    
    # GBPUSD - Good for scalping but more volatile
    # Characteristics: More volatile than EURUSD, trending behavior
    "GBPUSD": {
        # Risk/Reward
        "tp_atr_mult": 1.2,
        "sl_atr_mult": 1.0,
        "max_spread_pips": 2.0,
        "min_atr_pips": 3.0,
        "max_atr_pips": 25.0,
        "max_trades_per_session": 15,
        # Confluence
        "min_confluence": 3,
        # ADX - Standard for trending
        "adx_strong": 25,
        "adx_trending": 20,
        # MACD - Slightly slower for noise reduction
        "macd_fast": 7,
        "macd_slow": 15,
        "macd_signal": 5,
        # Bollinger
        "bb_period": 10,
        "bb_std": 2.0,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # USDJPY - Excellent liquidity, BOJ intervention risk
    # Characteristics: Trending in sessions, respect round numbers
    "USDJPY": {
        # Risk/Reward
        "tp_atr_mult": 1.0,
        "sl_atr_mult": 0.8,
        "max_spread_pips": 1.5,
        "min_atr_pips": 2.0,
        "max_atr_pips": 20.0,
        "max_trades_per_session": 18,
        # Confluence
        "min_confluence": 3,
        # ADX - Works well with trending
        "adx_strong": 25,
        "adx_trending": 20,
        # RSI - Standard
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        # Session focus - Best during Tokyo + London
        "asian_session_boost": 1.5,  # Extra weight for Asian session
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # EURJPY - Cross pair, good volatility
    # Characteristics: Follows EUR and JPY sentiment
    "EURJPY": {
        # Risk/Reward
        "tp_atr_mult": 1.2,
        "sl_atr_mult": 1.0,
        "max_spread_pips": 2.5,
        "min_atr_pips": 3.0,
        "max_atr_pips": 25.0,
        "max_trades_per_session": 12,
        # Confluence
        "min_confluence": 3,
        # ADX
        "adx_strong": 25,
        "adx_trending": 20,
        # Bollinger - Tighter for breakouts
        "bb_period": 10,
        "bb_std": 1.8,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # AUDUSD - Commodity currency, range-bound often
    # Characteristics: Influenced by China/commodities, ranges more
    "AUDUSD": {
        # Risk/Reward
        "tp_atr_mult": 1.0,
        "sl_atr_mult": 0.8,
        "max_spread_pips": 1.8,
        "min_atr_pips": 2.0,
        "max_atr_pips": 18.0,
        "max_trades_per_session": 12,
        # Confluence
        "min_confluence": 3,
        # ADX - Lower threshold (often ranges)
        "adx_strong": 23,
        "adx_trending": 18,
        # Bollinger - Good for mean reversion
        "bb_period": 12,
        "bb_std": 2.0,
        # RSI - Wider for mean reversion
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        # Session focus - Best during Asian
        "asian_session_boost": 1.3,
        # Strategy preference
        "prefer_mean_reversion": True,
        "prefer_momentum": False,
    },
    
    # USDCAD - Correlated with oil
    # Characteristics: Range-bound, influenced by oil prices
    "USDCAD": {
        # Risk/Reward
        "tp_atr_mult": 1.0,
        "sl_atr_mult": 0.9,
        "max_spread_pips": 2.0,
        "min_atr_pips": 2.5,
        "max_atr_pips": 20.0,
        "max_trades_per_session": 10,
        # Confluence
        "min_confluence": 3,
        # ADX - Lower for ranging pair
        "adx_strong": 23,
        "adx_trending": 18,
        # Bollinger
        "bb_period": 12,
        "bb_std": 2.0,
        # Strategy preference
        "prefer_mean_reversion": True,
        "prefer_momentum": False,
    },
    
    # XAUUSD (Gold) - NOT recommended for HFT (wide spreads, volatile)
    # But if you must, use very tight filters
    "XAUUSD": {
        # Risk/Reward - Wider due to volatility
        "tp_atr_mult": 1.5,
        "sl_atr_mult": 1.2,
        "max_spread_pips": 5.0,   # Gold has wider spreads
        "min_atr_pips": 10.0,     # Gold moves more
        "max_atr_pips": 50.0,
        "max_trades_per_session": 8,  # Fewer trades
        # Confluence - Higher requirement
        "min_confluence": 4,
        "require_ema_alignment": True,
        # ADX - Need strong trends only
        "adx_strong": 28,
        "adx_trending": 23,
        "adx_filter_weak": True,
        # Bollinger - Wider for volatile asset
        "bb_period": 14,
        "bb_std": 2.2,
        # RSI - Wider extremes
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        # MACD - Slower for noise
        "macd_fast": 8,
        "macd_slow": 17,
        "macd_signal": 6,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # BTCUSD - NOT recommended for HFT (extremely volatile)
    "BTCUSD": {
        # Risk/Reward
        "tp_atr_mult": 1.5,
        "sl_atr_mult": 1.2,
        "max_spread_pips": 50.0,  # BTC has very wide spreads
        "min_atr_pips": 100.0,
        "max_atr_pips": 1000.0,
        "max_trades_per_session": 5,  # Very few trades
        # Confluence - Very high requirement
        "min_confluence": 5,
        # ADX - Only trade strong trends
        "adx_strong": 30,
        "adx_trending": 25,
        "adx_filter_weak": True,
        # Bollinger - Very wide
        "bb_period": 14,
        "bb_std": 2.5,
        # MACD - Much slower
        "macd_fast": 10,
        "macd_slow": 21,
        "macd_signal": 7,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # GBPJPY - "The Beast" - volatile but tradeable
    "GBPJPY": {
        # Risk/Reward
        "tp_atr_mult": 1.3,
        "sl_atr_mult": 1.0,
        "max_spread_pips": 3.0,
        "min_atr_pips": 5.0,
        "max_atr_pips": 40.0,
        "max_trades_per_session": 10,
        # Confluence
        "min_confluence": 4,
        # ADX - Strong trend requirement
        "adx_strong": 27,
        "adx_trending": 22,
        # Bollinger - Slightly wider
        "bb_period": 12,
        "bb_std": 2.1,
        # RSI - Moderate
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        # MACD
        "macd_fast": 7,
        "macd_slow": 15,
        "macd_signal": 5,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # EURGBP - Range-bound cross, excellent for mean reversion
    "EURGBP": {
        # Risk/Reward
        "tp_atr_mult": 0.9,
        "sl_atr_mult": 0.7,
        "max_spread_pips": 1.5,
        "min_atr_pips": 1.5,
        "max_atr_pips": 12.0,
        "max_trades_per_session": 15,
        # Confluence
        "min_confluence": 3,
        # ADX - Very low threshold (always ranging)
        "adx_strong": 20,
        "adx_trending": 15,
        # Bollinger - Tight for mean reversion
        "bb_period": 10,
        "bb_std": 1.8,
        # RSI - Tight for reversals
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        # Strategy preference - Strong mean reversion bias
        "prefer_mean_reversion": True,
        "prefer_momentum": False,
    },
    
    # NAS100 (NASDAQ) - Index scalping
    "NAS100": {
        # Risk/Reward
        "tp_atr_mult": 1.2,
        "sl_atr_mult": 1.0,
        "max_spread_pips": 3.0,
        "min_atr_pips": 20.0,
        "max_atr_pips": 150.0,
        "max_trades_per_session": 10,
        # Confluence
        "min_confluence": 4,
        # ADX
        "adx_strong": 26,
        "adx_trending": 21,
        # Session focus - Only trade NY session
        "ny_session_only": True,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
    
    # US30 (Dow Jones) - Index scalping
    "US30": {
        # Risk/Reward
        "tp_atr_mult": 1.2,
        "sl_atr_mult": 1.0,
        "max_spread_pips": 3.0,
        "min_atr_pips": 25.0,
        "max_atr_pips": 200.0,
        "max_trades_per_session": 8,
        # Confluence
        "min_confluence": 4,
        # ADX
        "adx_strong": 26,
        "adx_trending": 21,
        # Session focus - Only trade NY session
        "ny_session_only": True,
        # Strategy preference
        "prefer_mean_reversion": False,
        "prefer_momentum": True,
    },
}

# HFT Timeframes (all M1 for true scalping)
HFT_INSTRUMENT_TIMEFRAMES: Dict[str, str] = {
    # Major pairs - M1 scalping
    "EURUSD": "M1",
    "GBPUSD": "M1",
    "USDJPY": "M1",
    "AUDUSD": "M1",
    "USDCAD": "M1",
    # Cross pairs
    "EURJPY": "M1",
    "GBPJPY": "M1",
    "EURGBP": "M1",
    # Commodities (still M1 but with tighter filters)
    "XAUUSD": "M1",
    # Crypto (M1 but very selective)
    "BTCUSD": "M1",
    # Indices
    "NAS100": "M1",
    "US30": "M1",
}


def get_hft_params_for_instrument(instrument: str) -> Dict[str, Any]:
    """Get merged HFT params for a specific instrument."""
    params = dict(HFT_STRATEGY_PARAMS)
    if instrument in HFT_INSTRUMENT_PARAMS:
        params.update(HFT_INSTRUMENT_PARAMS[instrument])
    return params


def get_hft_timeframe_for_instrument(instrument: str) -> str:
    """Get HFT timeframe for instrument (always M1)."""
    return HFT_INSTRUMENT_TIMEFRAMES.get(instrument, "M1")

# Default instrument (best performer from backtests)
DEFAULT_INSTRUMENT = "XAUUSD"  # Changed - XAUUSD is the star!

# Risk settings
DEFAULT_RISK_PERCENT = 2.5  # 2.5% risk per trade (optimized from backtests)

# Initial balance for backtesting
INITIAL_BALANCE = 200_000.0

# Backtest defaults
BACKTEST_DAYS = 30
TIMEFRAME = "M5"


# ============================================================================
# MEAN REVERSION STRATEGY CONFIGURATION
# A proven algorithmic trading strategy based on statistical mean reversion
# ============================================================================

MEAN_REVERSION_PARAMS: Dict[str, Any] = {
    # Timeframe - M5 works well for mean reversion
    "timeframe": "M5",
    "min_candles": 50,
    
    # ========================================
    # BOLLINGER BANDS (Primary Indicator)
    # ========================================
    "bb_period": 20,          # Standard 20-period
    "bb_std_dev": 2.0,        # 2 standard deviations
    "bb_touch_entry": True,   # Enter when price touches band
    "bb_close_beyond": True,  # Or closes beyond band
    
    # ========================================
    # RSI CONFIRMATION
    # ========================================
    "rsi_period": 14,         # Standard 14-period RSI
    "rsi_oversold": 35,       # Relaxed oversold level
    "rsi_overbought": 65,     # Relaxed overbought level
    "rsi_extreme_os": 25,     # Extreme oversold
    "rsi_extreme_ob": 75,     # Extreme overbought
    "require_rsi_confirm": True,  # Require RSI confirmation
    
    # ========================================
    # ADX FILTER (Critical - Range Detection)
    # ========================================
    "adx_period": 14,
    "adx_ranging_threshold": 30,  # Raised to allow more trades
    "require_ranging_market": True,
    
    # ========================================
    # STOCHASTIC CONFIRMATION
    # ========================================
    "use_stochastic": True,
    "stoch_k_period": 14,
    "stoch_d_period": 3,
    "stoch_smooth": 3,
    "stoch_oversold": 25,
    "stoch_overbought": 75,
    
    # ========================================
    # Z-SCORE FILTER
    # ========================================
    "min_z_score": 1.2,  # Lowered from 1.5 for more trades
    
    # ========================================
    # TARGETS & STOPS
    # ========================================
    "target_middle_band": True,  # Target is ALWAYS the mean
    "sl_atr_mult": 1.0,          # Tighter stop for better R:R
    "atr_period": 14,
    "tp_atr_mult": 2.0,          # Fallback if not using middle band
    
    # ========================================
    # SESSION FILTERING
    # ========================================
    "sessions": ("London", "NewYork", "Asian"),
    "prefer_asian": True,  # Asian often ranges
    
    # ========================================
    # TRADE MANAGEMENT
    # ========================================
    "max_trades_per_session": 8,
    "min_minutes_between_trades": 10,
    "max_daily_trades": 20,
    
    # ========================================
    # CONFLUENCE
    # ========================================
    "min_confluence": 2,
    
    # ========================================
    # RISK
    # ========================================
    "risk_percent": 1.5,  # Higher risk due to higher win rate
    
    "debug": True,  # Enable debug for testing
}

# Mean Reversion per-instrument params
# Best instruments: Range-bound pairs with mean-reverting behavior
MEAN_REVERSION_INSTRUMENT_PARAMS: Dict[str, Dict[str, Any]] = {
    # EURUSD - Excellent for mean reversion
    # The most traded pair, often ranges, reverts to mean
    "EURUSD": {
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "min_z_score": 1.5,
        "adx_ranging_threshold": 25,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sl_atr_mult": 1.5,
        "min_confluence": 3,
        "prefer_asian": False,  # London/NY better for EUR
    },
    
    # AUDNZD - BEST for mean reversion!
    # Highly correlated pairs = natural range-bound behavior
    "AUDNZD": {
        "bb_period": 20,
        "bb_std_dev": 1.8,  # Tighter bands
        "min_z_score": 1.3,  # Lower threshold (ranges more)
        "adx_ranging_threshold": 20,  # Very often ranging
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "sl_atr_mult": 1.2,
        "min_confluence": 2,
        "prefer_asian": True,  # Asian session for AUD
    },
    
    # AUDCAD - Also excellent for mean reversion
    # Both commodity currencies, similar behavior
    "AUDCAD": {
        "bb_period": 20,
        "bb_std_dev": 1.8,
        "min_z_score": 1.3,
        "adx_ranging_threshold": 22,
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "sl_atr_mult": 1.3,
        "min_confluence": 2,
        "prefer_asian": True,
    },
    
    # USDCHF - Safe haven, often ranges
    "USDCHF": {
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "min_z_score": 1.5,
        "adx_ranging_threshold": 25,
        "rsi_oversold": 30,
        "rsi_overbought": 70,
        "sl_atr_mult": 1.5,
        "min_confluence": 3,
    },
    
    # EURGBP - Classic range-bound cross
    "EURGBP": {
        "bb_period": 20,
        "bb_std_dev": 1.8,
        "min_z_score": 1.4,
        "adx_ranging_threshold": 22,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "sl_atr_mult": 1.2,
        "min_confluence": 2,
    },
    
    # USDJPY - Can range or trend, be cautious
    "USDJPY": {
        "bb_period": 20,
        "bb_std_dev": 2.0,
        "min_z_score": 1.8,  # Higher threshold
        "adx_ranging_threshold": 23,
        "rsi_oversold": 28,
        "rsi_overbought": 72,
        "sl_atr_mult": 1.5,
        "min_confluence": 4,  # Higher confluence needed
    },
    
    # XAUUSD - Gold can mean revert but volatile
    "XAUUSD": {
        "bb_period": 20,
        "bb_std_dev": 2.2,  # Wider bands for volatility
        "min_z_score": 2.0,  # Strong deviation needed
        "adx_ranging_threshold": 25,
        "rsi_oversold": 25,
        "rsi_overbought": 75,
        "sl_atr_mult": 2.0,  # Wider stop
        "min_confluence": 4,
    },
}

# Mean Reversion timeframes - M5 or M15 work best
MEAN_REVERSION_TIMEFRAMES: Dict[str, str] = {
    "EURUSD": "M5",
    "GBPUSD": "M5",
    "USDJPY": "M5",
    "USDCHF": "M5",
    "AUDNZD": "M15",  # Slower timeframe for range pairs
    "AUDCAD": "M15",
    "EURGBP": "M15",
    "AUDUSD": "M5",
    "XAUUSD": "M5",
    "BTCUSD": "M15",  # Slower for crypto
}


def get_mean_reversion_params_for_instrument(instrument: str) -> Dict[str, Any]:
    """Get merged Mean Reversion params for a specific instrument."""
    params = dict(MEAN_REVERSION_PARAMS)
    if instrument in MEAN_REVERSION_INSTRUMENT_PARAMS:
        params.update(MEAN_REVERSION_INSTRUMENT_PARAMS[instrument])
    return params


def get_mean_reversion_timeframe_for_instrument(instrument: str) -> str:
    """Get Mean Reversion timeframe for instrument."""
    return MEAN_REVERSION_TIMEFRAMES.get(instrument, "M5")


# ============================================================================
# SUPERTREND STRATEGY CONFIGURATION  
# Proven trend-following strategy from TradingView (Dark Vector / SuperTrend)
# Uses ATR-based trailing stops + Hull MA confirmation + Choppiness filter
# ============================================================================

SUPERTREND_PARAMS: Dict[str, Any] = {
    # Timeframe
    "timeframe": "M5",
    "min_candles": 100,
    
    # ========================================
    # SUPERTREND CORE SETTINGS
    # ========================================
    "atr_period": 10,           # ATR period for volatility
    "atr_multiplier": 3.0,      # Multiplier for SuperTrend bands
    
    # ========================================
    # HULL MOVING AVERAGE FILTER
    # ========================================
    "hma_period": 50,           # Hull MA period
    "use_hma_filter": True,     # Require HMA alignment
    
    # ========================================
    # CHOPPINESS INDEX (Noise Gate)
    # ========================================
    "chop_period": 14,          # Lookback for choppiness
    "chop_threshold": 61.8,     # Above = choppy, avoid trading
    "use_chop_filter": True,    # Whether to use filter
    
    # ========================================
    # VOLUME FILTER
    # ========================================
    "volume_filter": False,     # Forex has no real volume
    "volume_period": 20,
    "volume_threshold": 0.8,
    
    # ========================================
    # RISK MANAGEMENT
    # ========================================
    "sl_atr_multiplier": 1.5,   # SL beyond SuperTrend line
    "tp_rr_ratio": 2.0,         # Risk:Reward ratio
    "trailing_stop": True,      # Use SuperTrend as trailing stop
    
    # ========================================
    # POSITION & SIGNAL CONTROL
    # ========================================
    "risk_per_trade": 0.015,    # 1.5% risk per trade
    "max_positions": 1,         # Only 1 position at a time
    "alternate_signals": True,  # Force alternating Long/Short
    
    # ========================================
    # SESSION FILTERING
    # ========================================
    "sessions": ("London", "NewYork"),
    "use_killzones": True,
    
    # ========================================
    # DEBUG
    # ========================================
    "debug": False,
}

# SuperTrend per-instrument params
SUPERTREND_INSTRUMENT_PARAMS: Dict[str, Dict[str, Any]] = {
    # EURUSD - Major pair, medium volatility
    "EURUSD": {
        "atr_multiplier": 2.5,
        "chop_threshold": 60,
        "hma_period": 50,
        "tp_rr_ratio": 2.0,
    },
    
    # GBPUSD - More volatile than EUR
    "GBPUSD": {
        "atr_multiplier": 2.8,
        "chop_threshold": 58,
        "hma_period": 50,
        "tp_rr_ratio": 2.0,
    },
    
    # USDJPY - Trending behavior
    "USDJPY": {
        "atr_multiplier": 2.5,
        "chop_threshold": 60,
        "hma_period": 50,
        "tp_rr_ratio": 2.0,
    },
    
    # XAUUSD - Gold, very volatile
    "XAUUSD": {
        "atr_multiplier": 3.5,
        "chop_threshold": 55,
        "sl_atr_multiplier": 2.0,
        "hma_period": 40,
        "tp_rr_ratio": 2.5,
    },
    
    # BTCUSD - Crypto, extremely volatile
    "BTCUSD": {
        "atr_multiplier": 4.0,
        "chop_threshold": 50,
        "sl_atr_multiplier": 2.5,
        "hma_period": 30,
        "tp_rr_ratio": 2.5,
    },
    
    # GBPJPY - "The Beast", very volatile
    "GBPJPY": {
        "atr_multiplier": 3.5,
        "chop_threshold": 55,
        "sl_atr_multiplier": 2.0,
        "hma_period": 40,
        "tp_rr_ratio": 2.5,
    },
}

# SuperTrend timeframes
SUPERTREND_TIMEFRAMES: Dict[str, str] = {
    "EURUSD": "M5",
    "GBPUSD": "M5",
    "USDJPY": "M5",
    "XAUUSD": "M5",
    "BTCUSD": "M15",  # Slower for crypto
    "GBPJPY": "M5",
}


def get_supertrend_params_for_instrument(instrument: str) -> Dict[str, Any]:
    """Get merged SuperTrend params for a specific instrument."""
    params = dict(SUPERTREND_PARAMS)
    if instrument in SUPERTREND_INSTRUMENT_PARAMS:
        params.update(SUPERTREND_INSTRUMENT_PARAMS[instrument])
    return params


def get_supertrend_timeframe_for_instrument(instrument: str) -> str:
    """Get SuperTrend timeframe for instrument."""
    return SUPERTREND_TIMEFRAMES.get(instrument, "M5")


__all__ = [
    "STRATEGY_NAME",
    "STRATEGY_PARAMS",
    "INSTRUMENT_PARAMS",
    "INSTRUMENT_TIMEFRAMES",
    "get_params_for_instrument",
    "get_timeframe_for_instrument",
    # HFT Scalper exports
    "HFT_STRATEGY_PARAMS",
    "HFT_INSTRUMENT_PARAMS",
    "HFT_INSTRUMENT_TIMEFRAMES",
    "get_hft_params_for_instrument",
    "get_hft_timeframe_for_instrument",
    # Mean Reversion exports
    "MEAN_REVERSION_PARAMS",
    "MEAN_REVERSION_INSTRUMENT_PARAMS",
    "MEAN_REVERSION_TIMEFRAMES",
    "get_mean_reversion_params_for_instrument",
    "get_mean_reversion_timeframe_for_instrument",
    # SuperTrend exports
    "SUPERTREND_PARAMS",
    "SUPERTREND_INSTRUMENT_PARAMS",
    "SUPERTREND_TIMEFRAMES",
    "get_supertrend_params_for_instrument",
    "get_supertrend_timeframe_for_instrument",
    # General
    "DEFAULT_INSTRUMENT",
    "DEFAULT_RISK_PERCENT",
    "INITIAL_BALANCE",
    "BACKTEST_DAYS",
    "TIMEFRAME",
]
