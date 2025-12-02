"""ICT Smart Money Concepts strategy implementation - Enhanced Edition.

This strategy implements the full ICT (Inner Circle Trader) methodology including:
- Liquidity sweeps and grabs
- Order blocks, breaker blocks, and mitigation blocks
- Fair value gaps with consequent encroachment
- Optimal Trade Entry (OTE) zones (61.8-78.6% Fibonacci)
- Killzone-based session filtering
- AMD (Accumulation-Manipulation-Distribution) patterns
- Silver Bullet setups
- Market structure shifts (MSS) and breaks of structure (BOS)
- Institutional candle detection
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from datetime import datetime
from typing import Deque, Iterable, List, Optional, Tuple

from core.order_types import OrderSide, StrategySignal, TakeProfitLevel
from core.utils import (
    AMDPattern,
    AsianRange,
    ATRDisplacement,
    BreakerBlock,
    Candle,
    CISDSignal,
    CRTPattern,
    FairValueGap,
    GapMitigation,
    InstitutionalCandle,
    Killzone,
    LiquidityPool,
    LiquiditySweep,
    MitigationBlock,
    MomentumFVG,
    OTE,
    OrderBlock,
    PowerOf3,
    PremiumDiscount,
    SessionRange,
    SwingPoint,
    active_killzone,
    active_session,
    asian_range_swept,
    calculate_asian_range,
    calculate_atr,
    calculate_displacement,
    calculate_ote,
    calculate_premium_discount,
    calculate_session_range,
    check_session_sweep,
    classify_atr_displacement,
    detect_amd_pattern,
    detect_bos,
    detect_breaker_blocks,
    detect_cisd,
    detect_crt_pattern,
    detect_fair_value_gaps,
    detect_gap_mitigation,
    detect_institutional_candles,
    detect_liquidity_sweep,
    detect_mitigation_blocks,
    detect_momentum_fvgs,
    detect_mss,
    detect_order_blocks,
    detect_power_of_3,
    detect_unfilled_fvg,
    find_ote_from_swings,
    fvg_consequent_encroachment,
    get_all_session_ranges,
    get_entry_quality,
    get_strongest_fvg,
    identify_liquidity_pools,
    is_judas_swing,
    is_real_displacement,
    is_silver_bullet_window,
    nearest_liquidity_pool,
    price_in_discount,
    price_in_ote,
    price_in_premium,
    price_zone,
    recent_structure,
    swing_points,
)
from strategies import BaseStrategy, StrategyRegistry


@dataclass
class MarketContext:
    """Comprehensive container describing the current ICT market state."""

    direction: OrderSide
    sweep: Optional[LiquiditySweep]
    displacement: float
    atr: float
    session: str
    killzone: Optional[Killzone]
    
    # Order flow structures
    gap: Optional[FairValueGap]
    unfilled_gaps: List[FairValueGap]
    block: Optional[OrderBlock]
    breaker: Optional[BreakerBlock]
    mitigation: Optional[MitigationBlock]
    
    # Advanced ICT concepts
    ote: Optional[OTE]
    amd: Optional[AMDPattern]
    institutional_candles: List[InstitutionalCandle]
    is_silver_bullet: bool
    
    # Enhanced ICT concepts (from previous session)
    asian_range: Optional[AsianRange]
    premium_discount: Optional[PremiumDiscount]
    power_of_3: Optional[PowerOf3]
    liquidity_pools: List[LiquidityPool]
    entry_quality: float  # 0.0 - 1.0 based on time window
    
    # NEW: Advanced ICT concepts (from research)
    crt_pattern: Optional[CRTPattern]  # Candle Range Theory
    cisd_signal: Optional[CISDSignal]  # Change in State of Delivery
    atr_displacement: Optional[ATRDisplacement]  # ATR-based move classification
    momentum_fvg: Optional[MomentumFVG]  # Momentum-weighted FVG
    session_ranges: List[SessionRange]  # Session high/low tracking
    gap_mitigation: Optional[GapMitigation]  # Gap fill tracking
    
    # Confluence scoring
    confluence_score: int


class ICTSMCStrategy(BaseStrategy):
    """Full ICT Smart Money Concepts strategy with all major setups.
    
    Entry conditions (can be customized via params):
    1. Must be in a killzone (London, NY AM, NY PM)
    2. Liquidity sweep detected (or synthetic sweep fallback)
    3. Market structure confirmed (BOS/MSS)
    4. Entry in POI (FVG, OB, breaker, or OTE zone)
    5. Sufficient displacement for momentum confirmation
    
    Special modes:
    - Silver Bullet: FVG entries during 10-11 AM NY window
    - AMD: Enter on distribution phase after manipulation
    - Institutional: Follow large-body candle momentum
    """

    name = "ict_smc"
    DEFAULT_PARAMS = {
        # Core settings
        "min_candles": 50,  # Need more candles for HTF analysis
        "rr_target": 3.0,  # Higher R:R for quality setups
        "atr_period": 14,
        
        # Session filtering
        "sessions": ("London", "NewYork"),
        "use_killzones": True,
        "killzone_filter": ("London_Open", "NY_AM", "NY_Lunch", "NY_PM", "Silver_Bullet_NY"),
        
        # Displacement & momentum
        "displacement_lookback": 4,
        "min_displacement": 0.25,  # Slightly relaxed for more entries
        "require_institutional_candle": False,
        "institutional_body_ratio": 0.7,
        "institutional_volume_mult": 1.3,
        
        # Liquidity sweep
        "sweep_lookback": 25,  # Increased lookback for better levels
        "sweep_reentry_buffer_mult": 0.25,
        "sweep_reentry_bars": 5,
        "allow_synthetic_sweeps": True,
        
        # Market structure
        "bias_mode": "auto",  # "auto", "bos_only", "mss_only", "none"
        "require_bias_alignment": True,  # KEY: Require trend alignment
        
        # HTF (Higher Time Frame) Trend Filter
        "use_htf_filter": True,
        "htf_ema_fast": 20,  # 20-period EMA
        "htf_ema_slow": 50,  # 50-period EMA
        "htf_trend_bars": 100,  # Look back 100 bars for trend
        
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
        "ote_bonus_confluence": 2,
        
        # AMD Pattern
        "use_amd": True,
        "amd_accumulation_bars": 5,
        "amd_manipulation_threshold": 1.5,
        
        # Silver Bullet
        "silver_bullet_mode": True,
        "silver_bullet_bonus": 2,
        
        # Premium/Discount Zones
        "use_premium_discount": True,
        "require_optimal_zone": False,  # If True, only buy in discount, sell in premium
        "premium_discount_bonus": 2,  # Confluence bonus for optimal zone entry
        
        # Asian Range
        "use_asian_range": True,
        "asian_sweep_bonus": 2,  # Bonus when Asian high/low is swept
        
        # Power of 3 Pattern
        "use_power_of_3": True,
        "po3_bonus": 2,  # Bonus for confirmed PO3 pattern
        
        # Time-Based Entry Quality
        "use_time_quality": True,
        "min_time_quality": 0.5,  # Minimum entry quality (0.0-1.0)
        
        # Liquidity Pools
        "use_liquidity_pools": True,
        "liquidity_pool_bonus": 1,  # Bonus for targeting liquidity
        
        # NEW: CRT (Candle Range Theory) Pattern
        "use_crt": True,
        "crt_bonus": 3,  # Bonus for CRT pattern (high probability)
        "crt_tolerance": 0.002,  # Price tolerance for HTF level proximity
        
        # NEW: CISD (Change in State of Delivery)
        "use_cisd": True,
        "cisd_bonus": 3,  # Bonus for confirmed CISD
        "cisd_min_tolerance_ratio": 0.5,  # Minimum conviction for CISD
        
        # NEW: ATR Displacement Classification
        "use_atr_displacement": True,
        "reject_judas_swing": False,  # If True, skip trades during Judas moves
        "displacement_bonus": 2,  # Bonus for real displacement (2+ ATR)
        
        # NEW: Momentum-Weighted FVGs
        "use_momentum_fvg": True,
        "momentum_fvg_bonus": 2,  # Bonus for strong momentum FVG
        "momentum_rsi_period": 14,
        "momentum_strong_threshold": 0.7,  # RSI 70+ or 30-
        
        # NEW: Session Range Tracking
        "use_session_ranges": True,
        "session_sweep_bonus": 2,  # Bonus when prior session range is swept
        
        # NEW: Gap Mitigation
        "use_gap_mitigation": True,
        "gap_mitigation_bonus": 1,  # Bonus when gap is partially mitigated
        
        # NEW: Tiered Take Profit System
        # Partial exits at multiple R:R levels for better risk management
        "use_tiered_tp": True,
        "tp1_rr": 1.0,      # First TP at 1:1 R:R
        "tp1_percent": 0.33,  # Close 33% of position
        "tp2_rr": 2.0,      # Second TP at 2:1 R:R
        "tp2_percent": 0.33,  # Close another 33%
        "tp3_rr": 3.0,      # Third TP at full target
        "tp3_percent": 0.34,  # Close remaining 34%
        
        # Risk management
        "atr_buffer_mult": 0.25,  # Slightly wider stops
        "min_confluence": 3,  # Lower quality filter for more signals (was 4)
        
        # Volatility filter
        "use_volatility_filter": False,  # DISABLED - was blocking too many trades
        "min_atr_percentile": 10,  # Skip when volatility too low (was 20)
        "max_atr_percentile": 95,  # Skip when volatility extreme
        "atr_lookback": 50,  # Periods to calculate ATR percentile
        
        # Session-specific R:R targets (London tends to have better follow-through)
        "london_rr_bonus": 0.5,  # Add 0.5R to target in London
        "ny_rr_bonus": 0.0,
        
        # Trade quality requirements
        "min_grade": "B",  # Minimum grade to take trade (A+, A, B, C)
        
        # Debug & demo
        "demo_mode": False,
        "demo_stride": 5,
        "debug": False,
    }

    def __init__(self, context) -> None:
        super().__init__(context)
        self._candles: Deque[Candle] = deque(maxlen=500)
        self.params = {**self.DEFAULT_PARAMS, **(self.context.parameters or {})}

    def on_candle(self, candle: Candle) -> Optional[StrategySignal]:
        # Handle duplicate/update of forming candle (same timestamp, updated OHLC)
        if self._candles and self._candles[-1].timestamp == candle.timestamp:
            # Update the last candle with new values (forming candle updated)
            self._candles[-1] = candle
        else:
            # New candle, append it
            self._candles.append(candle)
        
        min_candles = int(self.params["min_candles"])
        
        if len(self._candles) < min_candles:
            self._debug(f"warming up: {len(self._candles)}/{min_candles}")
            return self._demo_signal(candle)

        # Session filtering
        session_name = active_session(candle.timestamp, self.context.session_windows)
        session_whitelist = set(self.params.get("sessions", self.DEFAULT_PARAMS["sessions"]))
        if session_name not in session_whitelist:
            self._debug(f"session filtered: {session_name}")
            return self._demo_signal(candle)

        # Volatility filter
        if self.params.get("use_volatility_filter"):
            if not self._check_volatility():
                self._debug("volatility filter rejected")
                return self._demo_signal(candle)

        # Build comprehensive market context
        market_ctx = self._build_market_context(session_name, candle)
        if market_ctx is None:
            return self._demo_signal(candle)

        # Confluence check
        min_confluence = int(self.params.get("min_confluence", 4))
        if market_ctx.confluence_score < min_confluence:
            self._debug(f"insufficient confluence: {market_ctx.confluence_score}/{min_confluence}")
            return self._demo_signal(candle)

        # Trade grading
        grade = self._grade_trade(market_ctx)
        min_grade = str(self.params.get("min_grade", "B"))
        if not self._meets_min_grade(grade, min_grade):
            self._debug(f"trade grade {grade} below minimum {min_grade}")
            return self._demo_signal(candle)

        # Calculate entry, stop, and target
        entry = self._determine_entry(candle, market_ctx)
        if entry is None:
            self._debug("entry calculation failed")
            return self._demo_signal(candle)

        stop_loss = self._stop_loss(candle, market_ctx)
        if stop_loss is None:
            self._debug("stop calculation failed")
            return self._demo_signal(candle)

        risk = abs(entry - stop_loss)
        if risk <= 0:
            return self._demo_signal(candle)

        # Scale R:R based on confluence and session
        base_rr = float(self.params["rr_target"])
        
        # Session-specific R:R bonus
        if session_name == "London":
            base_rr += float(self.params.get("london_rr_bonus", 0.5))
        elif session_name == "NewYork":
            base_rr += float(self.params.get("ny_rr_bonus", 0.0))
        
        # Confluence scaling
        if market_ctx.confluence_score >= 8:
            rr = base_rr * 1.3  # A+ setup - go for more
        elif market_ctx.confluence_score >= 6:
            rr = base_rr * 1.15  # A setup
        elif market_ctx.confluence_score >= 4:
            rr = base_rr  # B setup
        else:
            rr = base_rr * 0.8  # C setup (if allowed)
        
        if market_ctx.direction is OrderSide.BUY:
            take_profit = entry + rr * risk
        else:
            take_profit = entry - rr * risk

        # Validate TP
        if (market_ctx.direction is OrderSide.BUY and take_profit <= entry) or (
            market_ctx.direction is OrderSide.SELL and take_profit >= entry
        ):
            self._debug("invalid take profit")
            return self._demo_signal(candle)

        # Build tiered TP levels if enabled
        partial_tps = []
        if self.params.get("use_tiered_tp"):
            partial_tps = self._calculate_tiered_tps(entry, stop_loss, market_ctx.direction)
            if partial_tps:
                self._debug(f"tiered TPs: {len(partial_tps)} levels")

        # Build signal with confluence as confidence
        confidence = min(market_ctx.confluence_score / 10.0, 1.0)
        
        signal = StrategySignal(
            instrument=self.context.instrument,
            side=market_ctx.direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=self._candles[-1].timestamp,
            confidence=confidence,
            session=market_ctx.session,
            partial_tps=partial_tps,
        )
        
        self._debug(
            "SIGNAL | %s | entry=%.5f stop=%.5f tp=%.5f | confluence=%d | %s",
            market_ctx.direction.value.upper(),
            entry,
            stop_loss,
            take_profit,
            market_ctx.confluence_score,
            self._describe_setup(market_ctx),
        )
        
        return signal
    
    def _calculate_tiered_tps(
        self,
        entry: float,
        stop_loss: float,
        direction: OrderSide,
    ) -> List[TakeProfitLevel]:
        """Calculate tiered take profit levels.
        
        Creates multiple TP levels at different R:R ratios:
        - TP1: 1:1 R:R - Close 33% of position (secure profit)
        - TP2: 2:1 R:R - Close 33% of position (let winners run)
        - TP3: Full target R:R - Close remaining 34%
        
        This approach:
        - Locks in profit early
        - Reduces risk on remaining position
        - Allows participation in larger moves
        """
        risk = abs(entry - stop_loss)
        if risk <= 0:
            return []
        
        tp1_rr = float(self.params.get("tp1_rr", 1.0))
        tp1_pct = float(self.params.get("tp1_percent", 0.33))
        tp2_rr = float(self.params.get("tp2_rr", 2.0))
        tp2_pct = float(self.params.get("tp2_percent", 0.33))
        tp3_rr = float(self.params.get("tp3_rr", 3.0))
        tp3_pct = float(self.params.get("tp3_percent", 0.34))
        
        tps = []
        
        if direction is OrderSide.BUY:
            tp1_price = entry + (tp1_rr * risk)
            tp2_price = entry + (tp2_rr * risk)
            tp3_price = entry + (tp3_rr * risk)
        else:
            tp1_price = entry - (tp1_rr * risk)
            tp2_price = entry - (tp2_rr * risk)
            tp3_price = entry - (tp3_rr * risk)
        
        # Validate prices
        if tp1_price > 0:
            tps.append(TakeProfitLevel(price=tp1_price, percent=tp1_pct, rr_ratio=tp1_rr))
        if tp2_price > 0:
            tps.append(TakeProfitLevel(price=tp2_price, percent=tp2_pct, rr_ratio=tp2_rr))
        if tp3_price > 0:
            tps.append(TakeProfitLevel(price=tp3_price, percent=tp3_pct, rr_ratio=tp3_rr))
        
        return tps

    def _build_market_context(self, session_name: str, candle: Candle) -> Optional[MarketContext]:
        """Build comprehensive market context with all ICT concepts."""
        candles = list(self._candles)
        confluence_score = 0
        
        # Calculate ATR
        atr = calculate_atr(candles, period=int(self.params["atr_period"]))
        
        # Time-based entry quality
        entry_quality = get_entry_quality(candle.timestamp) if self.params.get("use_time_quality") else 1.0
        min_time_quality = float(self.params.get("min_time_quality", 0.5))
        if self.params.get("use_time_quality") and entry_quality < min_time_quality:
            self._debug(f"time quality too low: {entry_quality:.2f} < {min_time_quality}")
            # Don't reject, just don't add bonus
        elif entry_quality >= 0.9:
            confluence_score += 1  # Bonus for optimal time
            self._debug(f"optimal entry time: {entry_quality:.2f}")
        
        # Killzone check
        killzone = active_killzone(candle.timestamp) if self.params.get("use_killzones") else None
        allowed_killzones = set(self.params.get("killzone_filter", ()))
        if killzone and allowed_killzones and killzone.name in allowed_killzones:
            confluence_score += 1
            self._debug(f"killzone: {killzone.name}")
        
        # Silver Bullet check
        is_silver_bullet = is_silver_bullet_window(candle.timestamp)
        if is_silver_bullet and self.params.get("silver_bullet_mode"):
            confluence_score += int(self.params.get("silver_bullet_bonus", 2))
            self._debug("silver bullet window active")
        
        # Asian Range analysis
        asian_range = None
        if self.params.get("use_asian_range"):
            asian_range = calculate_asian_range(candles)
            if asian_range:
                high_swept, low_swept = asian_range_swept(candles[-20:], asian_range)
                if high_swept or low_swept:
                    confluence_score += int(self.params.get("asian_sweep_bonus", 2))
                    self._debug(f"Asian range swept: high={high_swept} low={low_swept}")
        
        # Liquidity sweep detection
        buffer_mult = float(self.params.get("sweep_reentry_buffer_mult", 0.3))
        sweep_buffer = atr * max(buffer_mult, 0.0)
        reentry_bars = max(int(self.params.get("sweep_reentry_bars", 5)), 1)
        
        sweep = detect_liquidity_sweep(
            candles,
            lookback=int(self.params["sweep_lookback"]),
            reentry_buffer=sweep_buffer,
            reentry_bars=reentry_bars,
        )
        
        if sweep:
            confluence_score += 2
            self._debug(f"liquidity sweep: {sweep.direction} at {sweep.swept_level:.5f}")
        elif self.params.get("allow_synthetic_sweeps"):
            sweep = self._synthetic_sweep(candles)
            if sweep:
                confluence_score += 1
                self._debug(f"synthetic sweep: {sweep.direction}")
        
        if not sweep:
            self._debug("no sweep detected")
            return None
        
        # Market structure analysis
        swings = recent_structure(swing_points(candles, depth=3), max_age_minutes=300)
        direction = OrderSide.BUY if sweep.direction == "buy" else OrderSide.SELL
        
        # Premium/Discount zone analysis
        premium_discount = None
        if self.params.get("use_premium_discount"):
            premium_discount = calculate_premium_discount(swings)
            if premium_discount:
                zone = price_zone(candle.close, premium_discount)
                optimal_zone = (
                    (zone == "discount" and direction is OrderSide.BUY) or
                    (zone == "premium" and direction is OrderSide.SELL)
                )
                if optimal_zone:
                    confluence_score += int(self.params.get("premium_discount_bonus", 2))
                    self._debug(f"optimal zone: {zone} for {direction.value}")
                elif self.params.get("require_optimal_zone"):
                    self._debug(f"wrong zone: {zone} for {direction.value}")
                    return None
        
        # Power of 3 pattern detection
        power_of_3 = None
        if self.params.get("use_power_of_3"):
            power_of_3 = detect_power_of_3(candles)
            if power_of_3 and power_of_3.manipulation_complete:
                if (power_of_3.predicted_direction == "bullish" and direction is OrderSide.BUY) or \
                   (power_of_3.predicted_direction == "bearish" and direction is OrderSide.SELL):
                    confluence_score += int(self.params.get("po3_bonus", 2))
                    self._debug(f"PO3 aligned: {power_of_3.predicted_direction}")
        
        # Liquidity pool targeting
        liquidity_pools = []
        if self.params.get("use_liquidity_pools"):
            liquidity_pools = identify_liquidity_pools(swings)
            target_pool = nearest_liquidity_pool(candle.close, liquidity_pools, direction.value.lower())
            if target_pool:
                confluence_score += int(self.params.get("liquidity_pool_bonus", 1))
                self._debug(f"targeting liquidity at {target_pool.level:.5f} (strength {target_pool.strength})")
        
        # HTF Trend Filter - check if trade aligns with higher timeframe trend
        if self.params.get("use_htf_filter"):
            htf_trend = self._get_htf_trend(candles)
            if htf_trend:
                if htf_trend == direction.value.lower():
                    confluence_score += 3  # Strong bonus for HTF alignment
                    self._debug(f"HTF trend aligned: {htf_trend}")
                else:
                    self._debug(f"HTF trend opposing: {htf_trend} vs {direction.value.lower()}")
                    # Still allow trade but no bonus - could reject if require_htf_alignment
                    if self.params.get("require_htf_alignment"):
                        return None
        
        bias = self._directional_bias(swings)
        if bias:
            if bias == sweep.direction:
                confluence_score += 2
                self._debug(f"bias aligned: {bias}")
            elif self.params.get("require_bias_alignment"):
                self._debug("bias misaligned, rejected")
                return None
        
        # Displacement check
        displacement = calculate_displacement(candles, lookback=int(self.params["displacement_lookback"]))
        min_disp = float(self.params["min_displacement"])
        if displacement >= min_disp:
            confluence_score += 1
            self._debug(f"displacement: {displacement:.2f}")
        elif displacement < min_disp * 0.5:
            self._debug(f"displacement too low: {displacement:.2f}")
            return None
        
        # Order block detection
        blocks = detect_order_blocks(swings, candles)
        block = self._select_block(blocks, direction)
        if block:
            confluence_score += 1
            self._debug(f"order block: {block.direction} at {block.low:.5f}-{block.high:.5f}")
        
        # Breaker block detection
        breaker = None
        if self.params.get("use_breaker_blocks"):
            breakers = detect_breaker_blocks(candles, blocks)
            breaker = self._select_breaker(breakers, direction)
            if breaker:
                confluence_score += 2  # Breakers are powerful
                self._debug(f"breaker block: {breaker.direction}")
        
        # Mitigation block detection
        mitigation = None
        if self.params.get("use_mitigation_blocks"):
            mitigations = detect_mitigation_blocks(candles, blocks)
            mitigation = self._select_mitigation(mitigations, direction)
            if mitigation:
                confluence_score += 1
                self._debug(f"mitigation block: tested {mitigation.times_tested}x")
        
        # Fair value gap detection
        all_gaps = detect_fair_value_gaps(candles)
        unfilled_gaps = detect_unfilled_fvg(candles, all_gaps) if self.params.get("prefer_unfilled_fvg") else all_gaps
        gap_tolerance = atr * float(self.params["gap_tolerance_mult"])
        gap = self._select_gap(unfilled_gaps, direction, candle.close, gap_tolerance)
        
        if gap:
            confluence_score += 2
            ce_level = fvg_consequent_encroachment(gap)
            self._debug(f"FVG: {gap.direction} CE={ce_level:.5f}")
        
        # Validate required POIs
        if self.params.get("require_gap") and not gap:
            self._debug("gap required but missing")
            return None
        if self.params.get("require_block") and not block and not breaker:
            self._debug("block required but missing")
            return None
        
        # OTE calculation
        ote = None
        if self.params.get("use_ote"):
            ote = find_ote_from_swings(swings, direction.value.lower())
            if ote and price_in_ote(candle.close, ote):
                confluence_score += int(self.params.get("ote_bonus_confluence", 2))
                self._debug(f"price in OTE zone: {ote.ote_lower:.5f}-{ote.ote_upper:.5f}")
            elif self.params.get("require_ote"):
                self._debug("OTE required but price not in zone")
                return None
        
        # AMD pattern detection
        amd = None
        if self.params.get("use_amd"):
            amd = detect_amd_pattern(
                candles,
                min_accumulation_bars=int(self.params.get("amd_accumulation_bars", 5)),
                manipulation_threshold=float(self.params.get("amd_manipulation_threshold", 1.5)),
            )
            if amd and amd.direction == direction.value.lower():
                confluence_score += 2
                self._debug(f"AMD pattern: {amd.phase} -> {amd.direction}")
        
        # Institutional candle detection
        inst_candles = []
        if self.params.get("require_institutional_candle") or True:
            inst_candles = detect_institutional_candles(
                candles,
                min_body_ratio=float(self.params.get("institutional_body_ratio", 0.7)),
                volume_mult=float(self.params.get("institutional_volume_mult", 1.3)),
            )
            if inst_candles:
                # Check if last institutional candle aligns with direction
                last_inst = inst_candles[-1]
                inst_dir = "buy" if last_inst.candle.close > last_inst.candle.open else "sell"
                if inst_dir == direction.value.lower():
                    confluence_score += 1
                    self._debug(f"institutional candle aligned: {inst_dir}")
            elif self.params.get("require_institutional_candle"):
                self._debug("institutional candle required but missing")
                return None
        
        # NEW: CRT (Candle Range Theory) Pattern Detection
        crt_pattern = None
        if self.params.get("use_crt"):
            # Use swing levels as HTF levels for CRT detection
            htf_levels = [s.price for s in swings] if swings else None
            crt_pattern = detect_crt_pattern(
                candles,
                htf_levels=htf_levels,
                tolerance_mult=float(self.params.get("crt_tolerance", 0.002)),
            )
            if crt_pattern:
                crt_dir = "bullish" if direction is OrderSide.BUY else "bearish"
                if crt_pattern.direction == crt_dir:
                    confluence_score += int(self.params.get("crt_bonus", 3))
                    self._debug(f"CRT pattern detected: {crt_pattern.direction} at {crt_pattern.swept_level:.5f}")
        
        # NEW: CISD (Change in State of Delivery) Detection
        cisd_signal = None
        if self.params.get("use_cisd"):
            cisd_signal = detect_cisd(
                candles,
                swings,
                min_tolerance_ratio=float(self.params.get("cisd_min_tolerance_ratio", 0.5)),
            )
            if cisd_signal:
                cisd_dir = "bullish" if direction is OrderSide.BUY else "bearish"
                if cisd_signal.new_bias == cisd_dir:
                    confluence_score += int(self.params.get("cisd_bonus", 3))
                    self._debug(f"CISD confirmed: {cisd_signal.old_bias} -> {cisd_signal.new_bias} (ratio={cisd_signal.tolerance_ratio:.2f})")
        
        # NEW: ATR Displacement Classification
        atr_displacement = None
        if self.params.get("use_atr_displacement"):
            atr_displacement = classify_atr_displacement(
                candles,
                atr_period=int(self.params.get("atr_period", 14)),
            )
            if atr_displacement:
                if atr_displacement.is_fake_move and self.params.get("reject_judas_swing"):
                    self._debug(f"Judas swing detected (zone={atr_displacement.zone}), rejecting")
                    return None
                elif atr_displacement.is_real_displacement:
                    confluence_score += int(self.params.get("displacement_bonus", 2))
                    self._debug(f"Real displacement: {atr_displacement.move_atr_multiple:.2f} ATR")
                elif atr_displacement.is_fake_move:
                    self._debug(f"Judas swing zone: {atr_displacement.move_atr_multiple:.2f} ATR (caution)")
        
        # NEW: Momentum-Weighted FVG Detection
        momentum_fvg = None
        if self.params.get("use_momentum_fvg"):
            momentum_gaps = detect_momentum_fvgs(
                candles,
                rsi_period=int(self.params.get("momentum_rsi_period", 14)),
                strong_threshold=float(self.params.get("momentum_strong_threshold", 0.7)),
            )
            desired_dir = "bullish" if direction is OrderSide.BUY else "bearish"
            momentum_fvg = get_strongest_fvg(
                momentum_gaps,
                direction=desired_dir,
                price=candle.close,
                tolerance=gap_tolerance,
            )
            if momentum_fvg and momentum_fvg.likely_to_hold:
                confluence_score += int(self.params.get("momentum_fvg_bonus", 2))
                self._debug(f"Strong momentum FVG: strength={momentum_fvg.momentum_strength:.2f}, RSI={momentum_fvg.rsi_at_formation:.1f}")
        
        # NEW: Session Range Tracking
        session_ranges = []
        if self.params.get("use_session_ranges"):
            session_ranges = get_all_session_ranges(candles)
            for sr in session_ranges:
                if not sr.is_active:  # Only check completed sessions
                    high_swept, low_swept = check_session_sweep(candles, sr)
                    if (high_swept and direction is OrderSide.SELL) or (low_swept and direction is OrderSide.BUY):
                        confluence_score += int(self.params.get("session_sweep_bonus", 2))
                        self._debug(f"Session range swept: {sr.session_name} (high={high_swept}, low={low_swept})")
                        break
        
        # NEW: Gap Mitigation Detection
        gap_mitigation = None
        if self.params.get("use_gap_mitigation"):
            gap_mitigation = detect_gap_mitigation(candles)
            if gap_mitigation and gap_mitigation.half_mitigated and not gap_mitigation.fully_mitigated:
                # Gap partially filled - good for continuation
                gap_dir = "bullish" if gap_mitigation.direction == "up" else "bearish"
                trade_dir = "bullish" if direction is OrderSide.BUY else "bearish"
                if gap_dir == trade_dir:
                    confluence_score += int(self.params.get("gap_mitigation_bonus", 1))
                    self._debug(f"Gap mitigation: {gap_mitigation.direction} gap 50% filled")
        
        return MarketContext(
            direction=direction,
            sweep=sweep,
            displacement=displacement,
            atr=atr,
            session=session_name,
            killzone=killzone,
            gap=gap,
            unfilled_gaps=unfilled_gaps,
            block=block,
            breaker=breaker,
            mitigation=mitigation,
            ote=ote,
            amd=amd,
            institutional_candles=inst_candles,
            is_silver_bullet=is_silver_bullet,
            asian_range=asian_range,
            premium_discount=premium_discount,
            power_of_3=power_of_3,
            liquidity_pools=liquidity_pools,
            entry_quality=entry_quality,
            # NEW: Advanced ICT concepts
            crt_pattern=crt_pattern,
            cisd_signal=cisd_signal,
            atr_displacement=atr_displacement,
            momentum_fvg=momentum_fvg,
            session_ranges=session_ranges,
            gap_mitigation=gap_mitigation,
            confluence_score=confluence_score,
        )

    def _directional_bias(self, swings: Iterable[SwingPoint]) -> Optional[str]:
        """Determine directional bias from market structure."""
        swings_list = list(swings)
        if len(swings_list) < 4:
            return None
        
        mode = str(self.params.get("bias_mode", "auto")).lower()
        if mode == "none":
            return None
        
        if mode in {"auto", "bos_only"}:
            bos = detect_bos(swings_list)
            if bos:
                return "buy" if bos.kind == "high" else "sell"
            if mode == "bos_only":
                return None
        
        if mode in {"auto", "mss_only"}:
            mss = detect_mss(swings_list)
            if mss:
                return "sell" if mss.kind == "high" else "buy"
        
        return None

    def _get_htf_trend(self, candles: List[Candle]) -> Optional[str]:
        """Determine higher timeframe trend using EMA crossover.
        
        Uses fast EMA vs slow EMA to determine trend:
        - If fast EMA > slow EMA: bullish trend (buy)
        - If fast EMA < slow EMA: bearish trend (sell)
        
        Also checks if price is above/below EMAs for confirmation.
        """
        fast_period = int(self.params.get("htf_ema_fast", 20))
        slow_period = int(self.params.get("htf_ema_slow", 50))
        lookback = int(self.params.get("htf_trend_bars", 100))
        
        if len(candles) < slow_period + 5:
            return None
        
        # Use recent candles for trend analysis
        analysis_candles = candles[-lookback:] if len(candles) >= lookback else candles
        closes = [c.close for c in analysis_candles]
        
        # Calculate EMAs
        fast_ema = self._calculate_ema(closes, fast_period)
        slow_ema = self._calculate_ema(closes, slow_period)
        
        if fast_ema is None or slow_ema is None:
            return None
        
        current_price = closes[-1]
        
        # Strong trend confirmation:
        # 1. Fast EMA above/below slow EMA
        # 2. Price above/below fast EMA (extra confirmation)
        if fast_ema > slow_ema:
            if current_price > fast_ema:
                return "buy"  # Strong uptrend
            elif current_price > slow_ema:
                return "buy"  # Moderate uptrend
        elif fast_ema < slow_ema:
            if current_price < fast_ema:
                return "sell"  # Strong downtrend
            elif current_price < slow_ema:
                return "sell"  # Moderate downtrend
        
        return None  # No clear trend

    @staticmethod
    def _calculate_ema(data: List[float], period: int) -> Optional[float]:
        """Calculate Exponential Moving Average."""
        if len(data) < period:
            return None
        
        multiplier = 2 / (period + 1)
        ema = sum(data[:period]) / period  # SMA for initial EMA
        
        for price in data[period:]:
            ema = (price - ema) * multiplier + ema
        
        return ema

    @staticmethod
    def _synthetic_sweep(candles: Iterable[Candle]) -> Optional[LiquiditySweep]:
        """Create synthetic sweep from recent price action."""
        candles_list = list(candles)
        if len(candles_list) < 5:
            return None
        
        last = candles_list[-1]
        recent = candles_list[-8:-1]
        
        if not recent:
            return None
        
        direction = "buy" if last.close > last.open else "sell"
        
        if direction == "buy":
            level = min(c.low for c in recent)
            if last.low <= level * 1.001:  # Slight tolerance
                return LiquiditySweep(direction="buy", swept_level=level, timestamp=last.timestamp)
        else:
            level = max(c.high for c in recent)
            if last.high >= level * 0.999:
                return LiquiditySweep(direction="sell", swept_level=level, timestamp=last.timestamp)
        
        return None

    @staticmethod
    def _select_gap(
        gaps: Iterable[FairValueGap],
        direction: OrderSide,
        price: float,
        tolerance: float,
    ) -> Optional[FairValueGap]:
        """Select the most relevant FVG for the trade direction."""
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        best_gap = None
        best_distance = float('inf')
        
        for gap in gaps:
            if gap.direction != desired:
                continue
            
            # Prefer gaps that price is inside or close to
            if gap.contains(price):
                return gap
            
            # Check proximity
            distance = min(abs(price - gap.upper), abs(price - gap.lower))
            if distance <= tolerance and distance < best_distance:
                best_gap = gap
                best_distance = distance
        
        return best_gap

    @staticmethod
    def _select_block(blocks: Iterable[OrderBlock], direction: OrderSide) -> Optional[OrderBlock]:
        """Select the most recent order block for the trade direction."""
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        for block in reversed(list(blocks)):
            if block.direction == desired:
                return block
        return None

    @staticmethod
    def _select_breaker(breakers: Iterable[BreakerBlock], direction: OrderSide) -> Optional[BreakerBlock]:
        """Select the most recent breaker block for the trade direction."""
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        for breaker in reversed(list(breakers)):
            if breaker.direction == desired:
                return breaker
        return None

    @staticmethod
    def _select_mitigation(mitigations: Iterable[MitigationBlock], direction: OrderSide) -> Optional[MitigationBlock]:
        """Select the best mitigation block (prefer less tested ones)."""
        desired = "bullish" if direction is OrderSide.BUY else "bearish"
        candidates = [m for m in mitigations if m.direction == desired]
        if not candidates:
            return None
        # Prefer blocks tested fewer times (fresher)
        return min(candidates, key=lambda m: m.times_tested)

    def _determine_entry(self, candle: Candle, ctx: MarketContext) -> Optional[float]:
        """Determine optimal entry price based on available POIs."""
        entry = candle.close
        
        # Priority 1: OTE zone entry
        if ctx.ote and price_in_ote(candle.close, ctx.ote):
            midpoint = (ctx.ote.ote_upper + ctx.ote.ote_lower) / 2
            if ctx.direction is OrderSide.BUY:
                entry = min(entry, midpoint)
            else:
                entry = max(entry, midpoint)
        
        # Priority 2: FVG consequent encroachment
        if ctx.gap:
            ce = fvg_consequent_encroachment(ctx.gap)
            if ctx.direction is OrderSide.BUY:
                entry = max(entry, ce)
            else:
                entry = min(entry, ce)
        
        # Priority 3: Breaker block
        elif ctx.breaker:
            midpoint = (ctx.breaker.high + ctx.breaker.low) / 2
            if ctx.direction is OrderSide.BUY:
                entry = max(entry, midpoint)
            else:
                entry = min(entry, midpoint)
        
        # Priority 4: Order block
        elif ctx.block:
            midpoint = (ctx.block.high + ctx.block.low) / 2
            if ctx.direction is OrderSide.BUY:
                entry = max(entry, midpoint)
            else:
                entry = min(entry, midpoint)
        
        # Priority 5: Mitigation block
        elif ctx.mitigation:
            entry = ctx.mitigation.mitigation_level
        
        return entry if entry > 0 else None

    def _stop_loss(self, candle: Candle, ctx: MarketContext) -> Optional[float]:
        """Calculate stop loss with ATR buffer beyond key level."""
        buffer = ctx.atr * float(self.params.get("atr_buffer_mult", 0.2))
        
        if ctx.direction is OrderSide.BUY:
            # Stop below the lowest protective level
            reference = candle.low
            if ctx.sweep:
                reference = min(reference, ctx.sweep.swept_level)
            if ctx.block:
                reference = min(reference, ctx.block.low)
            if ctx.breaker:
                reference = min(reference, ctx.breaker.low)
            if ctx.ote:
                reference = min(reference, ctx.ote.ote_lower)
            stop = reference - buffer
        else:
            # Stop above the highest protective level
            reference = candle.high
            if ctx.sweep:
                reference = max(reference, ctx.sweep.swept_level)
            if ctx.block:
                reference = max(reference, ctx.block.high)
            if ctx.breaker:
                reference = max(reference, ctx.breaker.high)
            if ctx.ote:
                reference = max(reference, ctx.ote.ote_upper)
            stop = reference + buffer
        
        return stop if stop > 0 else None

    def _check_volatility(self) -> bool:
        """Check if current volatility is within acceptable range.
        
        Returns True if volatility is acceptable, False otherwise.
        Avoids trading when volatility is too low (choppy) or too high (risky).
        """
        candles = list(self._candles)
        lookback = int(self.params.get("atr_lookback", 50))
        
        if len(candles) < lookback:
            return True  # Not enough data, allow trade
        
        # Calculate ATR for each period in lookback
        atr_period = int(self.params.get("atr_period", 14))
        atrs = []
        for i in range(atr_period, len(candles)):
            window = candles[i - atr_period:i]
            if len(window) >= atr_period:
                tr_sum = sum(max(c.high - c.low, abs(c.high - window[j-1].close if j > 0 else c.high - c.open), 
                                 abs(c.low - window[j-1].close if j > 0 else c.low - c.open))
                            for j, c in enumerate(window[-atr_period:]))
                atrs.append(tr_sum / atr_period)
        
        if len(atrs) < 10:
            return True
        
        current_atr = atrs[-1]
        sorted_atrs = sorted(atrs)
        
        # Calculate percentile of current ATR
        percentile = (sorted_atrs.index(min(sorted_atrs, key=lambda x: abs(x - current_atr))) / len(sorted_atrs)) * 100
        
        min_percentile = float(self.params.get("min_atr_percentile", 20))
        max_percentile = float(self.params.get("max_atr_percentile", 95))
        
        if percentile < min_percentile:
            self._debug(f"volatility too low: {percentile:.1f}th percentile")
            return False
        if percentile > max_percentile:
            self._debug(f"volatility too high: {percentile:.1f}th percentile")
            return False
        
        return True

    def _grade_trade(self, ctx: MarketContext) -> str:
        """Grade the trade quality based on confluence and setup quality.
        
        A+ (10+): All major factors aligned - sweep + breaker + OTE + killzone + trend
        A (7-9): Strong setup with most factors
        B (4-6): Decent setup with core factors
        C (1-3): Marginal setup, low probability
        """
        score = ctx.confluence_score
        
        # Bonus for key combinations
        bonus = 0
        if ctx.sweep and ctx.breaker and ctx.ote:
            bonus += 2  # Triple confluence
        if ctx.is_silver_bullet and ctx.gap:
            bonus += 1  # Silver bullet with FVG
        if ctx.amd and ctx.killzone:
            bonus += 1  # AMD in killzone
        # Enhanced ICT concept bonuses
        if ctx.power_of_3 and ctx.power_of_3.manipulation_complete:
            bonus += 1  # Confirmed PO3 pattern
        if ctx.premium_discount and ctx.entry_quality >= 0.9:
            bonus += 1  # Optimal zone + optimal time
        if ctx.asian_range and ctx.sweep:
            # Check if sweep was at Asian level
            if abs(ctx.sweep.swept_level - ctx.asian_range.high) / ctx.atr < 0.5 or \
               abs(ctx.sweep.swept_level - ctx.asian_range.low) / ctx.atr < 0.5:
                bonus += 1  # Asian range sweep
        
        # NEW: Advanced ICT concept bonuses
        if ctx.crt_pattern:
            bonus += 2  # CRT is a high-probability reversal pattern
        if ctx.cisd_signal and ctx.cisd_signal.tolerance_ratio >= 0.7:
            bonus += 2  # Strong CISD confirmation
        if ctx.atr_displacement and ctx.atr_displacement.is_real_displacement:
            bonus += 1  # Real institutional displacement
        if ctx.momentum_fvg and ctx.momentum_fvg.likely_to_hold:
            bonus += 1  # Strong momentum FVG
        if ctx.session_ranges:
            # Check for session sweep alignment
            for sr in ctx.session_ranges:
                if not sr.is_active and (sr.high_swept or sr.low_swept):
                    bonus += 1
                    break
        
        adjusted_score = score + bonus
        
        if adjusted_score >= 10:
            return "A+"
        elif adjusted_score >= 7:
            return "A"
        elif adjusted_score >= 4:
            return "B"
        else:
            return "C"

    def _meets_min_grade(self, grade: str, min_grade: str) -> bool:
        """Check if trade grade meets minimum requirement."""
        grade_order = {"A+": 4, "A": 3, "B": 2, "C": 1}
        return grade_order.get(grade, 0) >= grade_order.get(min_grade, 2)

    def _describe_setup(self, ctx: MarketContext) -> str:
        """Create human-readable description of the setup."""
        parts = []
        if ctx.is_silver_bullet:
            parts.append("SB")
        if ctx.killzone:
            parts.append(ctx.killzone.name)
        if ctx.sweep:
            parts.append(f"sweep:{ctx.sweep.direction}")
        if ctx.amd:
            parts.append("AMD")
        if ctx.gap:
            parts.append("FVG")
        if ctx.breaker:
            parts.append("breaker")
        if ctx.block:
            parts.append("OB")
        if ctx.ote:
            parts.append("OTE")
        # Enhanced ICT concepts in description
        if ctx.power_of_3 and ctx.power_of_3.manipulation_complete:
            parts.append("PO3")
        if ctx.premium_discount:
            zone = price_zone(ctx.sweep.swept_level if ctx.sweep else 0, ctx.premium_discount)
            if zone in ("premium", "discount"):
                parts.append(zone.upper()[:4])  # PREM or DISC
        if ctx.asian_range:
            parts.append("ASIAN")
        if ctx.entry_quality >= 0.9:
            parts.append("OPT_TIME")
        # NEW: Advanced ICT concepts in description
        if ctx.crt_pattern:
            parts.append("CRT")
        if ctx.cisd_signal:
            parts.append("CISD")
        if ctx.atr_displacement:
            if ctx.atr_displacement.is_real_displacement:
                parts.append("DISP")
            elif ctx.atr_displacement.is_fake_move:
                parts.append("JUDAS")
        if ctx.momentum_fvg:
            if ctx.momentum_fvg.likely_to_hold:
                parts.append("MFVG+")
            else:
                parts.append("MFVG")
        if ctx.session_ranges:
            for sr in ctx.session_ranges:
                if not sr.is_active:
                    parts.append(f"S:{sr.session_name[:3]}")
                    break
        if ctx.gap_mitigation and ctx.gap_mitigation.half_mitigated:
            parts.append("GAP50")
        return "+".join(parts) if parts else "basic"

    def _debug(self, message: str, *args) -> None:
        """Print debug message if debug mode is enabled."""
        if bool(self.params.get("debug", False)):
            text = message % args if args else message
            print(f"[ICT-SMC] {text}")

    def _demo_signal(self, candle: Candle) -> Optional[StrategySignal]:
        """Generate demo signal for testing (when demo_mode is enabled)."""
        if not bool(self.params.get("demo_mode")):
            return None
        
        stride = max(int(self.params.get("demo_stride", 5)), 1)
        if len(self._candles) < stride or len(self._candles) % stride:
            return None
        
        direction = OrderSide.BUY if candle.close >= candle.open else OrderSide.SELL
        entry = candle.close
        buffer = max(candle.range, 1e-5)
        rr = float(self.params.get("rr_target", 2.0))
        
        if direction is OrderSide.BUY:
            stop_loss = entry - buffer
            take_profit = entry + rr * buffer
        else:
            stop_loss = entry + buffer
            take_profit = entry - rr * buffer
        
        if stop_loss <= 0 or take_profit <= 0:
            return None
        
        session_name = active_session(candle.timestamp, self.context.session_windows) or "Demo"
        
        return StrategySignal(
            instrument=self.context.instrument,
            side=direction,
            entry=entry,
            stop_loss=stop_loss,
            take_profit=take_profit,
            timestamp=candle.timestamp,
            confidence=1.0,
            session=session_name,
        )


# Register the strategy
StrategyRegistry.register(ICTSMCStrategy)

__all__ = ["ICTSMCStrategy"]
