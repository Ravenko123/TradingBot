"""Utility helpers shared across trading and backtesting flows."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from statistics import mean, pstdev
from typing import Iterable, Iterator, List, Optional, Sequence, Tuple

from config.settings import SessionWindow


@dataclass(slots=True)
class Candle:
    """Normalized OHLCV candle structure."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2


@dataclass(slots=True)
class SwingPoint:
    """Represents a swing high or low used for structure analysis."""

    index: int
    price: float
    timestamp: datetime
    kind: str  # "high" or "low"


@dataclass(slots=True)
class FairValueGap:
    """Simplified 3-candle imbalance."""

    start_index: int
    end_index: int
    upper: float
    lower: float
    direction: str  # "bullish" or "bearish"

    def contains(self, price: float) -> bool:
        return self.lower <= price <= self.upper


@dataclass(slots=True)
class OrderBlock:
    """Lightweight order-block representation derived from swings."""

    index: int
    high: float
    low: float
    direction: str
    origin_time: datetime


@dataclass(slots=True)
class LiquiditySweep:
    """Represents a simple buy/sell-side liquidity sweep."""

    direction: str  # "buy" or "sell"
    swept_level: float
    timestamp: datetime


@dataclass(slots=True)
class BreakerBlock:
    """Breaker block: a failed order block that flips polarity after being violated.
    
    When a bullish OB fails (price breaks below), the zone becomes a bearish breaker.
    When a bearish OB fails (price breaks above), the zone becomes a bullish breaker.
    Price often returns to test these zones before continuing in the new direction.
    """

    index: int
    high: float
    low: float
    direction: str  # "bullish" or "bearish" (the NEW direction after flip)
    origin_time: datetime
    original_ob_direction: str  # the original OB direction that failed


@dataclass(slots=True)
class MitigationBlock:
    """Mitigation block: an order block that has been partially filled/mitigated.
    
    When price returns to an OB and reacts (but doesn't fully break through),
    the OB is considered mitigated. The remaining unfilled portion often 
    provides strong support/resistance on subsequent tests.
    """

    index: int
    high: float
    low: float
    direction: str  # "bullish" or "bearish"
    origin_time: datetime
    mitigation_level: float  # the price level where mitigation occurred
    times_tested: int


@dataclass(slots=True)
class Killzone:
    """ICT Killzone time window for high-probability trading."""

    name: str
    start: time
    end: time
    timezone: str  # typically "America/New_York" for ICT concepts


@dataclass(slots=True)
class OTE:
    """Optimal Trade Entry zone (61.8% - 78.6% Fibonacci retracement)."""

    swing_high: float
    swing_low: float
    ote_upper: float  # 61.8% level
    ote_lower: float  # 78.6% level
    direction: str  # "bullish" or "bearish"


@dataclass(slots=True)
class InstitutionalCandle:
    """Large-body candle indicating institutional activity."""

    index: int
    candle: Candle
    body_ratio: float  # body size relative to range
    volume_ratio: float  # volume relative to average


@dataclass(slots=True)
class AMDPattern:
    """Accumulation-Manipulation-Distribution pattern."""

    phase: str  # "accumulation", "manipulation", "distribution"
    start_index: int
    end_index: int
    range_high: float
    range_low: float
    direction: str  # expected move direction after manipulation


def load_candles(path: str) -> List[Candle]:
    """Load OHLCV data from CSV."""

    candles: List[Candle] = []
    with open(path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            candles.append(
                Candle(
                    timestamp=datetime.fromisoformat(str(row["timestamp"])),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )
            )
    return candles


def active_session(ts: datetime, sessions: Iterable[SessionWindow]) -> Optional[str]:
    """Return the configured session name covering the supplied timestamp."""

    current = ts.time()
    for session in sessions:
        if _within_window(current, session.start, session.end):
            return session.name
    return None


def _within_window(current: time, start: time, end: time) -> bool:
    if start <= end:
        return start <= current <= end
    return current >= start or current <= end


def floor_to_minute(ts: datetime) -> datetime:
    return ts.replace(second=0, microsecond=0)


def rolling_atr(candles: Sequence[Candle], period: int = 14) -> float:
    if len(candles) < period + 1:
        raise ValueError("Not enough candles for ATR")
    tr_values: List[float] = []
    for idx in range(1, period + 1):
        cur = candles[-idx]
        prev = candles[-idx - 1]
        tr_values.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
    return sum(tr_values) / len(tr_values)


def displacement_score(candles: Sequence[Candle], lookback: int = 5) -> float:
    if len(candles) < lookback:
        return 0.0
    window = candles[-lookback:]
    avg_range = mean(c.range for c in window)
    last = candles[-1]
    body = abs(last.close - last.open)
    return body / avg_range if avg_range else 0.0


def swing_points(candles: Sequence[Candle], depth: int = 3) -> List[SwingPoint]:
    swings: List[SwingPoint] = []
    if len(candles) < depth * 2 + 1:
        return swings  # Not enough candles for swing detection
    
    for idx in range(depth, len(candles) - depth):
        window = candles[idx - depth : idx + depth + 1]
        pivot = candles[idx]
        highs = [c.high for c in window if c is not pivot]
        lows = [c.low for c in window if c is not pivot]
        
        # Skip if we don't have enough comparison points
        if not highs or not lows:
            continue
            
        if pivot.high > max(highs):
            swings.append(SwingPoint(index=idx, price=pivot.high, timestamp=pivot.timestamp, kind="high"))
        elif pivot.low < min(lows):
            swings.append(SwingPoint(index=idx, price=pivot.low, timestamp=pivot.timestamp, kind="low"))
    return swings


def recent_structure(swings: Sequence[SwingPoint], max_age_minutes: int = 180) -> List[SwingPoint]:
    if not swings:
        return []
    cutoff = swings[-1].timestamp - timedelta(minutes=max_age_minutes)
    return [s for s in swings if s.timestamp >= cutoff]


def detect_bos(swings: Sequence[SwingPoint]) -> Optional[SwingPoint]:
    if len(swings) < 4:
        return None
    a, b, c, d = swings[-4:]
    if a.kind == "low" and c.kind == "low" and b.kind == "high" and d.kind == "high":
        if c.price > a.price and d.price > b.price:
            return d
    if a.kind == "high" and c.kind == "high" and b.kind == "low" and d.kind == "low":
        if c.price < a.price and d.price < b.price:
            return d
    return None


def detect_mss(swings: Sequence[SwingPoint]) -> Optional[SwingPoint]:
    if len(swings) < 5:
        return None
    last_swings = swings[-5:]
    highs = [s for s in last_swings if s.kind == "high"]
    lows = [s for s in last_swings if s.kind == "low"]
    if len(highs) >= 2 and len(lows) >= 2:
        if highs[-1].price < highs[-2].price and lows[-1].price < lows[-2].price:
            return highs[-1]
        if highs[-1].price > highs[-2].price and lows[-1].price > lows[-2].price:
            return lows[-1]
    return None


def detect_fair_value_gaps(candles: Sequence[Candle]) -> List[FairValueGap]:
    gaps: List[FairValueGap] = []
    for idx in range(2, len(candles)):
        c0, c1, c2 = candles[idx - 2], candles[idx - 1], candles[idx]
        if c0.high < c2.low:
            gaps.append(
                FairValueGap(
                    start_index=idx - 2,
                    end_index=idx,
                    upper=c1.low,
                    lower=c0.high,
                    direction="bullish",
                )
            )
        elif c0.low > c2.high:
            gaps.append(
                FairValueGap(
                    start_index=idx - 2,
                    end_index=idx,
                    upper=c0.low,
                    lower=c1.high,
                    direction="bearish",
                )
            )
    return gaps


def detect_order_blocks(swings: Sequence[SwingPoint], candles: Sequence[Candle]) -> List[OrderBlock]:
    blocks: List[OrderBlock] = []
    for swing in swings[-8:]:
        candle = candles[swing.index]
        if swing.kind == "high":
            blocks.append(
                OrderBlock(index=swing.index, high=candle.high, low=candle.open, direction="bearish", origin_time=candle.timestamp)
            )
        else:
            blocks.append(
                OrderBlock(index=swing.index, high=candle.open, low=candle.low, direction="bullish", origin_time=candle.timestamp)
            )
    return blocks


def equity_curve_stats(values: Sequence[float]) -> Tuple[float, float]:
    if len(values) < 2:
        return 0.0, 0.0
    returns = [(values[i] - values[i - 1]) / values[i - 1] for i in range(1, len(values)) if values[i - 1] != 0]
    if not returns:
        return 0.0, 0.0
    return mean(returns), pstdev(returns)


def microstructure_path(candle: Candle, slices: int) -> Iterator[float]:
    if slices <= 0:
        raise ValueError("slices must be positive")
    half = max(slices // 2, 1)
    up_step = (candle.high - candle.open) / half if half else 0.0
    down_step = (candle.open - candle.low) / max(slices - half, 1)
    price = candle.open
    for _ in range(half):
        price += up_step
        yield min(price, candle.high)
    for _ in range(slices - half):
        price -= down_step
        yield max(price, candle.low)


def pip_distance(price_a: float, price_b: float, tick_size: float) -> float:
    return abs(price_a - price_b) / tick_size


def calculate_atr(candles: Sequence[Candle], period: int = 14) -> float:
    """Backward-compatible ATR helper wrapping rolling_atr."""

    return rolling_atr(candles, period)


def calculate_displacement(candles: Sequence[Candle], lookback: int = 5) -> float:
    """Legacy wrapper for displacement score."""

    return displacement_score(candles, lookback)


def detect_liquidity_sweep(
    candles: Sequence[Candle],
    lookback: int = 40,
    reentry_buffer: float = 0.0,
    reentry_bars: int = 3,
) -> Optional[LiquiditySweep]:
    """Detect a liquidity grab that breaks a prior extreme and returns inside within a few candles."""

    total = len(candles)
    if total < 3:
        return None
    window = list(candles)[-min(max(lookback, reentry_bars + 2), total):]
    buffer = max(reentry_buffer, 0.0)
    recent = min(reentry_bars, len(window) - 1)
    if recent <= 0:
        return None
    start_idx = len(window) - recent
    for idx in range(start_idx, len(window)):
        candidate = window[idx]
        prior_slice = window[:idx]
        if not prior_slice:
            continue
        high = max(c.high for c in prior_slice)
        low = min(c.low for c in prior_slice)
        body_low = min(candidate.open, candidate.close)
        body_high = max(candidate.open, candidate.close)
        if candidate.high > high and body_low <= high + buffer:
            return LiquiditySweep(direction="sell", swept_level=high, timestamp=candidate.timestamp)
        if candidate.low < low and body_high >= low - buffer:
            return LiquiditySweep(direction="buy", swept_level=low, timestamp=candidate.timestamp)
    return None


# ---------------------------------------------------------------------------
# ICT KILLZONES - High-probability trading windows
# ---------------------------------------------------------------------------

ICT_KILLZONES: List[Killzone] = [
    Killzone(name="London_Open", start=time(2, 0), end=time(5, 0), timezone="America/New_York"),
    Killzone(name="NY_AM", start=time(9, 30), end=time(11, 0), timezone="America/New_York"),
    Killzone(name="NY_Lunch", start=time(12, 0), end=time(13, 30), timezone="America/New_York"),
    Killzone(name="NY_PM", start=time(14, 0), end=time(16, 0), timezone="America/New_York"),
    Killzone(name="Silver_Bullet_London", start=time(3, 0), end=time(4, 0), timezone="America/New_York"),
    Killzone(name="Silver_Bullet_NY", start=time(10, 0), end=time(11, 0), timezone="America/New_York"),
    Killzone(name="Silver_Bullet_PM", start=time(14, 0), end=time(15, 0), timezone="America/New_York"),
]


def active_killzone(ts: datetime, killzones: Optional[Sequence[Killzone]] = None) -> Optional[Killzone]:
    """Return the ICT killzone that contains the given timestamp.
    
    IMPORTANT: ts is assumed to be MT5 server time (typically UTC+2/UTC+3).
    Killzones are defined in New York time, so we convert before checking.
    """
    import pytz
    
    zones = killzones or ICT_KILLZONES
    
    # Convert MT5 server time to New York time
    # MT5 server is typically UTC+2 (Europe/Helsinki or similar)
    try:
        server_tz = pytz.timezone('Etc/GMT-2')  # UTC+2 (note: Etc/GMT signs are inverted)
        ny_tz = pytz.timezone('America/New_York')
        
        # Make server time timezone-aware and convert to NY
        if ts.tzinfo is None:
            server_aware = server_tz.localize(ts)
        else:
            server_aware = ts
        ny_time = server_aware.astimezone(ny_tz)
        current = ny_time.time()
    except Exception:
        # Fallback to raw time if conversion fails
        current = ts.time()
    
    for kz in zones:
        if kz.start <= kz.end:
            if kz.start <= current <= kz.end:
                return kz
        else:
            # handles overnight windows
            if current >= kz.start or current <= kz.end:
                return kz
    return None


def is_silver_bullet_window(ts: datetime) -> bool:
    """Check if timestamp falls within an ICT Silver Bullet window (10-11 AM NY)."""
    kz = active_killzone(ts)
    return kz is not None and "Silver_Bullet" in kz.name


# ---------------------------------------------------------------------------
# BREAKER BLOCKS - Failed order blocks that flip polarity
# ---------------------------------------------------------------------------

def detect_breaker_blocks(
    candles: Sequence[Candle],
    order_blocks: Sequence[OrderBlock],
) -> List[BreakerBlock]:
    """Detect breaker blocks: order blocks that have been violated and flipped.
    
    A bullish OB that gets broken to the downside becomes a bearish breaker.
    A bearish OB that gets broken to the upside becomes a bullish breaker.
    """
    breakers: List[BreakerBlock] = []
    if not order_blocks or len(candles) < 2:
        return breakers
    
    for ob in order_blocks:
        if ob.index >= len(candles) - 1:
            continue
        
        # Check subsequent candles for violation
        for i in range(ob.index + 1, len(candles)):
            c = candles[i]
            if ob.direction == "bullish":
                # Bullish OB violated when price closes below OB low
                if c.close < ob.low:
                    breakers.append(BreakerBlock(
                        index=ob.index,
                        high=ob.high,
                        low=ob.low,
                        direction="bearish",  # flipped!
                        origin_time=ob.origin_time,
                        original_ob_direction="bullish",
                    ))
                    break
            else:  # bearish OB
                # Bearish OB violated when price closes above OB high
                if c.close > ob.high:
                    breakers.append(BreakerBlock(
                        index=ob.index,
                        high=ob.high,
                        low=ob.low,
                        direction="bullish",  # flipped!
                        origin_time=ob.origin_time,
                        original_ob_direction="bearish",
                    ))
                    break
    
    return breakers


# ---------------------------------------------------------------------------
# MITIGATION BLOCKS - Order blocks that have been partially filled
# ---------------------------------------------------------------------------

def detect_mitigation_blocks(
    candles: Sequence[Candle],
    order_blocks: Sequence[OrderBlock],
) -> List[MitigationBlock]:
    """Detect mitigation blocks: OBs that price has returned to and reacted from.
    
    When price taps into an OB but doesn't break through, the OB is 'mitigated'.
    These zones often provide strong levels on subsequent tests.
    """
    mitigated: List[MitigationBlock] = []
    if not order_blocks or len(candles) < 2:
        return mitigated
    
    for ob in order_blocks:
        if ob.index >= len(candles) - 1:
            continue
        
        times_tested = 0
        mitigation_level = 0.0
        
        for i in range(ob.index + 1, len(candles)):
            c = candles[i]
            if ob.direction == "bullish":
                # Check if price tapped into the bullish OB zone
                if c.low <= ob.high and c.low >= ob.low:
                    times_tested += 1
                    mitigation_level = c.low
                    # Check for reaction (bullish candle after tap)
                    if c.close > c.open:
                        break
            else:  # bearish OB
                # Check if price tapped into the bearish OB zone
                if c.high >= ob.low and c.high <= ob.high:
                    times_tested += 1
                    mitigation_level = c.high
                    # Check for reaction (bearish candle after tap)
                    if c.close < c.open:
                        break
        
        if times_tested > 0:
            mitigated.append(MitigationBlock(
                index=ob.index,
                high=ob.high,
                low=ob.low,
                direction=ob.direction,
                origin_time=ob.origin_time,
                mitigation_level=mitigation_level,
                times_tested=times_tested,
            ))
    
    return mitigated


# ---------------------------------------------------------------------------
# OPTIMAL TRADE ENTRY (OTE) - 61.8% to 78.6% Fibonacci zone
# ---------------------------------------------------------------------------

def calculate_ote(swing_high: float, swing_low: float, direction: str) -> OTE:
    """Calculate the Optimal Trade Entry zone (61.8% - 78.6% Fib retracement).
    
    For bullish setups: OTE is measured from swing low to high, looking for 
    retracement entries.
    For bearish setups: OTE is measured from swing high to low.
    """
    swing_range = swing_high - swing_low
    
    if direction == "bullish":
        # Bullish: retracement from high, buy in OTE zone
        ote_upper = swing_high - (swing_range * 0.618)
        ote_lower = swing_high - (swing_range * 0.786)
    else:
        # Bearish: retracement from low, sell in OTE zone
        ote_lower = swing_low + (swing_range * 0.618)
        ote_upper = swing_low + (swing_range * 0.786)
    
    return OTE(
        swing_high=swing_high,
        swing_low=swing_low,
        ote_upper=max(ote_upper, ote_lower),
        ote_lower=min(ote_upper, ote_lower),
        direction=direction,
    )


def price_in_ote(price: float, ote: OTE) -> bool:
    """Check if price is within the OTE zone."""
    return ote.ote_lower <= price <= ote.ote_upper


def find_ote_from_swings(swings: Sequence[SwingPoint], direction: str) -> Optional[OTE]:
    """Find OTE zone from recent swing points."""
    if len(swings) < 2:
        return None
    
    highs = [s for s in swings if s.kind == "high"]
    lows = [s for s in swings if s.kind == "low"]
    
    if not highs or not lows:
        return None
    
    recent_high = max(highs, key=lambda s: s.index)
    recent_low = max(lows, key=lambda s: s.index)
    
    return calculate_ote(recent_high.price, recent_low.price, direction)


# ---------------------------------------------------------------------------
# INSTITUTIONAL CANDLES - Large displacement candles
# ---------------------------------------------------------------------------

def detect_institutional_candles(
    candles: Sequence[Candle],
    min_body_ratio: float = 0.7,
    volume_mult: float = 1.5,
    lookback: int = 20,
) -> List[InstitutionalCandle]:
    """Detect institutional candles: large-body, high-volume displacement moves.
    
    These candles often mark the start of a new leg or confirm direction.
    """
    if len(candles) < lookback + 1:
        return []
    
    inst_candles: List[InstitutionalCandle] = []
    avg_volume = mean(c.volume for c in candles[-lookback:-1]) if lookback > 1 else 1.0
    avg_volume = max(avg_volume, 1e-9)
    
    for idx in range(lookback, len(candles)):
        c = candles[idx]
        body = abs(c.close - c.open)
        body_ratio = body / c.range if c.range > 0 else 0.0
        volume_ratio = c.volume / avg_volume if avg_volume > 0 else 0.0
        
        if body_ratio >= min_body_ratio and volume_ratio >= volume_mult:
            inst_candles.append(InstitutionalCandle(
                index=idx,
                candle=c,
                body_ratio=body_ratio,
                volume_ratio=volume_ratio,
            ))
    
    return inst_candles


# ---------------------------------------------------------------------------
# AMD PATTERN - Accumulation, Manipulation, Distribution
# ---------------------------------------------------------------------------

def detect_amd_pattern(
    candles: Sequence[Candle],
    min_accumulation_bars: int = 5,
    manipulation_threshold: float = 1.5,
) -> Optional[AMDPattern]:
    """Detect Accumulation-Manipulation-Distribution pattern.
    
    1. Accumulation: tight range (consolidation)
    2. Manipulation: false breakout (liquidity grab)
    3. Distribution: true move in opposite direction
    
    This is the classic ICT setup for catching institutional moves.
    """
    if len(candles) < min_accumulation_bars + 3:
        return None
    
    # Look for accumulation (low volatility period)
    recent = candles[-min_accumulation_bars - 3:]
    accumulation = recent[:min_accumulation_bars]
    
    acc_high = max(c.high for c in accumulation)
    acc_low = min(c.low for c in accumulation)
    acc_range = acc_high - acc_low
    
    if acc_range <= 0:
        return None
    
    avg_candle_range = mean(c.range for c in accumulation)
    
    # Check for manipulation (break of accumulation range)
    potential_manip = recent[min_accumulation_bars:-1]
    distribution_candle = recent[-1]
    
    manipulation_detected = False
    direction = ""
    
    for c in potential_manip:
        if c.high > acc_high + (avg_candle_range * 0.5):
            # Upside manipulation - expect bearish move
            if distribution_candle.close < distribution_candle.open:
                manipulation_detected = True
                direction = "bearish"
                break
        if c.low < acc_low - (avg_candle_range * 0.5):
            # Downside manipulation - expect bullish move
            if distribution_candle.close > distribution_candle.open:
                manipulation_detected = True
                direction = "bullish"
                break
    
    if not manipulation_detected:
        return None
    
    return AMDPattern(
        phase="distribution",
        start_index=len(candles) - len(recent),
        end_index=len(candles) - 1,
        range_high=acc_high,
        range_low=acc_low,
        direction=direction,
    )


# ---------------------------------------------------------------------------
# ENHANCED FAIR VALUE GAP DETECTION
# ---------------------------------------------------------------------------

def detect_unfilled_fvg(
    candles: Sequence[Candle],
    gaps: Optional[Sequence[FairValueGap]] = None,
) -> List[FairValueGap]:
    """Return only FVGs that haven't been filled (price hasn't returned through them)."""
    if gaps is None:
        gaps = detect_fair_value_gaps(candles)
    
    unfilled: List[FairValueGap] = []
    
    for gap in gaps:
        if gap.end_index >= len(candles):
            continue
        
        filled = False
        for i in range(gap.end_index + 1, len(candles)):
            c = candles[i]
            if gap.direction == "bullish":
                # Bullish FVG filled if price trades down through it
                if c.low <= gap.lower:
                    filled = True
                    break
            else:
                # Bearish FVG filled if price trades up through it
                if c.high >= gap.upper:
                    filled = True
                    break
        
        if not filled:
            unfilled.append(gap)
    
    return unfilled


def fvg_consequent_encroachment(gap: FairValueGap) -> float:
    """Calculate the 50% level (consequent encroachment) of an FVG.
    
    This is a key ICT level where price often reacts.
    """
    return (gap.upper + gap.lower) / 2


# ---------------------------------------------------------------------------
# ASIAN SESSION RANGE - Key liquidity levels for London/NY
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class AsianRange:
    """Asian session high/low used as liquidity levels."""
    
    high: float
    low: float
    mid: float  # 50% level (equilibrium)
    start_time: datetime
    end_time: datetime
    swept_high: bool = False
    swept_low: bool = False


def calculate_asian_range(candles: Sequence[Candle]) -> Optional[AsianRange]:
    """Calculate the Asian session high/low for liquidity targeting.
    
    Asian session in NY time: 7 PM - 12 AM (previous day's evening)
    This range is key for London/NY liquidity sweeps.
    """
    import pytz
    
    if len(candles) < 20:
        return None
    
    try:
        server_tz = pytz.timezone('Etc/GMT-2')  # MT5 server typically UTC+2
        ny_tz = pytz.timezone('America/New_York')
        
        asian_candles = []
        
        for c in candles[-100:]:  # Look at recent candles
            if c.timestamp.tzinfo is None:
                server_aware = server_tz.localize(c.timestamp)
            else:
                server_aware = c.timestamp
            ny_time = server_aware.astimezone(ny_tz)
            hour = ny_time.hour
            
            # Asian session: 7 PM - 12 AM NY time (previous trading day)
            if hour >= 19 or hour < 0:
                asian_candles.append(c)
        
        if len(asian_candles) < 5:
            return None
        
        high = max(c.high for c in asian_candles)
        low = min(c.low for c in asian_candles)
        mid = (high + low) / 2
        
        return AsianRange(
            high=high,
            low=low,
            mid=mid,
            start_time=asian_candles[0].timestamp,
            end_time=asian_candles[-1].timestamp,
        )
    except Exception:
        return None


def asian_range_swept(candles: Sequence[Candle], asian: AsianRange) -> Tuple[bool, bool]:
    """Check if Asian high/low have been swept.
    
    Returns (high_swept, low_swept).
    """
    high_swept = any(c.high > asian.high for c in candles)
    low_swept = any(c.low < asian.low for c in candles)
    return high_swept, low_swept


# ---------------------------------------------------------------------------
# PREMIUM/DISCOUNT ZONES - Trade at optimal price levels
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PremiumDiscount:
    """Premium and Discount zones based on current range."""
    
    range_high: float
    range_low: float
    equilibrium: float  # 50% level
    premium_zone: float  # 75% and above (sell zone)
    discount_zone: float  # 25% and below (buy zone)


def calculate_premium_discount(swings: Sequence[SwingPoint]) -> Optional[PremiumDiscount]:
    """Calculate premium/discount zones from recent structure.
    
    ICT concept: Only buy in discount (below 50%), only sell in premium (above 50%).
    This ensures we're always trading at optimal price levels.
    """
    if len(swings) < 2:
        return None
    
    highs = [s.price for s in swings if s.kind == "high"]
    lows = [s.price for s in swings if s.kind == "low"]
    
    if not highs or not lows:
        return None
    
    range_high = max(highs)
    range_low = min(lows)
    swing_range = range_high - range_low
    
    if swing_range <= 0:
        return None
    
    equilibrium = range_low + (swing_range * 0.5)
    premium_zone = range_low + (swing_range * 0.75)
    discount_zone = range_low + (swing_range * 0.25)
    
    return PremiumDiscount(
        range_high=range_high,
        range_low=range_low,
        equilibrium=equilibrium,
        premium_zone=premium_zone,
        discount_zone=discount_zone,
    )


def price_in_premium(price: float, zones: PremiumDiscount) -> bool:
    """Check if price is in premium zone (ideal for sells)."""
    return price >= zones.premium_zone


def price_in_discount(price: float, zones: PremiumDiscount) -> bool:
    """Check if price is in discount zone (ideal for buys)."""
    return price <= zones.discount_zone


def price_zone(price: float, zones: PremiumDiscount) -> str:
    """Return which zone price is in: 'premium', 'discount', or 'equilibrium'."""
    if price >= zones.premium_zone:
        return "premium"
    elif price <= zones.discount_zone:
        return "discount"
    else:
        return "equilibrium"


# ---------------------------------------------------------------------------
# POWER OF 3 (PO3) - AMD Entry Timing
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class PowerOf3:
    """Power of 3 pattern: Open, High/Low of day prediction.
    
    ICT concept: The first move of the day is often the opposite direction
    of the true daily move (manipulation before distribution).
    """
    
    session_open: float
    session_high: float
    session_low: float
    predicted_direction: str  # "bullish" or "bearish"
    manipulation_complete: bool


def detect_power_of_3(
    candles: Sequence[Candle],
    session_start_hour: int = 9,  # NY session start
) -> Optional[PowerOf3]:
    """Detect Power of 3 pattern for session trading.
    
    Pattern:
    1. Accumulation: First 30 mins (9:00-9:30)
    2. Manipulation: False move (9:30-10:00) 
    3. Distribution: True move (10:00 onwards)
    """
    import pytz
    
    if len(candles) < 20:
        return None
    
    try:
        server_tz = pytz.timezone('Etc/GMT-2')
        ny_tz = pytz.timezone('America/New_York')
        
        # Find session open candle
        session_candles = []
        session_open = None
        
        for c in candles[-50:]:
            if c.timestamp.tzinfo is None:
                server_aware = server_tz.localize(c.timestamp)
            else:
                server_aware = c.timestamp
            ny_time = server_aware.astimezone(ny_tz)
            
            # Collect NY session candles (9:00 onwards)
            if ny_time.hour >= session_start_hour and ny_time.hour < 16:
                session_candles.append(c)
                if session_open is None:
                    session_open = c.open
        
        if len(session_candles) < 10 or session_open is None:
            return None
        
        session_high = max(c.high for c in session_candles)
        session_low = min(c.low for c in session_candles)
        current_price = session_candles[-1].close
        
        # Determine predicted direction based on manipulation
        # If early session took out lows, expect bullish
        # If early session took out highs, expect bearish
        early_candles = session_candles[:6]  # First 30 mins (assuming 5-min candles)
        
        early_high = max(c.high for c in early_candles)
        early_low = min(c.low for c in early_candles)
        
        # Check if manipulation occurred
        later_candles = session_candles[6:] if len(session_candles) > 6 else []
        manipulation_complete = False
        predicted_direction = "unknown"
        
        if later_candles:
            later_low = min(c.low for c in later_candles)
            later_high = max(c.high for c in later_candles)
            
            # Low taken early, then reversed = bullish PO3
            if early_low < session_open * 0.999 and current_price > session_open:
                predicted_direction = "bullish"
                manipulation_complete = True
            # High taken early, then reversed = bearish PO3
            elif early_high > session_open * 1.001 and current_price < session_open:
                predicted_direction = "bearish"
                manipulation_complete = True
        
        if predicted_direction == "unknown":
            return None
        
        return PowerOf3(
            session_open=session_open,
            session_high=session_high,
            session_low=session_low,
            predicted_direction=predicted_direction,
            manipulation_complete=manipulation_complete,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# TIME-BASED ENTRY WINDOWS - Optimal entry times
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class TimeWindow:
    """Optimal time window for entries based on ICT research."""
    
    name: str
    start: time
    end: time
    session: str
    quality: float  # 0.0 - 1.0 quality score


# Optimal ICT entry windows based on research
ICT_OPTIMAL_WINDOWS: List[TimeWindow] = [
    # London Session
    TimeWindow("London_Killzone_Open", time(2, 0), time(4, 0), "London", 0.9),
    TimeWindow("London_Midday", time(4, 0), time(6, 0), "London", 0.7),
    # NY Session
    TimeWindow("NY_Open_Killzone", time(9, 30), time(10, 30), "NewYork", 0.95),
    TimeWindow("Silver_Bullet_10AM", time(10, 0), time(11, 0), "NewYork", 1.0),  # Best ICT window
    TimeWindow("NY_Lunch_Avoid", time(12, 0), time(13, 0), "NewYork", 0.3),  # Low quality
    TimeWindow("NY_PM_Killzone", time(14, 0), time(15, 0), "NewYork", 0.85),
    TimeWindow("Silver_Bullet_2PM", time(14, 0), time(15, 0), "NewYork", 0.9),
]


def get_entry_quality(ts: datetime) -> float:
    """Get the quality score for entering at this time.
    
    Returns 0.0 - 1.0 where 1.0 is optimal entry time.
    Used to adjust confluence/confidence scoring.
    """
    import pytz
    
    try:
        server_tz = pytz.timezone('Etc/GMT-2')
        ny_tz = pytz.timezone('America/New_York')
        
        if ts.tzinfo is None:
            server_aware = server_tz.localize(ts)
        else:
            server_aware = ts
        ny_time = server_aware.astimezone(ny_tz)
        current = ny_time.time()
        
        for window in ICT_OPTIMAL_WINDOWS:
            if window.start <= window.end:
                if window.start <= current <= window.end:
                    return window.quality
            else:
                if current >= window.start or current <= window.end:
                    return window.quality
        
        return 0.5  # Default quality for unspecified times
    except Exception:
        return 0.5


# ---------------------------------------------------------------------------
# REFINED LIQUIDITY LEVELS
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class LiquidityPool:
    """A cluster of stop-losses creating a liquidity pool."""
    
    level: float
    side: str  # "buy" (stops below) or "sell" (stops above)
    strength: int  # number of swing points creating this level
    last_touch: datetime


def identify_liquidity_pools(swings: Sequence[SwingPoint], tolerance_mult: float = 0.001) -> List[LiquidityPool]:
    """Identify liquidity pools from swing clusters.
    
    Multiple equal lows = buy-side liquidity (stops below)
    Multiple equal highs = sell-side liquidity (stops above)
    """
    pools: List[LiquidityPool] = []
    
    if len(swings) < 3:
        return pools
    
    # Group highs and lows separately
    highs = [(s.price, s.timestamp) for s in swings if s.kind == "high"]
    lows = [(s.price, s.timestamp) for s in swings if s.kind == "low"]
    
    # Find clusters in highs (sell-side liquidity)
    for i, (price, ts) in enumerate(highs):
        cluster_count = 1
        for j, (other_price, other_ts) in enumerate(highs):
            if i != j and abs(price - other_price) / price <= tolerance_mult:
                cluster_count += 1
        
        if cluster_count >= 2:
            pools.append(LiquidityPool(
                level=price,
                side="sell",
                strength=cluster_count,
                last_touch=ts,
            ))
    
    # Find clusters in lows (buy-side liquidity)
    for i, (price, ts) in enumerate(lows):
        cluster_count = 1
        for j, (other_price, other_ts) in enumerate(lows):
            if i != j and abs(price - other_price) / price <= tolerance_mult:
                cluster_count += 1
        
        if cluster_count >= 2:
            pools.append(LiquidityPool(
                level=price,
                side="buy",
                strength=cluster_count,
                last_touch=ts,
            ))
    
    # Remove duplicates and sort by strength
    seen_levels: set = set()
    unique_pools: List[LiquidityPool] = []
    for pool in sorted(pools, key=lambda p: p.strength, reverse=True):
        level_key = round(pool.level, 5)
        if level_key not in seen_levels:
            seen_levels.add(level_key)
            unique_pools.append(pool)
    
    return unique_pools


def nearest_liquidity_pool(price: float, pools: Sequence[LiquidityPool], direction: str) -> Optional[LiquidityPool]:
    """Find the nearest liquidity pool in the given direction.
    
    For buys: find sell-side liquidity above price
    For sells: find buy-side liquidity below price
    """
    if not pools:
        return None
    
    if direction == "buy":
        # Looking for sell-side liquidity above current price
        candidates = [p for p in pools if p.side == "sell" and p.level > price]
    else:
        # Looking for buy-side liquidity below current price
        candidates = [p for p in pools if p.side == "buy" and p.level < price]
    
    if not candidates:
        return None
    
    # Return closest by distance
    return min(candidates, key=lambda p: abs(p.level - price))


# ---------------------------------------------------------------------------
# CRT (CANDLE RANGE THEORY) - 3-Candle Reversal Pattern
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CRTPattern:
    """Candle Range Theory pattern: 3-candle liquidity sweep reversal.
    
    ICT concept:
    1. Price approaches HTF level (support/resistance)
    2. Previous candle closes near the level
    3. Next candle sweeps the high/low (grabs liquidity)
    4. Current candle reverses and breaks opposite side
    Entry: On close of reversal candle
    Stop: Below/above swept level
    Target: CRT high/low (range of the 3 candles)
    """
    
    index: int
    direction: str  # "bullish" or "bearish"
    sweep_candle_index: int
    reversal_candle_index: int
    swept_level: float
    entry_level: float
    stop_level: float
    target_level: float
    htf_level: Optional[float]  # The HTF level that was approached
    timestamp: datetime


def detect_crt_pattern(
    candles: Sequence[Candle],
    htf_levels: Optional[Sequence[float]] = None,
    tolerance_mult: float = 0.002,
) -> Optional[CRTPattern]:
    """Detect Candle Range Theory reversal pattern.
    
    CRT Setup Requirements:
    1. Price near HTF level (or recent swing high/low)
    2. Candle 1: Approaches level
    3. Candle 2: Sweeps liquidity (breaks level briefly)
    4. Candle 3: Reverses and closes beyond opposite side of Candle 1
    
    Args:
        candles: Price candle sequence
        htf_levels: Optional list of higher timeframe levels to watch
        tolerance_mult: Proximity tolerance as multiplier of price
    
    Returns:
        CRTPattern if detected, None otherwise
    """
    if len(candles) < 5:
        return None
    
    # Get the last 3 candles for CRT analysis
    c1, c2, c3 = candles[-3], candles[-2], candles[-1]
    
    # If no HTF levels provided, use recent swing extremes
    if htf_levels is None or len(htf_levels) == 0:
        recent = candles[-20:]
        htf_levels = [
            max(c.high for c in recent),
            min(c.low for c in recent),
        ]
    
    tolerance = c3.close * tolerance_mult
    
    # Check for BULLISH CRT (sweep low, reverse up)
    for level in htf_levels:
        # Check if level is near recent lows
        if level >= c3.close:
            continue
        
        # Candle 1 should close near the level
        if abs(c1.low - level) > tolerance * 2:
            continue
        
        # Candle 2 sweeps below the level (liquidity grab)
        if c2.low >= level:
            continue
        
        # Candle 2 sweeps below Candle 1 low
        if c2.low >= c1.low:
            continue
        
        # Candle 3 must close above Candle 1 high (reversal confirmation)
        if c3.close <= c1.high:
            continue
        
        # Candle 3 should be bullish
        if c3.close <= c3.open:
            continue
        
        # CRT range
        crt_high = max(c1.high, c2.high, c3.high)
        crt_low = min(c1.low, c2.low, c3.low)
        
        return CRTPattern(
            index=len(candles) - 1,
            direction="bullish",
            sweep_candle_index=len(candles) - 2,
            reversal_candle_index=len(candles) - 1,
            swept_level=c2.low,
            entry_level=c3.close,
            stop_level=crt_low,
            target_level=crt_high,
            htf_level=level,
            timestamp=c3.timestamp,
        )
    
    # Check for BEARISH CRT (sweep high, reverse down)
    for level in htf_levels:
        # Check if level is near recent highs
        if level <= c3.close:
            continue
        
        # Candle 1 should close near the level
        if abs(c1.high - level) > tolerance * 2:
            continue
        
        # Candle 2 sweeps above the level (liquidity grab)
        if c2.high <= level:
            continue
        
        # Candle 2 sweeps above Candle 1 high
        if c2.high <= c1.high:
            continue
        
        # Candle 3 must close below Candle 1 low (reversal confirmation)
        if c3.close >= c1.low:
            continue
        
        # Candle 3 should be bearish
        if c3.close >= c3.open:
            continue
        
        # CRT range
        crt_high = max(c1.high, c2.high, c3.high)
        crt_low = min(c1.low, c2.low, c3.low)
        
        return CRTPattern(
            index=len(candles) - 1,
            direction="bearish",
            sweep_candle_index=len(candles) - 2,
            reversal_candle_index=len(candles) - 1,
            swept_level=c2.high,
            entry_level=c3.close,
            stop_level=crt_high,
            target_level=crt_low,
            htf_level=level,
            timestamp=c3.timestamp,
        )
    
    return None


# ---------------------------------------------------------------------------
# CISD (CHANGE IN STATE OF DELIVERY) - Structure Shift Confirmation
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class CISDSignal:
    """Change in State of Delivery: marks the transition from one directional bias to another.
    
    CISD occurs when:
    1. A swing point is formed
    2. Price sweeps liquidity beyond that swing
    3. Price then reverses and breaks back through with conviction
    4. The prior directional bias is now invalidated
    
    This is stronger than a simple BOS (Break of Structure) because it requires
    the liquidity sweep first, confirming institutional involvement.
    """
    
    index: int
    old_bias: str  # "bullish" or "bearish" 
    new_bias: str  # "bullish" or "bearish"
    swing_broken: float  # The swing level that was broken
    sweep_level: float  # Where liquidity was swept
    confirmation_level: float  # Where CISD was confirmed
    tolerance_ratio: float  # How convincing the shift was (0-1)
    timestamp: datetime


def detect_cisd(
    candles: Sequence[Candle],
    swings: Sequence[SwingPoint],
    min_tolerance_ratio: float = 0.5,
) -> Optional[CISDSignal]:
    """Detect Change in State of Delivery.
    
    CISD Detection Algorithm:
    1. Identify recent swing highs/lows
    2. Check if price swept beyond swing (liquidity grab)
    3. Check if price then broke back through the swing in opposite direction
    4. Calculate tolerance ratio (conviction of the shift)
    
    Args:
        candles: Price candle sequence
        swings: Recent swing points
        min_tolerance_ratio: Minimum ratio for valid CISD (0-1)
    
    Returns:
        CISDSignal if detected, None otherwise
    """
    if len(candles) < 10 or len(swings) < 4:
        return None
    
    recent_candles = candles[-15:]
    recent_swings = list(swings)[-6:]
    
    # Get the most recent highs and lows
    highs = [s for s in recent_swings if s.kind == "high"]
    lows = [s for s in recent_swings if s.kind == "low"]
    
    if len(highs) < 2 or len(lows) < 2:
        return None
    
    current = candles[-1]
    
    # Check for BULLISH CISD (was bearish, now bullish)
    # Pattern: Price swept below a swing low, then reversed and broke above prior swing high
    lowest_swing = min(lows, key=lambda s: s.price)
    highest_swing = max(highs, key=lambda s: s.price)
    
    # Find if price swept below lowest swing
    sweep_low = any(c.low < lowest_swing.price for c in recent_candles)
    
    if sweep_low:
        # Check if current price broke back above a prior swing high
        prior_high = sorted(highs, key=lambda s: s.index)[-2] if len(highs) >= 2 else None
        if prior_high and current.close > prior_high.price:
            # Calculate tolerance ratio (how convincing)
            sweep_depth = lowest_swing.price - min(c.low for c in recent_candles)
            swing_range = highest_swing.price - lowest_swing.price
            tolerance_ratio = min(sweep_depth / swing_range, 1.0) if swing_range > 0 else 0.0
            
            if tolerance_ratio >= min_tolerance_ratio:
                return CISDSignal(
                    index=len(candles) - 1,
                    old_bias="bearish",
                    new_bias="bullish",
                    swing_broken=prior_high.price,
                    sweep_level=min(c.low for c in recent_candles),
                    confirmation_level=current.close,
                    tolerance_ratio=tolerance_ratio,
                    timestamp=current.timestamp,
                )
    
    # Check for BEARISH CISD (was bullish, now bearish)
    # Pattern: Price swept above a swing high, then reversed and broke below prior swing low
    sweep_high = any(c.high > highest_swing.price for c in recent_candles)
    
    if sweep_high:
        # Check if current price broke back below a prior swing low
        prior_low = sorted(lows, key=lambda s: s.index)[-2] if len(lows) >= 2 else None
        if prior_low and current.close < prior_low.price:
            # Calculate tolerance ratio
            sweep_height = max(c.high for c in recent_candles) - highest_swing.price
            swing_range = highest_swing.price - lowest_swing.price
            tolerance_ratio = min(sweep_height / swing_range, 1.0) if swing_range > 0 else 0.0
            
            if tolerance_ratio >= min_tolerance_ratio:
                return CISDSignal(
                    index=len(candles) - 1,
                    old_bias="bullish",
                    new_bias="bearish",
                    swing_broken=prior_low.price,
                    sweep_level=max(c.high for c in recent_candles),
                    confirmation_level=current.close,
                    tolerance_ratio=tolerance_ratio,
                    timestamp=current.timestamp,
                )
    
    return None


# ---------------------------------------------------------------------------
# ATR-BASED DISPLACEMENT ZONES - Fake vs Real Moves
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class ATRDisplacement:
    """ATR-based move classification for ICT trading.
    
    Based on ICT teaching:
    - ±0.5 ATR from open = Fake moves / Judas Swing zone
    - ±1.0 ATR = Normal range
    - ±2.0 ATR = Real displacement / Silver Bullet entries
    
    Helps identify:
    - Judas Swings (false moves to grab liquidity)
    - True displacement (institutional momentum)
    - Entry quality based on move magnitude
    """
    
    current_price: float
    session_open: float
    atr: float
    move_atr_multiple: float  # Current move in ATR units
    zone: str  # "judas", "normal", "displacement"
    is_fake_move: bool
    is_real_displacement: bool


def classify_atr_displacement(
    candles: Sequence[Candle],
    session_open: Optional[float] = None,
    atr_period: int = 14,
) -> Optional[ATRDisplacement]:
    """Classify current price move relative to ATR bands.
    
    ICT uses ATR multiples to distinguish:
    - Fake moves (Judas Swings) within 0.5 ATR
    - Normal moves within 1.0 ATR
    - Real displacement beyond 2.0 ATR
    
    Real displacement indicates institutional activity and
    higher probability continuation.
    
    Args:
        candles: Price candle sequence
        session_open: Session open price (defaults to detecting from candles)
        atr_period: Period for ATR calculation
    
    Returns:
        ATRDisplacement classification
    """
    if len(candles) < atr_period + 5:
        return None
    
    atr = calculate_atr(candles, period=atr_period)
    current = candles[-1]
    
    # Determine session open if not provided
    if session_open is None:
        # Use open of first candle in recent session-like grouping
        import pytz
        try:
            server_tz = pytz.timezone('Etc/GMT-2')
            ny_tz = pytz.timezone('America/New_York')
            
            for c in candles[-50:]:
                if c.timestamp.tzinfo is None:
                    server_aware = server_tz.localize(c.timestamp)
                else:
                    server_aware = c.timestamp
                ny_time = server_aware.astimezone(ny_tz)
                
                # Session open at 9:30 NY or nearest
                if ny_time.hour == 9 and ny_time.minute < 35:
                    session_open = c.open
                    break
        except Exception:
            pass
        
        if session_open is None:
            # Fallback to recent range
            session_open = candles[-20].open
    
    # Calculate move from session open
    move = current.close - session_open
    move_atr_multiple = abs(move) / atr if atr > 0 else 0.0
    
    # Classify the move
    if move_atr_multiple <= 0.5:
        zone = "judas"
        is_fake = True
        is_displacement = False
    elif move_atr_multiple <= 1.0:
        zone = "normal"
        is_fake = False
        is_displacement = False
    elif move_atr_multiple >= 2.0:
        zone = "displacement"
        is_fake = False
        is_displacement = True
    else:
        zone = "transition"
        is_fake = False
        is_displacement = False
    
    return ATRDisplacement(
        current_price=current.close,
        session_open=session_open,
        atr=atr,
        move_atr_multiple=move_atr_multiple,
        zone=zone,
        is_fake_move=is_fake,
        is_real_displacement=is_displacement,
    )


def is_judas_swing(candles: Sequence[Candle], atr_period: int = 14) -> bool:
    """Check if current move is a Judas Swing (fake move).
    
    Judas Swing: Initial move in wrong direction to grab liquidity
    before the real move. Typically within 0.5 ATR of session open.
    """
    displacement = classify_atr_displacement(candles, atr_period=atr_period)
    return displacement is not None and displacement.is_fake_move


def is_real_displacement(candles: Sequence[Candle], atr_period: int = 14) -> bool:
    """Check if current move shows real displacement (2+ ATR).
    
    Real displacement indicates institutional involvement and
    high probability of continuation.
    """
    displacement = classify_atr_displacement(candles, atr_period=atr_period)
    return displacement is not None and displacement.is_real_displacement


# ---------------------------------------------------------------------------
# MOMENTUM-WEIGHTED FVGs - RSI-based strength ranking
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class MomentumFVG(FairValueGap):
    """Fair Value Gap with momentum strength assessment.
    
    FVGs formed during strong momentum (high RSI for bullish, low RSI for bearish)
    are more likely to hold and act as continuation levels.
    
    FVGs formed during weak momentum are more likely to get filled.
    """
    
    momentum_strength: float  # 0.0 - 1.0 strength score
    rsi_at_formation: float
    likely_to_hold: bool


def calculate_rsi(candles: Sequence[Candle], period: int = 14) -> float:
    """Calculate RSI (Relative Strength Index)."""
    if len(candles) < period + 1:
        return 50.0  # Neutral if not enough data
    
    gains = []
    losses = []
    
    for i in range(1, period + 1):
        change = candles[-i].close - candles[-i - 1].close
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains) / period
    avg_loss = sum(losses) / period
    
    if avg_loss == 0:
        return 100.0
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi


def detect_momentum_fvgs(
    candles: Sequence[Candle],
    rsi_period: int = 14,
    strong_threshold: float = 0.7,  # RSI > 70 for bullish, < 30 for bearish
) -> List[MomentumFVG]:
    """Detect Fair Value Gaps with momentum strength assessment.
    
    Ranks FVGs by the momentum present at their formation:
    - Strong bullish FVG: Formed when RSI > 70
    - Strong bearish FVG: Formed when RSI < 30
    - Weak FVGs: Formed during neutral RSI (likely to fill)
    
    Args:
        candles: Price candle sequence
        rsi_period: RSI calculation period
        strong_threshold: RSI threshold for strong momentum (default 70/30)
    
    Returns:
        List of MomentumFVG with strength assessments
    """
    if len(candles) < rsi_period + 5:
        return []
    
    momentum_gaps: List[MomentumFVG] = []
    
    # Calculate RSI at each point
    rsi_values = []
    for i in range(rsi_period + 1, len(candles) + 1):
        rsi = calculate_rsi(candles[:i], period=rsi_period)
        rsi_values.append(rsi)
    
    # Detect FVGs and assess momentum
    for idx in range(2, len(candles)):
        c0, c1, c2 = candles[idx - 2], candles[idx - 1], candles[idx]
        
        # Get RSI at formation time (if available)
        rsi_idx = idx - rsi_period - 1
        rsi_at_formation = rsi_values[rsi_idx] if rsi_idx >= 0 and rsi_idx < len(rsi_values) else 50.0
        
        gap = None
        momentum_strength = 0.0
        likely_to_hold = False
        
        # Bullish FVG
        if c0.high < c2.low:
            gap_upper = c1.low
            gap_lower = c0.high
            direction = "bullish"
            
            # Strong bullish momentum: RSI > 70
            if rsi_at_formation >= (strong_threshold * 100):
                momentum_strength = min((rsi_at_formation - 50) / 50, 1.0)
                likely_to_hold = True
            else:
                momentum_strength = max((rsi_at_formation - 30) / 40, 0.0)
                likely_to_hold = False
            
            gap = MomentumFVG(
                start_index=idx - 2,
                end_index=idx,
                upper=gap_upper,
                lower=gap_lower,
                direction=direction,
                momentum_strength=momentum_strength,
                rsi_at_formation=rsi_at_formation,
                likely_to_hold=likely_to_hold,
            )
        
        # Bearish FVG
        elif c0.low > c2.high:
            gap_upper = c0.low
            gap_lower = c1.high
            direction = "bearish"
            
            # Strong bearish momentum: RSI < 30
            if rsi_at_formation <= (100 - strong_threshold * 100):
                momentum_strength = min((50 - rsi_at_formation) / 50, 1.0)
                likely_to_hold = True
            else:
                momentum_strength = max((70 - rsi_at_formation) / 40, 0.0)
                likely_to_hold = False
            
            gap = MomentumFVG(
                start_index=idx - 2,
                end_index=idx,
                upper=gap_upper,
                lower=gap_lower,
                direction=direction,
                momentum_strength=momentum_strength,
                rsi_at_formation=rsi_at_formation,
                likely_to_hold=likely_to_hold,
            )
        
        if gap:
            momentum_gaps.append(gap)
    
    return momentum_gaps


def get_strongest_fvg(
    gaps: Sequence[MomentumFVG],
    direction: str,
    price: float,
    tolerance: float,
) -> Optional[MomentumFVG]:
    """Get the strongest momentum FVG for the given direction.
    
    Prioritizes:
    1. FVGs likely to hold (strong momentum)
    2. FVGs closest to current price
    3. Higher momentum strength
    """
    candidates = []
    
    for gap in gaps:
        if gap.direction != direction:
            continue
        
        # Check if price is inside or near the gap
        if gap.contains(price) or min(abs(price - gap.upper), abs(price - gap.lower)) <= tolerance:
            candidates.append(gap)
    
    if not candidates:
        return None
    
    # Sort by: likely_to_hold (desc), momentum_strength (desc), distance to price (asc)
    candidates.sort(key=lambda g: (
        -int(g.likely_to_hold),
        -g.momentum_strength,
        min(abs(price - g.upper), abs(price - g.lower)),
    ))
    
    return candidates[0]


# ---------------------------------------------------------------------------
# SESSION RANGE TRACKING - AM/PM/London session high/low
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class SessionRange:
    """High/Low range for a specific trading session.
    
    Sessions:
    - London: 02:00-05:00 ET
    - AM: 09:30-12:00 ET
    - NY Lunch: 12:00-13:30 ET
    - PM: 13:30-16:00 ET
    """
    
    session_name: str
    high: float
    low: float
    mid: float
    open: float
    close: float
    start_time: datetime
    end_time: Optional[datetime]
    is_active: bool
    high_swept: bool = False
    low_swept: bool = False


# Session definitions in NY time
SESSION_DEFINITIONS = {
    "London": (time(2, 0), time(5, 0)),
    "AM": (time(9, 30), time(12, 0)),
    "NY_Lunch": (time(12, 0), time(13, 30)),
    "PM": (time(13, 30), time(16, 0)),
}


def calculate_session_range(
    candles: Sequence[Candle],
    session_name: str,
) -> Optional[SessionRange]:
    """Calculate the high/low range for a specific session.
    
    These ranges act as liquidity targets for subsequent sessions.
    E.g., London range is often swept during NY session.
    
    Args:
        candles: Price candle sequence
        session_name: One of "London", "AM", "NY_Lunch", "PM"
    
    Returns:
        SessionRange with high/low levels
    """
    if session_name not in SESSION_DEFINITIONS:
        return None
    
    if len(candles) < 10:
        return None
    
    start_time, end_time = SESSION_DEFINITIONS[session_name]
    
    import pytz
    try:
        server_tz = pytz.timezone('Etc/GMT-2')
        ny_tz = pytz.timezone('America/New_York')
        
        session_candles = []
        session_start = None
        session_end = None
        
        for c in candles[-200:]:  # Look at recent candles
            if c.timestamp.tzinfo is None:
                server_aware = server_tz.localize(c.timestamp)
            else:
                server_aware = c.timestamp
            ny_time = server_aware.astimezone(ny_tz)
            current_time = ny_time.time()
            
            # Check if candle is within session
            if start_time <= current_time < end_time:
                session_candles.append(c)
                if session_start is None:
                    session_start = c.timestamp
                session_end = c.timestamp
        
        if len(session_candles) < 3:
            return None
        
        high = max(c.high for c in session_candles)
        low = min(c.low for c in session_candles)
        mid = (high + low) / 2
        session_open = session_candles[0].open
        session_close = session_candles[-1].close
        
        # Check if session is currently active
        current = candles[-1]
        if current.timestamp.tzinfo is None:
            current_aware = server_tz.localize(current.timestamp)
        else:
            current_aware = current.timestamp
        current_ny = current_aware.astimezone(ny_tz)
        is_active = start_time <= current_ny.time() < end_time
        
        return SessionRange(
            session_name=session_name,
            high=high,
            low=low,
            mid=mid,
            open=session_open,
            close=session_close,
            start_time=session_start,
            end_time=session_end,
            is_active=is_active,
        )
    except Exception:
        return None


def get_all_session_ranges(candles: Sequence[Candle]) -> List[SessionRange]:
    """Get ranges for all trading sessions."""
    ranges = []
    for session_name in SESSION_DEFINITIONS.keys():
        range_data = calculate_session_range(candles, session_name)
        if range_data:
            ranges.append(range_data)
    return ranges


def check_session_sweep(
    candles: Sequence[Candle],
    session_range: SessionRange,
    lookback: int = 20,
) -> Tuple[bool, bool]:
    """Check if a session's high/low has been swept.
    
    Returns (high_swept, low_swept).
    Session sweeps are high-probability reversal signals.
    """
    if session_range.is_active:
        # Can't sweep active session
        return False, False
    
    recent = candles[-lookback:] if len(candles) >= lookback else candles
    
    high_swept = any(c.high > session_range.high for c in recent)
    low_swept = any(c.low < session_range.low for c in recent)
    
    return high_swept, low_swept


# ---------------------------------------------------------------------------
# GAP MITIGATION TRACKING
# ---------------------------------------------------------------------------

@dataclass(slots=True)
class GapMitigation:
    """Tracks opening gap mitigation.
    
    ICT concept: Gaps often fill to 50% level before continuation.
    This tracks gaps between session close and next session open.
    """
    
    gap_start: float  # Previous close
    gap_end: float  # Current open
    gap_size: float
    direction: str  # "up" or "down"
    fifty_percent_level: float
    fully_mitigated: bool
    half_mitigated: bool
    mitigation_level: float


def detect_gap_mitigation(
    candles: Sequence[Candle],
    mitigation_threshold: float = 0.5,
) -> Optional[GapMitigation]:
    """Detect opening gap and its mitigation status.
    
    Args:
        candles: Price candle sequence  
        mitigation_threshold: What % of gap must be filled (default 50%)
    
    Returns:
        GapMitigation status if gap detected
    """
    if len(candles) < 5:
        return None
    
    # Look for gap in recent candles
    for i in range(1, min(10, len(candles))):
        prev_close = candles[-i - 1].close
        current_open = candles[-i].open
        
        gap_size = abs(current_open - prev_close)
        
        # Minimum gap threshold (0.1% of price)
        if gap_size < prev_close * 0.001:
            continue
        
        direction = "up" if current_open > prev_close else "down"
        fifty_percent = prev_close + (current_open - prev_close) * 0.5
        
        # Check mitigation in subsequent candles
        subsequent = candles[-i:]
        
        if direction == "up":
            # Gap up - check if price came back down to fill
            lowest = min(c.low for c in subsequent)
            mitigation_level = lowest
            fill_pct = (current_open - lowest) / gap_size if gap_size > 0 else 0
        else:
            # Gap down - check if price came back up to fill
            highest = max(c.high for c in subsequent)
            mitigation_level = highest
            fill_pct = (highest - current_open) / gap_size if gap_size > 0 else 0
        
        half_mitigated = fill_pct >= 0.5
        fully_mitigated = fill_pct >= 1.0
        
        return GapMitigation(
            gap_start=prev_close,
            gap_end=current_open,
            gap_size=gap_size,
            direction=direction,
            fifty_percent_level=fifty_percent,
            fully_mitigated=fully_mitigated,
            half_mitigated=half_mitigated,
            mitigation_level=mitigation_level,
        )
    
    return None


__all__ = [
    # Core data structures
    "Candle",
    "SwingPoint",
    "FairValueGap",
    "OrderBlock",
    "LiquiditySweep",
    "BreakerBlock",
    "MitigationBlock",
    "Killzone",
    "OTE",
    "InstitutionalCandle",
    "AMDPattern",
    "AsianRange",
    "PremiumDiscount",
    "PowerOf3",
    "TimeWindow",
    "LiquidityPool",
    # NEW: Advanced ICT structures
    "CRTPattern",
    "CISDSignal",
    "ATRDisplacement",
    "MomentumFVG",
    "SessionRange",
    "GapMitigation",
    # Constants
    "ICT_KILLZONES",
    "ICT_OPTIMAL_WINDOWS",
    "SESSION_DEFINITIONS",
    # Basic utilities
    "load_candles",
    "active_session",
    "active_killzone",
    "is_silver_bullet_window",
    "rolling_atr",
    "displacement_score",
    "calculate_atr",
    "calculate_displacement",
    # Liquidity detection
    "detect_liquidity_sweep",
    "identify_liquidity_pools",
    "nearest_liquidity_pool",
    # Block detection
    "detect_breaker_blocks",
    "detect_mitigation_blocks",
    "detect_order_blocks",
    # OTE functions
    "calculate_ote",
    "price_in_ote",
    "find_ote_from_swings",
    # Institutional detection
    "detect_institutional_candles",
    "detect_amd_pattern",
    # FVG functions
    "detect_fair_value_gaps",
    "detect_unfilled_fvg",
    "fvg_consequent_encroachment",
    "detect_momentum_fvgs",
    "get_strongest_fvg",
    "calculate_rsi",
    # Asian Range
    "calculate_asian_range",
    "asian_range_swept",
    # Premium/Discount
    "calculate_premium_discount",
    "price_in_premium",
    "price_in_discount",
    "price_zone",
    # Power of 3
    "detect_power_of_3",
    # Time-based
    "get_entry_quality",
    # NEW: CRT Pattern
    "detect_crt_pattern",
    # NEW: CISD Detection
    "detect_cisd",
    # NEW: ATR Displacement
    "classify_atr_displacement",
    "is_judas_swing",
    "is_real_displacement",
    # NEW: Session Ranges
    "calculate_session_range",
    "get_all_session_ranges",
    "check_session_sweep",
    # NEW: Gap Mitigation
    "detect_gap_mitigation",
    # Structure analysis
    "swing_points",
    "recent_structure",
    "detect_bos",
    "detect_mss",
    # Helpers
    "equity_curve_stats",
    "microstructure_path",
    "floor_to_minute",
    "pip_distance",
]