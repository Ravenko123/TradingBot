"""Broker implementations for live MT5 trading and simulation."""

from __future__ import annotations

import random
from typing import Dict, Optional

from config.settings import Instrument, SETTINGS
from core.order_types import Fill, IOCOrder
from core.utils import Candle
from core.logger import get_logger

# MT5 import
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

trade_logger = get_logger("trades")
system_logger = get_logger("system")


# =============================================================================
# SYMBOL MAPPING - Must match mt5_client.py
# =============================================================================
SYMBOL_MAP: Dict[str, str] = {
    "BTCUSD": "BTCUSD",
    "USDJPY": "USDJPY+",
    "XAUUSD": "XAUUSD+",
    "GBPJPY": "GBPJPY+",
    "EURUSD": "EURUSD+",
    "GBPUSD": "GBPUSD+",
}


def get_mt5_symbol(internal_symbol: str) -> str:
    """Convert internal symbol to MT5 broker symbol."""
    normalized = internal_symbol.upper().strip()
    if normalized in SYMBOL_MAP:
        return SYMBOL_MAP[normalized]
    if "BTC" in normalized:
        return normalized
    return normalized + "+"


def has_open_position(symbol: str) -> bool:
    """Check if there's an open position for this symbol on MT5."""
    if mt5 is None:
        return False
    
    mt5_symbol = get_mt5_symbol(symbol)
    positions = mt5.positions_get(symbol=mt5_symbol)
    
    if positions is None or len(positions) == 0:
        return False
    
    return True


def get_open_position_info(symbol: str) -> Optional[Dict]:
    """Get info about open position for this symbol."""
    if mt5 is None:
        return None
    
    mt5_symbol = get_mt5_symbol(symbol)
    positions = mt5.positions_get(symbol=mt5_symbol)
    
    if positions is None or len(positions) == 0:
        return None
    
    pos = positions[0]  # Take first position for this symbol
    return {
        "ticket": pos.ticket,
        "symbol": pos.symbol,
        "type": "buy" if pos.type == 0 else "sell",
        "volume": pos.volume,
        "price_open": pos.price_open,
        "sl": pos.sl,
        "tp": pos.tp,
        "profit": pos.profit,
    }


# =============================================================================
# MT5 LIVE BROKER - REAL TRADES
# =============================================================================

class MT5Broker:
    """REAL MT5 broker that sends actual orders to MetaTrader 5.
    
    This broker uses mt5.order_send() to execute REAL trades.
    Make sure MT5 is connected before using this broker.
    """

    def __init__(
        self,
        instrument: Instrument,
        *,
        magic_number: int = 123456,
        deviation: int = 20,
        risk_percent: float = 0.025,  # 2.5% default
    ) -> None:
        self.instrument = instrument
        self.magic_number = magic_number
        self.deviation = deviation  # Max slippage in points
        self.risk_percent = risk_percent
        
        if mt5 is None:
            raise RuntimeError("MetaTrader5 package not installed!")
    
    def get_account_balance(self) -> float:
        """Get REAL account balance from MT5."""
        info = mt5.account_info()
        if info is None:
            return 0.0
        return info.balance
    
    def get_account_equity(self) -> float:
        """Get REAL account equity from MT5."""
        info = mt5.account_info()
        if info is None:
            return 0.0
        return info.equity
    
    def calculate_lot_size(self, symbol: str, entry: float, stop_loss: float) -> float:
        """Calculate proper lot size based on REAL account balance and risk %.
        
        This calculates lots directly, not units.
        Rounds normally (0.234 -> 0.23, 0.235 -> 0.24), minimum 0.01.
        """
        balance = self.get_account_balance()
        if balance <= 0:
            return 0.01
        
        # Risk amount in dollars
        risk_amount = balance * self.risk_percent
        
        # Stop loss distance in price
        sl_distance = abs(entry - stop_loss)
        if sl_distance <= 0:
            return 0.01
        
        # Get symbol info for pip value calculation
        mt5_symbol = get_mt5_symbol(symbol)
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            mt5.symbol_select(mt5_symbol, True)
            info = mt5.symbol_info(mt5_symbol)
        
        if info is None:
            return 0.01  # Minimum fallback
        
        # Calculate pip value per lot
        tick_size = info.trade_tick_size
        tick_value = info.trade_tick_value
        
        if tick_size <= 0 or tick_value <= 0:
            return 0.01
        
        # How many ticks is our SL?
        sl_ticks = sl_distance / tick_size
        
        # Value lost per lot if SL is hit
        loss_per_lot = sl_ticks * tick_value
        
        if loss_per_lot <= 0:
            return 0.01
        
        # Lots needed for our risk amount
        lots = risk_amount / loss_per_lot
        
        # Round to 2 decimal places (standard rounding)
        lots = round(lots, 2)
        
        # Enforce minimum 0.01
        lots = max(0.01, lots)
        
        # Enforce broker max if available
        if info.volume_max > 0:
            lots = min(lots, info.volume_max)
        
        system_logger.info(
            f"[LOT CALC] Balance=${balance:.2f} Risk={self.risk_percent*100:.1f}% "
            f"RiskAmt=${risk_amount:.2f} SL={sl_distance:.2f} -> {lots:.2f} lots"
        )
        
        return lots

    def _get_symbol_info(self, symbol: str):
        """Get MT5 symbol info for lot size calculations."""
        mt5_symbol = get_mt5_symbol(symbol)
        info = mt5.symbol_info(mt5_symbol)
        if info is None:
            # Try to enable the symbol first
            mt5.symbol_select(mt5_symbol, True)
            info = mt5.symbol_info(mt5_symbol)
        return info

    def _get_current_price(self, symbol: str, side: str) -> Optional[float]:
        """Get current bid/ask price from MT5."""
        mt5_symbol = get_mt5_symbol(symbol)
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            return None
        return tick.ask if side == "buy" else tick.bid

    def _units_to_lots(self, symbol: str, units: float) -> float:
        """Convert unit quantity to MT5 lots.
        
        For forex: 1 lot = 100,000 units
        For gold: 1 lot = 100 oz
        For BTC: 1 lot = 1 BTC (usually)
        """
        info = self._get_symbol_info(symbol)
        if info is None:
            # Fallback defaults
            if "XAU" in symbol:
                contract_size = 100  # 100 oz per lot
            elif "BTC" in symbol:
                contract_size = 1  # 1 BTC per lot
            else:
                contract_size = 100000  # Standard forex
        else:
            contract_size = info.trade_contract_size
        
        lots = units / contract_size
        
        # Round to symbol's lot step
        if info and info.volume_step > 0:
            lots = round(lots / info.volume_step) * info.volume_step
        else:
            lots = round(lots, 2)
        
        # Enforce min/max lots
        if info:
            lots = max(info.volume_min, min(lots, info.volume_max))
        else:
            lots = max(0.01, min(lots, 100.0))
        
        return lots

    async def execute_ioc(self, order: IOCOrder, candle: Candle) -> Fill:
        """Execute a REAL order on MT5.
        
        This sends an actual market order to your MT5 broker!
        Uses CURRENT LIVE PRICE from MT5 tick data.
        SL/TP are calculated as OFFSETS from the signal, applied to current price.
        """
        mt5_symbol = get_mt5_symbol(order.instrument)
        
        # Enable symbol if not visible
        if not mt5.symbol_select(mt5_symbol, True):
            return Fill(0.0, None, "rejected", f"symbol_not_found: {mt5_symbol}")
        
        # Get CURRENT LIVE price from MT5
        tick = mt5.symbol_info_tick(mt5_symbol)
        if tick is None:
            return Fill(0.0, None, "rejected", "no_tick_data")
        
        # Use CURRENT price - this is the LIVE price, not some old candle
        if order.side.value == "buy":
            order_type = mt5.ORDER_TYPE_BUY
            current_price = tick.ask
        else:
            order_type = mt5.ORDER_TYPE_SELL
            current_price = tick.bid
        
        # =================================================================
        # Calculate SL/TP based on CURRENT PRICE
        # The signal provides risk/reward structure, we apply it to live price
        # =================================================================
        if order.entry_price is not None and order.stop_loss is not None:
            # Calculate the RISK (distance from entry to SL) from the signal
            signal_risk = abs(order.entry_price - order.stop_loss)
            
            # Calculate R:R ratio from signal
            if order.take_profit is not None:
                signal_reward = abs(order.take_profit - order.entry_price)
                rr_ratio = signal_reward / signal_risk if signal_risk > 0 else 3.0
            else:
                rr_ratio = 3.0
            
            # Apply the same risk/reward to CURRENT price
            if order.side.value == "buy":
                live_sl = current_price - signal_risk
                live_tp = current_price + (signal_risk * rr_ratio)
            else:
                live_sl = current_price + signal_risk
                live_tp = current_price - (signal_risk * rr_ratio)
            
            # Calculate lots using CURRENT price and live SL
            lots = self.calculate_lot_size(order.instrument, current_price, live_sl)
            
            # Update order with live prices
            order.stop_loss = live_sl
            order.take_profit = live_tp
            order.entry_price = current_price
        else:
            # Fallback to old method if no entry/SL provided
            lots = self._units_to_lots(order.instrument, order.quantity)
            live_sl = order.stop_loss
            live_tp = order.take_profit
        
        # Ensure minimum 0.01
        lots = max(0.01, round(lots, 2))
        
        if lots <= 0:
            return Fill(0.0, None, "rejected", "invalid_lot_size")
        
        # Get symbol digits for price normalization
        info = mt5.symbol_info(mt5_symbol)
        digits = info.digits if info else 5
        
        # Normalize prices to proper decimal places
        price_normalized = round(current_price, digits)
        sl_normalized = round(order.stop_loss, digits) if order.stop_loss else None
        tp_normalized = round(order.take_profit, digits) if order.take_profit else None
        
        # Get minimum stop distance in points
        stop_level = info.trade_stops_level if info else 0
        point = info.point if info else 0.00001
        
        # Also get current spread (SL/TP often need to be at least spread away)
        current_spread = tick.ask - tick.bid
        spread_in_points = current_spread / point if point > 0 else 0
        
        # Use the LARGER of: stops_level, spread * 2, or 30 points minimum
        # Add extra buffer to account for price movement between calculation and execution
        min_points = max(stop_level, spread_in_points * 2, 30)
        min_stop_distance = min_points * point
        
        system_logger.debug(
            f"[STOPS] {mt5_symbol}: stops_level={stop_level} spread={spread_in_points:.1f} "
            f"-> using min_dist={min_stop_distance:.5f}"
        )
        
        # Validate and fix SL/TP relative to CURRENT price (not signal entry price)
        # For BUY: SL must be BELOW price, TP must be ABOVE price
        # For SELL: SL must be ABOVE price, TP must be BELOW price
        if order.side.value == "buy":
            # BUY order
            if sl_normalized is not None:
                max_sl = price_normalized - min_stop_distance
                if sl_normalized >= max_sl:
                    # SL is too close or above entry - adjust it down
                    old_sl = sl_normalized
                    sl_normalized = round(max_sl - point, digits)
                    print(f"      [ADJUST] SL {old_sl} -> {sl_normalized} (must be below {max_sl:.{digits}f})")
            if tp_normalized is not None:
                min_tp = price_normalized + min_stop_distance
                if tp_normalized <= min_tp:
                    old_tp = tp_normalized
                    tp_normalized = round(min_tp + point, digits)
                    print(f"      [ADJUST] TP {old_tp} -> {tp_normalized} (must be above {min_tp:.{digits}f})")
        else:
            # SELL order
            if sl_normalized is not None:
                min_sl = price_normalized + min_stop_distance
                if sl_normalized <= min_sl:
                    # SL is too close or below entry - adjust it up
                    old_sl = sl_normalized
                    sl_normalized = round(min_sl + point, digits)
                    print(f"      [ADJUST] SL {old_sl} -> {sl_normalized} (must be above {min_sl:.{digits}f})")
            if tp_normalized is not None:
                max_tp = price_normalized - min_stop_distance
                if tp_normalized >= max_tp:
                    old_tp = tp_normalized
                    tp_normalized = round(max_tp - point, digits)
                    print(f"      [ADJUST] TP {old_tp} -> {tp_normalized} (must be below {max_tp:.{digits}f})")
        
        # Build the order request WITH SL and TP
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": mt5_symbol,
            "volume": lots,
            "type": order_type,
            "price": price_normalized,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "ULTIMA2.0",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        # Add SL and TP to the order (normalized)
        if sl_normalized is not None:
            request["sl"] = sl_normalized
        if tp_normalized is not None:
            request["tp"] = tp_normalized
        
        # Log the order attempt
        system_logger.info(
            f"[MT5] Sending ORDER: {order.side.value.upper()} {lots:.2f} lots {mt5_symbol} @ {price_normalized} SL={sl_normalized} TP={tp_normalized}"
        )
        print(f"\n[MT5] >>> SENDING REAL ORDER <<<")
        print(f"      Symbol: {mt5_symbol}")
        print(f"      Side: {order.side.value.upper()}")
        print(f"      Lots: {lots:.2f}")
        print(f"      Price: {price_normalized}")
        print(f"      SL: {sl_normalized}")
        print(f"      TP: {tp_normalized}")
        
        # SEND THE ORDER!
        result = mt5.order_send(request)
        
        if result is None:
            error = mt5.last_error()
            system_logger.error(f"[MT5] Order failed: {error}")
            print(f"      [FAILED] Error: {error}")
            return Fill(0.0, None, "rejected", f"mt5_error: {error}")
        
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            system_logger.error(f"[MT5] Order rejected: {result.retcode} - {result.comment}")
            print(f"      [REJECTED] Code: {result.retcode} - {result.comment}")
            return Fill(0.0, None, "rejected", f"retcode_{result.retcode}: {result.comment}")
        
        # SUCCESS!
        filled_price = result.price
        filled_volume = result.volume
        ticket = result.order  # MT5 ticket number
        
        # Convert lots back to units for internal tracking
        info = self._get_symbol_info(order.instrument)
        contract_size = info.trade_contract_size if info else 100000
        filled_units = filled_volume * contract_size
        
        system_logger.info(
            f"[MT5] ORDER FILLED: {filled_volume:.2f} lots @ {filled_price:.5f} | Ticket: {ticket}"
        )
        print(f"      [FILLED] {filled_volume:.2f} lots @ {filled_price:.5f}")
        print(f"      [TICKET] {ticket}")
        
        trade_logger.info(
            f"MT5 TRADE | {order.side.value.upper()} | {mt5_symbol} | "
            f"Lots={filled_volume:.2f} | Price={filled_price:.5f} | Ticket={ticket}"
        )
        
        return Fill(
            filled_quantity=filled_units,
            price=filled_price,
            status="filled",
            reason=None,
            ticket=ticket,
        )

    def get_positions(self) -> list:
        """Get all open positions from MT5."""
        positions = mt5.positions_get()
        return list(positions) if positions else []

    def close_position(self, ticket: int) -> bool:
        """Close a specific position by ticket number."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        symbol = pos.symbol
        volume = pos.volume
        
        # Opposite order to close
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "ULTIMA2.0 close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        return result and result.retcode == mt5.TRADE_RETCODE_DONE

    def modify_position_sl(self, ticket: int, new_sl: float) -> bool:
        """Modify the stop loss of an existing position (for trailing SL)."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": pos.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": pos.tp,  # Keep existing TP
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            system_logger.info(f"[MT5] SL modified to {new_sl:.5f} for ticket {ticket}")
            print(f"      [TRAIL] SL moved to {new_sl:.5f}")
            return True
        else:
            system_logger.warning(f"[MT5] Failed to modify SL: {result.comment if result else 'no result'}")
            return False

    def partial_close(self, ticket: int, lots_to_close: float) -> bool:
        """Close a portion of a position (for partial profit taking)."""
        position = mt5.positions_get(ticket=ticket)
        if not position:
            return False
        
        pos = position[0]
        symbol = pos.symbol
        
        # Can't close more than we have
        lots_to_close = min(lots_to_close, pos.volume)
        lots_to_close = round(lots_to_close, 2)
        
        if lots_to_close < 0.01:
            return False
        
        # Opposite order to close partial
        if pos.type == mt5.POSITION_TYPE_BUY:
            order_type = mt5.ORDER_TYPE_SELL
            price = mt5.symbol_info_tick(symbol).bid
        else:
            order_type = mt5.ORDER_TYPE_BUY
            price = mt5.symbol_info_tick(symbol).ask
        
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": lots_to_close,
            "type": order_type,
            "position": ticket,
            "price": price,
            "deviation": self.deviation,
            "magic": self.magic_number,
            "comment": "ULTIMA2.0 partial",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }
        
        result = mt5.order_send(request)
        if result and result.retcode == mt5.TRADE_RETCODE_DONE:
            system_logger.info(f"[MT5] Partial close: {lots_to_close:.2f} lots at {result.price:.5f}")
            print(f"      [PARTIAL] Closed {lots_to_close:.2f} lots @ {result.price:.5f}")
            return True
        else:
            system_logger.warning(f"[MT5] Partial close failed: {result.comment if result else 'no result'}")
            return False


# =============================================================================
# SIMULATED BROKER - For backtesting only
# =============================================================================

class SimulatedBroker:
    """Simulated broker for backtesting. Does NOT execute real trades."""

    def __init__(
        self,
        instrument: Instrument,
        *,
        spread: float | None = None,
        slippage: float | None = None,
    ) -> None:
        self.instrument = instrument
        self.exec_cfg = SETTINGS.execution
        self._spread = spread if spread is not None else self.exec_cfg.default_spread
        self._slippage = slippage if slippage is not None else self.exec_cfg.default_slippage

    async def execute_ioc(self, order: IOCOrder, candle: Candle) -> Fill:
        """Simulate IOC order execution (for backtesting only)."""

        volume_units = candle.volume
        if volume_units <= 0:
            volume_units = self.instrument.liquidity_per_min
        elif volume_units < self.instrument.contract_size:
            volume_units *= self.instrument.contract_size

        available = min(self.instrument.liquidity_per_min, volume_units)
        available *= self.exec_cfg.min_liquidity_ratio
        min_floor = self.instrument.contract_size * self.exec_cfg.min_liquidity_ratio
        available = max(available, min_floor)
        if available <= 0:
            return Fill(filled_quantity=0.0, price=None, status="rejected", reason="no_liquidity")

        fill_qty = min(order.quantity, available)
        fill_ratio = fill_qty / order.quantity if order.quantity else 0.0
        if fill_ratio < self.exec_cfg.partial_fill_threshold:
            return Fill(filled_quantity=0.0, price=None, status="cancelled", reason="insufficient_liquidity")

        spread_half = self._spread / 2
        slip = self._slippage * random.uniform(0.5, 1.5)
        mid = candle.mid
        if order.side.value == "buy":
            price = min(candle.high, mid + spread_half + slip)
        else:
            price = max(candle.low, mid - spread_half - slip)

        if order.limit_price is not None:
            if order.side.value == "buy" and price > order.limit_price:
                return Fill(0.0, None, "cancelled", "limit_price_exceeded")
            if order.side.value == "sell" and price < order.limit_price:
                return Fill(0.0, None, "cancelled", "limit_price_exceeded")

        return Fill(filled_quantity=fill_qty, price=price, status="filled", reason=None)


__all__ = ["SimulatedBroker", "MT5Broker", "get_mt5_symbol"]