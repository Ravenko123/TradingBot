"""
ULTIMA 2.0 - ICT/SMC Live Trading Bot
======================================
Main entry point for LIVE TRADING with MetaTrader 5.

Features:
- Real-time MT5 candle scanning
- Multi-instrument trading  
- Telegram control integration
- Full ICT/SMC strategy execution
- Risk management and position tracking

Usage:
    python bot/main.py                      # Run live with defaults
    python bot/main.py --telegram           # Run with Telegram control
    python bot/main.py --instruments BTCUSD XAUUSD  # Trade specific pairs
    python bot/main.py --mode backtest --data data.csv  # Backtest mode
"""

from __future__ import annotations

import argparse
import asyncio
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from backtest.engine import BacktestEngine, run_backtest_sync
from backtest.metrics import export_equity_curve, export_report_json, export_trades_csv, plot_equity_curve
from config.settings import SETTINGS, Instrument
from config.strategy_config import (
    STRATEGY_NAME,
    STRATEGY_PARAMS,
    DEFAULT_INSTRUMENT,
    DEFAULT_RISK_PERCENT,
    TIMEFRAME,
    get_params_for_instrument,
    get_timeframe_for_instrument,
)
from core.broker import SimulatedBroker, MT5Broker
from core.datafeed import CSVDataFeed
from core.execution import ExecutionEngine
from core.logger import configure_logging, get_logger
from core.mt5_client import (
    MT5NotAvailable,
    ensure_initialized as ensure_mt5_initialized,
    fetch_candles as fetch_mt5_candles,
    shutdown as shutdown_mt5,
)
from core.utils import Candle
from strategies import StrategyContext, StrategyRegistry

# Telegram integration (optional)
try:
    from core.telegram_bot import TelegramController, create_telegram_controller
    TELEGRAM_AVAILABLE = True
except ImportError:
    TELEGRAM_AVAILABLE = False
    TelegramController = None  # type: ignore
    create_telegram_controller = None  # type: ignore

# Type alias for optional Telegram controller
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from core.telegram_bot import TelegramController as _TelegramController
    TelegramControllerType = _TelegramController
else:
    TelegramControllerType = object


system_logger = get_logger("system")
error_logger = get_logger("errors")


# =============================================================================
# CONFIGURATION
# =============================================================================

# How often to scan for new candles (seconds)
SCAN_INTERVAL = 1

# Lookback for initial candle history
LOOKBACK_CANDLES = 200

# All tradeable instruments
ALL_TRADEABLE = ["BTCUSD", "USDJPY", "XAUUSD", "GBPJPY", "EURUSD", "GBPUSD"]


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="ULTIMA 2.0 - ICT/SMC Trading Bot (Live & Backtest)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python bot/main.py                                    # Live trade BTCUSD, USDJPY, XAUUSD
  python bot/main.py --telegram                         # With Telegram control
  python bot/main.py --instruments BTCUSD               # Trade only BTCUSD
  python bot/main.py --dry-run                          # Simulate without real trades
  python bot/main.py --mode backtest --data data.csv   # Backtest on historical data
        """
    )
    # Live trading arguments
    parser.add_argument(
        "--instruments", 
        nargs="+", 
        default=["BTCUSD", "USDJPY", "XAUUSD", "GBPJPY", "EURUSD", "GBPUSD"],
        help="Instruments to trade (default: all 6 instruments)"
    )
    parser.add_argument(
        "--timeframe",
        default=TIMEFRAME,
        help=f"Candle timeframe (default: {TIMEFRAME})"
    )
    parser.add_argument(
        "--scan-interval",
        type=float,
        default=SCAN_INTERVAL,
        help=f"Seconds between scans (default: {SCAN_INTERVAL})"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without executing real trades (simulation mode)"
    )
    parser.add_argument(
        "--mt5-path",
        default=None,
        help="Path to MT5 terminal (auto-detect if not specified)"
    )
    
    # Shared arguments
    parser.add_argument("--mode", choices=["live", "backtest"], default="live", help="Execution mode")
    parser.add_argument("--risk", type=float, default=DEFAULT_RISK_PERCENT, help="Risk percent per trade")
    parser.add_argument("--telegram", action="store_true", help="Enable Telegram control")
    parser.add_argument("--telegram-settings", default="telegram_settings.json", help="Path to Telegram settings")
    
    # Backtest-only arguments
    parser.add_argument("--data", default="data/sample_data.csv", help="CSV file for backtest mode")
    parser.add_argument("--instrument", default=DEFAULT_INSTRUMENT, help="Instrument symbol (backtest mode)")
    parser.add_argument("--strategy", default=STRATEGY_NAME, help="Strategy name")
    parser.add_argument("--export", help="Equity curve export path (backtest mode)")
    parser.add_argument("--plot", help="PNG path for equity chart (backtest mode)")
    parser.add_argument("--trades-csv", help="Export trades to CSV (backtest mode)")
    parser.add_argument("--report-json", help="Export JSON report (backtest mode)")
    parser.add_argument("--spread", type=float, help="Override broker spread")
    parser.add_argument("--slippage", type=float, help="Override broker slippage")
    
    return parser


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_next_candle_time(timeframe: str) -> datetime:
    """Calculate when the next candle will close."""
    now = datetime.now()
    
    # Parse timeframe (M1, M5, M15, H1, etc.)
    tf_map = {
        "M1": 1, "M2": 2, "M3": 3, "M5": 5, "M10": 10, "M15": 15, "M30": 30,
        "H1": 60, "H2": 120, "H4": 240, "D1": 1440,
    }
    minutes = tf_map.get(timeframe.upper(), 5)
    
    # Calculate next candle close time
    current_minute = now.hour * 60 + now.minute
    next_candle_minute = ((current_minute // minutes) + 1) * minutes
    next_hour = next_candle_minute // 60
    next_min = next_candle_minute % 60
    
    next_close = now.replace(hour=next_hour % 24, minute=next_min, second=0, microsecond=0)
    if next_close <= now:
        next_close = next_close.replace(hour=(next_hour + 24) % 24)
    
    return next_close


def format_countdown(seconds: int) -> str:
    """Format seconds as MM:SS countdown."""
    mins = seconds // 60
    secs = seconds % 60
    return f"{mins}:{secs:02d}"


# =============================================================================
# INSTRUMENT RUNNER
# =============================================================================

class InstrumentRunner:
    """Manages strategy and execution for a single instrument."""
    
    def __init__(
        self,
        symbol: str,
        timeframe: str,
        risk_percent: float,
        strategy_params: Dict,
        telegram: Optional[Any] = None,
        dry_run: bool = False,
    ):
        self.symbol = symbol
        self.timeframe = timeframe
        self.risk_percent = risk_percent
        self.telegram = telegram
        self.dry_run = dry_run
        
        # Get instrument config
        self.instrument = self._get_instrument(symbol)
        
        # Initialize broker - USE REAL MT5 BROKER FOR LIVE TRADING!
        if dry_run:
            self.broker = SimulatedBroker(self.instrument)
            print(f"  [{symbol}] Using SIMULATED broker (dry-run mode)")
        else:
            self.broker = MT5Broker(self.instrument)
            print(f"  [{symbol}] Using REAL MT5 broker - LIVE TRADES!")
        
        self.engine = ExecutionEngine(
            self.instrument,
            broker=self.broker,
            risk_per_trade=risk_percent / 100.0,
        )
        
        # Initialize strategy
        context = StrategyContext(
            instrument=symbol,
            risk_per_trade=risk_percent / 100.0,
            session_windows=SETTINGS.sessions,
            parameters=strategy_params,
        )
        self.strategy = StrategyRegistry.create(STRATEGY_NAME, context)
        
        # Candle history
        self.candles: List[Candle] = []
        self.last_candle_time: Optional[datetime] = None
        
        system_logger.info(f"[INIT] {symbol} | TF={timeframe} Risk={risk_percent}%")
    
    def _get_instrument(self, symbol: str) -> Instrument:
        """Get instrument from settings or create default."""
        if symbol in SETTINGS.allowed_instruments:
            return SETTINGS.allowed_instruments[symbol]
        
        # Default instrument config
        return Instrument(
            symbol=symbol,
            pip_value=10.0,
            tick_size=0.0001,
            contract_size=100_000.0,
            base_currency="USD",
            liquidity_per_min=10_000_000,
            margin_rate=0.02,
        )
    
    def load_history(self, mt5_path: Optional[str] = None) -> None:
        """Load initial candle history from MT5 and warm up strategy."""
        try:
            days_needed = max(5, LOOKBACK_CANDLES // 288 + 1)
            candles = fetch_mt5_candles(
                self.symbol,
                self.timeframe,
                days=days_needed,
                terminal_path=mt5_path,
            )
            
            self.candles = candles[-LOOKBACK_CANDLES:] if len(candles) > LOOKBACK_CANDLES else candles
            
            if self.candles:
                self.last_candle_time = self.candles[-1].timestamp
                system_logger.info(
                    f"[DATA] {self.symbol}: Loaded {len(self.candles)} candles | "
                    f"Last: {self.last_candle_time}"
                )
                
                # Warm up strategy with historical candles (except the last one)
                # This builds up the strategy's internal state (swing points, order blocks, etc.)
                warmup_candles = self.candles[:-1] if len(self.candles) > 1 else []
                for candle in warmup_candles:
                    # Feed candle to strategy without acting on signals (warming up)
                    self.strategy.on_candle(candle)
                
                print(f"  [WARMUP] {self.symbol}: Processed {len(warmup_candles)} historical candles")
            else:
                system_logger.warning(f"[WARN] {self.symbol}: No candles loaded!")
                
        except Exception as e:
            error_logger.error(f"[ERROR] {self.symbol}: Failed to load history: {e}")
            raise
    
    async def check_initial_setup(self) -> None:
        """Check the most recent candle for an immediate trading setup."""
        if not self.candles:
            return
        
        # Check if Telegram controls allow trading
        if self.telegram:
            if not self.telegram.is_trading_allowed():
                return
            if not self.telegram.is_instrument_enabled(self.symbol):
                return
        
        # Process the latest candle for a potential signal
        latest_candle = self.candles[-1]
        print(f"  [CHECK] {self.symbol}: Checking latest candle for setup...")
        await self.process_candle(latest_candle)
    
    def fetch_new_candles(self, mt5_path: Optional[str] = None) -> List[Candle]:
        """Fetch any new candles since last check."""
        try:
            candles = fetch_mt5_candles(
                self.symbol,
                self.timeframe,
                days=1,
                terminal_path=mt5_path,
            )
            
            if not candles:
                return []
            
            new_candles = []
            for candle in candles:
                if self.last_candle_time is None or candle.timestamp > self.last_candle_time:
                    new_candles.append(candle)
                    self.candles.append(candle)
            
            # Trim history
            if len(self.candles) > LOOKBACK_CANDLES * 2:
                self.candles = self.candles[-LOOKBACK_CANDLES:]
            
            if new_candles:
                self.last_candle_time = new_candles[-1].timestamp
                
            return new_candles
            
        except Exception as e:
            error_logger.warning(f"[WARN] {self.symbol}: Error fetching candles: {e}")
            return []
    
    async def process_candle(self, candle: Candle) -> None:
        """Process a single candle through the strategy."""
        try:
            # Check Telegram controls
            if self.telegram:
                if not self.telegram.is_trading_allowed():
                    return
                if not self.telegram.is_instrument_enabled(self.symbol):
                    return
                
                # Update risk dynamically
                new_risk = self.telegram.get_risk_per_trade() / 100.0
                if new_risk != self.engine.risk.risk_per_trade:
                    self.engine.risk.risk_per_trade = new_risk
            
            # Generate signal
            signal = self.strategy.on_candle(candle)
            
            if signal:
                system_logger.info(
                    f"[SIGNAL] {self.symbol}: {signal.side.value.upper()} | "
                    f"Entry={signal.entry:.5f} SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f}"
                )
                
                if not self.dry_run:
                    await self.engine.handle_signal(signal, candle)
                    
                    if self.telegram and self.engine.position:
                        self.telegram.notify_trade(
                            instrument=self.symbol,
                            side=signal.side.value,
                            entry=signal.entry,
                            sl=signal.stop_loss,
                            tp=signal.take_profit,
                        )
            
            # Update position (check SL/TP)
            trade = self.engine.update_position(candle)
            
            if trade:
                result = "WIN" if trade.pnl > 0 else "LOSS"
                system_logger.info(
                    f"[{result}] {self.symbol}: CLOSED | PnL=${trade.pnl:+.2f} R={trade.r_multiple:+.2f}"
                )
                
                if self.telegram:
                    self.telegram.notify_close(
                        instrument=trade.instrument,
                        side=trade.side.value,
                        pnl=trade.pnl,
                        r_multiple=trade.r_multiple,
                    )
                    
        except Exception as e:
            error_logger.error(f"[ERROR] {self.symbol}: Error processing candle: {e}")


# =============================================================================
# LIVE TRADING LOOP
# =============================================================================

async def run_live_trading(
    instruments: List[str],
    timeframe: str,
    risk_percent: float,
    scan_interval: float,
    telegram: Optional[Any],
    mt5_path: Optional[str],
    dry_run: bool,
) -> None:
    """Main live trading loop with MT5."""
    
    # Initialize MT5
    system_logger.info("[MT5] Connecting to MetaTrader 5...")
    try:
        ensure_mt5_initialized(path=mt5_path)
        system_logger.info("[OK] MT5 connected!")
    except Exception as e:
        error_logger.error(f"[ERROR] Failed to connect to MT5: {e}")
        raise
    
    # Create runners for each instrument with per-instrument params
    runners: Dict[str, InstrumentRunner] = {}
    for symbol in instruments:
        try:
            # Get instrument-specific params (OPTIMIZED per instrument)
            instrument_params = get_params_for_instrument(symbol)
            instrument_tf = get_timeframe_for_instrument(symbol)
            
            # NOTE: We NO LONGER override rr_target and min_confluence from Telegram
            # Each instrument has its own optimized values from backtesting:
            # BTCUSD: RR=2.2, XAUUSD: RR=4.0, USDJPY: RR=1.7, etc.
            # Telegram can still control: pause, risk%, enabled instruments
            
            runner = InstrumentRunner(
                symbol=symbol,
                timeframe=instrument_tf,  # Use per-instrument timeframe!
                risk_percent=risk_percent,
                strategy_params=instrument_params,  # Use per-instrument params!
                telegram=telegram,
                dry_run=dry_run,
            )
            runner.load_history(mt5_path)
            runners[symbol] = runner
            print(f"  [{symbol}] TF={instrument_tf}, RR={instrument_params['rr_target']}, Conf={instrument_params['min_confluence']}")
        except Exception as e:
            error_logger.error(f"[ERROR] Failed to initialize {symbol}: {e}")
    
    if not runners:
        error_logger.error("[ERROR] No instruments initialized! Exiting.")
        return
    
    symbols_str = ", ".join(runners.keys())
    system_logger.info(f"[LIVE] TRADING STARTED | Instruments: {symbols_str}")
    
    if telegram:
        telegram.broadcast(
            f"*BOT STARTED*\n\n"
            f"Mode: {'DRY RUN' if dry_run else 'LIVE'}\n"
            f"Instruments: {symbols_str}\n"
            f"Timeframe: {timeframe}\n"
            f"Risk: {risk_percent:.1f}%"
        )
    
    # Main loop - check for entries EVERY SECOND
    scan_count = 0
    last_entry_time = time.time()  # Track time since last entry
    
    try:
        while True:
            loop_start = time.time()
            scan_count += 1
            
            # Check Telegram emergency stop
            if telegram and telegram.state.emergency_stop:
                system_logger.warning("[ALERT] Emergency stop - pausing...")
                await asyncio.sleep(scan_interval)
                continue
            
            # Check Telegram pause state
            if telegram and not telegram.is_trading_allowed():
                await asyncio.sleep(scan_interval)
                continue
            
            # UPDATE RISK FROM TELEGRAM ON EVERY SCAN
            # This ensures risk changes take effect immediately
            if telegram:
                current_risk = telegram.get_risk_per_trade() / 100.0
                for runner in runners.values():
                    if runner.engine.risk._risk_per_trade != current_risk:
                        runner.engine.risk._risk_per_trade = current_risk
                        system_logger.info(f"[RISK] Updated to {current_risk*100:.1f}%")
            
            # Fetch latest candles and check for signals on EVERY scan
            signals_found = 0
            positions_updated = 0
            
            # Periodic status log (every 60 seconds)
            if scan_count % 60 == 0:
                system_logger.info(f"[SCAN] {scan_count} scans completed, checking all instruments...")
            
            for symbol, runner in runners.items():
                if telegram and not telegram.is_instrument_enabled(symbol):
                    continue
                
                # Always fetch latest candles (includes current forming candle)
                new_candles = runner.fetch_new_candles(mt5_path)
                
                # Check MT5 for existing position (sync with broker)
                from core.broker import has_open_position, get_open_position_info
                mt5_has_position = has_open_position(symbol)
                
                # Sync engine position state with MT5
                if mt5_has_position and runner.engine.position is None:
                    # MT5 has position but engine doesn't know - skip signals
                    pass
                elif not mt5_has_position and runner.engine.position is not None:
                    # Position was closed on MT5 (manually or by SL/TP)
                    print(f"\n[CLOSED] {symbol}: Position closed on MT5")
                    system_logger.info(f"[CLOSED] {symbol}: Position closed on MT5")
                    runner.engine.position = None
                
                # Process new candles into strategy (for history)
                for candle in new_candles:
                    runner.strategy.on_candle(candle)
                
                # ALWAYS check fresh candle from MT5 for signals (even if no "new" candles)
                # This ensures we catch signals on forming candles that update in real-time
                signal = None
                if not mt5_has_position:
                    # Get latest candle fresh from MT5
                    mt5_symbol = {"BTCUSD": "BTCUSD", "XAUUSD": "XAUUSD+", "USDJPY": "USDJPY+", 
                                  "GBPJPY": "GBPJPY+", "EURUSD": "EURUSD+", "GBPUSD": "GBPUSD+"}.get(symbol, symbol)
                    import MetaTrader5 as mt5
                    tf_map = {"M1": mt5.TIMEFRAME_M1, "M5": mt5.TIMEFRAME_M5, "M15": mt5.TIMEFRAME_M15}
                    tf = tf_map.get(runner.timeframe, mt5.TIMEFRAME_M5)
                    rates = mt5.copy_rates_from_pos(mt5_symbol, tf, 0, 1)
                    if rates is not None and len(rates) > 0:
                        r = rates[0]
                        from core.datafeed import Candle
                        from datetime import datetime
                        fresh_candle = Candle(
                            timestamp=datetime.fromtimestamp(r['time']),
                            open=r['open'], high=r['high'], low=r['low'], close=r['close'],
                            volume=r['tick_volume']
                        )
                        # Update the last candle in runner's buffer with fresh data
                        if runner.candles and runner.candles[-1].timestamp == fresh_candle.timestamp:
                            runner.candles[-1] = fresh_candle
                        elif runner.candles and fresh_candle.timestamp > runner.candles[-1].timestamp:
                            # New candle started, add it
                            runner.candles.append(fresh_candle)
                            runner.last_candle_time = fresh_candle.timestamp
                        
                        # Check for signal
                        signal = runner.strategy.on_candle(fresh_candle)
                
                # If we got a signal, execute it
                if signal and not mt5_has_position:
                    signals_found += 1
                    current_risk_pct = runner.engine.risk._risk_per_trade * 100
                    print(f"\n[SIGNAL] {symbol}: {signal.side.value.upper()} @ {signal.entry:.5f}")
                    print(f"         SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f} Risk={current_risk_pct:.1f}%")
                    
                    # Check daily drawdown limit (from Telegram control)
                    if telegram and telegram.is_daily_drawdown_enabled():
                        max_loss = telegram.get_max_daily_loss_pct()
                        current_pnl_pct = (telegram.state.pnl_today / SETTINGS.initial_balance) * 100
                        if current_pnl_pct <= -max_loss:
                            print(f"         [SKIP] Daily drawdown limit hit: {current_pnl_pct:.1f}% <= -{max_loss:.1f}%")
                            system_logger.warning(f"[RISK] Daily drawdown limit blocked trade: {current_pnl_pct:.1f}%")
                            continue
                    
                    system_logger.info(
                        f"[SIGNAL] {symbol}: {signal.side.value.upper()} | "
                        f"Entry={signal.entry:.5f} SL={signal.stop_loss:.5f} TP={signal.take_profit:.5f} Risk={current_risk_pct:.1f}%"
                    )
                    
                    if not runner.dry_run:
                        # Use the latest candle from our buffer for execution
                        exec_candle = runner.candles[-1] if runner.candles else None
                        if exec_candle:
                            await runner.engine.handle_signal(signal, exec_candle)
                            
                            if runner.engine.position:
                                last_entry_time = time.time()  # Reset timer!
                                print(f"         [OK] Position opened!")
                                
                                if runner.telegram:
                                    runner.telegram.notify_trade(
                                        instrument=symbol,
                                        side=signal.side.value,
                                        entry=signal.entry,
                                        sl=signal.stop_loss,
                                        tp=signal.take_profit,
                                    )
                
                # Only update trailing SL on MT5 - DO NOT simulate exits
                # Real exits are detected by MT5 position disappearing (above)
                if runner.engine.position and runner.candles and mt5_has_position:
                    latest_candle = runner.candles[-1]
                    # Just update trailing - ignore the returned trade (that's simulated)
                    runner.engine.update_position(latest_candle)
                    positions_updated += 1
            
            # Calculate time since last entry
            seconds_waiting = int(time.time() - last_entry_time)
            wait_mins = seconds_waiting // 60
            wait_secs = seconds_waiting % 60
            
            # Count active positions
            active_positions = sum(1 for r in runners.values() if r.engine.position is not None)
            
            # Display status
            now = datetime.now().strftime("%H:%M:%S")
            if active_positions > 0:
                status_msg = f"\r[{now}] {active_positions} position(s) open | Monitoring...              "
            else:
                # Show scan count every 60 scans (about once per minute)
                scan_indicator = "." * (scan_count % 4)
                status_msg = f"\r[{now}] Scanning{scan_indicator} ({wait_mins}m {wait_secs:02d}s)              "
            
            sys.stdout.write(status_msg)
            sys.stdout.flush()
            
            # Sleep until next scan
            elapsed = time.time() - loop_start
            sleep_time = max(0.1, scan_interval - elapsed)
            await asyncio.sleep(sleep_time)
            
    except KeyboardInterrupt:
        system_logger.info("[STOP] Shutdown signal received")
    finally:
        shutdown_mt5()
        system_logger.info("[MT5] Disconnected")
        
        if telegram:
            total_pnl = sum(r.engine.balance - SETTINGS.initial_balance for r in runners.values())
            telegram.broadcast(
                f"*BOT STOPPED*\n\nSession P&L: ${total_pnl:+,.2f}"
            )


# Global for signal handlers
_telegram: Optional[Any] = None


async def run_live(
    data_path: str,
    instrument_key: str,
    strategy_name: str,
    risk_percent: float,
    speed: float,
    loop_data: bool,
    max_candles: Optional[int],
    spread: Optional[float],
    slippage: Optional[float],
    telegram: Optional[Any] = None,
) -> None:
    """Legacy CSV-based simulation (kept for compatibility)."""
    global _telegram
    _telegram = telegram
    
    instrument = SETTINGS.allowed_instruments[instrument_key]
    broker = SimulatedBroker(instrument, spread=spread, slippage=slippage)
    
    # Get risk from Telegram if available, otherwise use CLI arg
    if telegram:
        risk_percent = telegram.get_risk_per_trade()
    
    # Convert risk percent to decimal (2.5% -> 0.025)
    risk_per_trade = risk_percent / 100.0
    
    engine = ExecutionEngine(
        instrument, 
        broker=broker,
        risk_per_trade=risk_per_trade,
    )
    
    # Build strategy params - merge shared config with Telegram overrides
    strategy_params = STRATEGY_PARAMS.copy()
    if telegram:
        strategy_params["rr_target"] = telegram.get_rr_target()
        strategy_params["min_confluence"] = telegram.get_min_confluence()
    
    context = StrategyContext(
        instrument=instrument.symbol,
        risk_per_trade=risk_per_trade,
        session_windows=SETTINGS.sessions,
        parameters=strategy_params,
    )
    strategy = StrategyRegistry.create(strategy_name, context)

    system_logger.info(
        "Starting live mode | instrument=%s strategy=%s risk=%.2f%%",
        instrument_key, strategy_name, risk_percent
    )
    
    if telegram:
        telegram.broadcast(
            f"*BOT STARTED*\n\n"
            f"Instrument: {instrument_key}\n"
            f"Risk: {risk_percent:.1f}%\n"
            f"R:R: {strategy_params['rr_target']:.1f}:1"
        )

    feed = CSVDataFeed(
        path=data_path,
        speed=speed,
        loop=loop_data,
        max_candles=max_candles,
    )

    async for candle in feed:
        # Check Telegram controls
        if telegram:
            # Check if trading is paused/stopped
            if not telegram.is_trading_allowed():
                continue
            
            # Check if this instrument is enabled
            if not telegram.is_instrument_enabled(instrument_key):
                continue
            
            # Update risk dynamically
            new_risk = telegram.get_risk_per_trade() / 100.0
            if new_risk != engine.risk.risk_per_trade:
                engine.risk.risk_per_trade = new_risk
        
        signal_result = strategy.on_candle(candle)
        if signal_result:
            await engine.handle_signal(signal_result, candle)
            
            # Notify Telegram about new trade
            if telegram and engine.position:
                telegram.notify_trade(
                    instrument=instrument_key,
                    side=signal_result.side.value,
                    entry=signal_result.entry,
                    sl=signal_result.stop_loss,
                    tp=signal_result.take_profit,
                )
        
        # Check for closed trades
        prev_position = engine.position
        trade = engine.update_position(candle)
        
        # Notify Telegram about closed trade
        if trade and telegram:
            telegram.notify_close(
                instrument=trade.instrument,
                side=trade.side.value,
                pnl=trade.pnl,
                r_multiple=trade.r_multiple,
            )

    system_logger.info("Live session completed | balance=%.2f", engine.balance)
    
    if telegram:
        telegram.broadcast(f"*SESSION COMPLETE*\n\nFinal balance: ${engine.balance:,.2f}")


# =============================================================================
# MAIN ENTRY POINT
# =============================================================================

def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    
    # Always try to initialize Telegram (it's a core feature)
    telegram = None
    if TELEGRAM_AVAILABLE:
        try:
            # Look for settings in bot/ folder first
            settings_path = args.telegram_settings
            if not Path(settings_path).exists():
                # Try bot/ subfolder
                bot_path = Path("bot") / settings_path
                if bot_path.exists():
                    settings_path = str(bot_path)
            
            telegram = create_telegram_controller(settings_path)
            telegram.start()
            print("[OK] Telegram control enabled")
        except FileNotFoundError:
            print("[WARN] telegram_settings.json not found - Telegram disabled")
        except Exception as e:
            print(f"[WARN] Telegram init failed: {e}")
    else:
        print("[WARN] Telegram not available. Install: pip install requests")
    
    try:
        if args.mode == "backtest":
            # BACKTEST MODE - use CSV data
            engine = BacktestEngine(
                data_path=args.data,
                instrument=args.instrument,
                strategy_name=args.strategy,
                spread=args.spread,
                slippage=args.slippage,
                strategy_params=dict(STRATEGY_PARAMS),  # Use shared config!
            )
            result = run_backtest_sync(engine)
            system_logger.info("Backtest complete | instrument=%s strategy=%s", args.instrument, args.strategy)
            print("\n[REPORT] Performance Report:")
            for key, value in result.report.as_dict().items():
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
            if result.session_stats:
                print("\n[SESSIONS] Session Stats:")
                for session, stats in result.session_stats.items():
                    win_rate = (stats.wins / stats.trades * 100) if stats.trades else 0.0
                    avg_r = stats.total_r / stats.trades if stats.trades else 0.0
                    print(
                        f"  {session}: trades={stats.trades} win_rate={win_rate:.2f}% avg_R={avg_r:.2f} pnl={stats.pnl:.2f}"
                    )
            if args.export:
                export_equity_curve(args.export, result.timestamps, result.equity_curve)
                print(f"\nEquity curve exported to {args.export}")
            if args.plot:
                plot_equity_curve(args.plot, result.timestamps, result.equity_curve)
                print(f"Equity chart saved to {args.plot}")
            if args.trades_csv:
                export_trades_csv(args.trades_csv, result.trades)
                print(f"Trades exported to {args.trades_csv}")
            if args.report_json:
                export_report_json(args.report_json, result.report, result.session_stats)
                print(f"JSON report saved to {args.report_json}")
        else:
            # LIVE MODE - use MT5 real-time data
            valid_instruments = []
            for symbol in args.instruments:
                symbol = symbol.upper()
                if symbol in ALL_TRADEABLE:
                    valid_instruments.append(symbol)
                else:
                    system_logger.warning(f"[WARN] Unknown instrument: {symbol}")
            
            if not valid_instruments:
                error_logger.error("[ERROR] No valid instruments specified!")
                return
            
            # Print startup banner - use risk from Telegram if available
            actual_risk = telegram.get_risk_per_trade() if telegram else args.risk
            print("=" * 60)
            print("[BOT] ULTIMA 2.0 - ICT/SMC LIVE TRADING BOT")
            print("=" * 60)
            print(f"   Instruments : {', '.join(valid_instruments)}")
            print(f"   Timeframe   : {args.timeframe}")
            print(f"   Risk/Trade  : {actual_risk}%")
            print(f"   Scan Every  : {args.scan_interval}s")
            print(f"   Telegram    : {'ON' if telegram else 'OFF'}")
            print(f"   Mode        : {'DRY RUN' if args.dry_run else 'LIVE TRADING'}")
            print("=" * 60)
            print()
            
            if not args.dry_run:
                print("[!] LIVE TRADING MODE - Real trades will be executed!")
                print("    Press Ctrl+C to stop")
                print()
            
            asyncio.run(
                run_live_trading(
                    instruments=valid_instruments,
                    timeframe=args.timeframe,
                    risk_percent=actual_risk,  # Use Telegram risk, not args.risk
                    scan_interval=args.scan_interval,
                    telegram=telegram,
                    mt5_path=args.mt5_path,
                    dry_run=args.dry_run,
                )
            )
            
    except KeyboardInterrupt:
        system_logger.info("Received shutdown signal")
        if telegram:
            telegram.broadcast("*BOT STOPPED*\n\nManual shutdown received.")
    except MT5NotAvailable as e:
        error_logger.error(f"[ERROR] MT5 not available: {e}")
        print("\n[ERROR] MetaTrader 5 is not running or not installed!")
        print("   1. Make sure MT5 is installed and running")
        print("   2. Install MT5 Python package: pip install MetaTrader5")
    except Exception:
        error_logger.exception("Fatal error while running mode=%s", args.mode)
        if telegram:
            telegram.broadcast("*ERROR*\n\nBot crashed! Check logs.")
        raise
    finally:
        if telegram:
            telegram.stop()


if __name__ == "__main__":  # pragma: no cover
    main()