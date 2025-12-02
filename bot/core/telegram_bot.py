"""
ULTIMA 2.0 - Telegram Control Integration
==========================================
Complete Telegram control for the ICT/SMC trading bot.

Features:
- Instrument enable/disable (BTCUSD, USDJPY, XAUUSD, GBPJPY, EURUSD, GBPUSD)
- Risk management controls
- Live status monitoring
- Emergency stop/resume
- Performance tracking
- Strategy parameter tuning

Author: Ravenko123
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set

logger = logging.getLogger(__name__)


# All available instruments (even if disabled by default)
ALL_INSTRUMENTS = {
    "BTCUSD": {"name": "Bitcoin/USD", "enabled_default": True},
    "USDJPY": {"name": "USD/JPY", "enabled_default": True},
    "XAUUSD": {"name": "Gold/USD", "enabled_default": True},
    "GBPJPY": {"name": "GBP/JPY", "enabled_default": False},
    "EURUSD": {"name": "EUR/USD", "enabled_default": False},
    "GBPUSD": {"name": "GBP/USD", "enabled_default": False},
}


@dataclass
class TelegramState:
    """Persistent state for Telegram control."""
    
    # Trading control
    is_paused: bool = False
    emergency_stop: bool = False
    emergency_stop_time: Optional[str] = None
    
    # Enabled instruments
    enabled_instruments: Set[str] = field(default_factory=lambda: {"BTCUSD", "USDJPY", "XAUUSD"})
    
    # Risk settings (overrides)
    risk_per_trade: float = 2.5  # Percentage
    max_daily_loss_pct: float = 5.0  # Percentage
    max_drawdown_pct: float = 15.0  # Max drawdown before stopping
    daily_drawdown_enabled: bool = True  # Whether daily drawdown limit is active
    
    # Strategy overrides
    rr_target: float = 3.0
    min_confluence: int = 4
    
    # Session tracking
    trades_today: int = 0
    pnl_today: float = 0.0
    last_trade_time: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "is_paused": self.is_paused,
            "emergency_stop": self.emergency_stop,
            "emergency_stop_time": self.emergency_stop_time,
            "enabled_instruments": list(self.enabled_instruments),
            "risk_per_trade": self.risk_per_trade,
            "max_daily_loss_pct": self.max_daily_loss_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "daily_drawdown_enabled": self.daily_drawdown_enabled,
            "rr_target": self.rr_target,
            "min_confluence": self.min_confluence,
            "trades_today": self.trades_today,
            "pnl_today": self.pnl_today,
            "last_trade_time": self.last_trade_time,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TelegramState":
        state = cls()
        state.is_paused = data.get("is_paused", False)
        state.emergency_stop = data.get("emergency_stop", False)
        state.emergency_stop_time = data.get("emergency_stop_time")
        state.enabled_instruments = set(data.get("enabled_instruments", ["BTCUSD", "USDJPY", "XAUUSD"]))
        state.risk_per_trade = data.get("risk_per_trade", 2.5)
        state.max_daily_loss_pct = data.get("max_daily_loss_pct", 5.0)
        state.max_drawdown_pct = data.get("max_drawdown_pct", 15.0)
        state.daily_drawdown_enabled = data.get("daily_drawdown_enabled", True)
        state.rr_target = data.get("rr_target", 3.0)
        state.min_confluence = data.get("min_confluence", 4)
        state.trades_today = data.get("trades_today", 0)
        state.pnl_today = data.get("pnl_today", 0.0)
        state.last_trade_time = data.get("last_trade_time")
        return state


class TelegramController:
    """
    Telegram control interface for the ICT/SMC trading bot.
    
    Commands:
    - /status - Show bot status
    - /instruments - Show/toggle instruments
    - /enable <symbol> - Enable instrument
    - /disable <symbol> - Disable instrument
    - /risk <percent> - Set risk per trade
    - /pause - Pause trading
    - /resume - Resume trading
    - /stop - Emergency stop
    - /performance - Show performance
    - /help - Show commands
    """
    
    def __init__(
        self,
        token: str,
        allowed_user_ids: Iterable[int],
        state_path: str = "telegram_state.json",
        poll_interval: float = 1.0,  # Fast polling for instant responses
    ) -> None:
        try:
            import requests as _requests
        except ImportError as exc:
            raise RuntimeError(
                "Telegram requires 'requests' package. Install: pip install requests"
            ) from exc
        
        self._requests = _requests
        self._api_base = f"https://api.telegram.org/bot{token.strip()}"
        self._allowed_user_ids: Set[int] = {int(uid) for uid in allowed_user_ids if uid}
        self._poll_interval = max(1.0, float(poll_interval))
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._update_offset: Optional[int] = None
        
        # State management - use absolute path relative to this file's directory
        # This ensures we always use the same state file regardless of CWD
        bot_dir = Path(__file__).parent.parent  # Go up from core/ to bot/
        self._state_path = bot_dir / state_path
        self.state = self._load_state()
        
        # Command queue for main bot to consume
        self.command_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        
        logger.info("📱 Telegram Controller initialized")
    
    def _load_state(self) -> TelegramState:
        """Load state from disk."""
        if self._state_path.exists():
            try:
                with open(self._state_path, 'r') as f:
                    return TelegramState.from_dict(json.load(f))
            except Exception as e:
                logger.warning(f"Failed to load state: {e}")
        return TelegramState()
    
    def _save_state(self) -> None:
        """Save state to disk."""
        try:
            with open(self._state_path, 'w') as f:
                json.dump(self.state.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save state: {e}")
    
    # =========================================================================
    # LIFECYCLE
    # =========================================================================
    
    def start(self) -> None:
        """Start Telegram polling."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(
            target=self._run_loop, 
            name="TelegramPolling", 
            daemon=True
        )
        self._thread.start()
        logger.info("📱 Telegram polling started")
    
    def stop(self, timeout: float = 5.0) -> None:
        """Stop Telegram polling."""
        self._stop_event.set()
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=timeout)
        logger.info("📱 Telegram polling stopped")
    
    # =========================================================================
    # MESSAGING
    # =========================================================================
    
    def send_message(self, chat_id: int, text: str, parse_mode: str = "Markdown") -> bool:
        """Send message to a chat."""
        payload = {"chat_id": chat_id, "text": text, "parse_mode": parse_mode}
        try:
            response = self._requests.post(
                f"{self._api_base}/sendMessage",
                data=payload,
                timeout=10,
            )
            return response.status_code < 400
        except Exception as e:
            logger.error(f"Failed to send message: {e}")
            return False
    
    def broadcast(self, text: str) -> None:
        """Broadcast message to all allowed users."""
        for user_id in self._allowed_user_ids:
            self.send_message(user_id, text)
    
    def notify_trade(self, instrument: str, side: str, entry: float, sl: float, tp: float) -> None:
        """Notify about a new trade."""
        msg = (
            f"🔔 *NEW TRADE*\n\n"
            f"📈 {instrument} {side.upper()}\n"
            f"Entry: {entry:.5f}\n"
            f"SL: {sl:.5f}\n"
            f"TP: {tp:.5f}\n"
            f"R:R: {abs(tp - entry) / abs(entry - sl):.1f}:1"
        )
        self.broadcast(msg)
    
    def notify_close(self, instrument: str, side: str, pnl: float, r_multiple: float) -> None:
        """Notify about a closed trade."""
        icon = "✅" if pnl > 0 else "❌"
        msg = (
            f"{icon} *TRADE CLOSED*\n\n"
            f"📈 {instrument} {side.upper()}\n"
            f"P&L: ${pnl:+.2f}\n"
            f"R: {r_multiple:+.2f}R"
        )
        self.broadcast(msg)
        
        # Update state
        self.state.trades_today += 1
        self.state.pnl_today += pnl
        self.state.last_trade_time = datetime.now().isoformat()
        self._save_state()
    
    # =========================================================================
    # COMMAND HANDLERS
    # =========================================================================
    
    def _handle_command(self, chat_id: int, command: str, args: List[str]) -> None:
        """Process a command and send response."""
        
        handlers = {
            "start": self._cmd_start,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "instruments": self._cmd_instruments,
            "enable": self._cmd_enable,
            "disable": self._cmd_disable,
            "risk": self._cmd_risk,
            "drawdown": self._cmd_drawdown,
            "rr": self._cmd_rr,
            "confluence": self._cmd_confluence,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "stop": self._cmd_stop,
            "performance": self._cmd_performance,
            "reset_daily": self._cmd_reset_daily,
        }
        
        handler = handlers.get(command, self._cmd_unknown)
        response = handler(args)
        self.send_message(chat_id, response)
    
    def _cmd_start(self, args: List[str]) -> str:
        return (
            "🤖 *ULTIMA 2.0 ICT/SMC Bot*\n\n"
            "Welcome! Use /help to see available commands.\n\n"
            "Quick start:\n"
            "• /status - Bot status\n"
            "• /instruments - Manage pairs\n"
            "• /risk 2.5 - Set risk %"
        )
    
    def _cmd_help(self, args: List[str]) -> str:
        return (
            "📱 *TELEGRAM COMMANDS*\n\n"
            "*Status:*\n"
            "• /status - Bot status\n"
            "• /performance - Today's P&L\n\n"
            "*Instruments:*\n"
            "• /instruments - List all\n"
            "• /enable BTCUSD - Enable pair\n"
            "• /disable EURUSD - Disable pair\n\n"
            "*Risk:*\n"
            "• /risk - Show risk setting\n"
            "• /risk 2.5 - Set 2.5% risk\n"
            "• /drawdown - Show/toggle daily limit\n"
            "• /drawdown 5.0 - Set 5% daily limit\n"
            "• /drawdown off - Disable limit\n"
            "• /rr - Show per-instrument R:R\n"
            "• /confluence - Show per-instrument conf\n\n"
            "*Control:*\n"
            "• /pause - Pause trading\n"
            "• /resume - Resume trading\n"
            "• /stop - Emergency stop\n\n"
            "*Maintenance:*\n"
            "• /reset\\_daily - Reset daily stats"
        )
    
    def _cmd_status(self, args: List[str]) -> str:
        status_icon = "🔴" if self.state.emergency_stop else ("⏸️" if self.state.is_paused else "🟢")
        status_text = "STOPPED" if self.state.emergency_stop else ("PAUSED" if self.state.is_paused else "RUNNING")
        
        enabled = ", ".join(sorted(self.state.enabled_instruments)) or "None"
        
        return (
            f"🤖 *BOT STATUS*\n\n"
            f"Status: {status_icon} {status_text}\n\n"
            f"*Instruments:*\n{enabled}\n\n"
            f"*Risk Settings:*\n"
            f"• Risk/trade: {self.state.risk_per_trade:.1f}%\n"
            f"• Daily limit: {'✅ ' + str(self.state.max_daily_loss_pct) + '%' if self.state.daily_drawdown_enabled else '❌ OFF'}\n"
            f"• R:R target: {self.state.rr_target:.1f}:1\n"
            f"• Min confluence: {self.state.min_confluence}\n\n"
            f"*Today:*\n"
            f"• Trades: {self.state.trades_today}\n"
            f"• P&L: ${self.state.pnl_today:+.2f}"
        )
    
    def _cmd_instruments(self, args: List[str]) -> str:
        lines = ["📊 *INSTRUMENTS*\n"]
        
        for symbol, info in ALL_INSTRUMENTS.items():
            enabled = symbol in self.state.enabled_instruments
            icon = "✅" if enabled else "❌"
            lines.append(f"{icon} {symbol} - {info['name']}")
        
        lines.append("\n*Commands:*")
        lines.append("• /enable BTCUSD")
        lines.append("• /disable EURUSD")
        
        return "\n".join(lines)
    
    def _cmd_enable(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: /enable <SYMBOL>\nExample: /enable BTCUSD"
        
        symbol = args[0].upper()
        if symbol not in ALL_INSTRUMENTS:
            valid = ", ".join(ALL_INSTRUMENTS.keys())
            return f"❌ Unknown instrument: {symbol}\nValid: {valid}"
        
        if symbol in self.state.enabled_instruments:
            return f"ℹ️ {symbol} is already enabled"
        
        self.state.enabled_instruments.add(symbol)
        self._save_state()
        
        return f"✅ {symbol} enabled!\n\nActive: {', '.join(sorted(self.state.enabled_instruments))}"
    
    def _cmd_disable(self, args: List[str]) -> str:
        if not args:
            return "❌ Usage: /disable <SYMBOL>\nExample: /disable EURUSD"
        
        symbol = args[0].upper()
        if symbol not in ALL_INSTRUMENTS:
            valid = ", ".join(ALL_INSTRUMENTS.keys())
            return f"❌ Unknown instrument: {symbol}\nValid: {valid}"
        
        if symbol not in self.state.enabled_instruments:
            return f"ℹ️ {symbol} is already disabled"
        
        self.state.enabled_instruments.discard(symbol)
        self._save_state()
        
        remaining = ", ".join(sorted(self.state.enabled_instruments)) or "None"
        return f"❌ {symbol} disabled!\n\nActive: {remaining}"
    
    def _cmd_risk(self, args: List[str]) -> str:
        if not args:
            return (
                f"📊 *RISK SETTINGS*\n\n"
                f"• Risk/trade: {self.state.risk_per_trade:.1f}%\n"
                f"• Max daily loss: {self.state.max_daily_loss_pct:.1f}%\n"
                f"• Max drawdown: {self.state.max_drawdown_pct:.1f}%\n\n"
                f"Set: /risk 2.5"
            )
        
        try:
            new_risk = float(args[0])
            if new_risk < 0.1 or new_risk > 10:
                return "❌ Risk must be between 0.1% and 10%"
            
            old_risk = self.state.risk_per_trade
            self.state.risk_per_trade = new_risk
            self._save_state()
            
            return f"✅ Risk updated!\n   {old_risk:.1f}% → {new_risk:.1f}%"
        except ValueError:
            return "❌ Invalid number. Example: /risk 2.5"
    
    def _cmd_drawdown(self, args: List[str]) -> str:
        """Control daily drawdown limit - enable/disable or set custom limit."""
        if not args:
            # Show current status
            status = "✅ ENABLED" if self.state.daily_drawdown_enabled else "❌ DISABLED"
            return (
                f"📊 *DAILY DRAWDOWN LIMIT*\n\n"
                f"Status: {status}\n"
                f"Limit: {self.state.max_daily_loss_pct:.1f}%\n\n"
                f"*Commands:*\n"
                f"• /drawdown on - Enable limit\n"
                f"• /drawdown off - Disable limit\n"
                f"• /drawdown 5.0 - Set 5% limit"
            )
        
        arg = args[0].lower()
        
        # Enable/Disable
        if arg in ("on", "enable", "yes", "true"):
            self.state.daily_drawdown_enabled = True
            self._save_state()
            return f"✅ Daily drawdown limit ENABLED\nLimit: {self.state.max_daily_loss_pct:.1f}%"
        
        if arg in ("off", "disable", "no", "false"):
            self.state.daily_drawdown_enabled = False
            self._save_state()
            return "⚠️ Daily drawdown limit DISABLED\n\n_Warning: No daily loss protection!_"
        
        # Set custom limit
        try:
            new_limit = float(arg)
            if new_limit < 1.0 or new_limit > 50.0:
                return "❌ Limit must be between 1% and 50%"
            
            old_limit = self.state.max_daily_loss_pct
            self.state.max_daily_loss_pct = new_limit
            self.state.daily_drawdown_enabled = True  # Auto-enable when setting limit
            self._save_state()
            
            return f"✅ Daily drawdown limit updated!\n   {old_limit:.1f}% → {new_limit:.1f}%\n\nStatus: ✅ ENABLED"
        except ValueError:
            return "❌ Invalid input.\n\nExamples:\n• /drawdown 5.0\n• /drawdown on\n• /drawdown off"
    
    def _cmd_rr(self, args: List[str]) -> str:
        # NOTE: R:R is now per-instrument from backtesting optimization
        # This command is kept for reference but doesn't override per-instrument settings
        return (
            "ℹ️ *R:R is now per-instrument (optimized)*\n\n"
            "• BTCUSD: 2.2:1\n"
            "• XAUUSD: 4.0:1\n"
            "• USDJPY: 1.7:1\n"
            "• GBPJPY: 3.0:1\n"
            "• EURUSD: 1.5:1\n"
            "• GBPUSD: 1.7:1\n\n"
            "_These are optimized from backtesting_"
        )
    
    def _cmd_confluence(self, args: List[str]) -> str:
        # NOTE: Confluence is now per-instrument from backtesting optimization
        # This command is kept for reference but doesn't override per-instrument settings
        return (
            "ℹ️ *Confluence is now per-instrument (optimized)*\n\n"
            "• BTCUSD: 4\n"
            "• XAUUSD: 3\n"
            "• USDJPY: 1\n"
            "• GBPJPY: 1\n"
            "• EURUSD: 2\n"
            "• GBPUSD: 2\n\n"
            "_These are optimized from backtesting_"
        )
    
    def _cmd_pause(self, args: List[str]) -> str:
        if self.state.is_paused:
            return "ℹ️ Bot is already paused"
        
        self.state.is_paused = True
        self._save_state()
        
        return "⏸️ *TRADING PAUSED*\n\nNo new trades will be opened.\nUse /resume to continue."
    
    def _cmd_resume(self, args: List[str]) -> str:
        if self.state.emergency_stop:
            return "🚨 Emergency stop is active!\nClear with /resume after checking positions."
        
        was_paused = self.state.is_paused or self.state.emergency_stop
        self.state.is_paused = False
        self.state.emergency_stop = False
        self.state.emergency_stop_time = None
        self._save_state()
        
        if was_paused:
            return "▶️ *TRADING RESUMED*\n\nBot is now active!"
        return "ℹ️ Bot was already running"
    
    def _cmd_stop(self, args: List[str]) -> str:
        self.state.emergency_stop = True
        self.state.emergency_stop_time = datetime.now().isoformat()
        self._save_state()
        
        return (
            "🚨 *EMERGENCY STOP ACTIVATED*\n\n"
            "All trading STOPPED immediately!\n"
            "Open positions should be reviewed.\n\n"
            "Use /resume to restart."
        )
    
    def _cmd_performance(self, args: List[str]) -> str:
        win_icon = "📈" if self.state.pnl_today >= 0 else "📉"
        
        return (
            f"📊 *TODAY'S PERFORMANCE*\n\n"
            f"Trades: {self.state.trades_today}\n"
            f"P&L: {win_icon} ${self.state.pnl_today:+.2f}\n\n"
            f"Last trade: {self.state.last_trade_time or 'None'}"
        )
    
    def _cmd_reset_daily(self, args: List[str]) -> str:
        self.state.trades_today = 0
        self.state.pnl_today = 0.0
        self._save_state()
        
        return "✅ Daily stats reset!"
    
    def _cmd_unknown(self, args: List[str]) -> str:
        return "❓ Unknown command. Use /help to see available commands."
    
    # =========================================================================
    # POLLING LOOP
    # =========================================================================
    
    def _run_loop(self) -> None:
        """Main polling loop."""
        while not self._stop_event.is_set():
            self._poll_once()
            self._stop_event.wait(self._poll_interval)
    
    def _poll_once(self) -> None:
        """Poll for new messages."""
        params: Dict[str, Any] = {"timeout": 10}
        if self._update_offset is not None:
            params["offset"] = self._update_offset
        
        try:
            response = self._requests.get(
                f"{self._api_base}/getUpdates",
                params=params,
                timeout=15,
            )
        except Exception as e:
            logger.warning(f"Telegram poll error: {e}")
            return
        
        try:
            data = response.json()
        except ValueError:
            return
        
        if not data.get("ok", False):
            return
        
        for update in data.get("result", []):
            update_id = update.get("update_id")
            if isinstance(update_id, int):
                self._update_offset = update_id + 1
            
            message = update.get("message") or {}
            chat = message.get("chat") or {}
            chat_id = chat.get("id")
            sender = message.get("from") or {}
            user_id = sender.get("id")
            
            if not chat_id or not user_id:
                continue
            
            # Check authorization
            if self._allowed_user_ids and user_id not in self._allowed_user_ids:
                logger.warning(f"Unauthorized Telegram user: {user_id}")
                continue
            
            text = (message.get("text") or "").strip()
            if not text.startswith("/"):
                continue
            
            # Parse command
            parts = text.split()
            command = parts[0][1:].lower().split("@")[0]
            args = parts[1:]
            
            # Handle command
            self._handle_command(chat_id, command, args)
    
    # =========================================================================
    # PUBLIC GETTERS FOR BOT
    # =========================================================================
    
    def is_trading_allowed(self) -> bool:
        """Check if trading is allowed."""
        return not self.state.emergency_stop and not self.state.is_paused
    
    def is_instrument_enabled(self, symbol: str) -> bool:
        """Check if an instrument is enabled."""
        return symbol in self.state.enabled_instruments
    
    def get_enabled_instruments(self) -> Set[str]:
        """Get set of enabled instruments."""
        return self.state.enabled_instruments.copy()
    
    def get_risk_per_trade(self) -> float:
        """Get risk per trade as percentage."""
        return self.state.risk_per_trade
    
    def get_rr_target(self) -> float:
        """Get R:R target."""
        return self.state.rr_target
    
    def get_min_confluence(self) -> int:
        """Get minimum confluence score."""
        return self.state.min_confluence
    
    def is_daily_drawdown_enabled(self) -> bool:
        """Check if daily drawdown limit is enabled."""
        return self.state.daily_drawdown_enabled
    
    def get_max_daily_loss_pct(self) -> float:
        """Get max daily loss percentage."""
        return self.state.max_daily_loss_pct


def load_telegram_settings(path: str = "telegram_settings.json") -> Dict[str, Any]:
    """Load Telegram settings from JSON file."""
    settings_path = Path(path)
    
    # Check multiple locations
    search_paths = [
        settings_path,
        Path(__file__).parent.parent.parent / "telegram_settings.json",
        Path.cwd() / "telegram_settings.json",
    ]
    
    for p in search_paths:
        if p.exists():
            with open(p, 'r') as f:
                return json.load(f)
    
    raise FileNotFoundError(f"telegram_settings.json not found in: {search_paths}")


def create_telegram_controller(settings_path: str = "telegram_settings.json") -> TelegramController:
    """Create TelegramController from settings file."""
    settings = load_telegram_settings(settings_path)
    
    return TelegramController(
        token=settings["bot_token"],
        allowed_user_ids=settings.get("allowed_user_ids", []),
    )


__all__ = [
    "TelegramController",
    "TelegramState",
    "ALL_INSTRUMENTS",
    "load_telegram_settings",
    "create_telegram_controller",
]
