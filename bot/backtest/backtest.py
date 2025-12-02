"""Straightforward run harness for ICT SMC backtests."""

from __future__ import annotations

import argparse
import atexit
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backtest.engine import BacktestEngine, run_backtest_sync  # type: ignore  # noqa: E402
from backtest.metrics import (  # type: ignore  # noqa: E402
    export_equity_curve,
    export_report_json,
    export_trades_csv,
    plot_equity_curve,
)
from config.settings import SETTINGS  # type: ignore  # noqa: E402
from config.strategy_config import (  # type: ignore  # noqa: E402
    STRATEGY_NAME,
    STRATEGY_PARAMS,
    DEFAULT_INSTRUMENT,
    DEFAULT_RISK_PERCENT,
    INITIAL_BALANCE,
    BACKTEST_DAYS,
    TIMEFRAME,
    get_params_for_instrument,
    get_timeframe_for_instrument,
)
from core.logger import configure_logging, get_logger  # type: ignore  # noqa: E402
from core.mt5_client import (  # type: ignore  # noqa: E402
    MT5NotAvailable,
    ensure_initialized as ensure_mt5_initialized,
    fetch_candles as fetch_mt5_candles,
    shutdown as shutdown_mt5,
)
from core.utils import Candle, load_candles  # type: ignore  # noqa: E402


# ---------------------------------------------------------------------------
# QUICK KNOBS — edit these without hunting through the file.
# ---------------------------------------------------------------------------
DATA_PATH: Optional[Path] = None  # Set to a CSV for offline mode, leave None for MT5.
USE_MT5 = True
MT5_TERMINAL_PATH: Optional[str] = None  # Point to terminal64.exe if auto-discovery fails.
INSTRUMENTS = list(SETTINGS.allowed_instruments.keys())

# These now come from shared config (config/strategy_config.py):
# - STRATEGY_NAME
# - STRATEGY_PARAMS  
# - DEFAULT_INSTRUMENT
# - DEFAULT_RISK_PERCENT
# - INITIAL_BALANCE
# - BACKTEST_DAYS
# - TIMEFRAME

DAYS = BACKTEST_DAYS
SPREAD = None
SLIPPAGE = None
LOG_LEVEL = "INFO"
AUTO_EXPORT = False
EXPORT_DIR = SETTINGS.backtest.export_dir


def _percent_to_fraction(value: float) -> float:
    return max(value, 0.0) / 100.0


RISK_FRACTION = _percent_to_fraction(DEFAULT_RISK_PERCENT)


@dataclass
class RunConfig:
    """Minimal run configuration state."""

    data_path: Optional[Path]
    use_mt5: bool
    mt5_path: Optional[str]
    timeframe: str
    instruments: List[str]
    strategy_name: str
    lookback_days: Optional[int]
    initial_balance: float
    log_level: str
    risk_per_trade: Optional[float]
    spread: Optional[float]
    slippage: Optional[float]
    auto_export: bool
    export_dir: Path
    strategy_params: Dict[str, Any]
    start_timestamp: Optional[datetime] = None
    end_timestamp: Optional[datetime] = None

    def copy(self) -> "RunConfig":
        return RunConfig(
            data_path=self.data_path,
            use_mt5=self.use_mt5,
            mt5_path=self.mt5_path,
            timeframe=self.timeframe,
            instruments=list(self.instruments),
            strategy_name=self.strategy_name,
            lookback_days=self.lookback_days,
            initial_balance=self.initial_balance,
            log_level=self.log_level,
            risk_per_trade=self.risk_per_trade,
            spread=self.spread,
            slippage=self.slippage,
            auto_export=self.auto_export,
            export_dir=self.export_dir,
            strategy_params=dict(self.strategy_params),
            start_timestamp=self.start_timestamp,
            end_timestamp=self.end_timestamp,
        )

    def as_dict(self) -> Dict[str, Any]:
        return {
            "data_path": str(self.data_path) if self.data_path else None,
            "use_mt5": self.use_mt5,
            "mt5_path": self.mt5_path,
            "timeframe": self.timeframe,
            "instruments": self.instruments,
            "strategy_name": self.strategy_name,
            "lookback_days": self.lookback_days,
            "initial_balance": self.initial_balance,
            "log_level": self.log_level,
            "risk_per_trade": self.risk_per_trade,
            "spread": self.spread,
            "slippage": self.slippage,
            "auto_export": self.auto_export,
            "export_dir": str(self.export_dir),
            "strategy_params": self.strategy_params,
            "start_timestamp": self.start_timestamp.isoformat() if self.start_timestamp else None,
            "end_timestamp": self.end_timestamp.isoformat() if self.end_timestamp else None,
        }


BASE_CONFIG = RunConfig(
    data_path=Path(DATA_PATH) if DATA_PATH else None,
    use_mt5=USE_MT5,
    mt5_path=MT5_TERMINAL_PATH,
    timeframe=TIMEFRAME,
    instruments=list(INSTRUMENTS),
    strategy_name=STRATEGY_NAME,
    lookback_days=DAYS,
    initial_balance=INITIAL_BALANCE,
    log_level=LOG_LEVEL,
    risk_per_trade=RISK_FRACTION,
    spread=SPREAD,
    slippage=SLIPPAGE,
    auto_export=AUTO_EXPORT,
    export_dir=Path(EXPORT_DIR),
    strategy_params=dict(STRATEGY_PARAMS),
    start_timestamp=None,
    end_timestamp=None,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="ICT SMC Backtest Runner")
    parser.add_argument("--data", default=None, help="CSV data file (disables MT5 mode)")
    parser.add_argument("--instrument", help="Instrument symbol (omit to run every configured instrument)")
    parser.add_argument("--all-instruments", action="store_true", help="Force run across all configured instruments")
    parser.add_argument("--days", type=int, help="Rolling lookback window in days (e.g. --days 30)")
    parser.add_argument("--start", help="ISO timestamp for lookback start (inclusive)")
    parser.add_argument("--end", help="ISO timestamp for lookback end (inclusive)")
    parser.add_argument("--strategy", default=BASE_CONFIG.strategy_name, help="Strategy name registered in the factory")
    parser.add_argument("--timeframe", default=None, help="MT5 timeframe code (e.g. M1, M5, H1). If omitted, uses per-instrument optimal TF.")
    parser.add_argument("--mt5-path", help="Explicit path to MetaTrader 5 terminal64.exe")
    parser.add_argument("--use-csv", action="store_true", help="Force CSV mode even if MT5 is configured")
    parser.add_argument("--no-mt5", action="store_true", help="Alias for --use-csv")
    parser.add_argument("--spread", type=float, help="Override broker spread (in price units)")
    parser.add_argument("--slippage", type=float, help="Override broker slippage (in price units)")
    parser.add_argument("--risk", type=float, help="Risk percent per trade (e.g. --risk 1.5 for 1.5%)")
    parser.add_argument("--initial-balance", type=float, help="Override starting balance for this run")
    parser.add_argument("--log-level", choices=["DEBUG", "INFO", "WARNING", "ERROR"], help="Set log level for system/trade logs")
    parser.add_argument("--param", action="append", help="Override strategy tuning, e.g. --param min_candles=120")
    parser.add_argument("--export", help="Explicit CSV export path for equity curve")
    parser.add_argument("--plot", help="PNG export path for equity curve")
    parser.add_argument("--trades-csv", help="CSV export path for trade blotter")
    parser.add_argument("--report-json", help="JSON export path for performance + sessions")
    parser.add_argument("--auto-export", action="store_true", help="Force auto-export even if admin panel toggle is off")
    parser.add_argument("--export-dir", help="Directory for auto exports when running batches")
    parser.add_argument("--config-dump", action="store_true", help="Print the resolved configuration and exit")
    parser.add_argument("--no-progress", action="store_true", help="Disable the candle-level progress bar")
    return parser


def _coerce_strategy_value(raw: str) -> Any:
    token = raw.strip()
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    if "," in token and not token.startswith("["):
        parts = [part.strip() for part in token.split(",") if part.strip()]
        if len(parts) > 1:
            return tuple(parts)
    try:
        if "." in token or "e" in token.lower():
            return float(token)
        return int(token)
    except ValueError:
        return token


def parse_param_overrides(values: Sequence[str] | None) -> Dict[str, Any]:
    overrides: Dict[str, Any] = {}
    if not values:
        return overrides
    for raw in values:
        if not raw or "=" not in raw:
            continue
        key, value = raw.split("=", 1)
        overrides[key.strip()] = _coerce_strategy_value(value)
    return overrides


_MT5_SHUTDOWN_REGISTERED = False


def initialize_data_provider(config: RunConfig) -> None:
    global _MT5_SHUTDOWN_REGISTERED
    if not config.use_mt5:
        return
    try:
        ensure_mt5_initialized(config.mt5_path)
    except MT5NotAvailable as exc:
        raise SystemExit(str(exc)) from exc
    except Exception as exc:  # pragma: no cover - depends on local terminal state
        raise SystemExit(f"Failed to initialize MetaTrader5: {exc}") from exc
    if not _MT5_SHUTDOWN_REGISTERED:
        atexit.register(shutdown_mt5)
        _MT5_SHUTDOWN_REGISTERED = True


def load_candles_for_instrument(instrument: str, config: RunConfig) -> List[Candle]:
    if config.use_mt5:
        if not config.lookback_days:
            raise SystemExit("Set DAYS when using MT5 data.")
        try:
            return fetch_mt5_candles(
                symbol=instrument,
                timeframe=config.timeframe,
                days=config.lookback_days,
                terminal_path=config.mt5_path,
            )
        except MT5NotAvailable as exc:
            raise SystemExit(str(exc)) from exc
        except Exception as exc:
            raise SystemExit(f"MT5 data pull failed for {instrument}: {exc}") from exc
    if not config.data_path:
        raise SystemExit("CSV mode active but no --data path supplied.")
    candles = load_candles(str(config.data_path))
    if not candles:
        raise SystemExit(f"No candles found in {config.data_path}")
    return candles


def apply_cli_overrides(config: RunConfig, args: argparse.Namespace, params: Dict[str, Any]) -> None:
    if args.data:
        config.data_path = Path(args.data)
        config.use_mt5 = False
    if args.initial_balance is not None:
        config.initial_balance = args.initial_balance
    if args.log_level:
        config.log_level = args.log_level
    if args.export_dir:
        config.export_dir = Path(args.export_dir)
    if args.auto_export:
        config.auto_export = True
    if args.days is not None:
        config.lookback_days = max(1, args.days)
    if args.timeframe:
        config.timeframe = args.timeframe.upper()
    if args.mt5_path:
        config.mt5_path = args.mt5_path
    if getattr(args, "use_csv", False) or getattr(args, "no_mt5", False):
        config.use_mt5 = False
    if args.start:
        config.start_timestamp = datetime.fromisoformat(args.start)
    if args.end:
        config.end_timestamp = datetime.fromisoformat(args.end)
    if args.spread is not None:
        config.spread = args.spread
    if args.slippage is not None:
        config.slippage = args.slippage
    if getattr(args, "risk", None) is not None:
        config.risk_per_trade = _percent_to_fraction(args.risk)
    if getattr(args, "strategy", None):
        config.strategy_name = args.strategy
    if params:
        config.strategy_params.update(params)
    if config.use_mt5:
        config.data_path = None
    elif not config.data_path:
        raise SystemExit("CSV mode selected but no --data path was provided.")


def set_log_levels(level: str) -> None:
    desired = getattr(logging, level.upper(), logging.INFO)
    for name in ("system", "trades", "errors"):
        logging.getLogger(name).setLevel(desired)


def print_report(result, instrument: str, config: RunConfig) -> None:
    print(f"\n=== {instrument} Performance Report ===")
    window_minutes = int((result.data_end - result.data_start).total_seconds() // 60)
    window_days = (result.data_end.date() - result.data_start.date()).days + 1
    print(
        "  Data window : {start:%Y-%m-%d %H:%M} → {end:%Y-%m-%d %H:%M} ({minutes}m, {days} day{plural})".format(
            start=result.data_start,
            end=result.data_end,
            minutes=window_minutes,
            days=window_days,
            plural="s" if window_days != 1 else "",
        )
    )
    print(f"  Candles     : {result.candle_count:,}")
    print(f"  Strategy    : {config.strategy_name}")
    if config.use_mt5:
        print(f"  Source      : MT5 {config.timeframe}")
    elif config.data_path:
        print(f"  Source      : CSV {config.data_path}")
    risk_fraction = config.risk_per_trade if config.risk_per_trade is not None else SETTINGS.risk.risk_per_trade
    print(f"  Risk/trade  : {risk_fraction:.2%} of equity")
    if config.lookback_days:
        print(f"  Lookback    : last {config.lookback_days} day(s)")
    elif config.start_timestamp or config.end_timestamp:
        print(f"  Lookback    : custom window")
    for key, value in result.report.as_dict().items():
        print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    if result.session_stats:
        print("  Sessions:")
        for session, stats in result.session_stats.items():
            win_rate = (stats.wins / stats.trades * 100) if stats.trades else 0.0
            avg_r = stats.total_r / stats.trades if stats.trades else 0.0
            print(
                f"    {session}: trades={stats.trades} win_rate={win_rate:.2f}% avg_R={avg_r:.2f} pnl={stats.pnl:.2f}"
            )


def handle_exports(
    result,
    instrument: str,
    strategy: str,
    args: argparse.Namespace,
    config: RunConfig,
) -> None:
    export_dir = Path(args.export_dir) if args.export_dir else config.export_dir
    export_dir.mkdir(parents=True, exist_ok=True)

    if args.export:
        export_path = Path(args.export)
        export_equity_curve(export_path, result.timestamps, result.equity_curve)
        print(f"Equity curve exported to {export_path}")
    elif config.auto_export or args.auto_export:
        path = export_dir / f"{instrument}_{strategy}_equity.csv"
        export_equity_curve(path, result.timestamps, result.equity_curve)
        print(f"Auto-exported equity curve to {path}")

    if args.plot:
        plot_equity_curve(Path(args.plot), result.timestamps, result.equity_curve)
        print(f"Equity chart saved to {args.plot}")

    trades_path = args.trades_csv
    if trades_path:
        export_trades_csv(Path(trades_path), result.trades)
        print(f"Trades exported to {trades_path}")
    elif (config.auto_export or args.auto_export) and result.trades:
        path = export_dir / f"{instrument}_{strategy}_trades.csv"
        export_trades_csv(path, result.trades)
        print(f"Auto-exported trades to {path}")

    report_path = args.report_json
    if report_path:
        export_report_json(Path(report_path), result.report, result.session_stats)
        print(f"Report JSON saved to {report_path}")
    elif config.auto_export or args.auto_export:
        path = export_dir / f"{instrument}_{strategy}_report.json"
        export_report_json(path, result.report, result.session_stats)
        print(f"Auto-exported report to {path}")


def run_single_backtest(
    instrument: str,
    args: argparse.Namespace,
    config: RunConfig,
) -> None:
    system_logger = get_logger("system")
    lookback_msg = f"last {config.lookback_days} days" if config.lookback_days else "full dataset"
    source_msg = f"MT5 {config.timeframe}" if config.use_mt5 else f"CSV {config.data_path}"
    print(
        "\n→ {instrument}: {strategy} | {lookback} [{source}] @ balance {balance:,.2f}".format(
            instrument=instrument,
            strategy=config.strategy_name,
            lookback=lookback_msg,
            source=source_msg,
            balance=config.initial_balance,
        )
    )
    spread = args.spread if args.spread is not None else config.spread
    slippage = args.slippage if args.slippage is not None else config.slippage
    candles = load_candles_for_instrument(instrument, config)
    engine = BacktestEngine(
        instrument=instrument,
        data_path=str(config.data_path) if config.data_path else None,
        strategy_name=config.strategy_name,
        spread=spread,
        slippage=slippage,
        initial_balance=config.initial_balance,
        strategy_params=config.strategy_params,
        start_timestamp=config.start_timestamp,
        end_timestamp=config.end_timestamp,
        lookback_days=config.lookback_days,
        risk_per_trade=config.risk_per_trade,
        show_progress=not args.no_progress,
        candles=candles,
    )
    result = run_backtest_sync(engine)
    system_logger.info(
        "Backtest complete | instrument=%s strategy=%s trades=%s",
        instrument,
        config.strategy_name,
        len(result.trades),
    )
    print_report(result, instrument, config)
    if not result.trades:
        print("  Note: No qualifying setups triggered. Consider easing ICT params (e.g. --param min_displacement=0.8) or enabling demo_mode for smoke tests.")
    handle_exports(result, instrument, config.strategy_name, args, config)


def main() -> None:
    configure_logging()
    parser = build_parser()
    args = parser.parse_args()
    config = BASE_CONFIG.copy()
    param_overrides = parse_param_overrides(args.param)
    apply_cli_overrides(config, args, param_overrides)
    initialize_data_provider(config)

    if args.config_dump:
        print(json.dumps(config.as_dict(), indent=2))
        return
    set_log_levels(config.log_level)
    if args.all_instruments or not args.instrument:
        instruments = list(dict.fromkeys(config.instruments))
    else:
        instruments = [args.instrument]
    if not instruments:
        raise SystemExit("No instruments selected. Update RUN CONFIG at the top or supply --instrument symbol.")
    for symbol in instruments:
        # Create per-instrument config with optimized params
        inst_config = config.copy()
        # Start with per-instrument defaults, then overlay CLI --param overrides
        inst_params = get_params_for_instrument(symbol)
        inst_params.update(param_overrides)  # CLI params take precedence
        inst_config.strategy_params = inst_params
        # CLI --timeframe takes precedence, then per-instrument, then base config default
        if args.timeframe:
            inst_config.timeframe = args.timeframe.upper()
        else:
            inst_config.timeframe = get_timeframe_for_instrument(symbol) or inst_config.timeframe
        run_single_backtest(symbol, args, inst_config)


if __name__ == "__main__":  # pragma: no cover
    main()