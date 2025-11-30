# TradingBot

ICT/SMC backtesting harness driven directly from MetaTrader 5 data.

## Requirements

- Python 3.12+
- MetaTrader 5 terminal installed and logged in (demo or live)
- Project dependencies:

```powershell
pip install -r requirements.txt
```

## Quick Start

1. Launch MT5 so the terminal is connected to your broker feed.
2. Adjust the quick knobs at the top of `backtest/backtest.py` (e.g., `TIMEFRAME`, `DAYS`, `MT5_TERMINAL_PATH`).
3. Run a backtest—this pulls exactly the number of days requested from MT5:

```powershell
python backtest/backtest.py --instrument EURUSD+ --days 15
```

The runner will show a progress bar per instrument plus a performance block that includes the tested window, MT5 timeframe, and risk settings.

### Risk per trade

- `RISK_PERCENT` in `backtest/backtest.py` expresses the per-trade risk in percent (default matches `config.settings` at 0.75%).
- Override it from the CLI with `--risk 2.5` (meaning 2.5% of equity per trade) when you need ad-hoc tweaks.

### Strategy tuning

- `STRATEGY_NAME` stays on `ict_smc`; the `STRATEGY_PARAMS` dict in `backtest/backtest.py` is the single “optimal” ICT/SMC profile.
- Adjust any value in `STRATEGY_PARAMS` to make the change permanent, or pass overrides via CLI: `--param min_candles=60 --param min_displacement=0.7`.
- Multiple overrides are supported; each `--param` uses `key=value` pairs and basic type coercion (bool, int, float, comma-delimited tuples).
- To make higher-timeframe candles count as sweeps, bump `sweep_reentry_buffer_mult` (0.2 = 20% of ATR tolerance). Set `require_bias_alignment=true` if you want BOS/MSS agreement to be mandatory.
- If you finish a run with `trade_count: 0`, progressively relax `min_displacement`, `sweep_lookback`, `gap_tolerance_mult`, or temporarily enable `demo_mode=true` for smoke tests.

## CSV fallback (offline dev/testing)

If MT5 isn’t available, you can temporarily point the runner at a CSV file:

```powershell
python backtest/backtest.py --data data/sample_data.csv --use-csv --instrument EURUSD+
```

This mode bypasses MT5 initialization but is only meant for smoke tests.
