"""Backtest performance metrics and reporting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.execution import SessionStats, TradeRecord


@dataclass
class PerformanceReport:
    """Structured summary of a backtest run."""

    total_return_pct: float
    sharpe_ratio: float
    profit_factor: float
    max_drawdown_pct: float
    win_rate_pct: float
    trade_count: int
    avg_r_multiple: float

    def as_dict(self) -> dict:
        return {
            "total_return_pct": self.total_return_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "profit_factor": self.profit_factor,
            "max_drawdown_pct": self.max_drawdown_pct,
            "win_rate_pct": self.win_rate_pct,
            "trade_count": self.trade_count,
            "avg_r_multiple": self.avg_r_multiple,
        }


def compute_performance(equity_curve: Sequence[float], trades: Sequence[TradeRecord]) -> PerformanceReport:
    """Compute performance statistics from equity and trades."""

    returns = np.diff(equity_curve) / equity_curve[:-1]
    sharpe = 0.0 if returns.size == 0 else (np.mean(returns) / (np.std(returns) + 1e-9)) * np.sqrt(252)

    gains = [t.pnl for t in trades if t.pnl > 0]
    losses = [-t.pnl for t in trades if t.pnl < 0]
    profit_factor = (sum(gains) / sum(losses)) if losses else float("inf")

    peak = equity_curve[0]
    max_drawdown = 0.0
    for value in equity_curve:
        if value > peak:
            peak = value
        drawdown = (value - peak) / peak
        max_drawdown = min(max_drawdown, drawdown)

    wins = sum(1 for t in trades if t.pnl > 0)
    win_rate = (wins / len(trades) * 100) if trades else 0.0

    total_return = (equity_curve[-1] / equity_curve[0] - 1) * 100
    avg_r = float(np.mean([t.r_multiple for t in trades])) if trades else 0.0

    return PerformanceReport(
        total_return_pct=total_return,
        sharpe_ratio=sharpe,
        profit_factor=profit_factor,
        max_drawdown_pct=abs(max_drawdown * 100),
        win_rate_pct=win_rate,
        trade_count=len(trades),
        avg_r_multiple=avg_r,
    )


def export_equity_curve(path: str | Path, timestamps: Iterable[str], equity_values: Sequence[float]) -> None:
    """Persist the equity curve to CSV for later analysis."""

    ts, eq = _align_series(timestamps, equity_values)
    frame = pd.DataFrame({"timestamp": ts, "equity": eq})
    frame.to_csv(Path(path), index=False)


def plot_equity_curve(path: str | Path, timestamps: Iterable[str], equity_values: Sequence[float]) -> None:
    """Render an equity curve chart to disk using Matplotlib."""

    ts_raw, eq = _align_series(timestamps, equity_values)
    ts = pd.to_datetime(ts_raw)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(ts, eq, color="#1f77b4", linewidth=1.5)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Time")
    ax.set_ylabel("Equity")
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.autofmt_xdate()
    fig.tight_layout()
    output = Path(path)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def export_trades_csv(path: str | Path, trades: Sequence[TradeRecord]) -> None:
    """Write trade blotter information to CSV."""

    rows = []
    columns = [
        "instrument",
        "side",
        "entry_price",
        "exit_price",
        "quantity",
        "opened_at",
        "closed_at",
        "pnl",
        "r_multiple",
        "session",
    ]
    for trade in trades:
        rows.append(
            {
                "instrument": trade.instrument,
                "side": trade.side.value,
                "entry_price": trade.entry_price,
                "exit_price": trade.exit_price,
                "quantity": trade.quantity,
                "opened_at": trade.opened_at.isoformat(),
                "closed_at": trade.closed_at.isoformat(),
                "pnl": trade.pnl,
                "r_multiple": trade.r_multiple,
                "session": trade.session,
            }
        )
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(Path(path), index=False)


def export_report_json(
    path: str | Path,
    report: PerformanceReport,
    session_stats: dict[str, SessionStats],
) -> None:
    """Persist performance report plus session stats to JSON."""

    payload = {
        "report": report.as_dict(),
        "sessions": {
            name: {
                "trades": stats.trades,
                "wins": stats.wins,
                "losses": stats.losses,
                "pnl": stats.pnl,
                "total_r": stats.total_r,
            }
            for name, stats in session_stats.items()
        },
    }
    Path(path).write_text(json.dumps(payload, indent=2))


def _align_series(timestamps: Iterable[str], equity_values: Sequence[float]) -> Tuple[List[str], List[float]]:
    """Ensure timestamps and equity arrays have matching lengths."""

    ts = list(timestamps)
    eq = list(equity_values)
    if not eq:
        return ts, eq
    if len(eq) == len(ts):
        return ts, eq
    if len(eq) == len(ts) + 1:
        anchor = ts[0] if ts else pd.Timestamp.utcnow().isoformat()
        ts_with_anchor = [anchor] + ts
        return ts_with_anchor, eq
    raise ValueError("Timestamp and equity series lengths are incompatible")


__all__ = ["PerformanceReport", "compute_performance", "export_equity_curve", "plot_equity_curve"]