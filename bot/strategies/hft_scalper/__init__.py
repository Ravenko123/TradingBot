"""HFT Scalping Strategy package.

This strategy implements High Frequency Trading / Scalping concepts:
- M1 timeframe for rapid entries/exits
- Small pip targets (3-10 pips per trade)
- Tight stop losses (2-5 pips)
- High trade frequency (many small wins)
- Momentum and spread-based entries
- Session liquidity optimization
"""

from strategies.hft_scalper.strategy import HFTScalperStrategy

__all__ = ["HFTScalperStrategy"]
