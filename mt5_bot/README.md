# Simple EURUSD Trading Bot - Graduation Project 2026

A clean, educational MetaTrader 5 Expert Advisor (EA) demonstrating automated trading with Telegram notifications.

---

## 📋 Project Overview

**Purpose**: Educational graduation project demonstrating algorithmic trading concepts

**Platform**: MetaTrader 5 (MT5 Demo Account)

**Instrument**: EURUSD (Euro vs US Dollar)

**Timeframe**: M15 (15-minute candles)

**Language**: MQL5 (Expert Advisor) + Python (Telegram notifications)

---

## 🎯 Trading Strategy

### Simple Trend Following with EMA Crossover + RSI Filter

**Indicators Used:**
- EMA 50 (Fast Moving Average)
- EMA 200 (Slow Moving Average)  
- RSI 14 (Relative Strength Index)

### Entry Rules

**BUY Signal:**
- EMA 50 > EMA 200 (uptrend confirmed)
- RSI > 50 (bullish momentum)

**SELL Signal:**
- EMA 50 < EMA 200 (downtrend confirmed)
- RSI < 50 (bearish momentum)

### Risk Management

- **Stop Loss**: 30 pips (fixed)
- **Take Profit**: 60 pips (fixed)
- **Position Size**: 0.1 lot (configurable)
- **Max Concurrent Trades**: 1 (only one position at a time)

---

## 🚀 Installation & Setup

### Step 1: Install MetaTrader 5

1. Download MT5 from your broker or [MetaQuotes website](https://www.metatrader5.com/)
2. Install and open a **DEMO account**
3. Select EURUSD in Market Watch

### Step 2: Install the Expert Advisor

1. Copy `SimpleEA_EURUSD.mq5` to:
   ```
   C:\Users\YourName\AppData\Roaming\MetaQuotes\Terminal\[TERMINAL_ID]\MQL5\Experts\
   ```

2. Open MetaEditor (F4 in MT5) or compile from MT5
3. Compile the EA (F7 or Compile button)
4. Compiled file will be created: `SimpleEA_EURUSD.ex5`

### Step 3: Set Up Telegram Notifications (Optional)

#### Get Your Bot Token:
1. Open Telegram and search for `@BotFather`
2. Send `/newbot` and follow instructions
3. Copy the **Bot Token** (e.g., `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`)

#### Get Your Chat ID:
1. Search for `@userinfobot` on Telegram
2. Start a chat - it will show your **Chat ID** (e.g., `123456789`)

#### Configure Python Script:
1. Open `telegram_bot.py`
2. Replace these lines:
   ```python
   TELEGRAM_BOT_TOKEN = "123456789:ABCdefGHIjklMNOpqrsTUVwxyz"  # Your bot token
   TELEGRAM_CHAT_ID = "123456789"  # Your chat ID
   ```

3. Install Python dependencies:
   ```bash
   pip install requests
   ```

4. Test your configuration:
   ```bash
   python test_telegram.py
   ```

### Step 4: Run the EA

1. Open MT5 and load **EURUSD M15 chart**
2. Drag `SimpleEA_EURUSD` from Navigator → Expert Advisors
3. In EA settings dialog:
   - **LotSize**: 0.1 (adjust based on account size)
   - **StopLossPips**: 30
   - **TakeProfitPips**: 60
   - **EnableTelegramNotifications**: true
   - **TelegramScriptPath**: Full path to `telegram_bot.py`
4. Click **OK**
5. Enable **AutoTrading** (button in toolbar)

---

## 📊 How It Works

### Initialization
- EA loads and creates indicator handles (EMA 50, EMA 200, RSI 14)
- Prints confirmation message in MT5 Journal

### On Every Tick (Price Update)
1. **Check existing positions** - Only 1 trade at a time
2. **Read indicator values** - Current EMA and RSI levels
3. **Evaluate conditions**:
   - If BUY conditions met → Open BUY trade
   - If SELL conditions met → Open SELL trade
4. **Set SL/TP** automatically
5. **Send Telegram notification** (if enabled)

### Trade Lifecycle
1. **Entry**: Trade opens when signals align
2. **Management**: MT5 handles SL/TP automatically
3. **Exit**: Position closes at SL or TP
4. **Notification**: Telegram alert sent with profit/loss

---

## 📁 Project Files

```
mt5_graduation_project/
│
├── SimpleEA_EURUSD.mq5          # Main Expert Advisor (MQL5)
├── telegram_bot.py               # Telegram notification script (Python)
├── test_telegram.py              # Test script to verify Telegram setup
└── README.md                     # This documentation
```

---

## 🎓 Educational Value

This project demonstrates:

1. **Algorithmic Trading Concepts**
   - Trend detection (EMA crossover)
   - Momentum confirmation (RSI)
   - Automated order placement

2. **Risk Management**
   - Fixed stop-loss and take-profit
   - Position size control
   - Single-position rule

3. **MQL5 Programming**
   - Indicator handling (`iMA`, `iRSI`)
   - Trade execution (`OrderSend`)
   - Event handling (`OnTick`, `OnTradeTransaction`)

4. **System Integration**
   - Calling external Python scripts
   - API integration (Telegram Bot API)
   - Real-time notifications

---

## ⚙️ Configuration Options

You can adjust these parameters in the EA settings:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `LotSize` | 0.1 | Position size in lots |
| `StopLossPips` | 30 | Stop loss distance in pips |
| `TakeProfitPips` | 60 | Take profit distance in pips |
| `MagicNumber` | 123456 | Unique EA identifier |
| `EMA_Fast_Period` | 50 | Fast EMA period |
| `EMA_Slow_Period` | 200 | Slow EMA period |
| `RSI_Period` | 14 | RSI calculation period |
| `EnableTelegramNotifications` | true | Enable/disable alerts |
| `TelegramScriptPath` | telegram_bot.py | Path to Python script |

---

## 🔍 Testing & Validation

### Strategy Tester (Backtesting)

1. In MT5, press **Ctrl+R** to open Strategy Tester
2. Select:
   - **Expert Advisor**: SimpleEA_EURUSD
   - **Symbol**: EURUSD
   - **Period**: M15
   - **Date Range**: Last 3-6 months
3. Click **Start**
4. Review results in the **Results** and **Graph** tabs

**Note**: Telegram notifications won't work in Strategy Tester (external scripts disabled)

### Live Testing on Demo

1. Run on DEMO account first
2. Monitor for 1-2 weeks
3. Check:
   - Trade entries match strategy logic
   - SL/TP levels are correct
   - Telegram notifications arrive
4. Review performance in MT5 Journal and Account History

---

## 📱 Sample Telegram Notifications

**Trade Opened:**
```
🤖 MT5 Trading Bot
2026-01-10 14:30:25

🟢 BUY Trade Opened
Symbol: EURUSD
Price: 1.08500
SL: 1.08200
TP: 1.09100
```

**Trade Closed (Profit):**
```
🤖 MT5 Trading Bot
2026-01-10 16:45:10

✅ Trade Closed (PROFIT)
Symbol: EURUSD
P/L: 45.50 USD
```

**Trade Closed (Loss):**
```
🤖 MT5 Trading Bot
2026-01-10 18:20:33

❌ Trade Closed (LOSS)
Symbol: EURUSD
P/L: -22.30 USD
```

---

## 🐛 Troubleshooting

### EA Not Opening Trades
- Check AutoTrading is enabled (green button in MT5)
- Verify EA is attached to EURUSD M15 chart
- Look at Journal for error messages
- Ensure market is open (EURUSD trades 24/5)

### Telegram Not Working
- Run `test_telegram.py` to verify configuration
- Check Bot Token and Chat ID are correct
- Ensure you've started a chat with your bot on Telegram
- Verify internet connection

### Compilation Errors
- Make sure you're using MT5 (not MT4)
- Check MQL5 syntax in MetaEditor
- Update MT5 to latest version

### Orders Rejected
- Check account balance (sufficient margin)
- Verify lot size is valid for your broker
- Ensure EURUSD is available for trading

---

## 📈 Performance Expectations

**⚠️ Important Disclaimer:**

This is an **educational project**, not a profit-guaranteed system.

- **Purpose**: Demonstrate automated trading concepts
- **Results**: Will vary based on market conditions
- **Risk**: Demo account only for learning
- **Reality**: Most simple strategies need optimization for profitability

**Never risk real money without proper testing and risk management!**

---

## 📝 Project Report Guide

For your graduation presentation, include:

1. **Introduction**
   - Algorithmic trading overview
   - Why EURUSD and M15 timeframe

2. **Strategy Explanation**
   - EMA crossover logic
   - RSI confirmation
   - Visual examples (screenshots)

3. **Implementation**
   - MQL5 code structure
   - Key functions explained
   - Telegram integration

4. **Testing Results**
   - Backtest performance
   - Demo account results (if available)
   - Charts and statistics

5. **Conclusion**
   - What you learned
   - Challenges faced
   - Potential improvements

---

## 🎯 Next Steps & Improvements

After mastering this basic EA, you could:

1. **Add trailing stop-loss** for better profit capture
2. **Implement breakeven logic** to protect profits
3. **Add time filters** (avoid low-volatility periods)
4. **Use dynamic SL/TP** based on ATR
5. **Test multiple timeframes** (M5, H1, etc.)
6. **Optimize parameters** using Strategy Tester
7. **Add more indicators** (MACD, Bollinger Bands)
8. **Implement position sizing** based on risk percentage

---

## 📞 Support

For graduation project questions:
- Review MQL5 documentation: https://www.mql5.com/en/docs
- MetaTrader forum: https://www.mql5.com/en/forum
- Telegram Bot API: https://core.telegram.org/bots/api

---

## ✅ Final Checklist

Before demonstration:

- [ ] MT5 installed and demo account created
- [ ] EA compiled successfully (no errors)
- [ ] EA attached to EURUSD M15 chart
- [ ] AutoTrading enabled
- [ ] Telegram bot configured and tested
- [ ] Test message received on Telegram
- [ ] EA parameters reviewed and set
- [ ] Journal shows "EA Initialized" message
- [ ] At least one trade executed (for demo)
- [ ] Screenshots prepared for presentation

---

**Good luck with your graduation project! 🎓📈**

---

*Version 1.0 - January 2026*
*Educational Project - For Demo Trading Only*
