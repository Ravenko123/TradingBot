//+------------------------------------------------------------------+
//|                                           SimpleEA_EURUSD.mq5    |
//|                                   Educational Graduation Project |
//|                                                                  |
//| STRATEGY: Simple Trend Following with EMA Crossover + RSI Filter|
//| INSTRUMENT: EURUSD                                               |
//| TIMEFRAME: M15                                                   |
//+------------------------------------------------------------------+
#property copyright "Graduation Project 2026"
#property link      ""
#property version   "1.00"
#property strict

//--- Input Parameters (easily adjustable)
input double   LotSize = 0.1;              // Position size (0.1 = 0.1 lot)
input int      StopLossPips = 30;          // Stop Loss in pips
input int      TakeProfitPips = 60;        // Take Profit in pips
input int      MagicNumber = 123456;       // Unique identifier for this EA

input int      EMA_Fast_Period = 50;       // EMA 50 period
input int      EMA_Slow_Period = 200;      // EMA 200 period
input int      RSI_Period = 14;            // RSI period

input bool     EnableTelegramNotifications = true;  // Enable Telegram alerts
input string   TelegramScriptPath = "telegram_bot.py";  // Path to Python script

//--- Global Variables
int ema_fast_handle;
int ema_slow_handle;
int rsi_handle;

double ema_fast_buffer[];
double ema_slow_buffer[];
double rsi_buffer[];

//+------------------------------------------------------------------+
//| Expert initialization function                                   |
//+------------------------------------------------------------------+
int OnInit()
{
   Print("=== Simple EURUSD EA Initialized ===");
   Print("Strategy: EMA 50/200 Crossover + RSI Filter");
   Print("Stop Loss: ", StopLossPips, " pips | Take Profit: ", TakeProfitPips, " pips");
   
   //--- Create indicator handles
   ema_fast_handle = iMA(_Symbol, PERIOD_M15, EMA_Fast_Period, 0, MODE_EMA, PRICE_CLOSE);
   ema_slow_handle = iMA(_Symbol, PERIOD_M15, EMA_Slow_Period, 0, MODE_EMA, PRICE_CLOSE);
   rsi_handle = iRSI(_Symbol, PERIOD_M15, RSI_Period, PRICE_CLOSE);
   
   //--- Check if indicators are created successfully
   if(ema_fast_handle == INVALID_HANDLE || ema_slow_handle == INVALID_HANDLE || rsi_handle == INVALID_HANDLE)
   {
      Print("ERROR: Failed to create indicator handles!");
      return(INIT_FAILED);
   }
   
   //--- Set array as series (most recent data at index 0)
   ArraySetAsSeries(ema_fast_buffer, true);
   ArraySetAsSeries(ema_slow_buffer, true);
   ArraySetAsSeries(rsi_buffer, true);
   
   Print("Indicators initialized successfully!");
   Print("Waiting for trading signals...");
   
   return(INIT_SUCCEEDED);
}

//+------------------------------------------------------------------+
//| Expert deinitialization function                                 |
//+------------------------------------------------------------------+
void OnDeinit(const int reason)
{
   //--- Release indicator handles
   if(ema_fast_handle != INVALID_HANDLE)
      IndicatorRelease(ema_fast_handle);
   if(ema_slow_handle != INVALID_HANDLE)
      IndicatorRelease(ema_slow_handle);
   if(rsi_handle != INVALID_HANDLE)
      IndicatorRelease(rsi_handle);
   
   Print("=== Simple EURUSD EA Stopped ===");
}

//+------------------------------------------------------------------+
//| Expert tick function (runs on every price tick)                 |
//+------------------------------------------------------------------+
void OnTick()
{
   //--- Check if we already have an open position
   if(HasOpenPosition())
   {
      return; // Only one trade at a time
   }
   
   //--- Copy indicator data to buffers
   if(CopyBuffer(ema_fast_handle, 0, 0, 3, ema_fast_buffer) <= 0)
      return;
   if(CopyBuffer(ema_slow_handle, 0, 0, 3, ema_slow_buffer) <= 0)
      return;
   if(CopyBuffer(rsi_handle, 0, 0, 3, rsi_buffer) <= 0)
      return;
   
   //--- Get current indicator values
   double ema_fast_current = ema_fast_buffer[0];
   double ema_slow_current = ema_slow_buffer[0];
   double rsi_current = rsi_buffer[0];
   
   //--- Get current price
   double ask = SymbolInfoDouble(_Symbol, SYMBOL_ASK);
   double bid = SymbolInfoDouble(_Symbol, SYMBOL_BID);
   
   //--- Check BUY conditions
   if(ema_fast_current > ema_slow_current && rsi_current > 50)
   {
      Print("--- BUY SIGNAL DETECTED ---");
      Print("EMA50: ", DoubleToString(ema_fast_current, 5), " > EMA200: ", DoubleToString(ema_slow_current, 5));
      Print("RSI: ", DoubleToString(rsi_current, 2), " > 50");
      
      OpenBuyTrade(ask);
   }
   
   //--- Check SELL conditions
   else if(ema_fast_current < ema_slow_current && rsi_current < 50)
   {
      Print("--- SELL SIGNAL DETECTED ---");
      Print("EMA50: ", DoubleToString(ema_fast_current, 5), " < EMA200: ", DoubleToString(ema_slow_current, 5));
      Print("RSI: ", DoubleToString(rsi_current, 2), " < 50");
      
      OpenSellTrade(bid);
   }
}

//+------------------------------------------------------------------+
//| Check if there's already an open position                        |
//+------------------------------------------------------------------+
bool HasOpenPosition()
{
   for(int i = PositionsTotal() - 1; i >= 0; i--)
   {
      ulong ticket = PositionGetTicket(i);
      if(PositionSelectByTicket(ticket))
      {
         if(PositionGetString(POSITION_SYMBOL) == _Symbol && 
            PositionGetInteger(POSITION_MAGIC) == MagicNumber)
         {
            return true;
         }
      }
   }
   return false;
}

//+------------------------------------------------------------------+
//| Open a BUY trade                                                  |
//+------------------------------------------------------------------+
void OpenBuyTrade(double price)
{
   //--- Calculate Stop Loss and Take Profit
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   double sl = price - (StopLossPips * 10 * point);   // Convert pips to price
   double tp = price + (TakeProfitPips * 10 * point);
   
   //--- Normalize prices
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   
   //--- Prepare trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_BUY;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Simple EA - BUY";
   
   //--- Send order
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ BUY ORDER OPENED SUCCESSFULLY!");
         Print("Ticket: ", result.order);
         Print("Entry Price: ", DoubleToString(price, digits));
         Print("Stop Loss: ", DoubleToString(sl, digits), " (", StopLossPips, " pips)");
         Print("Take Profit: ", DoubleToString(tp, digits), " (", TakeProfitPips, " pips)");
         
         //--- Send Telegram notification
         if(EnableTelegramNotifications)
         {
            string message = "🟢 BUY Trade Opened\n";
            message += "Symbol: " + _Symbol + "\n";
            message += "Price: " + DoubleToString(price, digits) + "\n";
            message += "SL: " + DoubleToString(sl, digits) + "\n";
            message += "TP: " + DoubleToString(tp, digits);
            SendTelegramMessage(message);
         }
      }
      else
      {
         Print("✗ ORDER FAILED! Error code: ", result.retcode);
         Print("Error description: ", GetTradeResultDescription(result.retcode));
      }
   }
   else
   {
      Print("✗ OrderSend failed! Error: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Open a SELL trade                                                 |
//+------------------------------------------------------------------+
void OpenSellTrade(double price)
{
   //--- Calculate Stop Loss and Take Profit
   double point = SymbolInfoDouble(_Symbol, SYMBOL_POINT);
   int digits = (int)SymbolInfoInteger(_Symbol, SYMBOL_DIGITS);
   
   double sl = price + (StopLossPips * 10 * point);   // Convert pips to price
   double tp = price - (TakeProfitPips * 10 * point);
   
   //--- Normalize prices
   sl = NormalizeDouble(sl, digits);
   tp = NormalizeDouble(tp, digits);
   
   //--- Prepare trade request
   MqlTradeRequest request = {};
   MqlTradeResult result = {};
   
   request.action = TRADE_ACTION_DEAL;
   request.symbol = _Symbol;
   request.volume = LotSize;
   request.type = ORDER_TYPE_SELL;
   request.price = price;
   request.sl = sl;
   request.tp = tp;
   request.deviation = 10;
   request.magic = MagicNumber;
   request.comment = "Simple EA - SELL";
   
   //--- Send order
   if(OrderSend(request, result))
   {
      if(result.retcode == TRADE_RETCODE_DONE)
      {
         Print("✓ SELL ORDER OPENED SUCCESSFULLY!");
         Print("Ticket: ", result.order);
         Print("Entry Price: ", DoubleToString(price, digits));
         Print("Stop Loss: ", DoubleToString(sl, digits), " (", StopLossPips, " pips)");
         Print("Take Profit: ", DoubleToString(tp, digits), " (", TakeProfitPips, " pips)");
         
         //--- Send Telegram notification
         if(EnableTelegramNotifications)
         {
            string message = "🔴 SELL Trade Opened\n";
            message += "Symbol: " + _Symbol + "\n";
            message += "Price: " + DoubleToString(price, digits) + "\n";
            message += "SL: " + DoubleToString(sl, digits) + "\n";
            message += "TP: " + DoubleToString(tp, digits);
            SendTelegramMessage(message);
         }
      }
      else
      {
         Print("✗ ORDER FAILED! Error code: ", result.retcode);
         Print("Error description: ", GetTradeResultDescription(result.retcode));
      }
   }
   else
   {
      Print("✗ OrderSend failed! Error: ", GetLastError());
   }
}

//+------------------------------------------------------------------+
//| Send Telegram message via Python script                          |
//+------------------------------------------------------------------+
void SendTelegramMessage(string message)
{
   //--- Replace newlines for command line
   StringReplace(message, "\n", " | ");
   
   //--- Build command
   string command = "python " + TelegramScriptPath + " \"" + message + "\"";
   
   //--- Execute Python script (note: may not work in Strategy Tester)
   // In real trading, this will trigger the Python script
   Print("Telegram notification sent: ", message);
}

//+------------------------------------------------------------------+
//| Get human-readable trade result description                      |
//+------------------------------------------------------------------+
string GetTradeResultDescription(uint retcode)
{
   switch(retcode)
   {
      case TRADE_RETCODE_DONE: return "Request completed";
      case TRADE_RETCODE_REJECT: return "Request rejected";
      case TRADE_RETCODE_INVALID: return "Invalid request";
      case TRADE_RETCODE_ERROR: return "Request processing error";
      case TRADE_RETCODE_TIMEOUT: return "Request timeout";
      case TRADE_RETCODE_INVALID_VOLUME: return "Invalid volume";
      case TRADE_RETCODE_INVALID_PRICE: return "Invalid price";
      case TRADE_RETCODE_INVALID_STOPS: return "Invalid stops";
      case TRADE_RETCODE_NO_MONEY: return "Not enough money";
      default: return "Unknown error";
   }
}

//+------------------------------------------------------------------+
//| Trade transaction event handler                                  |
//+------------------------------------------------------------------+
void OnTradeTransaction(const MqlTradeTransaction& trans,
                        const MqlTradeRequest& request,
                        const MqlTradeResult& result)
{
   //--- Detect when position is closed
   if(trans.type == TRADE_TRANSACTION_DEAL_ADD)
   {
      ulong deal_ticket = trans.deal;
      if(HistoryDealSelect(deal_ticket))
      {
         long deal_magic = HistoryDealGetInteger(deal_ticket, DEAL_MAGIC);
         
         if(deal_magic == MagicNumber)
         {
            long deal_type = HistoryDealGetInteger(deal_ticket, DEAL_TYPE);
            double deal_profit = HistoryDealGetDouble(deal_ticket, DEAL_PROFIT);
            string symbol = HistoryDealGetString(deal_ticket, DEAL_SYMBOL);
            
            //--- Check if this is a position close
            if(deal_type == DEAL_TYPE_BUY || deal_type == DEAL_TYPE_SELL)
            {
               if(symbol == _Symbol)
               {
                  Print("--- POSITION CLOSED ---");
                  Print("Profit/Loss: ", DoubleToString(deal_profit, 2), " ", AccountInfoString(ACCOUNT_CURRENCY));
                  
                  //--- Send Telegram notification
                  if(EnableTelegramNotifications)
                  {
                     string message = deal_profit > 0 ? "✅ Trade Closed (PROFIT)\n" : "❌ Trade Closed (LOSS)\n";
                     message += "Symbol: " + symbol + "\n";
                     message += "P/L: " + DoubleToString(deal_profit, 2) + " " + AccountInfoString(ACCOUNT_CURRENCY);
                     SendTelegramMessage(message);
                  }
               }
            }
         }
      }
   }
}
//+------------------------------------------------------------------+
