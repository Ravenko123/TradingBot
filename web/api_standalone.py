"""
Simple Flask API for Ultima Dashboard
Standalone version - doesn't require all bot modules
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime
import pandas as pd

app = Flask(__name__)
CORS(app)

# Paths - look for data in parent directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRADES_FILE = os.path.join(BASE_DIR, "trades.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "results.csv")
TELEGRAM_STATE_FILE = os.path.join(BASE_DIR, "telegram_state.json")

print(f"Looking for trades in: {TRADES_FILE}")
print(f"Base directory: {BASE_DIR}")


def load_telegram_state():
    """Load bot state from telegram_state.json"""
    try:
        if os.path.exists(TELEGRAM_STATE_FILE):
            with open(TELEGRAM_STATE_FILE, 'r') as f:
                return json.load(f)
        return {"is_running": False, "strategy": None}
    except Exception as e:
        print(f"Error loading telegram state: {e}")
        return {"is_running": False, "strategy": None}


def load_trades():
    """Load trades from trades.csv"""
    try:
        if os.path.exists(TRADES_FILE):
            df = pd.read_csv(TRADES_FILE)
            if not df.empty:
                return df
        return pd.DataFrame()
    except Exception as e:
        print(f"Error loading trades: {e}")
        return pd.DataFrame()


def calculate_metrics(trades_df):
    """Calculate performance metrics from trades"""
    if trades_df.empty:
        return {
            "total_trades": 0,
            "winning_trades": 0,
            "losing_trades": 0,
            "win_rate": 0,
            "total_pnl": 0,
            "average_win": 0,
            "average_loss": 0,
            "profit_factor": 0,
            "sharpe_ratio": 0,
            "max_drawdown": 0,
            "total_return": 0
        }
    
    # Ensure pnl column exists
    pnl_col = None
    if 'pnl' in trades_df.columns:
        pnl_col = 'pnl'
    elif 'profit' in trades_df.columns:
        pnl_col = 'profit'
    elif 'PnL' in trades_df.columns:
        pnl_col = 'PnL'
    else:
        # Try to create pnl from entry/exit prices
        if 'entry_price' in trades_df.columns and 'exit_price' in trades_df.columns:
            pnl_col = 'entry_price'  # Will handle differently below
        else:
            return {
                "total_trades": len(trades_df),
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0,
                "total_pnl": 0,
                "average_win": 0,
                "average_loss": 0,
                "profit_factor": 0,
                "sharpe_ratio": 0,
                "max_drawdown": 0,
                "total_return": 0
            }
    
    total_trades = len(trades_df)
    
    # Get pnl values
    if pnl_col == 'entry_price':
        pnl_values = (trades_df['exit_price'] - trades_df['entry_price']) * trades_df['quantity']
    else:
        pnl_values = trades_df[pnl_col]
    
    winning_trades = len(pnl_values[pnl_values > 0])
    losing_trades = len(pnl_values[pnl_values < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = pnl_values.sum()
    
    wins = pnl_values[pnl_values > 0]
    losses = pnl_values[pnl_values < 0]
    
    average_win = wins.mean() if len(wins) > 0 else 0
    average_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    if len(pnl_values) > 1:
        sharpe_ratio = (pnl_values.mean() / pnl_values.std()) * (252 ** 0.5) if pnl_values.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cumulative_pnl = pnl_values.cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Calculate total return
    starting_balance = 10000
    total_return = (total_pnl / starting_balance * 100) if starting_balance > 0 else 0
    
    return {
        "total_trades": int(total_trades),
        "winning_trades": int(winning_trades),
        "losing_trades": int(losing_trades),
        "win_rate": round(win_rate, 2),
        "total_pnl": round(total_pnl, 2),
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "profit_factor": round(profit_factor, 2),
        "sharpe_ratio": round(sharpe_ratio, 2),
        "max_drawdown": round(max_drawdown, 2),
        "total_return": round(total_return, 2)
    }


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200


@app.route('/api/status', methods=['GET'])
def get_status():
    """Get bot status"""
    state = load_telegram_state()
    return jsonify({
        "status": "running" if state.get("is_running", False) else "stopped",
        "active_strategy": state.get("strategy", None),
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get current bot metrics"""
    trades_df = load_trades()
    metrics = calculate_metrics(trades_df)
    
    starting_balance = 10000
    current_balance = starting_balance + metrics['total_pnl']
    
    return jsonify({
        "balance": round(current_balance, 2),
        "pnl": metrics['total_pnl'],
        "pnl_percentage": round((metrics['total_pnl'] / starting_balance * 100), 2) if starting_balance > 0 else 0,
        "win_rate": metrics['win_rate'],
        "total_trades": metrics['total_trades'],
        "sharpe_ratio": metrics['sharpe_ratio'],
        "profit_factor": metrics['profit_factor'],
        "max_drawdown": metrics['max_drawdown'],
        "total_return": metrics['total_return'],
        "average_win": metrics['average_win'],
        "average_loss": metrics['average_loss'],
        "timestamp": datetime.now().isoformat()
    })


@app.route('/api/equity', methods=['GET'])
def get_equity_curve():
    """Get equity curve data"""
    trades_df = load_trades()
    
    if trades_df.empty:
        return jsonify({"labels": [], "data": []})
    
    # Get timestamp column
    timestamp_col = None
    if 'timestamp' in trades_df.columns:
        timestamp_col = 'timestamp'
    elif 'exit_time' in trades_df.columns:
        timestamp_col = 'exit_time'
    elif 'time' in trades_df.columns:
        timestamp_col = 'time'
    
    if timestamp_col:
        trades_df[timestamp_col] = pd.to_datetime(trades_df[timestamp_col])
        trades_df = trades_df.sort_values(timestamp_col)
    
    # Get pnl column
    pnl_col = 'pnl' if 'pnl' in trades_df.columns else ('profit' if 'profit' in trades_df.columns else 'PnL')
    
    # Calculate cumulative PnL
    starting_balance = 10000
    trades_df['cumulative_pnl'] = trades_df[pnl_col].cumsum()
    trades_df['equity'] = starting_balance + trades_df['cumulative_pnl']
    
    # Prepare data for chart
    if timestamp_col:
        labels = trades_df[timestamp_col].dt.strftime('%Y-%m-%d %H:%M').tolist()
    else:
        labels = list(range(len(trades_df)))
    
    data = trades_df['equity'].tolist()
    
    return jsonify({
        "labels": labels,
        "data": data
    })


@app.route('/api/trades', methods=['GET'])
def get_trades():
    """Get trade history"""
    limit = request.args.get('limit', 50, type=int)
    trades_df = load_trades()
    
    if trades_df.empty:
        return jsonify([])
    
    # Sort by most recent
    timestamp_col = None
    if 'timestamp' in trades_df.columns:
        timestamp_col = 'timestamp'
    elif 'exit_time' in trades_df.columns:
        timestamp_col = 'exit_time'
    elif 'time' in trades_df.columns:
        timestamp_col = 'time'
    
    if timestamp_col:
        trades_df = trades_df.sort_values(timestamp_col, ascending=False)
    
    trades_df = trades_df.head(limit)
    
    # Convert to list of dicts
    trades = []
    for _, row in trades_df.iterrows():
        try:
            trade = {
                "symbol": str(row.get('symbol', row.get('Symbol', 'N/A'))),
                "type": str(row.get('type', row.get('Type', row.get('direction', 'N/A')))),
                "entry_price": float(row.get('entry_price', row.get('Entry_Price', 0))),
                "exit_price": float(row.get('exit_price', row.get('Exit_Price', 0))),
                "quantity": float(row.get('quantity', row.get('Quantity', 0))),
                "pnl": float(row.get('pnl', row.get('profit', row.get('PnL', 0)))),
                "timestamp": str(row.get('timestamp', row.get('exit_time', row.get('time', 'N/A'))))
            }
            trades.append(trade)
        except Exception as e:
            print(f"Error processing trade row: {e}")
            continue
    
    return jsonify(trades)


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get open positions (placeholder)"""
    return jsonify([])


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get available strategies"""
    strategies = [
        {
            "name": "Grid Trading",
            "code": "grid",
            "active": True,
            "description": "Grid Trading strategy with ATR spacing"
        },
        {
            "name": "ICT SMC",
            "code": "ict_smc",
            "active": False,
            "description": "Inner Circle Trader & Smart Money Concepts"
        },
        {
            "name": "HFT Scalper",
            "code": "hft_scalper",
            "active": False,
            "description": "High-Frequency Trading Scalper"
        },
        {
            "name": "Mean Reversion",
            "code": "mean_reversion",
            "active": False,
            "description": "Mean Reversion Strategy"
        }
    ]
    return jsonify(strategies)


@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the bot"""
    try:
        data = request.json or {}
        strategy = data.get('strategy', 'grid')
        
        state = load_telegram_state()
        state['is_running'] = True
        state['strategy'] = strategy
        
        with open(TELEGRAM_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        
        return jsonify({
            "success": True,
            "message": f"Bot started with {strategy} strategy"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/bot/stop', methods=['POST'])
def stop_bot():
    """Stop the bot"""
    try:
        state = load_telegram_state()
        state['is_running'] = False
        
        with open(TELEGRAM_STATE_FILE, 'w') as f:
            json.dump(state, f, indent=4)
        
        return jsonify({
            "success": True,
            "message": "Bot stopped successfully"
        })
    except Exception as e:
        return jsonify({
            "success": False,
            "message": str(e)
        }), 500


@app.route('/api/performance', methods=['GET'])
def get_performance():
    """Get performance metrics by symbol"""
    trades_df = load_trades()
    
    if trades_df.empty or 'symbol' not in trades_df.columns:
        return jsonify([])
    
    performance = []
    for symbol in trades_df['symbol'].unique():
        symbol_trades = trades_df[trades_df['symbol'] == symbol]
        metrics = calculate_metrics(symbol_trades)
        
        performance.append({
            "symbol": symbol,
            "trades": metrics['total_trades'],
            "win_rate": metrics['win_rate'],
            "pnl": metrics['total_pnl'],
            "profit_factor": metrics['profit_factor']
        })
    
    return jsonify(performance)


if __name__ == '__main__':
    print("\n" + "="*60)
    print("Ultima API Server Starting")
    print("="*60)
    print(f"Base Directory: {BASE_DIR}")
    print(f"Trades File: {TRADES_FILE}")
    print(f"Files Exist:")
    print(f"  - trades.csv: {os.path.exists(TRADES_FILE)}")
    print(f"  - telegram_state.json: {os.path.exists(TELEGRAM_STATE_FILE)}")
    print("="*60)
    print("API running on http://0.0.0.0:5000")
    print("Endpoints:")
    print("  - GET  /api/health")
    print("  - GET  /api/status")
    print("  - GET  /api/metrics")
    print("  - GET  /api/equity")
    print("  - GET  /api/trades")
    print("  - POST /api/bot/start")
    print("  - POST /api/bot/stop")
    print("="*60 + "\n")
    
    app.run(host='0.0.0.0', port=5000, debug=True)
