"""
Flask API backend for trading bot dashboard
Provides real-time metrics, trades, and bot control
"""

from flask import Flask, jsonify, request
from flask_cors import CORS
import json
import os
import sys
from datetime import datetime, timedelta
import pandas as pd

# Add bot directory to path to import bot modules
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(SCRIPT_DIR)
BOT_DIR = os.path.join(PARENT_DIR, "bot")
sys.path.insert(0, BOT_DIR)
sys.path.insert(0, PARENT_DIR)

try:
    from config.settings import SETTINGS
except ImportError:
    SETTINGS = {"account": {"initial_balance": 10000}}

try:
    from core.logger import Logger
except ImportError:
    class Logger:
        def __init__(self, name):
            self.name = name
        def error(self, msg):
            print(f"ERROR: {msg}")
        def info(self, msg):
            print(f"INFO: {msg}")

app = Flask(__name__)
CORS(app)  # Enable CORS for frontend access

# Paths - use parent directory
BASE_DIR = PARENT_DIR
TRADES_FILE = os.path.join(BASE_DIR, "trades.csv")
RESULTS_FILE = os.path.join(BASE_DIR, "results.csv")
TELEGRAM_STATE_FILE = os.path.join(BASE_DIR, "telegram_state.json")

logger = Logger(__name__)


def load_telegram_state():
    """Load bot state from telegram_state.json"""
    try:
        if os.path.exists(TELEGRAM_STATE_FILE):
            with open(TELEGRAM_STATE_FILE, 'r') as f:
                return json.load(f)
        return {"is_running": False, "strategy": None}
    except Exception as e:
        logger.error(f"Error loading telegram state: {e}")
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
        logger.error(f"Error loading trades: {e}")
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
    if 'pnl' not in trades_df.columns and 'profit' in trades_df.columns:
        trades_df['pnl'] = trades_df['profit']
    
    total_trades = len(trades_df)
    winning_trades = len(trades_df[trades_df['pnl'] > 0])
    losing_trades = len(trades_df[trades_df['pnl'] < 0])
    
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0
    total_pnl = trades_df['pnl'].sum()
    
    wins = trades_df[trades_df['pnl'] > 0]['pnl']
    losses = trades_df[trades_df['pnl'] < 0]['pnl']
    
    average_win = wins.mean() if len(wins) > 0 else 0
    average_loss = abs(losses.mean()) if len(losses) > 0 else 0
    
    gross_profit = wins.sum() if len(wins) > 0 else 0
    gross_loss = abs(losses.sum()) if len(losses) > 0 else 0
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else 0
    
    # Calculate Sharpe ratio (simplified)
    if len(trades_df) > 1:
        returns = trades_df['pnl']
        sharpe_ratio = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
    else:
        sharpe_ratio = 0
    
    # Calculate max drawdown
    cumulative_pnl = trades_df['pnl'].cumsum()
    running_max = cumulative_pnl.expanding().max()
    drawdown = cumulative_pnl - running_max
    max_drawdown = abs(drawdown.min()) if len(drawdown) > 0 else 0
    
    # Calculate total return (assuming starting balance from settings)
    starting_balance = SETTINGS.get('account', {}).get('initial_balance', 10000)
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
    
    # Get account balance (simplified - would need MT5 integration for real balance)
    starting_balance = SETTINGS.get('account', {}).get('initial_balance', 10000)
    current_balance = starting_balance + metrics['total_pnl']
    
    return jsonify({
        "balance": round(current_balance, 2),
        "pnl": metrics['total_pnl'],
        "pnl_percentage": round((metrics['total_pnl'] / starting_balance * 100), 2),
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
    
    # Ensure we have timestamp column
    if 'timestamp' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['timestamp'])
        trades_df = trades_df.sort_values('timestamp')
    elif 'exit_time' in trades_df.columns:
        trades_df['timestamp'] = pd.to_datetime(trades_df['exit_time'])
        trades_df = trades_df.sort_values('timestamp')
    
    # Calculate cumulative PnL
    starting_balance = SETTINGS.get('account', {}).get('initial_balance', 10000)
    trades_df['cumulative_pnl'] = trades_df['pnl'].cumsum()
    trades_df['equity'] = starting_balance + trades_df['cumulative_pnl']
    
    # Prepare data for chart
    labels = trades_df['timestamp'].dt.strftime('%Y-%m-%d %H:%M').tolist()
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
    
    # Sort by most recent and limit
    if 'timestamp' in trades_df.columns:
        trades_df = trades_df.sort_values('timestamp', ascending=False)
    elif 'exit_time' in trades_df.columns:
        trades_df = trades_df.sort_values('exit_time', ascending=False)
    
    trades_df = trades_df.head(limit)
    
    # Convert to list of dicts
    trades = []
    for _, row in trades_df.iterrows():
        trade = {
            "symbol": row.get('symbol', 'N/A'),
            "type": row.get('type', row.get('direction', 'N/A')),
            "entry_price": round(row.get('entry_price', 0), 5),
            "exit_price": round(row.get('exit_price', 0), 5),
            "quantity": round(row.get('quantity', row.get('position_size', 0)), 2),
            "pnl": round(row.get('pnl', row.get('profit', 0)), 2),
            "timestamp": row.get('timestamp', row.get('exit_time', 'N/A'))
        }
        trades.append(trade)
    
    return jsonify(trades)


@app.route('/api/positions', methods=['GET'])
def get_positions():
    """Get open positions (placeholder - would need MT5 integration)"""
    # This would require real-time MT5 connection
    # For now, return empty array
    return jsonify([])


@app.route('/api/strategies', methods=['GET'])
def get_strategies():
    """Get available strategies and their status"""
    from config.strategy_config import STRATEGY_CONFIGS
    
    strategies = []
    state = load_telegram_state()
    active_strategy = state.get("strategy", None)
    
    for strategy_name, config in STRATEGY_CONFIGS.items():
        if strategy_name in ['grid', 'mean_reversion', 'hft_scalper', 'supertrend', 'ict_smc']:
            strategies.append({
                "name": strategy_name.replace('_', ' ').title(),
                "code": strategy_name,
                "active": strategy_name == active_strategy,
                "description": f"{strategy_name.replace('_', ' ').title()} trading strategy"
            })
    
    return jsonify(strategies)


@app.route('/api/bot/start', methods=['POST'])
def start_bot():
    """Start the bot (placeholder - would need bot integration)"""
    try:
        data = request.json
        strategy = data.get('strategy', 'grid')
        
        # This would require actual bot control
        # For now, just update state file
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
    """Stop the bot (placeholder - would need bot integration)"""
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
    
    if trades_df.empty:
        return jsonify([])
    
    # Group by symbol
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


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    })


if __name__ == '__main__':
    logger.info("Starting trading bot API server...")
    app.run(host='0.0.0.0', port=5000, debug=True)
