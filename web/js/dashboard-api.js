// API Integration for Dashboard
const API_BASE_URL = 'http://localhost:5000/api';

// API Service
const API = {
    async getStatus() {
        try {
            const response = await fetch(`${API_BASE_URL}/status`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching status:', error);
            return null;
        }
    },

    async getMetrics() {
        try {
            const response = await fetch(`${API_BASE_URL}/metrics`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching metrics:', error);
            return null;
        }
    },

    async getEquityCurve() {
        try {
            const response = await fetch(`${API_BASE_URL}/equity`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching equity curve:', error);
            return null;
        }
    },

    async getTrades(limit = 50) {
        try {
            const response = await fetch(`${API_BASE_URL}/trades?limit=${limit}`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching trades:', error);
            return [];
        }
    },

    async getPositions() {
        try {
            const response = await fetch(`${API_BASE_URL}/positions`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching positions:', error);
            return [];
        }
    },

    async getStrategies() {
        try {
            const response = await fetch(`${API_BASE_URL}/strategies`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching strategies:', error);
            return [];
        }
    },

    async startBot(strategy) {
        try {
            const response = await fetch(`${API_BASE_URL}/bot/start`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ strategy })
            });
            return await response.json();
        } catch (error) {
            console.error('Error starting bot:', error);
            return { success: false, message: error.message };
        }
    },

    async stopBot() {
        try {
            const response = await fetch(`${API_BASE_URL}/bot/stop`, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                }
            });
            return await response.json();
        } catch (error) {
            console.error('Error stopping bot:', error);
            return { success: false, message: error.message };
        }
    },

    async getPerformance() {
        try {
            const response = await fetch(`${API_BASE_URL}/performance`);
            return await response.json();
        } catch (error) {
            console.error('Error fetching performance:', error);
            return [];
        }
    }
};

// Update dashboard with real data
async function updateDashboardWithAPI() {
    // Update bot status
    const status = await API.getStatus();
    if (status) {
        const statusBadge = document.querySelector('.status-badge');
        const statusText = document.querySelector('.bot-status span');
        if (statusBadge && statusText) {
            if (status.status === 'running') {
                statusBadge.classList.remove('bg-gray');
                statusBadge.classList.add('bg-success');
                statusText.textContent = 'Bot Running';
            } else {
                statusBadge.classList.remove('bg-success');
                statusBadge.classList.add('bg-gray');
                statusText.textContent = 'Bot Stopped';
            }
        }
    }

    // Update metrics
    const metrics = await API.getMetrics();
    if (metrics) {
        updateMetricsDisplay(metrics);
    }

    // Update equity chart
    const equityData = await API.getEquityCurve();
    if (equityData && window.equityChart) {
        updateEquityChart(equityData);
    }

    // Update trade history
    const trades = await API.getTrades(50);
    if (trades) {
        updateTradeHistory(trades);
    }

    // Update positions
    const positions = await API.getPositions();
    if (positions) {
        updatePositions(positions);
    }

    // Update strategies
    const strategies = await API.getStrategies();
    if (strategies) {
        updateStrategiesList(strategies);
    }
}

function updateMetricsDisplay(metrics) {
    // Update stat cards
    const balanceEl = document.querySelector('#balance-value');
    const pnlEl = document.querySelector('#pnl-value');
    const winRateEl = document.querySelector('#winrate-value');
    const tradesEl = document.querySelector('#trades-value');

    if (balanceEl) balanceEl.textContent = '$' + metrics.balance.toLocaleString();
    if (pnlEl) {
        pnlEl.textContent = (metrics.pnl >= 0 ? '+' : '') + '$' + metrics.pnl.toLocaleString();
        pnlEl.className = metrics.pnl >= 0 ? 'text-success' : 'text-danger';
    }
    if (winRateEl) winRateEl.textContent = metrics.win_rate.toFixed(1) + '%';
    if (tradesEl) tradesEl.textContent = metrics.total_trades;

    // Update performance metrics
    const sharpeEl = document.querySelector('#sharpe-value');
    const profitFactorEl = document.querySelector('#profit-factor-value');
    const maxDdEl = document.querySelector('#max-dd-value');
    const avgWinEl = document.querySelector('#avg-win-value');
    const avgLossEl = document.querySelector('#avg-loss-value');
    const returnEl = document.querySelector('#return-value');

    if (sharpeEl) sharpeEl.textContent = metrics.sharpe_ratio.toFixed(2);
    if (profitFactorEl) profitFactorEl.textContent = metrics.profit_factor.toFixed(2);
    if (maxDdEl) maxDdEl.textContent = metrics.max_drawdown.toFixed(2) + '%';
    if (avgWinEl) avgWinEl.textContent = '$' + metrics.average_win.toFixed(2);
    if (avgLossEl) avgLossEl.textContent = '$' + metrics.average_loss.toFixed(2);
    if (returnEl) returnEl.textContent = metrics.total_return.toFixed(2) + '%';
}

function updateEquityChart(data) {
    if (!window.equityChart) return;
    
    window.equityChart.data.labels = data.labels;
    window.equityChart.data.datasets[0].data = data.data;
    window.equityChart.update('none'); // Update without animation
}

function updateTradeHistory(trades) {
    const tbody = document.querySelector('#trade-history-tbody');
    if (!tbody) return;

    tbody.innerHTML = '';
    
    trades.forEach(trade => {
        const row = document.createElement('tr');
        const pnlClass = trade.pnl >= 0 ? 'text-success' : 'text-danger';
        const pnlSign = trade.pnl >= 0 ? '+' : '';
        
        row.innerHTML = `
            <td>${trade.symbol}</td>
            <td><span class="badge ${trade.type === 'BUY' || trade.type === 'LONG' ? 'badge-success' : 'badge-danger'}">${trade.type}</span></td>
            <td>$${trade.entry_price.toFixed(5)}</td>
            <td>$${trade.exit_price.toFixed(5)}</td>
            <td>${trade.quantity}</td>
            <td class="${pnlClass}">${pnlSign}$${Math.abs(trade.pnl).toFixed(2)}</td>
            <td>${new Date(trade.timestamp).toLocaleString()}</td>
        `;
        tbody.appendChild(row);
    });
}

function updatePositions(positions) {
    const tbody = document.querySelector('#positions-tbody');
    if (!tbody) return;

    if (positions.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="text-center text-muted">No open positions</td></tr>';
        return;
    }

    tbody.innerHTML = '';
    
    positions.forEach(position => {
        const row = document.createElement('tr');
        const pnlClass = position.unrealized_pnl >= 0 ? 'text-success' : 'text-danger';
        const pnlSign = position.unrealized_pnl >= 0 ? '+' : '';
        
        row.innerHTML = `
            <td>${position.symbol}</td>
            <td><span class="badge ${position.type === 'BUY' ? 'badge-success' : 'badge-danger'}">${position.type}</span></td>
            <td>$${position.entry_price.toFixed(5)}</td>
            <td>${position.quantity}</td>
            <td class="${pnlClass}">${pnlSign}$${Math.abs(position.unrealized_pnl).toFixed(2)}</td>
            <td>${new Date(position.entry_time).toLocaleString()}</td>
        `;
        tbody.appendChild(row);
    });
}

function updateStrategiesList(strategies) {
    const container = document.querySelector('#strategies-list');
    if (!container) return;

    container.innerHTML = '';
    
    strategies.forEach(strategy => {
        const item = document.createElement('div');
        item.className = 'strategy-item';
        item.innerHTML = `
            <div class="strategy-info">
                <h4>${strategy.name}</h4>
                <p>${strategy.description}</p>
            </div>
            <div class="strategy-actions">
                <span class="badge ${strategy.active ? 'badge-success' : 'badge-secondary'}">${strategy.active ? 'Active' : 'Inactive'}</span>
                <button class="btn btn-sm btn-secondary" onclick="toggleStrategy('${strategy.code}')">
                    ${strategy.active ? 'Stop' : 'Start'}
                </button>
            </div>
        `;
        container.appendChild(item);
    });
}

// Bot control functions
async function handleStartBot() {
    const result = await API.startBot('grid');
    if (result.success) {
        showNotification('Success', result.message, 'success');
        await updateDashboardWithAPI();
    } else {
        showNotification('Error', result.message, 'error');
    }
}

async function handleStopBot() {
    const result = await API.stopBot();
    if (result.success) {
        showNotification('Success', result.message, 'success');
        await updateDashboardWithAPI();
    } else {
        showNotification('Error', result.message, 'error');
    }
}

async function toggleStrategy(strategyCode) {
    // Implementation for strategy toggle
    const strategies = await API.getStrategies();
    const strategy = strategies.find(s => s.code === strategyCode);
    
    if (strategy && strategy.active) {
        await handleStopBot();
    } else {
        await API.startBot(strategyCode);
        await updateDashboardWithAPI();
    }
}

function showNotification(title, message, type) {
    // Simple notification - you can enhance this with a proper notification library
    alert(`${title}: ${message}`);
}

// Initialize API integration
document.addEventListener('DOMContentLoaded', function() {
    // Initial data load
    updateDashboardWithAPI();
    
    // Set up auto-refresh every 30 seconds
    setInterval(updateDashboardWithAPI, 30000);
    
    // Add event listeners for bot controls
    const startBtn = document.querySelector('#start-bot-btn');
    const stopBtn = document.querySelector('#stop-bot-btn');
    
    if (startBtn) startBtn.addEventListener('click', handleStartBot);
    if (stopBtn) stopBtn.addEventListener('click', handleStopBot);
});
