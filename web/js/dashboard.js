// Dashboard functionality
let botActive = true;
let refreshInterval;

document.addEventListener('DOMContentLoaded', function() {
    initializeDashboard();
    startAutoRefresh();
});

function initializeDashboard() {
    initializeEquityChart();
    loadTradeHistory();
    updateDashboardData();
    initializeEventListeners();
}

function initializeEquityChart() {
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;

    // Sample data - in real implementation, fetch from API
    const dates = generateDateLabels(30);
    const equityData = generateEquityData(10000, 30);

    new Chart(ctx, {
        type: 'line',
        data: {
            labels: dates,
            datasets: [{
                label: 'Account Balance',
                data: equityData,
                borderColor: '#00d4aa',
                backgroundColor: 'rgba(0, 212, 170, 0.1)',
                tension: 0.4,
                fill: true,
                pointRadius: 0,
                pointHoverRadius: 6,
                pointBackgroundColor: '#00d4aa',
                pointBorderColor: '#fff',
                pointBorderWidth: 2
            }]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            interaction: {
                intersect: false,
                mode: 'index'
            },
            plugins: {
                legend: {
                    display: false
                },
                tooltip: {
                    backgroundColor: '#14181f',
                    titleColor: '#a0aec0',
                    bodyColor: '#fff',
                    borderColor: '#2d3748',
                    borderWidth: 1,
                    padding: 12,
                    displayColors: false,
                    callbacks: {
                        label: function(context) {
                            return '$' + context.parsed.y.toLocaleString('en-US', {
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2
                            });
                        }
                    }
                }
            },
            scales: {
                y: {
                    beginAtZero: false,
                    grid: {
                        color: 'rgba(255, 255, 255, 0.05)'
                    },
                    ticks: {
                        color: '#a0aec0',
                        callback: function(value) {
                            return '$' + value.toLocaleString('en-US');
                        }
                    }
                },
                x: {
                    grid: {
                        display: false
                    },
                    ticks: {
                        color: '#a0aec0',
                        maxTicksLimit: 10
                    }
                }
            }
        }
    });
}

function generateDateLabels(days) {
    const labels = [];
    const today = new Date();
    for (let i = days - 1; i >= 0; i--) {
        const date = new Date(today);
        date.setDate(date.getDate() - i);
        labels.push(date.toLocaleDateString('en-US', { month: 'short', day: 'numeric' }));
    }
    return labels;
}

function generateEquityData(initialBalance, days) {
    const data = [initialBalance];
    let balance = initialBalance;
    
    for (let i = 1; i < days; i++) {
        // Simulate realistic growth with some volatility
        const dailyReturn = 0.05 + (Math.random() * 0.1 - 0.02); // 5% avg daily with volatility
        balance = balance * (1 + dailyReturn);
        data.push(balance);
    }
    
    return data;
}

function loadTradeHistory() {
    const historyTable = document.getElementById('historyTable');
    if (!historyTable) return;

    // Sample trade history - in real implementation, fetch from API
    const trades = [
        {
            time: '2025-12-20 23:15',
            symbol: 'BTCUSD',
            side: 'BUY',
            size: 0.01,
            entry: 87941.78,
            exit: 88016.83,
            pnl: 75.05,
            return: 0.085
        },
        {
            time: '2025-12-20 22:45',
            symbol: 'EURUSD',
            side: 'SELL',
            size: 1.0,
            entry: 1.17559,
            exit: 1.17409,
            pnl: 150.00,
            return: 12.77
        },
        {
            time: '2025-12-20 21:30',
            symbol: 'BTCUSD',
            side: 'SELL',
            size: 0.01,
            entry: 88242.59,
            exit: 88167.54,
            pnl: 75.05,
            return: 1.28
        },
        {
            time: '2025-12-20 20:15',
            symbol: 'EURUSD',
            side: 'BUY',
            size: 1.0,
            entry: 1.17234,
            exit: 1.17284,
            pnl: 50.00,
            return: 4.26
        }
    ];

    historyTable.innerHTML = trades.map(trade => `
        <tr>
            <td>${trade.time}</td>
            <td><strong>${trade.symbol}</strong></td>
            <td><span class="badge badge-${trade.side === 'BUY' ? 'success' : 'danger'}">${trade.side}</span></td>
            <td>${trade.size}</td>
            <td>$${trade.entry.toFixed(5)}</td>
            <td>$${trade.exit.toFixed(5)}</td>
            <td class="${trade.pnl > 0 ? 'positive' : 'negative'}">$${trade.pnl.toFixed(2)}</td>
            <td class="${trade.return > 0 ? 'positive' : 'negative'}">${trade.return.toFixed(2)}%</td>
        </tr>
    `).join('');
}

async function updateDashboardData() {
    try {
        // In real implementation, fetch from API: /api/bot/status
        const data = {
            balance: 161330,
            pnl: 151330,
            pnlPercent: 1513.3,
            winRate: 87.5,
            totalTrades: 319,
            sharpeRatio: 1.58,
            profitFactor: 3.37,
            maxDrawdown: -12.5,
            avgWin: 534,
            avgLoss: -158,
            largestWin: 3802
        };

        // Update UI
        document.getElementById('balance').textContent = data.balance.toLocaleString('en-US');
        document.getElementById('pnl').textContent = data.pnl.toLocaleString('en-US');
        document.getElementById('balanceChange').textContent = `+${data.pnlPercent.toFixed(1)}%`;
        document.getElementById('pnlChange').textContent = `+${(data.pnl / 10000).toFixed(1)}x`;
        document.getElementById('winRate').textContent = data.winRate.toFixed(1);
        document.getElementById('totalTrades').textContent = data.totalTrades;
        document.getElementById('sharpeRatio').textContent = data.sharpeRatio.toFixed(2);
        document.getElementById('profitFactor').textContent = data.profitFactor.toFixed(2);
        document.getElementById('maxDrawdown').textContent = `${data.maxDrawdown.toFixed(1)}%`;
        document.getElementById('avgWin').textContent = `+$${data.avgWin}`;
        document.getElementById('avgLoss').textContent = `$${data.avgLoss}`;
        document.getElementById('largestWin').textContent = `+$${data.largestWin.toLocaleString('en-US')}`;

    } catch (error) {
        console.error('Error updating dashboard data:', error);
    }
}

function initializeEventListeners() {
    // Refresh button
    const refreshBtn = document.getElementById('refreshBtn');
    if (refreshBtn) {
        refreshBtn.addEventListener('click', () => {
            refreshBtn.querySelector('i').classList.add('fa-spin');
            updateDashboardData();
            loadTradeHistory();
            setTimeout(() => {
                refreshBtn.querySelector('i').classList.remove('fa-spin');
            }, 1000);
        });
    }

    // Start/Stop button
    const startStopBtn = document.getElementById('startStopBtn');
    if (startStopBtn) {
        startStopBtn.addEventListener('click', () => {
            botActive = !botActive;
            updateBotStatus();
        });
    }

    // Sidebar links
    document.querySelectorAll('.sidebar-link').forEach(link => {
        link.addEventListener('click', (e) => {
            document.querySelectorAll('.sidebar-link').forEach(l => l.classList.remove('active'));
            e.currentTarget.classList.add('active');
        });
    });
}

function updateBotStatus() {
    const statusIndicator = document.querySelector('.status-indicator');
    const statusText = document.querySelector('.bot-status span:last-child');
    const startStopBtn = document.getElementById('startStopBtn');

    if (botActive) {
        statusIndicator.classList.add('status-active');
        statusIndicator.classList.remove('status-inactive');
        statusText.textContent = 'Bot Active';
        startStopBtn.innerHTML = '<i class="fas fa-stop"></i> Stop Bot';
        startStopBtn.classList.remove('btn-primary');
        startStopBtn.classList.add('btn-outline');
    } else {
        statusIndicator.classList.remove('status-active');
        statusIndicator.classList.add('status-inactive');
        statusText.textContent = 'Bot Stopped';
        startStopBtn.innerHTML = '<i class="fas fa-play"></i> Start Bot';
        startStopBtn.classList.remove('btn-outline');
        startStopBtn.classList.add('btn-primary');
    }
}

function startAutoRefresh() {
    // Refresh data every 30 seconds
    refreshInterval = setInterval(() => {
        if (botActive) {
            updateDashboardData();
        }
    }, 30000);
}

function toggleStrategy(strategyId) {
    console.log('Toggle strategy:', strategyId);
    // In real implementation, call API to pause/resume strategy
}

function configureStrategy(strategyId) {
    console.log('Configure strategy:', strategyId);
    // In real implementation, open configuration modal
}

// Cleanup on page unload
window.addEventListener('beforeunload', () => {
    if (refreshInterval) {
        clearInterval(refreshInterval);
    }
});
