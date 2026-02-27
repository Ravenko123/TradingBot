// dashboard logic
// NOTE: perfModeEnabled, enablePerfMode, shouldEnablePerfMode, runPerfProbe
// are declared in main.js which loads first — do NOT redeclare here.

const API_BASE_URL = 'http://localhost:5000/api';
let equityChart = null;
let startStopPendingAction = null;
let statusSnapshot = null;
let heartbeatTicker = null;

// backtest animation state
let btChart = null;
let btSeries = null;
let btAnimTimer = null;
let btPaused = false;
let btStopped = false;
let btCandles = [];
let btTrades = [];
let btMarkers = [];
let btIdx = 0;
let btStats = { trades: 0, wins: 0, pnl: 0, balance: 10000 };
let btSlLine = null;
let btTpLine = null;
let btEntryLine = null;
let btTradeLines = [];
let btManualPanUntil = 0;
let btEquityChart = null;
let btRunInProgress = false;
let btLiveEquityCurve = [];
let btPairAnalyticsRows = [];
let btReportSymbol = 'EURUSD';
let btReportDays = 60;
let btGrossProfit = 0;
let btGrossLoss = 0;
let btEquitySampleCounter = 0;

function getToken() {
    // Support both old and new key
    return localStorage.getItem('zenith_token') || localStorage.getItem('ultima_token');
}

function requireAuth() {
    if (!getToken()) window.location.href = 'login.html';
}

async function apiFetch(path, options = {}) {
    const token = getToken();
    const headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
    if (token) headers['Authorization'] = `Bearer ${token}`;

    try {
        const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
        if (response.status === 401) {
            localStorage.removeItem('zenith_token');
            localStorage.removeItem('ultima_token');
            window.location.href = 'login.html';
            return null;
        }
        if (!response.ok) return null;
        return await response.json();
    } catch (e) {
        console.error('API error:', path, e);
        return null;
    }
}

async function apiFetchWithMeta(path, options = {}) {
    const token = getToken();
    const headers = Object.assign({ 'Content-Type': 'application/json' }, options.headers || {});
    if (token) headers['Authorization'] = `Bearer ${token}`;

    try {
        const response = await fetch(`${API_BASE_URL}${path}`, { ...options, headers });
        if (response.status === 401) {
            localStorage.removeItem('zenith_token');
            localStorage.removeItem('ultima_token');
            window.location.href = 'login.html';
            return { ok: false, status: 401, data: null };
        }

        let data = null;
        try {
            data = await response.json();
        } catch (_) {
            data = null;
        }
        return { ok: response.ok, status: response.status, data };
    } catch (e) {
        console.error('API error:', path, e);
        return { ok: false, status: 0, data: null };
    }
}

function isMt5UnavailableError(payload) {
    if (!payload) return false;
    if (payload.error_code === 'MT5_NOT_AVAILABLE') return true;

    const text = `${payload.message || ''}\n${payload.logs || ''}`.toLowerCase();
    return (
        text.includes('mt5 init failed') ||
        text.includes('cannot start without mt5 connection') ||
        text.includes("no module named 'metatrader5'") ||
        text.includes('no module named "metatrader5"')
    );
}

function openMt5HelpPopup(payload = {}) {
    const modal = document.getElementById('mt5HelpModal');
    if (!modal) return;

    const messageEl = document.getElementById('mt5HelpMessage');
    const linkEl = document.getElementById('mt5InstallLink');

    if (messageEl) {
        messageEl.textContent = payload.help_message || 'The bot could not start because MetaTrader 5 is not available on this machine.';
    }
    if (linkEl) {
        linkEl.href = payload.help_url || 'https://www.metatrader5.com/en/download';
    }

    modal.classList.add('is-open');
    modal.setAttribute('aria-hidden', 'false');
}

function closeMt5HelpPopup() {
    const modal = document.getElementById('mt5HelpModal');
    if (!modal) return;
    modal.classList.remove('is-open');
    modal.setAttribute('aria-hidden', 'true');
}

function setStatusText(el, message, isError = false) {
    if (!el) return;
    el.textContent = message;
    el.style.color = isError ? '#ff6b6b' : '#6b7a8d';
}

let botConfigCache = null;

function num(v, digits = 2) {
    const n = Number(v || 0);
    return Number.isFinite(n) ? n.toFixed(digits) : '0.00';
}

function formatTime(value) {
    if (!value) return '—';
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return '—';
    return d.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' });
}

function getAgeSeconds(value) {
    if (!value) return null;
    const d = new Date(value);
    if (Number.isNaN(d.getTime())) return null;
    return Math.max(0, Math.floor((Date.now() - d.getTime()) / 1000));
}

function renderStartStopButton({ isRunning = false, loading = false, action = null } = {}) {
    const btn = document.getElementById('startStopBtn');
    if (!btn) return;

    btn.classList.add('start-stop-btn');
    const icon = isRunning ? 'stop' : 'play';
    const label = isRunning ? 'Stop Bot' : 'Start Bot';
    const loadingLabel = action === 'stop' ? 'Stopping…' : 'Starting…';

    btn.innerHTML = `
        <span class="btn-icon"><i class="fas fa-${icon}"></i></span>
        <span class="btn-label">${loading ? loadingLabel : label}</span>
        <span class="btn-loader" aria-hidden="true"></span>
    `;

    btn.classList.toggle('is-loading', loading);
    btn.classList.toggle('is-stealth-loading', loading);
    if (loading) btn.setAttribute('aria-busy', 'true');
    else btn.removeAttribute('aria-busy');
}

function tickLiveStatusClock() {
    if (!statusSnapshot) return;

    const botLastUpdate = document.getElementById('botLastUpdate');
    const heartbeat = document.getElementById('botHeartbeat');
    const processStatus = document.getElementById('botProcessStatus');

    const statusAge = getAgeSeconds(statusSnapshot.timestamp);
    if (botLastUpdate) {
        if (statusSnapshot.timestamp && statusAge !== null) {
            botLastUpdate.textContent = `${formatTime(statusSnapshot.timestamp)} · ${statusAge}s ago`;
        } else {
            botLastUpdate.textContent = '—';
        }
    }

    const hbSource = statusSnapshot.last_scan_time || statusSnapshot.timestamp;
    const hbAge = getAgeSeconds(hbSource);
    if (heartbeat) {
        if (hbSource && hbAge !== null) {
            heartbeat.textContent = `${formatTime(hbSource)} · ${hbAge}s ago`;
        } else {
            heartbeat.textContent = '—';
        }
    }

    if (processStatus && statusSnapshot.process_alive) {
        if (statusSnapshot.last_scan_time && hbAge !== null) {
            processStatus.textContent = `Active · last scan ${hbAge}s ago`;
        } else {
            processStatus.textContent = 'Active';
        }
    }
}

function startHeartbeatTicker() {
    if (heartbeatTicker) return;
    heartbeatTicker = setInterval(tickLiveStatusClock, 1000);
}

// render backtest results
function renderRunResult(result) {
    if (!result) return 'No result data yet.';

    if (result.mode === 'walk_forward') {
        return `<div class="result-grid">
            <div class="result-item"><span class="label">Mode</span><span class="value">Walk-Forward</span></div>
            <div class="result-item"><span class="label">Periods</span><span class="value">${result.total_periods || 0}</span></div>
            <div class="result-item"><span class="label">Profitable</span><span class="value">${result.profitable_periods || 0}</span></div>
            <div class="result-item"><span class="label">Consistency</span><span class="value">${num(result.consistency)}%</span></div>
            <div class="result-item"><span class="label">Avg Profit / Period</span><span class="value">$${num(result.average_profit_per_period)}</span></div>
        </div>`;
    }

    if (result.mode === 'split') {
        const m = result.test?.metrics || {};
        return `<div class="result-grid">
            <div class="result-item"><span class="label">Mode</span><span class="value">Split (${num(result.split_ratio, 2)})</span></div>
            <div class="result-item"><span class="label">Test Trades</span><span class="value">${m.total_trades || 0}</span></div>
            <div class="result-item"><span class="label">Test Win Rate</span><span class="value">${num(m.win_rate)}%</span></div>
            <div class="result-item"><span class="label">Test Profit</span><span class="value">$${num(m.total_profit)}</span></div>
            <div class="result-item"><span class="label">Profit Factor</span><span class="value">${num(m.profit_factor)}</span></div>
        </div>`;
    }

    if (result.mode === 'monte_carlo') {
        const mc = result.monte_carlo || {};
        const sm = result.standard?.metrics || {};
        return `<div class="result-grid">
            <div class="result-item"><span class="label">Mode</span><span class="value">Monte Carlo</span></div>
            <div class="result-item"><span class="label">Iterations</span><span class="value">${mc.iterations || 0}</span></div>
            <div class="result-item"><span class="label">P10 Balance</span><span class="value">$${num(mc.p10)}</span></div>
            <div class="result-item"><span class="label">P50 Balance</span><span class="value">$${num(mc.p50)}</span></div>
            <div class="result-item"><span class="label">P90 Balance</span><span class="value">$${num(mc.p90)}</span></div>
            <div class="result-item"><span class="label">Base Profit</span><span class="value">$${num(sm.total_profit)}</span></div>
        </div>`;
    }

    const m = result.metrics || {};
    return `<div class="result-grid">
        <div class="result-item"><span class="label">Mode</span><span class="value">Standard</span></div>
        <div class="result-item"><span class="label">Trades</span><span class="value">${m.total_trades || 0}</span></div>
        <div class="result-item"><span class="label">Win Rate</span><span class="value">${num(m.win_rate)}%</span></div>
        <div class="result-item"><span class="label">Total Profit</span><span class="value">$${num(m.total_profit)}</span></div>
        <div class="result-item"><span class="label">Profit Factor</span><span class="value">${num(m.profit_factor)}</span></div>
        <div class="result-item"><span class="label">Max Drawdown</span><span class="value">$${num(m.max_drawdown)}</span></div>
    </div>`;
}

// api calls
async function loadStatus() {
    const status = await apiFetch('/status');
    if (!status) return;
    statusSnapshot = status;

    const indicator = document.querySelector('.status-indicator');
    const statusText = document.getElementById('botStatusText');

    const activityState = document.getElementById('botActivityState');
    const engineStatus = document.getElementById('botEngineStatus');
    const processStatus = document.getElementById('botProcessStatus');
    const strategy = document.getElementById('botStrategy');
    const pid = document.getElementById('botPid');
    const lastUpdate = document.getElementById('lastUpdate');

    const isRunning = status.status === 'running';

    if (isRunning) {
        indicator?.classList.add('status-active');
        indicator?.classList.remove('status-inactive');
        if (statusText) statusText.textContent = 'Bot Running';
    } else {
        indicator?.classList.remove('status-active');
        indicator?.classList.add('status-inactive');
        if (statusText) statusText.textContent = 'Bot Stopped';
    }

    if (!startStopPendingAction) {
        renderStartStopButton({ isRunning, loading: false });
    }

    if (activityState) {
        activityState.textContent = isRunning ? 'Live' : 'Idle';
        activityState.classList.toggle('is-live', isRunning);
    }
    if (engineStatus) {
        if (status.engine_status === 'running_stale') {
            engineStatus.textContent = 'Running (No recent scan)';
        } else if (status.engine_status === 'running') {
            engineStatus.textContent = 'Running';
        } else if (status.engine_status === 'process_only') {
            engineStatus.textContent = 'Process alive (state mismatch)';
        } else {
            engineStatus.textContent = 'Stopped';
        }
    }
    if (processStatus) {
        if (!status.process_alive) {
            processStatus.textContent = 'Not running';
        } else if (typeof status.last_scan_age_s === 'number') {
            processStatus.textContent = `Active · last scan ${status.last_scan_age_s}s ago`;
        } else {
            processStatus.textContent = 'Active';
        }
    }
    if (strategy) strategy.textContent = status.active_strategy || '—';
    if (pid) pid.textContent = status.pid ? String(status.pid) : '—';
    if (lastUpdate) lastUpdate.textContent = formatTime(new Date().toISOString());

    tickLiveStatusClock();
}

function refreshStatCardColors() {
    document.querySelectorAll('.stat-card').forEach(card => {
        const valEl = card.querySelector('.stat-value [id]') || card.querySelector('.stat-value span');
        if (!valEl) return;
        const raw = valEl.textContent.replace(/[^0-9.\-]/g, '');
        const num = parseFloat(raw);
        const isZero = isNaN(num) || num === 0;
        card.classList.toggle('zero-val', isZero);
    });
}

async function loadMetrics() {
    const m = await apiFetch('/metrics');
    if (!m) return;
    const $ = id => document.getElementById(id);
    if ($('balance')) $('balance').textContent = Number(m.balance || 0).toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    if ($('balanceChange')) $('balanceChange').textContent = `${Number(m.pnl_percentage || 0).toFixed(2)}%`;
    if ($('totalTrades')) $('totalTrades').textContent = Number(m.total_trades || 0);
    if ($('winRate')) $('winRate').textContent = Number(m.win_rate || 0).toFixed(1);
    if ($('pnl')) $('pnl').textContent = Number(m.profit_factor || 0).toFixed(2);
    if ($('sharpeRatio')) $('sharpeRatio').textContent = Number(m.sharpe_ratio || 0).toFixed(2);
    if ($('profitFactor')) $('profitFactor').textContent = Number(m.profit_factor || 0).toFixed(2);
    if ($('maxDrawdown')) $('maxDrawdown').textContent = `${Number(m.max_drawdown_pct ?? m.max_drawdown ?? 0).toFixed(2)}`;
    if ($('avgWin')) $('avgWin').textContent = `+$${Number(m.average_win || 0).toFixed(2)}`;
    if ($('avgLoss')) $('avgLoss').textContent = `-$${Number(m.average_loss || 0).toFixed(2)}`;
    if ($('largestWin')) $('largestWin').textContent = '+$0.00';
    refreshStatCardColors();
}

async function loadTrades() {
    const trades = await apiFetch('/trades?limit=50');
    if (!trades) return;
    const table = document.getElementById('historyTable');
    if (!table) return;
    if (!trades.length) { table.innerHTML = '<tr><td colspan="8">No trades yet</td></tr>'; return; }
    table.innerHTML = trades.map(t => `<tr>
        <td>${t.timestamp}</td><td><strong>${t.symbol}</strong></td>
        <td><span class="badge badge-${t.type === 'BUY' ? 'success' : 'danger'}">${t.type}</span></td>
        <td>${t.quantity}</td><td>${t.entry_price}</td><td>${t.exit_price}</td>
        <td class="${t.pnl >= 0 ? 'positive' : 'negative'}">${t.pnl.toFixed(2)}</td>
        <td>${t.pnl >= 0 ? '+' : ''}${t.pnl.toFixed(2)}</td>
    </tr>`).join('');
}

async function loadEquity() {
    const data = await apiFetch('/equity');
    if (!data) return;
    const ctx = document.getElementById('equityChart');
    if (!ctx) return;

    if (!equityChart) {
        equityChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: data.labels,
                datasets: [{
                    label: 'Equity',
                    data: data.data,
                    borderColor: '#00ff87',
                    backgroundColor: (context) => {
                        const chart = context.chart;
                        const { ctx: c, chartArea } = chart;
                        if (!chartArea) return 'transparent';
                        const g = c.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                        g.addColorStop(0, 'rgba(0, 255, 135, 0.12)');
                        g.addColorStop(1, 'rgba(0, 255, 135, 0)');
                        return g;
                    },
                    tension: 0.3, fill: true, pointRadius: 0, borderWidth: 2
                }]
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: { ticks: { color: '#4a5568', maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } }, grid: { color: 'rgba(255,255,255,0.03)' } },
                    y: { ticks: { color: '#4a5568', font: { family: "'JetBrains Mono'", size: 10 } }, grid: { color: 'rgba(255,255,255,0.03)' } }
                }
            }
        });
    } else {
        equityChart.data.labels = data.labels;
        equityChart.data.datasets[0].data = data.data;
        equityChart.update('none');
    }
}

async function handleStartStop() {
    const status = await apiFetch('/status');
    if (!status) return;

    const btn = document.getElementById('startStopBtn');
    const action = status.status === 'running' ? 'stop' : 'start';
    startStopPendingAction = action;
    renderStartStopButton({ isRunning: status.status === 'running', loading: true, action });
    if (btn) {
        btn.disabled = true;
    }

    let resp;
    if (status.status === 'running') {
        resp = await apiFetchWithMeta('/bot/stop', { method: 'POST' });
    } else {
        resp = await apiFetchWithMeta('/bot/start', { method: 'POST', body: JSON.stringify({ strategy: 'ict_smc' }) });
    }

    startStopPendingAction = null;
    if (btn) {
        btn.disabled = false;
    }

    if (!resp?.ok) {
        renderStartStopButton({ isRunning: status.status === 'running', loading: false });
        if (btn) btn.classList.add('is-error');
        setTimeout(() => btn?.classList.remove('is-error'), 500);
        if (action === 'start' && isMt5UnavailableError(resp?.data)) {
            openMt5HelpPopup(resp.data);
        }
        return;
    }
    
    if (btn) {
        btn.classList.add('is-success');
        setTimeout(() => btn?.classList.remove('is-success'), 600);
    }
    
    await loadStatus();
}

    async function loadTelegramConfig() {
        const cfg = await apiFetch('/telegram/config');
        if (!cfg) return;

        const tokenInput = document.getElementById('telegramBotToken');
        const chatInput = document.getElementById('telegramChatId');
        const statusEl = document.getElementById('telegramConfigStatus');

        if (tokenInput) tokenInput.value = '';
        if (chatInput) chatInput.value = cfg.chat_id || '';

        if (statusEl) {
            if (cfg.configured) {
                setStatusText(statusEl, `Telegram connected (${cfg.bot_token_masked || 'token saved'})`);
            } else if (cfg.has_bot_token || cfg.has_chat_id) {
                setStatusText(statusEl, 'Telegram partially configured. Save both token and chat ID.', true);
            } else {
                setStatusText(statusEl, 'Telegram not configured yet.');
            }
        }
    }

    async function saveTelegramConfig() {
        const tokenInput = document.getElementById('telegramBotToken');
        const chatInput = document.getElementById('telegramChatId');
        const statusEl = document.getElementById('telegramConfigStatus');
        const btn = document.getElementById('saveTelegramConfigBtn');

        const botToken = (tokenInput?.value || '').trim();
        const chatId = (chatInput?.value || '').trim();

        if (!botToken || !chatId) {
            setStatusText(statusEl, 'Both Bot Token and Chat ID are required.', true);
            return;
        }

        if (btn) {
            btn.disabled = true;
            btn.classList.add('is-loading');
        }

        const res = await apiFetch('/telegram/config', {
            method: 'POST',
            body: JSON.stringify({ bot_token: botToken, chat_id: chatId }),
        });

        if (btn) {
            btn.disabled = false;
            btn.classList.remove('is-loading');
        }

        if (!res || res.success === false) {
            setStatusText(statusEl, 'Failed to save Telegram login.', true);
            return;
        }

        if (tokenInput) tokenInput.value = '';
        setStatusText(statusEl, 'Telegram login saved. Restart bot to apply.');
    }

async function runBacktest() {
    // kill any running animation
    if (btAnimTimer) { clearTimeout(btAnimTimer); btAnimTimer = null; }

    const symbol = document.getElementById('btSymbol').value;
    const days = parseInt(document.getElementById('btDays').value) || 60;
    const status = document.getElementById('btStatus');
    const runBtn = document.getElementById('runBacktestBtn');
    const pauseBtn = document.getElementById('btPauseBtn');
    const stopBtn = document.getElementById('btStopBtn');

    btRunInProgress = true;
    if (pauseBtn) { pauseBtn.disabled = true; pauseBtn.innerHTML = '<i class="fas fa-pause"></i>'; }
    if (stopBtn) stopBtn.disabled = true;

    setStatusText(status, 'Running real MT5 no-lookahead backtest...');
    if (runBtn) { 
        runBtn.disabled = true; 
        runBtn.classList.add('is-loading');
        runBtn.innerHTML = '<i class="fas fa-hourglass-half"></i> Loading data…'; 
    }

    const riskPct = parseFloat(document.getElementById('btRiskPct')?.value) || 1.0;

    // Always enqueue a REAL persisted run via backtest_improved.py
    try {
        const queued = await apiFetch('/backtest/run', {
            method: 'POST',
            body: JSON.stringify({ symbol, mode: 'standard', split_ratio: 0.7, mc_iterations: 200 })
        });
        if (queued?.success && queued?.run_id) {
            setStatusText(status, `Queued real engine run: ${queued.run_id} (backtest_improved.py)`);
            setTimeout(async () => {
                await loadRuns();
                await loadRobustness();
            }, 1200);
        }
    } catch (e) {
        console.warn('Could not queue real persisted backtest run:', e);
    }

    let data = null;
    try {
        data = await apiFetch('/backtest/simulate', {
            method: 'POST',
            body: JSON.stringify({ symbol, days, risk_pct: riskPct })
        });
        if (!data) {
            throw new Error('No response from server');
        }
    } catch (e) {
        console.error('Backtest fetch failed:', e);
        setStatusText(status, `Error: ${e.message || 'Server connection failed'}`, true);
    }

    if (runBtn) { 
        runBtn.disabled = false; 
        runBtn.innerHTML = '<i class="fas fa-play"></i> Run Backtest';
        runBtn.classList.remove('is-loading'); 
    }

    if (!data || !data.candles || !data.candles.length) {
        btRunInProgress = false;
        const reason = data?.meta?.message || 'Failed to generate backtest data — check configuration';
        setStatusText(status, reason, true);
        if (runBtn) {
            runBtn.classList.add('is-error');
            setTimeout(() => runBtn?.classList.remove('is-error'), 500);
        }
        return;
    }
    
    if (runBtn) {
        runBtn.classList.add('is-success');
        setTimeout(() => runBtn?.classList.remove('is-success'), 600);
    }

    btCandles = data.candles;
    btTrades = data.trades || [];
    btMarkers = [];
    btIdx = 0;
    btPaused = false;
    btStopped = false;
    btStats = { trades: 0, wins: 0, pnl: 0, balance: 10000 };
    btSlLine = null;
    btTpLine = null;
    btEntryLine = null;
    btTradeLines = [];
    btLiveEquityCurve = [10000];
    btPairAnalyticsRows = [];
    btReportSymbol = symbol;
    btReportDays = days;
    btGrossProfit = 0;
    btGrossLoss = 0;
    btEquitySampleCounter = 0;

    // show chart area
    const chartArea = document.getElementById('btChartArea');
    chartArea.style.display = 'block';
    document.getElementById('btChartSymbol').textContent = symbol;
    document.getElementById('btProgress').textContent = '0%';

    // reset controls
    if (pauseBtn) { pauseBtn.innerHTML = '<i class="fas fa-pause"></i>'; pauseBtn.disabled = false; }
    if (stopBtn) stopBtn.disabled = false;

    updateBtLiveStats();
    const backtestNote = data?.meta?.message || '';
    setStatusText(status, backtestNote ? `Data loaded · ${backtestNote}` : 'Data loaded');
    initBtChart();
    await renderBacktestReport(data);
    setStatusText(status, 'Animating candles...');

    setTimeout(animateNextCandle, 300);
}

function renderBtEquityCurve(equityCurve = []) {
    const canvas = document.getElementById('btEquityChart');
    if (!canvas) return;

    const labels = equityCurve.map((_, i) => i + 1);

    if (!btEquityChart) {
        btEquityChart = new Chart(canvas, {
            type: 'line',
            data: {
                labels,
                datasets: [{
                    label: 'Equity',
                    data: equityCurve,
                    borderColor: '#00ff87',
                    backgroundColor: 'rgba(0,255,135,0.12)',
                    fill: true,
                    tension: 0.22,
                    pointRadius: 0,
                    borderWidth: 2,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: {
                    x: {
                        ticks: { color: '#6b7a8d', maxTicksLimit: 8, font: { family: "'JetBrains Mono'", size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    },
                    y: {
                        ticks: { color: '#6b7a8d', font: { family: "'JetBrains Mono'", size: 10 } },
                        grid: { color: 'rgba(255,255,255,0.03)' },
                    }
                }
            }
        });
    } else {
        btEquityChart.data.labels = labels;
        btEquityChart.data.datasets[0].data = equityCurve;
        btEquityChart.update('none');
    }
}

function renderPairAnalyticsTable(rows = [], selectedSymbol = '', selectedMetrics = null) {
    const table = document.getElementById('btPairAnalyticsTable');
    if (!table) return;

    const merged = [...rows];
    if (selectedSymbol && selectedMetrics) {
        const idx = merged.findIndex((r) => r.symbol === selectedSymbol);
        const row = {
            symbol: selectedSymbol,
            total_trades: Number(selectedMetrics.total_trades || 0),
            win_rate: Number(selectedMetrics.win_rate || 0),
            total_profit: Number(selectedMetrics.total_profit || 0),
            profit_factor: Number(selectedMetrics.profit_factor || 0),
            max_drawdown: Number(selectedMetrics.max_drawdown || 0),
            return_pct: Number(selectedMetrics.return_pct || 0),
            mode: 'live-runner',
        };
        if (idx >= 0) merged[idx] = row;
        else merged.unshift(row);
    }

    if (!merged.length) {
        table.innerHTML = '<tr><td colspan="8">No pair analytics available yet.</td></tr>';
        return;
    }

    table.innerHTML = merged.map((r) => {
        const p = Number(r.total_profit || 0);
        const dd = Number(r.max_drawdown || 0);
        const ret = Number(r.return_pct || 0);
        const isSel = r.symbol === selectedSymbol;
        return `<tr${isSel ? ' style="background: rgba(0,255,135,0.06);"' : ''}>
            <td><strong>${r.symbol || '—'}</strong></td>
            <td>${Number(r.total_trades || 0)}</td>
            <td>${num(r.win_rate)}%</td>
            <td class="${p >= 0 ? 'positive' : 'negative'}">${p >= 0 ? '+' : ''}$${num(p)}</td>
            <td>${num(r.profit_factor)}</td>
            <td class="negative">$${num(dd)}</td>
            <td class="${ret >= 0 ? 'positive' : 'negative'}">${ret >= 0 ? '+' : ''}${num(ret)}%</td>
            <td>${r.mode || '—'}</td>
        </tr>`;
    }).join('');
}

function calculateBtMaxDrawdown(equityCurve = []) {
    if (!Array.isArray(equityCurve) || equityCurve.length < 2) return 0;
    let peak = Number(equityCurve[0] || 0);
    let maxDd = 0;
    for (const point of equityCurve) {
        const value = Number(point || 0);
        if (value > peak) peak = value;
        const dd = peak - value;
        if (dd > maxDd) maxDd = dd;
    }
    return maxDd;
}

function buildLiveBacktestMetrics() {
    const totalTrades = Number(btStats.trades || 0);
    const winRate = totalTrades > 0 ? (btStats.wins / totalTrades * 100) : 0;
    const totalProfit = Number(btStats.pnl || 0);
    const maxDrawdown = calculateBtMaxDrawdown(btLiveEquityCurve);
    const returnPct = ((Number(btStats.balance || 10000) - 10000) / 10000) * 100;
    const profitFactor = btGrossLoss > 0 ? (btGrossProfit / btGrossLoss) : (btGrossProfit > 0 ? 99 : 0);

    return {
        total_trades: totalTrades,
        win_rate: winRate,
        total_profit: totalProfit,
        profit_factor: profitFactor,
        max_drawdown: maxDrawdown,
        return_pct: returnPct,
    };
}

function updateBacktestReportLive(statusLabel = 'Running') {
    const report = document.getElementById('btReport');
    if (!report) return;

    const m = buildLiveBacktestMetrics();
    const set = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = value;
    };

    set('btRepTrades', Number(m.total_trades || 0));
    set('btRepWinRate', `${num(m.win_rate)}%`);
    set('btRepProfit', `${Number(m.total_profit || 0) >= 0 ? '+' : ''}$${num(m.total_profit)}`);
    set('btRepPF', num(m.profit_factor));
    set('btRepDD', `$${num(m.max_drawdown)}`);
    set('btRepReturn', `${Number(m.return_pct || 0) >= 0 ? '+' : ''}${num(m.return_pct)}%`);

    const metaEl = document.getElementById('btReportMeta');
    if (metaEl) {
        metaEl.textContent = `${btReportSymbol} · ${btReportDays}d · ${Number(m.total_trades || 0)} trades · ${statusLabel}`;
    }

    const curve = btLiveEquityCurve.length ? btLiveEquityCurve : [10000];
    renderBtEquityCurve(curve);
    renderPairAnalyticsTable(btPairAnalyticsRows, btReportSymbol, m);
    report.style.display = 'block';
}

async function renderBacktestReport(data) {
    const report = document.getElementById('btReport');
    if (!report || !data) return;

    try {
        const pairData = await apiFetch('/backtest/pairs-analytics');
        btPairAnalyticsRows = pairData?.analytics || [];
    } catch (e) {
        btPairAnalyticsRows = [];
    }

    btReportSymbol = data.symbol || btReportSymbol;
    btReportDays = Number(data.days || btReportDays || 60);
    report.style.display = 'block';
    updateBacktestReportLive('Running');
}

function initBtChart() {
    const container = document.getElementById('btChartContainer');
    if (!container) return;
    container.innerHTML = '';
    btManualPanUntil = 0;

    if (btChart) { btChart.remove(); btChart = null; }

    const chartHeight = Math.max(420, Math.round(container.clientHeight || 560));

    btChart = LightweightCharts.createChart(container, {
        width: container.clientWidth,
        height: chartHeight,
        layout: {
            background: { type: 'solid', color: 'transparent' },
            textColor: '#6b7a8d',
            fontFamily: "'JetBrains Mono', monospace",
            fontSize: 11
        },
        grid: {
            vertLines: { color: 'rgba(255, 255, 255, 0.03)' },
            horzLines: { color: 'rgba(255, 255, 255, 0.03)' }
        },
        crosshair: {
            mode: LightweightCharts.CrosshairMode.Normal,
            vertLine: { color: 'rgba(96, 239, 255, 0.3)', width: 1, labelBackgroundColor: '#1a2535' },
            horzLine: { color: 'rgba(96, 239, 255, 0.3)', width: 1, labelBackgroundColor: '#1a2535' }
        },
        rightPriceScale: { borderColor: 'rgba(255, 255, 255, 0.06)' },
        timeScale: {
            borderColor: 'rgba(255, 255, 255, 0.06)',
            timeVisible: true,
            secondsVisible: false,
            rightOffset: 18,
            barSpacing: 6,
            minBarSpacing: 2
        },
        handleScroll: { mouseWheel: true, pressedMouseMove: true, horzTouchDrag: true, vertTouchDrag: false },
        handleScale: { axisPressedMouseMove: true, mouseWheel: true, pinch: true }
    });

    btSeries = btChart.addCandlestickSeries({
        upColor: '#00ff87',
        downColor: '#ff6b6b',
        borderDownColor: '#ff6b6b',
        borderUpColor: '#00ff87',
        wickDownColor: 'rgba(255, 107, 107, 0.5)',
        wickUpColor: 'rgba(0, 255, 135, 0.5)'
    });

    try {
        btSeries.priceScale().applyOptions({
            autoScale: true,
            scaleMargins: { top: 0.14, bottom: 0.16 },
        });
    } catch (e) {}

    // auto-resize
    new ResizeObserver(entries => {
        if (btChart) {
            const nextHeight = Math.max(420, Math.round(entries[0].contentRect.height || 560));
            btChart.applyOptions({ width: entries[0].contentRect.width, height: nextHeight });
        }
    }).observe(container);

    const markManualPan = () => {
        btManualPanUntil = Date.now() + 3500;
    };
    ['wheel', 'mousedown', 'touchstart'].forEach((eventName) => {
        container.addEventListener(eventName, markManualPan, { passive: true });
    });
}

function trackBacktestPrice(force = false) {
    if (!btSeries) return;
    if (!force && Date.now() < btManualPanUntil) return;

    try {
        btSeries.priceScale().applyOptions({ autoScale: true });
    } catch (e) {}
}

function animateNextCandle() {
    if (!btSeries || !btChart || !btCandles.length) {
        btRunInProgress = false;
        const status = document.getElementById('btStatus');
        setStatusText(status, 'Backtest animation could not start (chart init failed).', true);
        return;
    }

    if (btPaused || btStopped || btIdx >= btCandles.length) {
        if (btIdx >= btCandles.length) finishBacktest();
        return;
    }

    btSeries.update(btCandles[btIdx]);
    checkTradeMarkers(btIdx);

    // progress
    const pct = Math.round((btIdx / btCandles.length) * 100);
    const progEl = document.getElementById('btProgress');
    if (progEl) progEl.textContent = pct + '%';

    btEquitySampleCounter++;
    if (btEquitySampleCounter % 8 === 0) {
        btLiveEquityCurve.push(Number(btStats.balance || 10000));
        updateBacktestReportLive('Running');
    }

    if (Date.now() >= btManualPanUntil) {
        btChart.timeScale().scrollToRealTime();
    }
    trackBacktestPrice();
    btIdx++;

    const speed = parseInt(document.getElementById('btSpeed').value) || 40;
    const delay = Math.max(1, Math.round(300 / speed));
    btAnimTimer = setTimeout(animateNextCandle, delay);
}

function checkTradeMarkers(barIdx) {
    for (const t of btTrades) {
        if (t.entryBar === barIdx) {
            // entry arrow marker
            btMarkers.push({
                time: btCandles[barIdx].time,
                position: t.type === 'BUY' ? 'belowBar' : 'aboveBar',
                color: t.type === 'BUY' ? '#00ff87' : '#ff6b6b',
                shape: t.type === 'BUY' ? 'arrowUp' : 'arrowDown',
                text: t.type
            });
            btSeries.setMarkers(btMarkers.slice());

            // draw SL / TP / Entry price lines
            try {
                btEntryLine = btSeries.createPriceLine({
                    price: t.entryPrice,
                    color: 'rgba(96, 239, 255, 0.5)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    axisLabelVisible: false,
                    title: '',
                });
                btSlLine = btSeries.createPriceLine({
                    price: t.sl,
                    color: 'rgba(255, 107, 107, 0.7)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'SL',
                });
                btTpLine = btSeries.createPriceLine({
                    price: t.tp,
                    color: 'rgba(0, 255, 135, 0.7)',
                    lineWidth: 1,
                    lineStyle: LightweightCharts.LineStyle.Dashed,
                    axisLabelVisible: true,
                    title: 'TP',
                });
            } catch (e) { /* price line creation can fail silently */ }
        }

        if (t.exitBar === barIdx) {
            // exit marker
            const isWin = t.profit > 0;
            btMarkers.push({
                time: btCandles[barIdx].time,
                position: isWin ? 'aboveBar' : 'belowBar',
                color: isWin ? '#00ff87' : '#ff6b6b',
                shape: 'circle',
                text: t.reason + (isWin ? ' ✓' : ' ✗')
            });
            btSeries.setMarkers(btMarkers.slice());

            // remove SL/TP/entry price lines
            try {
                if (btSlLine) { btSeries.removePriceLine(btSlLine); btSlLine = null; }
                if (btTpLine) { btSeries.removePriceLine(btTpLine); btTpLine = null; }
                if (btEntryLine) { btSeries.removePriceLine(btEntryLine); btEntryLine = null; }
            } catch (e) {}

            // draw trade result line (entry → exit)
            try {
                const lineColor = isWin ? 'rgba(0, 255, 135, 0.4)' : 'rgba(255, 107, 107, 0.4)';
                const tradeLine = btChart.addLineSeries({
                    color: lineColor,
                    lineWidth: 2,
                    lineStyle: LightweightCharts.LineStyle.Dotted,
                    crosshairMarkerVisible: false,
                    priceLineVisible: false,
                    lastValueVisible: false,
                });
                tradeLine.setData([
                    { time: btCandles[t.entryBar].time, value: t.entryPrice },
                    { time: btCandles[t.exitBar].time, value: t.exitPrice },
                ]);
                btTradeLines.push(tradeLine);
            } catch (e) {}

            // floating profit/loss popup
            showTradePopup(t.profit, isWin, t.reason);

            // update stats
            btStats.trades++;
            if (t.profit > 0) btStats.wins++;
            btStats.pnl += t.profit;
            btStats.balance = t.balance;
            if (t.profit > 0) btGrossProfit += Number(t.profit || 0);
            else btGrossLoss += Math.abs(Number(t.profit || 0));
            btLiveEquityCurve.push(Number(btStats.balance || 10000));
            updateBtLiveStats();
            updateBacktestReportLive('Running');
        }
    }
}

function showTradePopup(profit, isWin, reason) {
    const container = document.getElementById('btChartContainer');
    if (!container) return;

    const el = document.createElement('div');
    el.className = isWin ? 'bt-popup-win' : 'bt-popup-loss';

    const sign = profit >= 0 ? '+' : '';
    const icon = isWin ? '💰' : '🔻';
    el.innerHTML = `<span>${icon} ${sign}$${Math.abs(profit).toFixed(2)}</span>`;

    // position randomly in the right half of the chart
    el.style.right = (20 + Math.random() * 80) + 'px';
    el.style.top = (30 + Math.random() * 50) + '%';
    container.style.position = 'relative';
    container.appendChild(el);

    setTimeout(() => { if (el.parentNode) el.remove(); }, 2200);
}

function updateBtLiveStats() {
    const wr = btStats.trades > 0 ? (btStats.wins / btStats.trades * 100).toFixed(1) : null;
    const el = id => document.getElementById(id);

    if (el('btLiveTrades')) el('btLiveTrades').textContent = btStats.trades;
    if (el('btLiveWinRate')) el('btLiveWinRate').textContent = wr ? wr + '%' : '\u2014';

    const pnlEl = el('btLivePnl');
    if (pnlEl) {
        pnlEl.textContent = (btStats.pnl >= 0 ? '+' : '') + '$' + btStats.pnl.toFixed(2);
        pnlEl.className = 'bt-stat-value ' + (btStats.pnl >= 0 ? 'positive' : 'negative');
    }

    if (el('btLiveBalance')) {
        el('btLiveBalance').textContent = '$' + btStats.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }

    const headBalance = el('btHeadBalance');
    if (headBalance) {
        headBalance.textContent = '$' + btStats.balance.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
    }
}

function toggleBtPause() {
    btPaused = !btPaused;
    const btn = document.getElementById('btPauseBtn');
    if (btn) btn.innerHTML = btPaused ? '<i class="fas fa-play"></i>' : '<i class="fas fa-pause"></i>';
    if (!btPaused && !btStopped) animateNextCandle();
}

function skipToEnd() {
    if (btAnimTimer) { clearTimeout(btAnimTimer); btAnimTimer = null; }
    btStopped = true;

    // remove any active price lines before bulk dump
    try {
        if (btSlLine) { btSeries.removePriceLine(btSlLine); btSlLine = null; }
        if (btTpLine) { btSeries.removePriceLine(btTpLine); btTpLine = null; }
        if (btEntryLine) { btSeries.removePriceLine(btEntryLine); btEntryLine = null; }
    } catch (e) {}

    // suppress popups during bulk render
    const origShow = showTradePopup;
    showTradePopup = () => {};

    // dump all remaining candles at once
    while (btIdx < btCandles.length) {
        btSeries.update(btCandles[btIdx]);
        checkTradeMarkers(btIdx);
        btIdx++;
    }
    btChart.timeScale().scrollToRealTime();
    trackBacktestPrice(true);

    showTradePopup = origShow;
    finishBacktest();
}

function finishBacktest() {
    btRunInProgress = false;
    btLiveEquityCurve.push(Number(btStats.balance || 10000));
    updateBacktestReportLive('Complete');
    const progEl = document.getElementById('btProgress');
    if (progEl) progEl.textContent = 'Complete';
    const btn = document.getElementById('btPauseBtn');
    if (btn) btn.disabled = true;
    const stopBtn = document.getElementById('btStopBtn');
    if (stopBtn) stopBtn.disabled = true;
    const status = document.getElementById('btStatus');
    setStatusText(status, 'Backtest animation complete.');
}

async function runSuite() {
    const s = document.getElementById('btStatus');
    setStatusText(s, 'Starting full suite (7 symbols × 4 modes)...');
    const r = await apiFetch('/backtest/run-suite', { method: 'POST', body: JSON.stringify({ modes: ['standard', 'walk_forward', 'split', 'monte_carlo'] }) });
    if (r && r.success) { setStatusText(s, `Started ${r.count} runs`); setTimeout(async () => { await loadRuns(); await loadRobustness(); }, 1200); }
    else setStatusText(s, 'Failed to start suite', true);
}

async function loadRuns() {
    const data = await apiFetch('/backtest/runs');
    const table = document.getElementById('runsTable');
    if (!data || !table) return;
    const runs = data.runs || [];
    if (!runs.length) { table.innerHTML = '<tr><td colspan="6">No runs yet</td></tr>'; return; }
    table.innerHTML = runs.map(r => `<tr>
        <td>${r.id}</td><td>${r.symbol}</td><td>${r.mode}</td><td>${r.status}</td><td>${r.created_at}</td>
        <td><button class="btn btn-sm btn-outline" data-run="${r.id}">View</button></td>
    </tr>`).join('');
    table.querySelectorAll('button[data-run]').forEach(btn => {
        btn.addEventListener('click', () => loadRunResult(btn.dataset.run));
    });
}

async function loadRunResult(runId) {
    const el = document.getElementById('runResult');
    if (!el) return;
    el.textContent = 'Loading result...';
    const data = await apiFetch(`/backtest/results/${runId}`);
    if (!data) return;
    if (data.result) el.innerHTML = renderRunResult(data.result);
    else el.textContent = data.log || data.status || 'No result yet';
}

async function loadRobustness() {
    const el = document.getElementById('robustnessResult');
    if (!el) return;
    const data = await apiFetch('/backtest/robustness');
    if (!data || !data.summary) { el.textContent = 'Robustness: no data yet'; return; }
    const s = data.summary;
    const rows = (data.details || []).slice(0, 28).map(d =>
        `<tr><td>${d.symbol}</td><td>${d.mode}</td><td>${num(d.value)}</td><td class="${d.passed ? 'positive' : 'negative'}">${d.passed ? 'PASS' : 'FAIL'}</td></tr>`
    ).join('');
    el.innerHTML = `<div><strong>Robustness Score:</strong> ${s.score}% | <strong>Passed:</strong> ${s.passed}/${s.checks} | <strong>Verdict:</strong> ${s.verdict}</div>
        <table class="robust-table"><thead><tr><th>Symbol</th><th>Mode</th><th>Value</th><th>Status</th></tr></thead>
        <tbody>${rows || '<tr><td colspan="4">No checks yet</td></tr>'}</tbody></table>`;
}

async function exportReportPdf() {
    const report = await apiFetch('/backtest/report-data');
    if (!report) return;
    const html = `<html><head><title>Zenith Backtest Report</title>
        <style>body{font-family:'Inter',Arial,sans-serif;padding:24px;background:#0a1018;color:#e8ecf1;}
        h1{background:linear-gradient(135deg,#00ff87,#60efff);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin-bottom:4px;}
        .meta{color:#6b7a8d;font-size:0.9rem;} table{width:100%;border-collapse:collapse;margin-top:16px;}
        td,th{border:1px solid #1a2535;padding:10px;text-align:left;font-size:0.9rem;} th{background:#111a25;color:#6b7a8d;text-transform:uppercase;font-size:0.75rem;letter-spacing:0.5px;}
        .ok{color:#00ff87;} .bad{color:#ff6b6b;}</style></head>
        <body><h1>Zenith Backtest Report</h1>
        <div class="meta">Generated: ${report.generated_at}</div>
        <div class="meta">User: ${report.user?.email || ''}</div>
        <p style="margin:16px 0;">Total runs: <b>${report.runs_total}</b> | Completed: <b>${report.runs_done}</b></p>
        <h3 style="color:#60efff;">Robustness</h3>
        <p>Score: <b style="color:#00ff87;">${report.robustness?.summary?.score || 0}%</b> | Verdict: <b>${report.robustness?.summary?.verdict || 'n/a'}</b></p>
        <table><thead><tr><th>Symbol</th><th>Mode</th><th>Value</th><th>Status</th></tr></thead><tbody>
        ${(report.robustness?.details || []).map(d => `<tr><td>${d.symbol}</td><td>${d.mode}</td><td>${num(d.value)}</td><td class="${d.passed ? 'ok' : 'bad'}">${d.passed ? 'PASS' : 'FAIL'}</td></tr>`).join('')}
        </tbody></table><script>window.onload=()=>window.print();<\/script></body></html>`;
    const w = window.open('', '_blank');
    if (!w) return;
    w.document.write(html);
    w.document.close();
}

async function loadMt5Status() {
    const status = await apiFetch('/mt5/status');
    const el = document.getElementById('mt5Status');
    if (!el) return;
    if (!status || !status.connected) {
        setStatusText(el, 'Not connected');
        return;
    }

    if (status.login && status.server) {
        setStatusText(el, `Connected: ${status.login} (${status.server})`);
        return;
    }

    if (status.runtime_connected) {
        const runtimeLabel = status.runtime_state ? ` (${status.runtime_state})` : '';
        setStatusText(el, `Connected via runtime${runtimeLabel}`);
        return;
    }

    setStatusText(el, 'Connected');
}

async function connectMt5() {
    const account_id = document.getElementById('mt5Account').value.trim();
    const server = document.getElementById('mt5Server').value.trim();
    const login = document.getElementById('mt5Login').value.trim();
    const s = document.getElementById('mt5Status');
    const btn = document.getElementById('mt5ConnectBtn');
    
    if (btn) {
        btn.classList.add('is-loading');
        btn.disabled = true;
    }
    
    setStatusText(s, 'Saving connection...');
    const r = await apiFetch('/mt5/connect', { method: 'POST', body: JSON.stringify({ account_id, server, login }) });
    
    if (btn) {
        btn.classList.remove('is-loading');
        btn.disabled = false;
    }
    
    if (r && r.success) {
        setStatusText(s, 'Connected');
        if (btn) {
            btn.classList.add('is-success');
            setTimeout(() => btn?.classList.remove('is-success'), 600);
        }
    } else {
        setStatusText(s, 'Connection failed', true);
        if (btn) {
            btn.classList.add('is-error');
            setTimeout(() => btn?.classList.remove('is-error'), 500);
        }
    }
}

async function disconnectMt5() {
    const s = document.getElementById('mt5Status');
    await apiFetch('/mt5/disconnect', { method: 'POST' });
    setStatusText(s, 'Disconnected');
}

function renderBotSymbols(cfg) {
    const grid = document.getElementById('botSymbolsGrid');
    if (!grid || !cfg) return;

    const enabled = new Set(cfg.enabled_symbols || []);
    const symbols = cfg.available_symbols || [];

    grid.innerHTML = symbols.map((symbol) => {
        const active = enabled.has(symbol);
        return `<button type="button" class="bot-symbol-chip ${active ? 'active' : 'inactive'}" data-symbol="${symbol}" aria-pressed="${active ? 'true' : 'false'}">
            <span class="chip-main">
                <span class="chip-dot"></span>
                <span class="chip-symbol">${symbol}</span>
            </span>
            <span class="chip-state">${active ? 'ON' : 'OFF'}</span>
        </button>`;
    }).join('');

    grid.querySelectorAll('.bot-symbol-chip').forEach((el) => {
        el.addEventListener('click', async () => {
            const symbol = el.getAttribute('data-symbol');
            if (!symbol || !botConfigCache) return;
            if (el.classList.contains('is-saving')) return;

            const prevEnabled = [...(botConfigCache.enabled_symbols || [])];
            const enabledSet = new Set(botConfigCache.enabled_symbols || []);
            const nextActive = !enabledSet.has(symbol);

            if (nextActive) enabledSet.add(symbol);
            else enabledSet.delete(symbol);

            el.classList.add('is-saving');
            el.classList.toggle('active', nextActive);
            el.classList.toggle('inactive', !nextActive);
            el.setAttribute('aria-pressed', nextActive ? 'true' : 'false');
            const state = el.querySelector('.chip-state');
            if (state) state.textContent = nextActive ? 'ON' : 'OFF';

            botConfigCache.enabled_symbols = [...enabledSet];
            const saved = await saveBotConfig(
                { enabled_symbols: [...enabledSet] },
                `${symbol} ${nextActive ? 'enabled' : 'disabled'}`,
                { silentRender: true }
            );

            el.classList.remove('is-saving');
            if (!saved) {
                botConfigCache.enabled_symbols = prevEnabled;
                renderBotSymbols(botConfigCache);
            }
        });
    });
}

async function loadBotConfig() {
    const cfg = await apiFetch('/bot/config');
    if (!cfg) return;

    botConfigCache = cfg;
    const riskInput = document.getElementById('botRiskInput');
    if (riskInput) riskInput.value = Number(cfg.risk_percent || 1).toFixed(2);
    const dailyDdInput = document.getElementById('botDailyDdInput');
    if (dailyDdInput) dailyDdInput.value = Number(cfg.max_daily_drawdown_pct || 5).toFixed(2);
    const marginCapInput = document.getElementById('botMarginCapInput');
    if (marginCapInput) marginCapInput.value = Number(cfg.max_margin_usage_pct || 20).toFixed(2);
    const ddAdjInput = document.getElementById('botDdAdjustmentInput');
    if (ddAdjInput) ddAdjInput.value = Number(cfg.daily_drawdown_adjustment_usd || 0).toFixed(2);
    renderBotSymbols(cfg);
}

async function saveBotConfig(payload, successMessage = 'Saved', options = {}) {
    const statusEl = document.getElementById('botConfigStatus');
    const res = await apiFetch('/bot/config', {
        method: 'POST',
        body: JSON.stringify(payload),
    });

    if (!res || res.success === false) {
        setStatusText(statusEl, 'Failed to save bot config', true);
        return false;
    }

    botConfigCache = res;
    const riskInput = document.getElementById('botRiskInput');
    if (riskInput) riskInput.value = Number(res.risk_percent || 1).toFixed(2);
    const dailyDdInput = document.getElementById('botDailyDdInput');
    if (dailyDdInput) dailyDdInput.value = Number(res.max_daily_drawdown_pct || 5).toFixed(2);
    const marginCapInput = document.getElementById('botMarginCapInput');
    if (marginCapInput) marginCapInput.value = Number(res.max_margin_usage_pct || 20).toFixed(2);
    const ddAdjInput = document.getElementById('botDdAdjustmentInput');
    if (ddAdjInput) ddAdjInput.value = Number(res.daily_drawdown_adjustment_usd || 0).toFixed(2);
    if (!options.silentRender) renderBotSymbols(res);
    setStatusText(statusEl, successMessage);
    return true;
}

async function saveBotRisk() {
    const riskInput = document.getElementById('botRiskInput');
    if (!riskInput) return;
    const risk = Number(riskInput.value);
    if (!Number.isFinite(risk) || risk < 0.10 || risk > 2.00) {
        setStatusText(document.getElementById('botConfigStatus'), 'Risk must be 0.10 - 2.00', true);
        return;
    }
    await saveBotConfig({ risk_percent: risk }, `Risk saved (${risk.toFixed(2)}%)`);
}

async function saveBotGuards() {
    const dailyDdInput = document.getElementById('botDailyDdInput');
    const marginCapInput = document.getElementById('botMarginCapInput');
    const ddAdjInput = document.getElementById('botDdAdjustmentInput');
    if (!dailyDdInput || !marginCapInput || !ddAdjInput) return;

    const dailyDd = Number(dailyDdInput.value);
    const marginCap = Number(marginCapInput.value);
    const ddAdj = Number(ddAdjInput.value);

    if (!Number.isFinite(dailyDd) || dailyDd < 0.5 || dailyDd > 25) {
        setStatusText(document.getElementById('botConfigStatus'), 'Daily DD must be 0.5 - 25.0', true);
        return;
    }
    if (!Number.isFinite(marginCap) || marginCap < 5 || marginCap > 95) {
        setStatusText(document.getElementById('botConfigStatus'), 'Margin cap must be 5 - 95', true);
        return;
    }
    if (!Number.isFinite(ddAdj)) {
        setStatusText(document.getElementById('botConfigStatus'), 'Daily DD adjustment must be a valid number', true);
        return;
    }

    await saveBotConfig({
        max_daily_drawdown_pct: dailyDd,
        max_margin_usage_pct: marginCap,
        daily_drawdown_adjustment_usd: ddAdj,
    }, `Guard limits saved (DD ${dailyDd.toFixed(2)}%, Margin ${marginCap.toFixed(1)}%)`);
}

async function resetDrawdownAdjustment() {
    const statusEl = document.getElementById('botConfigStatus');
    const res = await apiFetch('/bot/config/reset-drawdown', { method: 'POST' });
    if (!res || res.success === false) {
        setStatusText(statusEl, 'Failed to reset daily drawdown adjustment', true);
        return;
    }
    botConfigCache = res;
    const ddAdjInput = document.getElementById('botDdAdjustmentInput');
    if (ddAdjInput) ddAdjInput.value = Number(res.daily_drawdown_adjustment_usd || 0).toFixed(2);
    setStatusText(statusEl, `Daily DD reset applied (PnL estimate: ${Number(res.daily_pnl_estimate || 0).toFixed(2)} USD)`);
}

async function setAllSymbols(enabled) {
    if (!botConfigCache) await loadBotConfig();
    if (!botConfigCache) return;

    const symbols = enabled ? (botConfigCache.available_symbols || []) : [];
    const prevEnabled = [...(botConfigCache.enabled_symbols || [])];
    botConfigCache.enabled_symbols = [...symbols];
    renderBotSymbols(botConfigCache);

    const saved = await saveBotConfig(
        { enabled_symbols: symbols },
        enabled ? 'All symbols enabled' : 'All symbols disabled'
    );
    if (!saved) {
        botConfigCache.enabled_symbols = prevEnabled;
        renderBotSymbols(botConfigCache);
    }
}

async function loadBotMonitor() {
    const [activity, positionsRes, logsRes] = await Promise.all([
        apiFetch('/bot/activity?lines=220'),
        apiFetch('/bot/positions'),
        apiFetch('/bot/logs?lines=140'),
    ]);

    const summary = activity?.summary || {};
    const setText = (id, value) => {
        const el = document.getElementById(id);
        if (el) el.textContent = String(value);
    };

    setText('liveSignalsCount', summary.signals || 0);
    setText('liveOpenedCount', summary.opened || 0);
    setText('liveFailedCount', summary.failed || 0);

    const positions = positionsRes?.positions || [];
    const posMsg = positionsRes?.message || '';
    setText('liveOpenPositionsCount', positions.length);

    const posTable = document.getElementById('liveOpenPositionsTable');
    if (posTable) {
        if (!positions.length) {
            const hint = posMsg ? `<br><small style="color:var(--z-text-3);font-size:0.75rem">${posMsg}</small>` : '';
            posTable.innerHTML = `<tr class="empty-state-row"><td colspan="7">No open positions${hint}</td></tr>`;
        } else {
            posTable.innerHTML = positions.map((p) => `
                <tr>
                    <td>${p.ticket}</td>
                    <td><strong>${p.symbol}</strong></td>
                    <td><span class="badge badge-${p.direction === 'BUY' ? 'success' : 'danger'}">${p.direction}</span></td>
                    <td>${Number(p.volume).toFixed(2)}</td>
                    <td>${Number(p.open_price).toFixed(5)}</td>
                    <td>${Number(p.current_price).toFixed(5)}</td>
                    <td class="${Number(p.profit) >= 0 ? 'positive' : 'negative'}">${Number(p.profit).toFixed(2)}</td>
                </tr>
            `).join('');
        }
    }

    const events = activity?.events || [];
    const eventTable = document.getElementById('botActivityTable');
    if (eventTable) {
        if (!events.length) {
            eventTable.innerHTML = '<tr class="empty-state-row"><td colspan="3">No activity events yet</td></tr>';
        } else {
            eventTable.innerHTML = events.slice(-70).reverse().map((ev) => `
                <tr>
                    <td><span class="badge badge-${ev.type === 'opened' ? 'success' : (ev.type === 'failed' ? 'danger' : 'neutral')}">${ev.type}</span></td>
                    <td>${ev.symbol || '—'}</td>
                    <td>${ev.message || ''}</td>
                </tr>
            `).join('');
        }
    }

    const rawEl = document.getElementById('botRawLog');
    if (rawEl) rawEl.textContent = logsRes?.logs || 'No logs yet.';
}

async function logout() {
    await apiFetch('/auth/logout', { method: 'POST' });
    localStorage.removeItem('zenith_token');
    localStorage.removeItem('ultima_token');
    localStorage.removeItem('zenith_email');
    window.location.href = 'index.html';
}

// init
function initEvents() {
    const bind = (id, fn) => { const el = document.getElementById(id); if (el) el.addEventListener('click', fn); };

    bind('refreshBtn', async () => {
        const btn = document.getElementById('refreshBtn');
        btn.querySelector('i').classList.add('fa-spin');
        await Promise.all([loadMetrics(), loadStatus(), loadRuns(), loadTrades(), loadEquity(), loadRobustness(), loadMt5Status(), loadBotConfig(), loadBotMonitor()]);
        setTimeout(() => btn.querySelector('i').classList.remove('fa-spin'), 700);
    });

    bind('startStopBtn', handleStartStop);
    bind('runBacktestBtn', () => {
        if (btRunInProgress) return;
        runBacktest();
    });
    bind('btPauseBtn', toggleBtPause);
    bind('btStopBtn', skipToEnd);
    bind('mt5ConnectBtn', connectMt5);
    bind('mt5DisconnectBtn', disconnectMt5);
    bind('runMonteCarloBtn', runMonteCarlo);
    bind('logoutBtn', logout);
    bind('saveBotRiskBtn', saveBotRisk);
    bind('saveBotGuardsBtn', saveBotGuards);
    bind('resetDrawdownAdjBtn', resetDrawdownAdjustment);
    bind('enableAllSymbolsBtn', () => setAllSymbols(true));
    bind('disableAllSymbolsBtn', () => setAllSymbols(false));
    bind('reloadBotConfigBtn', loadBotConfig);
    bind('refreshLiveMonitorBtn', loadBotMonitor);
    bind('saveTelegramConfigBtn', saveTelegramConfig);
    bind('mt5HelpCloseBtn', closeMt5HelpPopup);
    bind('mt5HelpOkBtn', closeMt5HelpPopup);
    bind('mt5HelpBackdrop', closeMt5HelpPopup);
}

// ─── Monte Carlo Simulation ───
async function runMonteCarlo() {
    const btn = document.getElementById('runMonteCarloBtn');
    const resultsEl = document.getElementById('mcResults');
    const loadingEl = document.getElementById('mcLoading');
    const emptyEl = document.getElementById('mcEmpty');

    // need trades from the last backtest
    if (!btTrades || btTrades.length === 0) {
        if (emptyEl) {
            emptyEl.querySelector('.empty-state-title').textContent = 'No Trades Available';
            emptyEl.querySelector('.empty-state-desc').innerHTML = 'Run a backtest first so there are trades to shuffle.';
        }
        return;
    }

    // show loading
    if (emptyEl) emptyEl.style.display = 'none';
    if (resultsEl) resultsEl.style.display = 'none';
    if (loadingEl) loadingEl.style.display = 'flex';
    if (btn) { btn.classList.add('is-loading'); btn.disabled = true; }

    // run client-side monte carlo (fast, no server needed)
    const ITERATIONS = 200;
    const INITIAL = 10000;
    const profits = btTrades.map(t => t.profit || 0);
    const endingBalances = [];
    const drawdowns = [];

    await new Promise(resolve => setTimeout(resolve, 50)); // let UI update

    for (let i = 0; i < ITERATIONS; i++) {
        // fisher-yates shuffle
        const shuffled = profits.slice();
        for (let j = shuffled.length - 1; j > 0; j--) {
            const k = Math.floor(Math.random() * (j + 1));
            [shuffled[j], shuffled[k]] = [shuffled[k], shuffled[j]];
        }

        let balance = INITIAL;
        let peak = INITIAL;
        let maxDD = 0;
        for (const p of shuffled) {
            balance += p;
            if (balance > peak) peak = balance;
            const dd = (peak - balance) / peak;
            if (dd > maxDD) maxDD = dd;
        }
        endingBalances.push(balance);
        drawdowns.push(maxDD * 100);
    }

    endingBalances.sort((a, b) => a - b);
    const p10 = endingBalances[Math.floor(ITERATIONS * 0.10)];
    const p50 = endingBalances[Math.floor(ITERATIONS * 0.50)];
    const p90 = endingBalances[Math.floor(ITERATIONS * 0.90)];
    const avgDD = drawdowns.reduce((a, b) => a + b, 0) / drawdowns.length;
    const baseProfit = btStats.pnl;

    // hide loading, show results
    if (loadingEl) loadingEl.style.display = 'none';
    if (resultsEl) resultsEl.style.display = 'block';
    if (btn) { btn.classList.remove('is-loading'); btn.disabled = false; btn.classList.add('is-success'); setTimeout(() => btn.classList.remove('is-success'), 1500); }

    // populate percentile values
    const minBal = endingBalances[0];
    const maxBal = endingBalances[endingBalances.length - 1];
    const range = maxBal - minBal || 1;

    const setEl = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
    const setBar = (id, pct) => { const e = document.getElementById(id); if (e) e.style.width = Math.max(4, pct) + '%'; };

    setEl('mcP10', '$' + num(p10));
    setEl('mcP50', '$' + num(p50));
    setEl('mcP90', '$' + num(p90));
    setBar('mcP10Bar', ((p10 - minBal) / range) * 100);
    setBar('mcP50Bar', ((p50 - minBal) / range) * 100);
    setBar('mcP90Bar', ((p90 - minBal) / range) * 100);

    setEl('mcAvgDD', num(avgDD) + '%');
    setEl('mcIterations', ITERATIONS.toString());
    setEl('mcBaseProfit', (baseProfit >= 0 ? '+$' : '-$') + num(Math.abs(baseProfit)));
    
    const verdictEl = document.getElementById('mcVerdict');
    if (verdictEl) {
        if (p10 >= INITIAL) {
            verdictEl.textContent = 'Robust ✓';
            verdictEl.className = 'mc-stat-value positive';
        } else if (p50 >= INITIAL) {
            verdictEl.textContent = 'Moderate';
            verdictEl.className = 'mc-stat-value';
        } else {
            verdictEl.textContent = 'Weak ✗';
            verdictEl.className = 'mc-stat-value negative';
        }
    }

    setEl('mcHistMin', '$' + num(minBal));
    setEl('mcHistMax', '$' + num(maxBal));

    // build histogram
    const histEl = document.getElementById('mcHistogram');
    if (histEl) {
        const BINS = 25;
        const binSize = range / BINS;
        const bins = new Array(BINS).fill(0);
        for (const b of endingBalances) {
            let idx = Math.floor((b - minBal) / binSize);
            if (idx >= BINS) idx = BINS - 1;
            bins[idx]++;
        }
        const maxBin = Math.max(...bins);
        histEl.innerHTML = bins.map((count, i) => {
            const pct = maxBin > 0 ? (count / maxBin) * 100 : 0;
            const binVal = minBal + (i + 0.5) * binSize;
            const colorClass = binVal < INITIAL ? 'mc-hist-red' : binVal < INITIAL * 1.05 ? 'mc-hist-yellow' : 'mc-hist-green';
            return `<div class="mc-hist-bar ${colorClass}" style="height: ${Math.max(2, pct)}%;" title="${count} outcomes around $${num(binVal)}"></div>`;
        }).join('');
    }
}

// sidebar nav
function initSidebarNav() {
    const navLinks = Array.from(document.querySelectorAll('.sidebar-nav a[data-section]'));
    const sections = Array.from(document.querySelectorAll('.dashboard-main > section'));
    const sidebar = document.querySelector('.sidebar');
    const toggleBtn = document.getElementById('mobileNavToggle');
    const validSections = new Set(sections.map(sec => sec.id));
    
    function switchToSection(sectionId) {
        if (!sectionId || !validSections.has(sectionId)) return;

        sections.forEach(sec => {
            sec.style.display = 'none';
            sec.classList.remove('active');
        });
        
        const target = document.getElementById(sectionId);
        if (target) {
            target.style.display = 'flex';
            target.classList.add('active');
        }
        
        navLinks.forEach(link => link.classList.remove('active'));
        const activeLink = document.querySelector(`.sidebar-nav a[data-section="${sectionId}"]`);
        if (activeLink) activeLink.classList.add('active');
        
        if (sidebar && toggleBtn) {
            sidebar.classList.remove('active');
            toggleBtn.classList.remove('active');
        }
    }

    navLinks.forEach(link => {
        link.addEventListener('click', (e) => {
            e.preventDefault();
            const sectionId = link.getAttribute('data-section');
            if (sectionId) {
                switchToSection(sectionId);
                if (window.location.hash !== `#${sectionId}`) {
                    history.replaceState(null, '', `#${sectionId}`);
                }
            }
        });
    });

    // Also wire up any other data-section links in the page (e.g. "View All", "New Backtest")
    document.querySelectorAll('[data-section]').forEach(el => {
        if (navLinks.indexOf(el) !== -1) return; // skip sidebar links already bound
        el.addEventListener('click', (e) => {
            const sectionId = el.getAttribute('data-section');
            if (sectionId && validSections.has(sectionId)) {
                e.preventDefault();
                switchToSection(sectionId);
                history.replaceState(null, '', `#${sectionId}`);
            }
        });
    });

    window.addEventListener('hashchange', () => {
        const sectionId = (window.location.hash || '').replace('#', '');
        if (sectionId) switchToSection(sectionId);
    });

    if (toggleBtn) {
        toggleBtn.addEventListener('click', () => {
            if (sidebar) sidebar.classList.toggle('active');
            toggleBtn.classList.toggle('active');
        });
    }

    sections.forEach(sec => sec.style.display = 'none');
    const initialSection = (window.location.hash || '').replace('#', '');
    if (initialSection && validSections.has(initialSection)) {
        switchToSection(initialSection);
    } else {
        switchToSection('overview');
    }
}

// === Dashboard cosmic effects ===
function initDashNebula() {
    const container = document.getElementById('dashNebulaCanvas');
    if (!container) return;
    if (perfModeEnabled) return;
    const canvas = document.createElement('canvas');
    canvas.style.cssText = 'width:100%;height:100%;';
    container.appendChild(canvas);
    const ctx = canvas.getContext('2d');
    let w, h;
    const stars = [];
    const STAR_COUNT = 70;

    function resize() {
        w = canvas.width = window.innerWidth;
        h = canvas.height = window.innerHeight;
    }
    resize();
    window.addEventListener('resize', resize);

    const colors = ['rgba(0,255,135,', 'rgba(96,239,255,', 'rgba(167,139,250,'];
    for (let i = 0; i < STAR_COUNT; i++) {
        stars.push({
            x: Math.random() * w,
            y: Math.random() * h,
            r: Math.random() * 1.2 + 0.3,
            c: colors[Math.floor(Math.random() * 3)],
            speed: Math.random() * 0.15 + 0.02,
            phase: Math.random() * Math.PI * 2
        });
    }

    let lastFrame = 0;
    function draw(t) {
        if (perfModeEnabled) return;
        if (t - lastFrame < 40) {
            requestAnimationFrame(draw);
            return;
        }
        lastFrame = t;
        ctx.clearRect(0, 0, w, h);
        for (const s of stars) {
            const twinkle = 0.4 + 0.6 * Math.sin(t * 0.001 * s.speed * 10 + s.phase);
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
            ctx.fillStyle = s.c + twinkle.toFixed(2) + ')';
            ctx.fill();
            // faint glow
            ctx.beginPath();
            ctx.arc(s.x, s.y, s.r * 3, 0, Math.PI * 2);
            ctx.fillStyle = s.c + (twinkle * 0.08).toFixed(3) + ')';
            ctx.fill();
        }
        requestAnimationFrame(draw);
    }
    requestAnimationFrame(draw);
}

function initDashGhostPairs() {
    const layer = document.getElementById('ghostPairsLayer');
    if (!layer) return;
    if (perfModeEnabled) return;
    const pairs = ['EUR/USD', 'GBP/USD', 'USD/JPY', 'BTC/USD', 'AUD/USD', 'XAU/USD', 'ETH/USD', 'USD/CHF'];
    const colorClasses = ['', 'alt-color', 'purple-color'];

    function spawn() {
        const el = document.createElement('span');
        el.className = 'ghost-pair ' + colorClasses[Math.floor(Math.random() * 3)];
        el.textContent = pairs[Math.floor(Math.random() * pairs.length)];
        el.style.left = Math.random() * 90 + 5 + '%';
        const size = (Math.random() * 1.2 + 0.6).toFixed(2);
        el.style.fontSize = size + 'rem';
        const dur = (Math.random() * 30 + 25).toFixed(1);
        el.style.setProperty('--dur', dur + 's');
        el.style.setProperty('--rot', (Math.random() * 10 - 5).toFixed(1) + 'deg');
        el.style.opacity = '0';
        layer.appendChild(el);
        setTimeout(() => el.remove(), parseFloat(dur) * 1000);
    }

    // initial burst
    for (let i = 0; i < 6; i++) setTimeout(spawn, i * 3000);
    setInterval(spawn, 4000);
}

document.addEventListener('DOMContentLoaded', () => {
    requireAuth();
    initSidebarNav();
    initEvents();
    startHeartbeatTicker();
    renderStartStopButton({ isRunning: false, loading: false });
    Promise.all([loadStatus(), loadMetrics(), loadTrades(), loadEquity(), loadRuns(), loadRobustness(), loadMt5Status(), loadBotConfig(), loadBotMonitor(), loadTelegramConfig()]);
    setInterval(loadRuns, 5000);
    setInterval(loadStatus, 5000);
    setInterval(loadBotMonitor, 7000);
});
