# Zenith Trading Bot

Web interface for the Zenith algo trading system. Built for my graduation project.

## Setup

```
pip install -r requirements.txt
python alpha_api.py
```

Opens at http://127.0.0.1:5000

## Pages

- `/` — landing page
- `/login.html` / `/signup.html` — auth
- `/dashboard.html` — main dashboard (backtests, charts, bot controls)
- `/profile.html` — account settings
- `/docs.html` — documentation
- `/about.html` — project info

## Stack

- Python / Flask backend
- Vanilla JS frontend
- Chart.js for charts
- SQLite for storage
- MT5 integration for live trading
