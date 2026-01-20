# TradingBot Web Interface

Professional web interface for the TradingBot algorithmic trading system. This graduation project features a modern, responsive dashboard with real-time trading metrics and comprehensive documentation.

## 🌟 Features

- **Landing Page**: Modern hero section with feature showcase and performance metrics
- **Live Dashboard**: Real-time trading metrics, equity curves, and trade history
- **Comprehensive Documentation**: Full API reference, strategy guides, and setup instructions
- **About Page**: Project overview, technology stack, and academic context
- **Responsive Design**: Optimized for desktop, tablet, and mobile devices
- **Dark Theme**: Professional dark UI with cyan accent colors
- **Real-time Updates**: Auto-refreshing data every 30 seconds
- **Interactive Charts**: Chart.js integration for data visualization

## 📁 Structure

```
web/
├── index.html          # Landing page
├── dashboard.html      # Live trading dashboard
├── docs.html          # Documentation
├── about.html         # About page
├── api.py             # Flask API backend
├── requirements.txt   # Python dependencies
├── css/
│   ├── style.css      # Main stylesheet
│   ├── dashboard.css  # Dashboard styles
│   ├── docs.css       # Documentation styles
│   └── about.css      # About page styles
└── js/
    ├── main.js        # Main JavaScript
    ├── dashboard.js   # Dashboard functionality
    └── dashboard-api.js # API integration
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd web
pip install -r requirements.txt
```

### 2. Start the API Backend

```bash
python api.py
```

The API will start on `http://localhost:5000`

### 3. Open the Website

Open `index.html` in your browser or use a local server:

```bash
# Using Python's built-in server
python -m http.server 8000

# Then visit http://localhost:8000
```

## 🔌 API Endpoints

### Status & Metrics
- `GET /api/status` - Bot status
- `GET /api/metrics` - Performance metrics
- `GET /api/equity` - Equity curve data
- `GET /api/performance` - Performance by symbol

### Trading Data
- `GET /api/trades?limit=50` - Trade history
- `GET /api/positions` - Open positions
- `GET /api/strategies` - Available strategies

### Bot Control
- `POST /api/bot/start` - Start the bot
- `POST /api/bot/stop` - Stop the bot

## 🎨 Design System

### Colors
- **Primary**: #00d4aa (Cyan/Teal)
- **Background**: #0a0e12 (Dark)
- **Card Background**: #14181f (Dark Gray)
- **Secondary Background**: #1a1f2e (Blue Gray)
- **Text Primary**: #ffffff (White)
- **Text Secondary**: #8b92a7 (Gray)

### Typography
- **Headings**: Poppins, sans-serif
- **Body**: Inter, sans-serif
- **Code**: Courier New, monospace

## 📊 Dashboard Features

### Overview Section
- Account balance with real-time updates
- Total PnL with percentage change
- Win rate indicator
- Total trades counter
- Equity curve chart with timeframe selector
- Performance metrics grid

### Trading Section
- Active strategies list
- Strategy metrics (PnL, trades, win rate)
- Start/stop controls
- Configuration buttons

### Positions Section
- Open positions table
- Current PnL for each position
- Quick close buttons

### History Section
- Complete trade history
- Sortable columns
- Pagination support

## 🔧 Configuration

### API Configuration

Edit `js/dashboard-api.js` to change the API URL:

```javascript
const API_BASE_URL = 'http://localhost:5000/api';
```

### Refresh Interval

Change the auto-refresh interval in `js/dashboard-api.js`:

```javascript
// Default: 30 seconds
setInterval(updateDashboardWithAPI, 30000);
```

## 📈 Performance Metrics

The dashboard displays comprehensive performance metrics:

- **Total Return**: Overall profit/loss percentage
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss
- **Average Win/Loss**: Average profit vs loss per trade

## 🎓 Academic Project

This website was created as part of a graduation project demonstrating:

- Full-stack web development skills
- RESTful API design and implementation
- Modern frontend development with vanilla JavaScript
- Responsive UI/UX design principles
- Data visualization techniques
- Real-time data integration
- Professional documentation writing

## 📱 Responsive Design

The website is fully responsive with breakpoints for:

- Desktop: 1920px+
- Laptop: 1024px - 1919px
- Tablet: 768px - 1023px
- Mobile: 320px - 767px

## 🌐 Browser Support

- Chrome 90+
- Firefox 88+
- Safari 14+
- Edge 90+

## 🔒 Security Notes

- This is a demo/academic project - implement proper authentication for production
- API currently has CORS enabled for all origins
- No rate limiting implemented - add for production use
- Store sensitive credentials securely (not in code)

## 📝 License

This is an academic project for educational purposes only. Not licensed for commercial use.

## 🤝 Contributing

This is a graduation project, but feedback and suggestions are welcome!

## 📧 Contact

For questions about this project:
- Email: student@university.edu
- GitHub: github.com/yourusername/tradingbot

## 🙏 Acknowledgments

- Chart.js for data visualization
- Flask for API backend
- Font Awesome for icons (if used)
- Inspiration from modern trading platforms

---

**⚠️ Disclaimer**: This is an academic project for educational purposes only. Past performance does not guarantee future results. Always test thoroughly before using with real money.
