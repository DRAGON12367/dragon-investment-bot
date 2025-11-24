# Professional Live Trading Dashboard

A beautiful, professional web-based dashboard for 24/7 monitoring of stocks and cryptocurrencies.

## Features

- **Real-time Market Data**: Live prices for stocks and cryptocurrencies
- **Dual Market View**: Separate sections for stocks and crypto
- **Trading Signals**: ML-generated buy/sell signals with confidence scores
- **Portfolio Tracking**: Real-time portfolio value and position tracking
- **Price Charts**: Interactive price history charts
- **Auto-refresh**: Configurable auto-refresh intervals
- **Professional UI**: Modern, clean interface built with Streamlit

## Quick Start

### Option 1: Using Streamlit directly

```bash
streamlit run gui/dashboard.py
```

### Option 2: Using the runner script

```bash
python run_dashboard.py
```

The dashboard will open in your default web browser at `http://localhost:8501`

## Dashboard Sections

### 1. Header
- Last update timestamp
- Total assets being tracked
- Stock count
- Cryptocurrency count

### 2. Portfolio Summary
- Total portfolio value
- Available cash
- Number of open positions
- Total invested amount

### 3. Market Overview
- **Stocks Table**: Real-time stock prices, changes, volume
- **Cryptocurrencies Table**: Real-time crypto prices, changes, volume
- Color-coded price changes (green for positive, red for negative)

### 4. Trading Signals
- **Stock Signals**: ML-generated signals for stocks
- **Crypto Signals**: ML-generated signals for cryptocurrencies
- Shows: Symbol, Action (BUY/SELL), Confidence, Price, Stop Loss, Take Profit

### 5. Price Charts
- Interactive price history charts
- Select any symbol to view its price chart
- Real-time updates

### 6. Settings Sidebar
- Auto-refresh toggle
- Refresh interval slider (5-300 seconds)
- Manual refresh button
- Market filters (show/hide stocks/crypto)
- Statistics display

## Configuration

The dashboard uses the same configuration as the main bot. You can customize:

- **Watchlists**: Edit `stock_data_provider.py` and `crypto_data_provider.py`
- **Refresh Rate**: Use the sidebar slider (default: 30 seconds)
- **Display Options**: Toggle stocks/crypto visibility in sidebar

## Running 24/7

To run the dashboard continuously:

### On Windows:
```bash
# Run in background
start /B streamlit run gui/dashboard.py
```

### On Linux/Mac:
```bash
# Run in background
nohup streamlit run gui/dashboard.py &
```

### Using Screen (Linux/Mac):
```bash
screen -S dashboard
streamlit run gui/dashboard.py
# Press Ctrl+A then D to detach
```

### Using PM2 (Node.js process manager):
```bash
npm install -g pm2
pm2 start "streamlit run gui/dashboard.py" --name trading-dashboard
pm2 save
```

## Accessing Remotely

By default, Streamlit only allows local access. To access from other devices:

1. Edit `~/.streamlit/config.toml` (create if it doesn't exist):
```toml
[server]
address = "0.0.0.0"
port = 8501
```

2. Restart the dashboard

3. Access via `http://YOUR_IP:8501`

## Troubleshooting

### Dashboard not updating
- Check if auto-refresh is enabled
- Manually refresh using the "Refresh Now" button
- Check console for errors

### No data showing
- Ensure the bot has run at least once to initialize
- Check internet connection
- Verify market data providers are working

### Port already in use
- Change the port: `streamlit run gui/dashboard.py --server.port 8502`
- Or kill the existing process using port 8501

## Tips

1. **Keep it running**: Use screen, tmux, or PM2 for 24/7 operation
2. **Monitor resources**: The dashboard is lightweight but monitor CPU/memory
3. **Customize refresh**: Adjust refresh interval based on your needs (lower = more updates but more API calls)
4. **Multiple tabs**: Open multiple browser tabs for different views
5. **Mobile access**: Streamlit dashboards work on mobile browsers

## Next Steps

- Add email/SMS alerts for significant signals
- Add historical performance charts
- Add backtesting results
- Add more technical indicators
- Add portfolio allocation charts

