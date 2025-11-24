# 🌙 24/7 AI Monitoring - Most Profitable Opportunities

## Overview

The bot now includes a **24/7 AI monitoring system** that continuously scans stocks and cryptocurrencies to find the **absolute most profitable opportunities** and tells you **exactly when to sell**.

## Features

### 🏆 AI Profit Analyzer
- **Smart Opportunity Detection**: Analyzes 50+ cryptocurrencies and 18+ stocks
- **Profit Scoring**: Ranks opportunities by profit potential (0-100%)
- **Multi-Factor Analysis**: 
  - Momentum scoring
  - Volume analysis
  - Volatility assessment
  - Historical support/resistance
- **Risk/Reward Calculation**: Shows exact risk-reward ratios

### 🚨 Exact Sell Timing
- **Take Profit Signals**: Automatically triggers when profit target reached
- **Trailing Stop Loss**: Protects profits as price rises
- **Stop Loss Protection**: Limits losses
- **Momentum Reversal**: Detects when to exit before reversal
- **Time-Based Exits**: Exits stale positions

### 📊 24/7 Live Tracking
- **Continuous Scanning**: Scans every 30-60 seconds (configurable)
- **Real-time Updates**: Live prices for stocks and crypto
- **Position Tracking**: Monitors all open positions
- **Profit/Loss Tracking**: Real-time P&L for each position

## Running 24/7 Monitor

### Option 1: Command Line Monitor
```bash
python monitor_24_7.py
```

This runs a continuous monitoring service that:
- Scans markets every 30-60 seconds
- Finds top profit opportunities
- Detects sell signals
- Logs everything to console and file

### Option 2: Dashboard (Recommended)
```bash
streamlit run gui/dashboard.py
```

The dashboard shows:
- **Top Profit Opportunities**: Ranked by AI profit score
- **Sell Signals**: Exact timing with reasons
- **Live Market Data**: Real-time prices
- **Portfolio Status**: Current positions and P&L

## Dashboard Sections

### 1. 🏆 Top Profit Opportunities
Shows the **absolute most profitable** opportunities ranked by AI:
- Rank (#1, #2, #3...)
- Symbol and type (Stock/Crypto)
- Action (STRONG_BUY, BUY)
- Current price
- Target price (where to sell)
- Profit score (%)
- Profit potential (%)
- Risk/Reward ratio
- 24h change

### 2. 🚨 SELL SIGNALS - Exact Timing
Shows **exactly when to sell** with:
- **CRITICAL**: Red alerts - Sell immediately
- **HIGH**: Yellow warnings - Sell soon
- **MEDIUM/LOW**: Information - Consider selling

Each signal shows:
- Symbol
- Reason (TAKE_PROFIT, TRAILING_STOP, STOP_LOSS, etc.)
- Current price
- Entry price
- Profit/Loss %
- Profit/Loss $
- Hold duration
- Exact message

### 3. 📊 Market Overview
Live tables showing:
- All stocks with prices, changes, volume
- All cryptocurrencies with prices, changes, volume
- Real-time updates

### 4. 🎯 Trading Signals
ML-generated signals with confidence scores

### 5. 📈 Price Charts
Interactive charts for any symbol

## How It Works

### Finding Profitable Opportunities

1. **Scans 50+ Cryptocurrencies**:
   - Bitcoin, Ethereum, Solana, Cardano, and 46+ more
   - Real-time prices from CoinGecko
   - 24h change, volume, market cap

2. **Scans 18+ Stocks**:
   - AAPL, MSFT, GOOGL, TSLA, NVDA, and more
   - Real-time prices from Yahoo Finance
   - Intraday data

3. **AI Analysis**:
   - Calculates profit potential
   - Scores momentum, volume, volatility
   - Ranks by profit score
   - Identifies best entry points

4. **Recommendations**:
   - STRONG_BUY: Profit score > 60%
   - BUY: Profit score > 40%
   - Shows target price (where to sell)
   - Shows stop loss (risk limit)

### Exact Sell Timing

The bot tracks every position and monitors:

1. **Take Profit**: Sells when profit target reached (default: 10%)
2. **Trailing Stop**: Protects profits - if up 5%+, triggers if drops 5% from peak
3. **Stop Loss**: Sells if loss exceeds limit (default: 5%)
4. **Momentum Reversal**: Detects sharp drops and secures profits
5. **Time Exit**: Exits positions held >30 days with minimal gain

## Configuration

Edit `config.json` or set environment variables:

```json
{
  "trading_interval": 30,
  "take_profit_percentage": 0.10,
  "stop_loss_percentage": 0.05,
  "max_position_size": 0.1
}
```

## Running 24/7

### Windows (Background)
```bash
# Run in background
start /B python monitor_24_7.py
```

### Linux/Mac (Background)
```bash
# Run in background
nohup python monitor_24_7.py > monitor.log 2>&1 &
```

### Using Screen
```bash
screen -S monitor
python monitor_24_7.py
# Press Ctrl+A then D to detach
```

### Using PM2
```bash
pm2 start monitor_24_7.py --name "ai-monitor" --interpreter python
pm2 save
```

## Logs

The monitor logs to:
- Console (real-time)
- `logs/bot.log` (file)

Look for:
- 🏆 Top opportunities
- 🚨 Sell signals
- 📊 Scan summaries
- 💰 Portfolio updates

## Example Output

```
🏆 TOP PROFIT OPPORTUNITIES:
  1. BTC (crypto) - STRONG_BUY @ $43,250.00 | Profit Score: 72.5% | Target: $47,575.00 (+10.0%)
  2. ETH (crypto) - STRONG_BUY @ $2,580.00 | Profit Score: 68.3% | Target: $2,838.00 (+10.0%)
  3. TSLA (stock) - BUY @ $245.50 | Profit Score: 55.2% | Target: $270.05 (+10.0%)

🚨 SELL SIGNALS DETECTED:
  🚨 AAPL: Take profit target reached: 10.25% gain | SELL @ $185.50
  ⚠️ DOGE: Trailing stop triggered: 8.50% profit protected | SELL @ $0.085
```

## Tips

1. **Monitor Dashboard**: Keep dashboard open for visual monitoring
2. **Check Logs**: Review logs for detailed analysis
3. **Adjust Intervals**: Faster scans = more opportunities but more API calls
4. **Set Alerts**: Configure email/SMS for critical sell signals
5. **Review Daily**: Check top opportunities daily

---

**The bot now provides 24/7 AI-powered monitoring to find the most profitable opportunities and tell you exactly when to sell!** 🚀📈

