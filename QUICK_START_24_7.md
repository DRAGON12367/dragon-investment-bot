# 🚀 Quick Start - 24/7 AI Profit Monitor

## What's New

✅ **Fixed Crypto Data**: Now scans 50+ cryptocurrencies reliably  
✅ **AI Profit Analyzer**: Finds the absolute most profitable opportunities  
✅ **Exact Sell Timing**: Shows exactly when to sell with reasons  
✅ **24/7 Monitoring**: Continuous scanning every 30-60 seconds  
✅ **Live Dashboard**: Professional GUI with real-time updates  

## Quick Start

### 1. Run the Dashboard (Recommended)
```bash
streamlit run gui/dashboard.py
```

The dashboard will show:
- 🏆 **Top Profit Opportunities** - Ranked by AI profit score
- 🚨 **SELL SIGNALS** - Exact timing with reasons
- 📊 **Live Market Data** - Stocks & Crypto
- 💰 **Portfolio Status** - Real-time P&L

### 2. Run 24/7 Monitor (Background)
```bash
python monitor_24_7.py
```

Runs continuously, logging opportunities and sell signals.

## What You'll See

### Top Profit Opportunities
```
🏆 TOP PROFIT OPPORTUNITIES:
  1. BTC (crypto) - STRONG_BUY @ $43,250.00
     Profit Score: 72.5% | Target: $47,575.00 (+10.0%)
     Risk/Reward: 2.00 | 24h Change: +3.2%
  
  2. ETH (crypto) - STRONG_BUY @ $2,580.00
     Profit Score: 68.3% | Target: $2,838.00 (+10.0%)
     Risk/Reward: 2.00 | 24h Change: +2.8%
```

### Sell Signals
```
🚨 SELL SIGNALS DETECTED:
  🚨 AAPL: Take profit target reached: 10.25% gain
     SELL @ $185.50 | Profit: $1,025.00
     Priority: HIGH
  
  ⚠️ DOGE: Trailing stop triggered: 8.50% profit protected
     SELL @ $0.085 | Profit: $850.00
     Priority: HIGH
```

## Features

### AI Profit Scoring
- **Momentum Analysis**: Tracks price momentum
- **Volume Analysis**: Identifies high-volume opportunities
- **Volatility Assessment**: Finds volatile assets for profit
- **Support/Resistance**: Identifies good entry points
- **Risk/Reward**: Calculates optimal risk-reward ratios

### Sell Signal Types
1. **TAKE_PROFIT**: Profit target reached (default: 10%)
2. **TRAILING_STOP**: Protects profits as price rises
3. **STOP_LOSS**: Limits losses (default: 5%)
4. **MOMENTUM_REVERSAL**: Detects reversals early
5. **TIME_EXIT**: Exits stale positions

### 24/7 Scanning
- **Stocks**: 18+ major stocks (AAPL, TSLA, NVDA, etc.)
- **Crypto**: 50+ cryptocurrencies (BTC, ETH, SOL, etc.)
- **Real-time**: Updates every 30-60 seconds
- **Automatic**: Runs continuously without intervention

## Configuration

Edit `config.json`:
```json
{
  "trading_interval": 30,
  "take_profit_percentage": 0.10,
  "stop_loss_percentage": 0.05,
  "max_position_size": 0.1
}
```

## Running 24/7

### Windows
```bash
# Background
start /B python monitor_24_7.py

# Or use Task Scheduler for auto-start
```

### Linux/Mac
```bash
# Background with nohup
nohup python monitor_24_7.py > monitor.log 2>&1 &

# Or use screen
screen -S monitor
python monitor_24_7.py
# Ctrl+A then D to detach
```

## Dashboard Features

1. **Top Profit Opportunities Table**
   - Ranked by AI profit score
   - Shows target prices
   - Risk/reward ratios
   - 24h changes

2. **Sell Signals Section**
   - Color-coded by priority
   - Exact sell prices
   - Profit/loss amounts
   - Reasons for selling

3. **Live Market Data**
   - Separate tables for stocks/crypto
   - Real-time prices
   - Volume and changes

4. **Auto-Refresh**
   - Configurable intervals (5-300 seconds)
   - Manual refresh button
   - Last update timestamp

## Tips

1. **Keep Dashboard Open**: Best for visual monitoring
2. **Check Logs**: Detailed analysis in `logs/bot.log`
3. **Adjust Intervals**: Faster = more opportunities but more API calls
4. **Review Daily**: Check top opportunities each day
5. **Set Alerts**: Configure notifications for critical signals

## Troubleshooting

### No Crypto Data
- Check internet connection
- CoinGecko API may be rate-limited (wait a minute)
- Fallback will use price data only
- Check logs for specific errors

### Dashboard Not Updating
- Enable auto-refresh in sidebar
- Click "Refresh Now" button
- Check console for errors

---

**Your 24/7 AI profit monitor is ready! It will continuously find the most profitable opportunities and tell you exactly when to sell.** 🎯📈

