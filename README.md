# AI Investment Bot - Live Trading (Stocks & Crypto)

An automated trading bot that scans both stock and cryptocurrency markets in real-time. The bot uses machine learning algorithms, technical indicators, and risk management to generate and execute trading signals automatically.

## Features

- **Professional Live Dashboard**: Beautiful web-based GUI for 24/7 monitoring
- **Dual Market Scanning**: Scans both stock and cryptocurrency markets simultaneously
- **Real-time Data**: 
  - Stocks via Yahoo Finance (yfinance)
  - Cryptocurrencies via CoinGecko API
- **Machine Learning**: Random Forest classifier for price prediction
- **Technical Analysis**: RSI, MACD, Bollinger Bands, Moving Averages
- **Risk Management**: Position sizing, stop-loss, take-profit, daily loss limits
- **Paper Trading**: Simulated portfolio for testing strategies
- **Automated Execution**: Executes trades based on ML predictions and risk parameters

## Prerequisites

- Python 3.8 or higher
- Internet connection for market data

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd ai-investment-bot
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. (Optional) Set up environment variables:
Create a `.env` file in the root directory:
```env
LOG_LEVEL=INFO
TRADING_INTERVAL=60
MAX_POSITION_SIZE=0.1
INITIAL_CASH=10000
```

Alternatively, you can create a `config.json` file:
```json
{
  "trading_interval": 60,
  "max_position_size": 0.1,
  "min_confidence_threshold": 0.7,
  "max_daily_loss": 0.02,
  "stop_loss_percentage": 0.05,
  "take_profit_percentage": 0.10,
  "initial_cash": 10000.0
}
```

## Configuration

### Trading Parameters

- `trading_interval`: Time in seconds between trading iterations (default: 60)
- `max_position_size`: Maximum position size as fraction of portfolio (default: 0.1 = 10%)
- `min_confidence_threshold`: Minimum ML confidence to execute trade (default: 0.7)
- `initial_cash`: Starting cash for paper trading (default: 10000.0)

### Risk Management

- `max_daily_loss`: Maximum daily loss as fraction (default: 0.02 = 2%)
- `stop_loss_percentage`: Stop loss percentage (default: 0.05 = 5%)
- `take_profit_percentage`: Take profit percentage (default: 0.10 = 10%)

### Market Watchlists

The bot comes with default watchlists:

**Stocks**: AAPL, MSFT, GOOGL, AMZN, TSLA, META, NVDA, JPM, V, JNJ, WMT, PG, MA, UNH, HD, DIS, BAC, NFLX

**Cryptocurrencies**: BTC, ETH, BNB, XRP, ADA, SOL, DOT, DOGE, MATIC, AVAX, LINK, LTC, UNI, ETC, XLM

You can customize these in the code or via configuration.

## Usage

### Running the Live Dashboard (Recommended)

For a professional GUI experience with real-time monitoring:

```bash
streamlit run gui/dashboard.py
```

Or use the runner script:
```bash
python run_dashboard.py
```

The dashboard will open in your browser at `http://localhost:8501`

See [README_GUI.md](README_GUI.md) for detailed dashboard documentation.

### Running the Bot (Command Line)

For command-line operation:

```bash
python main.py
```

The bot will:
1. Connect to market data providers (Yahoo Finance & CoinGecko)
2. Load or create ML models
3. Start the trading loop:
   - Scan both stock and crypto markets
   - Fetch real-time market data
   - Generate trading signals using ML and technical indicators
   - Apply risk management filters
   - Execute approved trades (paper trading)

### Stopping the Bot

Press `Ctrl+C` to gracefully stop the bot. It will:
- Complete any pending operations
- Close API connections
- Save current state

## Project Structure

```
ai-investment-bot/
├── main.py                 # Main entry point
├── algorithms/             # Trading algorithms
│   ├── strategy.py        # Main trading strategy
│   ├── ml_model.py        # Machine learning model
│   └── technical_indicators.py  # Technical analysis
├── risk_management/        # Risk management
│   └── risk_manager.py    # Risk evaluation and position sizing
├── web_automation/         # Market data providers
│   ├── stock_data_provider.py  # Stock market data (Yahoo Finance)
│   ├── crypto_data_provider.py  # Crypto market data (CoinGecko)
│   ├── market_scanner.py  # Unified market scanner
│   └── broker_client.py   # Broker interface (paper trading)
├── data/                   # Data processing
│   └── data_processor.py  # Market data storage
├── utils/                  # Utilities
│   ├── config.py          # Configuration management
│   └── logger.py          # Logging setup
├── requirements.txt        # Python dependencies
└── README.md              # This file
```

## How It Works

1. **Market Scanning**: 
   - Scans stock market via Yahoo Finance API
   - Scans cryptocurrency market via CoinGecko API
   - Fetches real-time prices, volume, and market data

2. **Signal Generation**: 
   - Calculates technical indicators (RSI, MACD, etc.)
   - Uses ML model to predict price direction
   - Combines signals with confidence scores
   - Separates signals by asset type (stock/crypto)

3. **Risk Evaluation**:
   - Checks daily loss limits
   - Calculates position sizes
   - Validates cash availability
   - Applies stop-loss and take-profit levels

4. **Trade Execution**: Executes approved trades in simulated portfolio (paper trading)

## Market Data Sources

- **Stocks**: Yahoo Finance (yfinance) - Free, no API key required
- **Cryptocurrencies**: CoinGecko API - Free tier available, no API key required for basic usage

## Training the ML Model

The bot comes with a default model, but you can train it on historical data:

```python
from algorithms.ml_model import MLModel
from utils.config import Config
import pandas as pd

config = Config()
model = MLModel(config)

# Load your historical data
X = pd.DataFrame(...)  # Features
y = pd.Series(...)      # Labels (1 for buy, 0 for sell)

await model.train(X, y)
```

## Logging

Logs are saved to `logs/bot.log` and also displayed in the console. Log levels:
- `DEBUG`: Detailed information
- `INFO`: General information
- `WARNING`: Warning messages
- `ERROR`: Error messages

The bot logs:
- Market scan results (stocks vs crypto counts)
- Generated signals (separated by asset type)
- Trade executions
- Portfolio status

## Customizing Watchlists

You can customize the stocks and cryptocurrencies to scan:

```python
from web_automation.market_scanner import MarketScanner
from utils.config import Config

config = Config()
scanner = MarketScanner(config)

# Set custom stock watchlist
scanner.set_stock_watchlist(["AAPL", "TSLA", "NVDA"])

# Set custom crypto watchlist
scanner.set_crypto_watchlist(["bitcoin", "ethereum", "solana"])
```

## Paper Trading

The bot uses a simulated portfolio for paper trading. This allows you to:
- Test strategies without risking real money
- Track performance over time
- Debug and optimize algorithms

To connect to a real broker, you would need to:
1. Implement a broker-specific client
2. Replace the `BrokerClient` class
3. Add authentication and order execution logic

## Risk Warning

⚠️ **IMPORTANT**: 
- This bot is currently set up for **paper trading** (simulated)
- If you connect it to a real broker, it will execute real trades with real money
- Always test thoroughly before live trading
- Monitor the bot regularly
- Set appropriate risk limits
- Understand the risks involved

## Disclaimer

This software is for educational purposes only. Trading involves substantial risk of loss. Past performance does not guarantee future results. Use at your own risk.

## License

[Add your license here]

## Support

For issues or questions, please open an issue on the repository.
