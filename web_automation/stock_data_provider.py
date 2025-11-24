"""
Stock market data provider using Yahoo Finance.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
import yfinance as yf
import pandas as pd

from utils.config import Config


class StockDataProvider:
    """Provides real-time stock market data from Yahoo Finance."""
    
    def __init__(self, config: Config):
        """Initialize stock data provider."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.stocks")
        self.default_symbols = [
            # Technology (Expanded)
            "AAPL", "MSFT", "GOOGL", "GOOG", "AMZN", "META", "NVDA", "NFLX", "AMD",
            "INTC", "CRM", "ORCL", "ADBE", "CSCO", "IBM", "QCOM", "TXN", "AVGO",
            "NOW", "SNPS", "CDNS", "INTU", "ADSK", "ZM", "DOCN", "NET",
            "CRWD", "ZS", "PANW", "FTNT", "OKTA", "TEAM", "DDOG", "MDB", "SNOW",
            "PLTR", "RBLX", "U", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "LCID",
            # Finance (Expanded)
            "JPM", "BAC", "WFC", "GS", "MS", "C", "V", "MA", "AXP", "PYPL",
            "SCHW", "BLK", "COF", "USB", "PNC", "TFC", "BK", "STT", "FITB",
            "KEY", "HBAN", "CFG", "MTB", "ZION", "RF", "CMA", "WTFC", "ONB",
            # Healthcare (Expanded)
            "JNJ", "UNH", "PFE", "ABT", "TMO", "ABBV", "MRK", "LLY", "BMY",
            "AMGN", "GILD", "CVS", "CI", "HUM", "ELV", "CNC", "MOH",
            "BIIB", "REGN", "VRTX", "ILMN", "MRNA", "BNTX", "NVAX",
            "TDOC", "OMCL", "RGNX", "FOLD", "ARWR", "IONS", "ALKS", "SGMO",
            # Consumer (Expanded)
            "WMT", "HD", "MCD", "SBUX", "NKE", "TGT", "LOW", "COST", "TJX",
            "PG", "KO", "PEP", "DIS", "CMCSA", "VZ", "T", "NFLX", "AMZN",
            "TSLA", "F", "GM", "FORD", "HMC", "TM", "RIVN", "LCID", "NIO",
            "XPEV", "LI", "BZ", "RACE", "LULU", "DKS", "BBY", "GME",
            "AMC", "BBBY", "WEN", "YUM", "DPZ", "CMG",
            # Energy & Industrial (Expanded)
            "XOM", "CVX", "COP", "SLB", "BA", "CAT", "GE", "HON", "RTX",
            "LMT", "NOC", "GD", "TD", "MPC", "VLO", "PSX",
            "FANG", "OVV", "CTRA", "MTDR", "SM", "AR", "EQT", "RRC",
            # Real Estate & REITs
            "AMT", "PLD", "EQIX", "PSA", "WELL", "VICI", "SPG", "O", "DLR",
            "EXPI", "Z", "OPEN", "COMP", "HOUS", "RKT", "UWMC",
            # Materials & Mining
            "FCX", "NEM", "AA", "CLF", "STLD", "NUE", "CMC",
            "VALE", "RIO", "BHP", "SCCO", "TECK", "LAC", "ALB", "SQM",
            # Utilities
            "NEE", "DUK", "SO", "AEP", "SRE", "EXC", "XEL", "ES", "ETR",
            # Communication Services
            "GOOGL", "GOOG", "META", "NFLX", "DIS", "CMCSA", "VZ", "T", "TMUS",
            "LUMN", "ATUS", "SHEN", "CABO", "CHTR", "ROKU", "SPOT", "PINS", "SNAP",
            # Chinese Stocks
            "BABA", "JD", "PDD", "BIDU", "TME", "NIO", "XPEV", "LI", "BZ",
            "TAL", "EDU", "BILI", "WB", "DOYU", "HUYA", "MOMO", "VIPS",
            # ETFs & Index Funds (Expanded)
            "SPY", "QQQ", "DIA", "IWM", "VTI", "VOO", "VEA", "VWO", "AGG",
            "TLT", "GLD", "SLV", "USO", "UNG", "ARKK", "ARKQ", "ARKG", "ARKW",
            "TQQQ", "SQQQ", "SPXL", "SPXS", "UPRO", "SPXU", "FAS", "FAZ", "TNA",
            # Meme Stocks & High Volatility
            "GME", "AMC", "BBBY", "KOSS", "SNDL", "TLRY", "CGC",
            # Biotech & Pharma
            "MRNA", "BNTX", "NVAX", "GILD", "REGN", "VRTX", "BIIB", "ILMN",
            # Cloud & SaaS
            "CRM", "NOW", "TEAM", "ZM", "DOCN", "NET", "DDOG", "MDB", "SNOW",
            "PLTR", "U", "ESTC", "FROG", "BILL", "APPN", "VEEV", "WK",
            # Semiconductors
            "NVDA", "AMD", "INTC", "QCOM", "TXN", "AVGO", "MRVL", "SWKS", "QRVO",
            "MPWR", "MCHP", "ON", "WOLF", "ALGM", "DIOD", "POWI", "SLAB", "SITM",
            # Cybersecurity
            "CRWD", "ZS", "PANW", "FTNT", "OKTA", "S", "QLYS", "VRRM", "RDWR",
            "TENB", "RPD", "ESTC", "QLYS", "VRRM", "RDWR", "TENB", "RPD",
            # Fintech
            "PYPL", "COIN", "HOOD", "SOFI", "AFRM", "UPST", "LC", "NU",
            "PAG", "FISV", "FIS", "GPN", "WU", "FOUR", "BILL", "NCNO",
            # E-commerce & Retail
            "AMZN", "SHOP", "ETSY", "W", "CVNA", "VRM", "CARG", "ABG",
            # Gaming & Entertainment
            "RBLX", "U", "EA", "TTWO", "DKNG", "PENN", "LNW",
            # Space & Aerospace
            "RKLB", "SPCE", "RDW", "IRDM", "VSAT",
            # Electric Vehicles
            "TSLA", "RIVN", "LCID", "F", "GM", "NIO", "XPEV", "LI", "BZ",
            "FORD", "HMC", "TM", "RACE", "STLA", "VWAGY",
            # AI & Machine Learning
            "NVDA", "AMD", "INTC", "GOOGL", "MSFT", "META", "AAPL", "AMZN",
            "PLTR", "AI", "BBAI", "SOUN", "PRST",
            # Renewable Energy
            "ENPH", "SEDG", "RUN", "FSLR", "NEE", "BEP", "BEPC", "CWEN",
            # Cannabis
            "TLRY", "CGC", "ACB", "SNDL", "OGI", "CRON", "VFF",
            # Sports Betting
            "DKNG", "PENN", "BETZ", "ACHR", "LNW", "FLUT",
            # 3D Printing
            "DDD", "SSYS", "PRNT", "XONE", "MTLS", "PRNT",
            # Robotics & Automation
            "ROBO", "BOTZ", "IRBT", "TER", "ZBRA", "FANUY", "YASKY",
            # Quantum Computing
            "IONQ", "QUBT", "QBTS", "HON", "IBM", "GOOGL", "MSFT",
            # Metaverse & VR
            "META", "RBLX", "U", "NVDA", "AAPL", "MSFT", "GOOGL", "SONY",
            
            # Additional Large Cap Stocks
            "BRK.B", "JPM", "V", "MA", "WMT", "HD", "PG", "KO", "PEP", "DIS",
            "NFLX", "CMCSA", "T", "VZ", "XOM", "CVX", "JNJ", "UNH", "ABBV",
            
            # Mid-Cap Growth Stocks
            "SQ", "ROKU", "ZM", "DOCU", "FROG", "BILL", "ESTC", "APPN",
            "VEEV", "WK", "NCNO", "FOUR", "NU", "LC", "PAG", "FISV", "FIS",
            "GPN", "WU", "ABG", "CARG", "VRM", "CVNA", "W", "SHOP", "ETSY",
            
            # International Stocks (ADRs)
            "ASML", "TSM", "NVO", "UL", "BP", "SHEL", "GSK", "AZN", "DEO",
            "NVS", "RHHBY", "TM", "HMC", "SONY", "MUFG", "SMFG", "MFG",
            
            # Small Cap & Growth
            "UPST", "SOFI", "AFRM", "HOOD", "COIN", "RIVN", "LCID", "NIO",
            "XPEV", "LI", "BZ", "SPCE", "RKLB", "IRDM", "VSAT", "RDW",
            
            # Dividend Stocks
            "T", "VZ", "XOM", "CVX", "JNJ", "KO", "PEP", "PG", "WMT", "HD",
            "MCD", "SBUX", "NKE", "LOW", "COST", "TGT", "TJX",
            
            # Growth Tech
            "SNOW", "DDOG", "MDB", "NET", "CRWD", "ZS", "PANW", "FTNT",
            "OKTA", "S", "TEAM", "NOW", "CRM", "DOCN", "U", "PLTR",
            
            # Healthcare & Biotech
            "GILD", "REGN", "VRTX", "BIIB", "ILMN", "MRNA", "BNTX", "NVAX",
            "TDOC", "OMCL", "RGNX", "FOLD", "ARWR", "IONS", "ALKS", "SGMO",
            "BMRN", "FATE", "BEAM", "CRISPR", "EDIT", "NTLA", "VERV",
            
            # Energy & Commodities
            "SLB", "HAL", "OXY", "EOG", "DVN", "MRO", "APA", "FANG",
            "OVV", "CTRA", "MTDR", "SM", "AR", "EQT", "RRC", "SWN",
            
            # Industrial & Manufacturing
            "DE", "CAT", "EMR", "ETN", "ITW", "PH", "ROK", "CMI",
            "PCAR", "WWD", "FTV", "GGG", "AOS", "AME", "DOV",
            
            # Consumer Discretionary
            "NKE", "LULU", "DKS", "BBY", "GME", "AMC", "BBBY", "WEN",
            "YUM", "DPZ", "CMG", "SBUX", "MCD", "DIN", "BLMN", "CAKE",
            
            # Real Estate & REITs
            "AMT", "PLD", "EQIX", "PSA", "WELL", "VICI", "SPG", "O",
            "DLR", "EXPI", "Z", "OPEN", "COMP", "HOUS", "RKT", "UWMC",
            "RDFN", "REDF", "REAX", "REMAX",
            
            # Financial Services
            "COF", "USB", "PNC", "TFC", "BK", "STT", "FITB", "KEY",
            "HBAN", "CFG", "MTB", "ZION", "RF", "CMA", "WTFC", "ONB",
            "FNB", "HOMB", "UMBF", "TCBI", "WAFD",
            
            # Materials & Chemicals
            "LIN", "APD", "ECL", "SHW", "PPG", "DD", "DOW", "CE",
            "FMC", "NEM", "FCX", "AA", "CLF", "STLD", "NUE", "CMC",
            "VALE", "RIO", "BHP", "SCCO", "TECK", "LAC", "ALB", "SQM",
            
            # Utilities
            "NEE", "DUK", "SO", "AEP", "SRE", "EXC", "XEL", "ES", "ETR",
            "PEG", "ED", "EIX", "FE", "AEE", "CMS", "CNP", "LNT",
            
            # Communication & Media
            "CHTR", "ROKU", "SPOT", "PINS", "SNAP", "TWTR", "PARA",
            "FOX", "FOXA", "NWSA", "NWS", "LSXMA", "LSXMB", "LSXMK",
            
            # Transportation & Logistics
            "UPS", "FDX", "JBHT", "CHRW", "XPO", "KNX", "ODFL", "ARCB",
            "HUBG", "WERN", "RLGT", "MRTN", "PTSI",
            
            # Aerospace & Defense
            "LMT", "NOC", "GD", "RTX", "BA", "HWM", "TXT", "HXL",
            "HEI", "CW", "AIR", "SPR", "HII", "LDOS",
            
            # Food & Beverage
            "MDLZ", "GIS", "K", "CPB", "SJM", "HSY", "CAG", "HRL",
            "TSN", "BG", "ADM", "INGR", "FLO", "LW",
            
            # Retail & E-commerce
            "TGT", "WMT", "COST", "HD", "LOW", "TJX", "ROST", "BBY",
            "DKS", "ANF", "AEO", "GPS", "URBN", "DKS", "HIBB",
            
            # Total: 500+ stocks across all sectors
        ]
        
    async def get_market_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get real-time market data for stock symbols.
        
        Args:
            symbols: List of stock symbols. If None, uses default watchlist.
            
        Returns:
            Dictionary of market data keyed by symbol
        """
        if not symbols:
            symbols = self.default_symbols
        
        market_data = {}
        
        try:
            # Fetch data in larger batches for faster loading
            batch_size = 50  # Increased from 20 to 50 for faster loading
            for i in range(0, len(symbols), batch_size):
                batch = symbols[i:i + batch_size]
                self.logger.debug(f"Fetching stock batch {i//batch_size + 1}/{(len(symbols) + batch_size - 1)//batch_size}")
                
                # Fetch data for batch in parallel
                tasks = [self._fetch_symbol_data(symbol) for symbol in batch]
                results = await asyncio.gather(*tasks, return_exceptions=True)
                
                for symbol, result in zip(batch, results):
                    if isinstance(result, Exception):
                        self.logger.debug(f"Error fetching {symbol}: {result}")
                        continue
                    if result:
                        market_data[symbol] = result
                
                # Reduced delay between batches for faster loading
                if i + batch_size < len(symbols):
                    await asyncio.sleep(0.2)  # Reduced from 0.5 to 0.2 seconds
                    
        except Exception as e:
            self.logger.error(f"Error fetching stock data: {e}", exc_info=True)
        
        if not market_data:
            self.logger.warning("No stock data returned - may be market closed or API issue")
        else:
            self.logger.info(f"✅ Successfully fetched {len(market_data)} stocks")
        
        return market_data
    
    async def _fetch_symbol_data(self, symbol: str) -> Optional[Dict[str, Any]]:
        """Fetch data for a single symbol with retry logic."""
        max_retries = 2
        for attempt in range(max_retries):
            try:
                ticker = yf.Ticker(symbol)
                
                # Try to get info with timeout
                try:
                    info = ticker.info
                except Exception as e:
                    self.logger.debug(f"Could not get info for {symbol}: {e}")
                    info = {}
                
                # Get recent price data - try multiple periods
                hist = None
                for period in ["1d", "5d", "1mo"]:
                    try:
                        hist = ticker.history(period=period, timeout=10)
                        if not hist.empty:
                            break
                    except Exception as e:
                        self.logger.debug(f"Could not get {period} data for {symbol}: {e}")
                        continue
                
                if hist is None or hist.empty:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)  # Retry after delay
                        continue
                    return None
                
                latest = hist.iloc[-1]
                current_price = info.get("currentPrice") or info.get("regularMarketPrice") or float(latest.get("Close", 0))
                
                if current_price == 0:
                    if attempt < max_retries - 1:
                        await asyncio.sleep(0.5)
                        continue
                    return None
                
                # Get high/low/volume from history
                high = float(hist["High"].max()) if "High" in hist.columns else float(current_price * 1.02)
                low = float(hist["Low"].min()) if "Low" in hist.columns else float(current_price * 0.98)
                volume = int(hist["Volume"].sum()) if "Volume" in hist.columns else 0
                
                # Calculate change
                previous_close = info.get("previousClose") or info.get("regularMarketPreviousClose")
                if previous_close:
                    change = current_price - previous_close
                    change_percent = (change / previous_close) * 100
                else:
                    # Estimate from history
                    if len(hist) > 1:
                        prev_close = float(hist.iloc[-2].get("Close", current_price))
                        change = current_price - prev_close
                        change_percent = (change / prev_close) * 100 if prev_close > 0 else 0
                    else:
                        change = 0
                        change_percent = 0
                
                return {
                    "symbol": symbol,
                    "asset_type": "stock",
                    "price": float(current_price),
                    "open": float(latest.get("Open", current_price)),
                    "high": float(high),
                    "low": float(low),
                    "close": float(current_price),
                    "volume": int(volume),
                    "previous_close": float(previous_close if previous_close else current_price),
                    "change": float(change),
                    "change_percent": float(change_percent),
                    "market_cap": info.get("marketCap"),
                    "timestamp": datetime.now().isoformat(),
                }
                
            except Exception as e:
                self.logger.debug(f"Error fetching {symbol} (attempt {attempt + 1}): {e}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(0.5)  # Retry after delay
                    continue
                return None
        
        return None
    
    async def get_historical_data(
        self, 
        symbol: str, 
        period: str = "1mo"
    ) -> pd.DataFrame:
        """
        Get historical data for a symbol.
        
        Args:
            symbol: Stock symbol
            period: Period to fetch (1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max)
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period=period)
            return hist
        except Exception as e:
            self.logger.error(f"Error fetching historical data for {symbol}: {e}")
            return pd.DataFrame()
    
    def get_watchlist_symbols(self) -> List[str]:
        """Get default watchlist symbols."""
        return self.default_symbols.copy()
    
    def get_top_symbols(self, count: int = 50) -> List[str]:
        """Get top N symbols for fast initial load."""
        # Return top symbols (first N from default list)
        return self.default_symbols[:count]
    
    def set_watchlist(self, symbols: List[str]):
        """Set custom watchlist."""
        self.default_symbols = symbols
        self.logger.info(f"Updated stock watchlist: {symbols}")

