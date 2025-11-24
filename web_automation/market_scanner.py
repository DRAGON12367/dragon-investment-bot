"""
Unified market scanner for both stocks and cryptocurrencies.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime

from web_automation.stock_data_provider import StockDataProvider
from web_automation.crypto_data_provider import CryptoDataProvider
from utils.config import Config


class MarketScanner:
    """Scans both stock and crypto markets for trading opportunities."""
    
    def __init__(self, config: Config):
        """Initialize market scanner."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.scanner")
        self.stock_provider = StockDataProvider(config)
        self.crypto_provider = CryptoDataProvider(config)
        self.scan_stocks = True
        self.scan_crypto = True
        
    async def connect(self):
        """Initialize connections."""
        await self.crypto_provider.connect()
        self._connected = True
        self.logger.info("Market scanner connected")
    
    async def disconnect(self):
        """Close connections."""
        await self.crypto_provider.disconnect()
        self.logger.info("Market scanner disconnected")
    
    async def scan_markets(
        self,
        stock_symbols: Optional[List[str]] = None,
        crypto_symbols: Optional[List[str]] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Scan both stock and crypto markets.
        
        Args:
            stock_symbols: List of stock symbols to scan
            crypto_symbols: List of crypto symbols to scan
            
        Returns:
            Dictionary with 'stocks' and 'crypto' keys containing market data
        """
        results = {
            "stocks": {},
            "crypto": {},
            "timestamp": datetime.now().isoformat()
        }
        
        # Scan stocks and crypto in parallel
        tasks = []
        
        if self.scan_stocks:
            tasks.append(self._scan_stocks(stock_symbols))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))
        
        if self.scan_crypto:
            tasks.append(self._scan_crypto(crypto_symbols))
        else:
            tasks.append(asyncio.create_task(asyncio.sleep(0)))
        
        stock_data, crypto_data = await asyncio.gather(*tasks, return_exceptions=True)
        
        if isinstance(stock_data, Exception):
            self.logger.error(f"Error scanning stocks: {stock_data}", exc_info=True)
            stock_data = {}
        
        if isinstance(crypto_data, Exception):
            self.logger.error(f"Error scanning crypto: {crypto_data}", exc_info=True)
            crypto_data = {}
        
        if stock_data:
            results["stocks"] = stock_data
            self.logger.info(f"✅ Scanned {len(stock_data)} stocks")
        else:
            self.logger.warning("⚠️ No stock data returned from scan (market may be closed or API issue)")
        
        if crypto_data:
            results["crypto"] = crypto_data
            self.logger.info(f"✅ Scanned {len(crypto_data)} cryptocurrencies")
        else:
            self.logger.warning("⚠️ No crypto data returned from scan (may be rate limited - will use cached data if available)")
            # Not a critical error - will use cached data
        
        return results
    
    async def _scan_stocks(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan stock market."""
        try:
            return await self.stock_provider.get_market_data(symbols)
        except Exception as e:
            self.logger.error(f"Error scanning stocks: {e}", exc_info=True)
            return {}
    
    async def _scan_crypto(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """Scan crypto market."""
        try:
            self.logger.info("Starting crypto market scan...")
            result = await self.crypto_provider.get_market_data(symbols)
            self.logger.info(f"Crypto scan returned {len(result)} items")
            return result
        except Exception as e:
            self.logger.error(f"Error scanning crypto: {e}", exc_info=True)
            return {}
    
    async def get_all_market_data(self, fast_load: bool = False) -> Dict[str, Any]:
        """
        Get all market data in a unified format.
        
        Args:
            fast_load: If True, only fetch top 50 cryptos and top 50 stocks for faster initial load
        
        Returns:
            Dictionary with all symbols (stocks and crypto) keyed by symbol
        """
        if fast_load:
            # Fast initial load - only top symbols
            top_crypto = self.crypto_provider.get_top_symbols(50)
            top_stocks = self.stock_provider.get_top_symbols(50)
            self.logger.info(f"Fast load mode: fetching top {len(top_crypto)} crypto and top {len(top_stocks)} stocks")
            scan_results = await self.scan_markets(stock_symbols=top_stocks, crypto_symbols=top_crypto)
        else:
            # Full load - all symbols
            scan_results = await self.scan_markets()
        
        # Combine stocks and crypto into single dictionary
        all_data = {}
        
        stocks_dict = scan_results.get("stocks", {})
        crypto_dict = scan_results.get("crypto", {})
        
        self.logger.debug(f"Combining data: {len(stocks_dict)} stocks, {len(crypto_dict)} crypto")
        
        for symbol, data in stocks_dict.items():
            all_data[symbol] = data
        
        for symbol, data in crypto_dict.items():
            all_data[symbol] = data
        
        # Log final counts
        final_stocks = len([s for s, d in all_data.items() if d.get('asset_type') == 'stock'])
        final_crypto = len([s for s, d in all_data.items() if d.get('asset_type') == 'crypto'])
        self.logger.info(f"Combined market data: {final_stocks} stocks, {final_crypto} crypto (total: {len(all_data)})")
        
        return all_data
    
    def set_scan_options(self, scan_stocks: bool = True, scan_crypto: bool = True):
        """Configure which markets to scan."""
        self.scan_stocks = scan_stocks
        self.scan_crypto = scan_crypto
        self.logger.info(f"Scan options: Stocks={scan_stocks}, Crypto={scan_crypto}")
    
    def set_stock_watchlist(self, symbols: List[str]):
        """Set stock watchlist."""
        self.stock_provider.set_watchlist(symbols)
    
    def set_crypto_watchlist(self, symbols: List[str]):
        """Set crypto watchlist."""
        self.crypto_provider.set_watchlist(symbols)

