"""
Cryptocurrency market data provider using CoinGecko API.
"""
import asyncio
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime
from pathlib import Path
import aiohttp
import requests

from utils.config import Config


class CryptoDataProvider:
    """Provides real-time cryptocurrency market data from CoinGecko."""
    
    BASE_URL = "https://api.coingecko.com/api/v3"
    
    def __init__(self, config: Config):
        """Initialize crypto data provider."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.crypto")
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Rate limiting and caching - INCREASED to reduce API calls
        self.last_fetch_time = 0
        self.cached_data = {}
        self.cache_duration = 300  # Cache for 5 minutes to reduce API calls (was 60 seconds)
        self.rate_limit_delay = 2  # Delay between requests in seconds
        self.max_retries = 3
        self.retry_delays = [5, 15, 30]  # Exponential backoff delays
        
        # Detect if we're running in Streamlit
        import sys
        self.use_requests = 'streamlit' in sys.modules or any('streamlit' in str(arg) for arg in sys.argv)
        if self.use_requests:
            self.logger.info("Streamlit detected - will use requests library for HTTP calls")
        
        # Expanded crypto symbols - Top 150+ cryptocurrencies
        # Focus on most liquid and popular coins across all categories
        self.default_symbols = [
            # Top Tier (Top 20) - Most important
            "bitcoin", "ethereum", "tether", "binancecoin", "solana",
            "ripple", "usd-coin", "staked-ether", "cardano", "dogecoin",
            "tron", "chainlink", "polygon", "polkadot", "litecoin",
            "bitcoin-cash", "avalanche-2", "shiba-inu", "uniswap", "ethereum-classic",
            
            # Second Tier (21-50) - Popular altcoins
            "stellar", "monero", "okb", "cosmos", "internet-computer",
            "filecoin", "aptos", "hedera-hashgraph", "near", "optimism",
            "arbitrum", "immutable-x", "vechain", "the-graph", "aave",
            "algorand", "eos", "tezos", "theta-token", "axie-infinity",
            "maker", "dash", "zcash", "decentraland", "the-sandbox",
            "gala", "enjincoin", "flow", "chiliz", "loopring",
            
            # Third Tier (51-80) - Emerging & DeFi
            "fantom", "multiversx", "sui", "pepe", "floki", "bonk",
            "injective-protocol", "render-token", "fetch-ai", "the-sandbox",
            "crypto-com-chain", "cronos", "kava", "thorchain", "osmosis",
            "jupiter-exchange-solana", "pyth-network", "sei-network", "celestia",
            "dymension", "tia", "sei", "jup", "pyth", "wld", "op", "arb",
            "manta-network", "altlayer", "xai", "ai16z", "tensor", "jito",
            
            # Fourth Tier (81-110) - Layer 2 & Scaling
            "polygon", "arbitrum", "optimism", "base", "zksync", "starknet",
            "metis", "boba-network", "immutable-x", "loopring", "polygon-zkevm",
            "scroll", "mantle", "linea", "taiko", "blast", "mode", "zora",
            "l2beat", "dydx", "perpetual-protocol", "gmx", "synthetix",
            
            # Fifth Tier (111-140) - Meme Coins & Social
            "dogecoin", "shiba-inu", "pepe", "floki", "bonk", "wif",
            "bome", "myro", "popcat", "grok", "turbo", "doge", "shib",
            "feg-token", "safemoon", "baby-doge-coin", "dogelon-mars",
            "shiba-prediction", "shibarium", "bone-shibaswap", "leash",
            
            # Sixth Tier (141-150+) - AI & Gaming Tokens
            "fetch-ai", "singularitynet", "ocean-protocol", "numeraire",
            "cortex", "deepbrain-chain", "matrix-ai-network", "agi-token",
            "axie-infinity", "the-sandbox", "decentraland", "gala", "enjincoin",
            "illuvium", "star-atlas", "alien-worlds", "splinterlands",
            "gods-unchained", "my-neighbor-alice", "treeverse", "emperor",
            
            # Additional High-Volume Coins
            "bitcoin-cash-sv", "bitcoin-gold", "zcash", "monero", "dash",
            "ravencoin", "digibyte", "vertcoin", "groestlcoin", "feathercoin"
        ]
        # Total: 150+ cryptocurrencies tracked
        
        # Mapping of common symbols to CoinGecko IDs
        self.symbol_mapping = {
            # Major cryptocurrencies
            "BTC": "bitcoin",
            "ETH": "ethereum",
            "BNB": "binancecoin",
            "XRP": "ripple",
            "ADA": "cardano",
            "SOL": "solana",
            "DOT": "polkadot",
            "DOGE": "dogecoin",
            "MATIC": "polygon",
            "AVAX": "avalanche-2",
            "LINK": "chainlink",
            "LTC": "litecoin",
            "UNI": "uniswap",
            "ETC": "ethereum-classic",
            "XLM": "stellar",
            "USDT": "tether",
            "USDC": "usd-coin",
            "BCH": "bitcoin-cash",
            "TRX": "tron",
            "ATOM": "cosmos",
            "ALGO": "algorand",
            "VET": "vechain",
            "FIL": "filecoin",
            "ICP": "internet-computer",
            "APT": "aptos",
            "HBAR": "hedera-hashgraph",
            "NEAR": "near",
            "OP": "optimism",
            "ARB": "arbitrum",
            "AAVE": "aave",
            "MKR": "maker",
            "GRT": "the-graph",
            "SAND": "the-sandbox",
            "MANA": "decentraland",
            "AXS": "axie-infinity",
            "THETA": "theta-token",
            "EGLD": "multiversx",
            "FTM": "fantom",
            "SUI": "sui",
            "PEPE": "pepe",
            "FLOKI": "floki",
            "BONK": "bonk",
        }
        
    async def connect(self):
        """Initialize HTTP session."""
        if not self.session and not self.use_requests:
            # Only create aiohttp session if not using requests
            try:
                self.session = aiohttp.ClientSession()
                self.logger.info("Using aiohttp for async HTTP requests")
            except Exception as e:
                self.logger.warning(f"Could not create aiohttp session: {e}, will use requests")
                self.use_requests = True
                self.session = None
    
    async def disconnect(self):
        """Close HTTP session."""
        if self.session:
            await self.session.close()
            self.session = None
    
    async def get_market_data(self, symbols: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Get real-time market data for cryptocurrency symbols.
        
        Args:
            symbols: List of crypto symbols or CoinGecko IDs. If None, uses default list.
            
        Returns:
            Dictionary of market data keyed by symbol
        """
        # Check cache first to reduce API calls
        import time
        current_time = time.time()
        cache_key = str(symbols) if symbols else "default"
        
        # Use cached data if available and not expired
        if (cache_key in self.cached_data and 
            current_time - self.last_fetch_time < self.cache_duration):
            self.logger.debug(f"Returning cached crypto data (age: {int(current_time - self.last_fetch_time)}s)")
            return self.cached_data[cache_key]
        
        # If rate limited recently, return cached data even if stale
        if (cache_key in self.cached_data and 
            hasattr(self, 'rate_limited_until') and 
            current_time < self.rate_limited_until):
            self.logger.warning("Rate limited - returning stale cached data")
            return self.cached_data[cache_key]
        
        # Ensure session is connected (or requests is ready)
        if self.session is None:
            await self.connect()
        
        # If session is still None, we're using requests (which is fine)
        # No need to error out - requests will be used via run_in_executor
        
        if not symbols:
            symbols = self.default_symbols
        else:
            # Convert common symbols to CoinGecko IDs
            symbols = [self.symbol_mapping.get(s.upper(), s.lower()) for s in symbols]
        
        market_data = {}
        
        try:
            # CoinGecko allows up to 250 IDs per request, but we limit to 50 to avoid rate limits
            ids = ",".join(symbols[:50])
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true",
                "include_24hr_vol": "true",
                "include_last_updated_at": "true"
            }
            
            # Retry logic with exponential backoff
            data = None
            for attempt in range(self.max_retries):
                try:
                    # Add delay between requests to avoid rate limits
                    if attempt > 0:
                        delay = self.retry_delays[min(attempt - 1, len(self.retry_delays) - 1)]
                        self.logger.info(f"Retrying crypto fetch (attempt {attempt + 1}/{self.max_retries}) after {delay}s delay...")
                        await asyncio.sleep(delay)
                    
                    # Use requests library wrapped in asyncio.run_in_executor for Streamlit compatibility
                    if self.use_requests or self.session is None:
                        # Fallback to synchronous requests
                        loop = asyncio.get_event_loop()
                        response = await loop.run_in_executor(
                            None, 
                            lambda: requests.get(url, params=params, timeout=30, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                        )
                        if response.status_code == 200:
                            try:
                                data = response.json()
                                if not data:
                                    self.logger.warning("Empty price data returned from CoinGecko")
                                else:
                                    break  # Success, exit retry loop
                            except Exception as e:
                                self.logger.warning(f"Error parsing price data: {e}")
                                data = None
                        elif response.status_code == 429:
                            # Rate limited - set backoff time
                            import time
                            self.rate_limited_until = time.time() + 60  # Wait 1 minute
                            self.logger.warning(f"CoinGecko API rate limit (429) - attempt {attempt + 1}/{self.max_retries}. Will retry after delay.")
                            if attempt < self.max_retries - 1:
                                continue  # Retry
                            else:
                                # Last attempt failed, use cached data
                                if cache_key in self.cached_data:
                                    self.logger.warning("Rate limited on final attempt - returning cached data")
                                    return self.cached_data[cache_key]
                        else:
                            self.logger.warning(f"Failed to get crypto prices: {response.status_code}")
                            if attempt < self.max_retries - 1:
                                continue  # Retry
                    else:
                        # Use aiohttp if available
                        await asyncio.sleep(self.rate_limit_delay)  # Delay to avoid rate limits
                        response = await self.session.get(url, params=params)
                        async with response:
                            if response.status == 200:
                                try:
                                    data = await response.json()
                                    if not data:
                                        self.logger.warning("Empty price data returned from CoinGecko")
                                    else:
                                        break  # Success, exit retry loop
                                except Exception as e:
                                    self.logger.warning(f"Error parsing price data: {e}")
                                    data = None
                            elif response.status == 429:
                                # Rate limited
                                import time
                                self.rate_limited_until = time.time() + 60
                                self.logger.warning(f"CoinGecko API rate limit (429) - attempt {attempt + 1}/{self.max_retries}")
                                if attempt < self.max_retries - 1:
                                    continue  # Retry
                                else:
                                    if cache_key in self.cached_data:
                                        self.logger.warning("Rate limited on final attempt - returning cached data")
                                        return self.cached_data[cache_key]
                            else:
                                self.logger.warning(f"Failed to get crypto prices: {response.status}")
                                if attempt < self.max_retries - 1:
                                    continue  # Retry
                except Exception as e:
                    self.logger.warning(f"Error on attempt {attempt + 1}: {e}")
                    if attempt < self.max_retries - 1:
                        continue  # Retry
                    else:
                        self.logger.error(f"All retry attempts failed: {e}")
                        data = None
                        break
            
            # Process the data if we got any
            if data:
                # Get detailed market data
                market_url = f"{self.BASE_URL}/coins/markets"
                market_params = {
                    "ids": ids,
                    "vs_currency": "usd",
                    "order": "market_cap_desc",
                    "per_page": "250",
                    "page": "1",
                    "sparkline": "false"
                }
                
                try:
                    # Add delay before market data request to avoid rate limits
                    await asyncio.sleep(self.rate_limit_delay)
                    
                    # Use requests library wrapped in asyncio for Streamlit compatibility
                    if self.use_requests or self.session is None:
                        # Fallback to synchronous requests
                        loop = asyncio.get_event_loop()
                        market_response = await loop.run_in_executor(
                            None,
                            lambda: requests.get(market_url, params=market_params, timeout=30, headers={
                                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
                            })
                        )
                        if market_response.status_code == 200:
                            market_info = market_response.json()
                            
                            # Create a lookup by ID
                            market_lookup = {item["id"]: item for item in market_info}
                            
                            for coin_id, price_data in data.items():
                                try:
                                    market_info_item = market_lookup.get(coin_id, {})
                                    
                                    # Get symbol from market info or use coin_id
                                    symbol = market_info_item.get("symbol", coin_id).upper()
                                    
                                    current_price = price_data.get("usd", 0)
                                    if current_price == 0:
                                        continue
                                    
                                    change_24h = price_data.get("usd_24h_change", 0) or 0
                                    
                                    market_data[symbol] = {
                                        "symbol": symbol,
                                        "asset_type": "crypto",
                                        "price": float(current_price),
                                        "open": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                                        "high": float(market_info_item.get("high_24h", current_price)),
                                        "low": float(market_info_item.get("low_24h", current_price)),
                                        "close": float(current_price),
                                        "volume": float(market_info_item.get("total_volume", 0)),
                                        "previous_close": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                                        "change": float(current_price * change_24h / 100) if change_24h else 0.0,
                                        "change_percent": float(change_24h) if change_24h else 0.0,
                                        "market_cap": market_info_item.get("market_cap"),
                                        "timestamp": datetime.now().isoformat(),
                                    }
                                except Exception as e:
                                    self.logger.debug(f"Error processing {coin_id}: {e}")
                                    continue
                        else:
                            error_text = market_response.text
                            self.logger.warning(f"Failed to get market details: {market_response.status_code} - {error_text[:100]}")
                            # Fallback: use price data only
                            self._process_price_data_only(data, market_data)
                    else:
                        # Use aiohttp if available
                        market_response = await self.session.get(market_url, params=market_params)
                        async with market_response:
                            if market_response.status == 200:
                                market_info = await market_response.json()
                                
                                # Create a lookup by ID
                                market_lookup = {item["id"]: item for item in market_info}
                                
                                for coin_id, price_data in data.items():
                                    try:
                                        market_info_item = market_lookup.get(coin_id, {})
                                        
                                        # Get symbol from market info or use coin_id
                                        symbol = market_info_item.get("symbol", coin_id).upper()
                                        
                                        current_price = price_data.get("usd", 0)
                                        if current_price == 0:
                                            continue
                                        
                                        change_24h = price_data.get("usd_24h_change", 0) or 0
                                        
                                        market_data[symbol] = {
                                            "symbol": symbol,
                                            "asset_type": "crypto",
                                            "price": float(current_price),
                                            "open": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                                            "high": float(market_info_item.get("high_24h", current_price)),
                                            "low": float(market_info_item.get("low_24h", current_price)),
                                            "close": float(current_price),
                                            "volume": float(market_info_item.get("total_volume", 0)),
                                            "previous_close": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                                            "change": float(current_price * change_24h / 100) if change_24h else 0.0,
                                            "change_percent": float(change_24h) if change_24h else 0.0,
                                            "market_cap": market_info_item.get("market_cap"),
                                            "timestamp": datetime.now().isoformat(),
                                        }
                                    except Exception as e:
                                        self.logger.debug(f"Error processing {coin_id}: {e}")
                                        continue
                            elif market_response.status == 429:
                                self.logger.warning("CoinGecko API rate limit (429) - using price data only")
                                # Fallback: use price data only
                                self._process_price_data_only(data, market_data)
                            else:
                                error_text = await market_response.text()
                                self.logger.warning(f"Failed to get market details: {market_response.status} - {error_text[:100]}")
                                # Fallback: use price data only
                                self._process_price_data_only(data, market_data)
                except Exception as e:
                    self.logger.warning(f"Error getting market details: {e}")
                    # Fallback: use price data only
                    self._process_price_data_only(data, market_data)
            else:
                error_text = "No response data"
                self.logger.warning(f"Failed to get crypto prices: {error_text[:100]}")
                
                # Try alternative endpoint
                try:
                    alt_url = f"{self.BASE_URL}/simple/price"
                    alt_params = {"ids": ids, "vs_currencies": "usd", "include_24hr_change": "true"}
                    if self.use_requests or self.session is None:
                        # Use requests
                        loop = asyncio.get_event_loop()
                        alt_response = await loop.run_in_executor(
                            None,
                            lambda: requests.get(alt_url, params=alt_params, timeout=30)
                        )
                        if alt_response.status_code == 200:
                            data = alt_response.json()
                            self.logger.info("Got crypto data from alternative endpoint")
                            self._process_price_data_only(data, market_data)
                    else:
                        # Use aiohttp
                        alt_response = await self.session.get(alt_url, params=alt_params)
                        async with alt_response:
                            if alt_response.status == 200:
                                data = await alt_response.json()
                                self.logger.info("Got crypto data from alternative endpoint")
                                self._process_price_data_only(data, market_data)
                except Exception as e:
                    self.logger.debug(f"Alternative endpoint also failed: {e}")
        
        except Exception as e:
            self.logger.error(f"Error fetching crypto data: {e}", exc_info=True)
        
        # If still no data, try a simpler approach with just top coins
        if not market_data:
            self.logger.warning("No crypto data from main fetch, attempting simplified fallback...")
            try:
                await asyncio.sleep(self.rate_limit_delay * 2)  # Wait longer before fallback
                await self._fetch_top_coins_simple(market_data)
            except Exception as e:
                self.logger.error(f"Fallback fetch also failed: {e}", exc_info=True)
        
        if not market_data:
            self.logger.error("❌ CRITICAL: No crypto data could be fetched after all attempts!")
            # Return cached data if available, even if stale
            if cache_key in self.cached_data:
                self.logger.warning("Returning stale cached data due to API failure")
                return self.cached_data[cache_key]
            # Try to load from saved data file
            try:
                saved_data = self._load_saved_data()
                if saved_data:
                    self.logger.info("Loaded crypto data from saved file")
                    return saved_data
            except Exception as e:
                self.logger.debug(f"Could not load saved data: {e}")
        else:
            self.logger.info(f"✅ Successfully fetched {len(market_data)} cryptocurrencies")
            # Update cache
            self.cached_data[cache_key] = market_data
            self.last_fetch_time = current_time
            # Save data for future use
            try:
                self._save_data(market_data)
            except Exception as e:
                self.logger.debug(f"Could not save data: {e}")
        
        return market_data
    
    def _save_data(self, data: Dict[str, Any]):
        """Save data to file for recovery."""
        try:
            import json
            data_dir = Path(self.config.data_directory)
            data_dir.mkdir(parents=True, exist_ok=True)
            file_path = data_dir / "crypto_data_backup.json"
            with open(file_path, 'w') as f:
                json.dump(data, f, indent=2)
            self.logger.debug(f"Saved crypto data backup to {file_path}")
        except Exception as e:
            self.logger.debug(f"Could not save data: {e}")
    
    def _load_saved_data(self) -> Optional[Dict[str, Any]]:
        """Load saved data from file."""
        try:
            import json
            import time
            data_dir = Path(self.config.data_directory)
            file_path = data_dir / "crypto_data_backup.json"
            if file_path.exists():
                with open(file_path, 'r') as f:
                    data = json.load(f)
                    # Check if data is recent (within 24 hours)
                    file_age = time.time() - file_path.stat().st_mtime
                    if file_age < 86400:  # 24 hours
                        self.logger.info(f"Loaded crypto data from backup (age: {int(file_age/3600)} hours)")
                        return data
                    else:
                        self.logger.debug("Backup data too old, not using")
        except Exception as e:
            self.logger.debug(f"Could not load saved data: {e}")
        return None
    
    def _process_price_data_only(self, data: Dict[str, Any], market_data: Dict[str, Any]):
        """Process price data when market details are unavailable."""
        if not data:
            return
        
        self.logger.info("Processing price data only (market details unavailable)")
        for coin_id, price_data in data.items():
            try:
                current_price = price_data.get("usd", 0)
                if current_price == 0:
                    continue
                change_24h = price_data.get("usd_24h_change", 0) or 0
                symbol = coin_id.upper()  # Use coin ID as symbol
                
                market_data[symbol] = {
                    "symbol": symbol,
                    "asset_type": "crypto",
                    "price": float(current_price),
                    "open": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                    "high": float(current_price * 1.05),  # Estimate
                    "low": float(current_price * 0.95),  # Estimate
                    "close": float(current_price),
                    "volume": 0.0,
                    "previous_close": float(current_price * (1 - change_24h / 100)) if change_24h else float(current_price),
                    "change": float(current_price * change_24h / 100) if change_24h else 0.0,
                    "change_percent": float(change_24h) if change_24h else 0.0,
                    "market_cap": None,
                    "timestamp": datetime.now().isoformat(),
                }
            except Exception as e:
                self.logger.debug(f"Error processing {coin_id}: {e}")
                continue
    
    async def _fetch_top_coins_simple(self, market_data: Dict[str, Any]):
        """Fetch top coins using a simpler method."""
        try:
            # Ensure we're connected (session or requests ready)
            if self.session is None and not self.use_requests:
                await self.connect()
            
            # If we're using requests, that's fine - we can proceed
            if not self.use_requests and not self.session:
                self.logger.error("Cannot fetch crypto: no HTTP session available")
                return
            
            # Just get top 5 by market cap to minimize API calls
            top_coins = ["bitcoin", "ethereum", "tether", "binancecoin", "solana"]
            
            ids = ",".join(top_coins)
            url = f"{self.BASE_URL}/simple/price"
            params = {
                "ids": ids,
                "vs_currencies": "usd",
                "include_24hr_change": "true"
            }
            
            self.logger.info(f"Attempting simple crypto fetch for: {ids}")
            try:
                # Use requests library wrapped in asyncio for Streamlit compatibility
                if self.use_requests or self.session is None:
                    # Fallback to synchronous requests
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(url, params=params, timeout=30)
                    )
                    if response.status_code == 200:
                        data = response.json()
                        if data:
                            self._process_price_data_only(data, market_data)
                            self.logger.info(f"Fetched {len(market_data)} cryptocurrencies using simple method")
                        else:
                            self.logger.warning("Simple fetch returned empty data")
                    elif response.status_code == 429:
                        self.logger.warning("CoinGecko API rate limit (429). Please wait before refreshing.")
                    else:
                        error_text = response.text
                        self.logger.error(f"Simple fetch failed: {response.status_code} - {error_text[:200]}")
                else:
                    # Use aiohttp if available
                    response = await self.session.get(url, params=params)
                    async with response:
                        if response.status == 200:
                            data = await response.json()
                            if data:
                                self._process_price_data_only(data, market_data)
                                self.logger.info(f"Fetched {len(market_data)} cryptocurrencies using simple method")
                            else:
                                self.logger.warning("Simple fetch returned empty data")
                        else:
                            error_text = await response.text()
                            self.logger.error(f"Simple fetch failed: {response.status} - {error_text[:200]}")
            except Exception as e:
                self.logger.error(f"Simple fetch error: {e}")
        except Exception as e:
            self.logger.error(f"Error in simple fetch: {e}", exc_info=True)
    
    async def get_historical_data(
        self, 
        coin_id: str, 
        days: int = 30
    ) -> Dict[str, Any]:
        """
        Get historical data for a cryptocurrency.
        
        Args:
            coin_id: CoinGecko coin ID
            days: Number of days of history
            
        Returns:
            Dictionary with price history
        """
        await self.connect()
        
        try:
            url = f"{self.BASE_URL}/coins/{coin_id}/market_chart"
            params = {
                "vs_currency": "usd",
                "days": days
            }
            
            try:
                # Use requests library wrapped in asyncio for Streamlit compatibility
                if self.session is None:
                    # Fallback to synchronous requests
                    loop = asyncio.get_event_loop()
                    response = await loop.run_in_executor(
                        None,
                        lambda: requests.get(url, params=params, timeout=30)
                    )
                    if response.status_code == 200:
                        return response.json()
                    else:
                        self.logger.error(f"Failed to get historical data: {response.status_code}")
                        return {}
                else:
                    # Use aiohttp if available
                    response = await self.session.get(url, params=params)
                    async with response:
                        if response.status == 200:
                            return await response.json()
                        else:
                            self.logger.error(f"Failed to get historical data: {response.status}")
                            return {}
            except Exception as e:
                self.logger.error(f"Error getting historical data: {e}")
                return {}
        except Exception as e:
            self.logger.error(f"Error fetching historical data: {e}")
            return {}
    
    def get_watchlist_symbols(self) -> List[str]:
        """Get default watchlist symbols."""
        return self.default_symbols.copy()
    
    def set_watchlist(self, symbols: List[str]):
        """Set custom watchlist."""
        self.default_symbols = symbols
        self.logger.info(f"Updated crypto watchlist: {symbols}")

