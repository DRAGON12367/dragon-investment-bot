"""
Broker client interface - Market data provider wrapper with simulated trading.
"""
import logging
from datetime import datetime
from web_automation.market_scanner import MarketScanner
from utils.config import Config


class BrokerClient(MarketScanner):
    """Broker client wrapper that provides market data and portfolio simulation."""
    
    def __init__(self, config: Config):
        """Initialize broker client with market scanner."""
        super().__init__(config)
        self.logger = logging.getLogger("ai_investment_bot.broker_client")
        # Simulated portfolio for paper trading
        initial_cash = getattr(config, 'initial_cash', 10000.0)
        self.simulated_portfolio = {
            "cash": initial_cash,
            "total_value": initial_cash,
            "positions": []
        }
    
    async def get_all_market_data(self, fast_load: bool = False) -> dict:
        """
        Get all market data with improved error handling.
        
        Args:
            fast_load: If True, only fetch top 100 cryptos and top 100 stocks for faster initial load
        
        Returns:
            Dictionary with all market data
        """
        try:
            result = await super().get_all_market_data(fast_load=fast_load)
            if result is None:
                self.logger.warning("get_all_market_data returned None, returning empty dict")
                return {}
            if not isinstance(result, dict):
                self.logger.warning(f"get_all_market_data returned {type(result)}, returning empty dict")
                return {}
            self.logger.info(f"✅ Broker client fetched {len(result)} assets")
            return result
        except Exception as e:
            self.logger.error(f"Error getting market data: {e}", exc_info=True)
            import traceback
            traceback.print_exc()
            # Return empty dict instead of crashing
            return {}
    
    async def get_portfolio_status(self) -> dict:
        """
        Get current portfolio status (simulated for paper trading).
        
        Returns:
            Dictionary with portfolio information
        """
        # Calculate total value from positions
        total_value = self.simulated_portfolio["cash"]
        for position in self.simulated_portfolio["positions"]:
            # In real implementation, would fetch current price
            total_value += position.get("value", 0)
        
        self.simulated_portfolio["total_value"] = total_value
        
        return {
            "cash": self.simulated_portfolio["cash"],
            "total_value": total_value,
            "positions": self.simulated_portfolio["positions"]
        }
    
    async def execute_trades(self, signals: list) -> list:
        """
        Execute trades (simulated for paper trading).
        
        Args:
            signals: List of trading signals
            
        Returns:
            List of execution results
        """
        results = []
        for signal in signals:
            try:
                result = await self._execute_simulated_trade(signal)
                results.append(result)
            except Exception as e:
                self.logger.error(f"Failed to execute trade: {e}")
                results.append({"error": str(e)})
        return results
    
    async def _execute_simulated_trade(self, signal: dict) -> dict:
        """Execute a simulated trade."""
        symbol = signal.get("symbol")
        action = signal.get("action")
        quantity = signal.get("quantity", 0)
        price = signal.get("price", 0)
        
        trade_value = quantity * price
        
        if action == "BUY":
            if trade_value > self.simulated_portfolio["cash"]:
                raise ValueError(f"Insufficient cash: need ${trade_value:.2f}, have ${self.simulated_portfolio['cash']:.2f}")
            
            self.simulated_portfolio["cash"] -= trade_value
            
            # Add or update position
            position = next((p for p in self.simulated_portfolio["positions"] if p["symbol"] == symbol), None)
            if position:
                position["quantity"] += quantity
                position["value"] = position["quantity"] * price
            else:
                self.simulated_portfolio["positions"].append({
                    "symbol": symbol,
                    "quantity": quantity,
                    "value": trade_value,
                    "asset_type": signal.get("asset_type", "unknown")
                })
        
        elif action == "SELL":
            position = next((p for p in self.simulated_portfolio["positions"] if p["symbol"] == symbol), None)
            if not position or position["quantity"] < quantity:
                raise ValueError(f"Insufficient position: trying to sell {quantity}, have {position['quantity'] if position else 0}")
            
            self.simulated_portfolio["cash"] += trade_value
            position["quantity"] -= quantity
            position["value"] = position["quantity"] * price
            
            if position["quantity"] == 0:
                self.simulated_portfolio["positions"].remove(position)
        
        return {
            "id": f"sim_{symbol}_{action}_{datetime.now().timestamp()}",
            "symbol": symbol,
            "action": action,
            "quantity": quantity,
            "price": price,
            "status": "filled"
        }
