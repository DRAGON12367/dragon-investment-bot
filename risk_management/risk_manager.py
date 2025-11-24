"""
Risk management system for live trading.
"""
import logging
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict

from utils.config import Config


class RiskManager:
    """Manages risk for live trading operations."""
    
    def __init__(self, config: Config):
        """Initialize risk manager."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.risk_manager")
        
        # Track daily performance
        self.daily_pnl = 0.0
        self.daily_start_value = 0.0
        self.last_reset_date = datetime.now().date()
        
        # Track positions
        self.open_positions: Dict[str, Dict[str, Any]] = {}
        self.daily_trades = defaultdict(int)
        
    def evaluate_signals(
        self, 
        signals: List[Dict[str, Any]], 
        portfolio_status: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Evaluate and filter trading signals based on risk parameters.
        
        Args:
            signals: List of trading signals
            portfolio_status: Current portfolio status
            
        Returns:
            List of approved signals with position sizing
        """
        self._reset_daily_tracking()
        
        approved_signals = []
        total_value = portfolio_status.get("total_value", 0)
        cash_available = portfolio_status.get("cash", 0)
        current_positions = portfolio_status.get("positions", [])
        
        # Update daily start value if needed
        if self.daily_start_value == 0:
            self.daily_start_value = total_value
        
        # Check daily loss limit
        current_pnl = total_value - self.daily_start_value
        daily_loss_pct = current_pnl / self.daily_start_value if self.daily_start_value > 0 else 0
        
        if daily_loss_pct <= -self.config.max_daily_loss:
            self.logger.warning(
                f"Daily loss limit reached: {daily_loss_pct:.2%}. "
                "Stopping trading for today."
            )
            return []
        
        for signal in signals:
            try:
                approved_signal = self._evaluate_single_signal(
                    signal, 
                    total_value, 
                    cash_available,
                    current_positions
                )
                
                if approved_signal:
                    approved_signals.append(approved_signal)
                    
            except Exception as e:
                self.logger.error(f"Error evaluating signal: {e}", exc_info=True)
        
        return approved_signals
    
    def _evaluate_single_signal(
        self,
        signal: Dict[str, Any],
        total_value: float,
        cash_available: float,
        current_positions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Evaluate a single trading signal."""
        symbol = signal.get("symbol")
        action = signal.get("action")
        price = signal.get("price", 0)
        confidence = signal.get("confidence", 0)
        
        if not symbol or not action or price <= 0:
            return None
        
        # Check confidence threshold
        if confidence < self.config.min_confidence_threshold:
            self.logger.debug(f"Signal rejected: confidence {confidence:.2f} below threshold")
            return None
        
        # Check if we already have a position
        existing_position = self._get_position(symbol, current_positions)
        
        if action == "BUY":
            # Check if we already have a long position
            if existing_position and existing_position.get("quantity", 0) > 0:
                self.logger.debug(f"Already have position in {symbol}, skipping buy")
                return None
            
            # Calculate position size
            # Aggressive growth signals can use larger position sizes
            source = signal.get('source', '')
            if source == 'AGGRESSIVE_GROWTH' and 'position_size_pct' in signal:
                # Use the aggressive growth strategy's calculated position size
                position_size_pct = signal['position_size_pct']
                position_value = total_value * position_size_pct
            else:
                # Standard position sizing
                position_value = total_value * self.config.max_position_size
            
            quantity = int(position_value / price)
            
            # Check if we have enough cash
            required_cash = quantity * price
            if required_cash > cash_available:
                quantity = int(cash_available / price)
                if quantity == 0:
                    self.logger.debug(f"Insufficient cash for {symbol}")
                    return None
            
            # Check daily trade limit per symbol
            if self.daily_trades[symbol] >= 5:  # Max 5 trades per symbol per day
                self.logger.debug(f"Daily trade limit reached for {symbol}")
                return None
            
        elif action == "SELL":
            # Check if we have a position to sell
            if not existing_position or existing_position.get("quantity", 0) <= 0:
                self.logger.debug(f"No position to sell for {symbol}")
                return None
            
            quantity = existing_position.get("quantity", 0)
        
        else:
            return None
        
        # Add position sizing and risk parameters
        signal["quantity"] = quantity
        signal["order_type"] = "market"  # Can be changed to "limit" if needed
        signal["stop_loss"] = signal.get("stop_loss", price * (1 - self.config.stop_loss_percentage))
        signal["take_profit"] = signal.get("take_profit", price * (1 + self.config.take_profit_percentage))
        
        self.daily_trades[symbol] += 1
        
        return signal
    
    def _get_position(
        self, 
        symbol: str, 
        current_positions: List[Dict[str, Any]]
    ) -> Optional[Dict[str, Any]]:
        """Get current position for a symbol."""
        for position in current_positions:
            if position.get("symbol") == symbol:
                return position
        return None
    
    def _reset_daily_tracking(self):
        """Reset daily tracking if it's a new day."""
        today = datetime.now().date()
        if today > self.last_reset_date:
            self.logger.info("Resetting daily tracking for new day")
            self.daily_pnl = 0.0
            self.daily_start_value = 0.0
            self.daily_trades.clear()
            self.last_reset_date = today
    
    def update_position(self, symbol: str, position_data: Dict[str, Any]):
        """Update tracked position."""
        self.open_positions[symbol] = position_data
    
    def remove_position(self, symbol: str):
        """Remove position from tracking."""
        if symbol in self.open_positions:
            del self.open_positions[symbol]

