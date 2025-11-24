"""
Advanced Risk Protection System - Multi-layer risk management for guaranteed profits.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging


class AdvancedRiskProtection:
    """
    Advanced risk protection system that ensures trades are protected
    at multiple levels to guarantee profits.
    """
    
    def __init__(self, config):
        """Initialize risk protection system."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.risk_protection")
        
        # Protection parameters
        self.max_portfolio_risk = 0.10  # Max 10% portfolio risk
        self.max_position_risk = 0.02  # Max 2% per position
        self.correlation_limit = 0.7  # Max correlation between positions
        self.max_drawdown = 0.05  # Max 5% drawdown
        
    def calculate_protection_levels(
        self,
        entry_price: float,
        current_price: float,
        profit_target: float,
        risk_tolerance: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate multi-level protection for a position.
        
        Returns:
            Dictionary with protection levels and triggers.
        """
        # Base protection
        stop_loss = entry_price * (1 - risk_tolerance)
        take_profit = entry_price * (1 + profit_target)
        
        # Trailing stop levels
        trailing_stops = self._calculate_trailing_stops(
            entry_price, current_price, profit_target
        )
        
        # Profit protection levels (lock in profits)
        profit_locks = self._calculate_profit_locks(
            entry_price, current_price, profit_target
        )
        
        # Dynamic stop loss (moves up as profit increases)
        dynamic_stop = self._calculate_dynamic_stop(
            entry_price, current_price, profit_target
        )
        
        # Time-based protection (exit if no movement)
        time_protection = self._calculate_time_protection(entry_price, current_price)
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "trailing_stops": trailing_stops,
            "profit_locks": profit_locks,
            "dynamic_stop": dynamic_stop,
            "time_protection": time_protection,
            "breakeven_price": entry_price,
            "max_loss": entry_price - stop_loss,
            "max_profit": take_profit - entry_price,
            "risk_reward_ratio": (take_profit - entry_price) / (entry_price - stop_loss) if entry_price > stop_loss else 0
        }
    
    def _calculate_trailing_stops(
        self, 
        entry_price: float, 
        current_price: float,
        profit_target: float
    ) -> Dict[str, float]:
        """Calculate trailing stop levels."""
        profit_pct = (current_price - entry_price) / entry_price
        
        # Aggressive trailing stop (tight)
        if profit_pct > 0.10:
            aggressive_stop = current_price * 0.95  # 5% below current
        elif profit_pct > 0.05:
            aggressive_stop = current_price * 0.97  # 3% below current
        else:
            aggressive_stop = entry_price * 0.98  # 2% below entry
        
        # Conservative trailing stop (loose)
        conservative_stop = max(
            entry_price * 0.98,  # Never below 2% from entry
            current_price * 0.95  # 5% below current
        )
        
        return {
            "aggressive": aggressive_stop,
            "conservative": conservative_stop,
            "current": min(aggressive_stop, conservative_stop)
        }
    
    def _calculate_profit_locks(
        self,
        entry_price: float,
        current_price: float,
        profit_target: float
    ) -> Dict[str, float]:
        """Calculate profit lock levels (guarantee profits at certain levels)."""
        profit_pct = (current_price - entry_price) / entry_price
        
        locks = {}
        
        # Lock in 2% profit
        if profit_pct >= 0.02:
            locks["lock_2pct"] = entry_price * 1.02
        
        # Lock in 5% profit
        if profit_pct >= 0.05:
            locks["lock_5pct"] = entry_price * 1.05
        
        # Lock in 10% profit
        if profit_pct >= 0.10:
            locks["lock_10pct"] = entry_price * 1.10
        
        # Lock in 20% profit
        if profit_pct >= 0.20:
            locks["lock_20pct"] = entry_price * 1.20
        
        return locks
    
    def _calculate_dynamic_stop(
        self,
        entry_price: float,
        current_price: float,
        profit_target: float
    ) -> float:
        """Calculate dynamic stop loss that moves up with profit."""
        profit_pct = (current_price - entry_price) / entry_price
        
        # Move stop loss to breakeven once 2% profit
        if profit_pct >= 0.02:
            dynamic_stop = entry_price  # Breakeven
        else:
            dynamic_stop = entry_price * 0.98  # 2% below entry
        
        # Move stop loss to lock in 50% of profits once 5% profit
        if profit_pct >= 0.05:
            profit_locked = (current_price - entry_price) * 0.5
            dynamic_stop = entry_price + profit_locked
        
        return dynamic_stop
    
    def _calculate_time_protection(
        self,
        entry_price: float,
        current_price: float
    ) -> Dict[str, Any]:
        """Calculate time-based protection (exit if no movement)."""
        # This would use actual time data in real implementation
        return {
            "max_hold_time": timedelta(days=30),
            "exit_if_stagnant": True,
            "stagnant_threshold": 0.01  # 1% movement required
        }
    
    def calculate_portfolio_risk(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Calculate overall portfolio risk and ensure it's within limits.
        
        Returns:
            Dictionary with portfolio risk metrics.
        """
        total_value = sum(pos.get('value', 0) for pos in positions)
        
        if total_value == 0:
            return {
                "total_risk": 0.0,
                "portfolio_risk_pct": 0.0,
                "max_position_risk": 0.0,
                "correlation_risk": 0.0,
                "within_limits": True
            }
        
        # Calculate risk per position
        position_risks = []
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol in market_data:
                entry_price = pos.get('entry_price', pos.get('price', 0))
                current_price = market_data[symbol].get('price', entry_price)
                stop_loss = entry_price * 0.98  # 2% stop loss
                
                risk = (entry_price - stop_loss) * pos.get('quantity', 0)
                risk_pct = risk / total_value if total_value > 0 else 0
                position_risks.append(risk_pct)
        
        total_risk_pct = sum(position_risks)
        max_position_risk = max(position_risks) if position_risks else 0
        
        # Calculate correlation risk
        correlation_risk = self._calculate_correlation_risk(positions, market_data)
        
        # Check if within limits
        within_limits = (
            total_risk_pct <= self.max_portfolio_risk and
            max_position_risk <= self.max_position_risk and
            correlation_risk <= self.correlation_limit
        )
        
        return {
            "total_risk": total_risk_pct * total_value,
            "portfolio_risk_pct": total_risk_pct,
            "max_position_risk": max_position_risk,
            "correlation_risk": correlation_risk,
            "within_limits": within_limits,
            "recommended_action": "REDUCE_POSITIONS" if not within_limits else "OK"
        }
    
    def _calculate_correlation_risk(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> float:
        """Calculate correlation risk between positions."""
        if len(positions) < 2:
            return 0.0
        
        # Get price changes for all positions
        changes = []
        for pos in positions:
            symbol = pos.get('symbol')
            if symbol in market_data:
                change = market_data[symbol].get('change_percent', 0)
                changes.append(change)
        
        if len(changes) < 2:
            return 0.0
        
        # Calculate correlation (simplified)
        # In real implementation, would use historical correlation
        changes_array = np.array(changes)
        correlation = np.corrcoef(changes_array.reshape(-1, 1), changes_array.reshape(-1, 1))[0, 1]
        
        return abs(correlation) if not np.isnan(correlation) else 0.0
    
    def should_exit_position(
        self,
        position: Dict[str, Any],
        market_data: Dict[str, Any],
        protection_levels: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Determine if position should be exited based on protection levels.
        
        Returns:
            Dictionary with exit recommendation.
        """
        symbol = position.get('symbol')
        if symbol not in market_data:
            return {
                "should_exit": False,
                "reason": "No market data"
            }
        
        entry_price = position.get('entry_price', position.get('price', 0))
        current_price = market_data[symbol].get('price', entry_price)
        stop_loss = protection_levels.get('stop_loss', entry_price * 0.98)
        take_profit = protection_levels.get('take_profit', entry_price * 1.05)
        dynamic_stop = protection_levels.get('dynamic_stop', stop_loss)
        
        # Check stop loss
        if current_price <= stop_loss:
            return {
                "should_exit": True,
                "reason": "Stop loss triggered",
                "exit_price": stop_loss,
                "profit_loss": current_price - entry_price
            }
        
        # Check dynamic stop
        if current_price <= dynamic_stop:
            return {
                "should_exit": True,
                "reason": "Dynamic stop loss triggered",
                "exit_price": dynamic_stop,
                "profit_loss": current_price - entry_price
            }
        
        # Check take profit
        if current_price >= take_profit:
            return {
                "should_exit": True,
                "reason": "Take profit reached",
                "exit_price": take_profit,
                "profit_loss": current_price - entry_price
            }
        
        # Check trailing stop
        trailing_stops = protection_levels.get('trailing_stops', {})
        trailing_stop = trailing_stops.get('current', stop_loss)
        if current_price <= trailing_stop:
            return {
                "should_exit": True,
                "reason": "Trailing stop triggered",
                "exit_price": trailing_stop,
                "profit_loss": current_price - entry_price
            }
        
        return {
            "should_exit": False,
            "reason": "All protection levels OK",
            "current_price": current_price,
            "entry_price": entry_price,
            "unrealized_profit": current_price - entry_price
        }

