"""
24/7 AI Profit Analyzer - Finds most profitable opportunities and optimal sell points.
"""
import logging
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import pandas as pd
import numpy as np

from utils.config import Config

# 5x UPGRADE - Ultra Advanced Features
try:
    from algorithms.ultra_advanced_indicators import UltraAdvancedIndicators
    from algorithms.ultra_advanced_strategies import UltraAdvancedStrategies
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False


class ProfitAnalyzer:
    """AI-powered profit analyzer for optimal buy/sell timing."""
    
    def __init__(self, config: Config):
        """Initialize profit analyzer."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.profit_analyzer")
        self.tracked_positions: Dict[str, Dict[str, Any]] = {}
        
        # 5x UPGRADE - Ultra Advanced Features
        if ULTRA_AVAILABLE:
            self.ultra_indicators = UltraAdvancedIndicators()
            self.ultra_strategies = UltraAdvancedStrategies()
        else:
            self.ultra_indicators = None
            self.ultra_strategies = None
        
        # Ultra Enhanced ML
        try:
            from algorithms.ultra_enhanced_ml import UltraEnhancedML
            self.ultra_enhanced_ml = UltraEnhancedML(config)
        except ImportError:
            self.ultra_enhanced_ml = None
        
        # Quantum ML
        try:
            from algorithms.quantum_ml_models import QuantumMLModels
            self.quantum_ml = QuantumMLModels(config)
        except ImportError:
            self.quantum_ml = None
        
        # Neural Evolution
        try:
            from algorithms.neural_evolution_models import NeuralEvolutionModels
            self.neural_evolution = NeuralEvolutionModels(config)
        except ImportError:
            self.neural_evolution = None
        
        # Hyper ML
        try:
            from algorithms.hyper_ml_models import HyperMLModels
            self.hyper_ml = HyperMLModels(config)
        except ImportError:
            self.hyper_ml = None
        
    def analyze_opportunities(
        self,
        market_data: Dict[str, Any],
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Dict[str, Any]]:
        """
        Analyze market data to find most profitable opportunities.
        
        Args:
            market_data: Current market data for all assets
            historical_data: Optional historical data for analysis
            
        Returns:
            List of ranked opportunities with profit potential
        """
        opportunities = []
        
        for symbol, data in market_data.items():
            try:
                opportunity = self._analyze_asset(symbol, data, historical_data)
                if opportunity:
                    opportunities.append(opportunity)
            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol}: {e}")
        
        # Rank by profit potential
        opportunities.sort(key=lambda x: x.get('profit_score', 0), reverse=True)
        
        return opportunities
    
    def _analyze_asset(
        self,
        symbol: str,
        data: Dict[str, Any],
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single asset for profit potential."""
        current_price = data.get('price', 0)
        if current_price <= 0:
            return None
        
        # Calculate profit metrics
        change_24h = data.get('change_percent', 0)
        volume = data.get('volume', 0)
        high_24h = data.get('high', current_price)
        low_24h = data.get('low', current_price)
        
        # Price momentum score
        price_position = (current_price - low_24h) / (high_24h - low_24h) if high_24h != low_24h else 0.5
        momentum_score = change_24h / 100 if change_24h > 0 else 0
        
        # Volume analysis
        volume_score = min(volume / 1e9, 1.0) if volume > 0 else 0  # Normalize to 1B
        
        # Volatility (opportunity for profit)
        volatility = abs(high_24h - low_24h) / current_price if current_price > 0 else 0
        volatility_score = min(volatility * 10, 1.0)  # Higher volatility = more opportunity
        
        # Calculate profit potential
        profit_potential = self._calculate_profit_potential(
            current_price,
            change_24h,
            volatility,
            historical_data.get(symbol) if historical_data else None
        )
        
        # Overall profit score
        profit_score = (
            momentum_score * 0.3 +
            volume_score * 0.2 +
            volatility_score * 0.2 +
            profit_potential * 0.3
        )
        
        # Determine action - Lower thresholds to show more opportunities, especially for crypto
        asset_type = data.get('asset_type', 'unknown')
        
        # For crypto, use lower thresholds to show more options
        if asset_type == 'crypto':
            if profit_score > 0.4:  # Lowered from 0.5 to show more STRONG_BUY
                action = 'STRONG_BUY'
            elif profit_score > 0.15:  # Lowered from 0.25 to show more BUY options
                action = 'BUY'
            elif profit_score > 0.05:  # Lowered from 0.1 to show even more options
                action = 'BUY'  # Still show as BUY for more options
            else:
                action = 'HOLD'
        else:  # Stocks
            if profit_score > 0.5:  # Lowered from 0.6
                action = 'STRONG_BUY'
            elif profit_score > 0.3:  # Lowered from 0.4
                action = 'BUY'
            elif profit_score > 0.15:  # Lowered from 0.2
                action = 'HOLD'
            else:
                action = 'AVOID'
        
        return {
            'symbol': symbol,
            'asset_type': asset_type,
            'current_price': current_price,
            'action': action,
            'profit_score': profit_score,
            'profit_potential': profit_potential,
            'momentum_score': momentum_score,
            'volume_score': volume_score,
            'volatility_score': volatility_score,
            'change_24h': change_24h,
            'recommended_entry': current_price,
            'target_price': current_price * (1 + profit_potential),
            'stop_loss': current_price * (1 - self.config.stop_loss_percentage),
            'risk_reward_ratio': profit_potential / self.config.stop_loss_percentage,
            'timestamp': datetime.now().isoformat()
        }
    
    def _calculate_profit_potential(
        self,
        current_price: float,
        change_24h: float,
        volatility: float,
        historical: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate potential profit percentage."""
        # Base potential from momentum
        base_potential = max(change_24h / 100, 0) * 2  # Amplify positive momentum
        
        # Volatility adds opportunity
        volatility_bonus = volatility * 0.5
        
        # Historical analysis if available
        historical_bonus = 0
        if historical is not None and not historical.empty:
            # Check if price is near support (good entry)
            recent_low = historical['low'].tail(20).min()
            if current_price <= recent_low * 1.05:  # Within 5% of recent low
                historical_bonus = 0.15
        
        total_potential = min(base_potential + volatility_bonus + historical_bonus, 0.5)  # Cap at 50%
        return total_potential
    
    def track_position(
        self,
        symbol: str,
        entry_price: float,
        quantity: float,
        asset_type: str
    ):
        """Start tracking a position for sell signals."""
        self.tracked_positions[symbol] = {
            'symbol': symbol,
            'entry_price': entry_price,
            'quantity': quantity,
            'asset_type': asset_type,
            'entry_time': datetime.now(),
            'highest_price': entry_price,
            'current_price': entry_price,
            'unrealized_pnl': 0.0,
            'unrealized_pnl_percent': 0.0
        }
        self.logger.info(f"Started tracking position: {symbol} @ ${entry_price:.2f}")
    
    def analyze_sell_signals(
        self,
        market_data: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Analyze when to sell tracked positions.
        
        Returns:
            List of sell signals with exact timing
        """
        sell_signals = []
        
        for symbol, position in self.tracked_positions.items():
            if symbol not in market_data:
                continue
            
            current_data = market_data[symbol]
            current_price = current_data.get('price', 0)
            
            if current_price <= 0:
                continue
            
            # Update position
            position['current_price'] = current_price
            position['unrealized_pnl'] = (current_price - position['entry_price']) * position['quantity']
            position['unrealized_pnl_percent'] = ((current_price - position['entry_price']) / position['entry_price']) * 100
            
            # Update highest price
            if current_price > position['highest_price']:
                position['highest_price'] = current_price
            
            # Analyze sell conditions
            sell_signal = self._should_sell(position, current_data)
            
            if sell_signal:
                sell_signal['symbol'] = symbol
                sell_signal['current_price'] = current_price
                sell_signal['entry_price'] = position['entry_price']
                sell_signal['profit'] = position['unrealized_pnl']
                sell_signal['profit_percent'] = position['unrealized_pnl_percent']
                sell_signal['hold_duration'] = str(datetime.now() - position['entry_time'])
                sell_signals.append(sell_signal)
        
        return sell_signals
    
    def _should_sell(
        self,
        position: Dict[str, Any],
        current_data: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """Determine if position should be sold."""
        current_price = position['current_price']
        entry_price = position['entry_price']
        highest_price = position['highest_price']
        profit_percent = position['unrealized_pnl_percent']
        
        # Take profit conditions
        if profit_percent >= self.config.take_profit_percentage * 100:
            return {
                'action': 'SELL',
                'reason': 'TAKE_PROFIT',
                'priority': 'HIGH',
                'message': f"Take profit target reached: {profit_percent:.2f}% gain"
            }
        
        # Trailing stop loss (protect profits)
        if profit_percent > 5:  # If up more than 5%
            trailing_stop = highest_price * 0.95  # 5% below highest
            if current_price < trailing_stop:
                return {
                    'action': 'SELL',
                    'reason': 'TRAILING_STOP',
                    'priority': 'HIGH',
                    'message': f"Trailing stop triggered: {profit_percent:.2f}% profit protected"
                }
        
        # Stop loss
        if profit_percent <= -self.config.stop_loss_percentage * 100:
            return {
                'action': 'SELL',
                'reason': 'STOP_LOSS',
                'priority': 'CRITICAL',
                'message': f"Stop loss triggered: {profit_percent:.2f}% loss"
            }
        
        # Technical sell signals
        change_24h = current_data.get('change_percent', 0)
        if change_24h < -10 and profit_percent > 0:  # Sharp drop but still profitable
            return {
                'action': 'SELL',
                'reason': 'MOMENTUM_REVERSAL',
                'priority': 'MEDIUM',
                'message': f"Momentum reversal detected: {change_24h:.2f}% drop, secure {profit_percent:.2f}% profit"
            }
        
        # Time-based exit (if holding too long without progress)
        hold_duration = datetime.now() - position['entry_time']
        if hold_duration.days > 30 and profit_percent < 2:
            return {
                'action': 'SELL',
                'reason': 'TIME_EXIT',
                'priority': 'LOW',
                'message': f"Position held {hold_duration.days} days with minimal gain: {profit_percent:.2f}%"
            }
        
        return None
    
    def get_top_opportunities(
        self,
        opportunities: List[Dict[str, Any]],
        limit: int = 10
    ) -> List[Dict[str, Any]]:
        """Get top N most profitable opportunities, prioritizing crypto."""
        if not opportunities:
            return []
        
        # Sort by profit score (descending)
        sorted_opps = sorted(opportunities, key=lambda x: x.get('profit_score', 0), reverse=True)
        
        # Separate crypto and stock
        crypto_opps = [o for o in sorted_opps if o.get('asset_type') == 'crypto']
        stock_opps = [o for o in sorted_opps if o.get('asset_type') == 'stock']
        
        # If limit is large, prioritize showing more cryptos
        if limit >= 20:
            # For large limits, show more cryptos
            crypto_limit = min(len(crypto_opps), int(limit * 0.7))  # 70% crypto
            stock_limit = min(len(stock_opps), limit - crypto_limit)  # Remaining for stocks
            return crypto_opps[:crypto_limit] + stock_opps[:stock_limit]
        else:
            # For small limits, just return top N sorted by score
            return sorted_opps[:limit]
    
    def get_position_status(self) -> Dict[str, Any]:
        """Get status of all tracked positions."""
        total_value = sum(
            pos['current_price'] * pos['quantity'] 
            for pos in self.tracked_positions.values()
        )
        total_pnl = sum(pos['unrealized_pnl'] for pos in self.tracked_positions.values())
        
        return {
            'total_positions': len(self.tracked_positions),
            'total_value': total_value,
            'total_pnl': total_pnl,
            'positions': list(self.tracked_positions.values())
        }

