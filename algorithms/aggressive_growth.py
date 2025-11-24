"""
Aggressive Growth Strategy - Turn $100 into $100,000 through compound growth and high-conviction trading.
WARNING: This is a high-risk strategy. Only use with money you can afford to lose.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict

from algorithms.technical_indicators import TechnicalIndicators
from algorithms.advanced_indicators import AdvancedIndicators
from algorithms.professional_analysis import ProfessionalAnalysis
from utils.config import Config

# 5x UPGRADE - Ultra Advanced Features
try:
    from algorithms.ultra_advanced_indicators import UltraAdvancedIndicators
    from algorithms.ultra_advanced_strategies import UltraAdvancedStrategies
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False


class AggressiveGrowthStrategy:
    """
    Ultra-aggressive growth strategy designed for maximum returns.
    
    Key Techniques:
    1. Compound Growth - Reinvest all profits immediately
    2. High-Conviction Position Sizing - Up to 50% of portfolio on highest confidence trades
    3. Momentum Breakout Detection - Find assets with explosive growth potential
    4. Multi-Timeframe Analysis - Confirm signals across multiple timeframes
    5. Trend Riding - Let winners run, cut losers quickly
    6. Volatility Exploitation - Target high-volatility assets with strong momentum
    """
    
    def __init__(self, config: Config):
        """Initialize aggressive growth strategy."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.aggressive_growth")
        self.technical_indicators = TechnicalIndicators()
        self.advanced_indicators = AdvancedIndicators()
        self.professional_analysis = ProfessionalAnalysis()
        
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
        
        # Growth tracking
        self.initial_capital = 100.0  # Starting with $100
        self.target_capital = 100000.0  # Target: $100,000
        self.compound_multiplier = self.target_capital / self.initial_capital  # 1000x
        
        # Strategy parameters
        self.max_position_size = 0.50  # Up to 50% on highest conviction trades
        self.min_conviction_threshold = 0.85  # Very high confidence required
        self.momentum_threshold = 0.15  # 15%+ momentum required
        self.volatility_min = 0.05  # Minimum 5% daily volatility
        
        # Performance tracking
        self.trade_history: List[Dict[str, Any]] = []
        self.compound_growth_rate = 0.0
        
    def analyze_growth_opportunities(
        self,
        market_data: Dict[str, Any],
        portfolio_value: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Dict[str, Any]]:
        """
        Find the highest-growth potential opportunities.
        
        Args:
            market_data: Current market data
            portfolio_value: Current total portfolio value
            historical_data: Historical price data for analysis
            
        Returns:
            Ranked list of growth opportunities with conviction scores
        """
        opportunities = []
        
        for symbol, data in market_data.items():
            try:
                opportunity = self._analyze_growth_potential(
                    symbol, 
                    data, 
                    portfolio_value,
                    historical_data.get(symbol) if historical_data else None
                )
                if opportunity and opportunity['conviction_score'] >= self.min_conviction_threshold:
                    opportunities.append(opportunity)
            except Exception as e:
                self.logger.debug(f"Error analyzing {symbol}: {e}")
        
        # Rank by conviction score and growth potential
        opportunities.sort(
            key=lambda x: (x['conviction_score'], x['growth_potential']), 
            reverse=True
        )
        
        return opportunities
    
    def _analyze_growth_potential(
        self,
        symbol: str,
        data: Dict[str, Any],
        portfolio_value: float,
        historical: Optional[pd.DataFrame] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze a single asset for aggressive growth potential."""
        current_price = data.get('price', data.get('close', 0))
        if current_price <= 0:
            return None
        
        # Extract key metrics
        change_24h = data.get('change_percent', 0)
        volume = data.get('volume', 0)
        high_24h = data.get('high', current_price)
        low_24h = data.get('low', current_price)
        
        # Calculate volatility
        volatility = abs(high_24h - low_24h) / current_price if current_price > 0 else 0
        
        # Skip if volatility too low (need volatility for explosive moves)
        if volatility < self.volatility_min:
            return None
        
        # Momentum analysis (critical for growth)
        momentum_score = self._calculate_momentum_score(
            current_price, 
            change_24h, 
            volatility,
            historical
        )
        
        # Breakout detection
        breakout_score = self._detect_breakout(
            current_price,
            high_24h,
            low_24h,
            historical
        )
        
        # Volume analysis (need strong volume for breakouts)
        volume_score = self._analyze_volume(volume, historical)
        
        # Trend strength
        trend_score = 0.0
        if historical is not None and not historical.empty:
            trend_metrics = self.professional_analysis.calculate_trend_strength(historical)
            trend_score = trend_metrics.get('trend_strength', 0.0)
        
        # Support/Resistance analysis
        support_resistance_score = 0.0
        if historical is not None and not historical.empty:
            sr_levels = self.professional_analysis.detect_support_resistance(historical)
            # Check if price is breaking above resistance (bullish)
            resistance_levels = sr_levels.get('resistance', [])
            if resistance_levels:
                nearest_resistance = min([r for r in resistance_levels if r > current_price], default=None)
                if nearest_resistance and current_price >= nearest_resistance * 0.98:
                    support_resistance_score = 0.9  # Breaking resistance
        
        # Calculate growth potential (how much can this move?)
        growth_potential = self._estimate_growth_potential(
            momentum_score,
            volatility,
            trend_score,
            historical
        )
        
        # Overall conviction score (weighted combination)
        conviction_score = (
            momentum_score * 0.35 +
            breakout_score * 0.25 +
            volume_score * 0.15 +
            trend_score * 0.15 +
            support_resistance_score * 0.10
        )
        
        # Calculate position size based on conviction
        position_size_pct = self._calculate_position_size(conviction_score)
        position_value = portfolio_value * position_size_pct
        
        # Calculate expected return and risk
        expected_return = growth_potential
        risk_reward_ratio = expected_return / (self.config.stop_loss_percentage * 2)  # Aggressive stop loss
        
        # Only proceed if conviction is high enough
        if conviction_score < self.min_conviction_threshold:
            return None
        
        # Calculate target price (aggressive targets)
        target_price = current_price * (1 + growth_potential)
        stop_loss_price = current_price * (1 - self.config.stop_loss_percentage * 1.5)  # Tighter stop
        
        return {
            'symbol': symbol,
            'asset_type': data.get('asset_type', 'unknown'),
            'current_price': current_price,
            'action': 'STRONG_BUY',
            'conviction_score': conviction_score,
            'growth_potential': growth_potential,
            'momentum_score': momentum_score,
            'breakout_score': breakout_score,
            'volume_score': volume_score,
            'trend_score': trend_score,
            'volatility': volatility,
            'change_24h': change_24h,
            'position_size_pct': position_size_pct,
            'position_value': position_value,
            'target_price': target_price,
            'stop_loss': stop_loss_price,
            'expected_return': expected_return,
            'risk_reward_ratio': risk_reward_ratio,
            'timestamp': datetime.now().isoformat(),
            'strategy': 'AGGRESSIVE_GROWTH'
        }
    
    def _calculate_momentum_score(
        self,
        current_price: float,
        change_24h: float,
        volatility: float,
        historical: Optional[pd.DataFrame] = None
    ) -> float:
        """Calculate momentum score (0-1)."""
        score = 0.0
        
        # 24h momentum (must be positive and strong)
        if change_24h > self.momentum_threshold * 100:  # 15%+ move
            score += 0.4
        elif change_24h > 0.10 * 100:  # 10%+ move
            score += 0.3
        elif change_24h > 0.05 * 100:  # 5%+ move
            score += 0.2
        
        # Historical momentum (if available)
        if historical is not None and len(historical) >= 20:
            # 7-day momentum
            if len(historical) >= 7:
                price_7d_ago = historical['close'].iloc[-7]
                momentum_7d = (current_price - price_7d_ago) / price_7d_ago
                if momentum_7d > 0.20:  # 20%+ in 7 days
                    score += 0.3
                elif momentum_7d > 0.10:
                    score += 0.2
            
            # Acceleration (rate of change increasing)
            if len(historical) >= 14:
                recent_change = (current_price - historical['close'].iloc[-7]) / historical['close'].iloc[-7]
                earlier_change = (historical['close'].iloc[-7] - historical['close'].iloc[-14]) / historical['close'].iloc[-14]
                if recent_change > earlier_change * 1.5:  # Accelerating
                    score += 0.3
        
        return min(score, 1.0)
    
    def _detect_breakout(
        self,
        current_price: float,
        high_24h: float,
        low_24h: float,
        historical: Optional[pd.DataFrame] = None
    ) -> float:
        """Detect breakout patterns (price breaking key levels)."""
        score = 0.0
        
        if historical is None or historical.empty:
            return 0.0
        
        if len(historical) < 20:
            return 0.0
        
        # Check if breaking above recent highs
        recent_high = historical['high'].tail(20).max()
        if current_price >= recent_high * 0.98:  # Near or above recent high
            score += 0.5
        
        # Check if breaking above longer-term resistance
        if len(historical) >= 50:
            long_high = historical['high'].tail(50).max()
            if current_price >= long_high * 0.98:
                score += 0.3
        
        # Check price position in range (want to be near top)
        price_range = high_24h - low_24h
        if price_range > 0:
            position_in_range = (current_price - low_24h) / price_range
            if position_in_range > 0.85:  # Top 15% of range
                score += 0.2
        
        return min(score, 1.0)
    
    def _analyze_volume(
        self,
        current_volume: float,
        historical: Optional[pd.DataFrame] = None
    ) -> float:
        """Analyze volume for breakout confirmation."""
        if historical is None or historical.empty or len(historical) < 20:
            return 0.5  # Neutral if no data
        
        # Compare to average volume
        avg_volume = historical['volume'].tail(20).mean()
        if avg_volume > 0:
            volume_ratio = current_volume / avg_volume
            if volume_ratio > 2.0:  # 2x average volume (strong)
                return 1.0
            elif volume_ratio > 1.5:  # 1.5x average
                return 0.8
            elif volume_ratio > 1.2:
                return 0.6
            else:
                return 0.4
        
        return 0.5
    
    def _estimate_growth_potential(
        self,
        momentum_score: float,
        volatility: float,
        trend_score: float,
        historical: Optional[pd.DataFrame] = None
    ) -> float:
        """Estimate potential growth percentage (0-1 = 0-100%)."""
        # Base potential from momentum
        base_potential = momentum_score * 0.30  # Up to 30% from momentum
        
        # Volatility adds opportunity (more volatility = more potential)
        volatility_bonus = min(volatility * 2, 0.40)  # Up to 40% from volatility
        
        # Trend strength bonus
        trend_bonus = trend_score * 0.20  # Up to 20% from trend
        
        # Historical breakout potential
        historical_bonus = 0.0
        if historical is not None and len(historical) >= 50:
            # Check for similar breakouts in past
            recent_high = historical['high'].tail(20).max()
            longer_high = historical['high'].tail(50).max()
            if recent_high >= longer_high * 0.95:  # Near all-time high
                historical_bonus = 0.30  # Could be explosive
        
        total_potential = min(
            base_potential + volatility_bonus + trend_bonus + historical_bonus,
            1.0  # Cap at 100% (2x)
        )
        
        return total_potential
    
    def _calculate_position_size(self, conviction_score: float) -> float:
        """Calculate position size based on conviction (0-50% of portfolio)."""
        # Linear scaling from 20% to 50% based on conviction
        min_size = 0.20  # Minimum 20% for high-conviction trades
        max_size = self.max_position_size  # Maximum 50%
        
        # Scale based on conviction above threshold
        if conviction_score >= 0.95:
            return max_size
        elif conviction_score >= 0.90:
            return min_size + (max_size - min_size) * 0.75
        elif conviction_score >= 0.85:
            return min_size + (max_size - min_size) * 0.50
        else:
            return min_size
    
    def calculate_compound_growth_plan(
        self,
        current_value: float,
        target_value: float
    ) -> Dict[str, Any]:
        """
        Calculate compound growth plan to reach target.
        
        Returns:
            Dictionary with growth plan details
        """
        if current_value <= 0:
            return {}
        
        multiplier_needed = target_value / current_value
        
        # Calculate required growth rate for different timeframes
        # Using compound interest formula: FV = PV * (1 + r)^n
        # Solving for r: r = (FV/PV)^(1/n) - 1
        
        plans = {}
        for years in [1, 2, 3, 5]:
            periods = years * 252  # Trading days per year
            daily_rate = (multiplier_needed ** (1 / periods)) - 1
            annual_rate = ((1 + daily_rate) ** 252) - 1
            
            plans[f"{years}_year"] = {
                'years': years,
                'daily_return_needed': daily_rate,
                'annual_return_needed': annual_rate,
                'total_multiplier': multiplier_needed
            }
        
        # Calculate current progress
        progress_pct = (current_value / target_value) * 100
        
        return {
            'current_value': current_value,
            'target_value': target_value,
            'multiplier_needed': multiplier_needed,
            'progress_pct': progress_pct,
            'growth_plans': plans
        }
    
    def optimize_reinvestment(
        self,
        portfolio_value: float,
        recent_profits: float,
        opportunities: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Optimize how to reinvest profits for maximum compound growth.
        
        Args:
            portfolio_value: Current total portfolio value
            recent_profits: Recent profits to reinvest
            opportunities: Available growth opportunities
            
        Returns:
            Reinvestment strategy
        """
        if not opportunities or recent_profits <= 0:
            return {'action': 'HOLD', 'reason': 'No opportunities or profits'}
        
        # Find best opportunity
        best_opportunity = opportunities[0]
        
        # Calculate how much to reinvest
        # Aggressive: Reinvest 100% of profits + use available cash
        reinvestment_amount = min(recent_profits, portfolio_value * 0.5)
        
        return {
            'action': 'REINVEST',
            'amount': reinvestment_amount,
            'target_symbol': best_opportunity['symbol'],
            'target_conviction': best_opportunity['conviction_score'],
            'expected_return': best_opportunity['expected_return'],
            'reason': f"Reinvesting profits into highest conviction opportunity: {best_opportunity['symbol']}"
        }
    
    def track_performance(self, trade_result: Dict[str, Any]):
        """Track trade performance for learning and optimization."""
        self.trade_history.append({
            **trade_result,
            'timestamp': datetime.now().isoformat()
        })
        
        # Keep only last 100 trades
        if len(self.trade_history) > 100:
            self.trade_history = self.trade_history[-100:]
        
        # Calculate compound growth rate
        if len(self.trade_history) >= 10:
            recent_trades = self.trade_history[-10:]
            total_return = 1.0
            for trade in recent_trades:
                if 'return_pct' in trade:
                    total_return *= (1 + trade['return_pct'] / 100)
            
            self.compound_growth_rate = (total_return ** (1 / len(recent_trades)) - 1) * 100
    
    def get_performance_summary(self) -> Dict[str, Any]:
        """Get performance summary for the aggressive growth strategy."""
        if not self.trade_history:
            return {'status': 'No trades yet'}
        
        winning_trades = [t for t in self.trade_history if t.get('return_pct', 0) > 0]
        losing_trades = [t for t in self.trade_history if t.get('return_pct', 0) <= 0]
        
        avg_win = np.mean([t['return_pct'] for t in winning_trades]) if winning_trades else 0
        avg_loss = np.mean([t['return_pct'] for t in losing_trades]) if losing_trades else 0
        
        win_rate = len(winning_trades) / len(self.trade_history) if self.trade_history else 0
        
        return {
            'total_trades': len(self.trade_history),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': win_rate,
            'avg_win_pct': avg_win,
            'avg_loss_pct': avg_loss,
            'compound_growth_rate': self.compound_growth_rate,
            'profit_factor': abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades and avg_loss != 0 else float('inf')
        }

