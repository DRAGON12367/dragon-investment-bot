"""
AI Investment Advisor - Provides live 24/7 investment guidance using all algorithms.
Designed for small accounts ($50+) to maximize growth potential.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta

from algorithms.profit_analyzer import ProfitAnalyzer
from algorithms.aggressive_growth import AggressiveGrowthStrategy
from algorithms.professional_analysis import ProfessionalAnalysis
from algorithms.technical_indicators import TechnicalIndicators
from utils.config import Config

# 5x UPGRADE - Ultra Advanced Features
try:
    from algorithms.ultra_advanced_indicators import UltraAdvancedIndicators
    from algorithms.ultra_advanced_ml_models import UltraAdvancedMLModels
    from algorithms.ultra_advanced_strategies import UltraAdvancedStrategies
    ULTRA_AVAILABLE = True
except ImportError:
    ULTRA_AVAILABLE = False


class InvestmentAdvisor:
    """
    Comprehensive investment advisor that combines all strategies to provide
    actionable guidance for small accounts ($50+).
    
    Provides:
    - Best investment opportunities ranked by potential
    - Risk-adjusted recommendations
    - Position sizing guidance
    - Entry/exit timing
    - Portfolio allocation advice
    """
    
    def __init__(self, config: Config):
        """Initialize investment advisor."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.investment_advisor")
        self.profit_analyzer = ProfitAnalyzer(config)
        self.aggressive_growth = AggressiveGrowthStrategy(config)
        self.professional_analysis = ProfessionalAnalysis()
        self.technical_indicators = TechnicalIndicators()
        
        # 5x UPGRADE - Ultra Advanced Features
        if ULTRA_AVAILABLE:
            self.ultra_indicators = UltraAdvancedIndicators()
            self.ultra_ml = UltraAdvancedMLModels(config)
            self.ultra_strategies = UltraAdvancedStrategies()
        else:
            self.ultra_indicators = None
            self.ultra_ml = None
            self.ultra_strategies = None
        
        # Hyper ML Models
        try:
            from algorithms.hyper_ml_models import HyperMLModels
            self.hyper_ml = HyperMLModels(config)
        except ImportError:
            self.hyper_ml = None
        
        # Ultra Enhanced ML
        try:
            from algorithms.ultra_enhanced_ml import UltraEnhancedML
            self.ultra_enhanced_ml = UltraEnhancedML(config)
        except ImportError:
            self.ultra_enhanced_ml = None
        
        # Small account strategy parameters
        self.min_account_size = 50.0  # Minimum $50
        self.max_positions_small = 3  # Max 3 positions for small accounts
        self.min_position_size = 0.20  # Minimum 20% per position
        self.max_position_size_small = 0.40  # Max 40% for small accounts
        
    def get_investment_guidance(
        self,
        account_balance: float,
        market_data: Dict[str, Any],
        current_positions: List[Dict[str, Any]] = None,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive investment guidance based on account size and market conditions.
        
        Args:
            account_balance: Current account balance
            market_data: Current market data
            current_positions: Current open positions
            historical_data: Historical data for analysis
            
        Returns:
            Comprehensive investment guidance dictionary
        """
        if account_balance < self.min_account_size:
            return {
                'status': 'insufficient_funds',
                'message': f'Minimum account size is ${self.min_account_size}. Current: ${account_balance:.2f}',
                'recommendation': 'Deposit more funds to start investing'
            }
        
        current_positions = current_positions or []
        
        # Analyze all opportunities using different strategies
        all_opportunities = self._analyze_all_opportunities(
            market_data,
            account_balance,
            historical_data
        )
        
        # Get best recommendations based on account size
        recommendations = self._generate_recommendations(
            all_opportunities,
            account_balance,
            current_positions
        )
        
        # Calculate portfolio allocation
        portfolio_allocation = self._calculate_portfolio_allocation(
            account_balance,
            current_positions,
            recommendations
        )
        
        # Get risk assessment
        risk_assessment = self._assess_risk(
            recommendations,
            account_balance,
            current_positions
        )
        
        # Generate action plan
        action_plan = self._generate_action_plan(
            recommendations,
            current_positions,
            account_balance
        )
        
        return {
            'timestamp': datetime.now().isoformat(),
            'account_balance': account_balance,
            'account_size_category': self._get_account_category(account_balance),
            'recommendations': recommendations,
            'portfolio_allocation': portfolio_allocation,
            'risk_assessment': risk_assessment,
            'action_plan': action_plan,
            'growth_projection': self._project_growth(account_balance, recommendations),
            'top_opportunities': all_opportunities[:10]  # Top 10 overall
        }
    
    def _analyze_all_opportunities(
        self,
        market_data: Dict[str, Any],
        account_balance: float,
        historical_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> List[Dict[str, Any]]:
        """Analyze opportunities using all available strategies."""
        all_opps = []
        
        # Check if market data is available
        if not market_data:
            self.logger.warning("No market data available for analysis")
            return all_opps
        
        # 1. Aggressive Growth Opportunities (highest potential)
        try:
            growth_opps = self.aggressive_growth.analyze_growth_opportunities(
                market_data,
                account_balance,
                historical_data
            )
        except Exception as e:
            self.logger.error(f"Error in aggressive growth analysis: {e}")
            growth_opps = []
        for opp in growth_opps:
            opp['strategy'] = 'AGGRESSIVE_GROWTH'
            opp['priority'] = 'HIGH'
            opp['risk_level'] = 'HIGH'
            all_opps.append(opp)
        
        # 2. Profit Analyzer Opportunities (balanced)
        try:
            profit_opps = self.profit_analyzer.analyze_opportunities(market_data, historical_data)
            for opp in profit_opps:
                if opp.get('profit_score', 0) > 0.5:  # Only high-scoring ones
                    opp['strategy'] = 'PROFIT_ANALYZER'
                    opp['priority'] = 'MEDIUM'
                    opp['risk_level'] = 'MEDIUM'
                    all_opps.append(opp)
        except Exception as e:
            self.logger.error(f"Error in profit analyzer: {e}")
        
        # 3. Rank and deduplicate
        # Sort by combined score
        for opp in all_opps:
            if 'conviction_score' in opp:
                opp['combined_score'] = opp['conviction_score']
            elif 'profit_score' in opp:
                opp['combined_score'] = opp['profit_score']
            else:
                opp['combined_score'] = 0.5
            
            # Adjust for account size (small accounts need higher returns)
            if account_balance < 500:
                opp['combined_score'] *= 1.2  # Boost score for small accounts
        
        # Remove duplicates (same symbol)
        seen_symbols = set()
        unique_opps = []
        for opp in sorted(all_opps, key=lambda x: x.get('combined_score', 0), reverse=True):
            symbol = opp.get('symbol')
            if symbol and symbol not in seen_symbols:
                seen_symbols.add(symbol)
                unique_opps.append(opp)
        
        return unique_opps
    
    def _generate_recommendations(
        self,
        opportunities: List[Dict[str, Any]],
        account_balance: float,
        current_positions: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Generate specific investment recommendations."""
        recommendations = []
        
        # Separate crypto and stock opportunities
        crypto_opportunities = [opp for opp in opportunities if opp.get('asset_type') == 'crypto']
        stock_opportunities = [opp for opp in opportunities if opp.get('asset_type') == 'stock']
        
        # Show many more crypto options - prioritize cryptos
        max_crypto_recommendations = 15  # Show up to 15 crypto options
        max_stock_recommendations = 5   # Show up to 5 stock options
        
        # Filter opportunities
        available_cash = account_balance - sum(
            pos.get('value', 0) for pos in current_positions
        )
        
        # Process crypto opportunities first (prioritize showing more cryptos)
        for opp in crypto_opportunities[:max_crypto_recommendations * 2]:  # Consider more, filter down
            if available_cash <= 0:
                break
            
            if len([r for r in recommendations if r.get('asset_type') == 'crypto']) >= max_crypto_recommendations:
                break
            
            # Check if we already have this position
            symbol = opp.get('symbol')
            existing_position = next(
                (p for p in current_positions if p.get('symbol') == symbol),
                None
            )
            
            if existing_position:
                continue  # Skip if we already own it
            
            # Calculate position size (smaller for more options)
            position_size_pct = self._calculate_optimal_position_size(
                opp,
                account_balance,
                len(current_positions) + len(recommendations)
            )
            
            # For showing many options, use smaller position sizes
            if len([r for r in recommendations if r.get('asset_type') == 'crypto']) >= 5:
                position_size_pct = min(position_size_pct, 0.10)  # Max 10% per position when showing many
            
            position_value = account_balance * position_size_pct
            
            if position_value > available_cash:
                position_value = available_cash * 0.9  # Use 90% of available
                position_size_pct = position_value / account_balance
            
            if position_value < 10:  # Minimum $10 position
                continue
            
            recommendation = {
                'symbol': symbol,
                'asset_type': opp.get('asset_type', 'unknown'),
                'action': 'BUY',
                'current_price': opp.get('current_price', 0),
                'target_price': opp.get('target_price', 0),
                'stop_loss': opp.get('stop_loss', 0),
                'position_size_pct': position_size_pct,
                'position_value': position_value,
                'quantity': int(position_value / opp.get('current_price', 1)),
                'expected_return_pct': opp.get('growth_potential', opp.get('profit_potential', 0)) * 100,
                'risk_level': opp.get('risk_level', 'MEDIUM'),
                'strategy': opp.get('strategy', 'UNKNOWN'),
                'confidence': opp.get('conviction_score', opp.get('profit_score', 0)),
                'timeframe': self._estimate_timeframe(opp),
                'reason': self._generate_reason(opp),
                'priority': opp.get('priority', 'MEDIUM')
            }
            
            recommendations.append(recommendation)
            available_cash -= position_value
        
        # Process stock opportunities (fewer, but still show some)
        for opp in stock_opportunities[:max_stock_recommendations * 2]:
            if available_cash <= 0:
                break
            
            if len([r for r in recommendations if r.get('asset_type') == 'stock']) >= max_stock_recommendations:
                break
            
            # Check if we already have this position
            symbol = opp.get('symbol')
            existing_position = next(
                (p for p in current_positions if p.get('symbol') == symbol),
                None
            )
            
            if existing_position:
                continue  # Skip if we already own it
            
            # Calculate position size
            position_size_pct = self._calculate_optimal_position_size(
                opp,
                account_balance,
                len(current_positions) + len(recommendations)
            )
            
            position_value = account_balance * position_size_pct
            
            if position_value > available_cash:
                position_value = available_cash * 0.9  # Use 90% of available
                position_size_pct = position_value / account_balance
            
            if position_value < 10:  # Minimum $10 position
                continue
            
            recommendation = {
                'symbol': symbol,
                'asset_type': opp.get('asset_type', 'unknown'),
                'action': 'BUY',
                'current_price': opp.get('current_price', 0),
                'target_price': opp.get('target_price', 0),
                'stop_loss': opp.get('stop_loss', 0),
                'position_size_pct': position_size_pct,
                'position_value': position_value,
                'quantity': int(position_value / opp.get('current_price', 1)),
                'expected_return_pct': opp.get('growth_potential', opp.get('profit_potential', 0)) * 100,
                'risk_level': opp.get('risk_level', 'MEDIUM'),
                'strategy': opp.get('strategy', 'UNKNOWN'),
                'confidence': opp.get('conviction_score', opp.get('profit_score', 0)),
                'timeframe': self._estimate_timeframe(opp),
                'reason': self._generate_reason(opp),
                'priority': opp.get('priority', 'MEDIUM')
            }
            
            recommendations.append(recommendation)
            available_cash -= position_value
        
        # Sort by confidence/score for final display
        recommendations.sort(key=lambda x: x.get('confidence', 0), reverse=True)
        
        return recommendations
    
    def _calculate_optimal_position_size(
        self,
        opportunity: Dict[str, Any],
        account_balance: float,
        current_positions_count: int
    ) -> float:
        """Calculate optimal position size based on opportunity and account size."""
        # Base position size
        if account_balance < 200:
            # Very small account - need to be more concentrated
            base_size = 0.35  # 35% per position
        elif account_balance < 500:
            base_size = 0.30  # 30% per position
        elif account_balance < 1000:
            base_size = 0.25  # 25% per position
        else:
            base_size = 0.20  # 20% per position
        
        # Adjust based on confidence/conviction
        confidence = opportunity.get('conviction_score', opportunity.get('profit_score', 0.5))
        if confidence > 0.9:
            base_size *= 1.2  # Increase for very high confidence
        elif confidence > 0.8:
            base_size *= 1.1
        
        # Adjust based on risk level
        risk_level = opportunity.get('risk_level', 'MEDIUM')
        if risk_level == 'HIGH':
            base_size *= 0.8  # Reduce for high risk
        elif risk_level == 'LOW':
            base_size *= 1.1
        
        # Cap at maximum
        return min(base_size, self.max_position_size_small)
    
    def _calculate_portfolio_allocation(
        self,
        account_balance: float,
        current_positions: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Calculate recommended portfolio allocation."""
        total_allocated = sum(pos.get('value', 0) for pos in current_positions)
        cash_available = account_balance - total_allocated
        
        # Calculate allocation for recommendations
        recommended_allocation = sum(rec.get('position_value', 0) for rec in recommendations)
        
        # Asset type breakdown
        asset_types = {}
        for pos in current_positions:
            asset_type = pos.get('asset_type', 'unknown')
            asset_types[asset_type] = asset_types.get(asset_type, 0) + pos.get('value', 0)
        
        for rec in recommendations:
            asset_type = rec.get('asset_type', 'unknown')
            asset_types[asset_type] = asset_types.get(asset_type, 0) + rec.get('position_value', 0)
        
        return {
            'total_value': account_balance,
            'cash_available': cash_available,
            'currently_allocated': total_allocated,
            'recommended_allocation': recommended_allocation,
            'cash_reserve_pct': (cash_available - recommended_allocation) / account_balance if account_balance > 0 else 0,
            'asset_type_breakdown': {
                k: {'value': v, 'percentage': (v / account_balance * 100) if account_balance > 0 else 0}
                for k, v in asset_types.items()
            },
            'diversification_score': self._calculate_diversification_score(current_positions, recommendations)
        }
    
    def _assess_risk(
        self,
        recommendations: List[Dict[str, Any]],
        account_balance: float,
        current_positions: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Assess overall portfolio risk."""
        high_risk_count = sum(1 for r in recommendations if r.get('risk_level') == 'HIGH')
        total_positions = len(current_positions) + len(recommendations)
        
        # Calculate concentration risk
        if account_balance > 0:
            largest_position = max(
                [p.get('value', 0) for p in current_positions] +
                [r.get('position_value', 0) for r in recommendations],
                default=0
            )
            concentration_risk = largest_position / account_balance
        else:
            concentration_risk = 0
        
        # Overall risk level
        if high_risk_count > total_positions * 0.5 or concentration_risk > 0.4:
            overall_risk = 'HIGH'
        elif high_risk_count > 0 or concentration_risk > 0.3:
            overall_risk = 'MEDIUM'
        else:
            overall_risk = 'LOW'
        
        return {
            'overall_risk_level': overall_risk,
            'high_risk_positions': high_risk_count,
            'concentration_risk': concentration_risk,
            'diversification': 'LOW' if total_positions < 3 else 'MEDIUM' if total_positions < 5 else 'HIGH',
            'recommendation': self._get_risk_recommendation(overall_risk, concentration_risk)
        }
    
    def _generate_action_plan(
        self,
        recommendations: List[Dict[str, Any]],
        current_positions: List[Dict[str, Any]],
        account_balance: float
    ) -> Dict[str, Any]:
        """Generate actionable investment plan."""
        actions = []
        
        # Priority 1: High-conviction opportunities
        high_priority = [r for r in recommendations if r.get('priority') == 'HIGH']
        if high_priority:
            actions.append({
                'action': 'BUY',
                'priority': 'HIGH',
                'symbols': [r['symbol'] for r in high_priority],
                'message': f"🚀 HIGH PRIORITY: Invest in {len(high_priority)} high-conviction opportunity(ies)",
                'total_investment': sum(r['position_value'] for r in high_priority)
            })
        
        # Priority 2: Medium opportunities
        medium_priority = [r for r in recommendations if r.get('priority') == 'MEDIUM']
        if medium_priority:
            actions.append({
                'action': 'BUY',
                'priority': 'MEDIUM',
                'symbols': [r['symbol'] for r in medium_priority],
                'message': f"📈 MEDIUM PRIORITY: Consider {len(medium_priority)} balanced opportunity(ies)",
                'total_investment': sum(r['position_value'] for r in medium_priority)
            })
        
        # Check for positions to sell
        # (This would be handled by profit_analyzer in main bot)
        
        return {
            'immediate_actions': actions,
            'total_recommended_investment': sum(r['position_value'] for r in recommendations),
            'cash_remaining': account_balance - sum(r['position_value'] for r in recommendations),
            'next_review': 'Monitor positions daily and review new opportunities weekly'
        }
    
    def _project_growth(
        self,
        current_balance: float,
        recommendations: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Project potential growth based on recommendations."""
        if not recommendations:
            return {'message': 'No recommendations to project'}
        
        # Calculate weighted average expected return
        total_value = sum(r['position_value'] for r in recommendations)
        if total_value == 0:
            return {'message': 'No investment value to project'}
        
        weighted_return = sum(
            r['position_value'] * r['expected_return_pct'] / 100
            for r in recommendations
        ) / total_value
        
        # Projections for different timeframes
        projections = {}
        for months in [1, 3, 6, 12]:
            # Assume monthly compounding
            monthly_return = weighted_return / 12  # Rough estimate
            projected_value = current_balance * ((1 + monthly_return) ** months)
            projections[f'{months}_month'] = {
                'projected_value': projected_value,
                'potential_gain': projected_value - current_balance,
                'gain_percentage': ((projected_value / current_balance) - 1) * 100
            }
        
        return {
            'current_balance': current_balance,
            'weighted_expected_return': weighted_return * 100,
            'projections': projections,
            'target_100k_years': self._estimate_years_to_100k(current_balance, weighted_return)
        }
    
    def _estimate_years_to_100k(self, current: float, annual_return: float) -> Optional[float]:
        """Estimate years to reach $100,000."""
        if current <= 0 or annual_return <= 0:
            return None
        
        target = 100000.0
        if current >= target:
            return 0
        
        # Using compound interest: FV = PV * (1 + r)^n
        # Solving for n: n = log(FV/PV) / log(1 + r)
        import math
        try:
            years = math.log(target / current) / math.log(1 + annual_return)
            return years
        except:
            return None
    
    def _get_account_category(self, balance: float) -> str:
        """Categorize account size."""
        if balance < 100:
            return 'MICRO'
        elif balance < 500:
            return 'SMALL'
        elif balance < 2000:
            return 'MEDIUM'
        elif balance < 10000:
            return 'LARGE'
        else:
            return 'VERY_LARGE'
    
    def _estimate_timeframe(self, opportunity: Dict[str, Any]) -> str:
        """Estimate investment timeframe."""
        strategy = opportunity.get('strategy', '')
        if strategy == 'AGGRESSIVE_GROWTH':
            return '1-4 weeks'  # Aggressive growth is shorter term
        else:
            return '2-8 weeks'  # Standard opportunities
    
    def _generate_reason(self, opportunity: Dict[str, Any]) -> str:
        """Generate human-readable reason for recommendation."""
        strategy = opportunity.get('strategy', '')
        confidence = opportunity.get('conviction_score', opportunity.get('profit_score', 0))
        expected_return = opportunity.get('growth_potential', opportunity.get('profit_potential', 0)) * 100
        
        if strategy == 'AGGRESSIVE_GROWTH':
            return f"High-conviction momentum breakout ({confidence:.0%} confidence) with {expected_return:.0f}% growth potential"
        else:
            return f"Strong profit opportunity ({confidence:.0%} confidence) with {expected_return:.0f}% upside potential"
    
    def _calculate_diversification_score(
        self,
        current_positions: List[Dict[str, Any]],
        recommendations: List[Dict[str, Any]]
    ) -> float:
        """Calculate diversification score (0-1)."""
        all_symbols = set()
        all_symbols.update(p.get('symbol') for p in current_positions)
        all_symbols.update(r.get('symbol') for r in recommendations)
        
        asset_types = set()
        asset_types.update(p.get('asset_type') for p in current_positions)
        asset_types.update(r.get('asset_type') for r in recommendations)
        
        # Score based on number of positions and asset types
        position_score = min(len(all_symbols) / 5, 1.0)  # Max score at 5 positions
        type_score = min(len(asset_types) / 2, 1.0)  # Max score at 2 asset types
        
        return (position_score * 0.7 + type_score * 0.3)
    
    def _get_risk_recommendation(self, overall_risk: str, concentration: float) -> str:
        """Get risk management recommendation."""
        if overall_risk == 'HIGH':
            return "⚠️ High risk detected. Consider reducing position sizes or adding more diversification."
        elif concentration > 0.4:
            return "⚠️ High concentration risk. Consider diversifying across more positions."
        else:
            return "✅ Risk level acceptable for growth strategy."

