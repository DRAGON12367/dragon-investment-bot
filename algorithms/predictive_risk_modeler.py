"""
PREDICTIVE RISK MODELER - 200X UPGRADE
Advanced risk modeling with VaR, CVaR, stress testing, and scenario analysis
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


class PredictiveRiskModeler:
    """
    Advanced predictive risk modeling system.
    
    Features:
    - Value at Risk (VaR)
    - Conditional VaR (CVaR)
    - Stress testing
    - Scenario analysis
    - Risk decomposition
    - Portfolio risk attribution
    - Dynamic risk limits
    """
    
    def __init__(self, confidence_level: float = 0.95):
        """Initialize risk modeler."""
        self.logger = logging.getLogger("ai_investment_bot.risk_modeler")
        self.confidence_level = confidence_level
        
    def calculate_var(
        self,
        returns: pd.Series,
        method: str = 'historical',
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Value at Risk (VaR).
        
        Methods:
        - 'historical': Historical simulation
        - 'parametric': Parametric (normal distribution)
        - 'monte_carlo': Monte Carlo simulation
        """
        try:
            if returns.empty:
                return {}
            
            conf_level = confidence_level or self.confidence_level
            alpha = 1 - conf_level
            
            if method == 'historical':
                # Historical VaR: percentile of historical returns
                var = np.percentile(returns, alpha * 100)
                
            elif method == 'parametric':
                # Parametric VaR: assumes normal distribution
                mean_return = returns.mean()
                std_return = returns.std()
                var = mean_return + std_return * stats.norm.ppf(alpha)
                
            elif method == 'monte_carlo':
                # Monte Carlo VaR
                mean_return = returns.mean()
                std_return = returns.std()
                
                # Generate random returns
                n_simulations = 10000
                simulated_returns = np.random.normal(mean_return, std_return, n_simulations)
                var = np.percentile(simulated_returns, alpha * 100)
                
            else:
                return {}
            
            return {
                'var': var,
                'var_pct': var * 100,
                'confidence_level': conf_level,
                'method': method
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating VaR: {e}")
            return {}
    
    def calculate_cvar(
        self,
        returns: pd.Series,
        confidence_level: Optional[float] = None
    ) -> Dict[str, float]:
        """
        Calculate Conditional VaR (CVaR) / Expected Shortfall.
        
        CVaR = Expected loss given that loss exceeds VaR
        """
        try:
            if returns.empty:
                return {}
            
            conf_level = confidence_level or self.confidence_level
            alpha = 1 - conf_level
            
            # Calculate VaR first
            var = np.percentile(returns, alpha * 100)
            
            # CVaR = mean of returns below VaR
            tail_returns = returns[returns <= var]
            
            if len(tail_returns) == 0:
                cvar = var
            else:
                cvar = tail_returns.mean()
            
            return {
                'cvar': cvar,
                'cvar_pct': cvar * 100,
                'var': var,
                'var_pct': var * 100,
                'confidence_level': conf_level,
                'tail_observations': len(tail_returns)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating CVaR: {e}")
            return {}
    
    def stress_test(
        self,
        portfolio_returns: pd.Series,
        stress_scenarios: List[float]
    ) -> Dict[str, Any]:
        """
        Stress test portfolio under various scenarios.
        
        Args:
            portfolio_returns: Historical portfolio returns
            stress_scenarios: List of stress return percentages (e.g., [-0.1, -0.2, -0.3])
        """
        try:
            if portfolio_returns.empty:
                return {}
            
            results = []
            
            for stress_return in stress_scenarios:
                # Calculate impact
                current_value = 10000.0  # Default portfolio value
                stressed_value = current_value * (1 + stress_return)
                loss = current_value - stressed_value
                loss_pct = abs(stress_return) * 100
                
                # Probability of this scenario (based on historical distribution)
                prob = (portfolio_returns <= stress_return).sum() / len(portfolio_returns) * 100
                
                results.append({
                    'scenario': f'{stress_return*100:.1f}% decline',
                    'stress_return': stress_return,
                    'stressed_value': stressed_value,
                    'loss': loss,
                    'loss_pct': loss_pct,
                    'probability': prob
                })
            
            return {
                'stress_scenarios': results,
                'worst_case': min(results, key=lambda x: x['stress_return']),
                'most_likely_severe': max([r for r in results if r['probability'] > 1], 
                                         key=lambda x: x['probability'], default=None)
            }
            
        except Exception as e:
            self.logger.error(f"Error in stress test: {e}")
            return {}
    
    def scenario_analysis(
        self,
        portfolio_weights: Dict[str, float],
        asset_returns: pd.DataFrame,
        scenarios: Dict[str, Dict[str, float]]
    ) -> Dict[str, Any]:
        """
        Analyze portfolio performance under different scenarios.
        
        Args:
            portfolio_weights: Current portfolio weights
            asset_returns: Historical returns for each asset
            scenarios: Dict of scenario_name -> Dict of asset -> return_multiplier
        """
        try:
            if asset_returns.empty:
                return {}
            
            # Calculate base portfolio return
            base_returns = []
            for asset, weight in portfolio_weights.items():
                if asset in asset_returns.columns:
                    asset_ret = asset_returns[asset].mean() * 252  # Annualized
                    base_returns.append(weight * asset_ret)
            
            base_portfolio_return = sum(base_returns)
            
            # Analyze each scenario
            scenario_results = []
            
            for scenario_name, multipliers in scenarios.items():
                scenario_returns = []
                
                for asset, weight in portfolio_weights.items():
                    if asset in asset_returns.columns:
                        multiplier = multipliers.get(asset, 1.0)  # Default: no change
                        asset_ret = asset_returns[asset].mean() * 252 * multiplier
                        scenario_returns.append(weight * asset_ret)
                
                scenario_portfolio_return = sum(scenario_returns)
                return_change = scenario_portfolio_return - base_portfolio_return
                
                scenario_results.append({
                    'scenario': scenario_name,
                    'portfolio_return': scenario_portfolio_return,
                    'return_change': return_change,
                    'return_change_pct': (return_change / (abs(base_portfolio_return) + 0.01)) * 100
                })
            
            return {
                'base_portfolio_return': base_portfolio_return,
                'scenarios': scenario_results,
                'best_case': max(scenario_results, key=lambda x: x['portfolio_return']),
                'worst_case': min(scenario_results, key=lambda x: x['portfolio_return'])
            }
            
        except Exception as e:
            self.logger.error(f"Error in scenario analysis: {e}")
            return {}
    
    def decompose_portfolio_risk(
        self,
        portfolio_weights: Dict[str, float],
        returns: pd.DataFrame,
        cov_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Decompose portfolio risk by asset.
        
        Shows contribution of each asset to total portfolio risk.
        """
        try:
            if returns.empty:
                return {}
            
            # Calculate covariance matrix if not provided
            if cov_matrix is None:
                cov_matrix = returns.cov() * 252  # Annualized
            
            # Portfolio variance
            weights_array = np.array([portfolio_weights.get(asset, 0.0) for asset in returns.columns])
            portfolio_var = np.dot(weights_array.T, np.dot(cov_matrix, weights_array))
            portfolio_vol = np.sqrt(portfolio_var)
            
            # Risk contribution of each asset
            marginal_contrib = np.dot(cov_matrix, weights_array) / portfolio_vol
            risk_contributions = weights_array * marginal_contrib
            
            # Percentage contribution
            contributions = {}
            for i, asset in enumerate(returns.columns):
                contrib_pct = (risk_contributions[i] / portfolio_vol) * 100 if portfolio_vol > 0 else 0.0
                contributions[asset] = {
                    'risk_contribution': risk_contributions[i],
                    'contribution_pct': contrib_pct,
                    'weight': portfolio_weights.get(asset, 0.0)
                }
            
            return {
                'portfolio_volatility': portfolio_vol,
                'portfolio_variance': portfolio_var,
                'risk_contributions': contributions,
                'total_risk': portfolio_vol
            }
            
        except Exception as e:
            self.logger.error(f"Error decomposing portfolio risk: {e}")
            return {}
    
    def calculate_dynamic_risk_limits(
        self,
        returns: pd.Series,
        base_limit: float = 0.02,
        volatility_multiplier: float = 1.5
    ) -> Dict[str, float]:
        """
        Calculate dynamic risk limits based on current volatility.
        
        Risk limits adjust with market volatility.
        """
        try:
            if returns.empty:
                return {'risk_limit': base_limit}
            
            # Current volatility
            current_vol = returns.rolling(20).std().iloc[-1] if len(returns) >= 20 else returns.std()
            annualized_vol = current_vol * np.sqrt(252)
            
            # Historical average volatility
            avg_vol = returns.std() * np.sqrt(252)
            
            # Volatility ratio
            vol_ratio = annualized_vol / (avg_vol + 1e-10)
            
            # Adjust risk limit
            if vol_ratio > 1.5:
                # High volatility: reduce risk limit
                risk_limit = base_limit / volatility_multiplier
            elif vol_ratio < 0.7:
                # Low volatility: can increase risk limit slightly
                risk_limit = base_limit * 1.1
            else:
                risk_limit = base_limit
            
            return {
                'risk_limit': risk_limit,
                'current_volatility': annualized_vol,
                'avg_volatility': avg_vol,
                'volatility_ratio': vol_ratio,
                'risk_adjustment': 'reduced' if vol_ratio > 1.5 else 'increased' if vol_ratio < 0.7 else 'normal'
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating dynamic risk limits: {e}")
            return {'risk_limit': base_limit}
    
    def get_comprehensive_risk_report(
        self,
        portfolio_returns: pd.Series,
        portfolio_weights: Optional[Dict[str, float]] = None,
        asset_returns: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive risk analysis report.
        """
        try:
            if portfolio_returns.empty:
                return {'status': 'insufficient_data'}
            
            # Calculate VaR and CVaR
            var_historical = self.calculate_var(portfolio_returns, method='historical')
            var_parametric = self.calculate_var(portfolio_returns, method='parametric')
            cvar = self.calculate_cvar(portfolio_returns)
            
            # Stress test
            stress_scenarios = [-0.05, -0.10, -0.15, -0.20, -0.30]
            stress_test = self.stress_test(portfolio_returns, stress_scenarios)
            
            # Dynamic risk limits
            risk_limits = self.calculate_dynamic_risk_limits(portfolio_returns)
            
            # Risk decomposition (if portfolio weights provided)
            risk_decomp = {}
            if portfolio_weights and asset_returns is not None:
                risk_decomp = self.decompose_portfolio_risk(portfolio_weights, asset_returns)
            
            return {
                'var_historical': var_historical,
                'var_parametric': var_parametric,
                'cvar': cvar,
                'stress_test': stress_test,
                'risk_limits': risk_limits,
                'risk_decomposition': risk_decomp,
                'portfolio_volatility': portfolio_returns.std() * np.sqrt(252) * 100,
                'max_drawdown': ((portfolio_returns.cumsum().expanding().max() - portfolio_returns.cumsum()) / 
                               portfolio_returns.cumsum().expanding().max()).min() * 100
            }
            
        except Exception as e:
            self.logger.error(f"Error getting risk report: {e}")
            return {'status': 'error', 'message': str(e)}

