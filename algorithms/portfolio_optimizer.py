"""
Advanced portfolio optimization using Modern Portfolio Theory and risk-adjusted returns.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from utils.config import Config


class PortfolioOptimizer:
    """Advanced portfolio optimization algorithms."""
    
    def __init__(self, config: Config):
        """Initialize portfolio optimizer."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.portfolio_optimizer")
    
    def optimize_portfolio(
        self,
        returns: pd.DataFrame,
        method: str = 'sharpe',
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Optimize portfolio allocation.
        
        Args:
            returns: DataFrame of asset returns
            method: Optimization method ('sharpe', 'min_variance', 'max_return')
            risk_free_rate: Risk-free rate for Sharpe ratio
            
        Returns:
            Dictionary with optimal weights and metrics
        """
        if returns.empty or len(returns.columns) == 0:
            return {}
        
        # Calculate expected returns and covariance
        mean_returns = returns.mean()
        cov_matrix = returns.cov()
        
        # Use Ledoit-Wolf shrinkage for better covariance estimation
        try:
            lw = LedoitWolf()
            cov_matrix = pd.DataFrame(
                lw.fit(returns).covariance_,
                index=cov_matrix.index,
                columns=cov_matrix.columns
            )
        except Exception:
            pass  # Fallback to regular covariance
        
        num_assets = len(mean_returns)
        
        # Objective functions
        if method == 'sharpe':
            def objective(weights):
                portfolio_return = np.sum(mean_returns * weights)
                portfolio_std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                sharpe = (portfolio_return - risk_free_rate) / portfolio_std
                return -sharpe  # Minimize negative Sharpe
        elif method == 'min_variance':
            def objective(weights):
                return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
        else:  # max_return
            def objective(weights):
                return -np.sum(mean_returns * weights)
        
        # Constraints
        constraints = {'type': 'eq', 'fun': lambda x: np.sum(x) - 1}
        
        # Bounds (long-only portfolio)
        bounds = tuple((0, 1) for _ in range(num_assets))
        
        # Initial guess (equal weights)
        initial_weights = np.array([1/num_assets] * num_assets)
        
        # Optimize
        try:
            result = minimize(
                objective,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                portfolio_return = np.sum(mean_returns * optimal_weights)
                portfolio_std = np.sqrt(np.dot(optimal_weights.T, np.dot(cov_matrix, optimal_weights)))
                sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_std if portfolio_std > 0 else 0
                
                return {
                    'weights': dict(zip(mean_returns.index, optimal_weights)),
                    'expected_return': float(portfolio_return),
                    'volatility': float(portfolio_std),
                    'sharpe_ratio': float(sharpe_ratio),
                    'method': method
                }
        except Exception as e:
            self.logger.error(f"Portfolio optimization error: {e}")
        
        return {}
    
    def calculate_risk_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """Calculate advanced risk metrics."""
        if returns.empty:
            return {}
        
        metrics = {}
        
        # Sharpe Ratio
        excess_returns = returns.mean() - risk_free_rate / 252  # Daily risk-free rate
        sharpe = excess_returns / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        metrics['sharpe_ratio'] = float(sharpe)
        
        # Sortino Ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() if len(downside_returns) > 0 else 0
        sortino = excess_returns / downside_std * np.sqrt(252) if downside_std > 0 else 0
        metrics['sortino_ratio'] = float(sortino)
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        metrics['max_drawdown'] = float(drawdown.min())
        
        # Value at Risk (VaR) - 95% confidence
        metrics['var_95'] = float(returns.quantile(0.05))
        
        # Conditional VaR (CVaR)
        cvar = returns[returns <= metrics['var_95']].mean()
        metrics['cvar_95'] = float(cvar) if not np.isnan(cvar) else 0
        
        return metrics
    
    def rebalance_portfolio(
        self,
        current_positions: Dict[str, float],
        target_weights: Dict[str, float],
        total_value: float
    ) -> List[Dict[str, Any]]:
        """
        Generate rebalancing trades.
        
        Args:
            current_positions: Current position values
            target_weights: Target portfolio weights
            total_value: Total portfolio value
            
        Returns:
            List of rebalancing trades
        """
        trades = []
        
        # Calculate target values
        target_values = {symbol: total_value * weight for symbol, weight in target_weights.items()}
        
        # Calculate current weights
        current_total = sum(current_positions.values())
        if current_total == 0:
            current_weights = {symbol: 0 for symbol in target_weights.keys()}
        else:
            current_weights = {symbol: value / current_total for symbol, value in current_positions.items()}
        
        # Generate trades
        for symbol in set(list(current_positions.keys()) + list(target_weights.keys())):
            current_value = current_positions.get(symbol, 0)
            target_value = target_values.get(symbol, 0)
            difference = target_value - current_value
            
            # Only trade if difference is significant (>1% of portfolio)
            if abs(difference) > total_value * 0.01:
                trades.append({
                    'symbol': symbol,
                    'action': 'BUY' if difference > 0 else 'SELL',
                    'value': abs(difference),
                    'current_weight': current_weights.get(symbol, 0),
                    'target_weight': target_weights.get(symbol, 0)
                })
        
        return trades

