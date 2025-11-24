"""
Advanced Risk Metrics - Institutional-grade risk analysis.
Wall Street Use: Professional risk management and position sizing.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class AdvancedRiskMetrics:
    """
    Calculate advanced risk metrics used by institutions.
    
    Metrics:
    1. Value at Risk (VaR)
    2. Conditional VaR (CVaR)
    3. Maximum Drawdown
    4. Calmar Ratio
    5. Sortino Ratio
    6. Beta (correlation to market)
    7. Sharpe Ratio
    8. Risk-Adjusted Returns
    """
    
    def __init__(self):
        """Initialize advanced risk metrics calculator."""
        self.logger = logging.getLogger("ai_investment_bot.risk_metrics")
        
    def calculate_var_cvar(
        self,
        returns: pd.Series,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Calculate Value at Risk (VaR) and Conditional VaR (CVaR).
        
        Wall Street Use: VaR tells you maximum expected loss at confidence level.
        """
        if returns.empty or len(returns) < 20:
            return {}
        
        returns = returns.dropna()
        
        # VaR (percentile method)
        var = np.percentile(returns, (1 - confidence_level) * 100)
        
        # CVaR (Expected Shortfall) - average loss beyond VaR
        cvar = returns[returns <= var].mean()
        
        # Annualized
        periods_per_year = 252
        var_annualized = var * np.sqrt(periods_per_year)
        cvar_annualized = cvar * np.sqrt(periods_per_year) if not np.isnan(cvar) else var_annualized
        
        return {
            'var': float(var),
            'cvar': float(cvar) if not np.isnan(cvar) else float(var),
            'var_annualized': float(var_annualized),
            'cvar_annualized': float(cvar_annualized),
            'confidence_level': confidence_level,
            'max_expected_loss_pct': abs(var_annualized) * 100
        }
    
    def calculate_max_drawdown(
        self,
        prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate Maximum Drawdown and related metrics.
        
        Wall Street Use: Worst peak-to-trough decline.
        """
        if prices.empty or len(prices) < 10:
            return {}
        
        # Cumulative returns
        cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
        
        # Running maximum
        running_max = cumulative.expanding().max()
        
        # Drawdown
        drawdown = (cumulative - running_max) / running_max
        
        max_dd = drawdown.min()
        max_dd_idx = drawdown.idxmin()
        
        # Find peak before drawdown
        peak_idx = cumulative[:max_dd_idx].idxmax() if max_dd_idx in cumulative.index else None
        peak_value = cumulative.loc[peak_idx] if peak_idx else cumulative.max()
        
        # Recovery time (if recovered)
        if max_dd_idx < len(cumulative) - 1:
            recovery = cumulative[max_dd_idx:].loc[cumulative[max_dd_idx:] >= peak_value]
            recovery_time = len(recovery) if len(recovery) > 0 else None
        else:
            recovery_time = None
        
        return {
            'max_drawdown': float(max_dd),
            'max_drawdown_pct': float(max_dd * 100),
            'peak_value': float(peak_value) if peak_value else None,
            'trough_value': float(cumulative.loc[max_dd_idx]) if max_dd_idx else None,
            'recovery_time_days': recovery_time,
            'current_drawdown': float(drawdown.iloc[-1]),
            'current_drawdown_pct': float(drawdown.iloc[-1] * 100)
        }
    
    def calculate_risk_adjusted_metrics(
        self,
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive risk-adjusted metrics.
        
        Wall Street Use: Compare returns adjusted for risk.
        """
        if returns.empty or len(returns) < 20:
            return {}
        
        returns = returns.dropna()
        periods_per_year = 252
        
        # Annualized metrics
        mean_return = returns.mean() * periods_per_year
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        excess_return = mean_return - risk_free_rate
        sharpe = excess_return / volatility if volatility > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else volatility
        sortino = excess_return / downside_std if downside_std > 0 else 0
        
        # Calmar Ratio (Return / Max Drawdown)
        prices = (1 + returns).cumprod()
        running_max = prices.expanding().max()
        drawdown = (prices - running_max) / running_max
        max_dd = abs(drawdown.min())
        calmar = mean_return / max_dd if max_dd > 0 else 0
        
        # Win rate
        winning_trades = (returns > 0).sum()
        win_rate = winning_trades / len(returns) if len(returns) > 0 else 0
        
        # Profit factor
        gross_profit = returns[returns > 0].sum()
        gross_loss = abs(returns[returns < 0].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        return {
            'sharpe_ratio': float(sharpe),
            'sortino_ratio': float(sortino),
            'calmar_ratio': float(calmar),
            'annual_return': float(mean_return),
            'annual_volatility': float(volatility),
            'win_rate': float(win_rate),
            'profit_factor': float(profit_factor) if profit_factor != float('inf') else 999.0,
            'risk_adjusted_return': float(sharpe * volatility)  # Risk-adjusted return
        }
    
    def calculate_beta(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate Beta (sensitivity to market).
        
        Wall Street Use: Measures how asset moves relative to market.
        """
        if asset_returns.empty or market_returns.empty:
            return {}
        
        # Align returns
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 20:
            return {}
        
        # Calculate beta
        covariance = aligned['asset'].cov(aligned['market'])
        market_variance = aligned['market'].var()
        
        beta = covariance / market_variance if market_variance > 0 else 1.0
        
        # Alpha (excess return)
        asset_mean = aligned['asset'].mean() * 252
        market_mean = aligned['market'].mean() * 252
        alpha = asset_mean - (beta * market_mean)
        
        # Correlation
        correlation = aligned['asset'].corr(aligned['market'])
        
        # Interpretation
        if beta > 1.5:
            sensitivity = 'VERY_HIGH'
        elif beta > 1.0:
            sensitivity = 'HIGH'
        elif beta > 0.5:
            sensitivity = 'MODERATE'
        else:
            sensitivity = 'LOW'
        
        return {
            'beta': float(beta),
            'alpha': float(alpha),
            'correlation': float(correlation),
            'sensitivity': sensitivity,
            'market_exposure': 'HIGH' if beta > 1.0 else 'LOW'
        }
    
    def comprehensive_risk_analysis(
        self,
        returns: pd.Series,
        prices: Optional[pd.Series] = None,
        market_returns: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive risk analysis."""
        if returns.empty:
            return {}
        
        results = {
            'timestamp': datetime.now().isoformat()
        }
        
        # VaR/CVaR
        try:
            var_cvar = self.calculate_var_cvar(returns)
            results['var_cvar'] = var_cvar
        except Exception as e:
            self.logger.debug(f"Error in VaR calculation: {e}")
        
        # Max Drawdown
        if prices is not None:
            try:
                drawdown = self.calculate_max_drawdown(prices)
                results['drawdown'] = drawdown
            except Exception as e:
                self.logger.debug(f"Error in drawdown calculation: {e}")
        
        # Risk-adjusted metrics
        try:
            risk_metrics = self.calculate_risk_adjusted_metrics(returns)
            results['risk_metrics'] = risk_metrics
        except Exception as e:
            self.logger.debug(f"Error in risk metrics: {e}")
        
        # Beta (if market data available)
        if market_returns is not None:
            try:
                beta = self.calculate_beta(returns, market_returns)
                results['beta'] = beta
            except Exception as e:
                self.logger.debug(f"Error in beta calculation: {e}")
        
        # Overall risk assessment
        risk_score = 0.0
        if 'var_cvar' in results:
            var_pct = abs(results['var_cvar'].get('var_annualized', 0))
            risk_score += min(var_pct * 10, 0.5)  # Up to 0.5
        
        if 'drawdown' in results:
            dd_pct = abs(results['drawdown'].get('max_drawdown_pct', 0)) / 100
            risk_score += min(dd_pct, 0.5)  # Up to 0.5
        
        risk_level = 'HIGH' if risk_score > 0.6 else 'MEDIUM' if risk_score > 0.3 else 'LOW'
        
        results['overall_risk_score'] = risk_score
        results['risk_level'] = risk_level
        
        return results

