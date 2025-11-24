"""
Professional investor analysis tools - Support/Resistance, Volume Profile, Correlation, Risk Metrics
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional

try:
    from scipy import stats
    from scipy.optimize import minimize
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


class ProfessionalAnalysis:
    """Professional-grade analysis tools for institutional investors."""
    
    def detect_support_resistance(
        self, 
        df: pd.DataFrame, 
        window: int = 20,
        min_touches: int = 2
    ) -> Dict[str, List[float]]:
        """
        Detect support and resistance levels using pivot points and clustering.
        
        Args:
            df: DataFrame with OHLCV data
            window: Window for pivot point detection
            min_touches: Minimum number of touches to consider a level
            
        Returns:
            Dictionary with 'support' and 'resistance' levels
        """
        if df.empty or len(df) < window * 2:
            return {'support': [], 'resistance': []}
        
        df = df.copy()
        
        # Find pivot highs and lows
        pivot_highs = []
        pivot_lows = []
        
        for i in range(window, len(df) - window):
            # Pivot high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                pivot_highs.append(df['high'].iloc[i])
            
            # Pivot low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                pivot_lows.append(df['low'].iloc[i])
        
        # Cluster similar levels (within 1% of each other)
        def cluster_levels(levels: List[float], tolerance: float = 0.01) -> List[float]:
            if not levels:
                return []
            
            levels = sorted(levels)
            clusters = []
            current_cluster = [levels[0]]
            
            for level in levels[1:]:
                if abs(level - current_cluster[-1]) / current_cluster[-1] <= tolerance:
                    current_cluster.append(level)
                else:
                    if len(current_cluster) >= min_touches:
                        clusters.append(np.mean(current_cluster))
                    current_cluster = [level]
            
            if len(current_cluster) >= min_touches:
                clusters.append(np.mean(current_cluster))
            
            return sorted(clusters)
        
        resistance = cluster_levels(pivot_highs)
        support = cluster_levels(pivot_lows)
        
        # Get current price for relevance filtering
        current_price = df['close'].iloc[-1]
        
        # Filter to relevant levels (within 20% of current price)
        resistance = [r for r in resistance if 0.8 * current_price <= r <= 1.2 * current_price]
        support = [s for s in support if 0.8 * current_price <= s <= 1.2 * current_price]
        
        return {
            'support': support[:5],  # Top 5 support levels
            'resistance': resistance[:5]  # Top 5 resistance levels
        }
    
    def calculate_volume_profile(
        self, 
        df: pd.DataFrame, 
        bins: int = 20
    ) -> pd.DataFrame:
        """
        Calculate volume profile (Price by Volume).
        
        Args:
            df: DataFrame with OHLCV data
            bins: Number of price bins
            
        Returns:
            DataFrame with price bins and volume distribution
        """
        if df.empty:
            return pd.DataFrame()
        
        df = df.copy()
        
        # Create price bins
        price_min = df['low'].min()
        price_max = df['high'].max()
        price_bins = np.linspace(price_min, price_max, bins + 1)
        
        # Distribute volume across price bins
        volume_profile = []
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            bin_center = (bin_low + bin_high) / 2
            
            # Calculate volume in this price range
            mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
            volume_in_bin = df.loc[mask, 'volume'].sum()
            
            volume_profile.append({
                'price': bin_center,
                'volume': volume_in_bin,
                'price_low': bin_low,
                'price_high': bin_high
            })
        
        return pd.DataFrame(volume_profile)
    
    def calculate_correlation_matrix(
        self, 
        price_data: Dict[str, pd.Series]
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for multiple assets.
        
        Args:
            price_data: Dictionary of symbol -> price series
            
        Returns:
            Correlation matrix DataFrame
        """
        if not price_data:
            return pd.DataFrame()
        
        # Align all series to common dates
        df = pd.DataFrame(price_data)
        
        # Calculate returns
        returns = df.pct_change().dropna()
        
        # Calculate correlation
        correlation = returns.corr()
        
        return correlation
    
    def calculate_risk_metrics(
        self, 
        returns: pd.Series,
        risk_free_rate: float = 0.02
    ) -> Dict[str, float]:
        """
        Calculate professional risk metrics.
        
        Args:
            returns: Series of returns
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Dictionary of risk metrics
        """
        if returns.empty or len(returns) < 2:
            return {}
        
        returns = returns.dropna()
        
        # Annualized metrics
        periods_per_year = 252  # Trading days
        
        # Mean return
        mean_return = returns.mean() * periods_per_year
        
        # Volatility (standard deviation)
        volatility = returns.std() * np.sqrt(periods_per_year)
        
        # Sharpe Ratio
        sharpe_ratio = (mean_return - risk_free_rate) / volatility if volatility > 0 else 0
        
        # Sortino Ratio (only downside deviation)
        downside_returns = returns[returns < 0]
        downside_std = downside_returns.std() * np.sqrt(periods_per_year) if len(downside_returns) > 0 else volatility
        sortino_ratio = (mean_return - risk_free_rate) / downside_std if downside_std > 0 else 0
        
        # Maximum Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Value at Risk (VaR) - 95% confidence
        var_95 = np.percentile(returns, 5) * np.sqrt(periods_per_year)
        
        # Conditional VaR (CVaR) - Expected loss beyond VaR
        cvar_95 = returns[returns <= np.percentile(returns, 5)].mean() * np.sqrt(periods_per_year) if len(returns[returns <= np.percentile(returns, 5)]) > 0 else var_95
        
        # Calmar Ratio (Return / Max Drawdown)
        calmar_ratio = mean_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'volatility': volatility,
            'var_95': var_95,
            'cvar_95': cvar_95,
            'calmar_ratio': calmar_ratio,
            'mean_return': mean_return
        }
    
    def calculate_trend_strength(
        self, 
        df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Calculate trend strength using multiple methods.
        
        Args:
            df: DataFrame with OHLCV data
            
        Returns:
            Dictionary of trend strength metrics
        """
        if df.empty or len(df) < 50:
            return {}
        
        df = df.copy()
        
        # Linear regression slope
        x = np.arange(len(df))
        y = df['close'].values
        
        if SCIPY_AVAILABLE:
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        else:
            # Fallback to numpy polyfit
            coeffs = np.polyfit(x, y, 1)
            slope = coeffs[0]
            # Calculate R-squared manually
            y_pred = np.polyval(coeffs, x)
            ss_res = np.sum((y - y_pred) ** 2)
            ss_tot = np.sum((y - np.mean(y)) ** 2)
            r_value = np.sqrt(1 - (ss_res / ss_tot)) if ss_tot > 0 else 0
        
        # Trend strength (R-squared)
        trend_strength = r_value ** 2
        
        # ADX (if available)
        adx = df.get('adx', pd.Series([0] * len(df))).iloc[-1] if 'adx' in df.columns else 0
        
        # Price momentum
        returns = df['close'].pct_change().dropna()
        momentum = returns.tail(20).mean() * 252  # Annualized
        
        # Price position relative to range
        price_range = df['high'].max() - df['low'].min()
        current_price = df['close'].iloc[-1]
        price_position = (current_price - df['low'].min()) / price_range if price_range > 0 else 0.5
        
        return {
            'trend_strength': trend_strength,
            'trend_direction': 1 if slope > 0 else -1,
            'adx': adx,
            'momentum': momentum,
            'price_position': price_position,
            'slope': slope
        }
    
    def detect_fibonacci_levels(
        self, 
        high: float, 
        low: float
    ) -> Dict[str, float]:
        """
        Calculate Fibonacci retracement levels.
        
        Args:
            high: Recent high price
            low: Recent low price
            
        Returns:
            Dictionary of Fibonacci levels
        """
        diff = high - low
        
        levels = {
            'fib_0': high,
            'fib_23.6': high - 0.236 * diff,
            'fib_38.2': high - 0.382 * diff,
            'fib_50': high - 0.5 * diff,
            'fib_61.8': high - 0.618 * diff,
            'fib_78.6': high - 0.786 * diff,
            'fib_100': low
        }
        
        return levels

