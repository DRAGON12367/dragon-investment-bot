"""
MARKET MICROSTRUCTURE ANALYZER - 200X UPGRADE
Advanced market microstructure analysis including order flow, spread analysis, and execution quality
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')


class MarketMicrostructureAnalyzer:
    """
    Advanced market microstructure analysis.
    
    Features:
    - Bid-ask spread estimation
    - Order flow imbalance
    - Price impact analysis
    - Market depth analysis
    - Execution quality metrics
    - Liquidity analysis
    - Trade size distribution
    - Tick-by-tick analysis
    """
    
    def __init__(self):
        """Initialize market microstructure analyzer."""
        self.logger = logging.getLogger("ai_investment_bot.market_microstructure")
        self.cache = {}
        
    def estimate_bid_ask_spread(
        self,
        df: pd.DataFrame,
        method: str = 'roll'
    ) -> pd.Series:
        """
        Estimate bid-ask spread from price data.
        
        Methods:
        - 'roll': Roll's spread estimator
        - 'corwin_schultz': Corwin-Schultz spread estimator
        - 'high_low': High-low spread estimator
        """
        try:
            if 'close' not in df.columns or len(df) < 2:
                return pd.Series()
            
            close = df['close']
            high = df.get('high', close)
            low = df.get('low', close)
            
            if method == 'roll':
                # Roll's spread estimator: sqrt(-Cov(Δp_t, Δp_{t-1}))
                returns = close.pct_change()
                cov = returns.rolling(2).cov(returns.shift(1))
                spread = np.sqrt(-cov * 2) * close
                return spread.fillna(0)
            
            elif method == 'corwin_schultz':
                # Corwin-Schultz spread estimator
                # Uses high-low prices over 2-day periods
                if len(df) < 2:
                    return pd.Series()
                
                beta = np.log(high / low) ** 2
                gamma = np.log(df['high'].rolling(2).max() / df['low'].rolling(2).min()) ** 2
                
                alpha = (np.sqrt(2 * beta - beta) - np.sqrt(beta)) / (3 - 2 * np.sqrt(2))
                spread = 2 * (np.exp(alpha) - 1) / (1 + np.exp(alpha)) * close
                
                return spread.fillna(0)
            
            else:  # high_low
                # Simple high-low spread
                spread = (high - low) / close
                return spread.fillna(0)
                
        except Exception as e:
            self.logger.error(f"Error estimating spread: {e}")
            return pd.Series()
    
    def analyze_order_flow_imbalance(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze order flow imbalance from price and volume data.
        
        Assumes:
        - Price increases indicate buying pressure
        - Price decreases indicate selling pressure
        """
        try:
            if 'close' not in df.columns or 'volume' not in df.columns:
                return {}
            
            close = df['close']
            volume = df['volume']
            
            # Price change direction
            price_change = close.diff()
            direction = np.where(price_change > 0, 1, np.where(price_change < 0, -1, 0))
            
            # Order flow = direction * volume
            order_flow = direction * volume
            
            # Cumulative order flow
            cumulative_of = order_flow.cumsum()
            
            # Order flow imbalance (normalized)
            of_imbalance = order_flow.rolling(window).sum() / volume.rolling(window).sum()
            
            # Recent imbalance
            recent_imbalance = of_imbalance.iloc[-1] if not of_imbalance.empty else 0.0
            
            # Classify imbalance
            if recent_imbalance > 0.3:
                imbalance_type = 'strong_buying'
            elif recent_imbalance > 0.1:
                imbalance_type = 'buying'
            elif recent_imbalance < -0.3:
                imbalance_type = 'strong_selling'
            elif recent_imbalance < -0.1:
                imbalance_type = 'selling'
            else:
                imbalance_type = 'balanced'
            
            return {
                'order_flow': order_flow,
                'cumulative_order_flow': cumulative_of,
                'imbalance': of_imbalance,
                'recent_imbalance': recent_imbalance,
                'imbalance_type': imbalance_type,
                'avg_imbalance': of_imbalance.mean() if not of_imbalance.empty else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing order flow imbalance: {e}")
            return {}
    
    def estimate_price_impact(
        self,
        df: pd.DataFrame,
        volume_threshold: float = 0.1
    ) -> Dict[str, Any]:
        """
        Estimate price impact of trades.
        
        Measures how much price moves per unit of volume.
        """
        try:
            if 'close' not in df.columns or 'volume' not in df.columns:
                return {}
            
            close = df['close']
            volume = df['volume']
            
            # Price change
            price_change = close.pct_change().abs()
            
            # Volume (normalized)
            avg_volume = volume.rolling(20).mean()
            volume_ratio = volume / (avg_volume + 1e-10)
            
            # Price impact = price_change / volume_ratio
            price_impact = price_change / (volume_ratio + 1e-10)
            
            # Filter for high-volume trades
            high_vol_mask = volume_ratio > volume_threshold
            high_vol_impact = price_impact[high_vol_mask]
            
            return {
                'price_impact': price_impact,
                'avg_price_impact': price_impact.mean() if not price_impact.empty else 0.0,
                'high_volume_impact': high_vol_impact.mean() if not high_vol_impact.empty else 0.0,
                'max_price_impact': price_impact.max() if not price_impact.empty else 0.0,
                'liquidity_score': 1.0 / (price_impact.mean() + 0.01) if not price_impact.empty else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error estimating price impact: {e}")
            return {}
    
    def analyze_market_depth(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Dict[str, Any]:
        """
        Analyze market depth from price and volume patterns.
        
        Higher depth = more liquidity = easier to trade large sizes
        """
        try:
            if 'close' not in df.columns or 'volume' not in df.columns:
                return {}
            
            close = df['close']
            volume = df['volume']
            high = df.get('high', close)
            low = df.get('low', close)
            
            # Price range
            price_range = (high - low) / close
            
            # Volume
            avg_volume = volume.rolling(window).mean()
            
            # Market depth = volume / price_range
            # Higher volume with lower range = deeper market
            market_depth = avg_volume / (price_range + 0.0001)
            
            # Normalize
            depth_score = (market_depth / market_depth.rolling(60).max()).fillna(0.5)
            
            return {
                'market_depth': market_depth,
                'depth_score': depth_score,
                'current_depth': market_depth.iloc[-1] if not market_depth.empty else 0.0,
                'avg_depth': market_depth.mean() if not market_depth.empty else 0.0,
                'depth_trend': 'increasing' if market_depth.iloc[-5:].mean() > market_depth.iloc[-20:-5].mean() else 'decreasing' if not market_depth.empty else 'unknown'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing market depth: {e}")
            return {}
    
    def calculate_execution_quality(
        self,
        trades: List[Dict],
        market_prices: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate execution quality metrics.
        
        Args:
            trades: List of trade dicts with 'timestamp', 'price', 'quantity'
            market_prices: Series of market prices at trade times
        """
        try:
            if not trades or market_prices.empty:
                return {}
            
            execution_costs = []
            slippage_rates = []
            
            for trade in trades:
                timestamp = trade.get('timestamp')
                execution_price = trade.get('price', 0)
                
                if timestamp in market_prices.index:
                    market_price = market_prices.loc[timestamp]
                    
                    # Calculate slippage
                    if trade.get('action') == 'BUY':
                        slippage = (execution_price / market_price - 1) * 100
                    else:  # SELL
                        slippage = (market_price / execution_price - 1) * 100
                    
                    slippage_rates.append(slippage)
                    execution_costs.append(abs(slippage))
            
            if not execution_costs:
                return {}
            
            return {
                'avg_slippage_pct': np.mean(slippage_rates),
                'avg_execution_cost': np.mean(execution_costs),
                'max_slippage': np.max(execution_costs),
                'min_slippage': np.min(execution_costs),
                'execution_quality_score': max(0, 100 - np.mean(execution_costs) * 10)  # Score out of 100
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating execution quality: {e}")
            return {}
    
    def analyze_trade_size_distribution(
        self,
        df: pd.DataFrame,
        volume_col: str = 'volume'
    ) -> Dict[str, Any]:
        """
        Analyze distribution of trade sizes.
        
        Helps identify institutional vs retail activity.
        """
        try:
            if volume_col not in df.columns:
                return {}
            
            volume = df[volume_col]
            
            # Calculate percentiles
            percentiles = [10, 25, 50, 75, 90, 95, 99]
            percentile_values = {f'p{p}': np.percentile(volume, p) for p in percentiles}
            
            # Identify large trades (top 5%)
            large_trade_threshold = np.percentile(volume, 95)
            large_trades = volume[volume > large_trade_threshold]
            
            # Large trade ratio
            large_trade_ratio = len(large_trades) / len(volume) if len(volume) > 0 else 0.0
            
            return {
                'percentiles': percentile_values,
                'mean_volume': volume.mean(),
                'median_volume': volume.median(),
                'std_volume': volume.std(),
                'large_trade_threshold': large_trade_threshold,
                'large_trade_count': len(large_trades),
                'large_trade_ratio': large_trade_ratio,
                'institutional_activity': 'high' if large_trade_ratio > 0.1 else 'moderate' if large_trade_ratio > 0.05 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing trade size distribution: {e}")
            return {}
    
    def get_microstructure_insights(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get comprehensive microstructure insights.
        """
        try:
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Estimate spread
            spread = self.estimate_bid_ask_spread(df, method='roll')
            
            # Order flow imbalance
            order_flow = self.analyze_order_flow_imbalance(df)
            
            # Price impact
            price_impact = self.estimate_price_impact(df)
            
            # Market depth
            market_depth = self.analyze_market_depth(df)
            
            # Trade size distribution
            trade_sizes = self.analyze_trade_size_distribution(df)
            
            return {
                'spread': {
                    'current': spread.iloc[-1] if not spread.empty else 0.0,
                    'avg': spread.mean() if not spread.empty else 0.0,
                    'series': spread
                },
                'order_flow': order_flow,
                'price_impact': price_impact,
                'market_depth': market_depth,
                'trade_sizes': trade_sizes,
                'liquidity_score': market_depth.get('depth_score', pd.Series([0.5])).iloc[-1] if isinstance(market_depth.get('depth_score'), pd.Series) else 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error getting microstructure insights: {e}")
            return {'status': 'error', 'message': str(e)}

