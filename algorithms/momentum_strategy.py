"""
Momentum Strategy - Buy winners, sell losers.
Wall Street Use: Trend following and momentum investing.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class MomentumStrategy:
    """
    Momentum trading strategy.
    
    Concept: Assets that performed well recently continue to perform well.
    Buy assets with strong momentum, avoid weak momentum.
    """
    
    def __init__(self):
        """Initialize momentum strategy."""
        self.logger = logging.getLogger("ai_investment_bot.momentum")
        
    def calculate_momentum_score(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive momentum score.
        """
        if df.empty or len(df) < 50:
            return {}
        
        prices = df['close']
        returns = prices.pct_change()
        
        # Multiple timeframe momentum
        momentum_5d = returns.tail(5).sum()
        momentum_10d = returns.tail(10).sum()
        momentum_20d = returns.tail(20).sum()
        momentum_50d = returns.tail(50).sum()
        
        # Price vs moving averages
        sma_20 = prices.rolling(20).mean().iloc[-1]
        sma_50 = prices.rolling(50).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Momentum score (weighted average)
        momentum_score = (
            momentum_5d * 0.3 +
            momentum_10d * 0.25 +
            momentum_20d * 0.25 +
            momentum_50d * 0.2
        )
        
        # Price position
        above_sma20 = current_price > sma_20
        above_sma50 = current_price > sma_50
        
        # Volume confirmation
        if 'volume' in df.columns:
            volume_trend = df['volume'].tail(10).mean() > df['volume'].tail(20).head(10).mean()
        else:
            volume_trend = True
        
        # Momentum signal
        if momentum_score > 0.05 and above_sma20 and above_sma50 and volume_trend:
            signal = 'STRONG_BUY'
            confidence = 0.85
        elif momentum_score > 0.02 and above_sma20:
            signal = 'BUY'
            confidence = 0.70
        elif momentum_score < -0.05:
            signal = 'STRONG_SELL'
            confidence = 0.85
        elif momentum_score < -0.02:
            signal = 'SELL'
            confidence = 0.70
        else:
            signal = 'HOLD'
            confidence = 0.50
        
        return {
            'momentum_score': float(momentum_score),
            'momentum_5d': float(momentum_5d),
            'momentum_10d': float(momentum_10d),
            'momentum_20d': float(momentum_20d),
            'momentum_50d': float(momentum_50d),
            'above_sma20': above_sma20,
            'above_sma50': above_sma50,
            'signal': signal,
            'confidence': confidence,
            'trend_strength': 'STRONG' if abs(momentum_score) > 0.05 else 'MODERATE' if abs(momentum_score) > 0.02 else 'WEAK'
        }
    
    def relative_strength(
        self,
        asset_returns: pd.Series,
        market_returns: pd.Series
    ) -> Dict[str, Any]:
        """
        Calculate relative strength vs market.
        
        Wall Street Use: Find assets outperforming the market.
        """
        if len(asset_returns) < 20 or len(market_returns) < 20:
            return {}
        
        # Align returns
        aligned = pd.DataFrame({
            'asset': asset_returns,
            'market': market_returns
        }).dropna()
        
        if len(aligned) < 20:
            return {}
        
        # Cumulative returns
        asset_cumulative = (1 + aligned['asset']).cumprod().iloc[-1] - 1
        market_cumulative = (1 + aligned['market']).cumprod().iloc[-1] - 1
        
        # Relative strength
        relative_strength = asset_cumulative - market_cumulative
        
        # Signal
        if relative_strength > 0.10:
            signal = 'STRONG_BUY'  # Outperforming by 10%+
            confidence = 0.80
        elif relative_strength > 0.05:
            signal = 'BUY'  # Outperforming by 5%+
            confidence = 0.65
        elif relative_strength < -0.10:
            signal = 'STRONG_SELL'  # Underperforming by 10%+
            confidence = 0.80
        elif relative_strength < -0.05:
            signal = 'SELL'  # Underperforming by 5%+
            confidence = 0.65
        else:
            signal = 'HOLD'
            confidence = 0.50
        
        return {
            'relative_strength': float(relative_strength),
            'asset_return': float(asset_cumulative),
            'market_return': float(market_cumulative),
            'outperformance': float(relative_strength * 100),
            'signal': signal,
            'confidence': confidence
        }

