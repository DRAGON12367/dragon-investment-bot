"""
Mean Reversion Strategy - Buy oversold, sell overbought.
Wall Street Use: Statistical arbitrage and pairs trading.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class MeanReversionStrategy:
    """
    Mean reversion trading strategy.
    
    Concept: Prices tend to return to their average (mean).
    Buy when price is below mean, sell when above mean.
    """
    
    def __init__(self):
        """Initialize mean reversion strategy."""
        self.logger = logging.getLogger("ai_investment_bot.mean_reversion")
        
    def calculate_z_score(
        self,
        prices: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """
        Calculate Z-score (how many standard deviations from mean).
        
        Z-score > 2 = overbought (sell)
        Z-score < -2 = oversold (buy)
        """
        mean = prices.rolling(window).mean()
        std = prices.rolling(window).std()
        z_score = (prices - mean) / std
        return z_score
    
    def detect_mean_reversion_opportunities(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Detect mean reversion trading opportunities.
        """
        if df.empty or len(df) < 30:
            return {}
        
        prices = df['close']
        
        # Calculate Z-score
        z_score = self.calculate_z_score(prices, window=20)
        current_z = z_score.iloc[-1]
        
        # Mean reversion signals
        if current_z < -2.0:
            signal = 'STRONG_BUY'  # Very oversold
            confidence = 0.85
        elif current_z < -1.5:
            signal = 'BUY'  # Oversold
            confidence = 0.70
        elif current_z > 2.0:
            signal = 'STRONG_SELL'  # Very overbought
            confidence = 0.85
        elif current_z > 1.5:
            signal = 'SELL'  # Overbought
            confidence = 0.70
        else:
            signal = 'HOLD'
            confidence = 0.50
        
        # Calculate mean and target
        mean_price = prices.rolling(20).mean().iloc[-1]
        current_price = prices.iloc[-1]
        
        # Expected return to mean
        if current_price < mean_price:
            expected_return = (mean_price - current_price) / current_price
            target_price = mean_price
        else:
            expected_return = (mean_price - current_price) / current_price
            target_price = mean_price
        
        return {
            'symbol': symbol,
            'strategy': 'MEAN_REVERSION',
            'signal': signal,
            'confidence': confidence,
            'z_score': float(current_z),
            'current_price': float(current_price),
            'mean_price': float(mean_price),
            'target_price': float(target_price),
            'expected_return': float(expected_return),
            'distance_from_mean_pct': float(abs(current_price - mean_price) / mean_price * 100)
        }
    
    def pairs_trading_opportunity(
        self,
        price1: pd.Series,
        price2: pd.Series,
        symbol1: str,
        symbol2: str
    ) -> Dict[str, Any]:
        """
        Pairs trading - trade the spread between two correlated assets.
        
        Wall Street Use: Statistical arbitrage between correlated stocks.
        """
        if len(price1) < 30 or len(price2) < 30:
            return {}
        
        # Calculate ratio
        ratio = price1 / price2
        
        # Z-score of ratio
        mean_ratio = ratio.rolling(20).mean()
        std_ratio = ratio.rolling(20).std()
        z_score = (ratio.iloc[-1] - mean_ratio.iloc[-1]) / std_ratio.iloc[-1]
        
        # Correlation
        returns1 = price1.pct_change().dropna()
        returns2 = price2.pct_change().dropna()
        correlation = returns1.corr(returns2)
        
        # Pairs trading signal
        if z_score > 2.0 and correlation > 0.7:
            # Ratio too high - sell asset1, buy asset2
            signal = {
                'action': 'PAIRS_TRADE',
                'sell': symbol1,
                'buy': symbol2,
                'z_score': float(z_score),
                'correlation': float(correlation),
                'confidence': 0.75
            }
        elif z_score < -2.0 and correlation > 0.7:
            # Ratio too low - buy asset1, sell asset2
            signal = {
                'action': 'PAIRS_TRADE',
                'buy': symbol1,
                'sell': symbol2,
                'z_score': float(z_score),
                'correlation': float(correlation),
                'confidence': 0.75
            }
        else:
            signal = None
        
        return signal if signal else {}

