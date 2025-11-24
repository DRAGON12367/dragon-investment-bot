"""
Volatility Trading Strategy - Trade based on volatility regimes.
Wall Street Use: VIX trading, volatility arbitrage, options strategies.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime


class VolatilityTradingStrategy:
    """
    Volatility-based trading strategy.
    
    Concept: Different strategies work in different volatility regimes.
    Low volatility = trend following, High volatility = mean reversion.
    """
    
    def __init__(self):
        """Initialize volatility trading strategy."""
        self.logger = logging.getLogger("ai_investment_bot.volatility")
        
    def detect_volatility_regime(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect current volatility regime.
        """
        if df.empty or len(df) < 50:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        # Calculate volatility
        current_vol = returns.tail(20).std() * np.sqrt(252)  # Annualized
        historical_vol = returns.rolling(60).std().dropna() * np.sqrt(252) if len(returns) >= 60 else pd.Series([current_vol])
        
        avg_vol = historical_vol.mean() if len(historical_vol) > 0 else current_vol
        
        # Volatility percentile
        vol_percentile = (current_vol > historical_vol).sum() / len(historical_vol) if len(historical_vol) > 0 else 0.5
        
        # Regime classification
        if vol_percentile > 0.75:
            regime = 'HIGH_VOLATILITY'
            strategy = 'MEAN_REVERSION'  # High vol = mean revert
        elif vol_percentile < 0.25:
            regime = 'LOW_VOLATILITY'
            strategy = 'TREND_FOLLOWING'  # Low vol = trend
        else:
            regime = 'NORMAL_VOLATILITY'
            strategy = 'MIXED'
        
        return {
            'volatility_regime': regime,
            'current_volatility': float(current_vol),
            'average_volatility': float(avg_vol),
            'volatility_percentile': float(vol_percentile),
            'recommended_strategy': strategy,
            'volatility_ratio': float(current_vol / avg_vol) if avg_vol > 0 else 1.0
        }
    
    def volatility_breakout_signal(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect volatility breakouts (sudden increase in volatility).
        
        Wall Street Use: Volatility breakouts often precede big moves.
        """
        if df.empty or len(df) < 30:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        # Short-term vs long-term volatility
        short_vol = returns.tail(5).std()
        long_vol = returns.tail(20).std()
        
        # Volatility spike
        vol_ratio = short_vol / long_vol if long_vol > 0 else 1.0
        
        # Breakout detection
        if vol_ratio > 2.0:
            signal = 'VOLATILITY_SPIKE'
            interpretation = 'BREAKOUT_IMMINENT'
            confidence = 0.75
        elif vol_ratio > 1.5:
            signal = 'VOLATILITY_INCREASE'
            interpretation = 'WATCH_CLOSELY'
            confidence = 0.60
        else:
            signal = 'NORMAL'
            interpretation = 'STABLE'
            confidence = 0.50
        
        return {
            'signal': signal,
            'interpretation': interpretation,
            'volatility_ratio': float(vol_ratio),
            'short_term_vol': float(short_vol),
            'long_term_vol': float(long_vol),
            'confidence': confidence,
            'trading_implication': 'REDUCE_POSITION_SIZE' if vol_ratio > 2.0 else 'NORMAL'
        }

