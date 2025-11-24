"""
Sentiment Analysis - Market sentiment and crowd psychology indicators.
Wall Street uses sentiment to identify contrarian opportunities.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
from collections import defaultdict


class SentimentAnalyzer:
    """
    Analyze market sentiment using price action and volume patterns.
    
    Techniques:
    1. Fear & Greed Index (simulated from price action)
    2. Put/Call Ratio (simulated from price patterns)
    3. Contrarian Signals (extreme sentiment = reversal)
    4. Crowd Psychology (herding behavior detection)
    5. News Sentiment (price reaction to moves)
    """
    
    def __init__(self):
        """Initialize sentiment analyzer."""
        self.logger = logging.getLogger("ai_investment_bot.sentiment")
        self.sentiment_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def calculate_fear_greed_index(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate Fear & Greed Index from price action.
        
        Wall Street Use: Extreme fear = buying opportunity, extreme greed = selling opportunity.
        """
        if df.empty or len(df) < 20:
            return {'index': 50, 'sentiment': 'NEUTRAL'}
        
        df = df.copy()
        
        # Components of Fear & Greed Index
        components = {}
        
        # 1. Volatility (25% weight)
        returns = df['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252)  # Annualized
        # High volatility = fear, low volatility = greed
        vol_score = max(0, min(100, 100 - (volatility * 200)))  # Invert
        components['volatility'] = vol_score
        
        # 2. Market Momentum (25% weight)
        momentum_10d = df['close'].pct_change(10).iloc[-1] * 100
        momentum_score = max(0, min(100, 50 + (momentum_10d * 10)))
        components['momentum'] = momentum_score
        
        # 3. Price Strength (25% weight)
        # How far from recent lows/highs
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        if recent_high != recent_low:
            price_position = (current_price - recent_low) / (recent_high - recent_low)
            strength_score = price_position * 100
        else:
            strength_score = 50
        components['strength'] = strength_score
        
        # 4. Volume Pattern (25% weight)
        # High volume on down moves = fear, high volume on up moves = greed
        volume_up = df[df['close'] > df['close'].shift(1)]['volume'].tail(10).mean()
        volume_down = df[df['close'] < df['close'].shift(1)]['volume'].tail(10).mean()
        
        if volume_up + volume_down > 0:
            volume_ratio = volume_up / (volume_up + volume_down)
            volume_score = volume_ratio * 100
        else:
            volume_score = 50
        components['volume'] = volume_score
        
        # Weighted average
        fear_greed_index = (
            components['volatility'] * 0.25 +
            components['momentum'] * 0.25 +
            components['strength'] * 0.25 +
            components['volume'] * 0.25
        )
        
        # Classify sentiment
        if fear_greed_index < 25:
            sentiment = 'EXTREME_FEAR'
            signal = 'BUY'  # Contrarian
        elif fear_greed_index < 45:
            sentiment = 'FEAR'
            signal = 'BUY'
        elif fear_greed_index < 55:
            sentiment = 'NEUTRAL'
            signal = 'NEUTRAL'
        elif fear_greed_index < 75:
            sentiment = 'GREED'
            signal = 'SELL'
        else:
            sentiment = 'EXTREME_GREED'
            signal = 'SELL'  # Contrarian
        
        return {
            'index': float(fear_greed_index),
            'sentiment': sentiment,
            'signal': signal,
            'components': components,
            'confidence': abs(fear_greed_index - 50) / 50  # Higher confidence at extremes
        }
    
    def detect_contrarian_opportunities(
        self,
        df: pd.DataFrame,
        fear_greed: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect contrarian trading opportunities.
        
        Wall Street Use: When everyone is fearful, be greedy. When everyone is greedy, be fearful.
        """
        if df.empty:
            return {}
        
        # Extreme sentiment = reversal opportunity
        index = fear_greed.get('index', 50)
        sentiment = fear_greed.get('sentiment', 'NEUTRAL')
        
        # Check for oversold/overbought conditions
        if 'rsi' in df.columns:
            rsi = df['rsi'].iloc[-1]
            oversold = rsi < 30
            overbought = rsi > 70
        else:
            oversold = False
            overbought = False
        
        # Contrarian signals
        buy_opportunity = (sentiment in ['EXTREME_FEAR', 'FEAR']) or oversold
        sell_opportunity = (sentiment in ['EXTREME_GREED', 'GREED']) or overbought
        
        # Price action confirmation
        recent_drop = df['close'].pct_change(5).iloc[-1] < -0.05  # 5% drop
        recent_surge = df['close'].pct_change(5).iloc[-1] > 0.05  # 5% surge
        
        contrarian_signal = 'NEUTRAL'
        if buy_opportunity and recent_drop:
            contrarian_signal = 'STRONG_BUY'
        elif buy_opportunity:
            contrarian_signal = 'BUY'
        elif sell_opportunity and recent_surge:
            contrarian_signal = 'STRONG_SELL'
        elif sell_opportunity:
            contrarian_signal = 'SELL'
        
        return {
            'contrarian_signal': contrarian_signal,
            'buy_opportunity': buy_opportunity,
            'sell_opportunity': sell_opportunity,
            'oversold': oversold,
            'overbought': overbought,
            'recent_drop': recent_drop,
            'recent_surge': recent_surge
        }
    
    def analyze_crowd_psychology(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze crowd psychology and herding behavior.
        
        Wall Street Use: Identify when retail is piling in (sell) vs when they're panicking (buy).
        """
        if df.empty or len(df) < 20:
            return {}
        
        # Herding indicators
        # 1. Volume surge on price moves (FOMO)
        price_change = df['close'].pct_change()
        volume_surge = volume > (volume.rolling(20).mean() * 1.5)
        
        fomo_events = (price_change > 0.02) & volume_surge  # 2%+ move with high volume
        panic_events = (price_change < -0.02) & volume_surge  # 2%+ drop with high volume
        
        # 2. Price acceleration (momentum chasing)
        momentum_change = price_change.diff()
        acceleration = momentum_change > 0.01  # Increasing momentum
        
        # 3. Volume-price divergence (smart money vs retail)
        price_trend = df['close'].tail(10).mean() > df['close'].tail(20).head(10).mean()
        volume_trend = volume.tail(10).mean() > volume.tail(20).head(10).mean()
        divergence = price_trend != volume_trend
        
        # Crowd psychology state
        if fomo_events.sum() > 2:
            psychology = 'FOMO'  # Fear of missing out - retail buying
            signal = 'SELL'  # Contrarian
        elif panic_events.sum() > 2:
            psychology = 'PANIC'  # Retail selling
            signal = 'BUY'  # Contrarian
        elif acceleration.sum() > 3:
            psychology = 'MOMENTUM_CHASING'
            signal = 'CAUTION'
        else:
            psychology = 'CALM'
            signal = 'NEUTRAL'
        
        return {
            'psychology': psychology,
            'signal': signal,
            'fomo_events': int(fomo_events.sum()),
            'panic_events': int(panic_events.sum()),
            'volume_price_divergence': bool(divergence),
            'herding_detected': fomo_events.sum() > 2 or panic_events.sum() > 2
        }
    
    def comprehensive_sentiment(
        self,
        df: pd.DataFrame,
        volume: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive sentiment analysis."""
        if df.empty:
            return {}
        
        if volume is None:
            volume = df.get('volume', pd.Series([0] * len(df)))
        
        # Fear & Greed
        fear_greed = self.calculate_fear_greed_index(df)
        
        # Contrarian opportunities
        contrarian = self.detect_contrarian_opportunities(df, fear_greed)
        
        # Crowd psychology
        psychology = self.analyze_crowd_psychology(df, volume)
        
        # Overall sentiment signal
        signals = []
        if fear_greed.get('signal') != 'NEUTRAL':
            signals.append(fear_greed['signal'])
        if contrarian.get('contrarian_signal') != 'NEUTRAL':
            signals.append(contrarian['contrarian_signal'])
        if psychology.get('signal') != 'NEUTRAL':
            signals.append(psychology['signal'])
        
        buy_count = signals.count('BUY') + signals.count('STRONG_BUY')
        sell_count = signals.count('SELL') + signals.count('STRONG_SELL')
        
        if buy_count > sell_count:
            overall_signal = 'BUY' if buy_count >= 2 else 'WEAK_BUY'
        elif sell_count > buy_count:
            overall_signal = 'SELL' if sell_count >= 2 else 'WEAK_SELL'
        else:
            overall_signal = 'NEUTRAL'
        
        return {
            'fear_greed': fear_greed,
            'contrarian': contrarian,
            'psychology': psychology,
            'overall_sentiment_signal': overall_signal,
            'sentiment_confidence': max(buy_count, sell_count) / len(signals) if signals else 0.0
        }

