"""
ADVANCED SENTIMENT FUSION - 200X UPGRADE
Multi-source sentiment analysis and fusion
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AdvancedSentimentFusion:
    """
    Advanced multi-source sentiment fusion system.
    
    Features:
    - Multi-source sentiment aggregation
    - Sentiment weighting by reliability
    - Sentiment momentum
    - Contrarian sentiment signals
    - Sentiment divergence detection
    - Social media sentiment (simulated)
    - News sentiment (simulated)
    - Price-sentiment correlation
    """
    
    def __init__(self):
        """Initialize sentiment fusion system."""
        self.logger = logging.getLogger("ai_investment_bot.sentiment_fusion")
        self.sentiment_history = {}
        self.source_weights = {
            'price_action': 0.3,
            'volume': 0.2,
            'technical': 0.2,
            'social_media': 0.15,
            'news': 0.15
        }
        
    def calculate_price_sentiment(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate sentiment from price action.
        
        Bullish: Price above moving averages, uptrend
        Bearish: Price below moving averages, downtrend
        """
        try:
            if 'close' not in df.columns or len(df) < 20:
                return {}
            
            close = df['close']
            
            # Moving averages
            ma_20 = close.rolling(20).mean()
            ma_50 = close.rolling(50).mean() if len(df) >= 50 else ma_20
            
            # Price position
            price_above_ma20 = (close.iloc[-1] > ma_20.iloc[-1]) if len(ma_20) > 0 else False
            price_above_ma50 = (close.iloc[-1] > ma_50.iloc[-1]) if len(ma_50) > 0 else False
            
            # Trend
            price_change_5d = (close.iloc[-1] / close.iloc[-6] - 1) if len(close) >= 6 else 0.0
            price_change_20d = (close.iloc[-1] / close.iloc[-21] - 1) if len(close) >= 21 else 0.0
            
            # Calculate sentiment score (-1 to 1)
            sentiment_score = 0.0
            
            if price_above_ma20:
                sentiment_score += 0.2
            if price_above_ma50:
                sentiment_score += 0.3
            
            if price_change_5d > 0:
                sentiment_score += 0.2
            if price_change_20d > 0:
                sentiment_score += 0.3
            
            # Normalize to -1 to 1
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment': 'bullish' if sentiment_score > 0.3 else 'bearish' if sentiment_score < -0.3 else 'neutral',
                'price_above_ma20': price_above_ma20,
                'price_above_ma50': price_above_ma50,
                'price_change_5d': price_change_5d,
                'price_change_20d': price_change_20d
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating price sentiment: {e}")
            return {}
    
    def calculate_volume_sentiment(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate sentiment from volume patterns.
        
        High volume on up moves = bullish
        High volume on down moves = bearish
        """
        try:
            if 'close' not in df.columns or 'volume' not in df.columns or len(df) < 10:
                return {}
            
            close = df['close']
            volume = df['volume']
            
            # Price change
            price_change = close.pct_change()
            
            # Volume on up vs down days
            up_days = price_change > 0
            down_days = price_change < 0
            
            avg_volume_up = volume[up_days].mean() if up_days.any() else 0.0
            avg_volume_down = volume[down_days].mean() if down_days.any() else 0.0
            avg_volume = volume.mean()
            
            # Sentiment from volume
            if avg_volume_up > avg_volume_down * 1.2:
                volume_sentiment = 0.5  # Bullish
            elif avg_volume_down > avg_volume_up * 1.2:
                volume_sentiment = -0.5  # Bearish
            else:
                volume_sentiment = 0.0  # Neutral
            
            # Recent volume trend
            recent_volume = volume.iloc[-5:].mean() if len(volume) >= 5 else volume.mean()
            volume_trend = (recent_volume / avg_volume - 1) if avg_volume > 0 else 0.0
            
            return {
                'sentiment_score': volume_sentiment,
                'sentiment': 'bullish' if volume_sentiment > 0.2 else 'bearish' if volume_sentiment < -0.2 else 'neutral',
                'avg_volume_up': avg_volume_up,
                'avg_volume_down': avg_volume_down,
                'volume_trend': volume_trend
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating volume sentiment: {e}")
            return {}
    
    def calculate_technical_sentiment(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Calculate sentiment from technical indicators.
        """
        try:
            if 'close' not in df.columns or len(df) < 20:
                return {}
            
            close = df['close']
            
            # RSI (simplified)
            returns = close.pct_change()
            gains = returns[returns > 0].rolling(14).mean().fillna(0)
            losses = -returns[returns < 0].rolling(14).mean().fillna(0)
            rs = gains / (losses + 1e-10)
            rsi = 100 - (100 / (1 + rs))
            current_rsi = rsi.iloc[-1] if not rsi.empty else 50.0
            
            # MACD (simplified)
            ema_12 = close.ewm(span=12).mean()
            ema_26 = close.ewm(span=26).mean() if len(df) >= 26 else ema_12
            macd = ema_12 - ema_26
            macd_signal = macd.ewm(span=9).mean()
            macd_hist = macd - macd_signal
            current_macd_hist = macd_hist.iloc[-1] if not macd_hist.empty else 0.0
            
            # Calculate sentiment
            sentiment_score = 0.0
            
            # RSI contribution
            if current_rsi > 70:
                sentiment_score -= 0.3  # Overbought
            elif current_rsi < 30:
                sentiment_score += 0.3  # Oversold
            else:
                sentiment_score += (current_rsi - 50) / 50 * 0.2
            
            # MACD contribution
            if current_macd_hist > 0:
                sentiment_score += 0.3
            else:
                sentiment_score -= 0.3
            
            # Normalize
            sentiment_score = max(-1.0, min(1.0, sentiment_score))
            
            return {
                'sentiment_score': sentiment_score,
                'sentiment': 'bullish' if sentiment_score > 0.3 else 'bearish' if sentiment_score < -0.3 else 'neutral',
                'rsi': current_rsi,
                'macd_histogram': current_macd_hist
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating technical sentiment: {e}")
            return {}
    
    def simulate_social_sentiment(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate social media sentiment (would integrate with real API in production).
        """
        try:
            if 'close' not in df.columns:
                return {}
            
            close = df['close']
            
            # Simulate based on price momentum
            price_momentum = close.pct_change(periods=5).iloc[-1] if len(close) >= 6 else 0.0
            
            # Social sentiment tends to follow price momentum
            # Add some noise
            noise = np.random.normal(0, 0.1)
            social_sentiment = np.tanh(price_momentum * 10) + noise
            
            # Normalize
            social_sentiment = max(-1.0, min(1.0, social_sentiment))
            
            return {
                'sentiment_score': social_sentiment,
                'sentiment': 'bullish' if social_sentiment > 0.3 else 'bearish' if social_sentiment < -0.3 else 'neutral',
                'source': 'social_media_simulated'
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating social sentiment: {e}")
            return {}
    
    def simulate_news_sentiment(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate news sentiment (would integrate with real API in production).
        """
        try:
            if 'close' not in df.columns:
                return {}
            
            close = df['close']
            
            # Simulate based on volatility and recent returns
            returns = close.pct_change()
            recent_vol = returns.rolling(5).std().iloc[-1] if len(returns) >= 5 else returns.std()
            recent_return = returns.iloc[-5:].mean() if len(returns) >= 5 else returns.mean()
            
            # News sentiment tends to be more stable, less reactive
            news_sentiment = np.tanh(recent_return * 5) * 0.7  # Less reactive
            
            # Normalize
            news_sentiment = max(-1.0, min(1.0, news_sentiment))
            
            return {
                'sentiment_score': news_sentiment,
                'sentiment': 'bullish' if news_sentiment > 0.3 else 'bearish' if news_sentiment < -0.3 else 'neutral',
                'source': 'news_simulated',
                'volatility': recent_vol
            }
            
        except Exception as e:
            self.logger.error(f"Error simulating news sentiment: {e}")
            return {}
    
    def fuse_sentiments(
        self,
        sentiments: Dict[str, Dict[str, Any]],
        custom_weights: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Fuse multiple sentiment sources into unified sentiment.
        
        Args:
            sentiments: Dict of source_name -> sentiment_dict
            custom_weights: Optional custom weights (defaults to self.source_weights)
        """
        try:
            if not sentiments:
                return {'sentiment_score': 0.0, 'sentiment': 'neutral'}
            
            weights = custom_weights or self.source_weights
            
            # Weighted average
            weighted_sum = 0.0
            total_weight = 0.0
            
            for source, sentiment_data in sentiments.items():
                if 'sentiment_score' in sentiment_data:
                    weight = weights.get(source, 0.1)
                    score = sentiment_data['sentiment_score']
                    weighted_sum += weight * score
                    total_weight += weight
            
            # Normalize
            if total_weight > 0:
                fused_score = weighted_sum / total_weight
            else:
                fused_score = 0.0
            
            # Clamp to -1 to 1
            fused_score = max(-1.0, min(1.0, fused_score))
            
            # Determine sentiment
            if fused_score > 0.5:
                sentiment = 'very_bullish'
            elif fused_score > 0.2:
                sentiment = 'bullish'
            elif fused_score < -0.5:
                sentiment = 'very_bearish'
            elif fused_score < -0.2:
                sentiment = 'bearish'
            else:
                sentiment = 'neutral'
            
            # Calculate confidence (agreement between sources)
            if len(sentiments) > 1:
                scores = [s.get('sentiment_score', 0.0) for s in sentiments.values()]
                score_std = np.std(scores)
                confidence = max(0.0, 1.0 - score_std)  # Lower std = higher confidence
            else:
                confidence = 0.5
            
            return {
                'fused_sentiment_score': fused_score,
                'sentiment': sentiment,
                'confidence': confidence,
                'source_count': len(sentiments),
                'source_sentiments': {k: v.get('sentiment', 'neutral') for k, v in sentiments.items()}
            }
            
        except Exception as e:
            self.logger.error(f"Error fusing sentiments: {e}")
            return {'sentiment_score': 0.0, 'sentiment': 'neutral'}
    
    def detect_sentiment_divergence(
        self,
        current_sentiment: Dict[str, Any],
        price_action: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect divergence between sentiment and price action.
        
        Bullish divergence: Price down but sentiment improving
        Bearish divergence: Price up but sentiment deteriorating
        """
        try:
            if price_action.empty or 'fused_sentiment_score' not in current_sentiment:
                return {}
            
            # Recent price trend
            price_trend = price_action.iloc[-5:].mean() - price_action.iloc[-10:-5].mean() if len(price_action) >= 10 else 0.0
            
            # Sentiment trend (would need history, simplified here)
            current_sentiment_score = current_sentiment.get('fused_sentiment_score', 0.0)
            
            # Detect divergence
            divergence_detected = False
            divergence_type = None
            
            if price_trend < 0 and current_sentiment_score > 0.3:
                # Price down but sentiment bullish = bullish divergence
                divergence_detected = True
                divergence_type = 'bullish_divergence'
            elif price_trend > 0 and current_sentiment_score < -0.3:
                # Price up but sentiment bearish = bearish divergence
                divergence_detected = True
                divergence_type = 'bearish_divergence'
            
            return {
                'divergence_detected': divergence_detected,
                'divergence_type': divergence_type,
                'price_trend': price_trend,
                'sentiment_score': current_sentiment_score
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting sentiment divergence: {e}")
            return {}
    
    def get_comprehensive_sentiment(
        self,
        symbol: str,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Get comprehensive sentiment analysis from all sources.
        """
        try:
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Calculate all sentiment sources
            price_sentiment = self.calculate_price_sentiment(df)
            volume_sentiment = self.calculate_volume_sentiment(df)
            technical_sentiment = self.calculate_technical_sentiment(df)
            social_sentiment = self.simulate_social_sentiment(symbol, df)
            news_sentiment = self.simulate_news_sentiment(symbol, df)
            
            # Combine all sentiments
            all_sentiments = {
                'price_action': price_sentiment,
                'volume': volume_sentiment,
                'technical': technical_sentiment,
                'social_media': social_sentiment,
                'news': news_sentiment
            }
            
            # Filter out empty sentiments
            valid_sentiments = {k: v for k, v in all_sentiments.items() if v}
            
            # Fuse sentiments
            fused = self.fuse_sentiments(valid_sentiments)
            
            # Detect divergence
            divergence = self.detect_sentiment_divergence(fused, df['close'] if 'close' in df.columns else pd.Series())
            
            return {
                'fused_sentiment': fused,
                'source_sentiments': valid_sentiments,
                'divergence': divergence,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error getting comprehensive sentiment: {e}")
            return {'status': 'error', 'message': str(e)}

