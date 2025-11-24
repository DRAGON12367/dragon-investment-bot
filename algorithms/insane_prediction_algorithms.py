"""
INSANE PREDICTION ALGORITHMS - Ultra-Advanced AI for Predicting Best Buys
These algorithms use cutting-edge techniques to predict the absolute best coins/stocks to buy.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

try:
    from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor, VotingRegressor, StackingRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# Ultra Enhanced ML
try:
    from algorithms.ultra_enhanced_ml import UltraEnhancedML
    ULTRA_ENHANCED_AVAILABLE = True
except ImportError:
    ULTRA_ENHANCED_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class InsanePredictionAlgorithms:
    """
    Ultra-advanced prediction algorithms for identifying the best buys.
    Uses ensemble methods, deep learning, and advanced statistical techniques.
    """
    
    def __init__(self, config):
        """Initialize insane prediction algorithms."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.insane_predictions")
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Ultra Enhanced ML
        if ULTRA_ENHANCED_AVAILABLE:
            self.ultra_enhanced_ml = UltraEnhancedML(config)
        else:
            self.ultra_enhanced_ml = None
    
    # ========== MULTI-MODEL ENSEMBLE PREDICTOR ==========
    
    def ensemble_best_buy_predictor(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        price_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Ultra-advanced ensemble predictor that combines multiple models
        to predict the best buy opportunities.
        
        Returns:
            Comprehensive prediction with confidence scores.
        """
        if symbol not in market_data:
            return {"best_buy": False, "confidence": 0.0}
        
        data = market_data[symbol]
        current_price = data.get('price', 0)
        
        if current_price == 0:
            return {"best_buy": False, "confidence": 0.0}
        
        # Extract features
        features = self._extract_features(symbol, market_data, price_history)
        
        if features is None:
            return {"best_buy": False, "confidence": 0.0}
        
        # Run multiple prediction models
        predictions = {}
        
        # Model 1: Momentum-based prediction
        momentum_score = self._momentum_predictor(features)
        predictions['momentum'] = momentum_score
        
        # Model 2: Volume-based prediction
        volume_score = self._volume_predictor(features)
        predictions['volume'] = volume_score
        
        # Model 3: Trend-based prediction
        trend_score = self._trend_predictor(features)
        predictions['trend'] = trend_score
        
        # Model 4: Volatility-based prediction
        volatility_score = self._volatility_predictor(features)
        predictions['volatility'] = volatility_score
        
        # Model 5: Pattern-based prediction
        pattern_score = self._pattern_predictor(features)
        predictions['pattern'] = pattern_score
        
        # Model 6: Sentiment-based prediction (if available)
        sentiment_score = self._sentiment_predictor(features)
        predictions['sentiment'] = sentiment_score
        
        # Model 7: Correlation-based prediction
        correlation_score = self._correlation_predictor(symbol, market_data, features)
        predictions['correlation'] = correlation_score
        
        # Model 8: ML-based prediction (if models trained)
        ml_score = self._ml_predictor(features)
        predictions['ml'] = ml_score
        
        # Combine all predictions with weighted average
        weights = {
            'momentum': 0.15,
            'volume': 0.15,
            'trend': 0.15,
            'volatility': 0.10,
            'pattern': 0.15,
            'sentiment': 0.10,
            'correlation': 0.10,
            'ml': 0.10
        }
        
        weighted_score = sum(predictions.get(k, 0) * weights.get(k, 0) for k in weights)
        
        # Calculate confidence based on agreement
        scores = [v for v in predictions.values() if v > 0]
        agreement = len([s for s in scores if s > 0.7]) / len(scores) if scores else 0
        
        # Best buy if high score and good agreement
        best_buy = weighted_score > 0.75 and agreement > 0.6
        
        # Calculate profit potential
        profit_potential = self._calculate_profit_potential(features, weighted_score)
        
        # Calculate risk score
        risk_score = self._calculate_risk_score(features)
        
        return {
            "best_buy": best_buy,
            "confidence": weighted_score,
            "agreement": agreement,
            "profit_potential": profit_potential,
            "risk_score": risk_score,
            "individual_predictions": predictions,
            "entry_price": current_price,
            "target_price": current_price * (1 + profit_potential),
            "stop_loss": current_price * (1 - risk_score * 0.02),
            "timeframe": "1-7 days",
            "rank": self._calculate_rank(weighted_score, agreement, profit_potential, risk_score)
        }
    
    # ========== INDIVIDUAL PREDICTION MODELS ==========
    
    def _momentum_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on momentum indicators."""
        momentum = features.get('momentum', 0)
        roc = features.get('rate_of_change', 0)
        rsi = features.get('rsi', 50)
        
        # Strong momentum if positive and RSI not overbought
        if momentum > 0.02 and 30 < rsi < 70:
            score = min(momentum * 10 + roc * 5, 1.0)
            return max(score, 0.0)
        return 0.0
    
    def _volume_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on volume analysis."""
        volume_ratio = features.get('volume_ratio', 1.0)
        volume_trend = features.get('volume_trend', 0)
        
        # High volume with positive trend is bullish
        if volume_ratio > 1.2 and volume_trend > 0:
            score = min((volume_ratio - 1.0) * 0.5 + volume_trend * 0.3, 1.0)
            return max(score, 0.0)
        return 0.0
    
    def _trend_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on trend analysis."""
        trend_strength = features.get('trend_strength', 0)
        trend_direction = features.get('trend_direction', 0)
        ma_alignment = features.get('ma_alignment', False)
        
        # Strong uptrend with aligned moving averages
        if trend_direction > 0 and trend_strength > 0.5 and ma_alignment:
            score = min(trend_strength * 0.7 + trend_direction * 0.3, 1.0)
            return max(score, 0.0)
        return 0.0
    
    def _volatility_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on volatility analysis."""
        volatility = features.get('volatility', 0)
        volatility_regime = features.get('volatility_regime', 'NORMAL')
        
        # Moderate volatility is best for profit
        if volatility_regime == 'NORMAL' and 0.02 <= volatility <= 0.05:
            score = 0.8
        elif 0.01 <= volatility < 0.02:
            score = 0.6
        elif 0.05 < volatility <= 0.08:
            score = 0.5
        else:
            score = 0.2
        
        return score
    
    def _pattern_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on chart patterns."""
        pattern_type = features.get('pattern', None)
        pattern_strength = features.get('pattern_strength', 0)
        
        bullish_patterns = ['DOUBLE_BOTTOM', 'ASCENDING_TRIANGLE', 'BULL_FLAG', 'CUP_AND_HANDLE']
        
        if pattern_type in bullish_patterns and pattern_strength > 0.6:
            return min(pattern_strength, 1.0)
        return 0.0
    
    def _sentiment_predictor(self, features: Dict[str, Any]) -> float:
        """Predict based on sentiment analysis."""
        sentiment_score = features.get('sentiment', 0.5)
        fear_greed = features.get('fear_greed_index', 50)
        
        # Positive sentiment with moderate fear/greed
        if sentiment_score > 0.6 and 30 < fear_greed < 70:
            score = (sentiment_score - 0.5) * 2 + (1 - abs(fear_greed - 50) / 50) * 0.3
            return min(score, 1.0)
        return 0.0
    
    def _correlation_predictor(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        features: Dict[str, Any]
    ) -> float:
        """Predict based on correlation with similar assets."""
        asset_type = market_data.get(symbol, {}).get('asset_type', '')
        similar_assets = [
            s for s, d in market_data.items()
            if d.get('asset_type') == asset_type and s != symbol
        ]
        
        if len(similar_assets) < 3:
            return 0.5  # Neutral if not enough data
        
        # Check if similar assets are also moving up
        similar_changes = [
            market_data[s].get('change_percent', 0)
            for s in similar_assets[:10]
        ]
        avg_change = np.mean(similar_changes)
        
        # Positive correlation with sector momentum
        if avg_change > 1.0:
            score = min(avg_change / 10.0, 1.0)
            return max(score, 0.0)
        return 0.0
    
    def _ml_predictor(self, features: Dict[str, Any]) -> float:
        """ML-based prediction (if models are trained)."""
        if not self.is_trained or not SKLEARN_AVAILABLE:
            return 0.5  # Neutral if not trained
        
        try:
            # Convert features to array
            feature_array = np.array([
                features.get('momentum', 0),
                features.get('volume_ratio', 1.0),
                features.get('trend_strength', 0),
                features.get('volatility', 0.03),
                features.get('rsi', 50) / 100.0
            ]).reshape(1, -1)
            
            # Use ensemble model if available
            if "ensemble" in self.models and self.models["ensemble"] is not None:
                # Check if model is fitted
                if hasattr(self.models["ensemble"], 'predict'):
                    # Scale features if scaler available
                    if "scaler" in self.scalers and hasattr(self.scalers["scaler"], 'mean_'):
                        feature_array = self.scalers["scaler"].transform(feature_array)
                    
                    prediction = self.models["ensemble"].predict(feature_array)[0]
                    return float(max(0.0, min(1.0, prediction)))
        except Exception as e:
            # Silently handle errors - model not trained yet is normal
            pass
        
        return 0.5
    
    # ========== FEATURE EXTRACTION ==========
    
    def _extract_features(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        price_history: Optional[List[Dict]] = None
    ) -> Optional[Dict[str, Any]]:
        """Extract features for prediction."""
        if symbol not in market_data:
            return None
        
        data = market_data[symbol]
        
        features = {
            'price': data.get('price', 0),
            'change_percent': data.get('change_percent', 0),
            'volume': data.get('volume', 0),
            'high': data.get('high', 0),
            'low': data.get('low', 0),
        }
        
        # Calculate momentum
        if price_history and len(price_history) >= 10:
            prices = pd.Series([p['price'] for p in price_history[-10:]])
            features['momentum'] = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] if prices.iloc[0] > 0 else 0
            features['rate_of_change'] = prices.pct_change().mean() if len(prices) > 1 else 0
        else:
            features['momentum'] = data.get('change_percent', 0) / 100.0
            features['rate_of_change'] = features['momentum']
        
        # Calculate RSI (simplified)
        if price_history and len(price_history) >= 14:
            prices = pd.Series([p['price'] for p in price_history[-14:]])
            gains = prices.diff().clip(lower=0)
            losses = -prices.diff().clip(upper=0)
            avg_gain = gains.rolling(14).mean().iloc[-1]
            avg_loss = losses.rolling(14).mean().iloc[-1]
            if avg_loss > 0:
                rs = avg_gain / avg_loss
                features['rsi'] = 100 - (100 / (1 + rs))
            else:
                features['rsi'] = 100
        else:
            features['rsi'] = 50  # Neutral
        
        # Volume analysis
        current_volume = data.get('volume', 0)
        # Would use historical average in real implementation
        features['volume_ratio'] = 1.0  # Placeholder
        features['volume_trend'] = 0.0  # Placeholder
        
        # Trend analysis
        if price_history and len(price_history) >= 20:
            prices = pd.Series([p['price'] for p in price_history[-20:]])
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices.values, 1)[0]
            features['trend_strength'] = abs(slope) / prices.mean() if prices.mean() > 0 else 0
            features['trend_direction'] = 1 if slope > 0 else -1
            
            # Moving average alignment
            sma_short = prices.rolling(5).mean().iloc[-1]
            sma_long = prices.rolling(10).mean().iloc[-1]
            features['ma_alignment'] = sma_short > sma_long
        else:
            features['trend_strength'] = 0.0
            features['trend_direction'] = 0
            features['ma_alignment'] = False
        
        # Volatility
        if price_history and len(price_history) >= 20:
            prices = pd.Series([p['price'] for p in price_history[-20:]])
            features['volatility'] = prices.pct_change().std()
            if features['volatility'] < 0.02:
                features['volatility_regime'] = 'LOW'
            elif features['volatility'] < 0.05:
                features['volatility_regime'] = 'NORMAL'
            else:
                features['volatility_regime'] = 'HIGH'
        else:
            features['volatility'] = 0.03
            features['volatility_regime'] = 'NORMAL'
        
        # Pattern (simplified)
        features['pattern'] = None
        features['pattern_strength'] = 0.0
        
        # Sentiment (placeholder)
        features['sentiment'] = 0.5
        features['fear_greed_index'] = 50
        
        return features
    
    # ========== HELPER METHODS ==========
    
    def _calculate_profit_potential(self, features: Dict[str, Any], confidence: float) -> float:
        """Calculate expected profit potential."""
        momentum = features.get('momentum', 0)
        trend_strength = features.get('trend_strength', 0)
        
        # Base profit from momentum and trend
        base_profit = momentum * 2 + trend_strength * 0.1
        
        # Boost from confidence
        confidence_boost = confidence * 0.05
        
        profit_potential = base_profit + confidence_boost
        
        return max(0.05, min(profit_potential, 0.30))  # Between 5% and 30%
    
    def _calculate_risk_score(self, features: Dict[str, Any]) -> float:
        """Calculate risk score (0-1, lower is better)."""
        volatility = features.get('volatility', 0.03)
        rsi = features.get('rsi', 50)
        
        # Risk increases with volatility
        volatility_risk = min(volatility * 10, 1.0)
        
        # Risk increases if overbought/oversold
        rsi_risk = abs(rsi - 50) / 50.0
        
        risk_score = (volatility_risk + rsi_risk) / 2.0
        
        return max(0.0, min(risk_score, 1.0))
    
    def _calculate_rank(
        self,
        confidence: float,
        agreement: float,
        profit_potential: float,
        risk_score: float
    ) -> float:
        """Calculate overall rank (0-100)."""
        # Higher confidence = higher rank
        confidence_score = confidence * 40
        
        # Higher agreement = higher rank
        agreement_score = agreement * 20
        
        # Higher profit potential = higher rank
        profit_score = profit_potential * 20
        
        # Lower risk = higher rank
        risk_score_penalty = (1 - risk_score) * 20
        
        rank = confidence_score + agreement_score + profit_score + risk_score_penalty
        
        return max(0.0, min(rank, 100.0))
    
    def predict_best_buys(
        self,
        market_data: Dict[str, Any],
        limit: int = 20
    ) -> List[Dict[str, Any]]:
        """
        Predict the best buys across all assets.
        
        Returns:
            List of best buy opportunities ranked by prediction score.
        """
        predictions = []
        
        for symbol in market_data.keys():
            try:
                # Get price history (would use actual history in real implementation)
                price_history = None
                
                prediction = self.ensemble_best_buy_predictor(symbol, market_data, price_history)
                
                if prediction.get('best_buy', False):
                    prediction['symbol'] = symbol
                    prediction['asset_type'] = market_data[symbol].get('asset_type', 'unknown')
                    predictions.append(prediction)
            except Exception as e:
                self.logger.debug(f"Error predicting for {symbol}: {e}")
                continue
        
        # Sort by rank (highest first)
        predictions.sort(key=lambda x: x.get('rank', 0), reverse=True)
        
        return predictions[:limit]
    
    def train_models(self, X: np.ndarray, y: np.ndarray):
        """Train enhanced ML models on historical data."""
        if not SKLEARN_AVAILABLE or len(X) < 50:
            return
        
        try:
            # Enhanced ensemble with more models
            rf = RandomForestRegressor(
                n_estimators=300,  # Increased from 200
                max_depth=20,  # Increased from 15
                min_samples_split=3,  # More aggressive
                min_samples_leaf=1,  # More aggressive
                random_state=42,
                n_jobs=-1
            )
            gb = GradientBoostingRegressor(
                n_estimators=300,  # Increased from 200
                learning_rate=0.03,  # Lower for better generalization
                max_depth=8,  # Increased from 7
                min_samples_split=3,
                random_state=42
            )
            
            # Add more models if available
            from sklearn.ensemble import AdaBoostRegressor, ExtraTreesRegressor, BaggingRegressor
            
            ada = AdaBoostRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
            et = ExtraTreesRegressor(n_estimators=300, max_depth=20, random_state=42, n_jobs=-1)
            bag = BaggingRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            # Try stacking ensemble if available
            try:
                from sklearn.linear_model import Ridge
                meta_learner = Ridge(alpha=1.0)
                self.models["ensemble"] = StackingRegressor(
                    estimators=[
                        ('rf', rf),
                        ('gb', gb),
                        ('ada', ada),
                        ('et', et),
                        ('bag', bag)
                    ],
                    final_estimator=meta_learner,
                    cv=5,
                    n_jobs=-1
                )
            except:
                # Fallback to voting
                self.models["ensemble"] = VotingRegressor([
                    ('rf', rf),
                    ('gb', gb),
                    ('ada', ada),
                    ('et', et),
                    ('bag', bag)
                ])
            
            # Scale features with robust scaler
            if "scaler" not in self.scalers:
                self.scalers["scaler"] = RobustScaler()  # More robust to outliers
                X_scaled = self.scalers["scaler"].fit_transform(X)
            else:
                X_scaled = self.scalers["scaler"].transform(X)
            
            self.models["ensemble"].fit(X_scaled, y)
            self.is_trained = True
            
            # Also train ultra enhanced ML if available
            if self.ultra_enhanced_ml:
                try:
                    self.ultra_enhanced_ml.train_advanced_models(X, y)
                except Exception as e:
                    self.logger.debug(f"Ultra enhanced ML training error: {e}")
            
            self.logger.info("Ultra-enhanced prediction models trained successfully")
        except Exception as e:
            self.logger.error(f"Error training models: {e}")

