"""
Machine Learning model for price prediction and trading signals.
"""
import warnings
# Suppress sklearn warnings about division by zero (expected when model not fitted)
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn')

import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler

from utils.config import Config


class MLModel:
    """Machine learning model for predicting stock price movements."""
    
    def __init__(self, config: Config):
        """Initialize ML model."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.ml_model")
        self.model: Optional[RandomForestClassifier] = None
        self.scaler: Optional[StandardScaler] = None
        self.model_path = Path(config.model_directory) / "trading_model.pkl"
        self.scaler_path = Path(config.model_directory) / "scaler.pkl"
        self.is_loaded = False
        
    async def load_model(self):
        """Load pre-trained model or create a new one."""
        if self.model_path.exists() and self.scaler_path.exists():
            try:
                with open(self.model_path, 'rb') as f:
                    self.model = pickle.load(f)
                with open(self.scaler_path, 'rb') as f:
                    self.scaler = pickle.load(f)
                self.is_loaded = True
                self.logger.info("Loaded pre-trained model")
            except Exception as e:
                self.logger.warning(f"Failed to load model: {e}. Creating new model.")
                await self._create_model()
        else:
            self.logger.info("No pre-trained model found. Creating new model.")
            await self._create_model()
    
    async def _create_model(self):
        """Create a new model with optimized parameters."""
        # Upgraded to use ensemble with better hyperparameters
        from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
        
        rf = RandomForestClassifier(
            n_estimators=300,  # Increased from 100
            max_depth=20,  # Increased from 10
            min_samples_split=3,  # More aggressive
            min_samples_leaf=1,  # More aggressive
            max_features='sqrt',  # Feature sampling
            random_state=42,
            n_jobs=-1
        )
        
        gb = GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )
        
        # Use voting ensemble for better accuracy
        self.model = VotingClassifier(
            estimators=[('rf', rf), ('gb', gb)],
            voting='soft'
        )
        
        # Use RobustScaler for better outlier handling
        from sklearn.preprocessing import RobustScaler
        self.scaler = RobustScaler()
        self.is_loaded = True
        self.logger.info("Created new upgraded ML model with ensemble")
    
    async def train(self, X: pd.DataFrame, y: pd.Series):
        """
        Train the model on historical data.
        
        Args:
            X: Feature matrix
            y: Target labels (1 for buy, 0 for sell/hold)
        """
        if not self.is_loaded:
            await self._create_model()
        
        self.logger.info("Training ML model...")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled, y)
        
        # Save model
        self._save_model()
        
        self.logger.info("Model training completed")
    
    def _save_model(self):
        """Save the trained model and scaler."""
        self.model_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(self.scaler_path, 'wb') as f:
            pickle.dump(self.scaler, f)
        
        self.logger.info(f"Model saved to {self.model_path}")
    
    async def predict(self, df: pd.DataFrame) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions for given data.
        
        Args:
            df: DataFrame with features (must include technical indicators)
            
        Returns:
            Dictionary of predictions keyed by symbol
        """
        if not self.is_loaded:
            await self.load_model()
        
        predictions = {}
        
        try:
            # Prepare features
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'volume_sma', 'price_change', 'volume_ratio'
            ]
            
            # Get available features
            available_features = [col for col in feature_columns if col in df.columns]
            
            if not available_features:
                self.logger.warning("No features available for prediction")
                return predictions
            
            # Group by symbol and predict
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                
                if len(symbol_data) == 0:
                    continue
                
                # Use the most recent row
                latest = symbol_data.iloc[-1]
                features = latest[available_features].values.reshape(1, -1)
                
                # Scale features - check if scaler is fitted
                try:
                    features_scaled = self.scaler.transform(features)
                except (AttributeError, ValueError):
                    # Scaler not fitted yet, use features as-is or fit with dummy data
                    self.logger.warning("Scaler not fitted, using raw features")
                    # Fit scaler with current features (minimal fit)
                    self.scaler.fit(features)
                    features_scaled = self.scaler.transform(features)
                
                # Check if model is fitted, if not, return neutral prediction
                try:
                    # Try to get feature names to check if model is fitted
                    _ = self.model.n_features_in_
                except AttributeError:
                    # Model not fitted, return neutral prediction
                    # Only log once to avoid spam
                    if not hasattr(self, '_model_not_fitted_logged'):
                        self.logger.warning("Model not fitted yet, returning neutral predictions (this is normal on first run)")
                        self._model_not_fitted_logged = True
                    predictions[symbol] = {
                        'direction': 'HOLD',
                        'confidence': 0.5,
                        'prediction': 0,
                    }
                    continue
                
                # Predict
                prediction = self.model.predict(features_scaled)[0]
                probabilities = self.model.predict_proba(features_scaled)[0]
                
                # Determine direction and confidence
                if prediction == 1:
                    direction = 'BUY'
                    confidence = float(probabilities[1])
                else:
                    direction = 'SELL'
                    confidence = float(probabilities[0])
                
                predictions[symbol] = {
                    'direction': direction,
                    'confidence': confidence,
                    'prediction': int(prediction),
                }
        
        except Exception as e:
            self.logger.error(f"Error in prediction: {e}", exc_info=True)
        
        return predictions

