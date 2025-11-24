"""
Advanced Machine Learning models for trading: XGBoost, LightGBM, Ensemble methods, LSTM.
Wall Street-grade prediction models.
"""
import logging
import pickle
from pathlib import Path
from typing import Dict, Any, Optional, List
import numpy as np
import pandas as pd
from sklearn.ensemble import (
    RandomForestClassifier, 
    GradientBoostingClassifier,
    VotingClassifier,
    StackingClassifier
)
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import lightgbm as lgb

# Deep learning imports
try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
    from tensorflow.keras.optimizers import Adam
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from utils.config import Config


class AdvancedMLModels:
    """Advanced ML models for sophisticated trading predictions."""
    
    def __init__(self, config: Config):
        """Initialize advanced ML models."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.advanced_ml")
        
        # Model storage
        self.models: Dict[str, Any] = {}
        self.scalers: Dict[str, Any] = {}
        self.model_dir = Path(config.model_directory)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Model types
        self.model_types = ['xgboost', 'lightgbm', 'ensemble', 'random_forest']
        if TENSORFLOW_AVAILABLE:
            self.model_types.append('lstm')
        self.current_model_type = 'ensemble'  # Default to ensemble
        
        # LSTM model storage
        self.lstm_model = None
        self.lstm_scaler = MinMaxScaler() if TENSORFLOW_AVAILABLE else None
        
    async def initialize(self):
        """Initialize all models."""
        self.logger.info("Initializing advanced ML models...")
        
        # Create models with optimized hyperparameters
        self.models['xgboost'] = xgb.XGBClassifier(
            n_estimators=500,  # Increased from 200
            max_depth=10,  # Increased from 8
            learning_rate=0.03,  # Lower for better generalization
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,  # Minimum loss reduction
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            eval_metric='logloss',
            n_jobs=-1
        )
        
        self.models['lightgbm'] = lgb.LGBMClassifier(
            n_estimators=500,  # Increased from 200
            max_depth=10,  # Increased from 8
            learning_rate=0.03,  # Lower for better generalization
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,  # L1 regularization
            reg_lambda=1.0,  # L2 regularization
            random_state=42,
            verbose=-1
        )
        
        self.models['random_forest'] = RandomForestClassifier(
            n_estimators=500,  # Increased from 200
            max_depth=20,  # Increased from 15
            min_samples_split=3,  # More aggressive
            min_samples_leaf=1,  # More aggressive
            max_features='sqrt',  # Feature sampling
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        
        # Add Gradient Boosting
        self.models['gradient_boosting'] = GradientBoostingClassifier(
            n_estimators=300,
            learning_rate=0.03,
            max_depth=8,
            min_samples_split=3,
            random_state=42
        )
        
        # Create enhanced ensemble model with stacking
        try:
            from sklearn.linear_model import LogisticRegression
            meta_learner = LogisticRegression(random_state=42, max_iter=1000)
            self.models['ensemble'] = StackingClassifier(
                estimators=[
                    ('xgb', self.models['xgboost']),
                    ('lgb', self.models['lightgbm']),
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting'])
                ],
                final_estimator=meta_learner,
                cv=5,  # 5-fold cross-validation
                n_jobs=-1
            )
        except Exception as e:
            # Fallback to voting if stacking fails
            self.logger.warning(f"Stacking failed: {e}, using voting ensemble")
            self.models['ensemble'] = VotingClassifier(
                estimators=[
                    ('xgb', self.models['xgboost']),
                    ('lgb', self.models['lightgbm']),
                    ('rf', self.models['random_forest']),
                    ('gb', self.models['gradient_boosting'])
                ],
                voting='soft',
                weights=[2, 2, 1, 1]  # Weight XGBoost and LightGBM more
            )
        
        # Initialize scalers with RobustScaler for better outlier handling
        for model_type in self.model_types:
            if model_type != 'lstm':
                self.scalers[model_type] = RobustScaler()
        
        # Initialize LSTM if available
        if TENSORFLOW_AVAILABLE and 'lstm' in self.model_types:
            self._create_lstm_model()
        
        self.logger.info("Advanced ML models initialized")
    
    def _create_lstm_model(self):
        """Create LSTM model for time series prediction."""
        if not TENSORFLOW_AVAILABLE:
            return
        
        try:
            # LSTM architecture for price prediction
            self.lstm_model = Sequential([
                Bidirectional(LSTM(50, return_sequences=True), input_shape=(60, 1)),
                Dropout(0.2),
                LSTM(50, return_sequences=False),
                Dropout(0.2),
                Dense(25),
                Dense(3, activation='softmax')  # BUY, SELL, HOLD
            ])
            
            self.lstm_model.compile(
                optimizer=Adam(learning_rate=0.001),
                loss='categorical_crossentropy',
                metrics=['accuracy']
            )
            
            self.logger.info("LSTM model created")
        except Exception as e:
            self.logger.warning(f"Could not create LSTM model: {e}")
            self.lstm_model = None
    
    async def train_models(
        self, 
        X: pd.DataFrame, 
        y: pd.Series,
        model_type: Optional[str] = None
    ):
        """
        Train models on historical data.
        
        Args:
            X: Feature matrix
            y: Target labels (1 for buy, 0 for sell/hold)
            model_type: Specific model to train, or None for all
        """
        model_types = [model_type] if model_type else self.model_types
        
        for mt in model_types:
            if mt not in self.models:
                continue
                
            self.logger.info(f"Training {mt} model...")
            
            # Scale features
            X_scaled = self.scalers[mt].fit_transform(X)
            
            # Train model
            self.models[mt].fit(X_scaled, y)
            
            # Evaluate
            scores = cross_val_score(
                self.models[mt], 
                X_scaled, 
                y, 
                cv=5, 
                scoring='accuracy'
            )
            self.logger.info(f"{mt} model accuracy: {scores.mean():.3f} (+/- {scores.std() * 2:.3f})")
            
            # Save model
            self._save_model(mt)
    
    async def predict(
        self, 
        df: pd.DataFrame,
        model_type: Optional[str] = None
    ) -> Dict[str, Dict[str, Any]]:
        """
        Generate predictions using advanced models.
        
        Args:
            df: DataFrame with features
            model_type: Model to use, or None for current default
            
        Returns:
            Dictionary of predictions with confidence scores
        """
        if not self.models:
            await self.initialize()
        
        model_type = model_type or self.current_model_type
        if model_type not in self.models:
            model_type = 'ensemble'
        
        predictions = {}
        
        try:
            feature_columns = [
                'rsi', 'macd', 'macd_signal', 'macd_hist',
                'sma_20', 'sma_50', 'ema_12', 'ema_26',
                'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
                'volume_sma', 'price_change', 'volume_ratio',
                'adx', 'stoch', 'williams_r', 'cci', 'atr'
            ]
            
            available_features = [col for col in feature_columns if col in df.columns]
            
            if not available_features:
                self.logger.warning("No features available for prediction")
                return predictions
            
            # Get ensemble predictions from multiple models
            ensemble_predictions = {}
            ensemble_confidences = {}
            
            for symbol in df['symbol'].unique():
                symbol_data = df[df['symbol'] == symbol]
                
                if len(symbol_data) == 0:
                    continue
                
                latest = symbol_data.iloc[-1]
                features = latest[available_features].values.reshape(1, -1)
                
                # Get predictions from all models
                model_predictions = []
                model_confidences = []
                
                # Traditional ML models
                for mt in ['xgboost', 'lightgbm', 'random_forest']:
                    if mt not in self.models:
                        continue
                    
                    try:
                        # Check if scaler is fitted
                        if not hasattr(self.scalers[mt], 'mean_') or self.scalers[mt].mean_ is None:
                            continue
                            
                        features_scaled = self.scalers[mt].transform(features)
                        pred = self.models[mt].predict(features_scaled)[0]
                        proba = self.models[mt].predict_proba(features_scaled)[0]
                        
                        model_predictions.append(pred)
                        model_confidences.append(max(proba))
                    except Exception as e:
                        self.logger.debug(f"Error with {mt} for {symbol}: {e}")
                
                # LSTM prediction (time series)
                if TENSORFLOW_AVAILABLE and self.lstm_model is not None and len(symbol_data) >= 60:
                    try:
                        lstm_pred = self._predict_lstm(symbol_data, available_features)
                        if lstm_pred:
                            model_predictions.append(lstm_pred['prediction'])
                            model_confidences.append(lstm_pred['confidence'])
                    except Exception as e:
                        self.logger.debug(f"Error with LSTM for {symbol}: {e}")
                
                # Ensemble decision
                if model_predictions:
                    avg_prediction = np.mean(model_predictions)
                    avg_confidence = np.mean(model_confidences)
                    
                    direction = 'BUY' if avg_prediction > 0.5 else 'SELL'
                    confidence = float(avg_confidence)
                    
                    predictions[symbol] = {
                        'direction': direction,
                        'confidence': confidence,
                        'prediction': int(avg_prediction > 0.5),
                        'model_type': model_type,
                        'ensemble_score': float(avg_prediction),
                        'model_count': len(model_predictions),  # How many models contributed
                    }
        
        except Exception as e:
            self.logger.error(f"Error in advanced prediction: {e}", exc_info=True)
        
        return predictions
    
    def _save_model(self, model_type: str):
        """Save a trained model."""
        model_path = self.model_dir / f"{model_type}_model.pkl"
        scaler_path = self.model_dir / f"{model_type}_scaler.pkl"
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.models[model_type], f)
        
        with open(scaler_path, 'wb') as f:
            pickle.dump(self.scalers[model_type], f)
        
        self.logger.info(f"Saved {model_type} model")
    
    async def load_models(self):
        """Load pre-trained models."""
        for model_type in self.model_types:
            model_path = self.model_dir / f"{model_type}_model.pkl"
            scaler_path = self.model_dir / f"{model_type}_scaler.pkl"
            
            if model_path.exists() and scaler_path.exists():
                try:
                    with open(model_path, 'rb') as f:
                        self.models[model_type] = pickle.load(f)
                    with open(scaler_path, 'rb') as f:
                        self.scalers[model_type] = pickle.load(f)
                    self.logger.info(f"Loaded {model_type} model")
                except Exception as e:
                    self.logger.warning(f"Failed to load {model_type}: {e}")
    
    def _predict_lstm(self, symbol_data: pd.DataFrame, feature_columns: List[str]) -> Optional[Dict[str, Any]]:
        """Predict using LSTM model."""
        if not TENSORFLOW_AVAILABLE or self.lstm_model is None:
            return None
        
        try:
            # Prepare time series data (last 60 periods)
            if len(symbol_data) < 60:
                return None
            
            # Use close price for LSTM
            price_data = symbol_data['close'].tail(60).values.reshape(-1, 1)
            
            # Scale data
            price_scaled = self.lstm_scaler.fit_transform(price_data)
            
            # Reshape for LSTM (samples, timesteps, features)
            X = price_scaled.reshape(1, 60, 1)
            
            # Predict
            prediction = self.lstm_model.predict(X, verbose=0)[0]
            
            # Convert to direction
            class_idx = np.argmax(prediction)
            directions = ['SELL', 'HOLD', 'BUY']
            direction = directions[class_idx]
            confidence = float(prediction[class_idx])
            
            return {
                'prediction': 1 if direction == 'BUY' else 0,
                'confidence': confidence,
                'direction': direction
            }
        except Exception as e:
            self.logger.debug(f"LSTM prediction error: {e}")
            return None
    
    def set_model_type(self, model_type: str):
        """Set the active model type."""
        if model_type in self.model_types:
            self.current_model_type = model_type
            self.logger.info(f"Switched to {model_type} model")

