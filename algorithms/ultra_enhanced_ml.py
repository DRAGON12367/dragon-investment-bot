"""
ULTRA ENHANCED ML MODELS - Next-generation machine learning for trading
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

try:
    from sklearn.ensemble import (
        RandomForestRegressor, GradientBoostingRegressor, 
        AdaBoostRegressor, ExtraTreesRegressor, VotingRegressor,
        StackingRegressor, BaggingRegressor
    )
    from sklearn.neural_network import MLPRegressor
    from sklearn.svm import SVR
    from sklearn.linear_model import Ridge, Lasso, ElasticNet
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


class UltraEnhancedML:
    """
    Ultra-enhanced ML models with advanced ensemble techniques.
    """
    
    def __init__(self, config):
        """Initialize ultra-enhanced ML models."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.ultra_enhanced_ml")
        self.models = {}
        self.scalers = {}
        self.is_trained = False
    
    def create_advanced_ensemble(self) -> Any:
        """Create advanced stacking ensemble with multiple layers."""
        if not SKLEARN_AVAILABLE:
            return None
        
        # Base models (Level 1)
        base_models = [
            ('rf', RandomForestRegressor(
                n_estimators=300,
                max_depth=20,
                min_samples_split=3,
                min_samples_leaf=1,
                random_state=42,
                n_jobs=-1
            )),
            ('gb', GradientBoostingRegressor(
                n_estimators=300,
                learning_rate=0.03,
                max_depth=8,
                min_samples_split=3,
                random_state=42
            )),
            ('ada', AdaBoostRegressor(
                n_estimators=200,
                learning_rate=0.1,
                random_state=42
            )),
            ('et', ExtraTreesRegressor(
                n_estimators=300,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            ))
        ]
        
        # Add XGBoost and LightGBM if available
        if XGBOOST_AVAILABLE:
            base_models.append(('xgb', xgb.XGBRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                n_jobs=-1
            )))
            base_models.append(('lgb', lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                max_depth=8,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )))
        
        # Meta-learner (Level 2)
        meta_learner = Ridge(alpha=1.0)
        
        # Create stacking ensemble
        try:
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,
                n_jobs=-1
            )
            return ensemble
        except:
            # Fallback to voting if stacking fails
            return VotingRegressor(base_models)
    
    def predict_with_confidence_intervals(
        self,
        model: Any,
        X: np.ndarray,
        n_bootstrap: int = 100
    ) -> Dict[str, Any]:
        """
        Predict with confidence intervals using bootstrap.
        
        Returns:
            Dictionary with prediction, lower bound, upper bound, confidence
        """
        if not SKLEARN_AVAILABLE or model is None:
            return {
                'prediction': 0.5,
                'lower': 0.3,
                'upper': 0.7,
                'confidence': 0.5
            }
        
        try:
            # Bootstrap predictions
            predictions = []
            for _ in range(n_bootstrap):
                # Sample with replacement
                indices = np.random.choice(len(X), size=len(X), replace=True)
                X_sample = X[indices]
                pred = model.predict(X_sample)
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate statistics
            mean_pred = np.mean(predictions, axis=0)
            std_pred = np.std(predictions, axis=0)
            
            # Confidence intervals (95%)
            lower = mean_pred - 1.96 * std_pred
            upper = mean_pred + 1.96 * std_pred
            
            # Confidence score (inverse of interval width)
            interval_width = upper - lower
            confidence = 1.0 / (1.0 + interval_width)
            confidence = np.clip(confidence, 0.0, 1.0)
            
            return {
                'prediction': float(mean_pred[0]) if len(mean_pred) > 0 else 0.5,
                'lower': float(lower[0]) if len(lower) > 0 else 0.3,
                'upper': float(upper[0]) if len(upper) > 0 else 0.7,
                'confidence': float(confidence[0]) if len(confidence) > 0 else 0.5
            }
        except Exception as e:
            self.logger.debug(f"Bootstrap prediction error: {e}")
            # Fallback to simple prediction
            try:
                pred = model.predict(X)
                return {
                    'prediction': float(pred[0]) if len(pred) > 0 else 0.5,
                    'lower': float(pred[0] * 0.9) if len(pred) > 0 else 0.3,
                    'upper': float(pred[0] * 1.1) if len(pred) > 0 else 0.7,
                    'confidence': 0.7
                }
            except:
                return {
                    'prediction': 0.5,
                    'lower': 0.3,
                    'upper': 0.7,
                    'confidence': 0.5
                }
    
    def train_advanced_models(self, X: np.ndarray, y: np.ndarray):
        """Train advanced ensemble models."""
        if not SKLEARN_AVAILABLE or len(X) < 50:
            return
        
        try:
            # Scale features
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
            
            # Create and train ensemble
            ensemble = self.create_advanced_ensemble()
            if ensemble:
                ensemble.fit(X_scaled, y)
                self.models['advanced_ensemble'] = ensemble
                self.is_trained = True
                
                # Calculate cross-validation score
                try:
                    cv_scores = cross_val_score(ensemble, X_scaled, y, cv=5, scoring='r2')
                    avg_score = np.mean(cv_scores)
                    self.logger.info(f"Ultra-enhanced ML models trained - CV R²: {avg_score:.3f}")
                except:
                    self.logger.info("Ultra-enhanced ML models trained successfully")
        except Exception as e:
            self.logger.error(f"Error training ultra-enhanced models: {e}")
    
    def predict_enhanced(
        self,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Enhanced prediction with confidence intervals.
        
        Returns:
            Dictionary with prediction and confidence metrics
        """
        if not self.is_trained or 'advanced_ensemble' not in self.models:
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'lower_bound': 0.3,
                'upper_bound': 0.7
            }
        
        try:
            # Scale features
            if 'main' in self.scalers:
                features_scaled = self.scalers['main'].transform(features)
            else:
                features_scaled = features
            
            # Get prediction with confidence intervals
            result = self.predict_with_confidence_intervals(
                self.models['advanced_ensemble'],
                features_scaled
            )
            
            return {
                'prediction': result['prediction'],
                'confidence': result['confidence'],
                'lower_bound': result['lower'],
                'upper_bound': result['upper']
            }
        except Exception as e:
            self.logger.debug(f"Enhanced prediction error: {e}")
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'lower_bound': 0.3,
                'upper_bound': 0.7
            }

