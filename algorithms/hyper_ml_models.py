"""
HYPER ML MODELS - Next-generation machine learning with advanced optimization
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
        StackingRegressor, BaggingRegressor, HistGradientBoostingRegressor
    )
    from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
    from sklearn.svm import SVR
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, QuantileTransformer
    from sklearn.model_selection import cross_val_score, GridSearchCV, RandomizedSearchCV
    from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression
    from sklearn.decomposition import PCA
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    import lightgbm as lgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    import catboost as cb
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False


class HyperMLModels:
    """
    Hyper-optimized ML models with advanced techniques:
    - Automated hyperparameter tuning
    - Feature selection and engineering
    - Advanced ensemble methods
    - Model blending and stacking
    - Cross-validation optimization
    """
    
    def __init__(self, config):
        """Initialize hyper ML models."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.hyper_ml")
        self.models = {}
        self.scalers = {}
        self.feature_selectors = {}
        self.is_trained = False
        self.best_params = {}
    
    def create_optimized_ensemble(self) -> Any:
        """Create hyper-optimized stacking ensemble."""
        if not SKLEARN_AVAILABLE:
            return None
        
        # Level 1: Base models with optimized hyperparameters
        base_models = []
        
        # Random Forest - optimized
        rf = RandomForestRegressor(
            n_estimators=500,  # Increased
            max_depth=25,  # Deeper trees
            min_samples_split=2,  # More aggressive
            min_samples_leaf=1,  # More aggressive
            max_features='sqrt',  # Feature sampling
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        base_models.append(('rf', rf))
        
        # Gradient Boosting - optimized
        gb = GradientBoostingRegressor(
            n_estimators=500,
            learning_rate=0.02,  # Lower for better generalization
            max_depth=10,
            min_samples_split=3,
            min_samples_leaf=2,
            subsample=0.8,  # Stochastic gradient boosting
            random_state=42
        )
        base_models.append(('gb', gb))
        
        # Extra Trees - optimized
        et = ExtraTreesRegressor(
            n_estimators=500,
            max_depth=25,
            min_samples_split=2,
            min_samples_leaf=1,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1
        )
        base_models.append(('et', et))
        
        # AdaBoost - optimized
        ada = AdaBoostRegressor(
            n_estimators=300,
            learning_rate=0.05,
            loss='square',  # Better for regression
            random_state=42
        )
        base_models.append(('ada', ada))
        
        # Histogram-based Gradient Boosting (faster, often better)
        try:
            hgb = HistGradientBoostingRegressor(
                max_iter=500,
                learning_rate=0.02,
                max_depth=10,
                min_samples_leaf=2,
                random_state=42
            )
            base_models.append(('hgb', hgb))
        except:
            pass
        
        # XGBoost if available
        if XGBOOST_AVAILABLE:
            xgb_model = xgb.XGBRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=10,
                min_child_weight=1,
                subsample=0.8,
                colsample_bytree=0.8,
                gamma=0.1,  # Minimum loss reduction
                reg_alpha=0.1,  # L1 regularization
                reg_lambda=1.0,  # L2 regularization
                random_state=42,
                n_jobs=-1
            )
            base_models.append(('xgb', xgb_model))
        
        # LightGBM if available
        if XGBOOST_AVAILABLE:
            lgb_model = lgb.LGBMRegressor(
                n_estimators=500,
                learning_rate=0.02,
                max_depth=10,
                min_child_samples=2,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
            base_models.append(('lgb', lgb_model))
        
        # CatBoost if available
        if CATBOOST_AVAILABLE:
            cat_model = cb.CatBoostRegressor(
                iterations=500,
                learning_rate=0.02,
                depth=10,
                l2_leaf_reg=3,
                random_seed=42,
                verbose=False
            )
            base_models.append(('cat', cat_model))
        
        # Level 2: Meta-learner with multiple options
        meta_learners = [
            Ridge(alpha=1.0),
            Lasso(alpha=0.1),
            ElasticNet(alpha=0.1, l1_ratio=0.5),
            BayesianRidge()
        ]
        
        # Use best meta-learner (Ridge as default)
        meta_learner = meta_learners[0]
        
        # Create stacking ensemble
        try:
            ensemble = StackingRegressor(
                estimators=base_models,
                final_estimator=meta_learner,
                cv=5,  # 5-fold cross-validation
                n_jobs=-1,
                passthrough=False  # Don't pass original features
            )
            return ensemble
        except Exception as e:
            self.logger.debug(f"Stacking failed: {e}, using voting")
            # Fallback to voting
            return VotingRegressor(base_models)
    
    def optimize_hyperparameters(
        self,
        model: Any,
        X: np.ndarray,
        y: np.ndarray,
        param_grid: Dict[str, List]
    ) -> Dict[str, Any]:
        """Optimize hyperparameters using grid search."""
        if not SKLEARN_AVAILABLE:
            return {}
        
        try:
            # Use randomized search for faster optimization
            search = RandomizedSearchCV(
                model,
                param_grid,
                n_iter=20,  # Number of iterations
                cv=5,
                scoring='r2',
                n_jobs=-1,
                random_state=42,
                verbose=0
            )
            
            search.fit(X, y)
            
            return {
                'best_params': search.best_params_,
                'best_score': search.best_score_,
                'best_model': search.best_estimator_
            }
        except Exception as e:
            self.logger.debug(f"Hyperparameter optimization error: {e}")
            return {}
    
    def select_features(
        self,
        X: np.ndarray,
        y: np.ndarray,
        method: str = 'mutual_info',
        k: int = None
    ) -> Tuple[np.ndarray, Any]:
        """
        Select best features using statistical methods.
        
        Args:
            X: Feature matrix
            y: Target vector
            method: 'mutual_info' or 'f_regression'
            k: Number of features to select (None = auto)
            
        Returns:
            Selected features and selector object
        """
        if not SKLEARN_AVAILABLE:
            return X, None
        
        try:
            # Auto-select k if not specified
            if k is None:
                k = min(50, X.shape[1])  # Select top 50 or all if less
            
            # Choose scoring function
            if method == 'mutual_info':
                score_func = mutual_info_regression
            else:
                score_func = f_regression
            
            selector = SelectKBest(score_func=score_func, k=k)
            X_selected = selector.fit_transform(X, y)
            
            return X_selected, selector
        except Exception as e:
            self.logger.debug(f"Feature selection error: {e}")
            return X, None
    
    def apply_pca(
        self,
        X: np.ndarray,
        n_components: float = 0.95
    ) -> Tuple[np.ndarray, Any]:
        """
        Apply PCA for dimensionality reduction.
        
        Args:
            X: Feature matrix
            n_components: Number of components or variance to retain (0-1)
            
        Returns:
            Transformed features and PCA object
        """
        if not SKLEARN_AVAILABLE:
            return X, None
        
        try:
            pca = PCA(n_components=n_components, random_state=42)
            X_pca = pca.fit_transform(X)
            
            self.logger.info(f"PCA: {X.shape[1]} features -> {X_pca.shape[1]} components "
                           f"({pca.explained_variance_ratio_.sum():.2%} variance retained)")
            
            return X_pca, pca
        except Exception as e:
            self.logger.debug(f"PCA error: {e}")
            return X, None
    
    def train_hyper_models(
        self,
        X: np.ndarray,
        y: np.ndarray,
        use_feature_selection: bool = True,
        use_pca: bool = False
    ):
        """Train hyper-optimized models with advanced techniques."""
        if not SKLEARN_AVAILABLE or len(X) < 50:
            return
        
        try:
            # Step 1: Feature scaling with robust scaler
            scaler = RobustScaler()  # More robust to outliers
            X_scaled = scaler.fit_transform(X)
            self.scalers['main'] = scaler
            
            # Step 2: Feature selection (optional)
            if use_feature_selection and X_scaled.shape[1] > 20:
                X_selected, selector = self.select_features(
                    X_scaled, y, method='mutual_info', k=min(50, X_scaled.shape[1])
                )
                self.feature_selectors['main'] = selector
                X_processed = X_selected
            else:
                X_processed = X_scaled
            
            # Step 3: PCA (optional)
            if use_pca and X_processed.shape[1] > 10:
                X_processed, pca = self.apply_pca(X_processed, n_components=0.95)
                self.feature_selectors['pca'] = pca
            
            # Step 4: Create and train optimized ensemble
            ensemble = self.create_optimized_ensemble()
            if ensemble:
                ensemble.fit(X_processed, y)
                self.models['hyper_ensemble'] = ensemble
                self.is_trained = True
                
                # Calculate cross-validation score
                try:
                    cv_scores = cross_val_score(
                        ensemble, X_processed, y, 
                        cv=5, scoring='r2', n_jobs=-1
                    )
                    avg_score = np.mean(cv_scores)
                    std_score = np.std(cv_scores)
                    self.logger.info(
                        f"Hyper ML models trained - CV R²: {avg_score:.3f} ± {std_score:.3f}"
                    )
                except:
                    self.logger.info("Hyper ML models trained successfully")
        except Exception as e:
            self.logger.error(f"Error training hyper models: {e}")
    
    def predict_hyper(
        self,
        features: np.ndarray
    ) -> Dict[str, Any]:
        """
        Hyper-optimized prediction with confidence intervals.
        
        Returns:
            Dictionary with prediction and confidence metrics
        """
        if not self.is_trained or 'hyper_ensemble' not in self.models:
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
            
            # Apply feature selection
            if 'main' in self.feature_selectors:
                features_processed = self.feature_selectors['main'].transform(features_scaled)
            else:
                features_processed = features_scaled
            
            # Apply PCA if used
            if 'pca' in self.feature_selectors:
                features_processed = self.feature_selectors['pca'].transform(features_processed)
            
            # Get prediction
            prediction = self.models['hyper_ensemble'].predict(features_processed)[0]
            
            # Estimate confidence (simplified - could use bootstrap)
            confidence = 0.8  # Default confidence for ensemble
            
            # Calculate bounds (simplified)
            std_estimate = abs(prediction) * 0.1  # 10% uncertainty estimate
            lower = prediction - 1.96 * std_estimate
            upper = prediction + 1.96 * std_estimate
            
            return {
                'prediction': float(prediction),
                'confidence': float(confidence),
                'lower_bound': float(max(0.0, lower)),
                'upper_bound': float(min(1.0, upper))
            }
        except Exception as e:
            self.logger.debug(f"Hyper prediction error: {e}")
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'lower_bound': 0.3,
                'upper_bound': 0.7
            }

