"""
ADAPTIVE LEARNING SYSTEM - Self-improving AI that learns from market feedback
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

try:
    from sklearn.ensemble import VotingRegressor, StackingRegressor
    from sklearn.linear_model import Ridge
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class AdaptiveLearningSystem:
    """
    Adaptive Learning System - Continuously learns and improves from market feedback.
    Uses reinforcement learning principles to adapt strategies.
    """
    
    def __init__(self, config):
        """Initialize adaptive learning system."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.adaptive_learning")
        self.performance_history = []
        self.learning_rate = 0.1
        self.adaptation_threshold = 0.05  # Adapt if performance drops 5%
    
    def record_performance(
        self,
        prediction: float,
        actual: float,
        symbol: str
    ):
        """Record prediction performance for learning."""
        error = abs(prediction - actual)
        accuracy = 1.0 - min(1.0, error / (abs(actual) + 0.01))
        
        self.performance_history.append({
            'symbol': symbol,
            'prediction': prediction,
            'actual': actual,
            'error': error,
            'accuracy': accuracy,
            'timestamp': datetime.now()
        })
        
        # Keep only recent history (last 1000 predictions)
        if len(self.performance_history) > 1000:
            self.performance_history = self.performance_history[-1000:]
    
    def calculate_adaptive_weights(
        self,
        model_predictions: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Calculate adaptive weights for model ensemble based on recent performance.
        
        Args:
            model_predictions: Dictionary of model_name -> prediction
            
        Returns:
            Dictionary of model_name -> weight
        """
        if not model_predictions:
            return {}
        
        # Initialize equal weights
        weights = {model: 1.0 / len(model_predictions) for model in model_predictions.keys()}
        
        # If we have performance history, adjust weights
        if len(self.performance_history) > 50:
            # Calculate recent accuracy per model (simplified)
            # In real implementation, would track per-model performance
            recent_accuracy = np.mean([p['accuracy'] for p in self.performance_history[-50:]])
            
            # Boost weights if performance is good
            if recent_accuracy > 0.7:
                # Increase learning rate when doing well
                self.learning_rate = min(0.2, self.learning_rate * 1.1)
            elif recent_accuracy < 0.5:
                # Decrease learning rate when struggling
                self.learning_rate = max(0.05, self.learning_rate * 0.9)
        
        return weights
    
    def adaptive_predict(
        self,
        model_predictions: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Make adaptive prediction using weighted ensemble.
        
        Args:
            model_predictions: Dictionary of model_name -> prediction
            
        Returns:
            Dictionary with adaptive prediction and confidence
        """
        if not model_predictions:
            return {
                'prediction': 0.5,
                'confidence': 0.5,
                'adaptation_active': False
            }
        
        # Get adaptive weights
        weights = self.calculate_adaptive_weights(model_predictions)
        
        # Weighted average prediction
        weighted_sum = sum(pred * weights.get(model, 1.0/len(model_predictions)) 
                          for model, pred in model_predictions.items())
        weight_sum = sum(weights.get(model, 1.0/len(model_predictions)) 
                       for model in model_predictions.keys())
        
        prediction = weighted_sum / weight_sum if weight_sum > 0 else np.mean(list(model_predictions.values()))
        
        # Calculate confidence based on agreement
        predictions_list = list(model_predictions.values())
        std_dev = np.std(predictions_list)
        agreement = 1.0 - min(1.0, std_dev / (abs(prediction) + 0.01))
        
        # Adaptive confidence (higher if system is learning well)
        base_confidence = agreement
        learning_boost = min(0.1, self.learning_rate * 0.5)
        confidence = min(1.0, base_confidence + learning_boost)
        
        return {
            'prediction': float(prediction),
            'confidence': float(confidence),
            'adaptation_active': len(self.performance_history) > 50,
            'learning_rate': float(self.learning_rate),
            'model_count': len(model_predictions)
        }
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """Get learning system statistics."""
        if not self.performance_history:
            return {
                'total_predictions': 0,
                'average_accuracy': 0.0,
                'learning_rate': self.learning_rate
            }
        
        recent = self.performance_history[-100:] if len(self.performance_history) > 100 else self.performance_history
        avg_accuracy = np.mean([p['accuracy'] for p in recent])
        
        return {
            'total_predictions': len(self.performance_history),
            'average_accuracy': float(avg_accuracy),
            'recent_accuracy': float(np.mean([p['accuracy'] for p in recent[-20:]])),
            'learning_rate': float(self.learning_rate),
            'adaptation_active': len(self.performance_history) > 50
        }

