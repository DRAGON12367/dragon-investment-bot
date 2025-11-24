"""
NEURAL ARCHITECTURE SEARCH - AutoML for optimal trading model architectures
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.model_selection import cross_val_score, TimeSeriesSplit
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

try:
    from tensorflow import keras
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, LSTM, GRU, Dropout, BatchNormalization
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False


class NeuralArchitectureSearch:
    """
    Neural Architecture Search for automatically finding optimal trading models.
    Uses evolutionary algorithms and reinforcement learning concepts.
    """
    
    def __init__(self, config):
        """Initialize NAS."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.nas")
        self.best_architectures = []
        self.performance_history = []
        
    def search_optimal_architecture(
        self,
        X: np.ndarray,
        y: np.ndarray,
        max_iterations: int = 20,
        population_size: int = 10
    ) -> Dict[str, Any]:
        """
        Search for optimal neural architecture using evolutionary approach.
        """
        if not SKLEARN_AVAILABLE:
            return {'best_model': None, 'score': 0}
        
        # Initialize population of architectures
        population = self._initialize_population(population_size, X.shape[1])
        
        best_architecture = None
        best_score = 0
        
        for iteration in range(max_iterations):
            # Evaluate each architecture
            scores = []
            for arch in population:
                try:
                    model = self._build_model_from_arch(arch, X.shape[1])
                    score = self._evaluate_model(model, X, y)
                    scores.append(score)
                    
                    if score > best_score:
                        best_score = score
                        best_architecture = arch
                except Exception as e:
                    self.logger.warning(f"Error evaluating architecture: {e}")
                    scores.append(0)
            
            # Select best architectures
            sorted_pop = sorted(zip(population, scores), key=lambda x: x[1], reverse=True)
            elite = [arch for arch, score in sorted_pop[:population_size // 2]]
            
            # Create new generation (crossover and mutation)
            new_population = elite.copy()
            while len(new_population) < population_size:
                # Crossover
                parent1 = np.random.choice(len(elite))
                parent2 = np.random.choice(len(elite))
                child = self._crossover(elite[parent1], elite[parent2])
                
                # Mutation
                child = self._mutate(child)
                new_population.append(child)
            
            population = new_population
            self.logger.info(f"Iteration {iteration + 1}: Best score = {best_score:.4f}")
        
        # Build final model
        if best_architecture:
            final_model = self._build_model_from_arch(best_architecture, X.shape[1])
            final_model.fit(X, y)
            
            return {
                'best_model': final_model,
                'architecture': best_architecture,
                'score': best_score
            }
        
        return {'best_model': None, 'score': 0}
    
    def _initialize_population(self, size: int, input_dim: int) -> List[Dict]:
        """Initialize population of random architectures."""
        population = []
        
        for _ in range(size):
            arch = {
                'type': np.random.choice(['tree', 'neural', 'ensemble']),
                'layers': np.random.randint(1, 5),
                'units': [np.random.choice([32, 64, 128, 256]) for _ in range(np.random.randint(1, 4))],
                'dropout': np.random.uniform(0.1, 0.5),
                'learning_rate': np.random.choice([0.001, 0.01, 0.1]),
                'n_estimators': np.random.choice([50, 100, 200]) if np.random.random() > 0.5 else 100
            }
            population.append(arch)
        
        return population
    
    def _build_model_from_arch(self, arch: Dict, input_dim: int):
        """Build model from architecture specification."""
        if arch['type'] == 'tree':
            if XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(
                    n_estimators=arch['n_estimators'],
                    learning_rate=arch['learning_rate'],
                    random_state=42
                )
            else:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(
                    n_estimators=arch['n_estimators'],
                    random_state=42
                )
        
        elif arch['type'] == 'neural' and TENSORFLOW_AVAILABLE:
            model = Sequential()
            model.add(Dense(arch['units'][0], activation='relu', input_dim=input_dim))
            model.add(BatchNormalization())
            model.add(Dropout(arch['dropout']))
            
            for units in arch['units'][1:]:
                model.add(Dense(units, activation='relu'))
                model.add(BatchNormalization())
                model.add(Dropout(arch['dropout']))
            
            model.add(Dense(1, activation='sigmoid'))
            model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
            return model
        
        else:
            # Default to tree model
            if XGBOOST_AVAILABLE:
                return xgb.XGBClassifier(random_state=42)
            else:
                from sklearn.ensemble import RandomForestClassifier
                return RandomForestClassifier(random_state=42)
    
    def _evaluate_model(self, model, X: np.ndarray, y: np.ndarray) -> float:
        """Evaluate model performance."""
        try:
            if TENSORFLOW_AVAILABLE and isinstance(model, keras.Model):
                # For neural networks, use simple train/test split
                split = int(len(X) * 0.8)
                X_train, X_test = X[:split], X[split:]
                y_train, y_test = y[:split], y[split:]
                
                model.fit(X_train, y_train, epochs=10, verbose=0, batch_size=32)
                predictions = (model.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
                score = accuracy_score(y_test, predictions)
            else:
                # For sklearn models, use cross-validation
                tscv = TimeSeriesSplit(n_splits=3)
                scores = cross_val_score(model, X, y, cv=tscv, scoring='accuracy', n_jobs=-1)
                score = np.mean(scores)
            
            return score
        except Exception as e:
            self.logger.warning(f"Error evaluating model: {e}")
            return 0.0
    
    def _crossover(self, parent1: Dict, parent2: Dict) -> Dict:
        """Crossover two architectures."""
        child = {}
        
        # Randomly inherit from parents
        child['type'] = np.random.choice([parent1['type'], parent2['type']])
        child['layers'] = np.random.choice([parent1['layers'], parent2['layers']])
        child['units'] = parent1['units'] if np.random.random() > 0.5 else parent2['units']
        child['dropout'] = (parent1['dropout'] + parent2['dropout']) / 2
        child['learning_rate'] = np.random.choice([parent1['learning_rate'], parent2['learning_rate']])
        child['n_estimators'] = np.random.choice([parent1['n_estimators'], parent2['n_estimators']])
        
        return child
    
    def _mutate(self, arch: Dict) -> Dict:
        """Mutate architecture."""
        if np.random.random() < 0.3:  # 30% mutation rate
            arch['type'] = np.random.choice(['tree', 'neural', 'ensemble'])
        
        if np.random.random() < 0.3:
            arch['layers'] = np.random.randint(1, 5)
        
        if np.random.random() < 0.3:
            arch['units'] = [np.random.choice([32, 64, 128, 256]) for _ in range(len(arch['units']))]
        
        if np.random.random() < 0.3:
            arch['dropout'] = np.random.uniform(0.1, 0.5)
        
        if np.random.random() < 0.3:
            arch['learning_rate'] = np.random.choice([0.001, 0.01, 0.1])
        
        return arch
    
    def auto_optimize_model(self, X: np.ndarray, y: np.ndarray) -> Any:
        """Automatically find and return optimized model."""
        result = self.search_optimal_architecture(X, y)
        return result.get('best_model')

