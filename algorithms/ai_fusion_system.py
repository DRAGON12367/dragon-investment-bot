"""
AI FUSION SYSTEM - Intelligently combines all algorithms for optimal predictions
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class AIFusionSystem:
    """
    AI Fusion System that intelligently combines all available algorithms
    to generate the most accurate trading signals.
    """
    
    def __init__(self, config):
        """Initialize AI Fusion System."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.ai_fusion")
        
        # Try to import all available systems
        self.systems = {}
        self.weights = {}
        
        # Base systems
        try:
            from algorithms.technical_indicators import TechnicalIndicators
            self.systems['technical'] = TechnicalIndicators()
            self.weights['technical'] = 1.0
        except ImportError:
            pass
        
        try:
            from algorithms.advanced_indicators import AdvancedIndicators
            self.systems['advanced'] = AdvancedIndicators()
            self.weights['advanced'] = 1.5
        except ImportError:
            pass
        
        # Ultra advanced
        try:
            from algorithms.ultra_advanced_indicators import UltraAdvancedIndicators
            self.systems['ultra'] = UltraAdvancedIndicators()
            self.weights['ultra'] = 2.0
        except ImportError:
            pass
        
        # Quantum
        try:
            from algorithms.quantum_advanced_indicators import QuantumAdvancedIndicators
            self.systems['quantum'] = QuantumAdvancedIndicators()
            self.weights['quantum'] = 2.5
        except ImportError:
            pass
        
        # Mega 50x
        try:
            from algorithms.mega_indicators_50x import MegaIndicators50X
            self.systems['mega_50x'] = MegaIndicators50X()
            self.weights['mega_50x'] = 3.0
        except ImportError:
            pass
        
        # Ultimate 100x
        try:
            from algorithms.ultimate_indicators_100x import UltimateIndicators100X
            self.systems['ultimate_100x'] = UltimateIndicators100X()
            self.weights['ultimate_100x'] = 4.0
        except ImportError:
            pass
        
        # ML Models
        try:
            from algorithms.ml_model import MLModel
            self.systems['ml_base'] = MLModel(config)
            self.weights['ml_base'] = 2.0
        except ImportError:
            pass
        
        try:
            from algorithms.advanced_ml_models import AdvancedMLModels
            self.systems['ml_advanced'] = AdvancedMLModels(config)
            self.weights['ml_advanced'] = 2.5
        except ImportError:
            pass
        
        try:
            from algorithms.mega_ml_models_50x import MegaMLModels50X
            self.systems['ml_mega_50x'] = MegaMLModels50X()
            self.weights['ml_mega_50x'] = 3.5
        except ImportError:
            pass
        
        try:
            from algorithms.ultimate_ml_models_100x import UltimateMLModels100X
            self.systems['ml_ultimate_100x'] = UltimateMLModels100X(config)
            self.weights['ml_ultimate_100x'] = 4.5
        except ImportError:
            pass
        
        # Strategies
        try:
            from algorithms.mega_strategies_50x import MegaStrategies50X
            self.systems['strategies_mega_50x'] = MegaStrategies50X()  # No config parameter
            self.weights['strategies_mega_50x'] = 3.0
        except ImportError:
            pass
        
        try:
            from algorithms.ultimate_strategies_100x import UltimateStrategies100X
            self.systems['strategies_ultimate_100x'] = UltimateStrategies100X(config)
            self.weights['strategies_ultimate_100x'] = 4.0
        except ImportError:
            pass
        
        # Profit guarantee
        try:
            from algorithms.profit_guarantee_system import ProfitGuaranteeSystem
            self.systems['profit_guarantee'] = ProfitGuaranteeSystem(config)
            self.weights['profit_guarantee'] = 5.0  # Highest weight
        except ImportError:
            pass
        
        self.logger.info(f"AI Fusion System initialized with {len(self.systems)} systems")
    
    def fuse_signals(
        self,
        symbol: str,
        market_data: Dict[str, Any],
        historical_data: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Fuse signals from all available systems.
        
        Returns comprehensive signal with confidence scores.
        """
        all_signals = []
        system_contributions = {}
        
        # Collect signals from all systems
        for system_name, system in self.systems.items():
            try:
                weight = self.weights.get(system_name, 1.0)
                
                if system_name.endswith('_100x') or system_name == 'ultimate_100x':
                    # Ultimate 100x systems
                    if hasattr(system, 'comprehensive_analysis') and historical_data is not None:
                        try:
                            # Check if historical_data has enough data
                            if len(historical_data) >= 20:  # Need minimum data points
                                analysis = system.comprehensive_analysis(historical_data)
                                signal = self._extract_signal_from_analysis(analysis, symbol, market_data)
                            else:
                                signal = None  # Not enough data
                        except Exception as e:
                            # Silently skip if analysis fails (don't spam errors)
                            signal = None
                    else:
                        signal = self._generate_basic_signal(system, symbol, market_data, historical_data)
                elif system_name.startswith('ml_'):
                    # ML models
                    signal = self._generate_ml_signal(system, symbol, market_data, historical_data)
                elif system_name.startswith('strategies_'):
                    # Strategy systems
                    if hasattr(system, 'select_best_strategy') and historical_data is not None:
                        result = system.select_best_strategy(historical_data)
                        signal = self._convert_strategy_to_signal(result, symbol, market_data)
                    else:
                        signal = self._generate_basic_signal(system, symbol, market_data, historical_data)
                elif system_name == 'profit_guarantee':
                    # Profit guarantee system
                    signal = self._generate_profit_guarantee_signal(system, symbol, market_data, historical_data)
                else:
                    # Other systems
                    signal = self._generate_basic_signal(system, symbol, market_data, historical_data)
                
                if signal:
                    signal['weight'] = weight
                    signal['system'] = system_name
                    all_signals.append(signal)
                    system_contributions[system_name] = signal
                    
            except Exception as e:
                self.logger.warning(f"Error getting signal from {system_name}: {e}")
                continue
        
        # Fuse signals
        fused_signal = self._fuse_signals(all_signals, symbol, market_data)
        
        # Add system contributions
        fused_signal['system_contributions'] = system_contributions
        fused_signal['total_systems'] = len(all_signals)
        
        return fused_signal
    
    def _extract_signal_from_analysis(self, analysis: Dict, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Extract signal from comprehensive analysis."""
        try:
            data = market_data.get(symbol, {})
            current_price = data.get('price', 0)
            
            if current_price == 0:
                return None
            
            # Get overall score
            overall_score = analysis.get('overall_profit_score', 0.5)
            
            # Determine action
            if overall_score > 0.7:
                action = 'BUY'
            elif overall_score < 0.3:
                action = 'SELL'
            else:
                action = 'HOLD'
            
            return {
                'action': action,
                'confidence': overall_score,
                'price': current_price
            }
        except:
            return None
    
    def _generate_ml_signal(self, system, symbol: str, market_data: Dict, historical_data: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Generate signal from ML model."""
        try:
            data = market_data.get(symbol, {})
            current_price = data.get('price', 0)
            
            if current_price == 0:
                return None
            
            # Try to get prediction
            if hasattr(system, 'predict') and historical_data is not None:
                try:
                    # Extract features (simplified)
                    features = self._extract_features(historical_data)
                    if features is not None and len(features) > 0:
                        prediction = system.predict(features)
                        confidence = abs(prediction - 0.5) * 2 if isinstance(prediction, (int, float)) else 0.5
                        
                        action = 'BUY' if prediction > 0.6 else ('SELL' if prediction < 0.4 else 'HOLD')
                        
                        return {
                            'action': action,
                            'confidence': confidence,
                            'price': current_price
                        }
                except:
                    pass
            
            # Fallback
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': current_price
            }
        except:
            return None
    
    def _generate_profit_guarantee_signal(self, system, symbol: str, market_data: Dict, historical_data: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Generate signal from profit guarantee system."""
        try:
            data = market_data.get(symbol, {})
            current_price = data.get('price', 0)
            
            if current_price == 0:
                return None
            
            if hasattr(system, 'analyze_profit_potential') and historical_data is not None:
                try:
                    result = system.analyze_profit_potential(symbol, market_data, historical_data)
                    if result.get('guaranteed_profit', False):
                        return {
                            'action': 'BUY',
                            'confidence': result.get('confidence', 0.9),
                            'price': current_price,
                            'guaranteed_profit': True
                        }
                except:
                    pass
            
            return None
        except:
            return None
    
    def _generate_basic_signal(self, system, symbol: str, market_data: Dict, historical_data: Optional[pd.DataFrame]) -> Optional[Dict]:
        """Generate basic signal from system."""
        try:
            data = market_data.get(symbol, {})
            current_price = data.get('price', 0)
            
            if current_price == 0:
                return None
            
            # Try common methods
            if hasattr(system, 'analyze'):
                try:
                    result = system.analyze(symbol, market_data)
                    return self._convert_to_signal(result, current_price)
                except:
                    pass
            
            return {
                'action': 'HOLD',
                'confidence': 0.5,
                'price': current_price
            }
        except:
            return None
    
    def _convert_strategy_to_signal(self, result: Dict, symbol: str, market_data: Dict) -> Optional[Dict]:
        """Convert strategy result to signal."""
        try:
            data = market_data.get(symbol, {})
            current_price = data.get('price', 0)
            
            if current_price == 0:
                return None
            
            action = result.get('action', 'HOLD')
            confidence = result.get('confidence', 0.5)
            
            return {
                'action': action,
                'confidence': confidence,
                'price': current_price
            }
        except:
            return None
    
    def _convert_to_signal(self, result: Any, current_price: float) -> Dict:
        """Convert various result types to signal."""
        if isinstance(result, dict):
            action = result.get('action', 'HOLD')
            confidence = result.get('confidence', result.get('score', 0.5))
        else:
            action = 'HOLD'
            confidence = 0.5
        
        return {
            'action': action,
            'confidence': confidence,
            'price': current_price
        }
    
    def _fuse_signals(self, signals: List[Dict], symbol: str, market_data: Dict) -> Dict[str, Any]:
        """Fuse multiple signals into one."""
        if not signals:
            data = market_data.get(symbol, {})
            return {
                'symbol': symbol,
                'action': 'HOLD',
                'confidence': 0,
                'price': data.get('price', 0),
                'fused': True
            }
        
        # Weighted voting
        buy_votes = 0
        sell_votes = 0
        hold_votes = 0
        total_weight = 0
        total_confidence = 0
        
        for signal in signals:
            weight = signal.get('weight', 1.0)
            action = signal.get('action', 'HOLD')
            confidence = signal.get('confidence', 0.5)
            
            total_weight += weight
            total_confidence += confidence * weight
            
            if action == 'BUY':
                buy_votes += weight * confidence
            elif action == 'SELL':
                sell_votes += weight * confidence
            else:
                hold_votes += weight
        
        # Determine final action
        if buy_votes > sell_votes and buy_votes > hold_votes:
            final_action = 'BUY'
            final_confidence = min(buy_votes / total_weight, 1.0)
        elif sell_votes > buy_votes and sell_votes > hold_votes:
            final_action = 'SELL'
            final_confidence = min(sell_votes / total_weight, 1.0)
        else:
            final_action = 'HOLD'
            final_confidence = 0.3
        
        # Average confidence
        avg_confidence = total_confidence / total_weight if total_weight > 0 else 0.5
        final_confidence = max(final_confidence, avg_confidence * 0.7)
        
        data = market_data.get(symbol, {})
        current_price = data.get('price', 0)
        
        return {
            'symbol': symbol,
            'action': final_action,
            'confidence': min(final_confidence, 1.0),
            'price': current_price,
            'buy_votes': buy_votes,
            'sell_votes': sell_votes,
            'hold_votes': hold_votes,
            'total_systems': len(signals),
            'fused': True
        }
    
    def _extract_features(self, df: pd.DataFrame) -> Optional[np.ndarray]:
        """Extract features from historical data."""
        try:
            if df is None or df.empty:
                return None
            
            features = []
            
            # Price features
            if 'close' in df.columns:
                close = df['close'].values
                features.extend([
                    close[-1] if len(close) > 0 else 0,
                    close[-1] / close[-5] if len(close) >= 5 else 1,
                    close[-1] / close[-10] if len(close) >= 10 else 1,
                ])
            
            # Volume features
            if 'volume' in df.columns:
                volume = df['volume'].values
                if len(volume) > 0:
                    features.append(volume[-1])
                    if len(volume) >= 10:
                        features.append(volume[-1] / np.mean(volume[-10:]))
            
            return np.array(features).reshape(1, -1) if features else None
        except:
            return None

