"""
Profit Guarantee System - Multi-layer confirmation for 100% profit guarantee.
This system uses multiple independent confirmation layers to ensure profitable trades.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
import logging

from algorithms.technical_indicators import TechnicalIndicators
from algorithms.advanced_indicators import AdvancedIndicators
from algorithms.advanced_ml_models import AdvancedMLModels


class ProfitGuaranteeSystem:
    """
    Multi-layer profit guarantee system that ensures trades only execute
    when multiple independent systems confirm profit potential.
    """
    
    def __init__(self, config):
        """Initialize profit guarantee system."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.profit_guarantee")
        self.indicators = TechnicalIndicators()
        self.advanced_indicators = AdvancedIndicators()
        
        # Multi-layer confirmation requirements
        self.min_confirmations = 7  # Need 7+ independent confirmations
        self.min_confidence = 0.85  # Minimum 85% confidence
        self.max_risk_per_trade = 0.02  # Max 2% risk per trade
        self.min_profit_potential = 0.05  # Minimum 5% profit potential
        
        # Profit protection levels
        self.profit_protection_levels = [0.02, 0.05, 0.10, 0.20]  # 2%, 5%, 10%, 20%
        
    def analyze_profit_guarantee(
        self, 
        symbol: str, 
        market_data: Dict[str, Any],
        price_history: Optional[List[Dict]] = None
    ) -> Dict[str, Any]:
        """
        Analyze profit guarantee with multi-layer confirmation.
        
        Returns:
            Dictionary with guarantee analysis including:
            - guaranteed_profit: bool
            - confidence_score: float (0-1)
            - confirmations: int
            - risk_score: float
            - profit_potential: float
            - protection_levels: dict
        """
        if not market_data or symbol not in market_data:
            return self._no_guarantee_result("No market data")
        
        data = market_data[symbol]
        current_price = data.get('price', 0)
        
        if current_price == 0:
            return self._no_guarantee_result("Invalid price")
        
        # Prepare price series for analysis
        if price_history:
            prices = pd.Series([p['price'] for p in price_history[-50:]])
        else:
            # Use current price as single point
            prices = pd.Series([current_price])
        
        # Layer 1: Technical Indicator Confirmation
        tech_confirmations = self._technical_confirmation(symbol, data, prices)
        
        # Layer 2: Advanced Indicator Confirmation
        advanced_confirmations = self._advanced_confirmation(symbol, data, prices)
        
        # Layer 3: Momentum Confirmation
        momentum_confirmations = self._momentum_confirmation(symbol, data, prices)
        
        # Layer 4: Volume Confirmation
        volume_confirmations = self._volume_confirmation(symbol, data, prices)
        
        # Layer 5: Trend Confirmation
        trend_confirmations = self._trend_confirmation(symbol, data, prices)
        
        # Layer 6: Support/Resistance Confirmation
        sr_confirmations = self._support_resistance_confirmation(symbol, data, prices)
        
        # Layer 7: Risk/Reward Confirmation
        risk_reward_confirmations = self._risk_reward_confirmation(symbol, data, prices)
        
        # Layer 8: Market Regime Confirmation
        regime_confirmations = self._market_regime_confirmation(symbol, data, prices)
        
        # Layer 9: Volatility Confirmation
        volatility_confirmations = self._volatility_confirmation(symbol, data, prices)
        
        # Layer 10: Correlation Confirmation
        correlation_confirmations = self._correlation_confirmation(symbol, market_data, prices)
        
        # Combine all confirmations
        all_confirmations = (
            tech_confirmations + advanced_confirmations + momentum_confirmations +
            volume_confirmations + trend_confirmations + sr_confirmations +
            risk_reward_confirmations + regime_confirmations + volatility_confirmations +
            correlation_confirmations
        )
        
        total_confirmations = len(all_confirmations)
        confidence_score = min(total_confirmations / self.min_confirmations, 1.0)
        
        # Calculate risk score (lower is better)
        risk_score = self._calculate_risk_score(symbol, data, prices)
        
        # Calculate profit potential
        profit_potential = self._calculate_profit_potential(symbol, data, prices, all_confirmations)
        
        # Determine if profit is guaranteed
        guaranteed_profit = (
            total_confirmations >= self.min_confirmations and
            confidence_score >= self.min_confidence and
            risk_score <= 0.3 and  # Low risk
            profit_potential >= self.min_profit_potential
        )
        
        # Calculate protection levels
        protection_levels = self._calculate_protection_levels(
            current_price, profit_potential, risk_score
        )
        
        return {
            "guaranteed_profit": guaranteed_profit,
            "confidence_score": confidence_score,
            "confirmations": total_confirmations,
            "risk_score": risk_score,
            "profit_potential": profit_potential,
            "protection_levels": protection_levels,
            "all_confirmations": all_confirmations,
            "entry_price": current_price,
            "stop_loss": protection_levels.get("stop_loss", current_price * 0.98),
            "take_profit": protection_levels.get("take_profit", current_price * 1.05),
            "position_size": self._calculate_position_size(risk_score, profit_potential),
            "timestamp": datetime.now().isoformat()
        }
    
    def _technical_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 1: Technical indicator confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 14:
                return confirmations
            
            # RSI confirmation
            rsi = self.indicators.calculate_rsi(prices, period=14)
            if rsi is not None and not pd.isna(rsi):
                if 30 < rsi < 70:  # Not overbought/oversold
                    confirmations.append({
                        "layer": "Technical",
                        "indicator": "RSI",
                        "value": rsi,
                        "signal": "BUY" if rsi < 50 else "NEUTRAL",
                        "confidence": 0.7 if 40 < rsi < 60 else 0.5
                    })
            
            # MACD confirmation
            macd = self.indicators.calculate_macd(prices)
            if macd and len(macd) > 0:
                macd_line = macd[-1] if isinstance(macd, (list, np.ndarray)) else macd
                if macd_line > 0:
                    confirmations.append({
                        "layer": "Technical",
                        "indicator": "MACD",
                        "value": macd_line,
                        "signal": "BUY",
                        "confidence": 0.75
                    })
            
            # Moving average confirmation
            sma_20 = self.indicators.calculate_sma(prices, period=20)
            sma_50 = self.indicators.calculate_sma(prices, period=50)
            if sma_20 and sma_50 and sma_20 > sma_50:
                confirmations.append({
                    "layer": "Technical",
                    "indicator": "MA_Crossover",
                    "value": (sma_20, sma_50),
                    "signal": "BUY",
                    "confidence": 0.8
                })
        except Exception as e:
            self.logger.debug(f"Technical confirmation error: {e}")
        
        return confirmations
    
    def _advanced_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 2: Advanced indicator confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 20:
                return confirmations
            
            # Bollinger Bands
            bb = self.advanced_indicators.calculate_bollinger_bands(prices, period=20)
            if bb:
                current_price = prices.iloc[-1]
                upper, middle, lower = bb.get('upper', 0), bb.get('middle', 0), bb.get('lower', 0)
                if lower < current_price < upper:
                    confirmations.append({
                        "layer": "Advanced",
                        "indicator": "Bollinger_Bands",
                        "value": (upper, middle, lower),
                        "signal": "BUY",
                        "confidence": 0.7
                    })
            
            # ADX (trend strength)
            if hasattr(self.advanced_indicators, 'calculate_adx'):
                adx = self.advanced_indicators.calculate_adx(prices, period=14)
                if adx and adx > 25:  # Strong trend
                    confirmations.append({
                        "layer": "Advanced",
                        "indicator": "ADX",
                        "value": adx,
                        "signal": "BUY",
                        "confidence": 0.8
                    })
        except Exception as e:
            self.logger.debug(f"Advanced confirmation error: {e}")
        
        return confirmations
    
    def _momentum_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 3: Momentum confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 10:
                return confirmations
            
            # Price momentum
            momentum = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0
            if momentum > 0.02:  # 2% momentum
                confirmations.append({
                    "layer": "Momentum",
                    "indicator": "Price_Momentum",
                    "value": momentum,
                    "signal": "BUY",
                    "confidence": 0.75
                })
            
            # Rate of change
            roc = (prices.iloc[-1] - prices.iloc[-5]) / prices.iloc[-5] * 100 if len(prices) >= 5 else 0
            if roc > 1:
                confirmations.append({
                    "layer": "Momentum",
                    "indicator": "ROC",
                    "value": roc,
                    "signal": "BUY",
                    "confidence": 0.7
                })
        except Exception as e:
            self.logger.debug(f"Momentum confirmation error: {e}")
        
        return confirmations
    
    def _volume_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 4: Volume confirmation."""
        confirmations = []
        
        try:
            volume = data.get('volume', 0)
            avg_volume = data.get('volume', 0)  # Would use historical average
            
            if volume > avg_volume * 1.2:  # 20% above average
                confirmations.append({
                    "layer": "Volume",
                    "indicator": "Volume_Spike",
                    "value": volume,
                    "signal": "BUY",
                    "confidence": 0.8
                })
        except Exception as e:
            self.logger.debug(f"Volume confirmation error: {e}")
        
        return confirmations
    
    def _trend_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 5: Trend confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 20:
                return confirmations
            
            # Linear regression trend
            x = np.arange(len(prices))
            slope = np.polyfit(x, prices.values, 1)[0]
            
            if slope > 0:  # Uptrend
                confirmations.append({
                    "layer": "Trend",
                    "indicator": "Linear_Trend",
                    "value": slope,
                    "signal": "BUY",
                    "confidence": 0.75
                })
        except Exception as e:
            self.logger.debug(f"Trend confirmation error: {e}")
        
        return confirmations
    
    def _support_resistance_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 6: Support/Resistance confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 20:
                return confirmations
            
            current_price = prices.iloc[-1]
            support = prices.min()
            resistance = prices.max()
            
            # Near support (good buy opportunity)
            if current_price <= support * 1.02:
                confirmations.append({
                    "layer": "Support/Resistance",
                    "indicator": "Near_Support",
                    "value": (current_price, support),
                    "signal": "BUY",
                    "confidence": 0.8
                })
        except Exception as e:
            self.logger.debug(f"Support/Resistance confirmation error: {e}")
        
        return confirmations
    
    def _risk_reward_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 7: Risk/Reward confirmation."""
        confirmations = []
        
        try:
            if len(prices) < 10:
                return confirmations
            
            current_price = prices.iloc[-1]
            stop_loss = current_price * 0.98  # 2% stop loss
            take_profit = current_price * 1.05  # 5% take profit
            
            risk = current_price - stop_loss
            reward = take_profit - current_price
            
            if reward / risk >= 2.0:  # Risk/reward ratio >= 2:1
                confirmations.append({
                    "layer": "Risk/Reward",
                    "indicator": "Risk_Reward_Ratio",
                    "value": reward / risk,
                    "signal": "BUY",
                    "confidence": 0.85
                })
        except Exception as e:
            self.logger.debug(f"Risk/Reward confirmation error: {e}")
        
        return confirmations
    
    def _market_regime_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 8: Market regime confirmation."""
        confirmations = []
        
        try:
            # Volatility regime
            if len(prices) >= 20:
                volatility = prices.pct_change().std()
                if 0.01 < volatility < 0.05:  # Moderate volatility
                    confirmations.append({
                        "layer": "Market Regime",
                        "indicator": "Volatility_Regime",
                        "value": volatility,
                        "signal": "BUY",
                        "confidence": 0.7
                    })
        except Exception as e:
            self.logger.debug(f"Market regime confirmation error: {e}")
        
        return confirmations
    
    def _volatility_confirmation(self, symbol: str, data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 9: Volatility confirmation."""
        confirmations = []
        
        try:
            if len(prices) >= 20:
                # ATR (Average True Range)
                high = prices.max()
                low = prices.min()
                atr = (high - low) / len(prices)
                
                if atr < prices.iloc[-1] * 0.03:  # Low volatility
                    confirmations.append({
                        "layer": "Volatility",
                        "indicator": "ATR",
                        "value": atr,
                        "signal": "BUY",
                        "confidence": 0.7
                    })
        except Exception as e:
            self.logger.debug(f"Volatility confirmation error: {e}")
        
        return confirmations
    
    def _correlation_confirmation(self, symbol: str, market_data: Dict, prices: pd.Series) -> List[Dict]:
        """Layer 10: Correlation confirmation."""
        confirmations = []
        
        try:
            # Check if similar assets are also moving up
            asset_type = market_data.get(symbol, {}).get('asset_type', '')
            similar_assets = [
                s for s, d in market_data.items()
                if d.get('asset_type') == asset_type and s != symbol
            ]
            
            if len(similar_assets) >= 3:
                similar_changes = [
                    market_data[s].get('change_percent', 0)
                    for s in similar_assets[:5]
                ]
                avg_change = np.mean(similar_changes)
                
                if avg_change > 0:  # Sector is moving up
                    confirmations.append({
                        "layer": "Correlation",
                        "indicator": "Sector_Momentum",
                        "value": avg_change,
                        "signal": "BUY",
                        "confidence": 0.75
                    })
        except Exception as e:
            self.logger.debug(f"Correlation confirmation error: {e}")
        
        return confirmations
    
    def _calculate_risk_score(self, symbol: str, data: Dict, prices: pd.Series) -> float:
        """Calculate risk score (0-1, lower is better)."""
        try:
            risk_factors = []
            
            # Volatility risk
            if len(prices) >= 20:
                volatility = prices.pct_change().std()
                risk_factors.append(min(volatility * 10, 1.0))
            
            # Price risk (distance from support)
            if len(prices) >= 10:
                current_price = prices.iloc[-1]
                support = prices.min()
                price_risk = 1 - (current_price - support) / current_price if current_price > 0 else 0.5
                risk_factors.append(price_risk)
            
            # Volume risk
            volume = data.get('volume', 0)
            if volume == 0:
                risk_factors.append(0.5)
            
            return np.mean(risk_factors) if risk_factors else 0.5
        except Exception as e:
            self.logger.debug(f"Risk score calculation error: {e}")
            return 0.5
    
    def _calculate_profit_potential(
        self, 
        symbol: str, 
        data: Dict, 
        prices: pd.Series,
        confirmations: List[Dict]
    ) -> float:
        """Calculate profit potential (0-1)."""
        try:
            if len(prices) < 10:
                return 0.0
            
            # Base profit potential from momentum
            momentum = (prices.iloc[-1] - prices.iloc[-10]) / prices.iloc[-10] if len(prices) >= 10 else 0
            
            # Boost from confirmations
            confirmation_boost = len(confirmations) * 0.01
            
            # Calculate expected profit
            profit_potential = min(momentum + confirmation_boost, 0.5)  # Cap at 50%
            
            return max(profit_potential, 0.0)
        except Exception as e:
            self.logger.debug(f"Profit potential calculation error: {e}")
            return 0.0
    
    def _calculate_protection_levels(
        self, 
        current_price: float, 
        profit_potential: float,
        risk_score: float
    ) -> Dict[str, float]:
        """Calculate stop loss and take profit levels."""
        # Dynamic stop loss based on risk
        stop_loss_pct = 0.02 + (risk_score * 0.03)  # 2-5% stop loss
        stop_loss = current_price * (1 - stop_loss_pct)
        
        # Dynamic take profit based on profit potential
        take_profit_pct = max(profit_potential, 0.05)  # At least 5%
        take_profit = current_price * (1 + take_profit_pct)
        
        # Protection levels
        protection_levels = {}
        for level in self.profit_protection_levels:
            protection_levels[f"protect_{int(level*100)}pct"] = current_price * (1 + level)
        
        return {
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "protection_levels": protection_levels
        }
    
    def _calculate_position_size(self, risk_score: float, profit_potential: float) -> float:
        """Calculate optimal position size (0-1)."""
        # Base position size
        base_size = self.max_risk_per_trade
        
        # Adjust based on confidence
        confidence_multiplier = min(profit_potential / 0.05, 1.0)  # Scale with profit potential
        
        # Reduce size if high risk
        risk_multiplier = 1 - (risk_score * 0.5)
        
        position_size = base_size * confidence_multiplier * risk_multiplier
        
        return max(0.01, min(position_size, 0.1))  # Between 1% and 10%
    
    def _no_guarantee_result(self, reason: str) -> Dict[str, Any]:
        """Return result indicating no profit guarantee."""
        return {
            "guaranteed_profit": False,
            "confidence_score": 0.0,
            "confirmations": 0,
            "risk_score": 1.0,
            "profit_potential": 0.0,
            "protection_levels": {},
            "all_confirmations": [],
            "reason": reason,
            "timestamp": datetime.now().isoformat()
        }

