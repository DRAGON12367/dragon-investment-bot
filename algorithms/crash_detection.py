"""
CRASH DETECTION SYSTEM - Detects imminent crashes and triggers urgent sell signals.
"""
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import logging


class CrashDetection:
    """
    Advanced crash detection system that identifies assets about to crash.
    Triggers urgent "SELL RIGHT NOW" signals.
    """
    
    def __init__(self, config):
        """Initialize crash detection system."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.crash_detection")
        
        # Crash detection thresholds
        self.crash_threshold = -0.05  # -5% drop triggers crash alert
        self.urgent_threshold = -0.10  # -10% drop = URGENT
        self.critical_threshold = -0.15  # -15% drop = CRITICAL
        
        # Volume spike threshold (indicates panic selling)
        self.volume_spike_threshold = 2.0  # 2x normal volume
        
        # Momentum crash indicators
        self.momentum_crash_threshold = -0.03  # -3% momentum = crash risk
        
    def detect_crashes(
        self,
        market_data: Dict[str, Any],
        price_history: Optional[Dict[str, List[Dict]]] = None
    ) -> List[Dict[str, Any]]:
        """
        Detect crashes across all assets.
        
        Returns:
            List of crash alerts with urgency levels.
        """
        crash_alerts = []
        
        for symbol, data in market_data.items():
            try:
                alert = self._analyze_crash_risk(symbol, data, price_history)
                if alert and alert.get('crash_detected', False):
                    crash_alerts.append(alert)
            except Exception as e:
                self.logger.debug(f"Error analyzing crash for {symbol}: {e}")
                continue
        
        # Sort by urgency (CRITICAL > URGENT > HIGH)
        urgency_order = {'CRITICAL': 0, 'URGENT': 1, 'HIGH': 2, 'MEDIUM': 3}
        crash_alerts.sort(key=lambda x: urgency_order.get(x.get('urgency', 'MEDIUM'), 3))
        
        return crash_alerts
    
    def _analyze_crash_risk(
        self,
        symbol: str,
        data: Dict[str, Any],
        price_history: Optional[Dict[str, List[Dict]]] = None
    ) -> Optional[Dict[str, Any]]:
        """Analyze crash risk for a single asset."""
        current_price = data.get('price', 0)
        if current_price == 0:
            return None
        
        # Get price change
        change_pct = data.get('change_percent', 0) / 100.0  # Convert to decimal
        
        # Get volume
        volume = data.get('volume', 0)
        
        # Analyze price history if available
        historical_crash_signals = 0
        momentum_crash = False
        
        if price_history and symbol in price_history:
            hist = price_history[symbol]
            if len(hist) >= 10:
                prices = pd.Series([p['price'] for p in hist[-10:]])
                
                # Check for rapid decline
                recent_change = (prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0] if prices.iloc[0] > 0 else 0
                if recent_change < -0.05:  # -5% in recent period
                    historical_crash_signals += 1
                
                # Check momentum
                momentum = prices.pct_change().mean()
                if momentum < self.momentum_crash_threshold:
                    momentum_crash = True
                    historical_crash_signals += 1
                
                # Check for accelerating decline
                if len(hist) >= 5:
                    recent_prices = prices.iloc[-5:]
                    decline_rate = (recent_prices.iloc[-1] - recent_prices.iloc[0]) / recent_prices.iloc[0]
                    if decline_rate < -0.03:  # -3% in last 5 periods
                        historical_crash_signals += 1
        
        # Determine crash severity
        crash_detected = False
        urgency = 'MEDIUM'
        sell_action = None
        
        # CRITICAL: -15% or more drop
        if change_pct <= self.critical_threshold:
            crash_detected = True
            urgency = 'CRITICAL'
            sell_action = 'SELL RIGHT NOW - CRITICAL CRASH DETECTED'
        
        # URGENT: -10% to -15% drop
        elif change_pct <= self.urgent_threshold:
            crash_detected = True
            urgency = 'URGENT'
            sell_action = 'SELL RIGHT NOW - URGENT CRASH DETECTED'
        
        # HIGH: -5% to -10% drop
        elif change_pct <= self.crash_threshold:
            crash_detected = True
            urgency = 'HIGH'
            sell_action = 'GET RID OF IT RIGHT NOW - CRASH DETECTED'
        
        # Check for volume spike (panic selling)
        if volume > 0:
            # Would compare to average volume in real implementation
            # For now, use high volume as additional signal
            if change_pct < -0.03 and volume > 1000000:  # High volume + drop
                if not crash_detected:
                    crash_detected = True
                    urgency = 'HIGH'
                    sell_action = 'GET RID OF IT RIGHT NOW - HIGH VOLUME SELLING'
                else:
                    # Upgrade urgency if volume spike
                    if urgency == 'HIGH':
                        urgency = 'URGENT'
                        sell_action = 'SELL RIGHT NOW - HIGH VOLUME CRASH'
        
        # Check historical signals
        if historical_crash_signals >= 2:
            if not crash_detected:
                crash_detected = True
                urgency = 'HIGH'
                sell_action = 'GET RID OF IT RIGHT NOW - MULTIPLE CRASH SIGNALS'
            elif urgency == 'HIGH':
                urgency = 'URGENT'
                sell_action = 'SELL RIGHT NOW - ACCELERATING DECLINE'
        
        # Momentum crash
        if momentum_crash and change_pct < -0.02:
            if not crash_detected:
                crash_detected = True
                urgency = 'HIGH'
                sell_action = 'GET RID OF IT RIGHT NOW - MOMENTUM CRASH'
            elif urgency == 'HIGH':
                urgency = 'URGENT'
        
        if not crash_detected:
            return None
        
        # Calculate expected loss if not sold
        expected_loss = abs(change_pct) * 100
        
        # Calculate stop loss price
        stop_loss = current_price * 0.95  # 5% below current
        
        return {
            'symbol': symbol,
            'asset_type': data.get('asset_type', 'unknown'),
            'crash_detected': True,
            'urgency': urgency,
            'sell_action': sell_action,
            'current_price': current_price,
            'price_change_pct': change_pct * 100,
            'expected_loss': expected_loss,
            'stop_loss': stop_loss,
            'volume': volume,
            'historical_signals': historical_crash_signals,
            'momentum_crash': momentum_crash,
            'timestamp': datetime.now().isoformat(),
            'reason': self._get_crash_reason(change_pct, historical_crash_signals, momentum_crash, volume)
        }
    
    def _get_crash_reason(
        self,
        change_pct: float,
        historical_signals: int,
        momentum_crash: bool,
        volume: float
    ) -> str:
        """Get human-readable crash reason."""
        reasons = []
        
        if change_pct <= -0.15:
            reasons.append("CRITICAL: Price dropped 15%+")
        elif change_pct <= -0.10:
            reasons.append("URGENT: Price dropped 10%+")
        elif change_pct <= -0.05:
            reasons.append("Price dropped 5%+")
        
        if historical_signals >= 2:
            reasons.append("Multiple crash signals detected")
        
        if momentum_crash:
            reasons.append("Negative momentum accelerating")
        
        if volume > 1000000 and change_pct < 0:
            reasons.append("High volume panic selling")
        
        return " | ".join(reasons) if reasons else "Crash pattern detected"
    
    def get_crash_list(self, crash_alerts: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """
        Get formatted list of crashing assets by type.
        
        Returns:
            Dictionary with 'stocks' and 'crypto' lists of symbol names
        """
        stocks = []
        crypto = []
        
        for alert in crash_alerts:
            symbol = alert.get('symbol', '')
            asset_type = alert.get('asset_type', 'unknown')
            
            if asset_type == 'stock':
                stocks.append(symbol)
            elif asset_type == 'crypto':
                crypto.append(symbol)
        
        return {
            'stocks': stocks,
            'crypto': crypto,
            'total': len(crash_alerts)
        }

