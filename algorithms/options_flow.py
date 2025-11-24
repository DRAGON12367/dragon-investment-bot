"""
Options Flow Analysis - Simulated options market analysis.
Wall Street Use: Options flow reveals institutional sentiment and positioning.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta


class OptionsFlowAnalyzer:
    """
    Simulate options flow analysis from price action.
    
    Techniques:
    1. Put/Call Ratio (simulated from price patterns)
    2. Options Volume Analysis
    3. Implied Volatility Signals
    4. Gamma Levels (where options dealers hedge)
    5. Max Pain Theory
    """
    
    def __init__(self):
        """Initialize options flow analyzer."""
        self.logger = logging.getLogger("ai_investment_bot.options_flow")
        
    def simulate_put_call_ratio(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Simulate Put/Call ratio from price action.
        
        Wall Street Use: High put/call ratio = bearish sentiment, low = bullish.
        Contrarian: Extreme ratios often signal reversals.
        """
        if df.empty or len(df) < 20:
            return {}
        
        # Simulate put/call ratio based on:
        # 1. Downward price pressure = more puts
        # 2. Upward price pressure = more calls
        # 3. Volatility = more options activity
        
        returns = df['close'].pct_change().dropna()
        
        # Downward moves = put buying
        down_moves = returns[returns < 0]
        put_volume = abs(down_moves.sum()) * 100  # Simulated
        
        # Upward moves = call buying
        up_moves = returns[returns > 0]
        call_volume = up_moves.sum() * 100  # Simulated
        
        # Put/Call ratio
        if call_volume > 0:
            put_call_ratio = put_volume / call_volume
        else:
            put_call_ratio = 2.0  # High if no calls
        
        # Interpretation
        if put_call_ratio > 1.5:
            sentiment = 'EXTREMELY_BEARISH'
            signal = 'BUY'  # Contrarian
        elif put_call_ratio > 1.2:
            sentiment = 'BEARISH'
            signal = 'WEAK_BUY'
        elif put_call_ratio < 0.7:
            sentiment = 'EXTREMELY_BULLISH'
            signal = 'SELL'  # Contrarian
        elif put_call_ratio < 0.9:
            sentiment = 'BULLISH'
            signal = 'WEAK_SELL'
        else:
            sentiment = 'NEUTRAL'
            signal = 'NEUTRAL'
        
        return {
            'put_call_ratio': float(put_call_ratio),
            'sentiment': sentiment,
            'signal': signal,
            'put_volume': float(put_volume),
            'call_volume': float(call_volume),
            'ratio_interpretation': 'BEARISH' if put_call_ratio > 1.0 else 'BULLISH'
        }
    
    def detect_gamma_levels(
        self,
        current_price: float,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect Gamma levels (where options dealers hedge).
        
        Wall Street Use: Large options positions create support/resistance at strike prices.
        """
        if df.empty:
            return {}
        
        # Simulate gamma levels at round numbers and recent highs/lows
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Common strike prices (round numbers)
        if current_price >= 100:
            strike_increment = 5
        elif current_price >= 10:
            strike_increment = 1
        else:
            strike_increment = 0.1
        
        # Generate potential strike levels
        strikes = []
        base = (current_price // strike_increment) * strike_increment
        
        for i in range(-5, 6):
            strike = base + (i * strike_increment)
            if strike > 0:
                strikes.append(strike)
        
        # Add key levels
        strikes.extend([recent_high, recent_low])
        strikes = sorted(set(strikes))
        
        # Gamma levels (where most options are)
        # Typically at-the-money and near-the-money
        gamma_levels = [s for s in strikes if abs(s - current_price) / current_price < 0.10]  # Within 10%
        
        # Max pain (price where most options expire worthless)
        # Simplified: midpoint of recent range
        max_pain = (recent_high + recent_low) / 2
        
        return {
            'gamma_levels': [float(s) for s in gamma_levels],
            'max_pain': float(max_pain),
            'distance_to_max_pain': float(abs(current_price - max_pain) / current_price * 100),
            'strike_levels': [float(s) for s in strikes],
            'nearest_gamma': min([abs(s - current_price) for s in gamma_levels]) if gamma_levels else None
        }
    
    def analyze_implied_volatility(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Analyze implied volatility signals (simulated from realized volatility).
        
        Wall Street Use: High IV = expensive options, Low IV = cheap options.
        """
        if df.empty or len(df) < 20:
            return {}
        
        # Realized volatility (proxy for IV)
        returns = df['close'].pct_change().dropna()
        realized_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # Historical volatility range
        historical_vol = returns.rolling(60).std().dropna() * np.sqrt(252) if len(returns) >= 60 else pd.Series([realized_vol])
        
        vol_percentile = (realized_vol > historical_vol).sum() / len(historical_vol) if len(historical_vol) > 0 else 0.5
        
        # IV interpretation
        if vol_percentile > 0.8:
            iv_regime = 'VERY_HIGH'
            iv_signal = 'SELL_VOLATILITY'  # Options expensive
        elif vol_percentile > 0.6:
            iv_regime = 'HIGH'
            iv_signal = 'NEUTRAL'
        elif vol_percentile < 0.2:
            iv_regime = 'VERY_LOW'
            iv_signal = 'BUY_VOLATILITY'  # Options cheap
        elif vol_percentile < 0.4:
            iv_regime = 'LOW'
            iv_signal = 'NEUTRAL'
        else:
            iv_regime = 'NORMAL'
            iv_signal = 'NEUTRAL'
        
        return {
            'realized_volatility': float(realized_vol),
            'iv_percentile': float(vol_percentile),
            'iv_regime': iv_regime,
            'iv_signal': iv_signal,
            'options_valuation': 'EXPENSIVE' if iv_regime in ['VERY_HIGH', 'HIGH'] else 'CHEAP' if iv_regime in ['VERY_LOW', 'LOW'] else 'FAIR'
        }
    
    def comprehensive_options_flow(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Comprehensive options flow analysis."""
        if df.empty:
            return {}
        
        current_price = df['close'].iloc[-1]
        
        results = {
            'timestamp': datetime.now().isoformat()
        }
        
        # Put/Call ratio
        try:
            pc_ratio = self.simulate_put_call_ratio(df)
            results['put_call_ratio'] = pc_ratio
        except Exception as e:
            self.logger.debug(f"Error in put/call ratio: {e}")
        
        # Gamma levels
        try:
            gamma = self.detect_gamma_levels(current_price, df)
            results['gamma_levels'] = gamma
        except Exception as e:
            self.logger.debug(f"Error in gamma levels: {e}")
        
        # Implied volatility
        try:
            iv = self.analyze_implied_volatility(df)
            results['implied_volatility'] = iv
        except Exception as e:
            self.logger.debug(f"Error in IV analysis: {e}")
        
        # Overall options signal
        signals = []
        if 'put_call_ratio' in results and results['put_call_ratio'].get('signal') != 'NEUTRAL':
            signals.append(results['put_call_ratio']['signal'])
        
        if signals:
            buy_count = signals.count('BUY') + signals.count('WEAK_BUY')
            sell_count = signals.count('SELL') + signals.count('WEAK_SELL')
            
            if buy_count > sell_count:
                overall_signal = 'BUY' if buy_count >= 2 else 'WEAK_BUY'
            elif sell_count > buy_count:
                overall_signal = 'SELL' if sell_count >= 2 else 'WEAK_SELL'
            else:
                overall_signal = 'NEUTRAL'
        else:
            overall_signal = 'NEUTRAL'
        
        results['overall_options_signal'] = overall_signal
        
        return results

