"""
Market Maker Levels - Identify where market makers set support/resistance.
Wall Street Technique: Market makers defend certain price levels.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class MarketMakerLevels:
    """
    Detect market maker support and resistance levels.
    
    Techniques:
    1. Round Number Levels (psychological levels)
    2. High Volume Nodes
    3. Price Rejection Levels
    4. Liquidity Pools
    5. Stop Loss Clusters
    """
    
    def __init__(self):
        """Initialize market maker level detector."""
        self.logger = logging.getLogger("ai_investment_bot.market_maker")
        self.level_history: Dict[str, List[float]] = defaultdict(list)
        
    def identify_round_numbers(
        self,
        current_price: float
    ) -> Dict[str, Any]:
        """
        Identify round number psychological levels.
        
        Wall Street Use: Market makers often set stops/limits at round numbers.
        """
        # Find nearest round numbers
        if current_price >= 1000:
            round_increment = 100
        elif current_price >= 100:
            round_increment = 10
        elif current_price >= 10:
            round_increment = 1
        elif current_price >= 1:
            round_increment = 0.1
        else:
            round_increment = 0.01
        
        # Nearest round numbers above and below
        below = (current_price // round_increment) * round_increment
        above = below + round_increment
        
        # Key levels (multiples of 5 or 10)
        key_levels = []
        for multiplier in [0.5, 1, 2, 5, 10]:
            level = round_increment * multiplier
            if level < current_price * 0.5:  # Not too far
                key_levels.append(level)
        
        return {
            'nearest_below': float(below),
            'nearest_above': float(above),
            'key_levels': [float(l) for l in key_levels],
            'round_increment': float(round_increment)
        }
    
    def detect_price_rejections(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect price rejection levels (where price bounced).
        
        Wall Street Use: Rejections indicate strong support/resistance.
        """
        if df.empty or len(df) < 20:
            return {}
        
        # Find wicks (price rejection)
        # Long lower wick = support rejection
        # Long upper wick = resistance rejection
        
        body_size = abs(df['close'] - df['open'])
        lower_wick = df[['open', 'close']].min(axis=1) - df['low']
        upper_wick = df['high'] - df[['open', 'close']].max(axis=1)
        
        # Significant wicks (wick > 2x body)
        significant_lower_wicks = lower_wick > (body_size * 2)
        significant_upper_wicks = upper_wick > (body_size * 2)
        
        # Support levels (where price rejected lower)
        support_levels = df.loc[significant_lower_wicks, 'low'].tolist()
        
        # Resistance levels (where price rejected higher)
        resistance_levels = df.loc[significant_upper_wicks, 'high'].tolist()
        
        # Current price position
        current_price = df['close'].iloc[-1]
        
        # Nearest support/resistance
        supports_below = [s for s in support_levels if s < current_price]
        resistances_above = [r for r in resistance_levels if r > current_price]
        
        nearest_support = max(supports_below) if supports_below else None
        nearest_resistance = min(resistances_above) if resistances_above else None
        
        return {
            'support_levels': sorted(set(support_levels))[-5:],  # Last 5
            'resistance_levels': sorted(set(resistance_levels))[-5:],
            'nearest_support': float(nearest_support) if nearest_support else None,
            'nearest_resistance': float(nearest_resistance) if nearest_resistance else None,
            'support_distance_pct': float((current_price - nearest_support) / current_price * 100) if nearest_support else None,
            'resistance_distance_pct': float((nearest_resistance - current_price) / current_price * 100) if nearest_resistance else None
        }
    
    def identify_liquidity_pools(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Identify liquidity pools (where stops are likely clustered).
        
        Wall Street Use: Market makers target liquidity pools.
        """
        if df.empty or len(df) < 30:
            return {}
        
        # Liquidity pools = areas with many stop losses
        # Typically: recent highs/lows, round numbers, support/resistance
        
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        
        # Stop loss clusters (above recent highs, below recent lows)
        # Bullish: stops above recent high
        # Bearish: stops below recent low
        
        bullish_liquidity = recent_high * 1.01  # 1% above high
        bearish_liquidity = recent_low * 0.99  # 1% below low
        
        # Liquidity zones
        liquidity_zones = []
        
        # Above current price (resistance liquidity)
        if current_price < recent_high:
            liquidity_zones.append({
                'price': float(bullish_liquidity),
                'type': 'RESISTANCE_LIQUIDITY',
                'strength': 'HIGH'
            })
        
        # Below current price (support liquidity)
        if current_price > recent_low:
            liquidity_zones.append({
                'price': float(bearish_liquidity),
                'type': 'SUPPORT_LIQUIDITY',
                'strength': 'HIGH'
            })
        
        return {
            'liquidity_pools': liquidity_zones,
            'bullish_liquidity': float(bullish_liquidity),
            'bearish_liquidity': float(bearish_liquidity),
            'liquidity_count': len(liquidity_zones)
        }
    
    def comprehensive_levels(
        self,
        df: pd.DataFrame,
        volume: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive market maker level analysis."""
        if df.empty:
            return {}
        
        current_price = df['close'].iloc[-1]
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'current_price': float(current_price)
        }
        
        # Round numbers
        try:
            round_nums = self.identify_round_numbers(current_price)
            results['round_numbers'] = round_nums
        except Exception as e:
            self.logger.debug(f"Error in round number identification: {e}")
        
        # Price rejections
        try:
            rejections = self.detect_price_rejections(df)
            results['rejections'] = rejections
        except Exception as e:
            self.logger.debug(f"Error in rejection detection: {e}")
        
        # Liquidity pools
        try:
            liquidity = self.identify_liquidity_pools(df, volume if volume is not None else df.get('volume', pd.Series([0] * len(df))))
            results['liquidity_pools'] = liquidity
        except Exception as e:
            self.logger.debug(f"Error in liquidity pool identification: {e}")
        
        # Key levels summary
        key_levels = []
        if 'round_numbers' in results:
            key_levels.extend(results['round_numbers'].get('key_levels', []))
        if 'rejections' in results:
            if results['rejections'].get('nearest_support'):
                key_levels.append(results['rejections']['nearest_support'])
            if results['rejections'].get('nearest_resistance'):
                key_levels.append(results['rejections']['nearest_resistance'])
        
        results['key_levels'] = sorted(set(key_levels))
        results['nearest_key_level'] = min([abs(l - current_price) for l in key_levels]) if key_levels else None
        
        return results

