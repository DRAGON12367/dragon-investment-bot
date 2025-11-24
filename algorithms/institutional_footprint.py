"""
Institutional Footprint Detection - Identify where big money is moving.
Wall Street Technique: Follow the institutions, not retail.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict


class InstitutionalFootprint:
    """
    Detect institutional trading activity and footprint.
    
    Techniques:
    1. Large Block Detection
    2. Unusual Volume Analysis
    3. Price Impact Analysis
    4. Accumulation/Distribution Patterns
    5. Institutional Support/Resistance Levels
    """
    
    def __init__(self):
        """Initialize institutional footprint detector."""
        self.logger = logging.getLogger("ai_investment_bot.institutional")
        self.footprint_history: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
    def detect_large_blocks(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect large block trades (institutional activity).
        
        Wall Street Use: Large blocks indicate institutional interest.
        """
        if df.empty or len(df) < 20:
            return {}
        
        # Calculate average volume
        avg_volume = volume.rolling(20).mean()
        
        # Large blocks (3x+ average volume)
        large_blocks = volume > (avg_volume * 3)
        very_large_blocks = volume > (avg_volume * 5)
        
        # Block direction (price movement on large volume)
        block_directions = []
        for idx in df[large_blocks].index:
            if idx > 0:
                price_change = df.loc[idx, 'close'] - df.loc[idx - 1, 'close']
                block_directions.append('BUY' if price_change > 0 else 'SELL')
        
        buy_blocks = block_directions.count('BUY')
        sell_blocks = block_directions.count('SELL')
        
        # Recent large blocks (last 5 periods)
        recent_large = large_blocks.tail(5).sum()
        recent_very_large = very_large_blocks.tail(5).sum()
        
        # Institutional activity level
        if recent_very_large > 0:
            activity = 'VERY_HIGH'
        elif recent_large >= 3:
            activity = 'HIGH'
        elif recent_large >= 1:
            activity = 'MODERATE'
        else:
            activity = 'LOW'
        
        # Net institutional flow
        net_flow = buy_blocks - sell_blocks
        flow_direction = 'ACCUMULATING' if net_flow > 0 else 'DISTRIBUTING' if net_flow < 0 else 'NEUTRAL'
        
        return {
            'large_blocks_count': int(large_blocks.sum()),
            'very_large_blocks_count': int(very_large_blocks.sum()),
            'recent_large_blocks': int(recent_large),
            'buy_blocks': buy_blocks,
            'sell_blocks': sell_blocks,
            'net_institutional_flow': net_flow,
            'flow_direction': flow_direction,
            'activity_level': activity,
            'institutional_presence': 'STRONG' if activity in ['HIGH', 'VERY_HIGH'] else 'WEAK'
        }
    
    def analyze_unusual_volume(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze unusual volume patterns (institutional accumulation/distribution).
        
        Wall Street Use: Unusual volume often precedes big moves.
        """
        if df.empty or len(df) < 50:
            return {}
        
        # Volume percentiles
        volume_percentiles = volume.rolling(50).quantile([0.25, 0.5, 0.75, 0.90, 0.95])
        
        current_volume = volume.iloc[-1]
        avg_volume = volume.rolling(50).mean().iloc[-1]
        
        # Volume classification
        if current_volume > volume_percentiles.iloc[-1, 4]:  # 95th percentile
            volume_type = 'EXTREME'
            significance = 'VERY_HIGH'
        elif current_volume > volume_percentiles.iloc[-1, 3]:  # 90th percentile
            volume_type = 'VERY_HIGH'
            significance = 'HIGH'
        elif current_volume > volume_percentiles.iloc[-1, 2]:  # 75th percentile
            volume_type = 'HIGH'
            significance = 'MODERATE'
        elif current_volume < volume_percentiles.iloc[-1, 0]:  # 25th percentile
            volume_type = 'LOW'
            significance = 'LOW'
        else:
            volume_type = 'NORMAL'
            significance = 'LOW'
        
        # Volume trend
        volume_trend = 'INCREASING' if volume.tail(5).mean() > volume.tail(20).head(5).mean() else 'DECREASING'
        
        # Price action on unusual volume
        price_change = df['close'].pct_change().iloc[-1]
        volume_ratio = current_volume / avg_volume if avg_volume > 0 else 1.0
        
        # Institutional interpretation
        if volume_type in ['EXTREME', 'VERY_HIGH'] and price_change > 0.02:
            interpretation = 'INSTITUTIONAL_ACCUMULATION'
            signal = 'BUY'
        elif volume_type in ['EXTREME', 'VERY_HIGH'] and price_change < -0.02:
            interpretation = 'INSTITUTIONAL_DISTRIBUTION'
            signal = 'SELL'
        elif volume_type in ['EXTREME', 'VERY_HIGH']:
            interpretation = 'INSTITUTIONAL_INTEREST'
            signal = 'WATCH'
        else:
            interpretation = 'NORMAL_ACTIVITY'
            signal = 'NEUTRAL'
        
        return {
            'volume_type': volume_type,
            'volume_ratio': float(volume_ratio),
            'significance': significance,
            'volume_trend': volume_trend,
            'interpretation': interpretation,
            'signal': signal,
            'price_change_on_volume': float(price_change * 100)
        }
    
    def detect_accumulation_zones(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Detect institutional accumulation zones.
        
        Wall Street Use: Institutions accumulate positions over time, not all at once.
        """
        if df.empty or len(df) < 30:
            return {}
        
        # Look for price consolidation with increasing volume
        price_range = df['high'].tail(20).max() - df['low'].tail(20).min()
        price_range_pct = price_range / df['close'].iloc[-1]
        
        # Consolidation = small price range
        is_consolidating = price_range_pct < 0.05  # Less than 5% range
        
        # Volume increasing during consolidation
        volume_increasing = volume.tail(5).mean() > volume.tail(20).head(5).mean()
        
        # Price near recent lows (accumulation zone)
        recent_low = df['low'].tail(20).min()
        current_price = df['close'].iloc[-1]
        near_low = abs(current_price - recent_low) / recent_low < 0.03  # Within 3%
        
        # Accumulation detected
        accumulation = is_consolidating and volume_increasing and near_low
        
        # Distribution zone (opposite)
        recent_high = df['high'].tail(20).max()
        near_high = abs(current_price - recent_high) / recent_high < 0.03
        distribution = is_consolidating and volume_increasing and near_high
        
        return {
            'accumulation_zone': accumulation,
            'distribution_zone': distribution,
            'consolidation_detected': is_consolidating,
            'price_range_pct': float(price_range_pct * 100),
            'zone_signal': 'ACCUMULATE' if accumulation else 'DISTRIBUTE' if distribution else 'NEUTRAL'
        }
    
    def identify_institutional_levels(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Identify key institutional support/resistance levels.
        
        Wall Street Use: Institutions defend certain price levels.
        """
        if df.empty or len(df) < 50:
            return {}
        
        # High volume nodes (where most trading happened)
        # These become support/resistance
        volume_profile = []
        price_bins = np.linspace(df['low'].min(), df['high'].max(), 20)
        
        for i in range(len(price_bins) - 1):
            bin_low = price_bins[i]
            bin_high = price_bins[i + 1]
            mask = (df['low'] <= bin_high) & (df['high'] >= bin_low)
            bin_volume = volume[mask].sum()
            volume_profile.append({
                'price': (bin_low + bin_high) / 2,
                'volume': bin_volume
            })
        
        # Find high volume nodes (POC - Point of Control)
        volume_profile_df = pd.DataFrame(volume_profile)
        poc_price = volume_profile_df.loc[volume_profile_df['volume'].idxmax(), 'price']
        
        # Value area (70% of volume)
        total_volume = volume_profile_df['volume'].sum()
        volume_profile_df = volume_profile_df.sort_values('volume', ascending=False)
        cumulative_volume = volume_profile_df['volume'].cumsum()
        value_area = volume_profile_df[cumulative_volume <= total_volume * 0.70]
        
        value_area_high = value_area['price'].max()
        value_area_low = value_area['price'].min()
        
        # Current price position
        current_price = df['close'].iloc[-1]
        if current_price > value_area_high:
            position = 'ABOVE_VALUE_AREA'
        elif current_price < value_area_low:
            position = 'BELOW_VALUE_AREA'
        else:
            position = 'IN_VALUE_AREA'
        
        return {
            'poc_price': float(poc_price),
            'value_area_high': float(value_area_high),
            'value_area_low': float(value_area_low),
            'current_position': position,
            'distance_to_poc': float(abs(current_price - poc_price) / poc_price * 100),
            'institutional_level': poc_price
        }
    
    def comprehensive_footprint(
        self,
        df: pd.DataFrame,
        volume: Optional[pd.Series] = None
    ) -> Dict[str, Any]:
        """Comprehensive institutional footprint analysis."""
        if df.empty:
            return {}
        
        if volume is None:
            volume = df.get('volume', pd.Series([0] * len(df)))
        
        results = {
            'timestamp': datetime.now().isoformat()
        }
        
        # Large blocks
        try:
            blocks = self.detect_large_blocks(df, volume)
            results['large_blocks'] = blocks
        except Exception as e:
            self.logger.debug(f"Error in large block detection: {e}")
        
        # Unusual volume
        try:
            unusual_vol = self.analyze_unusual_volume(df, volume)
            results['unusual_volume'] = unusual_vol
        except Exception as e:
            self.logger.debug(f"Error in unusual volume analysis: {e}")
        
        # Accumulation zones
        try:
            zones = self.detect_accumulation_zones(df, volume)
            results['accumulation_zones'] = zones
        except Exception as e:
            self.logger.debug(f"Error in accumulation zone detection: {e}")
        
        # Institutional levels
        try:
            levels = self.identify_institutional_levels(df, volume)
            results['institutional_levels'] = levels
        except Exception as e:
            self.logger.debug(f"Error in institutional level identification: {e}")
        
        # Overall institutional signal
        signals = []
        if 'large_blocks' in results and results['large_blocks'].get('flow_direction') == 'ACCUMULATING':
            signals.append('BUY')
        if 'unusual_volume' in results and results['unusual_volume'].get('signal') == 'BUY':
            signals.append('BUY')
        if 'accumulation_zones' in results and results['accumulation_zones'].get('zone_signal') == 'ACCUMULATE':
            signals.append('BUY')
        
        buy_count = signals.count('BUY')
        if buy_count >= 2:
            overall_signal = 'STRONG_BUY'
        elif buy_count == 1:
            overall_signal = 'BUY'
        else:
            overall_signal = 'NEUTRAL'
        
        results['overall_institutional_signal'] = overall_signal
        results['institutional_confidence'] = buy_count / 3.0 if buy_count > 0 else 0.0
        
        return results

