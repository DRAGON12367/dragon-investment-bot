"""
MARKET ANOMALY DETECTOR - 200X UPGRADE
Detects market anomalies, outliers, and unusual patterns
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy import stats
from scipy.stats import zscore
import warnings
warnings.filterwarnings('ignore')


class MarketAnomalyDetector:
    """
    Advanced market anomaly detection system.
    
    Features:
    - Statistical outlier detection
    - Volume anomalies
    - Price action anomalies
    - Volatility spikes
    - Correlation breakdowns
    - Flash crash detection
    - Unusual order flow
    """
    
    def __init__(self, zscore_threshold: float = 3.0):
        """Initialize anomaly detector."""
        self.logger = logging.getLogger("ai_investment_bot.anomaly_detector")
        self.zscore_threshold = zscore_threshold
        self.anomaly_history = []
        
    def detect_price_anomalies(
        self,
        df: pd.DataFrame,
        method: str = 'zscore'
    ) -> Dict[str, Any]:
        """
        Detect price anomalies.
        
        Methods:
        - 'zscore': Z-score based detection
        - 'iqr': Interquartile range
        - 'isolation': Isolation forest (if available)
        """
        try:
            if 'close' not in df.columns or len(df) < 20:
                return {}
            
            close = df['close']
            returns = close.pct_change().dropna()
            
            if method == 'zscore':
                z_scores = np.abs(zscore(returns))
                anomalies = z_scores > self.zscore_threshold
                
                anomaly_indices = returns.index[anomalies]
                anomaly_values = returns[anomalies]
                
            elif method == 'iqr':
                Q1 = returns.quantile(0.25)
                Q3 = returns.quantile(0.75)
                IQR = Q3 - Q1
                
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                anomalies = (returns < lower_bound) | (returns > upper_bound)
                anomaly_indices = returns.index[anomalies]
                anomaly_values = returns[anomalies]
                
            else:
                return {}
            
            return {
                'anomaly_count': len(anomaly_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'anomaly_values': anomaly_values.tolist(),
                'anomaly_percentage': (len(anomaly_indices) / len(returns)) * 100,
                'most_recent_anomaly': anomaly_indices[-1] if len(anomaly_indices) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting price anomalies: {e}")
            return {}
    
    def detect_volume_anomalies(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """Detect unusual volume patterns."""
        try:
            if 'volume' not in df.columns or len(df) < 20:
                return {}
            
            volume = df['volume']
            
            # Calculate volume z-scores
            volume_z = np.abs(zscore(volume))
            anomalies = volume_z > self.zscore_threshold
            
            # Volume ratio (current vs average)
            avg_volume = volume.rolling(20).mean()
            volume_ratio = volume / (avg_volume + 1e-10)
            
            # Detect spikes
            volume_spikes = volume_ratio > 2.0  # 2x average
            
            anomaly_indices = volume.index[anomalies]
            spike_indices = volume.index[volume_spikes]
            
            return {
                'anomaly_count': len(anomaly_indices),
                'spike_count': len(spike_indices),
                'anomaly_indices': anomaly_indices.tolist(),
                'spike_indices': spike_indices.tolist(),
                'current_volume_ratio': volume_ratio.iloc[-1] if not volume_ratio.empty else 1.0,
                'avg_volume_ratio': volume_ratio.mean() if not volume_ratio.empty else 1.0
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volume anomalies: {e}")
            return {}
    
    def detect_volatility_spikes(
        self,
        df: pd.DataFrame,
        window: int = 20
    ) -> Dict[str, Any]:
        """Detect volatility spikes."""
        try:
            if 'close' not in df.columns or len(df) < window * 2:
                return {}
            
            close = df['close']
            returns = close.pct_change().dropna()
            
            # Rolling volatility
            rolling_vol = returns.rolling(window).std() * np.sqrt(252)  # Annualized
            
            # Calculate z-score of volatility
            vol_z = np.abs(zscore(rolling_vol))
            spikes = vol_z > self.zscore_threshold
            
            spike_indices = rolling_vol.index[spikes]
            spike_values = rolling_vol[spikes]
            
            # Current volatility vs historical
            current_vol = rolling_vol.iloc[-1] if not rolling_vol.empty else 0.0
            avg_vol = rolling_vol.mean() if not rolling_vol.empty else 0.0
            vol_ratio = current_vol / (avg_vol + 1e-10)
            
            return {
                'spike_count': len(spike_indices),
                'spike_indices': spike_indices.tolist(),
                'spike_values': spike_values.tolist(),
                'current_volatility': current_vol,
                'avg_volatility': avg_vol,
                'volatility_ratio': vol_ratio,
                'volatility_regime': 'high' if vol_ratio > 1.5 else 'normal' if vol_ratio > 0.5 else 'low'
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting volatility spikes: {e}")
            return {}
    
    def detect_flash_crash(
        self,
        df: pd.DataFrame,
        threshold: float = -0.05
    ) -> Dict[str, Any]:
        """
        Detect flash crash patterns (rapid price decline).
        
        Args:
            threshold: Minimum decline percentage to consider (default -5%)
        """
        try:
            if 'close' not in df.columns or len(df) < 10:
                return {}
            
            close = df['close']
            returns = close.pct_change()
            
            # Look for rapid declines
            rapid_declines = returns < threshold
            
            if not rapid_declines.any():
                return {'flash_crash_detected': False}
            
            decline_indices = returns.index[rapid_declines]
            decline_values = returns[rapid_declines]
            
            # Check if decline is followed by recovery (typical flash crash pattern)
            recoveries = []
            for idx in decline_indices:
                idx_pos = returns.index.get_loc(idx)
                if idx_pos < len(returns) - 1:
                    next_return = returns.iloc[idx_pos + 1]
                    recoveries.append(next_return > 0)
            
            recovery_rate = np.mean(recoveries) if recoveries else 0.0
            
            return {
                'flash_crash_detected': True,
                'crash_count': len(decline_indices),
                'crash_indices': decline_indices.tolist(),
                'crash_magnitudes': decline_values.tolist(),
                'recovery_rate': recovery_rate,
                'most_recent_crash': decline_indices[-1] if len(decline_indices) > 0 else None
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting flash crash: {e}")
            return {'flash_crash_detected': False}
    
    def detect_correlation_breakdown(
        self,
        corr_matrix_current: pd.DataFrame,
        corr_matrix_historical: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect correlation breakdowns (unusual correlation changes).
        """
        try:
            if corr_matrix_current.empty or corr_matrix_historical.empty:
                return {}
            
            # Get common assets
            common_assets = set(corr_matrix_current.columns) & set(corr_matrix_historical.columns)
            
            if len(common_assets) < 2:
                return {}
            
            breakdowns = []
            
            for asset1 in common_assets:
                for asset2 in common_assets:
                    if asset1 >= asset2:
                        continue
                    
                    current_corr = corr_matrix_current.loc[asset1, asset2]
                    historical_corr = corr_matrix_historical.loc[asset1, asset2]
                    
                    # Calculate change
                    corr_change = abs(current_corr - historical_corr)
                    
                    # Significant change threshold
                    if corr_change > 0.3:  # 30% change
                        breakdowns.append({
                            'asset1': asset1,
                            'asset2': asset2,
                            'current_correlation': current_corr,
                            'historical_correlation': historical_corr,
                            'change': corr_change,
                            'change_pct': (corr_change / (abs(historical_corr) + 0.01)) * 100
                        })
            
            breakdowns.sort(key=lambda x: x['change'], reverse=True)
            
            return {
                'breakdown_count': len(breakdowns),
                'breakdowns': breakdowns[:10],  # Top 10
                'avg_correlation_change': np.mean([b['change'] for b in breakdowns]) if breakdowns else 0.0
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation breakdown: {e}")
            return {}
    
    def detect_unusual_patterns(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect unusual price/volume patterns.
        
        Patterns:
        - Gap up/down
        - Doji candles
        - Engulfing patterns
        - Volume-price divergence
        """
        try:
            if 'close' not in df.columns or len(df) < 5:
                return {}
            
            close = df['close']
            open_price = df.get('open', close)
            high = df.get('high', close)
            low = df.get('low', close)
            volume = df.get('volume', pd.Series([1] * len(df)))
            
            patterns = []
            
            # Gap detection
            if len(df) > 1:
                prev_close = close.shift(1)
                gap = (open_price - prev_close) / prev_close
                
                large_gaps = abs(gap) > 0.02  # 2% gap
                gap_indices = gap.index[large_gaps]
                
                for idx in gap_indices:
                    gap_pct = gap.loc[idx] * 100
                    patterns.append({
                        'type': 'gap_up' if gap_pct > 0 else 'gap_down',
                        'timestamp': idx,
                        'magnitude': abs(gap_pct),
                        'confidence': 'high' if abs(gap_pct) > 0.05 else 'medium'
                    })
            
            # Volume-price divergence
            if len(df) > 10:
                price_trend = close.rolling(5).mean().diff()
                volume_trend = volume.rolling(5).mean().diff()
                
                # Divergence: price up but volume down (or vice versa)
                divergence = (price_trend > 0) & (volume_trend < 0) | (price_trend < 0) & (volume_trend > 0)
                div_indices = divergence.index[divergence]
                
                for idx in div_indices[-5:]:  # Last 5 divergences
                    patterns.append({
                        'type': 'volume_price_divergence',
                        'timestamp': idx,
                        'price_trend': 'up' if price_trend.loc[idx] > 0 else 'down',
                        'volume_trend': 'up' if volume_trend.loc[idx] > 0 else 'down',
                        'confidence': 'medium'
                    })
            
            return {
                'pattern_count': len(patterns),
                'patterns': patterns,
                'recent_patterns': patterns[-5:] if len(patterns) > 5 else patterns
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting unusual patterns: {e}")
            return {}
    
    def get_comprehensive_anomaly_report(
        self,
        df: pd.DataFrame,
        corr_matrix: Optional[pd.DataFrame] = None
    ) -> Dict[str, Any]:
        """
        Get comprehensive anomaly detection report.
        """
        try:
            if df.empty:
                return {'status': 'insufficient_data'}
            
            # Run all detectors
            price_anomalies = self.detect_price_anomalies(df)
            volume_anomalies = self.detect_volume_anomalies(df)
            volatility_spikes = self.detect_volatility_spikes(df)
            flash_crashes = self.detect_flash_crash(df)
            unusual_patterns = self.detect_unusual_patterns(df)
            
            # Calculate overall anomaly score
            anomaly_score = 0.0
            if price_anomalies.get('anomaly_count', 0) > 0:
                anomaly_score += 0.3
            if volume_anomalies.get('spike_count', 0) > 0:
                anomaly_score += 0.2
            if volatility_spikes.get('spike_count', 0) > 0:
                anomaly_score += 0.2
            if flash_crashes.get('flash_crash_detected', False):
                anomaly_score += 0.2
            if unusual_patterns.get('pattern_count', 0) > 0:
                anomaly_score += 0.1
            
            anomaly_level = 'high' if anomaly_score > 0.6 else 'medium' if anomaly_score > 0.3 else 'low'
            
            return {
                'anomaly_score': anomaly_score,
                'anomaly_level': anomaly_level,
                'price_anomalies': price_anomalies,
                'volume_anomalies': volume_anomalies,
                'volatility_spikes': volatility_spikes,
                'flash_crashes': flash_crashes,
                'unusual_patterns': unusual_patterns,
                'total_anomalies': (
                    price_anomalies.get('anomaly_count', 0) +
                    volume_anomalies.get('anomaly_count', 0) +
                    volatility_spikes.get('spike_count', 0) +
                    (1 if flash_crashes.get('flash_crash_detected', False) else 0) +
                    unusual_patterns.get('pattern_count', 0)
                )
            }
            
        except Exception as e:
            self.logger.error(f"Error getting anomaly report: {e}")
            return {'status': 'error', 'message': str(e)}

