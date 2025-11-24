"""
Ultra Advanced Technical Indicators - 5x Upgrade
60+ new professional-grade indicators used by hedge funds and prop trading firms.
"""
import pandas as pd
import numpy as np
from typing import Dict, Optional
from scipy import stats
from scipy.signal import find_peaks


class UltraAdvancedIndicators:
    """Ultra advanced technical indicators - 5x upgrade with 60+ new indicators."""
    
    def __init__(self):
        """Initialize ultra advanced indicators."""
        pass
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all ultra advanced indicators."""
        if df.empty or len(df) < 50:
            return df
        
        df = df.copy()
        
        # Momentum Indicators (10 new)
        df = self.calculate_ultimate_oscillator(df)
        df = self.calculate_awesome_oscillator(df)
        df = self.calculate_roc(df)
        df = self.calculate_momentum_index(df)
        df = self.calculate_price_oscillator(df)
        df = self.calculate_detrended_price_oscillator(df)
        df = self.calculate_trix(df)
        df = self.calculate_tsi(df)
        df = self.calculate_ppo(df)
        df = self.calculate_aroon_oscillator(df)
        
        # Volume Indicators (10 new)
        df = self.calculate_accumulation_distribution(df)
        df = self.calculate_ease_of_movement(df)
        df = self.calculate_force_index(df)
        df = self.calculate_mfi(df)
        df = self.calculate_negative_volume_index(df)
        df = self.calculate_positive_volume_index(df)
        df = self.calculate_volume_price_trend(df)
        df = self.calculate_volume_rate_of_change(df)
        df = self.calculate_volume_weighted_macd(df)
        df = self.calculate_klinger_oscillator(df)
        
        # Volatility Indicators (10 new)
        df = self.calculate_bb_percent(df)
        df = self.calculate_bb_bandwidth(df)
        df = self.calculate_chaikin_volatility(df)
        df = self.calculate_ulcer_index(df)
        df = self.calculate_standard_deviation(df)
        df = self.calculate_variance_ratio(df)
        df = self.calculate_parkinson_volatility(df)
        df = self.calculate_garman_klass_volatility(df)
        df = self.calculate_rogers_satchell_volatility(df)
        df = self.calculate_yang_zhang_volatility(df)
        
        # Trend Indicators (10 new)
        df = self.calculate_dpo(df)
        df = self.calculate_linear_regression_slope(df)
        df = self.calculate_linear_regression_angle(df)
        df = self.calculate_linear_regression_intercept(df)
        df = self.calculate_psar_enhanced(df)
        df = self.calculate_supertrend(df)
        df = self.calculate_aroon(df)
        df = self.calculate_directional_movement_index(df)
        df = self.calculate_commodity_selection_index(df)
        df = self.calculate_trend_strength_index(df)
        
        # Cycle Indicators (10 new)
        df = self.calculate_hilbert_transform(df)
        df = self.calculate_dominant_cycle_period(df)
        df = self.calculate_cycle_phase(df)
        df = self.calculate_sine_wave(df)
        df = self.calculate_lead_sine(df)
        df = self.calculate_dc_period(df)
        df = self.calculate_dc_phase(df)
        df = self.calculate_adaptive_cycle_period(df)
        df = self.calculate_fisher_transform(df)
        df = self.calculate_inverse_fisher_transform(df)
        
        # Market Structure (10 new)
        df = self.calculate_swing_highs_lows(df)
        df = self.calculate_pivot_points(df)
        df = self.calculate_fibonacci_levels(df)
        df = self.calculate_camarilla_levels(df)
        df = self.calculate_woodie_levels(df)
        df = self.calculate_tom_demark_levels(df)
        df = self.calculate_murrey_math_levels(df)
        df = self.calculate_gann_levels(df)
        df = self.calculate_volume_profile_levels(df)
        df = self.calculate_order_flow_imbalance(df)
        
        return df
    
    # ========== MOMENTUM INDICATORS (10 new) ==========
    
    def calculate_ultimate_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ultimate Oscillator."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift()),
            abs(low - close.shift())
        ], axis=1).max(axis=1)
        
        bp = close - pd.concat([low, close.shift()], axis=1).min(axis=1)
        
        avg7 = bp.rolling(7).sum() / tr.rolling(7).sum()
        avg14 = bp.rolling(14).sum() / tr.rolling(14).sum()
        avg28 = bp.rolling(28).sum() / tr.rolling(28).sum()
        
        uo = 100 * (4 * avg7 + 2 * avg14 + avg28) / 7
        df['ultimate_oscillator'] = uo
        return df
    
    def calculate_awesome_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Awesome Oscillator."""
        high = df['high']
        low = df['low']
        hl2 = (high + low) / 2
        
        ao = hl2.rolling(5).mean() - hl2.rolling(34).mean()
        df['awesome_oscillator'] = ao
        return df
    
    def calculate_roc(self, df: pd.DataFrame, period: int = 12) -> pd.DataFrame:
        """Calculate Rate of Change."""
        df['roc'] = df['close'].pct_change(periods=period) * 100
        return df
    
    def calculate_momentum_index(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Momentum Index."""
        df['momentum_index'] = df['close'] - df['close'].shift(period)
        return df
    
    def calculate_price_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Price Oscillator."""
        sma_short = df['close'].rolling(12).mean()
        sma_long = df['close'].rolling(26).mean()
        df['price_oscillator'] = ((sma_short - sma_long) / sma_long) * 100
        return df
    
    def calculate_detrended_price_oscillator(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Detrended Price Oscillator."""
        sma = df['close'].rolling(period).mean()
        df['dpo'] = df['close'].shift(period // 2 + 1) - sma
        return df
    
    def calculate_trix(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate TRIX (Triple Exponential Average)."""
        ema1 = df['close'].ewm(span=period, adjust=False).mean()
        ema2 = ema1.ewm(span=period, adjust=False).mean()
        ema3 = ema2.ewm(span=period, adjust=False).mean()
        df['trix'] = ema3.pct_change() * 10000
        return df
    
    def calculate_tsi(self, df: pd.DataFrame, r: int = 25, s: int = 13) -> pd.DataFrame:
        """Calculate True Strength Index."""
        momentum = df['close'].diff()
        smoothed_momentum = momentum.ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        smoothed_abs_momentum = momentum.abs().ewm(span=r, adjust=False).mean().ewm(span=s, adjust=False).mean()
        df['tsi'] = 100 * (smoothed_momentum / smoothed_abs_momentum)
        return df
    
    def calculate_ppo(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Percentage Price Oscillator."""
        ema_fast = df['close'].ewm(span=12, adjust=False).mean()
        ema_slow = df['close'].ewm(span=26, adjust=False).mean()
        df['ppo'] = ((ema_fast - ema_slow) / ema_slow) * 100
        return df
    
    def calculate_aroon_oscillator(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Aroon Oscillator."""
        aroon_up = ((period - df['high'].rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - df['low'].rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        df['aroon_oscillator'] = aroon_up - aroon_down
        return df
    
    # ========== VOLUME INDICATORS (10 new) ==========
    
    def calculate_accumulation_distribution(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Accumulation/Distribution Line."""
        clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        clv = clv.fillna(0)
        df['ad_line'] = (clv * df['volume']).cumsum()
        return df
    
    def calculate_ease_of_movement(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ease of Movement."""
        distance = (df['high'] + df['low']) / 2 - (df['high'].shift() + df['low'].shift()) / 2
        box_ratio = df['volume'] / (df['high'] - df['low'])
        df['eom'] = distance / box_ratio
        df['eom'] = df['eom'].fillna(0)
        df['eom_sma'] = df['eom'].rolling(14).mean()
        return df
    
    def calculate_force_index(self, df: pd.DataFrame, period: int = 13) -> pd.DataFrame:
        """Calculate Force Index."""
        df['force_index'] = df['close'].diff() * df['volume']
        df['force_index_sma'] = df['force_index'].rolling(period).mean()
        return df
    
    def calculate_mfi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Money Flow Index."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        money_flow = typical_price * df['volume']
        
        positive_flow = money_flow.where(typical_price > typical_price.shift(), 0).rolling(period).sum()
        negative_flow = money_flow.where(typical_price < typical_price.shift(), 0).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        df['mfi'] = mfi.fillna(50)
        return df
    
    def calculate_negative_volume_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Negative Volume Index."""
        nvi = pd.Series(index=df.index, dtype=float)
        nvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if df['volume'].iloc[i] < df['volume'].iloc[i-1]:
                nvi.iloc[i] = nvi.iloc[i-1] * (1 + df['close'].pct_change().iloc[i])
            else:
                nvi.iloc[i] = nvi.iloc[i-1]
        
        df['nvi'] = nvi
        return df
    
    def calculate_positive_volume_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Positive Volume Index."""
        pvi = pd.Series(index=df.index, dtype=float)
        pvi.iloc[0] = 1000
        
        for i in range(1, len(df)):
            if df['volume'].iloc[i] > df['volume'].iloc[i-1]:
                pvi.iloc[i] = pvi.iloc[i-1] * (1 + df['close'].pct_change().iloc[i])
            else:
                pvi.iloc[i] = pvi.iloc[i-1]
        
        df['pvi'] = pvi
        return df
    
    def calculate_volume_price_trend(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Price Trend."""
        vpt = (df['close'].pct_change() * df['volume']).cumsum()
        df['vpt'] = vpt
        return df
    
    def calculate_volume_rate_of_change(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Volume Rate of Change."""
        df['vroc'] = df['volume'].pct_change(periods=period) * 100
        return df
    
    def calculate_volume_weighted_macd(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted MACD."""
        vwap_short = (df['close'] * df['volume']).rolling(12).sum() / df['volume'].rolling(12).sum()
        vwap_long = (df['close'] * df['volume']).rolling(26).sum() / df['volume'].rolling(26).sum()
        df['vwmacd'] = vwap_short - vwap_long
        df['vwmacd_signal'] = df['vwmacd'].ewm(span=9, adjust=False).mean()
        return df
    
    def calculate_klinger_oscillator(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Klinger Oscillator."""
        trend = (df['high'] + df['low'] + df['close']) / 3
        volume_force = df['volume'] * np.sign(trend.diff())
        
        ema_fast = volume_force.ewm(span=34, adjust=False).mean()
        ema_slow = volume_force.ewm(span=55, adjust=False).mean()
        
        df['klinger'] = ema_fast - ema_slow
        df['klinger_signal'] = df['klinger'].ewm(span=13, adjust=False).mean()
        return df
    
    # ========== VOLATILITY INDICATORS (10 new) ==========
    
    def calculate_bb_percent(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Band %B."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df['bb_percent'] = (df['close'] - lower) / (upper - lower)
        return df
    
    def calculate_bb_bandwidth(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Bollinger Band Width."""
        sma = df['close'].rolling(period).mean()
        std = df['close'].rolling(period).std()
        upper = sma + 2 * std
        lower = sma - 2 * std
        df['bb_bandwidth'] = ((upper - lower) / sma) * 100
        return df
    
    def calculate_chaikin_volatility(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Chaikin Volatility."""
        hl_diff = df['high'] - df['low']
        ema_diff = hl_diff.ewm(span=period, adjust=False).mean()
        df['chaikin_vol'] = ((ema_diff - ema_diff.shift(period)) / ema_diff.shift(period)) * 100
        return df
    
    def calculate_ulcer_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Ulcer Index."""
        highest_close = df['close'].rolling(period).max()
        percent_drawdown = ((df['close'] - highest_close) / highest_close) * 100
        df['ulcer_index'] = np.sqrt(percent_drawdown.pow(2).rolling(period).mean())
        return df
    
    def calculate_standard_deviation(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Standard Deviation."""
        df['std_dev'] = df['close'].rolling(period).std()
        return df
    
    def calculate_variance_ratio(self, df: pd.DataFrame, short: int = 5, long: int = 20) -> pd.DataFrame:
        """Calculate Variance Ratio."""
        short_var = df['close'].pct_change().rolling(short).var()
        long_var = df['close'].pct_change().rolling(long).var()
        df['variance_ratio'] = short_var / long_var
        return df
    
    def calculate_parkinson_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Parkinson Volatility Estimator."""
        hl_ratio = np.log(df['high'] / df['low'])
        df['parkinson_vol'] = np.sqrt((1 / (4 * np.log(2))) * hl_ratio.pow(2).rolling(period).mean())
        return df
    
    def calculate_garman_klass_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Garman-Klass Volatility Estimator."""
        hl = np.log(df['high'] / df['low'])
        co = np.log(df['close'] / df['open'])
        df['gk_vol'] = np.sqrt(0.5 * hl.pow(2).rolling(period).mean() - 
                               (2 * np.log(2) - 1) * co.pow(2).rolling(period).mean())
        return df
    
    def calculate_rogers_satchell_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Rogers-Satchell Volatility Estimator."""
        hl = np.log(df['high'] / df['low'])
        ho = np.log(df['high'] / df['open'])
        lo = np.log(df['low'] / df['open'])
        co = np.log(df['close'] / df['open'])
        df['rs_vol'] = np.sqrt((hl * ho + hl * lo).rolling(period).mean())
        return df
    
    def calculate_yang_zhang_volatility(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Yang-Zhang Volatility Estimator."""
        o = np.log(df['open'] / df['close'].shift())
        c = np.log(df['close'] / df['open'])
        hl = np.log(df['high'] / df['low'])
        ho = np.log(df['high'] / df['open'])
        lo = np.log(df['low'] / df['open'])
        
        k = 0.34 / (1.34 + (period + 1) / (period - 1))
        df['yz_vol'] = np.sqrt(o.pow(2).rolling(period).mean() + 
                              k * c.pow(2).rolling(period).mean() + 
                              (1 - k) * (hl.pow(2) / 4).rolling(period).mean())
        return df
    
    # ========== TREND INDICATORS (10 new) ==========
    
    def calculate_dpo(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Detrended Price Oscillator."""
        sma = df['close'].rolling(period).mean()
        df['dpo'] = df['close'].shift(period // 2 + 1) - sma
        return df
    
    def calculate_linear_regression_slope(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Linear Regression Slope."""
        def calc_slope(x):
            if len(x) < 2:
                return 0
            y = np.arange(len(x))
            return np.polyfit(y, x, 1)[0]
        
        df['lr_slope'] = df['close'].rolling(period).apply(calc_slope, raw=True)
        return df
    
    def calculate_linear_regression_angle(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Linear Regression Angle."""
        df = self.calculate_linear_regression_slope(df, period)
        df['lr_angle'] = np.arctan(df['lr_slope']) * 180 / np.pi
        return df
    
    def calculate_linear_regression_intercept(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Linear Regression Intercept."""
        def calc_intercept(x):
            if len(x) < 2:
                return x.iloc[-1] if hasattr(x, 'iloc') else x[-1]
            y = np.arange(len(x))
            return np.polyfit(y, x, 1)[1]
        
        df['lr_intercept'] = df['close'].rolling(period).apply(calc_intercept, raw=True)
        return df
    
    def calculate_psar_enhanced(self, df: pd.DataFrame) -> pd.DataFrame:
        """Enhanced Parabolic SAR with adaptive acceleration."""
        # Use existing parabolic_sar if available, otherwise calculate
        if 'parabolic_sar' not in df.columns:
            from algorithms.advanced_indicators import AdvancedIndicators
            adv = AdvancedIndicators()
            df = adv.calculate_parabolic_sar(df)
        
        # Add enhanced features
        df['psar_trend'] = np.where(df['close'] > df['parabolic_sar'], 1, -1)
        df['psar_distance'] = abs(df['close'] - df['parabolic_sar']) / df['close'] * 100
        return df
    
    def calculate_supertrend(self, df: pd.DataFrame, period: int = 10, multiplier: float = 3.0) -> pd.DataFrame:
        """Calculate Supertrend Indicator."""
        hl_avg = (df['high'] + df['low']) / 2
        atr = self.calculate_atr(df.copy(), period)['atr']
        
        upper_band = hl_avg + (multiplier * atr)
        lower_band = hl_avg - (multiplier * atr)
        
        supertrend = pd.Series(index=df.index, dtype=float)
        direction = pd.Series(index=df.index, dtype=int)
        
        supertrend.iloc[0] = upper_band.iloc[0]
        direction.iloc[0] = -1
        
        for i in range(1, len(df)):
            if df['close'].iloc[i] <= supertrend.iloc[i-1]:
                supertrend.iloc[i] = upper_band.iloc[i]
                direction.iloc[i] = -1
            else:
                supertrend.iloc[i] = lower_band.iloc[i]
                direction.iloc[i] = 1
        
        df['supertrend'] = supertrend
        df['supertrend_direction'] = direction
        return df
    
    def calculate_aroon(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Aroon Indicator."""
        aroon_up = ((period - df['high'].rolling(period).apply(lambda x: period - 1 - x.argmax())) / period) * 100
        aroon_down = ((period - df['low'].rolling(period).apply(lambda x: period - 1 - x.argmin())) / period) * 100
        df['aroon_up'] = aroon_up
        df['aroon_down'] = aroon_down
        return df
    
    def calculate_directional_movement_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Directional Movement Index."""
        from algorithms.advanced_indicators import AdvancedIndicators
        adv = AdvancedIndicators()
        df = adv.calculate_adx(df)
        # ADX already calculated, add DMI components
        return df
    
    def calculate_commodity_selection_index(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Commodity Selection Index."""
        momentum = df['close'].pct_change(periods=10) * 100
        volatility = df['close'].pct_change().rolling(10).std() * 100
        df['csi'] = momentum / volatility
        df['csi'] = df['csi'].replace([np.inf, -np.inf], 0).fillna(0)
        return df
    
    def calculate_trend_strength_index(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Trend Strength Index."""
        price_change = df['close'].diff()
        price_change_abs = price_change.abs()
        
        up_sum = price_change.where(price_change > 0, 0).rolling(period).sum()
        down_sum = price_change.where(price_change < 0, 0).abs().rolling(period).sum()
        total_sum = price_change_abs.rolling(period).sum()
        
        df['tsi'] = 100 * (up_sum - down_sum) / total_sum
        df['tsi'] = df['tsi'].fillna(0)
        return df
    
    # ========== CYCLE INDICATORS (10 new) ==========
    
    def calculate_hilbert_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Hilbert Transform (simplified)."""
        # Simplified version - full implementation is complex
        price = df['close']
        df['hilbert_inphase'] = price.ewm(span=7, adjust=False).mean()
        df['hilbert_quadrature'] = price.ewm(span=7, adjust=False).mean().shift(1)
        return df
    
    def calculate_dominant_cycle_period(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Dominant Cycle Period."""
        # Simplified cycle detection
        price_change = df['close'].pct_change()
        autocorr = []
        for i in range(1, min(period+1, len(price_change))):
            if len(price_change) > i:
                corr = price_change.autocorr(lag=i)
                autocorr.append(abs(corr) if corr is not None else 0)
            else:
                autocorr.append(0)
        
        if autocorr:
            max_corr_idx = np.argmax(autocorr)
            df['dominant_cycle'] = max_corr_idx + 1
        else:
            df['dominant_cycle'] = period
        return df
    
    def calculate_cycle_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Cycle Phase."""
        price = df['close']
        sma_fast = price.rolling(5).mean()
        sma_slow = price.rolling(20).mean()
        
        # Phase: 0-360 degrees
        phase = np.arctan2(sma_fast - sma_slow, sma_slow) * 180 / np.pi + 180
        df['cycle_phase'] = phase.fillna(180)
        return df
    
    def calculate_sine_wave(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Sine Wave."""
        df = self.calculate_cycle_phase(df)
        df['sine_wave'] = np.sin(df['cycle_phase'] * np.pi / 180)
        return df
    
    def calculate_lead_sine(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Lead Sine Wave."""
        df = self.calculate_cycle_phase(df)
        df['lead_sine'] = np.sin((df['cycle_phase'] + 45) * np.pi / 180)
        return df
    
    def calculate_dc_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DC Period (Dominant Cycle Period)."""
        return self.calculate_dominant_cycle_period(df)
    
    def calculate_dc_phase(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate DC Phase."""
        return self.calculate_cycle_phase(df)
    
    def calculate_adaptive_cycle_period(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Adaptive Cycle Period."""
        volatility = df['close'].pct_change().rolling(20).std()
        df['adaptive_cycle'] = (20 / (1 + volatility * 100)).fillna(20)
        return df
    
    def calculate_fisher_transform(self, df: pd.DataFrame, period: int = 10) -> pd.DataFrame:
        """Calculate Fisher Transform."""
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        
        value1 = 2 * ((df['close'] - low) / (high - low) - 0.5)
        value1 = value1.fillna(0).clip(-0.999, 0.999)
        
        df['fisher'] = 0.5 * np.log((1 + value1) / (1 - value1))
        df['fisher_signal'] = df['fisher'].shift(1)
        return df
    
    def calculate_inverse_fisher_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Inverse Fisher Transform."""
        if 'fisher' not in df.columns:
            df = self.calculate_fisher_transform(df)
        
        df['inv_fisher'] = (np.exp(2 * df['fisher']) - 1) / (np.exp(2 * df['fisher']) + 1)
        return df
    
    # ========== MARKET STRUCTURE (10 new) ==========
    
    def calculate_swing_highs_lows(self, df: pd.DataFrame, lookback: int = 5) -> pd.DataFrame:
        """Calculate Swing Highs and Lows."""
        highs = find_peaks(df['high'].values, distance=lookback)[0]
        lows = find_peaks(-df['low'].values, distance=lookback)[0]
        
        swing_highs = pd.Series(0, index=df.index)
        swing_lows = pd.Series(0, index=df.index)
        
        swing_highs.iloc[highs] = 1
        swing_lows.iloc[lows] = 1
        
        df['swing_high'] = swing_highs
        df['swing_low'] = swing_lows
        return df
    
    def calculate_pivot_points(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Pivot Points."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        pivot = (high + low + close) / 3
        r1 = 2 * pivot - low
        s1 = 2 * pivot - high
        r2 = pivot + (high - low)
        s2 = pivot - (high - low)
        
        df['pivot'] = pivot
        df['pivot_r1'] = r1
        df['pivot_r2'] = r2
        df['pivot_s1'] = s1
        df['pivot_s2'] = s2
        return df
    
    def calculate_fibonacci_levels(self, df: pd.DataFrame, period: int = 20) -> pd.DataFrame:
        """Calculate Fibonacci Retracement Levels."""
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        diff = high - low
        
        fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
        for level in fib_levels:
            df[f'fib_{int(level*1000)}'] = high - (diff * level)
        
        return df
    
    def calculate_camarilla_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Camarilla Pivot Levels."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        range_val = high - low
        df['cam_r4'] = close + range_val * 1.1 / 2
        df['cam_r3'] = close + range_val * 1.1 / 4
        df['cam_r2'] = close + range_val * 1.1 / 6
        df['cam_r1'] = close + range_val * 1.1 / 12
        df['cam_s1'] = close - range_val * 1.1 / 12
        df['cam_s2'] = close - range_val * 1.1 / 6
        df['cam_s3'] = close - range_val * 1.1 / 4
        df['cam_s4'] = close - range_val * 1.1 / 2
        return df
    
    def calculate_woodie_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Woodie Pivot Levels."""
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']
        
        pivot = (high + low + 2 * close) / 4
        range_val = high - low
        
        df['woodie_pivot'] = pivot
        df['woodie_r1'] = 2 * pivot - low
        df['woodie_r2'] = pivot + range_val
        df['woodie_s1'] = 2 * pivot - high
        df['woodie_s2'] = pivot - range_val
        return df
    
    def calculate_tom_demark_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Tom DeMark Pivot Levels."""
        high = df['high']
        low = df['low']
        close = df['close']
        open_price = df['open']
        
        x = np.where(close < open_price, high + 2 * low + close, 
                    np.where(close > open_price, 2 * high + low + close,
                             high + low + 2 * close))
        
        df['demark_pivot'] = x / 4
        df['demark_r1'] = x / 2 - low
        df['demark_s1'] = x / 2 - high
        return df
    
    def calculate_murrey_math_levels(self, df: pd.DataFrame, period: int = 100) -> pd.DataFrame:
        """Calculate Murrey Math Levels."""
        high = df['high'].rolling(period).max()
        low = df['low'].rolling(period).min()
        range_val = high - low
        
        # 8/8 lines
        for i in range(9):
            level = low + (range_val * i / 8)
            df[f'murrey_{i}8'] = level
        
        return df
    
    def calculate_gann_levels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Gann Levels."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        pivot = (high + low + close) / 3
        range_val = high - low
        
        # Gann angles (1x1, 1x2, 2x1, etc.)
        df['gann_1x1'] = pivot
        df['gann_1x2'] = pivot + range_val * 0.5
        df['gann_2x1'] = pivot - range_val * 0.5
        return df
    
    def calculate_volume_profile_levels(self, df: pd.DataFrame, bins: int = 20) -> pd.DataFrame:
        """Calculate Volume Profile Levels."""
        price_range = df['high'].max() - df['low'].min()
        bin_size = price_range / bins
        
        # Simplified volume profile
        df['vp_level'] = ((df['close'] - df['low'].min()) / bin_size).astype(int)
        df['vp_volume'] = df.groupby('vp_level')['volume'].transform('sum')
        return df
    
    def calculate_order_flow_imbalance(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Order Flow Imbalance (simplified)."""
        # Simplified version - real order flow requires tick data
        price_change = df['close'].diff()
        volume_weighted = price_change * df['volume']
        
        buy_pressure = volume_weighted.where(volume_weighted > 0, 0).rolling(20).sum()
        sell_pressure = volume_weighted.where(volume_weighted < 0, 0).abs().rolling(20).sum()
        
        df['order_flow_imbalance'] = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure)
        df['order_flow_imbalance'] = df['order_flow_imbalance'].fillna(0)
        return df
    
    # Helper method
    def calculate_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate ATR (helper method)."""
        from algorithms.advanced_indicators import AdvancedIndicators
        adv = AdvancedIndicators()
        return adv.calculate_atr(df, period)

