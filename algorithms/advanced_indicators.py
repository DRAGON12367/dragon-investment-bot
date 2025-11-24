"""
Advanced technical indicators for sophisticated trading analysis.
"""
import pandas as pd
import numpy as np
from typing import Dict


class AdvancedIndicators:
    """Calculate advanced technical indicators."""
    
    def calculate_all(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate all advanced indicators."""
        if df.empty:
            return df
        
        df = df.copy()
        
        # Advanced momentum indicators
        df = self.calculate_adx(df)
        df = self.calculate_stochastic(df)
        df = self.calculate_williams_r(df)
        df = self.calculate_cci(df)
        df = self.calculate_atr(df)
        
        # Volume indicators
        df = self.calculate_obv(df)
        df = self.calculate_cmf(df)
        df = self.calculate_vwap(df)
        
        # Trend indicators
        df = self.calculate_ichimoku(df)
        df = self.calculate_parabolic_sar(df)
        
        # Volatility indicators
        df = self.calculate_keltner_channels(df)
        df = self.calculate_donchian_channels(df)
        
        return df
    
    def calculate_adx(
        self, 
        df: pd.DataFrame, 
        period: int = 14
    ) -> pd.DataFrame:
        """Calculate Average Directional Index (ADX)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # Calculate Directional Movement
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        # Smooth the values
        atr = tr.rolling(window=period).mean()
        plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr)
        
        # Calculate ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()
        
        df['adx'] = adx
        df['plus_di'] = plus_di
        df['minus_di'] = minus_di
        
        return df
    
    def calculate_stochastic(
        self, 
        df: pd.DataFrame, 
        k_period: int = 14, 
        d_period: int = 3
    ) -> pd.DataFrame:
        """Calculate Stochastic Oscillator."""
        low_min = df['low'].rolling(window=k_period).min()
        high_max = df['high'].rolling(window=k_period).max()
        
        k_percent = 100 * ((df['close'] - low_min) / (high_max - low_min))
        d_percent = k_percent.rolling(window=d_period).mean()
        
        df['stoch_k'] = k_percent
        df['stoch_d'] = d_percent
        
        return df
    
    def calculate_williams_r(
        self, 
        df: pd.DataFrame, 
        period: int = 14
    ) -> pd.DataFrame:
        """Calculate Williams %R."""
        high_max = df['high'].rolling(window=period).max()
        low_min = df['low'].rolling(window=period).min()
        
        williams_r = -100 * ((high_max - df['close']) / (high_max - low_min))
        df['williams_r'] = williams_r
        
        return df
    
    def calculate_cci(
        self, 
        df: pd.DataFrame, 
        period: int = 20
    ) -> pd.DataFrame:
        """Calculate Commodity Channel Index (CCI)."""
        tp = (df['high'] + df['low'] + df['close']) / 3
        sma_tp = tp.rolling(window=period).mean()
        mad = tp.rolling(window=period).apply(lambda x: np.abs(x - x.mean()).mean())
        
        cci = (tp - sma_tp) / (0.015 * mad)
        df['cci'] = cci
        
        return df
    
    def calculate_atr(
        self, 
        df: pd.DataFrame, 
        period: int = 14
    ) -> pd.DataFrame:
        """Calculate Average True Range (ATR)."""
        high = df['high']
        low = df['low']
        close = df['close']
        
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        atr = tr.rolling(window=period).mean()
        df['atr'] = atr
        
        return df
    
    def calculate_obv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate On-Balance Volume (OBV)."""
        obv = (np.sign(df['close'].diff()) * df['volume']).fillna(0).cumsum()
        df['obv'] = obv
        return df
    
    def calculate_cmf(
        self, 
        df: pd.DataFrame, 
        period: int = 20
    ) -> pd.DataFrame:
        """Calculate Chaikin Money Flow (CMF)."""
        mfv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
        mfv = mfv * df['volume']
        cmf = mfv.rolling(window=period).sum() / df['volume'].rolling(window=period).sum()
        df['cmf'] = cmf
        return df
    
    def calculate_vwap(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Volume Weighted Average Price (VWAP)."""
        typical_price = (df['high'] + df['low'] + df['close']) / 3
        vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
        df['vwap'] = vwap
        return df
    
    def calculate_ichimoku(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate Ichimoku Cloud components."""
        high_9 = df['high'].rolling(window=9).max()
        low_9 = df['low'].rolling(window=9).min()
        tenkan_sen = (high_9 + low_9) / 2
        
        high_26 = df['high'].rolling(window=26).max()
        low_26 = df['low'].rolling(window=26).min()
        kijun_sen = (high_26 + low_26) / 2
        
        senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
        high_52 = df['high'].rolling(window=52).max()
        low_52 = df['low'].rolling(window=52).min()
        senkou_span_b = ((high_52 + low_52) / 2).shift(26)
        
        chikou_span = df['close'].shift(-26)
        
        df['ichimoku_tenkan'] = tenkan_sen
        df['ichimoku_kijun'] = kijun_sen
        df['ichimoku_senkou_a'] = senkou_span_a
        df['ichimoku_senkou_b'] = senkou_span_b
        df['ichimoku_chikou'] = chikou_span
        
        return df
    
    def calculate_parabolic_sar(
        self, 
        df: pd.DataFrame,
        af_start: float = 0.02,
        af_increment: float = 0.02,
        af_max: float = 0.2
    ) -> pd.DataFrame:
        """Calculate Parabolic SAR."""
        # Simplified Parabolic SAR calculation
        high = df['high']
        low = df['low']
        close = df['close']
        
        sar = pd.Series(index=df.index, dtype=float)
        trend = pd.Series(index=df.index, dtype=int)
        ep = pd.Series(index=df.index, dtype=float)
        af = pd.Series(index=df.index, dtype=float)
        
        sar.iloc[0] = low.iloc[0]
        trend.iloc[0] = 1
        ep.iloc[0] = high.iloc[0]
        af.iloc[0] = af_start
        
        for i in range(1, len(df)):
            prev_sar = sar.iloc[i-1]
            prev_trend = trend.iloc[i-1]
            prev_ep = ep.iloc[i-1]
            prev_af = af.iloc[i-1]
            
            if prev_trend == 1:  # Uptrend
                sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                if sar.iloc[i] > low.iloc[i]:
                    trend.iloc[i] = -1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = low.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = 1
                    if high.iloc[i] > prev_ep:
                        ep.iloc[i] = high.iloc[i]
                        af.iloc[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
            else:  # Downtrend
                sar.iloc[i] = prev_sar + prev_af * (prev_ep - prev_sar)
                if sar.iloc[i] < high.iloc[i]:
                    trend.iloc[i] = 1
                    sar.iloc[i] = prev_ep
                    ep.iloc[i] = high.iloc[i]
                    af.iloc[i] = af_start
                else:
                    trend.iloc[i] = -1
                    if low.iloc[i] < prev_ep:
                        ep.iloc[i] = low.iloc[i]
                        af.iloc[i] = min(prev_af + af_increment, af_max)
                    else:
                        ep.iloc[i] = prev_ep
                        af.iloc[i] = prev_af
        
        df['parabolic_sar'] = sar
        return df
    
    def calculate_keltner_channels(
        self, 
        df: pd.DataFrame,
        period: int = 20,
        multiplier: float = 2.0
    ) -> pd.DataFrame:
        """Calculate Keltner Channels."""
        ema = df['close'].ewm(span=period).mean()
        atr = self.calculate_atr(df.copy(), period=period)['atr']
        
        df['keltner_upper'] = ema + (multiplier * atr)
        df['keltner_middle'] = ema
        df['keltner_lower'] = ema - (multiplier * atr)
        
        return df
    
    def calculate_donchian_channels(
        self, 
        df: pd.DataFrame,
        period: int = 20
    ) -> pd.DataFrame:
        """Calculate Donchian Channels."""
        df['donchian_upper'] = df['high'].rolling(window=period).max()
        df['donchian_lower'] = df['low'].rolling(window=period).min()
        df['donchian_middle'] = (df['donchian_upper'] + df['donchian_lower']) / 2
        
        return df

