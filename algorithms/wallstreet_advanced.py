"""
Wall Street Advanced Features - Institutional-grade prediction algorithms.
Implements techniques used by hedge funds, prop trading firms, and market makers.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict
import warnings

try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

from algorithms.technical_indicators import TechnicalIndicators
from algorithms.advanced_indicators import AdvancedIndicators
from algorithms.professional_analysis import ProfessionalAnalysis
from algorithms.sentiment_analysis import SentimentAnalyzer
from algorithms.institutional_footprint import InstitutionalFootprint
from algorithms.market_maker_levels import MarketMakerLevels
from algorithms.options_flow import OptionsFlowAnalyzer
from algorithms.advanced_risk_metrics import AdvancedRiskMetrics
from utils.config import Config


class WallStreetAdvanced:
    """
    Advanced Wall Street techniques for institutional-grade predictions.
    
    Features:
    1. Order Flow Analysis - Simulated Level 2 data analysis
    2. Smart Money Concepts - Institutional trading patterns
    3. Market Regime Detection - Bull/Bear/Sideways identification
    4. Liquidity Analysis - Market depth and execution quality
    5. Multi-Timeframe Confluence - Signal confirmation across timeframes
    6. Volume Profile Analysis - Price-by-volume institutional levels
    7. Volatility Forecasting - GARCH and advanced volatility models
    8. Correlation Regime Analysis - Dynamic correlation detection
    9. Market Microstructure - Bid-ask spread and order book analysis
    10. Sentiment Momentum - Price action sentiment analysis
    """
    
    def __init__(self, config: Config):
        """Initialize Wall Street advanced features."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.wallstreet")
        self.technical_indicators = TechnicalIndicators()
        self.advanced_indicators = AdvancedIndicators()
        self.professional_analysis = ProfessionalAnalysis()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.institutional_footprint = InstitutionalFootprint()
        self.market_maker_levels = MarketMakerLevels()
        self.options_flow = OptionsFlowAnalyzer()
        self.risk_metrics = AdvancedRiskMetrics()
        
        # Market regime tracking
        self.market_regimes: Dict[str, str] = {}  # symbol -> regime
        self.regime_history: Dict[str, List[Tuple[datetime, str]]] = defaultdict(list)
        
        # Hyper ML Models for enhanced predictions
        try:
            from algorithms.hyper_ml_models import HyperMLModels
            self.hyper_ml = HyperMLModels(config)
        except ImportError:
            self.hyper_ml = None
        
    def analyze_order_flow(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze order flow patterns (simulated Level 2 analysis).
        
        Wall Street Technique: Institutional traders analyze order flow to predict
        price movements before they happen.
        
        Returns:
            Order flow metrics and signals
        """
        if df.empty or len(df) < 20:
            return {}
        
        df = df.copy()
        
        # Calculate order flow imbalance
        # Positive flow = more buying pressure, Negative = selling pressure
        price_change = df['close'].diff()
        volume_weighted = (price_change * volume).fillna(0)
        
        # Cumulative order flow
        cumulative_flow = volume_weighted.cumsum()
        
        # Order flow momentum (rate of change)
        flow_momentum = cumulative_flow.diff(5).fillna(0)
        
        # Large order detection (institutional activity)
        avg_volume = volume.rolling(20).mean()
        large_orders = volume > (avg_volume * 2)  # 2x average = large order
        
        # Order flow divergence (price vs flow)
        price_trend = df['close'].rolling(10).apply(lambda x: 1 if x.iloc[-1] > x.iloc[0] else -1)
        flow_trend = np.sign(cumulative_flow.diff(10))
        divergence = price_trend != flow_trend
        
        # Smart money accumulation/distribution
        smart_money_flow = volume_weighted[large_orders].sum()
        
        return {
            'cumulative_flow': float(cumulative_flow.iloc[-1]),
            'flow_momentum': float(flow_momentum.iloc[-1]),
            'large_orders_count': int(large_orders.sum()),
            'smart_money_flow': float(smart_money_flow),
            'flow_divergence': bool(divergence.iloc[-1]) if len(divergence) > 0 else False,
            'order_flow_signal': 'BUY' if cumulative_flow.iloc[-1] > 0 and flow_momentum.iloc[-1] > 0 else 'SELL' if cumulative_flow.iloc[-1] < 0 else 'NEUTRAL',
            'institutional_activity': 'HIGH' if large_orders.sum() > 5 else 'MEDIUM' if large_orders.sum() > 2 else 'LOW'
        }
    
    def detect_smart_money_concepts(
        self,
        df: pd.DataFrame
    ) -> Dict[str, Any]:
        """
        Detect Smart Money Concepts (SMC) - Institutional trading patterns.
        
        Wall Street Technique: Identifies how institutions trade:
        - Accumulation zones (where they buy)
        - Distribution zones (where they sell)
        - Break of structure (BOS)
        - Change of character (CHoCH)
        """
        if df.empty or len(df) < 50:
            return {}
        
        df = df.copy()
        
        # Identify swing highs and lows (structure points)
        window = 20
        swing_highs = []
        swing_lows = []
        
        for i in range(window, len(df) - window):
            # Swing high
            if df['high'].iloc[i] == df['high'].iloc[i-window:i+window+1].max():
                swing_highs.append((i, df['high'].iloc[i]))
            # Swing low
            if df['low'].iloc[i] == df['low'].iloc[i-window:i+window+1].min():
                swing_lows.append((i, df['low'].iloc[i]))
        
        # Detect Break of Structure (BOS)
        current_price = df['close'].iloc[-1]
        recent_high = df['high'].tail(20).max()
        recent_low = df['low'].tail(20).min()
        
        # Bullish BOS: Price breaks above recent swing high
        bullish_bos = current_price > recent_high * 0.98
        
        # Bearish BOS: Price breaks below recent swing low
        bearish_bos = current_price < recent_low * 1.02
        
        # Change of Character (CHoCH) - Trend reversal signal
        price_momentum = df['close'].pct_change(10).iloc[-1]
        prev_momentum = df['close'].pct_change(10).iloc[-11] if len(df) > 11 else 0
        
        choch = (price_momentum > 0 and prev_momentum < 0) or (price_momentum < 0 and prev_momentum > 0)
        
        # Accumulation/Distribution zones
        # Accumulation: Price consolidates near lows with increasing volume
        recent_low_price = df['low'].tail(20).min()
        price_near_low = abs(current_price - recent_low_price) / recent_low_price < 0.05
        volume_increasing = df['volume'].tail(5).mean() > df['volume'].tail(20).head(5).mean()
        accumulation = price_near_low and volume_increasing
        
        # Distribution: Price consolidates near highs with decreasing volume
        price_near_high = abs(current_price - recent_high) / recent_high < 0.05
        volume_decreasing = df['volume'].tail(5).mean() < df['volume'].tail(20).head(5).mean()
        distribution = price_near_high and volume_decreasing
        
        # Fair Value Gap (FVG) - Institutional inefficiency
        # Gap between candles that institutions often fill
        fvg_detected = False
        if len(df) >= 3:
            prev_high = df['high'].iloc[-3]
            prev_low = df['low'].iloc[-3]
            current_low = df['low'].iloc[-1]
            current_high = df['high'].iloc[-1]
            
            # Bullish FVG: Gap between previous high and current low
            if current_low > prev_high:
                fvg_detected = True
                fvg_type = 'BULLISH'
            # Bearish FVG: Gap between previous low and current high
            elif current_high < prev_low:
                fvg_detected = True
                fvg_type = 'BEARISH'
            else:
                fvg_type = 'NONE'
        else:
            fvg_type = 'NONE'
        
        return {
            'swing_highs_count': len(swing_highs),
            'swing_lows_count': len(swing_lows),
            'bullish_bos': bullish_bos,
            'bearish_bos': bearish_bos,
            'change_of_character': choch,
            'accumulation_zone': accumulation,
            'distribution_zone': distribution,
            'fair_value_gap': fvg_detected,
            'fvg_type': fvg_type,
            'smart_money_signal': 'BUY' if (bullish_bos or accumulation) and not bearish_bos else 'SELL' if (bearish_bos or distribution) and not bullish_bos else 'NEUTRAL'
        }
    
    def detect_market_regime(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> Dict[str, Any]:
        """
        Detect market regime: Bull, Bear, or Sideways.
        
        Wall Street Technique: Different strategies work in different regimes.
        Institutions adapt their strategies based on market regime.
        """
        if df.empty or len(df) < 50:
            return {'regime': 'UNKNOWN', 'confidence': 0.0}
        
        df = df.copy()
        
        # Calculate trend strength using multiple methods
        returns = df['close'].pct_change().dropna()
        
        # 1. Price trend (50-day vs 200-day)
        sma_50 = df['close'].rolling(50).mean()
        sma_200 = df['close'].rolling(200).mean() if len(df) >= 200 else sma_50
        
        price_above_sma50 = df['close'].iloc[-1] > sma_50.iloc[-1]
        price_above_sma200 = df['close'].iloc[-1] > sma_200.iloc[-1]
        sma50_above_sma200 = sma_50.iloc[-1] > sma_200.iloc[-1]
        
        # 2. Momentum
        momentum_20 = returns.tail(20).mean() * 252  # Annualized
        momentum_50 = returns.tail(50).mean() * 252 if len(returns) >= 50 else momentum_20
        
        # 3. Volatility regime
        volatility = returns.std() * np.sqrt(252)  # Annualized
        high_volatility = volatility > 0.30  # 30% annual volatility threshold
        
        # 4. ADX for trend strength
        if 'adx' in df.columns:
            adx = df['adx'].iloc[-1]
            strong_trend = adx > 25
        else:
            adx = 0
            strong_trend = False
        
        # Regime classification
        bullish_signals = 0
        bearish_signals = 0
        sideways_signals = 0
        
        # Bull market signals
        if price_above_sma50 and price_above_sma200 and sma50_above_sma200:
            bullish_signals += 3
        elif price_above_sma50:
            bullish_signals += 1
        
        if momentum_20 > 0.10:  # 10%+ annual return
            bullish_signals += 2
        elif momentum_20 > 0:
            bullish_signals += 1
        
        if strong_trend and df['close'].iloc[-1] > df['close'].iloc[-20]:
            bullish_signals += 1
        
        # Bear market signals
        if not price_above_sma50 and not price_above_sma200:
            bearish_signals += 3
        elif not price_above_sma50:
            bearish_signals += 1
        
        if momentum_20 < -0.10:  # -10% annual return
            bearish_signals += 2
        elif momentum_20 < 0:
            bearish_signals += 1
        
        if strong_trend and df['close'].iloc[-1] < df['close'].iloc[-20]:
            bearish_signals += 1
        
        # Sideways signals
        if abs(momentum_20) < 0.05 and not strong_trend:  # Low momentum, weak trend
            sideways_signals += 2
        
        if abs(df['close'].iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1] < 0.05:  # Close to MA
            sideways_signals += 1
        
        # Determine regime
        max_signals = max(bullish_signals, bearish_signals, sideways_signals)
        
        if max_signals == bullish_signals and bullish_signals >= 3:
            regime = 'BULL'
            confidence = min(bullish_signals / 6, 1.0)
        elif max_signals == bearish_signals and bearish_signals >= 3:
            regime = 'BEAR'
            confidence = min(bearish_signals / 6, 1.0)
        else:
            regime = 'SIDEWAYS'
            confidence = min(max_signals / 3, 1.0)
        
        # Update regime history
        self.market_regimes[symbol] = regime
        self.regime_history[symbol].append((datetime.now(), regime))
        if len(self.regime_history[symbol]) > 100:
            self.regime_history[symbol] = self.regime_history[symbol][-100:]
        
        return {
            'regime': regime,
            'confidence': confidence,
            'momentum_20d': momentum_20,
            'volatility': volatility,
            'high_volatility': high_volatility,
            'trend_strength': adx,
            'price_vs_sma50': 'ABOVE' if price_above_sma50 else 'BELOW',
            'price_vs_sma200': 'ABOVE' if price_above_sma200 else 'BELOW'
        }
    
    def analyze_liquidity(
        self,
        df: pd.DataFrame,
        volume: pd.Series
    ) -> Dict[str, Any]:
        """
        Analyze market liquidity and execution quality.
        
        Wall Street Technique: Institutions need to know liquidity before
        entering large positions to avoid slippage.
        """
        if df.empty:
            return {}
        
        df = df.copy()
        
        # Average daily volume
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        
        # Volume liquidity score
        volume_ratio = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        
        # Price impact (how much price moves per unit volume)
        price_change = abs(df['close'].pct_change().iloc[-1])
        volume_normalized = current_volume / avg_volume.iloc[-1] if avg_volume.iloc[-1] > 0 else 1.0
        price_impact = price_change / volume_normalized if volume_normalized > 0 else 0
        
        # Bid-ask spread estimation (using high-low range as proxy)
        daily_range = (df['high'] - df['low']) / df['close']
        avg_spread = daily_range.rolling(20).mean().iloc[-1]
        
        # Liquidity zones (where most trading happens)
        # High volume = high liquidity
        high_volume_zones = volume > (avg_volume * 1.5)
        liquidity_zones = df.loc[high_volume_zones, 'close'].tolist()
        
        # Execution quality score
        # Lower spread + higher volume = better execution
        execution_score = (1 / (avg_spread + 0.001)) * volume_ratio
        execution_score = min(execution_score / 10, 1.0)  # Normalize to 0-1
        
        return {
            'volume_ratio': float(volume_ratio),
            'liquidity_score': 'HIGH' if volume_ratio > 1.5 else 'MEDIUM' if volume_ratio > 0.8 else 'LOW',
            'estimated_spread': float(avg_spread * 100),  # As percentage
            'price_impact': float(price_impact),
            'execution_quality': execution_score,
            'execution_grade': 'EXCELLENT' if execution_score > 0.7 else 'GOOD' if execution_score > 0.4 else 'POOR',
            'liquidity_zones': liquidity_zones[-5:] if liquidity_zones else []  # Last 5 zones
        }
    
    def multi_timeframe_confluence(
        self,
        market_data: Dict[str, pd.DataFrame],
        symbol: str
    ) -> Dict[str, Any]:
        """
        Multi-timeframe analysis for signal confirmation.
        
        Wall Street Technique: Institutions confirm signals across multiple
        timeframes before entering positions.
        """
        if symbol not in market_data:
            return {}
        
        df = market_data[symbol]
        if df.empty or len(df) < 50:
            return {}
        
        # Simulate different timeframes by resampling
        # Daily (1D), 4-hour (4H), 1-hour (1H) equivalent
        signals = {
            'daily': None,
            'intraday': None,
            'short_term': None
        }
        
        # Daily timeframe (use full data)
        daily_trend = 'BULLISH' if df['close'].iloc[-1] > df['close'].iloc[-20] else 'BEARISH'
        daily_momentum = df['close'].pct_change(20).iloc[-1]
        
        # Intraday (last 20 periods)
        if len(df) >= 20:
            intraday_trend = 'BULLISH' if df['close'].iloc[-1] > df['close'].iloc[-10] else 'BEARISH'
            intraday_momentum = df['close'].pct_change(10).iloc[-1]
        else:
            intraday_trend = daily_trend
            intraday_momentum = daily_momentum
        
        # Short-term (last 5 periods)
        if len(df) >= 5:
            short_term_trend = 'BULLISH' if df['close'].iloc[-1] > df['close'].iloc[-3] else 'BEARISH'
            short_term_momentum = df['close'].pct_change(3).iloc[-1]
        else:
            short_term_trend = intraday_trend
            short_term_momentum = intraday_momentum
        
        # Count confluence
        bullish_count = sum([
            daily_trend == 'BULLISH',
            intraday_trend == 'BULLISH',
            short_term_trend == 'BULLISH'
        ])
        
        bearish_count = sum([
            daily_trend == 'BEARISH',
            intraday_trend == 'BEARISH',
            short_term_trend == 'BEARISH'
        ])
        
        # Confluence score (0-1)
        confluence_score = max(bullish_count, bearish_count) / 3.0
        
        # Signal
        if bullish_count >= 2:
            signal = 'STRONG_BUY' if bullish_count == 3 else 'BUY'
        elif bearish_count >= 2:
            signal = 'STRONG_SELL' if bearish_count == 3 else 'SELL'
        else:
            signal = 'NEUTRAL'
        
        return {
            'daily_trend': daily_trend,
            'intraday_trend': intraday_trend,
            'short_term_trend': short_term_trend,
            'confluence_score': confluence_score,
            'bullish_timeframes': bullish_count,
            'bearish_timeframes': bearish_count,
            'multi_tf_signal': signal,
            'momentum_alignment': 'ALIGNED' if bullish_count >= 2 or bearish_count >= 2 else 'MIXED'
        }
    
    def forecast_volatility(
        self,
        df: pd.DataFrame,
        periods: int = 5
    ) -> Dict[str, Any]:
        """
        Forecast volatility using GARCH-like approach.
        
        Wall Street Technique: Volatility forecasting for risk management
        and position sizing.
        """
        if df.empty or len(df) < 30:
            return {}
        
        returns = df['close'].pct_change().dropna()
        
        # Historical volatility
        historical_vol = returns.std() * np.sqrt(252)  # Annualized
        
        # Volatility clustering (GARCH effect)
        # High volatility tends to be followed by high volatility
        recent_vol = returns.tail(10).std() * np.sqrt(252)
        long_term_vol = returns.std() * np.sqrt(252)
        
        # Volatility forecast (simplified GARCH)
        # Weight recent volatility more heavily
        forecast_vol = 0.7 * recent_vol + 0.3 * long_term_vol
        
        # Volatility regime
        vol_regime = 'HIGH' if forecast_vol > 0.30 else 'MEDIUM' if forecast_vol > 0.15 else 'LOW'
        
        # Expected price range (using forecasted volatility)
        current_price = df['close'].iloc[-1]
        expected_range = current_price * forecast_vol * np.sqrt(periods / 252)
        
        return {
            'historical_volatility': float(historical_vol),
            'forecasted_volatility': float(forecast_vol),
            'volatility_regime': vol_regime,
            'expected_price_range': float(expected_range),
            'upper_bound': float(current_price + expected_range),
            'lower_bound': float(current_price - expected_range),
            'volatility_trend': 'INCREASING' if recent_vol > long_term_vol else 'DECREASING'
        }
    
    def analyze_correlation_regime(
        self,
        price_data: Dict[str, pd.Series],
        lookback: int = 60
    ) -> Dict[str, Any]:
        """
        Analyze correlation regime between assets.
        
        Wall Street Technique: Correlations change in different market regimes.
        During crises, correlations increase (everything moves together).
        """
        if len(price_data) < 2:
            return {}
        
        # Calculate returns
        returns_df = pd.DataFrame({symbol: series.pct_change().dropna() 
                                  for symbol, series in price_data.items()})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < lookback:
            return {}
        
        # Recent correlation matrix
        recent_returns = returns_df.tail(lookback)
        correlation_matrix = recent_returns.corr()
        
        # Average correlation
        # Exclude diagonal (self-correlation = 1.0)
        mask = np.triu(np.ones_like(correlation_matrix.values), k=1).astype(bool)
        avg_correlation = correlation_matrix.values[mask].mean()
        
        # Correlation regime
        if avg_correlation > 0.7:
            regime = 'HIGH_CORRELATION'  # Crisis mode - everything moves together
        elif avg_correlation > 0.4:
            regime = 'MODERATE_CORRELATION'
        else:
            regime = 'LOW_CORRELATION'  # Normal diversification
        
        # Correlation trend
        if len(returns_df) >= lookback * 2:
            older_returns = returns_df.iloc[-lookback*2:-lookback]
            older_corr = older_returns.corr()
            older_avg = older_corr.values[mask].mean()
            correlation_trend = 'INCREASING' if avg_correlation > older_avg else 'DECREASING'
        else:
            correlation_trend = 'STABLE'
        
        return {
            'average_correlation': float(avg_correlation),
            'correlation_regime': regime,
            'correlation_trend': correlation_trend,
            'diversification_benefit': 'LOW' if avg_correlation > 0.7 else 'HIGH' if avg_correlation < 0.4 else 'MODERATE'
        }
    
    def comprehensive_analysis(
        self,
        df: pd.DataFrame,
        symbol: str,
        volume: Optional[pd.Series] = None,
        market_data: Optional[Dict[str, pd.DataFrame]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive Wall Street analysis combining all techniques.
        
        Returns:
            Complete analysis with all advanced features
        """
        if df.empty:
            return {}
        
        if volume is None:
            volume = df.get('volume', pd.Series([0] * len(df)))
        
        results = {
            'symbol': symbol,
            'timestamp': datetime.now().isoformat(),
        }
        
        # Order Flow Analysis
        try:
            order_flow = self.analyze_order_flow(df, volume)
            results['order_flow'] = order_flow
        except Exception as e:
            self.logger.debug(f"Error in order flow analysis: {e}")
        
        # Smart Money Concepts
        try:
            smc = self.detect_smart_money_concepts(df)
            results['smart_money'] = smc
        except Exception as e:
            self.logger.debug(f"Error in SMC analysis: {e}")
        
        # Market Regime
        try:
            regime = self.detect_market_regime(df, symbol)
            results['market_regime'] = regime
        except Exception as e:
            self.logger.debug(f"Error in regime detection: {e}")
        
        # Liquidity Analysis
        try:
            liquidity = self.analyze_liquidity(df, volume)
            results['liquidity'] = liquidity
        except Exception as e:
            self.logger.debug(f"Error in liquidity analysis: {e}")
        
        # Multi-Timeframe Confluence
        if market_data:
            try:
                mtf = self.multi_timeframe_confluence(market_data, symbol)
                results['multi_timeframe'] = mtf
            except Exception as e:
                self.logger.debug(f"Error in MTF analysis: {e}")
        
        # Volatility Forecast
        try:
            vol_forecast = self.forecast_volatility(df)
            results['volatility_forecast'] = vol_forecast
        except Exception as e:
            self.logger.debug(f"Error in volatility forecast: {e}")
        
        # Sentiment Analysis
        try:
            sentiment = self.sentiment_analyzer.comprehensive_sentiment(df, volume)
            results['sentiment'] = sentiment
        except Exception as e:
            self.logger.debug(f"Error in sentiment analysis: {e}")
        
        # Institutional Footprint
        try:
            footprint = self.institutional_footprint.comprehensive_footprint(df, volume)
            results['institutional_footprint'] = footprint
        except Exception as e:
            self.logger.debug(f"Error in institutional footprint: {e}")
        
        # Market Maker Levels
        try:
            mm_levels = self.market_maker_levels.comprehensive_levels(df, volume)
            results['market_maker_levels'] = mm_levels
        except Exception as e:
            self.logger.debug(f"Error in market maker levels: {e}")
        
        # Options Flow
        try:
            options = self.options_flow.comprehensive_options_flow(df)
            results['options_flow'] = options
        except Exception as e:
            self.logger.debug(f"Error in options flow: {e}")
        
        # Advanced Risk Metrics
        try:
            returns = df['close'].pct_change().dropna()
            if len(returns) >= 20:
                risk_analysis = self.risk_metrics.comprehensive_risk_analysis(
                    returns,
                    prices=df['close']
                )
                results['risk_analysis'] = risk_analysis
        except Exception as e:
            self.logger.debug(f"Error in risk analysis: {e}")
        
        # Generate overall signal from all sources
        signals = []
        
        # Order flow
        if 'order_flow' in results and results['order_flow'].get('order_flow_signal'):
            signals.append(results['order_flow']['order_flow_signal'])
        
        # Smart money
        if 'smart_money' in results and results['smart_money'].get('smart_money_signal'):
            signals.append(results['smart_money']['smart_money_signal'])
        
        # Multi-timeframe
        if 'multi_timeframe' in results and results['multi_timeframe'].get('multi_tf_signal'):
            signals.append(results['multi_timeframe']['multi_tf_signal'])
        
        # Sentiment (contrarian)
        if 'sentiment' in results:
            sentiment_signal = results['sentiment'].get('overall_sentiment_signal', 'NEUTRAL')
            if sentiment_signal != 'NEUTRAL':
                signals.append(sentiment_signal)
        
        # Institutional footprint
        if 'institutional_footprint' in results:
            inst_signal = results['institutional_footprint'].get('overall_institutional_signal', 'NEUTRAL')
            if inst_signal != 'NEUTRAL':
                signals.append(inst_signal)
        
        buy_count = signals.count('BUY') + signals.count('STRONG_BUY') + signals.count('WEAK_BUY')
        sell_count = signals.count('SELL') + signals.count('STRONG_SELL') + signals.count('WEAK_SELL')
        
        if buy_count > sell_count and buy_count >= 2:
            overall_signal = 'STRONG_BUY' if buy_count >= 4 else 'BUY'
        elif sell_count > buy_count and sell_count >= 2:
            overall_signal = 'STRONG_SELL' if sell_count >= 4 else 'SELL'
        else:
            overall_signal = 'NEUTRAL'
        
        results['overall_signal'] = overall_signal
        results['signal_confidence'] = max(buy_count, sell_count) / len(signals) if signals else 0.0
        results['signal_sources'] = len(signals)
        
        return results

