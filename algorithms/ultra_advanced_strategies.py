"""
Ultra Advanced Trading Strategies - 5x Upgrade
20+ new professional trading strategies used by hedge funds and prop firms.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from scipy import stats
from scipy.optimize import minimize


class UltraAdvancedStrategies:
    """Ultra Advanced Trading Strategies - 20+ new strategies."""
    
    def __init__(self):
        """Initialize ultra advanced strategies."""
        self.logger = logging.getLogger("ai_investment_bot.ultra_strategies")
    
    # ========== BREAKOUT STRATEGIES (5 new) ==========
    
    def detect_breakout_opportunities(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Breakout Trading Opportunities."""
        if df.empty or len(df) < 50:
            return {}
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Donchian Channels
        upper_band = high.rolling(20).max()
        lower_band = low.rolling(20).min()
        
        # Volume confirmation
        avg_volume = volume.rolling(20).mean()
        volume_spike = volume > avg_volume * 1.5
        
        # Breakout signals
        bullish_breakout = (close > upper_band.shift(1)) & volume_spike
        bearish_breakout = (close < lower_band.shift(1)) & volume_spike
        
        current_price = close.iloc[-1]
        upper_level = upper_band.iloc[-1]
        lower_level = lower_band.iloc[-1]
        
        if bullish_breakout.iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Breakout',
                'signal': 'STRONG_BUY',
                'confidence': 0.85,
                'current_price': float(current_price),
                'target_price': float(upper_level * 1.05),
                'stop_loss': float(lower_level),
                'reason': 'Bullish breakout above resistance with volume confirmation'
            }
        elif bearish_breakout.iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Breakout',
                'signal': 'STRONG_SELL',
                'confidence': 0.85,
                'current_price': float(current_price),
                'target_price': float(lower_level * 0.95),
                'stop_loss': float(upper_level),
                'reason': 'Bearish breakdown below support with volume confirmation'
            }
        
        return {}
    
    def detect_triangle_breakout(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Triangle Pattern Breakouts."""
        if df.empty or len(df) < 30:
            return {}
        
        high = df['high']
        low = df['low']
        close = df['close']
        
        # Identify triangle pattern (simplified)
        recent_highs = high.tail(20)
        recent_lows = low.tail(20)
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        # Ascending triangle: horizontal resistance, rising support
        # Descending triangle: falling resistance, horizontal support
        # Symmetrical triangle: both converging
        
        if abs(high_trend) < 0.01 and low_trend > 0.01:  # Ascending triangle
            resistance = recent_highs.max()
            if close.iloc[-1] > resistance * 0.99:
                return {
                    'symbol': symbol,
                    'strategy': 'Triangle Breakout',
                    'signal': 'STRONG_BUY',
                    'confidence': 0.80,
                    'current_price': float(close.iloc[-1]),
                    'target_price': float(resistance * 1.1),
                    'pattern': 'Ascending Triangle'
                }
        
        return {}
    
    def detect_cup_and_handle(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Cup and Handle Pattern."""
        if df.empty or len(df) < 60:
            return {}
        
        close = df['close']
        high = df['high']
        
        # Simplified cup and handle detection
        lookback = min(60, len(close))
        prices = close.tail(lookback).values
        
        # Find cup (U-shaped pattern)
        min_idx = np.argmin(prices)
        left_side = prices[:min_idx]
        right_side = prices[min_idx:]
        
        if len(left_side) > 10 and len(right_side) > 10:
            # Check if right side is rising (handle)
            if np.polyfit(range(len(right_side)), right_side, 1)[0] > 0:
                cup_depth = (prices.max() - prices.min()) / prices.max()
                if 0.1 < cup_depth < 0.4:  # Reasonable cup depth
                    return {
                        'symbol': symbol,
                        'strategy': 'Cup and Handle',
                        'signal': 'BUY',
                        'confidence': 0.75,
                        'current_price': float(close.iloc[-1]),
                        'target_price': float(high.tail(60).max()),
                        'pattern': 'Cup and Handle'
                    }
        
        return {}
    
    def detect_head_and_shoulders(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Head and Shoulders Pattern."""
        if df.empty or len(df) < 50:
            return {}
        
        high = df['high']
        close = df['close']
        
        # Find peaks
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(high.values, distance=10, prominence=high.std())
        
        if len(peaks) >= 3:
            peak_values = high.iloc[peaks[-3:]].values
            # Head and shoulders: left shoulder < head > right shoulder
            if peak_values[0] < peak_values[1] > peak_values[2] and peak_values[0] > peak_values[2]:
                neckline = (peak_values[0] + peak_values[2]) / 2
                if close.iloc[-1] < neckline:
                    return {
                        'symbol': symbol,
                        'strategy': 'Head and Shoulders',
                        'signal': 'STRONG_SELL',
                        'confidence': 0.85,
                        'current_price': float(close.iloc[-1]),
                        'target_price': float(neckline - (peak_values[1] - neckline)),
                        'pattern': 'Head and Shoulders'
                    }
        
        return {}
    
    def detect_pennant_pattern(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Pennant Pattern."""
        if df.empty or len(df) < 30:
            return {}
        
        high = df['high']
        low = df['low']
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Pennant: small symmetrical triangle after strong move
        recent_highs = high.tail(15)
        recent_lows = low.tail(15)
        
        high_trend = np.polyfit(range(len(recent_highs)), recent_highs, 1)[0]
        low_trend = np.polyfit(range(len(recent_lows)), recent_lows, 1)[0]
        
        # Converging lines (pennant)
        if abs(high_trend) < 0.02 and abs(low_trend) < 0.02:
            # Check for preceding strong move
            prior_move = abs(close.iloc[-15] - close.iloc[-30]) / close.iloc[-30] if len(close) >= 30 else 0
            if prior_move > 0.05:  # 5%+ move
                # Volume should decrease in pennant
                if volume.tail(15).mean() < volume.tail(30).head(15).mean() * 0.8:
                    direction = 'BUY' if close.iloc[-15] > close.iloc[-30] else 'SELL'
                    return {
                        'symbol': symbol,
                        'strategy': 'Pennant',
                        'signal': f'STRONG_{direction}',
                        'confidence': 0.80,
                        'current_price': float(close.iloc[-1]),
                        'target_price': float(close.iloc[-1] * (1 + prior_move * (1 if direction == 'BUY' else -1))),
                        'pattern': 'Pennant'
                    }
        
        return {}
    
    # ========== ARBITRAGE STRATEGIES (5 new) ==========
    
    def detect_pairs_trading(self, df1: pd.DataFrame, df2: pd.DataFrame, 
                            symbol1: str, symbol2: str) -> Dict[str, Any]:
        """Detect Pairs Trading Opportunities."""
        if df1.empty or df2.empty or len(df1) < 30 or len(df2) < 30:
            return {}
        
        # Align data
        common_dates = df1.index.intersection(df2.index)
        if len(common_dates) < 30:
            return {}
        
        price1 = df1.loc[common_dates, 'close']
        price2 = df2.loc[common_dates, 'close']
        
        # Calculate spread
        spread = price1 - price2
        spread_mean = spread.rolling(30).mean().iloc[-1]
        spread_std = spread.rolling(30).std().iloc[-1]
        
        if spread_std == 0:
            return {}
        
        z_score = (spread.iloc[-1] - spread_mean) / spread_std
        
        # Trading signal
        if z_score > 2:  # Spread too wide - sell spread
            return {
                'symbol': f"{symbol1}-{symbol2}",
                'strategy': 'Pairs Trading',
                'signal': 'SELL_SPREAD',
                'confidence': 0.75,
                'z_score': float(z_score),
                'action': f'Sell {symbol1}, Buy {symbol2}',
                'reason': 'Spread is 2+ standard deviations above mean'
            }
        elif z_score < -2:  # Spread too narrow - buy spread
            return {
                'symbol': f"{symbol1}-{symbol2}",
                'strategy': 'Pairs Trading',
                'signal': 'BUY_SPREAD',
                'confidence': 0.75,
                'z_score': float(z_score),
                'action': f'Buy {symbol1}, Sell {symbol2}',
                'reason': 'Spread is 2+ standard deviations below mean'
            }
        
        return {}
    
    def detect_statistical_arbitrage(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Statistical Arbitrage Opportunities."""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Calculate mean reversion metrics
        mean = returns.mean()
        std = returns.std()
        current_return = returns.iloc[-1]
        
        z_score = (current_return - mean) / std if std > 0 else 0
        
        # Mean reversion signal
        if z_score < -2:
            return {
                'symbol': symbol,
                'strategy': 'Statistical Arbitrage',
                'signal': 'BUY',
                'confidence': 0.70,
                'current_price': float(close.iloc[-1]),
                'z_score': float(z_score),
                'expected_return': float(mean),
                'reason': 'Price significantly below statistical mean'
            }
        elif z_score > 2:
            return {
                'symbol': symbol,
                'strategy': 'Statistical Arbitrage',
                'signal': 'SELL',
                'confidence': 0.70,
                'current_price': float(close.iloc[-1]),
                'z_score': float(z_score),
                'expected_return': float(mean),
                'reason': 'Price significantly above statistical mean'
            }
        
        return {}
    
    def detect_correlation_arbitrage(self, symbols_data: Dict[str, pd.DataFrame]) -> List[Dict[str, Any]]:
        """Detect Correlation Arbitrage Opportunities."""
        opportunities = []
        
        symbols = list(symbols_data.keys())
        if len(symbols) < 2:
            return opportunities
        
        # Calculate correlation matrix
        closes = {}
        for symbol, df in symbols_data.items():
            if not df.empty and 'close' in df.columns:
                closes[symbol] = df['close']
        
        if len(closes) < 2:
            return opportunities
        
        # Find pairs with high correlation
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                if sym1 in closes and sym2 in closes:
                    # Align data
                    common = closes[sym1].index.intersection(closes[sym2].index)
                    if len(common) > 30:
                        corr = closes[sym1].loc[common].corr(closes[sym2].loc[common])
                        if corr > 0.8:  # High correlation
                            # Check for divergence
                            ratio = closes[sym1].loc[common] / closes[sym2].loc[common]
                            ratio_z = (ratio.iloc[-1] - ratio.mean()) / ratio.std()
                            
                            if abs(ratio_z) > 2:
                                opportunities.append({
                                    'symbol': f"{sym1}-{sym2}",
                                    'strategy': 'Correlation Arbitrage',
                                    'signal': 'TRADE_PAIR',
                                    'confidence': 0.75,
                                    'correlation': float(corr),
                                    'divergence_z': float(ratio_z)
                                })
        
        return opportunities
    
    def detect_index_arbitrage(self, df: pd.DataFrame, index_data: pd.DataFrame, 
                               symbol: str) -> Dict[str, Any]:
        """Detect Index Arbitrage Opportunities."""
        if df.empty or index_data.empty:
            return {}
        
        # Align data
        common = df.index.intersection(index_data.index)
        if len(common) < 20:
            return {}
        
        stock_return = df.loc[common, 'close'].pct_change()
        index_return = index_data.loc[common, 'close'].pct_change()
        
        # Calculate beta
        if len(stock_return.dropna()) > 20 and len(index_return.dropna()) > 20:
            beta = stock_return.cov(index_return) / index_return.var()
            expected_return = beta * index_return.iloc[-1]
            actual_return = stock_return.iloc[-1]
            
            alpha = actual_return - expected_return
            
            if abs(alpha) > 0.02:  # 2% alpha
                return {
                    'symbol': symbol,
                    'strategy': 'Index Arbitrage',
                    'signal': 'BUY' if alpha > 0 else 'SELL',
                    'confidence': 0.70,
                    'alpha': float(alpha),
                    'beta': float(beta),
                    'reason': f'Significant alpha: {alpha:.2%}'
                }
        
        return {}
    
    def detect_volatility_arbitrage(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Volatility Arbitrage Opportunities."""
        if df.empty or len(df) < 30:
            return {}
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Realized volatility
        realized_vol = returns.rolling(20).std() * np.sqrt(252)
        
        # Expected volatility (from historical)
        expected_vol = returns.std() * np.sqrt(252)
        
        current_vol = realized_vol.iloc[-1]
        vol_ratio = current_vol / expected_vol if expected_vol > 0 else 1
        
        # Volatility mispricing
        if vol_ratio > 1.5:  # Volatility too high
            return {
                'symbol': symbol,
                'strategy': 'Volatility Arbitrage',
                'signal': 'SELL_VOLATILITY',
                'confidence': 0.75,
                'vol_ratio': float(vol_ratio),
                'current_vol': float(current_vol),
                'expected_vol': float(expected_vol),
                'reason': 'Realized volatility significantly above expected'
            }
        elif vol_ratio < 0.5:  # Volatility too low
            return {
                'symbol': symbol,
                'strategy': 'Volatility Arbitrage',
                'signal': 'BUY_VOLATILITY',
                'confidence': 0.75,
                'vol_ratio': float(vol_ratio),
                'current_vol': float(current_vol),
                'expected_vol': float(expected_vol),
                'reason': 'Realized volatility significantly below expected'
            }
        
        return {}
    
    # ========== QUANTITATIVE STRATEGIES (5 new) ==========
    
    def detect_factor_momentum(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Factor Momentum Opportunities."""
        if df.empty or len(df) < 100:
            return {}
        
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Multiple factor scores
        price_momentum = close.pct_change(20).iloc[-1]
        volume_momentum = volume.pct_change(20).iloc[-1]
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        
        # Factor scores (normalized)
        factors = {
            'momentum': price_momentum,
            'volume': volume_momentum,
            'volatility': -volatility  # Lower vol is better
        }
        
        # Weighted factor score
        factor_score = (
            factors['momentum'] * 0.5 +
            factors['volume'] * 0.3 +
            factors['volatility'] * 0.2
        )
        
        if factor_score > 0.05:
            return {
                'symbol': symbol,
                'strategy': 'Factor Momentum',
                'signal': 'STRONG_BUY',
                'confidence': 0.80,
                'factor_score': float(factor_score),
                'factors': {k: float(v) for k, v in factors.items()},
                'current_price': float(close.iloc[-1])
            }
        elif factor_score < -0.05:
            return {
                'symbol': symbol,
                'strategy': 'Factor Momentum',
                'signal': 'STRONG_SELL',
                'confidence': 0.80,
                'factor_score': float(factor_score),
                'factors': {k: float(v) for k, v in factors.items()},
                'current_price': float(close.iloc[-1])
            }
        
        return {}
    
    def detect_quantitative_value(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Quantitative Value Opportunities."""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        
        # Value metrics (simplified - would use financials in real implementation)
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else sma_50
        
        # Price relative to moving averages
        price_to_sma50 = close.iloc[-1] / sma_50
        price_to_sma200 = close.iloc[-1] / sma_200
        
        # Value score (lower is better)
        value_score = (price_to_sma50 + price_to_sma200) / 2
        
        if value_score < 0.9:  # Undervalued
            return {
                'symbol': symbol,
                'strategy': 'Quantitative Value',
                'signal': 'BUY',
                'confidence': 0.75,
                'value_score': float(value_score),
                'current_price': float(close.iloc[-1]),
                'fair_value': float(sma_50),
                'reason': 'Price significantly below moving averages'
            }
        elif value_score > 1.1:  # Overvalued
            return {
                'symbol': symbol,
                'strategy': 'Quantitative Value',
                'signal': 'SELL',
                'confidence': 0.75,
                'value_score': float(value_score),
                'current_price': float(close.iloc[-1]),
                'fair_value': float(sma_50),
                'reason': 'Price significantly above moving averages'
            }
        
        return {}
    
    def detect_machine_learning_signals(self, df: pd.DataFrame, symbol: str, 
                                        ml_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """Detect ML-based Trading Signals."""
        if df.empty or not ml_predictions:
            return {}
        
        pred = ml_predictions.get(symbol, {})
        if not pred:
            return {}
        
        direction = pred.get('direction', 'HOLD')
        confidence = pred.get('confidence', 0.5)
        
        if confidence > 0.8 and direction in ['BUY', 'SELL']:
            return {
                'symbol': symbol,
                'strategy': 'Machine Learning',
                'signal': f'STRONG_{direction}',
                'confidence': confidence,
                'ml_model_confidence': confidence,
                'current_price': float(df['close'].iloc[-1]),
                'reason': f'ML model predicts {direction} with {confidence:.1%} confidence'
            }
        
        return {}
    
    def detect_regime_based_strategy(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Regime-Based Strategy Signals."""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        returns = close.pct_change().dropna()
        
        # Market regime detection
        volatility = returns.rolling(20).std().iloc[-1]
        trend = close.rolling(20).mean().iloc[-1] - close.rolling(50).mean().iloc[-1] if len(close) >= 50 else 0
        
        # Regime classification
        if volatility > returns.std() * 1.5:
            regime = 'HIGH_VOLATILITY'
            strategy = 'MEAN_REVERSION'
        elif trend > 0:
            regime = 'BULL_TREND'
            strategy = 'MOMENTUM'
        else:
            regime = 'BEAR_TREND'
            strategy = 'SHORT_MOMENTUM'
        
        # Generate signal based on regime
        if strategy == 'MOMENTUM' and close.iloc[-1] > close.rolling(20).mean().iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Regime-Based',
                'signal': 'BUY',
                'confidence': 0.75,
                'regime': regime,
                'strategy_type': strategy,
                'current_price': float(close.iloc[-1])
            }
        elif strategy == 'MEAN_REVERSION' and close.iloc[-1] < close.rolling(20).mean().iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Regime-Based',
                'signal': 'BUY',
                'confidence': 0.70,
                'regime': regime,
                'strategy_type': strategy,
                'current_price': float(close.iloc[-1])
            }
        
        return {}
    
    def detect_adaptive_strategy(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Adaptive Strategy Signals (combines multiple strategies)."""
        if df.empty or len(df) < 50:
            return {}
        
        # Combine signals from multiple strategies
        signals = []
        
        # Momentum signal
        momentum = df['close'].pct_change(20).iloc[-1]
        if momentum > 0.05:
            signals.append(('MOMENTUM', 'BUY', 0.7))
        elif momentum < -0.05:
            signals.append(('MOMENTUM', 'SELL', 0.7))
        
        # Mean reversion signal
        sma = df['close'].rolling(20).mean().iloc[-1]
        current = df['close'].iloc[-1]
        if current < sma * 0.95:
            signals.append(('MEAN_REVERSION', 'BUY', 0.6))
        elif current > sma * 1.05:
            signals.append(('MEAN_REVERSION', 'SELL', 0.6))
        
        # Breakout signal
        high_20 = df['high'].rolling(20).max().iloc[-1]
        if current > high_20 * 0.99:
            signals.append(('BREAKOUT', 'BUY', 0.75))
        
        # Aggregate signals
        if signals:
            buy_signals = [s for s in signals if s[1] == 'BUY']
            sell_signals = [s for s in signals if s[1] == 'SELL']
            
            if len(buy_signals) > len(sell_signals):
                avg_confidence = np.mean([s[2] for s in buy_signals])
                strategies = [s[0] for s in buy_signals]
                return {
                    'symbol': symbol,
                    'strategy': 'Adaptive',
                    'signal': 'STRONG_BUY' if avg_confidence > 0.7 else 'BUY',
                    'confidence': avg_confidence,
                    'strategies': strategies,
                    'current_price': float(current)
                }
            elif len(sell_signals) > len(buy_signals):
                avg_confidence = np.mean([s[2] for s in sell_signals])
                strategies = [s[0] for s in sell_signals]
                return {
                    'symbol': symbol,
                    'strategy': 'Adaptive',
                    'signal': 'STRONG_SELL' if avg_confidence > 0.7 else 'SELL',
                    'confidence': avg_confidence,
                    'strategies': strategies,
                    'current_price': float(current)
                }
        
        return {}
    
    # ========== HIGH-FREQUENCY STRATEGIES (5 new) ==========
    
    def detect_microstructure_signals(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Market Microstructure Signals."""
        if df.empty or len(df) < 20:
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Bid-ask spread proxy (high-low range)
        spread = (high - low) / close
        avg_spread = spread.rolling(20).mean().iloc[-1]
        current_spread = spread.iloc[-1]
        
        # Order flow imbalance
        price_change = close.diff()
        volume_weighted = price_change * volume
        buy_pressure = volume_weighted.where(volume_weighted > 0, 0).rolling(10).sum()
        sell_pressure = volume_weighted.where(volume_weighted < 0, 0).abs().rolling(10).sum()
        
        imbalance = (buy_pressure - sell_pressure) / (buy_pressure + sell_pressure + 1e-10)
        current_imbalance = imbalance.iloc[-1]
        
        if current_imbalance > 0.3 and current_spread < avg_spread:
            return {
                'symbol': symbol,
                'strategy': 'Microstructure',
                'signal': 'BUY',
                'confidence': 0.75,
                'order_flow_imbalance': float(current_imbalance),
                'spread_tight': current_spread < avg_spread,
                'current_price': float(close.iloc[-1])
            }
        
        return {}
    
    def detect_liquidity_provision(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Liquidity Provision Opportunities."""
        if df.empty or len(df) < 30:
            return {}
        
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Volume profile
        avg_volume = volume.rolling(20).mean()
        current_volume = volume.iloc[-1]
        
        # Price stability
        volatility = close.pct_change().rolling(20).std().iloc[-1]
        
        # Liquidity score
        liquidity_score = (current_volume / avg_volume.iloc[-1]) * (1 / (volatility + 0.001))
        
        if liquidity_score > 2:  # High liquidity
            return {
                'symbol': symbol,
                'strategy': 'Liquidity Provision',
                'signal': 'PROVIDE_LIQUIDITY',
                'confidence': 0.70,
                'liquidity_score': float(liquidity_score),
                'current_price': float(close.iloc[-1]),
                'reason': 'High liquidity, low volatility - good for market making'
            }
        
        return {}
    
    def detect_momentum_continuation(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Momentum Continuation Signals."""
        if df.empty or len(df) < 30:
            return {}
        
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Multiple timeframe momentum
        mom_5 = close.pct_change(5).iloc[-1]
        mom_10 = close.pct_change(10).iloc[-1]
        mom_20 = close.pct_change(20).iloc[-1]
        
        # Volume confirmation
        vol_trend = volume.tail(5).mean() > volume.tail(20).head(5).mean()
        
        # Momentum alignment
        if mom_5 > 0.02 and mom_10 > 0.01 and mom_20 > 0 and vol_trend:
            return {
                'symbol': symbol,
                'strategy': 'Momentum Continuation',
                'signal': 'STRONG_BUY',
                'confidence': 0.80,
                'momentum_5d': float(mom_5),
                'momentum_10d': float(mom_10),
                'momentum_20d': float(mom_20),
                'volume_confirmed': vol_trend,
                'current_price': float(close.iloc[-1])
            }
        elif mom_5 < -0.02 and mom_10 < -0.01 and mom_20 < 0 and vol_trend:
            return {
                'symbol': symbol,
                'strategy': 'Momentum Continuation',
                'signal': 'STRONG_SELL',
                'confidence': 0.80,
                'momentum_5d': float(mom_5),
                'momentum_10d': float(mom_10),
                'momentum_20d': float(mom_20),
                'volume_confirmed': vol_trend,
                'current_price': float(close.iloc[-1])
            }
        
        return {}
    
    def detect_reversal_patterns(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Reversal Patterns."""
        if df.empty or len(df) < 20:
            return {}
        
        close = df['close']
        high = df['high']
        low = df['low']
        
        # Hammer pattern (bullish reversal)
        open_price = df.get('open', close)
        body = abs(close - open_price)
        lower_shadow = pd.concat([open_price, close], axis=1).min(axis=1) - low
        upper_shadow = high - pd.concat([open_price, close], axis=1).max(axis=1)
        
        # Hammer: small body, long lower shadow, little/no upper shadow
        is_hammer = (body < (high - low) * 0.3) & (lower_shadow > body * 2) & (upper_shadow < body)
        
        if is_hammer.iloc[-1] and close.iloc[-1] < close.rolling(10).mean().iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Reversal Pattern',
                'signal': 'BUY',
                'confidence': 0.70,
                'pattern': 'Hammer',
                'current_price': float(close.iloc[-1]),
                'reason': 'Bullish reversal pattern detected'
            }
        
        # Shooting star (bearish reversal)
        is_shooting_star = (body < (high - low) * 0.3) & (upper_shadow > body * 2) & (lower_shadow < body)
        
        if is_shooting_star.iloc[-1] and close.iloc[-1] > close.rolling(10).mean().iloc[-1]:
            return {
                'symbol': symbol,
                'strategy': 'Reversal Pattern',
                'signal': 'SELL',
                'confidence': 0.70,
                'pattern': 'Shooting Star',
                'current_price': float(close.iloc[-1]),
                'reason': 'Bearish reversal pattern detected'
            }
        
        return {}
    
    def detect_trend_exhaustion(self, df: pd.DataFrame, symbol: str) -> Dict[str, Any]:
        """Detect Trend Exhaustion Signals."""
        if df.empty or len(df) < 50:
            return {}
        
        close = df['close']
        volume = df.get('volume', pd.Series(1, index=df.index))
        
        # Trend strength
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean() if len(close) >= 50 else sma_20
        
        trend = (sma_20.iloc[-1] - sma_50.iloc[-1]) / sma_50.iloc[-1]
        
        # Momentum divergence
        price_momentum = close.pct_change(10).iloc[-1]
        volume_trend = volume.tail(10).mean() / volume.tail(20).head(10).mean()
        
        # Exhaustion: strong trend but weakening momentum and volume
        if abs(trend) > 0.1:  # Strong trend
            if trend > 0 and price_momentum < 0 and volume_trend < 0.8:  # Bullish exhaustion
                return {
                    'symbol': symbol,
                    'strategy': 'Trend Exhaustion',
                    'signal': 'SELL',
                    'confidence': 0.75,
                    'trend_strength': float(trend),
                    'momentum_divergence': True,
                    'volume_divergence': True,
                    'current_price': float(close.iloc[-1]),
                    'reason': 'Bullish trend showing exhaustion signs'
                }
            elif trend < 0 and price_momentum > 0 and volume_trend < 0.8:  # Bearish exhaustion
                return {
                    'symbol': symbol,
                    'strategy': 'Trend Exhaustion',
                    'signal': 'BUY',
                    'confidence': 0.75,
                    'trend_strength': float(trend),
                    'momentum_divergence': True,
                    'volume_divergence': True,
                    'current_price': float(close.iloc[-1]),
                    'reason': 'Bearish trend showing exhaustion signs'
                }
        
        return {}
    
    # ========== PORTFOLIO OPTIMIZATION STRATEGIES (5 new) ==========
    
    def optimize_portfolio_allocation(self, symbols_data: Dict[str, pd.DataFrame], 
                                     risk_budget: float = 0.02) -> Dict[str, Any]:
        """Optimize Portfolio Allocation using Modern Portfolio Theory."""
        if len(symbols_data) < 2:
            return {}
        
        # Calculate returns and covariance
        returns_data = {}
        for symbol, df in symbols_data.items():
            if not df.empty and 'close' in df.columns:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {}
        
        # Align returns
        common_dates = set.intersection(*[set(r.index) for r in returns_data.values()])
        if len(common_dates) < 30:
            return {}
        
        aligned_returns = pd.DataFrame({sym: r.loc[list(common_dates)] for sym, r in returns_data.items()})
        
        mean_returns = aligned_returns.mean()
        cov_matrix = aligned_returns.cov()
        
        # Optimize portfolio
        n = len(mean_returns)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        def portfolio_return(weights):
            return np.dot(weights, mean_returns)
        
        # Constraints
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        
        # Optimize for maximum Sharpe ratio
        initial_weights = np.array([1/n] * n)
        
        try:
            result = minimize(
                lambda w: -portfolio_return(w) / np.sqrt(portfolio_variance(w)),
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                allocations = {symbol: float(w) for symbol, w in zip(mean_returns.index, optimal_weights) if w > 0.01}
                
                return {
                    'strategy': 'Portfolio Optimization',
                    'signal': 'OPTIMIZE_ALLOCATION',
                    'allocations': allocations,
                    'expected_return': float(portfolio_return(optimal_weights)),
                    'expected_volatility': float(np.sqrt(portfolio_variance(optimal_weights))),
                    'sharpe_ratio': float(portfolio_return(optimal_weights) / np.sqrt(portfolio_variance(optimal_weights)))
                }
        except:
            pass
        
        return {}
    
    def detect_risk_parity_allocation(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect Risk Parity Portfolio Allocation."""
        if len(symbols_data) < 2:
            return {}
        
        # Calculate risk contribution
        returns_data = {}
        for symbol, df in symbols_data.items():
            if not df.empty and 'close' in df.columns:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {}
        
        volatilities = {sym: r.std() * np.sqrt(252) for sym, r in returns_data.items()}
        
        # Risk parity: equal risk contribution
        inv_vol = {sym: 1 / vol if vol > 0 else 0 for sym, vol in volatilities.items()}
        total_inv_vol = sum(inv_vol.values())
        
        if total_inv_vol > 0:
            allocations = {sym: inv_vol[sym] / total_inv_vol for sym in inv_vol.keys()}
            
            return {
                'strategy': 'Risk Parity',
                'signal': 'RISK_PARITY_ALLOCATION',
                'allocations': allocations,
                'reason': 'Equal risk contribution from each asset'
            }
        
        return {}
    
    def detect_minimum_variance_portfolio(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect Minimum Variance Portfolio."""
        if len(symbols_data) < 2:
            return {}
        
        # Calculate covariance
        returns_data = {}
        for symbol, df in symbols_data.items():
            if not df.empty and 'close' in df.columns:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {}
        
        common_dates = set.intersection(*[set(r.index) for r in returns_data.values()])
        if len(common_dates) < 30:
            return {}
        
        aligned_returns = pd.DataFrame({sym: r.loc[list(common_dates)] for sym, r in returns_data.items()})
        cov_matrix = aligned_returns.cov()
        
        n = len(cov_matrix)
        
        def portfolio_variance(weights):
            return np.dot(weights.T, np.dot(cov_matrix, weights))
        
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = tuple((0, 1) for _ in range(n))
        initial_weights = np.array([1/n] * n)
        
        try:
            result = minimize(
                portfolio_variance,
                initial_weights,
                method='SLSQP',
                bounds=bounds,
                constraints=constraints
            )
            
            if result.success:
                optimal_weights = result.x
                allocations = {symbol: float(w) for symbol, w in zip(cov_matrix.index, optimal_weights) if w > 0.01}
                
                return {
                    'strategy': 'Minimum Variance',
                    'signal': 'MIN_VAR_ALLOCATION',
                    'allocations': allocations,
                    'portfolio_variance': float(portfolio_variance(optimal_weights))
                }
        except:
            pass
        
        return {}
    
    def detect_maximum_diversification(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect Maximum Diversification Portfolio."""
        if len(symbols_data) < 2:
            return {}
        
        # Calculate correlation matrix
        returns_data = {}
        for symbol, df in symbols_data.items():
            if not df.empty and 'close' in df.columns:
                returns_data[symbol] = df['close'].pct_change().dropna()
        
        if len(returns_data) < 2:
            return {}
        
        common_dates = set.intersection(*[set(r.index) for r in returns_data.values()])
        if len(common_dates) < 30:
            return {}
        
        aligned_returns = pd.DataFrame({sym: r.loc[list(common_dates)] for sym, r in returns_data.items()})
        corr_matrix = aligned_returns.corr()
        
        # Find low correlation pairs
        low_corr_pairs = []
        symbols = list(corr_matrix.index)
        for i, sym1 in enumerate(symbols):
            for sym2 in symbols[i+1:]:
                corr = corr_matrix.loc[sym1, sym2]
                if abs(corr) < 0.3:  # Low correlation
                    low_corr_pairs.append((sym1, sym2, corr))
        
        if low_corr_pairs:
            # Allocate more to low correlation pairs
            allocations = {}
            for sym1, sym2, corr in low_corr_pairs[:5]:  # Top 5 pairs
                allocations[sym1] = allocations.get(sym1, 0) + 0.1
                allocations[sym2] = allocations.get(sym2, 0) + 0.1
            
            # Normalize
            total = sum(allocations.values())
            if total > 0:
                allocations = {k: v / total for k, v in allocations.items()}
                
                return {
                    'strategy': 'Maximum Diversification',
                    'signal': 'DIVERSIFY_ALLOCATION',
                    'allocations': allocations,
                    'low_correlation_pairs': len(low_corr_pairs),
                    'reason': 'Focus on low correlation assets for maximum diversification'
                }
        
        return {}
    
    def detect_tactical_asset_allocation(self, symbols_data: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Detect Tactical Asset Allocation Signals."""
        if len(symbols_data) < 2:
            return {}
        
        # Score each asset
        scores = {}
        for symbol, df in symbols_data.items():
            if df.empty or len(df) < 20:
                continue
            
            close = df['close']
            
            # Multiple factors
            momentum = close.pct_change(20).iloc[-1]
            trend = 1 if close.iloc[-1] > close.rolling(20).mean().iloc[-1] else -1
            volatility = close.pct_change().rolling(20).std().iloc[-1]
            
            # Composite score
            score = momentum * 0.5 + trend * 0.3 - volatility * 0.2
            scores[symbol] = score
        
        if scores:
            # Allocate based on scores
            total_score = sum(abs(s) for s in scores.values())
            if total_score > 0:
                allocations = {sym: abs(score) / total_score for sym, score in scores.items()}
                
                return {
                    'strategy': 'Tactical Allocation',
                    'signal': 'TACTICAL_ALLOCATION',
                    'allocations': allocations,
                    'top_assets': sorted(scores.items(), key=lambda x: x[1], reverse=True)[:5]
                }
        
        return {}

