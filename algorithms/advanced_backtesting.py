"""
ADVANCED BACKTESTING ENGINE - 200X UPGRADE
Comprehensive backtesting system with walk-forward analysis, Monte Carlo simulation, and more
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


class AdvancedBacktestingEngine:
    """
    Advanced backtesting engine with multiple analysis methods.
    
    Features:
    - Walk-forward optimization
    - Monte Carlo simulation
    - Out-of-sample testing
    - Performance metrics (Sharpe, Sortino, Calmar, etc.)
    - Drawdown analysis
    - Trade analysis
    - Risk-adjusted returns
    """
    
    def __init__(self, initial_capital: float = 10000.0):
        """Initialize backtesting engine."""
        self.logger = logging.getLogger("ai_investment_bot.advanced_backtesting")
        self.initial_capital = initial_capital
        self.results_cache = {}
        
    def run_backtest(
        self,
        signals: pd.DataFrame,
        prices: pd.Series,
        commission: float = 0.001,
        slippage: float = 0.0005
    ) -> Dict[str, Any]:
        """
        Run comprehensive backtest.
        
        Args:
            signals: DataFrame with 'action' (BUY/SELL), 'price', 'timestamp'
            prices: Series of historical prices
            commission: Commission rate (default 0.1%)
            slippage: Slippage rate (default 0.05%)
        """
        try:
            if signals.empty or prices.empty:
                return {'error': 'Empty data'}
            
            # Initialize tracking
            capital = self.initial_capital
            position = 0
            position_price = 0.0
            trades = []
            equity_curve = [capital]
            timestamps = [signals.index[0] if hasattr(signals.index[0], 'date') else datetime.now()]
            
            # Process signals
            for idx, signal in signals.iterrows():
                action = signal.get('action', 'HOLD')
                price = signal.get('price', prices.get(idx, 0))
                
                if price <= 0:
                    continue
                
                # Apply slippage
                if action == 'BUY':
                    execution_price = price * (1 + slippage)
                elif action == 'SELL':
                    execution_price = price * (1 - slippage)
                else:
                    execution_price = price
                
                # Execute trade
                if action == 'BUY' and position == 0:
                    # Calculate position size (use available capital)
                    position_value = capital * 0.95  # Use 95% of capital
                    shares = int(position_value / execution_price)
                    cost = shares * execution_price * (1 + commission)
                    
                    if cost <= capital:
                        capital -= cost
                        position = shares
                        position_price = execution_price
                        
                        trades.append({
                            'timestamp': idx,
                            'action': 'BUY',
                            'price': execution_price,
                            'shares': shares,
                            'value': cost
                        })
                
                elif action == 'SELL' and position > 0:
                    # Sell position
                    proceeds = position * execution_price * (1 - commission)
                    capital += proceeds
                    
                    # Calculate P&L
                    pnl = proceeds - (position * position_price)
                    pnl_pct = (execution_price / position_price - 1) * 100
                    
                    trades.append({
                        'timestamp': idx,
                        'action': 'SELL',
                        'price': execution_price,
                        'shares': position,
                        'value': proceeds,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct
                    })
                    
                    position = 0
                    position_price = 0.0
                
                # Update equity curve
                current_value = capital + (position * execution_price if position > 0 else 0)
                equity_curve.append(current_value)
                timestamps.append(idx)
            
            # Close any open position at end
            if position > 0:
                final_price = prices.iloc[-1]
                proceeds = position * final_price * (1 - commission)
                capital += proceeds
                equity_curve[-1] = capital
            
            # Calculate metrics
            equity_series = pd.Series(equity_curve, index=timestamps[:len(equity_curve)])
            returns = equity_series.pct_change().dropna()
            
            metrics = self.calculate_performance_metrics(
                equity_series, returns, trades
            )
            
            return {
                'initial_capital': self.initial_capital,
                'final_capital': capital,
                'total_return': (capital / self.initial_capital - 1) * 100,
                'trades': trades,
                'equity_curve': equity_series,
                'returns': returns,
                'metrics': metrics,
                'total_trades': len([t for t in trades if t['action'] == 'SELL']),
                'winning_trades': len([t for t in trades if t.get('pnl', 0) > 0]),
                'losing_trades': len([t for t in trades if t.get('pnl', 0) < 0])
            }
            
        except Exception as e:
            self.logger.error(f"Error running backtest: {e}")
            return {'error': str(e)}
    
    def calculate_performance_metrics(
        self,
        equity: pd.Series,
        returns: pd.Series,
        trades: List[Dict]
    ) -> Dict[str, float]:
        """Calculate comprehensive performance metrics."""
        try:
            if equity.empty or returns.empty:
                return {}
            
            total_return = (equity.iloc[-1] / equity.iloc[0] - 1) * 100
            
            # Annualized return
            days = (equity.index[-1] - equity.index[0]).days if hasattr(equity.index[0], 'date') else 252
            annual_return = ((equity.iloc[-1] / equity.iloc[0]) ** (252 / max(days, 1)) - 1) * 100
            
            # Volatility (annualized)
            volatility = returns.std() * np.sqrt(252) * 100
            
            # Sharpe Ratio (assuming risk-free rate = 0)
            sharpe_ratio = (annual_return / volatility) if volatility > 0 else 0.0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = downside_returns.std() * np.sqrt(252) if len(downside_returns) > 0 else 0.01
            sortino_ratio = (annual_return / (downside_std * 100)) if downside_std > 0 else 0.0
            
            # Maximum Drawdown
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1) * 100
            max_drawdown = drawdown.min()
            
            # Calmar Ratio
            calmar_ratio = (annual_return / abs(max_drawdown)) if max_drawdown != 0 else 0.0
            
            # Win rate
            closed_trades = [t for t in trades if t['action'] == 'SELL' and 'pnl' in t]
            if closed_trades:
                winning = len([t for t in closed_trades if t['pnl'] > 0])
                win_rate = (winning / len(closed_trades)) * 100
                
                # Average win/loss
                wins = [t['pnl'] for t in closed_trades if t['pnl'] > 0]
                losses = [abs(t['pnl']) for t in closed_trades if t['pnl'] < 0]
                
                avg_win = np.mean(wins) if wins else 0.0
                avg_loss = np.mean(losses) if losses else 0.0
                profit_factor = (sum(wins) / sum(losses)) if losses and sum(losses) > 0 else 0.0
            else:
                win_rate = 0.0
                avg_win = 0.0
                avg_loss = 0.0
                profit_factor = 0.0
            
            return {
                'total_return_pct': total_return,
                'annual_return_pct': annual_return,
                'volatility_pct': volatility,
                'sharpe_ratio': sharpe_ratio,
                'sortino_ratio': sortino_ratio,
                'max_drawdown_pct': max_drawdown,
                'calmar_ratio': calmar_ratio,
                'win_rate_pct': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': profit_factor,
                'total_trades': len(closed_trades)
            }
            
        except Exception as e:
            self.logger.error(f"Error calculating metrics: {e}")
            return {}
    
    def walk_forward_optimization(
        self,
        historical_data: pd.DataFrame,
        train_period: int = 252,
        test_period: int = 63,
        step: int = 21
    ) -> Dict[str, Any]:
        """
        Walk-forward optimization to test strategy robustness.
        
        Args:
            historical_data: DataFrame with OHLCV data
            train_period: Training period in days
            test_period: Testing period in days
            step: Step size for rolling window
        """
        try:
            if len(historical_data) < train_period + test_period:
                return {'error': 'Insufficient data'}
            
            results = []
            start_idx = 0
            
            while start_idx + train_period + test_period <= len(historical_data):
                # Split data
                train_data = historical_data.iloc[start_idx:start_idx + train_period]
                test_data = historical_data.iloc[start_idx + train_period:start_idx + train_period + test_period]
                
                # Here you would:
                # 1. Train strategy on train_data
                # 2. Test on test_data
                # 3. Record results
                
                # For now, return structure
                results.append({
                    'train_start': train_data.index[0],
                    'train_end': train_data.index[-1],
                    'test_start': test_data.index[0],
                    'test_end': test_data.index[-1],
                    'train_period': len(train_data),
                    'test_period': len(test_data)
                })
                
                start_idx += step
            
            return {
                'total_windows': len(results),
                'windows': results,
                'avg_test_period': np.mean([r['test_period'] for r in results])
            }
            
        except Exception as e:
            self.logger.error(f"Error in walk-forward optimization: {e}")
            return {'error': str(e)}
    
    def monte_carlo_simulation(
        self,
        returns: pd.Series,
        n_simulations: int = 1000,
        days: int = 252
    ) -> Dict[str, Any]:
        """
        Monte Carlo simulation of future returns.
        
        Args:
            returns: Historical returns
            n_simulations: Number of simulations
            days: Number of days to simulate
        """
        try:
            if returns.empty:
                return {'error': 'Empty returns'}
            
            mean_return = returns.mean()
            std_return = returns.std()
            
            # Run simulations
            simulations = []
            for _ in range(n_simulations):
                # Generate random returns
                random_returns = np.random.normal(mean_return, std_return, days)
                # Calculate cumulative return
                cumulative = (1 + pd.Series(random_returns)).prod()
                simulations.append(cumulative)
            
            simulations = np.array(simulations)
            
            # Calculate statistics
            return {
                'mean_final_value': np.mean(simulations),
                'median_final_value': np.median(simulations),
                'std_final_value': np.std(simulations),
                'min_final_value': np.min(simulations),
                'max_final_value': np.max(simulations),
                'percentile_5': np.percentile(simulations, 5),
                'percentile_95': np.percentile(simulations, 95),
                'probability_profit': (simulations > 1.0).sum() / len(simulations) * 100,
                'simulations': simulations.tolist()[:100]  # Return first 100 for plotting
            }
            
        except Exception as e:
            self.logger.error(f"Error in Monte Carlo simulation: {e}")
            return {'error': str(e)}
    
    def analyze_drawdowns(
        self,
        equity: pd.Series
    ) -> Dict[str, Any]:
        """Analyze drawdown periods."""
        try:
            if equity.empty:
                return {}
            
            # Calculate drawdown
            cumulative = equity / equity.iloc[0]
            running_max = cumulative.expanding().max()
            drawdown = (cumulative / running_max - 1) * 100
            
            # Find drawdown periods
            in_drawdown = drawdown < 0
            drawdown_periods = []
            
            start_idx = None
            for i, is_dd in enumerate(in_drawdown):
                if is_dd and start_idx is None:
                    start_idx = i
                elif not is_dd and start_idx is not None:
                    # Drawdown ended
                    dd_period = drawdown.iloc[start_idx:i]
                    max_dd = dd_period.min()
                    duration = len(dd_period)
                    
                    drawdown_periods.append({
                        'start': equity.index[start_idx],
                        'end': equity.index[i-1],
                        'max_drawdown': max_dd,
                        'duration_days': duration
                    })
                    start_idx = None
            
            # If still in drawdown at end
            if start_idx is not None:
                dd_period = drawdown.iloc[start_idx:]
                max_dd = dd_period.min()
                duration = len(dd_period)
                
                drawdown_periods.append({
                    'start': equity.index[start_idx],
                    'end': equity.index[-1],
                    'max_drawdown': max_dd,
                    'duration_days': duration
                })
            
            return {
                'max_drawdown': drawdown.min(),
                'current_drawdown': drawdown.iloc[-1],
                'drawdown_periods': drawdown_periods,
                'avg_drawdown_duration': np.mean([d['duration_days'] for d in drawdown_periods]) if drawdown_periods else 0,
                'total_drawdown_periods': len(drawdown_periods)
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing drawdowns: {e}")
            return {}

