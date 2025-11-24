"""
Professional investor charts - Candlestick, Volume Profile, Correlation, Risk Metrics
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional

from algorithms.professional_analysis import ProfessionalAnalysis


class ProfessionalCharts:
    """Professional-grade charts for institutional investors."""
    
    def __init__(self):
        self.analysis = ProfessionalAnalysis()
    
    def render_candlestick_chart(
        self, 
        df: pd.DataFrame,
        symbol: str,
        indicators: Optional[Dict] = None
    ) -> go.Figure:
        """
        Create professional candlestick chart with indicators.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name
            indicators: Optional dictionary of indicators to overlay
            
        Returns:
            Plotly figure
        """
        fig = make_subplots(
            rows=3, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.03,
            row_heights=[0.6, 0.2, 0.2],
            subplot_titles=(f'{symbol} - Price Action', 'Volume', 'Indicators')
        )
        
        # Prepare x-axis data
        if isinstance(df.index, pd.DatetimeIndex):
            x_data = df.index
        else:
            x_data = list(range(len(df)))
        
        # Candlestick chart
        fig.add_trace(
            go.Candlestick(
                x=x_data,
                open=df['open'],
                high=df['high'],
                low=df['low'],
                close=df['close'],
                name='Price'
            ),
            row=1, col=1
        )
        
        # Add moving averages if available
        if indicators:
            if 'sma_20' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=df['sma_20'],
                        name='SMA 20',
                        line=dict(color='blue', width=1)
                    ),
                    row=1, col=1
                )
            
            if 'sma_50' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=df['sma_50'],
                        name='SMA 50',
                        line=dict(color='orange', width=1)
                    ),
                    row=1, col=1
                )
            
            # VWAP
            if 'vwap' in df.columns:
                fig.add_trace(
                    go.Scatter(
                        x=x_data,
                        y=df['vwap'],
                        name='VWAP',
                        line=dict(color='purple', width=1, dash='dash')
                    ),
                    row=1, col=1
                )
        
        # Volume bars
        colors = ['red' if df['close'].iloc[i] < df['open'].iloc[i] else 'green' 
                 for i in range(len(df))]
        fig.add_trace(
            go.Bar(
                x=x_data,
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # RSI if available
        if 'rsi' in df.columns:
            fig.add_trace(
                go.Scatter(
                    x=x_data,
                    y=df['rsi'],
                    name='RSI',
                    line=dict(color='purple', width=1)
                ),
                row=3, col=1
            )
            # Add RSI levels
            fig.add_hline(y=70, line_dash="dash", line_color="red", row=3, col=1)
            fig.add_hline(y=30, line_dash="dash", line_color="green", row=3, col=1)
            fig.add_hline(y=50, line_dash="dot", line_color="gray", row=3, col=1)
        
        # Update layout
        fig.update_layout(
            title=f'{symbol} - Professional Chart',
            xaxis_rangeslider_visible=False,
            height=800,
            showlegend=True,
            template='plotly_dark'
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        fig.update_yaxes(title_text="RSI", row=3, col=1)
        
        return fig
    
    def render_volume_profile(
        self, 
        df: pd.DataFrame,
        symbol: str
    ) -> go.Figure:
        """
        Create volume profile chart.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name
            
        Returns:
            Plotly figure
        """
        volume_profile = self.analysis.calculate_volume_profile(df)
        
        if volume_profile.empty:
            return go.Figure()
        
        fig = go.Figure()
        
        # Horizontal bar chart for volume profile
        fig.add_trace(
            go.Bar(
                x=volume_profile['volume'],
                y=volume_profile['price'],
                orientation='h',
                name='Volume Profile',
                marker_color='lightblue'
            )
        )
        
        # Add current price line
        current_price = df['close'].iloc[-1]
        fig.add_hline(
            y=current_price,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current Price: ${current_price:.2f}"
        )
        
        # POC (Point of Control) - price with highest volume
        poc_price = volume_profile.loc[volume_profile['volume'].idxmax(), 'price']
        fig.add_hline(
            y=poc_price,
            line_dash="dot",
            line_color="green",
            annotation_text=f"POC: ${poc_price:.2f}"
        )
        
        fig.update_layout(
            title=f'{symbol} - Volume Profile',
            xaxis_title='Volume',
            yaxis_title='Price',
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def render_correlation_heatmap(
        self, 
        correlation_matrix: pd.DataFrame
    ) -> go.Figure:
        """
        Create correlation heatmap.
        
        Args:
            correlation_matrix: Correlation matrix DataFrame
            
        Returns:
            Plotly figure
        """
        if correlation_matrix.empty:
            return go.Figure()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.index,
            colorscale='RdBu',
            zmid=0,
            text=correlation_matrix.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Correlation")
        ))
        
        fig.update_layout(
            title='Portfolio Correlation Matrix',
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def render_support_resistance_chart(
        self, 
        df: pd.DataFrame,
        symbol: str
    ) -> go.Figure:
        """
        Create chart with support and resistance levels.
        
        Args:
            df: DataFrame with OHLCV data
            symbol: Symbol name
            
        Returns:
            Plotly figure
        """
        levels = self.analysis.detect_support_resistance(df)
        
        fig = go.Figure()
        
        # Prepare x-axis data
        if isinstance(df.index, pd.DatetimeIndex):
            x_axis = df.index
        else:
            x_axis = list(range(len(df)))
        
        # Price line
        fig.add_trace(
            go.Scatter(
                x=x_axis,
                y=df['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            )
        )
        
        # Support levels
        for i, level in enumerate(levels['support']):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="green",
                annotation_text=f"Support {i+1}: ${level:.2f}",
                annotation_position="right"
            )
        
        # Resistance levels
        for i, level in enumerate(levels['resistance']):
            fig.add_hline(
                y=level,
                line_dash="dash",
                line_color="red",
                annotation_text=f"Resistance {i+1}: ${level:.2f}",
                annotation_position="right"
            )
        
        # Fibonacci levels
        if len(df) > 0:
            recent_high = df['high'].tail(50).max()
            recent_low = df['low'].tail(50).min()
            fib_levels = self.analysis.detect_fibonacci_levels(recent_high, recent_low)
            
            for name, level in fib_levels.items():
                if name != 'fib_0' and name != 'fib_100':
                    fig.add_hline(
                        y=level,
                        line_dash="dot",
                        line_color="orange",
                        opacity=0.5,
                        annotation_text=name.replace('fib_', 'Fib '),
                        annotation_position="left"
                    )
        
        fig.update_layout(
            title=f'{symbol} - Support & Resistance Analysis',
            xaxis_title='Time',
            yaxis_title='Price',
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        return fig
    
    def render_risk_metrics_dashboard(
        self, 
        risk_metrics: Dict[str, float],
        symbol: str
    ) -> None:
        """
        Render risk metrics dashboard.
        
        Args:
            risk_metrics: Dictionary of risk metrics
            symbol: Symbol name
        """
        st.subheader(f"📊 Risk Metrics - {symbol}")
        
        if not risk_metrics:
            st.info("Insufficient data for risk metrics")
            return
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Sharpe Ratio", f"{risk_metrics.get('sharpe_ratio', 0):.2f}")
            st.metric("Sortino Ratio", f"{risk_metrics.get('sortino_ratio', 0):.2f}")
        
        with col2:
            st.metric("Max Drawdown", f"{risk_metrics.get('max_drawdown', 0):.2%}")
            st.metric("Volatility", f"{risk_metrics.get('volatility', 0):.2%}")
        
        with col3:
            st.metric("VaR (95%)", f"{risk_metrics.get('var_95', 0):.2%}")
            st.metric("CVaR (95%)", f"{risk_metrics.get('cvar_95', 0):.2%}")
        
        with col4:
            st.metric("Calmar Ratio", f"{risk_metrics.get('calmar_ratio', 0):.2f}")
            st.metric("Mean Return", f"{risk_metrics.get('mean_return', 0):.2%}")

