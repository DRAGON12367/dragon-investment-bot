"""
Advanced Charts - Heat maps, performance charts, sector analysis, etc.
"""
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import numpy as np
from typing import Dict, List, Any, Optional


class AdvancedCharts:
    """Advanced visualization charts for trading dashboard."""
    
    def render_heatmap(
        self,
        data: pd.DataFrame,
        title: str = "Heat Map"
    ) -> go.Figure:
        """Render correlation or performance heat map."""
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale='RdYlGn',
            text=data.values,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Value")
        ))
        
        fig.update_layout(
            title=title,
            height=600,
            template='plotly_dark',
            xaxis_title="",
            yaxis_title=""
        )
        
        return fig
    
    def render_performance_chart(
        self,
        portfolio_value: List[float],
        benchmark: Optional[List[float]] = None,
        dates: Optional[List] = None
    ) -> go.Figure:
        """Render portfolio performance vs benchmark."""
        fig = go.Figure()
        
        x_data = dates if dates else list(range(len(portfolio_value)))
        
        fig.add_trace(go.Scatter(
            x=x_data,
            y=portfolio_value,
            mode='lines',
            name='Portfolio',
            line=dict(color='#00ff00', width=2)
        ))
        
        if benchmark:
            fig.add_trace(go.Scatter(
                x=x_data,
                y=benchmark,
                mode='lines',
                name='Benchmark',
                line=dict(color='#ff0000', width=2, dash='dash')
            ))
        
        fig.update_layout(
            title='Portfolio Performance',
            xaxis_title='Time',
            yaxis_title='Value ($)',
            height=400,
            template='plotly_dark',
            hovermode='x unified'
        )
        
        return fig
    
    def render_sector_performance(
        self,
        sector_data: Dict[str, float]
    ) -> go.Figure:
        """Render sector performance bar chart."""
        sectors = list(sector_data.keys())
        performance = list(sector_data.values())
        
        colors = ['green' if p > 0 else 'red' for p in performance]
        
        fig = go.Figure(data=go.Bar(
            x=sectors,
            y=performance,
            marker_color=colors,
            text=[f'{p:.2f}%' for p in performance],
            textposition='outside'
        ))
        
        fig.update_layout(
            title='Sector Performance',
            xaxis_title='Sector',
            yaxis_title='Performance (%)',
            height=400,
            template='plotly_dark',
            showlegend=False
        )
        
        return fig
    
    def render_risk_return_scatter(
        self,
        assets: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Render risk-return scatter plot."""
        symbols = []
        returns = []
        risks = []
        
        for symbol, data in assets.items():
            symbols.append(symbol)
            returns.append(data.get('return', 0))
            risks.append(data.get('risk', 0))
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=risks,
            y=returns,
            mode='markers+text',
            text=symbols,
            textposition='top center',
            marker=dict(
                size=10,
                color=returns,
                colorscale='RdYlGn',
                showscale=True,
                colorbar=dict(title="Return")
            )
        ))
        
        fig.update_layout(
            title='Risk-Return Analysis',
            xaxis_title='Risk (Volatility)',
            yaxis_title='Return (%)',
            height=500,
            template='plotly_dark'
        )
        
        return fig
    
    def render_momentum_heatmap(
        self,
        momentum_data: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """Render momentum heat map across assets."""
        symbols = list(momentum_data.keys())
        timeframes = ['5d', '10d', '20d', '50d']
        
        z_data = []
        for symbol in symbols:
            row = []
            data = momentum_data[symbol]
            for tf in timeframes:
                row.append(data.get(f'momentum_{tf}', 0))
            z_data.append(row)
        
        fig = go.Figure(data=go.Heatmap(
            z=z_data,
            x=timeframes,
            y=symbols,
            colorscale='RdYlGn',
            text=z_data,
            texttemplate='%{text:.2f}',
            textfont={"size": 10},
            colorbar=dict(title="Momentum")
        ))
        
        fig.update_layout(
            title='Momentum Heat Map',
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def render_volatility_regime_chart(
        self,
        volatility_history: pd.Series,
        regime_thresholds: Dict[str, float]
    ) -> go.Figure:
        """Render volatility regime chart."""
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=volatility_history.index,
            y=volatility_history.values,
            mode='lines',
            name='Volatility',
            line=dict(color='blue', width=2),
            fill='tozeroy'
        ))
        
        # Add regime thresholds
        high_threshold = regime_thresholds.get('high', 0.3)
        low_threshold = regime_thresholds.get('low', 0.15)
        
        fig.add_hline(
            y=high_threshold,
            line_dash="dash",
            line_color="red",
            annotation_text="High Volatility"
        )
        fig.add_hline(
            y=low_threshold,
            line_dash="dash",
            line_color="green",
            annotation_text="Low Volatility"
        )
        
        fig.update_layout(
            title='Volatility Regime',
            xaxis_title='Time',
            yaxis_title='Volatility',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def render_market_breadth(
        self,
        advancing: int,
        declining: int,
        unchanged: int
    ) -> go.Figure:
        """Render market breadth indicator."""
        labels = ['Advancing', 'Declining', 'Unchanged']
        values = [advancing, declining, unchanged]
        colors = ['green', 'red', 'gray']
        
        fig = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=0.4,
            marker_colors=colors
        )])
        
        fig.update_layout(
            title='Market Breadth',
            height=400,
            template='plotly_dark'
        )
        
        return fig
    
    def render_volume_analysis(
        self,
        df: pd.DataFrame,
        symbol: str
    ) -> go.Figure:
        """Render volume analysis chart."""
        fig = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            row_heights=[0.7, 0.3],
            subplot_titles=(f'{symbol} - Price & Volume', 'Volume Profile')
        )
        
        # Price
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=df['close'],
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Volume bars
        colors = ['green' if df['close'].iloc[i] >= df['open'].iloc[i] else 'red' 
                 for i in range(len(df))]
        
        fig.add_trace(
            go.Bar(
                x=df.index,
                y=df['volume'],
                name='Volume',
                marker_color=colors
            ),
            row=2, col=1
        )
        
        # Volume moving average
        if len(df) > 20:
            volume_ma = df['volume'].rolling(20).mean()
            fig.add_trace(
                go.Scatter(
                    x=df.index,
                    y=volume_ma,
                    mode='lines',
                    name='Volume MA',
                    line=dict(color='yellow', width=1, dash='dash')
                ),
                row=2, col=1
            )
        
        fig.update_layout(
            height=600,
            template='plotly_dark',
            showlegend=True
        )
        
        fig.update_xaxes(title_text="Time", row=2, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Volume", row=2, col=1)
        
        return fig

