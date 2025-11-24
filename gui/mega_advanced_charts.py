"""
MEGA ADVANCED CHARTS - Ultra-professional trading visualizations
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class MegaAdvancedCharts:
    """Mega-advanced charting system with institutional-grade visualizations."""
    
    def __init__(self):
        """Initialize mega advanced charts."""
        pass
    
    def market_depth_chart(
        self,
        market_data: Dict[str, Any],
        top_n: int = 15
    ) -> go.Figure:
        """
        Create market depth chart showing bid/ask pressure.
        
        Args:
            market_data: Current market data
            top_n: Number of top assets
            
        Returns:
            Plotly figure
        """
        # Calculate market depth indicators
        depth_data = []
        for symbol, data in list(market_data.items())[:top_n]:
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            change = data.get('change_percent', 0)
            
            # Bid depth (buying pressure)
            bid_depth = max(0, change) * (volume / 1000000) if volume > 0 else 0
            
            # Ask depth (selling pressure)
            ask_depth = abs(min(0, change)) * (volume / 1000000) if volume > 0 else 0
            
            depth_data.append({
                'Symbol': symbol,
                'Bid Depth': bid_depth,
                'Ask Depth': ask_depth,
                'Net Depth': bid_depth - ask_depth
            })
        
        df = pd.DataFrame(depth_data)
        df = df.sort_values('Net Depth', ascending=False)
        
        fig = go.Figure()
        
        # Bid depth (buying)
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Bid Depth'],
            name='Bid Depth (Buying)',
            marker_color='green',
            opacity=0.7
        ))
        
        # Ask depth (selling)
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=-df['Ask Depth'],
            name='Ask Depth (Selling)',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='⭐ Market Depth - Bid/Ask Pressure Analysis',
            xaxis_title='Symbol',
            yaxis_title='Market Depth',
            barmode='overlay',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def volatility_surface(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create 3D volatility surface showing risk across assets.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Plotly 3D figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        symbols = [p.get('symbol', '') for p in predictions[:20]]
        volatilities = [p.get('risk_score', 0.5) * 100 for p in predictions[:20]]
        profits = [p.get('profit_potential', 0) * 100 for p in predictions[:20]]
        ranks = [p.get('rank', 0) for p in predictions[:20]]
        
        fig = go.Figure(data=[go.Scatter3d(
            x=volatilities,
            y=profits,
            z=ranks,
            mode='markers+text',
            marker=dict(
                size=10,
                color=ranks,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Rank")
            ),
            text=symbols,
            textposition='middle center',
            hovertemplate='<b>%{text}</b><br>' +
                        'Volatility: %{x:.1f}%<br>' +
                        'Profit: %{y:.1f}%<br>' +
                        'Rank: %{z:.1f}<extra></extra>'
        )])
        
        fig.update_layout(
            title='⭐ Volatility Surface - 3D Risk/Reward Analysis',
            scene=dict(
                xaxis_title='Volatility (Risk)',
                yaxis_title='Profit Potential',
                zaxis_title='Overall Rank'
            ),
            height=600
        )
        
        return fig
    
    def correlation_network(
        self,
        market_data: Dict[str, Any],
        price_history: Optional[Dict[str, List]] = None
    ) -> go.Figure:
        """
        Create correlation network graph showing asset relationships.
        
        Args:
            market_data: Current market data
            price_history: Historical price data
            
        Returns:
            Plotly figure
        """
        # Calculate correlations
        symbols = list(market_data.keys())[:20]
        correlations = {}
        
        if price_history:
            for i, sym1 in enumerate(symbols):
                for sym2 in symbols[i+1:]:
                    if sym1 in price_history and sym2 in price_history:
                        hist1 = price_history[sym1]
                        hist2 = price_history[sym2]
                        
                        if len(hist1) > 10 and len(hist2) > 10:
                            prices1 = pd.Series([p['price'] for p in hist1[-20:]])
                            prices2 = pd.Series([p['price'] for p in hist2[-20:]])
                            
                            if len(prices1) == len(prices2):
                                corr = prices1.corr(prices2)
                                if not np.isnan(corr):
                                    correlations[f"{sym1}-{sym2}"] = corr
        
        if not correlations:
            return self._empty_chart("Insufficient data for correlation network")
        
        # Create network graph
        nodes = list(set([s for pair in correlations.keys() for s in pair.split('-')]))
        edges = [(pair.split('-')[0], pair.split('-')[1], abs(corr)) 
                 for pair, corr in correlations.items() if abs(corr) > 0.3]
        
        if not edges:
            return self._empty_chart("No strong correlations found")
        
        # Create edge traces
        edge_x = []
        edge_y = []
        for edge in edges:
            x0, y0 = hash(edge[0]) % 100, hash(edge[0]) % 100
            x1, y1 = hash(edge[1]) % 100, hash(edge[1]) % 100
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])
        
        # Create node traces
        node_x = [hash(node) % 100 for node in nodes]
        node_y = [hash(node) % 100 for node in nodes]
        
        fig = go.Figure()
        
        # Edges
        fig.add_trace(go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=0.5, color='#888'),
            hoverinfo='none',
            mode='lines',
            showlegend=False
        ))
        
        # Nodes
        fig.add_trace(go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=20, color='lightblue'),
            text=nodes,
            textposition='middle center',
            hoverinfo='text',
            name='Assets'
        ))
        
        fig.update_layout(
            title='⭐ Correlation Network - Asset Relationships',
            showlegend=False,
            hovermode='closest',
            height=500,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False)
        )
        
        return fig
    
    def performance_waterfall(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create waterfall chart showing cumulative performance potential.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Sort by rank
        sorted_preds = sorted(predictions, key=lambda x: x.get('rank', 0), reverse=True)[:15]
        
        symbols = [p.get('symbol', '') for p in sorted_preds]
        profits = [p.get('profit_potential', 0) * 100 for p in sorted_preds]
        cumulative = np.cumsum(profits)
        
        fig = go.Figure()
        
        # Waterfall bars
        fig.add_trace(go.Bar(
            x=symbols,
            y=profits,
            marker_color=['green' if p > 0 else 'red' for p in profits],
            text=[f"+{p:.1f}%" if p > 0 else f"{p:.1f}%" for p in profits],
            textposition='outside',
            name='Profit Potential'
        ))
        
        # Cumulative line
        fig.add_trace(go.Scatter(
            x=symbols,
            y=cumulative,
            mode='lines+markers',
            name='Cumulative',
            line=dict(color='gold', width=3),
            marker=dict(size=8, color='gold')
        ))
        
        fig.update_layout(
            title='⭐ Performance Waterfall - Cumulative Profit Potential',
            xaxis_title='Symbol',
            yaxis_title='Profit Potential (%)',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def prediction_heatmap_matrix(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create advanced heatmap matrix showing all prediction dimensions.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Get top predictions
        top_preds = sorted(predictions, key=lambda x: x.get('rank', 0), reverse=True)[:15]
        
        symbols = [p.get('symbol', '') for p in top_preds]
        
        # Create matrix data
        matrix_data = []
        for pred in top_preds:
            row = [
                pred.get('confidence', 0) * 100,
                pred.get('profit_potential', 0) * 100,
                pred.get('risk_score', 0) * 100,
                pred.get('rank', 0),
                pred.get('agreement', 0) * 100 if pred.get('agreement') else 0
            ]
            matrix_data.append(row)
        
        matrix = np.array(matrix_data).T
        
        fig = go.Figure(data=go.Heatmap(
            z=matrix,
            x=symbols,
            y=['Confidence', 'Profit %', 'Risk %', 'Rank', 'Agreement %'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Score"),
            text=[[f"{val:.1f}" for val in row] for row in matrix],
            texttemplate='%{text}',
            textfont={"size": 9}
        ))
        
        fig.update_layout(
            title='⭐ Prediction Matrix - Multi-Dimensional Analysis',
            xaxis_title='Symbol',
            yaxis_title='Metric',
            height=400,
            margin=dict(l=100, r=50, t=50, b=150)
        )
        
        return fig
    
    def trend_velocity_chart(
        self,
        price_history: Dict[str, List],
        symbols: List[str]
    ) -> go.Figure:
        """
        Create trend velocity chart showing price acceleration.
        
        Args:
            price_history: Historical price data
            symbols: List of symbols to analyze
            
        Returns:
            Plotly figure
        """
        velocity_data = []
        
        for symbol in symbols[:15]:
            if symbol in price_history:
                history = price_history[symbol]
                if len(history) > 10:
                    prices = pd.Series([p['price'] for p in history[-20:]])
                    
                    # Calculate velocity (first derivative)
                    velocity = prices.diff().mean()
                    
                    # Calculate acceleration (second derivative)
                    acceleration = prices.diff().diff().mean()
                    
                    velocity_data.append({
                        'Symbol': symbol,
                        'Velocity': velocity,
                        'Acceleration': acceleration,
                        'Trend Strength': abs(velocity) + abs(acceleration)
                    })
        
        if not velocity_data:
            return self._empty_chart("Insufficient data for trend velocity")
        
        df = pd.DataFrame(velocity_data)
        df = df.sort_values('Trend Strength', ascending=False)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Price Velocity', 'Price Acceleration'),
            vertical_spacing=0.15
        )
        
        # Velocity
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=df['Velocity'],
                marker_color=['green' if v > 0 else 'red' for v in df['Velocity']],
                name='Velocity'
            ),
            row=1, col=1
        )
        
        # Acceleration
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=df['Acceleration'],
                marker_color=['green' if a > 0 else 'red' for a in df['Acceleration']],
                name='Acceleration'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='⭐ Trend Velocity - Price Movement Speed',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Velocity", row=1, col=1)
        fig.update_yaxes(title_text="Acceleration", row=2, col=1)
        
        return fig
    
    def support_resistance_heatmap(
        self,
        price_history: Dict[str, List],
        symbols: List[str]
    ) -> go.Figure:
        """
        Create heatmap showing support/resistance levels.
        
        Args:
            price_history: Historical price data
            symbols: List of symbols
            
        Returns:
            Plotly figure
        """
        levels_data = []
        
        for symbol in symbols[:15]:
            if symbol in price_history:
                history = price_history[symbol]
                if len(history) > 20:
                    prices = pd.Series([p['price'] for p in history[-50:]])
                    
                    # Simple support/resistance detection
                    current_price = prices.iloc[-1]
                    price_range = prices.max() - prices.min()
                    
                    # Support level (lower bound)
                    support = prices.min()
                    support_strength = (current_price - support) / price_range if price_range > 0 else 0
                    
                    # Resistance level (upper bound)
                    resistance = prices.max()
                    resistance_strength = (resistance - current_price) / price_range if price_range > 0 else 0
                    
                    levels_data.append({
                        'Symbol': symbol,
                        'Current Price': current_price,
                        'Support': support,
                        'Resistance': resistance,
                        'Support Distance': support_strength * 100,
                        'Resistance Distance': resistance_strength * 100
                    })
        
        if not levels_data:
            return self._empty_chart("Insufficient data for support/resistance")
        
        df = pd.DataFrame(levels_data)
        
        fig = go.Figure()
        
        # Support distance
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Support Distance'],
            name='Distance to Support',
            marker_color='green',
            opacity=0.7
        ))
        
        # Resistance distance
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Resistance Distance'],
            name='Distance to Resistance',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='⭐ Support/Resistance Heatmap - Key Price Levels',
            xaxis_title='Symbol',
            yaxis_title='Distance (%)',
            barmode='group',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def _empty_chart(self, message: str) -> go.Figure:
        """Create an empty chart with a message."""
        fig = go.Figure()
        fig.add_annotation(
            x=0.5,
            y=0.5,
            text=message,
            showarrow=False,
            font=dict(size=16, color="gray")
        )
        fig.update_layout(
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            height=300
        )
        return fig

