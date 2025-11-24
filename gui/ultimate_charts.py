"""
ULTIMATE CHARTS - Most advanced trading visualizations
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class UltimateCharts:
    """Ultimate charting system with institutional-grade visualizations."""
    
    def __init__(self):
        """Initialize ultimate charts."""
        pass
    
    def market_leaderboard(
        self,
        market_data: Dict[str, Any],
        top_n: int = 30
    ) -> go.Figure:
        """
        Create interactive leaderboard showing top performers.
        
        Args:
            market_data: Current market data
            top_n: Number of top assets to show
            
        Returns:
            Plotly figure
        """
        # Calculate performance metrics
        leaderboard = []
        for symbol, data in list(market_data.items())[:top_n]:
            change = data.get('change_percent', 0)
            volume = data.get('volume', 0)
            price = data.get('price', 0)
            
            # Performance score
            score = (change * 0.5) + (np.log10(volume / 1000000 + 1) * 0.3) if volume > 0 else change
            
            leaderboard.append({
                'Symbol': symbol,
                'Change %': change,
                'Volume': volume / 1000000,  # In millions
                'Price': price,
                'Score': score,
                'Type': data.get('asset_type', 'unknown')
            })
        
        df = pd.DataFrame(leaderboard)
        df = df.sort_values('Score', ascending=False)
        
        # Color by asset type
        colors = ['#1f77b4' if t == 'stock' else '#ff7f0e' for t in df['Type']]
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Change %'],
            marker_color=colors,
            text=[f"{c:.2f}%" for c in df['Change %']],
            textposition='outside',
            name='24h Change',
            hovertemplate='<b>%{x}</b><br>' +
                        'Change: %{y:.2f}%<br>' +
                        'Volume: %{customdata:.1f}M<extra></extra>',
            customdata=df['Volume']
        ))
        
        fig.update_layout(
            title='⭐ Market Leaderboard - Top Performers',
            xaxis_title='Symbol',
            yaxis_title='24h Change (%)',
            height=600,
            showlegend=False,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def sector_heatmap(
        self,
        market_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Create sector heatmap showing performance by category.
        
        Args:
            market_data: Current market data
            
        Returns:
            Plotly figure
        """
        # Categorize assets
        sectors = {
            'Technology': [],
            'Finance': [],
            'Healthcare': [],
            'Consumer': [],
            'Energy': [],
            'Crypto': []
        }
        
        tech_stocks = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'NFLX', 'AMD']
        finance_stocks = ['JPM', 'BAC', 'WFC', 'GS', 'V', 'MA']
        healthcare_stocks = ['JNJ', 'UNH', 'PFE', 'ABT']
        consumer_stocks = ['WMT', 'HD', 'MCD', 'SBUX', 'NKE']
        energy_stocks = ['XOM', 'CVX', 'COP']
        
        for symbol, data in market_data.items():
            change = data.get('change_percent', 0)
            asset_type = data.get('asset_type', 'unknown')
            
            if asset_type == 'crypto':
                sectors['Crypto'].append(change)
            elif symbol in tech_stocks:
                sectors['Technology'].append(change)
            elif symbol in finance_stocks:
                sectors['Finance'].append(change)
            elif symbol in healthcare_stocks:
                sectors['Healthcare'].append(change)
            elif symbol in consumer_stocks:
                sectors['Consumer'].append(change)
            elif symbol in energy_stocks:
                sectors['Energy'].append(change)
            else:
                sectors['Technology'].append(change)  # Default
        
        # Calculate average change per sector
        sector_avg = {sector: np.mean(changes) if changes else 0 
                     for sector, changes in sectors.items()}
        
        # Create heatmap data
        sectors_list = list(sector_avg.keys())
        values = [sector_avg[s] for s in sectors_list]
        
        fig = go.Figure(data=go.Heatmap(
            z=[values],
            x=sectors_list,
            y=['Performance'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Avg Change %"),
            text=[[f"{v:.2f}%" for v in values]],
            texttemplate='%{text}',
            textfont={"size": 14}
        ))
        
        fig.update_layout(
            title='⭐ Sector Performance Heatmap',
            height=200,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return fig
    
    def price_action_analysis(
        self,
        price_history: Dict[str, List],
        symbol: str
    ) -> go.Figure:
        """
        Create comprehensive price action analysis chart.
        
        Args:
            price_history: Historical price data
            symbol: Symbol to analyze
            
        Returns:
            Plotly figure
        """
        if symbol not in price_history or len(price_history[symbol]) < 20:
            return self._empty_chart("Insufficient data")
        
        history = price_history[symbol]
        df = pd.DataFrame(history)
        df = df.sort_values('timestamp')
        
        if 'price' not in df.columns:
            return self._empty_chart("No price data")
        
        prices = df['price'].values
        
        # Calculate price action metrics
        returns = np.diff(prices) / prices[:-1]
        volatility = np.std(returns) * np.sqrt(252)  # Annualized
        
        # Create subplots
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('Price Action', 'Returns Distribution', 'Volatility'),
            vertical_spacing=0.1,
            row_heights=[0.5, 0.25, 0.25]
        )
        
        # Price action
        fig.add_trace(
            go.Scatter(
                x=df.index,
                y=prices,
                mode='lines',
                name='Price',
                line=dict(color='blue', width=2)
            ),
            row=1, col=1
        )
        
        # Returns distribution
        fig.add_trace(
            go.Histogram(
                x=returns * 100,
                nbinsx=30,
                name='Returns',
                marker_color='green'
            ),
            row=2, col=1
        )
        
        # Rolling volatility
        window = min(20, len(returns))
        rolling_vol = pd.Series(returns).rolling(window).std() * np.sqrt(252) * 100
        
        fig.add_trace(
            go.Scatter(
                x=df.index[window:],
                y=rolling_vol[window:],
                mode='lines',
                name='Volatility',
                line=dict(color='red', width=2)
            ),
            row=3, col=1
        )
        
        fig.update_layout(
            title=f'⭐ Price Action Analysis: {symbol}',
            height=800,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Time", row=3, col=1)
        fig.update_yaxes(title_text="Price", row=1, col=1)
        fig.update_yaxes(title_text="Frequency", row=2, col=1)
        fig.update_yaxes(title_text="Volatility %", row=3, col=1)
        
        return fig
    
    def market_correlation_matrix(
        self,
        market_data: Dict[str, Any],
        price_history: Optional[Dict[str, List]] = None
    ) -> go.Figure:
        """
        Create correlation matrix showing asset relationships.
        
        Args:
            market_data: Current market data
            price_history: Historical price data
            
        Returns:
            Plotly figure
        """
        if not price_history:
            return self._empty_chart("No price history available")
        
        # Get symbols with sufficient history
        symbols = []
        price_series = {}
        
        for symbol in list(market_data.keys())[:30]:  # Limit to 30 for performance
            if symbol in price_history and len(price_history[symbol]) > 20:
                history = price_history[symbol]
                prices = [p['price'] for p in history[-30:]]  # Last 30 data points
                if len(prices) == 30:
                    symbols.append(symbol)
                    price_series[symbol] = prices
        
        if len(symbols) < 5:
            return self._empty_chart("Insufficient data for correlation")
        
        # Calculate correlation matrix
        df = pd.DataFrame(price_series)
        corr_matrix = df.corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.index,
            colorscale='RdBu',
            zmid=0,
            showscale=True,
            colorbar=dict(title="Correlation"),
            text=corr_matrix.values.round(2),
            texttemplate='%{text}',
            textfont={"size": 8}
        ))
        
        fig.update_layout(
            title='⭐ Market Correlation Matrix',
            height=700,
            xaxis_title='Symbol',
            yaxis_title='Symbol',
            margin=dict(l=100, r=50, t=50, b=100)
        )
        
        return fig
    
    def volume_profile_advanced(
        self,
        price_history: Dict[str, List],
        symbol: str
    ) -> go.Figure:
        """
        Create advanced volume profile with price levels.
        
        Args:
            price_history: Historical price data
            symbol: Symbol to analyze
            
        Returns:
            Plotly figure
        """
        if symbol not in price_history or len(price_history[symbol]) < 20:
            return self._empty_chart("Insufficient data")
        
        history = price_history[symbol]
        df = pd.DataFrame(history)
        df = df.sort_values('timestamp')
        
        prices = df.get('price', df.get('close', []))
        volumes = df.get('volume', [0] * len(df))
        
        if len(prices) == 0:
            return self._empty_chart("No price data")
        
        # Create price bins
        price_min = min(prices)
        price_max = max(prices)
        bins = np.linspace(price_min, price_max, 20)
        
        # Calculate volume per price level
        volume_profile = []
        for i in range(len(bins) - 1):
            mask = (prices >= bins[i]) & (prices < bins[i+1])
            volume_at_level = sum(volumes[mask]) if hasattr(volumes, '__getitem__') else 0
            volume_profile.append({
                'price_level': (bins[i] + bins[i+1]) / 2,
                'volume': volume_at_level
            })
        
        vp_df = pd.DataFrame(volume_profile)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=vp_df['volume'],
            y=vp_df['price_level'],
            orientation='h',
            marker_color='lightblue',
            name='Volume Profile'
        ))
        
        # Add current price line
        current_price = prices.iloc[-1] if hasattr(prices, 'iloc') else prices[-1]
        fig.add_vline(
            x=max(vp_df['volume']) * 0.1,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Current: ${current_price:.2f}"
        )
        
        fig.update_layout(
            title=f'⭐ Advanced Volume Profile: {symbol}',
            xaxis_title='Volume',
            yaxis_title='Price Level',
            height=600
        )
        
        return fig
    
    def momentum_comparison(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create momentum comparison chart across assets.
        
        Args:
            predictions: List of predictions
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Extract momentum data
        momentum_data = []
        for pred in predictions[:20]:
            symbol = pred.get('symbol', '')
            confidence = pred.get('confidence', 0) * 100
            profit = pred.get('profit_potential', 0) * 100
            risk = pred.get('risk_score', 0) * 100
            
            # Momentum score
            momentum = (confidence * 0.4) + (profit * 0.4) - (risk * 0.2)
            
            momentum_data.append({
                'Symbol': symbol,
                'Momentum': momentum,
                'Confidence': confidence,
                'Profit': profit,
                'Risk': risk
            })
        
        df = pd.DataFrame(momentum_data)
        df = df.sort_values('Momentum', ascending=False)
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Momentum Score', 'Confidence vs Profit'),
            vertical_spacing=0.15
        )
        
        # Momentum bars
        fig.add_trace(
            go.Bar(
                x=df['Symbol'],
                y=df['Momentum'],
                marker_color='gold',
                name='Momentum'
            ),
            row=1, col=1
        )
        
        # Scatter: Confidence vs Profit
        fig.add_trace(
            go.Scatter(
                x=df['Confidence'],
                y=df['Profit'],
                mode='markers+text',
                text=df['Symbol'],
                textposition='top center',
                marker=dict(
                    size=10,
                    color=df['Risk'],
                    colorscale='RdYlGn',
                    showscale=True,
                    colorbar=dict(title="Risk", x=1.15)
                ),
                name='Confidence vs Profit'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='⭐ Momentum Comparison Analysis',
            height=700,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Symbol", row=1, col=1)
        fig.update_yaxes(title_text="Momentum Score", row=1, col=1)
        fig.update_xaxes(title_text="Confidence %", row=2, col=1)
        fig.update_yaxes(title_text="Profit Potential %", row=2, col=1)
        
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

