"""
Ultra Advanced Charts - Next-level visualizations for trading analysis
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class UltraAdvancedCharts:
    """Ultra-advanced charting system with professional trading visualizations."""
    
    def __init__(self):
        """Initialize ultra advanced charts."""
        pass
    
    def multi_timeframe_analysis(
        self,
        price_data: pd.DataFrame,
        symbol: str
    ) -> go.Figure:
        """
        Create multi-timeframe analysis chart showing different timeframes.
        
        Args:
            price_data: DataFrame with price history
            symbol: Symbol name
            
        Returns:
            Plotly figure with subplots
        """
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=('1-Minute View', '5-Minute View', '15-Minute View'),
            vertical_spacing=0.08,
            row_heights=[0.4, 0.3, 0.3]
        )
        
        # Ensure we have timestamp index
        if 'timestamp' in price_data.columns:
            price_data = price_data.set_index('timestamp')
        elif price_data.index.name != 'timestamp':
            # Create timestamp index if missing
            price_data.index = pd.date_range(end=pd.Timestamp.now(), periods=len(price_data), freq='1min')
        
        # Ensure we have OHLC data
        if 'open' not in price_data.columns:
            price_data['open'] = price_data['price']
        if 'high' not in price_data.columns:
            price_data['high'] = price_data['price'] * 1.01
        if 'low' not in price_data.columns:
            price_data['low'] = price_data['price'] * 0.99
        if 'close' not in price_data.columns:
            price_data['close'] = price_data['price']
        
        # 1-minute view (raw data)
        recent_data = price_data.iloc[-min(100, len(price_data)):]
        fig.add_trace(
            go.Candlestick(
                x=recent_data.index,
                open=recent_data['open'],
                high=recent_data['high'],
                low=recent_data['low'],
                close=recent_data['close'],
                name='1m'
            ),
            row=1, col=1
        )
        
        # 5-minute view (resampled)
        if len(price_data) > 20:
            try:
                price_5m = price_data.resample('5T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                if len(price_5m) > 0:
                    recent_5m = price_5m.iloc[-min(50, len(price_5m)):]
                    fig.add_trace(
                        go.Candlestick(
                            x=recent_5m.index,
                            open=recent_5m['open'],
                            high=recent_5m['high'],
                            low=recent_5m['low'],
                            close=recent_5m['close'],
                            name='5m'
                        ),
                        row=2, col=1
                    )
            except:
                pass
        
        # 15-minute view (resampled)
        if len(price_data) > 60:
            try:
                price_15m = price_data.resample('15T').agg({
                    'open': 'first',
                    'high': 'max',
                    'low': 'min',
                    'close': 'last'
                }).dropna()
                if len(price_15m) > 0:
                    recent_15m = price_15m.iloc[-min(30, len(price_15m)):]
                    fig.add_trace(
                        go.Candlestick(
                            x=recent_15m.index,
                            open=recent_15m['open'],
                            high=recent_15m['high'],
                            low=recent_15m['low'],
                            close=recent_15m['close'],
                            name='15m'
                        ),
                        row=3, col=1
                    )
            except:
                pass
        
        fig.update_layout(
            title=f'⭐ Multi-Timeframe Analysis: {symbol}',
            height=800,
            showlegend=False,
            xaxis_rangeslider_visible=False
        )
        
        return fig
    
    def order_flow_heatmap(
        self,
        market_data: Dict[str, Any],
        top_n: int = 20
    ) -> go.Figure:
        """
        Create order flow heatmap showing buy/sell pressure.
        
        Args:
            market_data: Dictionary of market data
            top_n: Number of top assets to show
            
        Returns:
            Plotly figure
        """
        # Calculate buy/sell pressure
        data = []
        for symbol, info in list(market_data.items())[:top_n]:
            change = info.get('change_percent', 0)
            volume = info.get('volume', 0)
            
            # Buy pressure (positive change + high volume)
            buy_pressure = max(0, change) * (volume / 1000000) if volume > 0 else 0
            
            # Sell pressure (negative change + high volume)
            sell_pressure = abs(min(0, change)) * (volume / 1000000) if volume > 0 else 0
            
            data.append({
                'Symbol': symbol,
                'Buy Pressure': buy_pressure,
                'Sell Pressure': sell_pressure,
                'Net Flow': buy_pressure - sell_pressure
            })
        
        df = pd.DataFrame(data)
        df = df.sort_values('Net Flow', ascending=False)
        
        fig = go.Figure()
        
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Buy Pressure'],
            name='Buy Pressure',
            marker_color='green',
            opacity=0.7
        ))
        
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=-df['Sell Pressure'],
            name='Sell Pressure',
            marker_color='red',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='⭐ Order Flow Heatmap - Buy vs Sell Pressure',
            xaxis_title='Symbol',
            yaxis_title='Pressure',
            barmode='overlay',
            height=500,
            hovermode='x unified'
        )
        
        return fig
    
    def momentum_matrix(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create momentum matrix showing price momentum vs prediction confidence.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        symbols = [p.get('symbol', '') for p in predictions]
        confidences = [p.get('confidence', 0) * 100 for p in predictions]
        profit_potentials = [p.get('profit_potential', 0) * 100 for p in predictions]
        ranks = [p.get('rank', 0) for p in predictions]
        
        # Create matrix
        fig = go.Figure(data=go.Heatmap(
            z=[ranks],
            x=symbols,
            y=['Momentum Score'],
            colorscale='Viridis',
            showscale=True,
            colorbar=dict(title="Rank"),
            text=[[f"{s}<br>Conf: {c:.1f}%<br>Profit: +{p:.1f}%" 
                   for s, c, p in zip(symbols, confidences, profit_potentials)]],
            texttemplate='%{text}',
            textfont={"size": 9}
        ))
        
        fig.update_layout(
            title='⭐ Momentum Matrix - Best Opportunities',
            xaxis_title='Symbol',
            yaxis_title='',
            height=200,
            margin=dict(l=50, r=50, t=50, b=150)
        )
        
        return fig
    
    def ai_confidence_radar(
        self,
        prediction: Dict[str, Any]
    ) -> go.Figure:
        """
        Create advanced radar chart with AI confidence breakdown.
        
        Args:
            prediction: Single prediction dictionary
            
        Returns:
            Plotly figure
        """
        individual = prediction.get('individual_predictions', {})
        
        categories = [k.replace('_', ' ').title() for k in individual.keys()]
        values = [v * 100 for v in individual.values()]
        
        # Add overall metrics
        categories.append('Overall Confidence')
        values.append(prediction.get('confidence', 0) * 100)
        
        categories.append('Profit Potential')
        values.append(prediction.get('profit_potential', 0) * 100)
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='AI Score',
            line_color='gold',
            marker=dict(size=8, color='gold')
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f'⭐ AI Confidence Radar: {prediction.get("symbol", "Unknown")}',
            height=500
        )
        
        return fig
    
    def profit_guarantee_chart(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create chart showing profit guarantee scores.
        
        Args:
            predictions: List of predictions with profit guarantee data
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Filter to best buys only
        best_buys = [p for p in predictions if p.get('best_buy', False)]
        
        if not best_buys:
            return self._empty_chart("No best buy predictions")
        
        symbols = [p.get('symbol', '') for p in best_buys[:15]]
        ranks = [p.get('rank', 0) for p in best_buys[:15]]
        profits = [p.get('profit_potential', 0) * 100 for p in best_buys[:15]]
        risks = [p.get('risk_score', 0) * 100 for p in best_buys[:15]]
        
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('Profit Potential', 'Risk Score'),
            vertical_spacing=0.15
        )
        
        # Profit potential bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=profits,
                name='Profit %',
                marker_color='green',
                text=[f"+{p:.1f}%" for p in profits],
                textposition='outside'
            ),
            row=1, col=1
        )
        
        # Risk score bars
        fig.add_trace(
            go.Bar(
                x=symbols,
                y=risks,
                name='Risk %',
                marker_color='red',
                text=[f"{r:.1f}%" for r in risks],
                textposition='outside'
            ),
            row=2, col=1
        )
        
        fig.update_layout(
            title='⭐ Profit Guarantee Analysis',
            height=600,
            showlegend=False
        )
        
        fig.update_xaxes(title_text="Symbol", row=2, col=1)
        fig.update_yaxes(title_text="Profit %", row=1, col=1)
        fig.update_yaxes(title_text="Risk %", row=2, col=1)
        
        return fig
    
    def smart_money_flow(
        self,
        market_data: Dict[str, Any],
        price_history: Optional[Dict[str, List]] = None
    ) -> go.Figure:
        """
        Create smart money flow chart showing institutional activity.
        
        Args:
            market_data: Current market data
            price_history: Historical price data
            
        Returns:
            Plotly figure
        """
        # Calculate smart money indicators
        flow_data = []
        for symbol, data in list(market_data.items())[:20]:
            price = data.get('price', 0)
            volume = data.get('volume', 0)
            change = data.get('change_percent', 0)
            
            # Smart money flow = price change * volume (weighted)
            smart_flow = (change / 100) * (volume / 1000000) if volume > 0 else 0
            
            flow_data.append({
                'Symbol': symbol,
                'Smart Money Flow': smart_flow,
                'Volume': volume / 1000000,  # In millions
                'Price Change': change
            })
        
        df = pd.DataFrame(flow_data)
        df = df.sort_values('Smart Money Flow', ascending=False)
        
        fig = go.Figure()
        
        # Color by flow direction
        colors = ['green' if x > 0 else 'red' for x in df['Smart Money Flow']]
        
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Smart Money Flow'],
            marker_color=colors,
            text=[f"{f:.2f}" for f in df['Smart Money Flow']],
            textposition='outside',
            name='Smart Money Flow',
            hovertemplate='<b>%{x}</b><br>' +
                        'Flow: %{y:.2f}<br>' +
                        'Volume: %{customdata:.1f}M<extra></extra>',
            customdata=df['Volume']
        ))
        
        fig.update_layout(
            title='⭐ Smart Money Flow - Institutional Activity',
            xaxis_title='Symbol',
            yaxis_title='Smart Money Flow',
            height=500,
            showlegend=False
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

