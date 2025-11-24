"""
QUANTUM CHARTS - Advanced quantum-inspired visualizations
"""
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class QuantumCharts:
    """Quantum-inspired charting system with advanced visualizations."""
    
    def __init__(self):
        """Initialize quantum charts."""
        pass
    
    def quantum_probability_cloud(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create quantum probability cloud showing prediction uncertainty.
        
        Args:
            predictions: List of predictions with confidence scores
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Extract data
        symbols = [p.get('symbol', '') for p in predictions[:20]]
        confidences = [p.get('confidence', 0) * 100 for p in predictions[:20]]
        profits = [p.get('profit_potential', 0) * 100 for p in predictions[:20]]
        
        # Quantum uncertainty (inverse of confidence)
        uncertainties = [100 - c for c in confidences]
        
        fig = go.Figure()
        
        # Quantum cloud scatter
        fig.add_trace(go.Scatter(
            x=confidences,
            y=profits,
            mode='markers+text',
            text=symbols,
            textposition='top center',
            marker=dict(
                size=[max(10, u * 0.5) for u in uncertainties],  # Size = uncertainty
                color=uncertainties,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title="Quantum Uncertainty %"),
                opacity=0.7,
                line=dict(width=2, color='white')
            ),
            name='Quantum Probability Cloud',
            hovertemplate='<b>%{text}</b><br>' +
                        'Confidence: %{x:.1f}%<br>' +
                        'Profit: %{y:.1f}%<br>' +
                        'Uncertainty: %{marker.color:.1f}%<extra></extra>'
        ))
        
        fig.update_layout(
            title='⭐ Quantum Probability Cloud - Prediction Uncertainty',
            xaxis_title='Confidence %',
            yaxis_title='Profit Potential %',
            height=600,
            template='plotly_dark'
        )
        
        return fig
    
    def quantum_superposition_chart(
        self,
        market_data: Dict[str, Any]
    ) -> go.Figure:
        """
        Create quantum superposition chart showing multiple market states.
        
        Args:
            market_data: Current market data
            
        Returns:
            Plotly figure
        """
        if not market_data:
            return self._empty_chart("No market data available")
        
        # Calculate quantum states (bullish, bearish, neutral)
        states = []
        for symbol, data in list(market_data.items())[:30]:
            change = data.get('change_percent', 0)
            volume = data.get('volume', 0)
            
            # Quantum state probabilities
            if change > 2:
                bull_prob = 0.8
                bear_prob = 0.1
                neutral_prob = 0.1
            elif change < -2:
                bull_prob = 0.1
                bear_prob = 0.8
                neutral_prob = 0.1
            else:
                bull_prob = 0.33
                bear_prob = 0.33
                neutral_prob = 0.34
            
            states.append({
                'Symbol': symbol,
                'Bullish': bull_prob * 100,
                'Bearish': bear_prob * 100,
                'Neutral': neutral_prob * 100,
                'Change': change
            })
        
        df = pd.DataFrame(states)
        df = df.sort_values('Change', ascending=False)
        
        fig = go.Figure()
        
        # Bullish probability
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Bullish'],
            name='Bullish Probability',
            marker_color='green',
            opacity=0.7
        ))
        
        # Bearish probability
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Bearish'],
            name='Bearish Probability',
            marker_color='red',
            opacity=0.7
        ))
        
        # Neutral probability
        fig.add_trace(go.Bar(
            x=df['Symbol'],
            y=df['Neutral'],
            name='Neutral Probability',
            marker_color='gray',
            opacity=0.7
        ))
        
        fig.update_layout(
            title='⭐ Quantum Superposition - Market State Probabilities',
            xaxis_title='Symbol',
            yaxis_title='Probability %',
            barmode='stack',
            height=600,
            xaxis={'tickangle': -45}
        )
        
        return fig
    
    def quantum_entanglement_network(
        self,
        correlations: Dict[str, Dict[str, float]]
    ) -> go.Figure:
        """
        Create quantum entanglement network showing asset correlations.
        
        Args:
            correlations: Dictionary of symbol -> {other_symbol: correlation}
            
        Returns:
            Plotly figure
        """
        if not correlations:
            return self._empty_chart("No correlation data available")
        
        # Create network graph data
        nodes = []
        edges = []
        
        symbols = list(correlations.keys())[:20]  # Limit for performance
        
        for i, symbol1 in enumerate(symbols):
            nodes.append({
                'id': symbol1,
                'label': symbol1,
                'size': 20
            })
            
            for symbol2, corr in correlations[symbol1].items():
                if symbol2 in symbols and abs(corr) > 0.7:  # Strong correlation
                    edges.append({
                        'source': symbol1,
                        'target': symbol2,
                        'value': abs(corr),
                        'color': 'green' if corr > 0 else 'red'
                    })
        
        # Create network visualization
        fig = go.Figure()
        
        # Add edges
        for edge in edges[:50]:  # Limit edges
            source_idx = symbols.index(edge['source'])
            target_idx = symbols.index(edge['target'])
            
            fig.add_trace(go.Scatter(
                x=[source_idx, target_idx],
                y=[source_idx, target_idx],
                mode='lines',
                line=dict(
                    width=edge['value'] * 5,
                    color=edge['color'],
                    opacity=0.3
                ),
                showlegend=False,
                hoverinfo='skip'
            ))
        
        # Add nodes
        fig.add_trace(go.Scatter(
            x=list(range(len(symbols))),
            y=list(range(len(symbols))),
            mode='markers+text',
            text=symbols,
            textposition='middle center',
            marker=dict(
                size=30,
                color='lightblue',
                line=dict(width=2, color='darkblue')
            ),
            name='Assets',
            hovertemplate='<b>%{text}</b><extra></extra>'
        ))
        
        fig.update_layout(
            title='⭐ Quantum Entanglement Network - Asset Correlations',
            height=700,
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False, zeroline=False),
            yaxis=dict(showgrid=False, showticklabels=False, zeroline=False)
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

