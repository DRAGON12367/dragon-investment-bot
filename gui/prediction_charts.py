"""
Advanced Prediction Charts - Visualizations for Best Buy Predictions
"""
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np
from typing import Dict, List, Any, Optional


class PredictionCharts:
    """Advanced charts for visualizing best buy predictions."""
    
    def __init__(self):
        """Initialize prediction charts."""
        pass
    
    def best_buy_heatmap(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create a heatmap showing best buy opportunities.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Prepare data
        symbols = [p.get('symbol', '') for p in predictions]
        confidences = [p.get('confidence', 0) * 100 for p in predictions]
        profit_potentials = [p.get('profit_potential', 0) * 100 for p in predictions]
        risk_scores = [p.get('risk_score', 0.5) * 100 for p in predictions]
        ranks = [p.get('rank', 0) for p in predictions]
        
        # Create matrix for heatmap
        data = []
        for i, pred in enumerate(predictions):
            data.append({
                'Symbol': pred.get('symbol', ''),
                'Confidence': confidences[i],
                'Profit Potential': profit_potentials[i],
                'Risk Score': risk_scores[i],
                'Rank': ranks[i],
                'Best Buy': 'YES' if pred.get('best_buy', False) else 'NO'
            })
        
        df = pd.DataFrame(data)
        
        # Create heatmap
        fig = go.Figure(data=go.Heatmap(
            z=[[p.get('rank', 0) for p in predictions]],
            x=symbols,
            y=['Best Buy Rank'],
            colorscale='RdYlGn',
            showscale=True,
            colorbar=dict(title="Rank Score"),
            text=[[f"{s}<br>Rank: {r:.1f}" for s, r in zip(symbols, ranks)]],
            texttemplate='%{text}',
            textfont={"size": 10}
        ))
        
        fig.update_layout(
            title="🔥 Best Buy Opportunities Heatmap",
            xaxis_title="Symbol",
            yaxis_title="",
            height=200,
            margin=dict(l=50, r=50, t=50, b=100)
        )
        
        return fig
    
    def prediction_confidence_chart(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create a bar chart showing prediction confidence scores.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Sort by confidence
        sorted_preds = sorted(predictions, key=lambda x: x.get('confidence', 0), reverse=True)
        
        symbols = [p.get('symbol', '') for p in sorted_preds[:15]]
        confidences = [p.get('confidence', 0) * 100 for p in sorted_preds[:15]]
        colors = ['#00cc00' if p.get('best_buy', False) else '#ff9900' for p in sorted_preds[:15]]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=confidences,
                marker_color=colors,
                text=[f"{c:.1f}%" for c in confidences],
                textposition='outside',
                name='Confidence'
            )
        ])
        
        fig.update_layout(
            title="📊 Prediction Confidence Scores",
            xaxis_title="Symbol",
            yaxis_title="Confidence (%)",
            yaxis=dict(range=[0, 100]),
            height=400,
            showlegend=False
        )
        
        return fig
    
    def profit_potential_chart(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create a scatter chart showing profit potential vs risk.
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        profit_potentials = [p.get('profit_potential', 0) * 100 for p in predictions]
        risk_scores = [p.get('risk_score', 0.5) * 100 for p in predictions]
        symbols = [p.get('symbol', '') for p in predictions]
        confidences = [p.get('confidence', 0) * 100 for p in predictions]
        best_buys = [p.get('best_buy', False) for p in predictions]
        
        # Create scatter plot
        fig = go.Figure()
        
        # Best buys in green
        best_buy_indices = [i for i, bb in enumerate(best_buys) if bb]
        if best_buy_indices:
            fig.add_trace(go.Scatter(
                x=[risk_scores[i] for i in best_buy_indices],
                y=[profit_potentials[i] for i in best_buy_indices],
                mode='markers+text',
                marker=dict(
                    size=[confidences[i] for i in best_buy_indices],
                    color='#00cc00',
                    line=dict(width=2, color='darkgreen')
                ),
                text=[symbols[i] for i in best_buy_indices],
                textposition='top center',
                name='Best Buy',
                hovertemplate='<b>%{text}</b><br>' +
                            'Profit Potential: %{y:.1f}%<br>' +
                            'Risk Score: %{x:.1f}%<br>' +
                            'Confidence: %{marker.size:.1f}%<extra></extra>'
            ))
        
        # Other predictions in orange
        other_indices = [i for i, bb in enumerate(best_buys) if not bb]
        if other_indices:
            fig.add_trace(go.Scatter(
                x=[risk_scores[i] for i in other_indices],
                y=[profit_potentials[i] for i in other_indices],
                mode='markers',
                marker=dict(
                    size=[confidences[i] for i in other_indices],
                    color='#ff9900',
                    opacity=0.6
                ),
                text=[symbols[i] for i in other_indices],
                name='Other',
                hovertemplate='<b>%{text}</b><br>' +
                            'Profit Potential: %{y:.1f}%<br>' +
                            'Risk Score: %{x:.1f}%<extra></extra>'
            ))
        
        fig.update_layout(
            title="💰 Profit Potential vs Risk Analysis",
            xaxis_title="Risk Score (%)",
            yaxis_title="Profit Potential (%)",
            height=500,
            hovermode='closest'
        )
        
        # Add quadrant lines
        fig.add_hline(y=10, line_dash="dash", line_color="gray", annotation_text="10% Profit")
        fig.add_vline(x=30, line_dash="dash", line_color="gray", annotation_text="30% Risk")
        
        return fig
    
    def prediction_radar_chart(
        self,
        prediction: Dict[str, Any]
    ) -> go.Figure:
        """
        Create a radar chart showing all prediction factors.
        
        Args:
            prediction: Single prediction dictionary
            
        Returns:
            Plotly figure
        """
        individual = prediction.get('individual_predictions', {})
        
        categories = list(individual.keys())
        values = [v * 100 for v in individual.values()]
        
        fig = go.Figure()
        
        fig.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name='Prediction Score',
            line_color='#1f77b4'
        ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 100]
                )),
            showlegend=True,
            title=f"🎯 Prediction Analysis: {prediction.get('symbol', 'Unknown')}",
            height=400
        )
        
        return fig
    
    def top_predictions_table(
        self,
        predictions: List[Dict[str, Any]],
        limit: int = 10
    ) -> pd.DataFrame:
        """
        Create a formatted table of top predictions.
        
        Args:
            predictions: List of prediction dictionaries
            limit: Number of top predictions to show
            
        Returns:
            Formatted DataFrame
        """
        if not predictions:
            return pd.DataFrame()
        
        # Sort by rank
        sorted_preds = sorted(predictions, key=lambda x: x.get('rank', 0), reverse=True)[:limit]
        
        data = []
        for i, pred in enumerate(sorted_preds, 1):
            data.append({
                'Rank': f"#{i}",
                'Symbol': pred.get('symbol', ''),
                'Type': pred.get('asset_type', 'unknown').upper(),
                'Best Buy': '✅ YES' if pred.get('best_buy', False) else '❌ NO',
                'Confidence': f"{pred.get('confidence', 0) * 100:.1f}%",
                'Profit Potential': f"+{pred.get('profit_potential', 0) * 100:.1f}%",
                'Risk Score': f"{pred.get('risk_score', 0) * 100:.1f}%",
                'Entry Price': f"${pred.get('entry_price', 0):,.2f}",
                'Target Price': f"${pred.get('target_price', 0):,.2f}",
                'Stop Loss': f"${pred.get('stop_loss', 0):,.2f}",
                'Overall Rank': f"{pred.get('rank', 0):.1f}/100"
            })
        
        return pd.DataFrame(data)
    
    def prediction_timeline(
        self,
        predictions: List[Dict[str, Any]]
    ) -> go.Figure:
        """
        Create a timeline showing prediction confidence over time.
        (Would use actual historical data in real implementation)
        
        Args:
            predictions: List of prediction dictionaries
            
        Returns:
            Plotly figure
        """
        if not predictions:
            return self._empty_chart("No predictions available")
        
        # Sort by rank
        sorted_preds = sorted(predictions, key=lambda x: x.get('rank', 0), reverse=True)[:10]
        
        symbols = [p.get('symbol', '') for p in sorted_preds]
        ranks = [p.get('rank', 0) for p in sorted_preds]
        
        fig = go.Figure(data=[
            go.Bar(
                x=symbols,
                y=ranks,
                marker_color='#1f77b4',
                text=[f"{r:.1f}" for r in ranks],
                textposition='outside',
                name='Rank'
            )
        ])
        
        fig.update_layout(
            title="📈 Top 10 Best Buy Rankings",
            xaxis_title="Symbol",
            yaxis_title="Rank Score (0-100)",
            yaxis=dict(range=[0, 100]),
            height=400,
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

