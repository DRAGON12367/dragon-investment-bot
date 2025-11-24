"""
Portfolio Analytics - Advanced portfolio analysis and optimization.
Wall Street Use: Professional portfolio management and risk assessment.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
from collections import defaultdict

try:
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class PortfolioAnalytics:
    """
    Advanced portfolio analytics and visualization.
    
    Features:
    1. Portfolio Heat Map
    2. Correlation Analysis
    3. Sector/Asset Allocation
    4. Risk Contribution Analysis
    5. Performance Attribution
    6. Portfolio Optimization
    """
    
    def __init__(self):
        """Initialize portfolio analytics."""
        self.logger = logging.getLogger("ai_investment_bot.portfolio_analytics")
        
    def generate_portfolio_heatmap(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate portfolio heat map showing performance.
        
        Wall Street Use: Visual representation of portfolio performance.
        """
        if not positions:
            return {}
        
        heatmap_data = []
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol not in market_data:
                continue
            
            asset_data = market_data[symbol]
            current_price = asset_data.get('price', asset_data.get('close', 0))
            entry_price = position.get('entry_price', current_price)
            quantity = position.get('quantity', 0)
            
            # Calculate P&L
            pnl = (current_price - entry_price) * quantity
            pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
            
            # Position value
            position_value = current_price * quantity
            
            # 24h change
            change_24h = asset_data.get('change_percent', 0)
            
            heatmap_data.append({
                'symbol': symbol,
                'asset_type': asset_data.get('asset_type', 'unknown'),
                'position_value': position_value,
                'pnl': pnl,
                'pnl_pct': pnl_pct,
                'change_24h': change_24h,
                'quantity': quantity,
                'entry_price': entry_price,
                'current_price': current_price
            })
        
        # Sort by P&L
        heatmap_data.sort(key=lambda x: x['pnl'], reverse=True)
        
        # Calculate totals
        total_value = sum(p['position_value'] for p in heatmap_data)
        total_pnl = sum(p['pnl'] for p in heatmap_data)
        total_pnl_pct = (total_pnl / sum(p['position_value'] - p['pnl'] for p in heatmap_data) * 100) if total_value > 0 else 0
        
        return {
            'positions': heatmap_data,
            'total_value': total_value,
            'total_pnl': total_pnl,
            'total_pnl_pct': total_pnl_pct,
            'position_count': len(heatmap_data),
            'winning_positions': len([p for p in heatmap_data if p['pnl'] > 0]),
            'losing_positions': len([p for p in heatmap_data if p['pnl'] < 0])
        }
    
    def analyze_correlation_clusters(
        self,
        price_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Analyze correlation clusters in portfolio.
        
        Wall Street Use: Identify correlated positions (diversification risk).
        """
        if len(price_data) < 2:
            return {}
        
        # Calculate returns
        returns_df = pd.DataFrame({symbol: series.pct_change().dropna() 
                                  for symbol, series in price_data.items()})
        returns_df = returns_df.dropna()
        
        if len(returns_df) < 20:
            return {}
        
        # Correlation matrix
        correlation_matrix = returns_df.corr()
        
        # Find highly correlated pairs
        high_correlation_pairs = []
        for i, symbol1 in enumerate(correlation_matrix.columns):
            for j, symbol2 in enumerate(correlation_matrix.columns):
                if i < j:  # Avoid duplicates
                    corr = correlation_matrix.loc[symbol1, symbol2]
                    if abs(corr) > 0.7:  # High correlation threshold
                        high_correlation_pairs.append({
                            'symbol1': symbol1,
                            'symbol2': symbol2,
                            'correlation': float(corr),
                            'type': 'POSITIVE' if corr > 0 else 'NEGATIVE'
                        })
        
        # Cluster analysis (if sklearn available)
        clusters = {}
        if SKLEARN_AVAILABLE and len(returns_df.columns) >= 3:
            try:
                # Use PCA for dimensionality reduction
                pca = PCA(n_components=min(3, len(returns_df.columns)))
                pca_data = pca.fit_transform(returns_df.T)
                
                # K-means clustering
                n_clusters = min(3, len(returns_df.columns))
                kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
                cluster_labels = kmeans.fit_predict(pca_data)
                
                # Group symbols by cluster
                for idx, symbol in enumerate(returns_df.columns):
                    cluster_id = int(cluster_labels[idx])
                    if cluster_id not in clusters:
                        clusters[cluster_id] = []
                    clusters[cluster_id].append(symbol)
            except Exception as e:
                self.logger.debug(f"Error in clustering: {e}")
        
        # Diversification score
        avg_correlation = correlation_matrix.values[np.triu(np.ones_like(correlation_matrix.values), k=1).astype(bool)].mean()
        diversification_score = 1 - abs(avg_correlation)  # Higher = more diversified
        
        return {
            'correlation_matrix': correlation_matrix.to_dict(),
            'high_correlation_pairs': high_correlation_pairs,
            'clusters': clusters,
            'average_correlation': float(avg_correlation),
            'diversification_score': float(diversification_score),
            'diversification_quality': 'EXCELLENT' if diversification_score > 0.7 else 'GOOD' if diversification_score > 0.4 else 'POOR'
        }
    
    def calculate_risk_contribution(
        self,
        positions: List[Dict[str, Any]],
        returns_data: Dict[str, pd.Series]
    ) -> Dict[str, Any]:
        """
        Calculate risk contribution of each position.
        
        Wall Street Use: Identify which positions contribute most to portfolio risk.
        """
        if not positions or not returns_data:
            return {}
        
        # Calculate position weights
        total_value = sum(p.get('value', 0) for p in positions)
        if total_value == 0:
            return {}
        
        position_weights = {}
        position_volatilities = {}
        
        for position in positions:
            symbol = position.get('symbol')
            if symbol not in returns_data:
                continue
            
            position_value = position.get('value', 0)
            weight = position_value / total_value if total_value > 0 else 0
            
            # Calculate volatility
            returns = returns_data[symbol].dropna()
            if len(returns) >= 20:
                volatility = returns.std() * np.sqrt(252)  # Annualized
            else:
                volatility = 0.20  # Default 20%
            
            position_weights[symbol] = weight
            position_volatilities[symbol] = volatility
        
        # Risk contribution
        risk_contributions = {}
        for symbol, weight in position_weights.items():
            vol = position_volatilities.get(symbol, 0)
            risk_contribution = weight * vol
            risk_contributions[symbol] = {
                'weight': float(weight),
                'volatility': float(vol),
                'risk_contribution': float(risk_contribution),
                'risk_contribution_pct': float(risk_contribution / sum(position_volatilities.values()) * 100) if sum(position_volatilities.values()) > 0 else 0
            }
        
        # Sort by risk contribution
        sorted_risk = sorted(risk_contributions.items(), 
                           key=lambda x: x[1]['risk_contribution'], 
                           reverse=True)
        
        return {
            'risk_contributions': dict(sorted_risk),
            'total_portfolio_risk': float(sum(r['risk_contribution'] for r in risk_contributions.values())),
            'top_risk_contributors': [symbol for symbol, _ in sorted_risk[:3]]
        }
    
    def performance_attribution(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Performance attribution analysis.
        
        Wall Street Use: Understand what's driving portfolio performance.
        """
        if not positions:
            return {}
        
        # Group by asset type
        by_type = defaultdict(list)
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                asset_type = market_data[symbol].get('asset_type', 'unknown')
                by_type[asset_type].append(position)
        
        # Calculate performance by type
        type_performance = {}
        for asset_type, type_positions in by_type.items():
            total_value = sum(p.get('value', 0) for p in type_positions)
            total_pnl = sum(
                (market_data[p['symbol']].get('price', 0) - p.get('entry_price', 0)) * p.get('quantity', 0)
                for p in type_positions if p['symbol'] in market_data
            )
            
            type_performance[asset_type] = {
                'position_count': len(type_positions),
                'total_value': total_value,
                'total_pnl': total_pnl,
                'pnl_pct': (total_pnl / (total_value - total_pnl) * 100) if total_value > 0 else 0
            }
        
        # Best and worst performers
        all_pnl = []
        for position in positions:
            symbol = position.get('symbol')
            if symbol in market_data:
                current_price = market_data[symbol].get('price', 0)
                entry_price = position.get('entry_price', current_price)
                pnl = (current_price - entry_price) * position.get('quantity', 0)
                all_pnl.append({
                    'symbol': symbol,
                    'pnl': pnl,
                    'pnl_pct': ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0
                })
        
        all_pnl.sort(key=lambda x: x['pnl'], reverse=True)
        
        return {
            'by_asset_type': type_performance,
            'best_performers': all_pnl[:3] if all_pnl else [],
            'worst_performers': all_pnl[-3:] if all_pnl else [],
            'total_positions': len(positions),
            'winning_count': len([p for p in all_pnl if p['pnl'] > 0]),
            'losing_count': len([p for p in all_pnl if p['pnl'] < 0])
        }
    
    def comprehensive_portfolio_analysis(
        self,
        positions: List[Dict[str, Any]],
        market_data: Dict[str, Any],
        price_history: Optional[Dict[str, pd.Series]] = None
    ) -> Dict[str, Any]:
        """Comprehensive portfolio analysis."""
        results = {
            'timestamp': datetime.now().isoformat()
        }
        
        # Heat map
        try:
            heatmap = self.generate_portfolio_heatmap(positions, market_data)
            results['heatmap'] = heatmap
        except Exception as e:
            self.logger.debug(f"Error in heatmap generation: {e}")
        
        # Correlation analysis
        if price_history and len(price_history) >= 2:
            try:
                correlation = self.analyze_correlation_clusters(price_history)
                results['correlation'] = correlation
            except Exception as e:
                self.logger.debug(f"Error in correlation analysis: {e}")
        
        # Risk contribution
        if price_history:
            try:
                returns_data = {symbol: series.pct_change().dropna() 
                              for symbol, series in price_history.items()}
                risk_contrib = self.calculate_risk_contribution(positions, returns_data)
                results['risk_contribution'] = risk_contrib
            except Exception as e:
                self.logger.debug(f"Error in risk contribution: {e}")
        
        # Performance attribution
        try:
            attribution = self.performance_attribution(positions, market_data)
            results['performance_attribution'] = attribution
        except Exception as e:
            self.logger.debug(f"Error in performance attribution: {e}")
        
        return results

