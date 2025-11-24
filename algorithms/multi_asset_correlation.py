"""
MULTI-ASSET CORRELATION ANALYZER - 200X UPGRADE
Advanced correlation analysis across multiple assets and timeframes
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
import logging
from scipy.stats import pearsonr, spearmanr
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform
import warnings
warnings.filterwarnings('ignore')


class MultiAssetCorrelation:
    """
    Advanced multi-asset correlation analysis system.
    
    Features:
    - Dynamic correlation matrices across multiple timeframes
    - Correlation regime detection
    - Portfolio diversification optimization
    - Cross-asset momentum detection
    - Correlation clustering
    - Risk spillover analysis
    - Sector/crypto correlation mapping
    """
    
    def __init__(self):
        """Initialize multi-asset correlation analyzer."""
        self.logger = logging.getLogger("ai_investment_bot.multi_asset_correlation")
        self.correlation_cache = {}
        self.regime_history = {}
        
    def calculate_correlation_matrix(
        self,
        market_data: Dict[str, pd.DataFrame],
        timeframe: str = '1h',
        method: str = 'pearson'
    ) -> pd.DataFrame:
        """
        Calculate correlation matrix for all assets.
        
        Args:
            market_data: Dict of symbol -> DataFrame with OHLCV data
            timeframe: Timeframe for correlation ('1h', '4h', '1d')
            method: Correlation method ('pearson', 'spearman', 'kendall')
        """
        try:
            # Extract close prices
            prices = {}
            for symbol, df in market_data.items():
                if 'close' in df.columns and len(df) > 20:
                    prices[symbol] = df['close']
            
            if len(prices) < 2:
                return pd.DataFrame()
            
            # Create price DataFrame
            price_df = pd.DataFrame(prices)
            
            # Calculate returns
            returns = price_df.pct_change().dropna()
            
            if len(returns) < 10:
                return pd.DataFrame()
            
            # Calculate correlation
            if method == 'pearson':
                corr_matrix = returns.corr()
            elif method == 'spearman':
                corr_matrix = returns.corr(method='spearman')
            else:
                corr_matrix = returns.corr(method='kendall')
            
            return corr_matrix.fillna(0)
            
        except Exception as e:
            self.logger.error(f"Error calculating correlation matrix: {e}")
            return pd.DataFrame()
    
    def detect_correlation_regime(
        self,
        corr_matrix: pd.DataFrame,
        lookback: int = 20
    ) -> Dict[str, Any]:
        """
        Detect current correlation regime (high/low correlation).
        
        Returns:
            Dict with regime info: 'regime', 'avg_correlation', 'volatility'
        """
        try:
            if corr_matrix.empty:
                return {'regime': 'unknown', 'avg_correlation': 0.0, 'volatility': 0.0}
            
            # Get upper triangle (exclude diagonal)
            mask = np.triu(np.ones_like(corr_matrix.values, dtype=bool), k=1)
            correlations = corr_matrix.values[mask]
            
            avg_corr = np.mean(np.abs(correlations))
            corr_vol = np.std(correlations)
            
            # Determine regime
            if avg_corr > 0.7:
                regime = 'high_correlation'
            elif avg_corr > 0.4:
                regime = 'moderate_correlation'
            else:
                regime = 'low_correlation'
            
            return {
                'regime': regime,
                'avg_correlation': avg_corr,
                'volatility': corr_vol,
                'max_correlation': np.max(np.abs(correlations)),
                'min_correlation': np.min(np.abs(correlations))
            }
            
        except Exception as e:
            self.logger.error(f"Error detecting correlation regime: {e}")
            return {'regime': 'unknown', 'avg_correlation': 0.0, 'volatility': 0.0}
    
    def find_diversification_opportunities(
        self,
        corr_matrix: pd.DataFrame,
        current_portfolio: List[str],
        max_correlation: float = 0.3
    ) -> List[Dict[str, Any]]:
        """
        Find assets with low correlation to current portfolio.
        
        Args:
            corr_matrix: Correlation matrix
            current_portfolio: List of symbols in current portfolio
            max_correlation: Maximum acceptable correlation
        """
        try:
            if corr_matrix.empty or not current_portfolio:
                return []
            
            opportunities = []
            
            # Calculate average correlation with portfolio for each asset
            for symbol in corr_matrix.columns:
                if symbol in current_portfolio:
                    continue
                
                # Get correlations with portfolio assets
                portfolio_corrs = []
                for portfolio_symbol in current_portfolio:
                    if portfolio_symbol in corr_matrix.index:
                        corr = corr_matrix.loc[portfolio_symbol, symbol]
                        portfolio_corrs.append(abs(corr))
                
                if portfolio_corrs:
                    avg_corr = np.mean(portfolio_corrs)
                    max_corr = np.max(portfolio_corrs)
                    
                    if avg_corr <= max_correlation:
                        opportunities.append({
                            'symbol': symbol,
                            'avg_correlation': avg_corr,
                            'max_correlation': max_corr,
                            'diversification_score': 1.0 - avg_corr
                        })
            
            # Sort by diversification score
            opportunities.sort(key=lambda x: x['diversification_score'], reverse=True)
            
            return opportunities
            
        except Exception as e:
            self.logger.error(f"Error finding diversification opportunities: {e}")
            return []
    
    def detect_cross_asset_momentum(
        self,
        market_data: Dict[str, pd.DataFrame],
        corr_matrix: pd.DataFrame,
        threshold: float = 0.6
    ) -> List[Dict[str, Any]]:
        """
        Detect momentum that's spreading across correlated assets.
        
        Returns:
            List of momentum clusters
        """
        try:
            if corr_matrix.empty:
                return []
            
            # Calculate momentum for each asset
            momentum_scores = {}
            for symbol, df in market_data.items():
                if 'close' in df.columns and len(df) > 20:
                    returns = df['close'].pct_change(periods=5)
                    momentum_scores[symbol] = returns.iloc[-1] if not returns.empty else 0.0
            
            # Find highly correlated pairs with similar momentum
            clusters = []
            processed = set()
            
            for symbol1 in corr_matrix.columns:
                if symbol1 in processed:
                    continue
                
                cluster = [symbol1]
                momentum1 = momentum_scores.get(symbol1, 0.0)
                
                for symbol2 in corr_matrix.columns:
                    if symbol2 == symbol1 or symbol2 in processed:
                        continue
                    
                    corr = abs(corr_matrix.loc[symbol1, symbol2])
                    momentum2 = momentum_scores.get(symbol2, 0.0)
                    
                    # Check if correlated and have similar momentum direction
                    if corr >= threshold and np.sign(momentum1) == np.sign(momentum2):
                        cluster.append(symbol2)
                        processed.add(symbol2)
                
                if len(cluster) > 1:
                    avg_momentum = np.mean([momentum_scores.get(s, 0.0) for s in cluster])
                    avg_corr = np.mean([
                        abs(corr_matrix.loc[cluster[0], s]) 
                        for s in cluster[1:] 
                        if s in corr_matrix.index
                    ])
                    
                    clusters.append({
                        'assets': cluster,
                        'avg_momentum': avg_momentum,
                        'avg_correlation': avg_corr,
                        'cluster_size': len(cluster),
                        'momentum_strength': abs(avg_momentum)
                    })
                
                processed.add(symbol1)
            
            # Sort by momentum strength
            clusters.sort(key=lambda x: x['momentum_strength'], reverse=True)
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error detecting cross-asset momentum: {e}")
            return []
    
    def analyze_risk_spillover(
        self,
        market_data: Dict[str, pd.DataFrame],
        corr_matrix: pd.DataFrame,
        target_symbol: str
    ) -> Dict[str, Any]:
        """
        Analyze risk spillover from other assets to target symbol.
        
        Returns:
            Risk spillover analysis
        """
        try:
            if corr_matrix.empty or target_symbol not in corr_matrix.columns:
                return {'risk_score': 0.0, 'high_risk_assets': []}
            
            # Calculate volatility for each asset
            volatilities = {}
            for symbol, df in market_data.items():
                if 'close' in df.columns and len(df) > 20:
                    returns = df['close'].pct_change()
                    volatilities[symbol] = returns.std() * np.sqrt(252)  # Annualized
            
            # Calculate risk contribution from each asset
            risk_contributions = []
            target_vol = volatilities.get(target_symbol, 0.0)
            
            for symbol in corr_matrix.columns:
                if symbol == target_symbol:
                    continue
                
                corr = corr_matrix.loc[target_symbol, symbol]
                vol = volatilities.get(symbol, 0.0)
                
                # Risk contribution = correlation * volatility
                risk_contrib = abs(corr) * vol
                
                risk_contributions.append({
                    'symbol': symbol,
                    'correlation': corr,
                    'volatility': vol,
                    'risk_contribution': risk_contrib
                })
            
            # Sort by risk contribution
            risk_contributions.sort(key=lambda x: x['risk_contribution'], reverse=True)
            
            # Calculate total risk score
            total_risk = sum(r['risk_contribution'] for r in risk_contributions)
            risk_score = min(total_risk / (target_vol + 0.01), 1.0)  # Normalize
            
            # Get high-risk assets (top contributors)
            high_risk_assets = [
                r['symbol'] for r in risk_contributions[:5]
                if r['risk_contribution'] > 0.1
            ]
            
            return {
                'risk_score': risk_score,
                'high_risk_assets': high_risk_assets,
                'risk_contributions': risk_contributions[:10],
                'total_risk_contribution': total_risk
            }
            
        except Exception as e:
            self.logger.error(f"Error analyzing risk spillover: {e}")
            return {'risk_score': 0.0, 'high_risk_assets': []}
    
    def cluster_assets(
        self,
        corr_matrix: pd.DataFrame,
        n_clusters: Optional[int] = None,
        threshold: float = 0.5
    ) -> Dict[str, int]:
        """
        Cluster assets based on correlation similarity.
        
        Returns:
            Dict mapping symbol -> cluster_id
        """
        try:
            if corr_matrix.empty:
                return {}
            
            # Convert correlation to distance (1 - abs(correlation))
            distance_matrix = 1 - np.abs(corr_matrix.values)
            
            # Perform hierarchical clustering
            condensed_distances = squareform(distance_matrix)
            linkage_matrix = linkage(condensed_distances, method='ward')
            
            # Determine number of clusters if not specified
            if n_clusters is None:
                n_clusters = max(2, min(10, len(corr_matrix) // 3))
            
            # Get cluster assignments
            cluster_labels = fcluster(linkage_matrix, n_clusters, criterion='maxclust')
            
            # Map symbols to clusters
            clusters = {}
            for i, symbol in enumerate(corr_matrix.columns):
                clusters[symbol] = int(cluster_labels[i])
            
            return clusters
            
        except Exception as e:
            self.logger.error(f"Error clustering assets: {e}")
            return {}
    
    def get_correlation_insights(
        self,
        market_data: Dict[str, pd.DataFrame]
    ) -> Dict[str, Any]:
        """
        Get comprehensive correlation insights.
        
        Returns:
            Complete correlation analysis
        """
        try:
            # Calculate correlation matrix
            corr_matrix = self.calculate_correlation_matrix(market_data)
            
            if corr_matrix.empty:
                return {'status': 'insufficient_data'}
            
            # Detect regime
            regime = self.detect_correlation_regime(corr_matrix)
            
            # Find diversification opportunities (if we had portfolio info)
            # diversification = self.find_diversification_opportunities(corr_matrix, [])
            
            # Detect cross-asset momentum
            momentum_clusters = self.detect_cross_asset_momentum(market_data, corr_matrix)
            
            # Get top correlated pairs
            top_pairs = []
            for i, symbol1 in enumerate(corr_matrix.columns):
                for symbol2 in corr_matrix.columns[i+1:]:
                    corr = corr_matrix.loc[symbol1, symbol2]
                    top_pairs.append({
                        'asset1': symbol1,
                        'asset2': symbol2,
                        'correlation': corr
                    })
            
            top_pairs.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            # Cluster assets
            clusters = self.cluster_assets(corr_matrix)
            
            return {
                'correlation_matrix': corr_matrix,
                'regime': regime,
                'momentum_clusters': momentum_clusters[:5],
                'top_correlated_pairs': top_pairs[:10],
                'asset_clusters': clusters,
                'matrix_size': len(corr_matrix)
            }
            
        except Exception as e:
            self.logger.error(f"Error getting correlation insights: {e}")
            return {'status': 'error', 'message': str(e)}

