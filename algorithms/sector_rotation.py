"""
Sector Rotation Strategy - Rotate between sectors based on market cycle.
Wall Street Use: Institutional sector allocation and market timing.
"""
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional
from datetime import datetime
from collections import defaultdict


class SectorRotationStrategy:
    """
    Sector rotation strategy.
    
    Concept: Different sectors outperform at different stages of economic cycle.
    Rotate investments based on market conditions.
    """
    
    def __init__(self):
        """Initialize sector rotation strategy."""
        self.logger = logging.getLogger("ai_investment_bot.sector_rotation")
        
        # Sector classifications
        self.sectors = {
            'TECH': ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA'],
            'FINANCE': ['JPM', 'BAC', 'WFC', 'GS', 'MS'],
            'HEALTHCARE': ['JNJ', 'PFE', 'UNH', 'ABT', 'TMO'],
            'ENERGY': ['XOM', 'CVX', 'COP', 'SLB'],
            'CONSUMER': ['WMT', 'HD', 'NKE', 'SBUX'],
            'INDUSTRIAL': ['BA', 'CAT', 'GE', 'HON']
        }
        
    def classify_sector(self, symbol: str) -> str:
        """Classify symbol into sector."""
        for sector, symbols in self.sectors.items():
            if symbol in symbols:
                return sector
        return 'OTHER'
    
    def analyze_sector_performance(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze performance by sector.
        """
        sector_performance = defaultdict(list)
        
        for symbol, data in market_data.items():
            if data.get('asset_type') != 'stock':
                continue
            
            sector = self.classify_sector(symbol)
            change_pct = data.get('change_percent', 0)
            volume = data.get('volume', 0)
            
            sector_performance[sector].append({
                'symbol': symbol,
                'change_pct': change_pct,
                'volume': volume,
                'price': data.get('price', 0)
            })
        
        # Calculate sector averages
        sector_stats = {}
        for sector, stocks in sector_performance.items():
            if stocks:
                avg_change = np.mean([s['change_pct'] for s in stocks])
                total_volume = sum([s['volume'] for s in stocks])
                sector_stats[sector] = {
                    'avg_change': float(avg_change),
                    'total_volume': float(total_volume),
                    'stock_count': len(stocks),
                    'performance_rank': 0  # Will be set later
                }
        
        # Rank sectors by performance
        sorted_sectors = sorted(sector_stats.items(), 
                              key=lambda x: x[1]['avg_change'], 
                              reverse=True)
        
        for rank, (sector, stats) in enumerate(sorted_sectors, 1):
            stats['performance_rank'] = rank
        
        # Identify leading sectors
        leading_sectors = [sector for sector, stats in sorted_sectors[:3]]
        
        return {
            'sector_performance': dict(sector_stats),
            'leading_sectors': leading_sectors,
            'sector_rankings': {sector: stats['performance_rank'] 
                              for sector, stats in sector_stats.items()}
        }
    
    def market_cycle_detection(
        self,
        market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Detect market cycle stage based on sector performance.
        
        Early Cycle: Technology, Consumer Discretionary lead
        Mid Cycle: Industrials, Materials lead
        Late Cycle: Energy, Financials lead
        Recession: Utilities, Consumer Staples lead
        """
        sector_analysis = self.analyze_sector_performance(market_data)
        sector_perf = sector_analysis.get('sector_performance', {})
        
        # Get top performing sectors
        tech_perf = sector_perf.get('TECH', {}).get('avg_change', 0)
        energy_perf = sector_perf.get('ENERGY', {}).get('avg_change', 0)
        finance_perf = sector_perf.get('FINANCE', {}).get('avg_change', 0)
        consumer_perf = sector_perf.get('CONSUMER', {}).get('avg_change', 0)
        
        # Determine cycle stage
        if tech_perf > 2.0 and consumer_perf > 1.0:
            cycle = 'EARLY_CYCLE'
            recommended_sectors = ['TECH', 'CONSUMER']
        elif energy_perf > 2.0 and finance_perf > 1.0:
            cycle = 'LATE_CYCLE'
            recommended_sectors = ['ENERGY', 'FINANCE']
        elif tech_perf < -1.0 and consumer_perf < -1.0:
            cycle = 'RECESSION'
            recommended_sectors = ['HEALTHCARE', 'CONSUMER']  # Defensive
        else:
            cycle = 'MID_CYCLE'
            recommended_sectors = ['INDUSTRIAL', 'TECH']
        
        return {
            'market_cycle': cycle,
            'recommended_sectors': recommended_sectors,
            'sector_analysis': sector_analysis,
            'rotation_signal': 'ROTATE' if cycle in ['LATE_CYCLE', 'RECESSION'] else 'HOLD'
        }

