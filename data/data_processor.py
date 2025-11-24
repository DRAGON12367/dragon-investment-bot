"""
Data processing utilities for market data.
"""
import logging
from pathlib import Path
from typing import List, Dict, Any
import pandas as pd
from datetime import datetime, timedelta

from utils.config import Config


class DataProcessor:
    """Process and store market data."""
    
    def __init__(self, config: Config):
        """Initialize data processor."""
        self.config = config
        self.logger = logging.getLogger("ai_investment_bot.data_processor")
        self.data_directory = Path(config.data_directory)
        self.data_directory.mkdir(parents=True, exist_ok=True)
    
    def save_market_data(self, market_data: Dict[str, Any], timestamp: datetime = None):
        """Save market data to file."""
        if timestamp is None:
            timestamp = datetime.now()
        
        filename = self.data_directory / f"market_data_{timestamp.strftime('%Y%m%d')}.csv"
        
        # Convert to DataFrame
        records = []
        for symbol, data in market_data.items():
            record = {"timestamp": timestamp, "symbol": symbol, **data}
            records.append(record)
        
        df = pd.DataFrame(records)
        
        # Append to file or create new
        if filename.exists():
            existing_df = pd.read_csv(filename)
            combined_df = pd.concat([existing_df, df], ignore_index=True)
            combined_df.to_csv(filename, index=False)
        else:
            df.to_csv(filename, index=False)
        
        self.logger.debug(f"Saved market data to {filename}")
    
    def load_historical_data(
        self, 
        symbol: str, 
        days: int = 30
    ) -> pd.DataFrame:
        """Load historical data for a symbol."""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # Load data from files
        dataframes = []
        current_date = start_date
        
        while current_date <= end_date:
            filename = self.data_directory / f"market_data_{current_date.strftime('%Y%m%d')}.csv"
            if filename.exists():
                df = pd.read_csv(filename)
                df = df[df['symbol'] == symbol]
                if not df.empty:
                    dataframes.append(df)
            current_date += timedelta(days=1)
        
        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            combined_df['timestamp'] = pd.to_datetime(combined_df['timestamp'])
            combined_df = combined_df.sort_values('timestamp')
            return combined_df
        
        return pd.DataFrame()

