"""
Configuration management for the AI Investment Bot.
"""
import os
import json
from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class Config:
    """Configuration class for the investment bot."""
    
    # Trading parameters
    trading_interval: int = 60  # seconds
    max_position_size: float = 0.1  # 10% of portfolio
    min_confidence_threshold: float = 0.7
    
    # Risk management
    max_daily_loss: float = 0.02  # 2% max daily loss
    stop_loss_percentage: float = 0.05  # 5% stop loss
    take_profit_percentage: float = 0.10  # 10% take profit
    
    # Broker API
    broker_api_key: Optional[str] = None
    broker_api_secret: Optional[str] = None
    broker_base_url: str = "https://api.broker.com"
    
    # Logging
    log_level: str = "INFO"
    log_file: Optional[str] = "logs/bot.log"
    
    # Data
    data_directory: str = "data"
    model_directory: str = "models"
    
    def __init__(self, config_path: Optional[str] = None):
        """Load configuration from file or environment variables."""
        if config_path and Path(config_path).exists():
            self.load_from_file(config_path)
        else:
            self.load_from_env()
            
        # Create necessary directories
        self._create_directories()
    
    def load_from_file(self, config_path: str):
        """Load configuration from JSON file."""
        with open(config_path, 'r') as f:
            config_data = json.load(f)
            for key, value in config_data.items():
                if hasattr(self, key):
                    setattr(self, key, value)
    
    def load_from_env(self):
        """Load configuration from environment variables."""
        self.broker_api_key = os.getenv("BROKER_API_KEY", self.broker_api_key)
        self.broker_api_secret = os.getenv("BROKER_API_SECRET", self.broker_api_secret)
        self.broker_base_url = os.getenv("BROKER_BASE_URL", self.broker_base_url)
        self.log_level = os.getenv("LOG_LEVEL", self.log_level)
        self.trading_interval = int(os.getenv("TRADING_INTERVAL", self.trading_interval))
        self.max_position_size = float(os.getenv("MAX_POSITION_SIZE", self.max_position_size))
    
    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        Path(self.data_directory).mkdir(parents=True, exist_ok=True)
        Path(self.model_directory).mkdir(parents=True, exist_ok=True)
        if self.log_file:
            Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            "trading_interval": self.trading_interval,
            "max_position_size": self.max_position_size,
            "min_confidence_threshold": self.min_confidence_threshold,
            "max_daily_loss": self.max_daily_loss,
            "stop_loss_percentage": self.stop_loss_percentage,
            "take_profit_percentage": self.take_profit_percentage,
            "broker_base_url": self.broker_base_url,
            "log_level": self.log_level,
        }

