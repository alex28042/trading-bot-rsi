"""
Configuration settings for the RSI-ADX Momentum Trading Strategy.

This module contains all configurable parameters for the trading system,
including strategy parameters, risk management settings, and data sources.
"""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class StrategyConfig:
    """Configuration parameters for the RSI-ADX momentum strategy."""

    # Technical indicator parameters
    rsi_period: int = 14
    adx_period: int = 14

    # Entry conditions
    rsi_entry_threshold: float = 50.0
    adx_minimum_threshold: float = 25.0

    # Exit conditions
    rsi_exit_threshold: float = 50.0

    # Risk management parameters
    stop_loss_percentage: float = 0.02  # 2% stop loss
    take_profit_percentage: float = 0.04  # 4% take profit
    position_size: float = 0.10  # 10% of portfolio per trade

    # Backtesting parameters
    initial_capital: float = 100000.0  # $100k initial capital
    commission_per_trade: float = 5.0  # $5 per trade

    # Data parameters
    symbol: str = "BTCUSDT"  # Default to BTC/USDT for crypto trading
    start_date: str = "2020-01-01"
    end_date: str = "2023-12-31"


@dataclass
class DataConfig:
    """Configuration for data sources and file paths."""

    # File paths
    data_directory: str = "data"
    logs_directory: str = "logs"
    results_directory: str = "results"

    # Data source settings
    data_source: str = "binance"  # Options: 'csv', 'yahoo', 'alpha_vantage', 'binance'
    csv_filename: Optional[str] = None

    # Cryptocurrency timeframe settings
    timeframe: str = "4h"  # Options: '1h', '4h', '1d' for crypto

    # API settings (if using external data sources)
    alpha_vantage_api_key: Optional[str] = None

    # Binance API settings (optional for public data)
    binance_api_key: Optional[str] = None
    binance_api_secret: Optional[str] = None
    binance_testnet: bool = True  # Use testnet for safe testing


@dataclass
class CryptoConfig:
    """Configuration specific to cryptocurrency trading."""

    # Position sizing for crypto (in BTC fractions)
    position_size_btc: float = 0.1  # 0.1 BTC per trade
    use_btc_position_sizing: bool = True  # Use BTC fractions instead of USD percentage

    # Crypto-specific risk management
    crypto_stop_loss_percentage: float = 0.02  # 2% stop loss (more volatile than stocks)
    crypto_take_profit_percentage: float = 0.04  # 4% take profit

    # Trading fees (Binance spot trading)
    maker_fee: float = 0.001  # 0.1% maker fee
    taker_fee: float = 0.001  # 0.1% taker fee

    # Supported timeframes for crypto
    supported_timeframes: list = None  # Will be set in __post_init__

    # Market hours (crypto trades 24/7)
    trading_hours_per_day: int = 24
    trading_days_per_year: int = 365

    def __post_init__(self):
        if self.supported_timeframes is None:
            self.supported_timeframes = ['1h', '4h', '1d']


@dataclass
class BacktestConfig:
    """Configuration for backtesting engine."""

    # Performance metrics settings
    risk_free_rate: float = 0.02  # 2% annual risk-free rate
    trading_days_per_year: int = 252

    # Reporting settings
    generate_plots: bool = True
    save_results: bool = True
    verbose_logging: bool = True


class Config:
    """Main configuration class that combines all settings."""

    def __init__(self):
        self.strategy = StrategyConfig()
        self.data = DataConfig()
        self.backtest = BacktestConfig()
        self.crypto = CryptoConfig()

        # Create necessary directories
        self._create_directories()

    def _create_directories(self):
        """Create necessary directories if they don't exist."""
        directories = [
            self.data.data_directory,
            self.data.logs_directory,
            self.data.results_directory
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def update_strategy_params(self, **kwargs):
        """Update strategy parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.strategy, key):
                setattr(self.strategy, key, value)
            else:
                raise ValueError(f"Unknown strategy parameter: {key}")

    def update_data_params(self, **kwargs):
        """Update data parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.data, key):
                setattr(self.data, key, value)
            else:
                raise ValueError(f"Unknown data parameter: {key}")

    def update_crypto_params(self, **kwargs):
        """Update crypto-specific parameters dynamically."""
        for key, value in kwargs.items():
            if hasattr(self.crypto, key):
                setattr(self.crypto, key, value)
            else:
                raise ValueError(f"Unknown crypto parameter: {key}")

    def to_dict(self):
        """Convert configuration to dictionary for logging/serialization."""
        return {
            'strategy': self.strategy.__dict__,
            'data': self.data.__dict__,
            'backtest': self.backtest.__dict__,
            'crypto': self.crypto.__dict__
        }


# Global configuration instance
config = Config()