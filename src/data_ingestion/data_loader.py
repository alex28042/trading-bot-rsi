"""
Data ingestion module for loading historical OHLC data.

This module provides functionality to load market data from various sources
including CSV files, Yahoo Finance, Alpha Vantage API, and Binance API for cryptocurrency data.
"""

import pandas as pd
import numpy as np
# import yfinance as yf  # Optional dependency
import logging
from pathlib import Path
from typing import Optional, Union
from datetime import datetime, timedelta

# Binance connector import (optional)
try:
    from ..binance_connector import BinanceDataFetcher
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Class for loading and preprocessing market data from various sources."""

    def __init__(self, data_config):
        """
        Initialize DataLoader with configuration.

        Args:
            data_config: DataConfig object containing data source settings
        """
        self.config = data_config
        self.data = None

    def load_data(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Load market data based on configured data source.

        Args:
            symbol: Stock symbol (e.g., 'AAPL')
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format

        Returns:
            DataFrame with OHLC data
        """
        logger.info(f"Loading data for {symbol} from {start_date} to {end_date}")

        if self.config.data_source == "csv":
            return self._load_from_csv(symbol)
        elif self.config.data_source == "yahoo":
            return self._load_from_yahoo(symbol, start_date, end_date)
        elif self.config.data_source == "alpha_vantage":
            return self._load_from_alpha_vantage(symbol)
        elif self.config.data_source == "binance":
            return self._load_from_binance(symbol, start_date, end_date)
        else:
            raise ValueError(f"Unsupported data source: {self.config.data_source}")

    def _load_from_csv(self, symbol: str) -> pd.DataFrame:
        """Load data from CSV file."""
        if self.config.csv_filename:
            file_path = Path(self.config.data_directory) / self.config.csv_filename
        else:
            file_path = Path(self.config.data_directory) / f"{symbol}.csv"

        if not file_path.exists():
            raise FileNotFoundError(f"CSV file not found: {file_path}")

        logger.info(f"Loading data from CSV: {file_path}")

        try:
            df = pd.read_csv(file_path)
            return self._standardize_dataframe(df)
        except Exception as e:
            logger.error(f"Error loading CSV file: {e}")
            raise

    def _load_from_yahoo(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load data from Yahoo Finance."""
        logger.info(f"Loading data from Yahoo Finance for {symbol}")

        try:
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            df = ticker.history(start=start_date, end=end_date)

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return self._standardize_dataframe(df)
        except ImportError:
            raise ImportError("yfinance package required for Yahoo Finance data. Install with: pip install yfinance")
        except Exception as e:
            logger.error(f"Error loading data from Yahoo Finance: {e}")
            raise

    def _load_from_alpha_vantage(self, symbol: str) -> pd.DataFrame:
        """Load data from Alpha Vantage API."""
        if not self.config.alpha_vantage_api_key:
            raise ValueError("Alpha Vantage API key not provided")

        logger.info(f"Loading data from Alpha Vantage for {symbol}")

        # Note: This would require alpha_vantage package
        # For now, we'll raise a NotImplementedError
        raise NotImplementedError("Alpha Vantage integration not implemented yet")

    def _load_from_binance(self, symbol: str, start_date: str, end_date: str) -> pd.DataFrame:
        """Load cryptocurrency data from Binance API."""
        if not BINANCE_AVAILABLE:
            raise ImportError("Binance connector not available. Install python-binance: pip install python-binance")

        logger.info(f"Loading data from Binance for {symbol}")

        try:
            # Get timeframe from config if available, otherwise default to 4h
            timeframe = getattr(self.config, 'timeframe', '4h')

            # Initialize Binance data fetcher
            fetcher = BinanceDataFetcher()

            # Validate symbol format for Binance (should be like BTCUSDT)
            if not symbol.endswith('USDT'):
                logger.warning(f"Symbol {symbol} might not be in correct Binance format. "
                             f"Consider using format like BTCUSDT")

            # Fetch data from Binance
            df = fetcher.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                start_date=start_date,
                end_date=end_date
            )

            if df.empty:
                raise ValueError(f"No data found for symbol {symbol}")

            return self._standardize_dataframe(df)

        except Exception as e:
            logger.error(f"Error loading data from Binance: {e}")
            raise

    def _standardize_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize DataFrame format and column names.

        Args:
            df: Raw DataFrame from data source

        Returns:
            Standardized DataFrame with consistent column names
        """
        # Create a copy to avoid modifying original
        df = df.copy()

        # Standardize column names (handle different naming conventions)
        column_mapping = {
            'Open': 'open',
            'High': 'high',
            'Low': 'low',
            'Close': 'close',
            'Volume': 'volume',
            'Adj Close': 'adj_close',
            'Date': 'date'
        }

        # Rename columns if they exist
        for old_name, new_name in column_mapping.items():
            if old_name in df.columns:
                df.rename(columns={old_name: new_name}, inplace=True)

        # Ensure date is in the index
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
        elif df.index.name != 'Date' and not isinstance(df.index, pd.DatetimeIndex):
            # Try to convert index to datetime if it's not already
            try:
                df.index = pd.to_datetime(df.index)
            except:
                logger.warning("Could not convert index to datetime")

        # Ensure we have the required columns
        required_columns = ['open', 'high', 'low', 'close', 'volume']
        missing_columns = [col for col in required_columns if col not in df.columns]

        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Sort by date
        df.sort_index(inplace=True)

        # Validate data quality
        self._validate_data(df)

        logger.info(f"Data loaded successfully: {len(df)} rows from {df.index[0]} to {df.index[-1]}")

        return df

    def _validate_data(self, df: pd.DataFrame) -> None:
        """
        Validate data quality and handle common issues.

        Args:
            df: DataFrame to validate
        """
        # Check for missing values
        if df.isnull().any().any():
            logger.warning("Data contains missing values. Performing forward fill.")
            df.fillna(method='ffill', inplace=True)

        # Check for negative prices
        price_columns = ['open', 'high', 'low', 'close']
        for col in price_columns:
            if (df[col] <= 0).any():
                logger.warning(f"Found non-positive values in {col} column")

        # Check for logical inconsistencies (high < low, etc.)
        if (df['high'] < df['low']).any():
            logger.warning("Found rows where high < low")

        if (df['high'] < df['close']).any():
            logger.warning("Found rows where high < close")

        if (df['low'] > df['close']).any():
            logger.warning("Found rows where low > close")

    def save_data_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """
        Save DataFrame to CSV file.

        Args:
            df: DataFrame to save
            filename: Name of the CSV file
        """
        file_path = Path(self.config.data_directory) / filename
        df.to_csv(file_path)
        logger.info(f"Data saved to: {file_path}")

    def get_data_summary(self, df: pd.DataFrame) -> dict:
        """
        Get summary statistics of the loaded data.

        Args:
            df: DataFrame to analyze

        Returns:
            Dictionary with summary statistics
        """
        return {
            'start_date': str(df.index[0].date()),
            'end_date': str(df.index[-1].date()),
            'total_days': len(df),
            'symbol': getattr(self, 'symbol', 'Unknown'),
            'price_range': {
                'min': float(df['close'].min()),
                'max': float(df['close'].max()),
                'avg': float(df['close'].mean())
            },
            'volume_stats': {
                'avg_volume': float(df['volume'].mean()),
                'max_volume': float(df['volume'].max())
            }
        }


def create_sample_data(symbol: str = "AAPL", days: int = 500) -> pd.DataFrame:
    """
    Create sample OHLC data for testing purposes.

    Args:
        symbol: Stock symbol
        days: Number of days of data to generate

    Returns:
        DataFrame with synthetic OHLC data
    """
    np.random.seed(42)  # For reproducible results

    # Generate date range
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    dates = pd.date_range(start=start_date, end=end_date, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    # Generate synthetic price data with some realistic patterns
    initial_price = 100.0
    returns = np.random.normal(0.001, 0.02, len(dates))  # Daily returns

    # Add some momentum and mean reversion
    for i in range(1, len(returns)):
        returns[i] += 0.1 * returns[i-1]  # Momentum

    # Calculate prices
    prices = [initial_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))

    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate intraday volatility
        daily_range = close * np.random.uniform(0.005, 0.03)

        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = low + np.random.uniform(0, high - low)

        # Ensure logical consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        volume = int(np.random.uniform(1000000, 10000000))

        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(data, index=dates)
    return df