"""
Binance Data Fetcher

Specialized module for fetching cryptocurrency market data from Binance API.
Optimized for BTC/USDT trading with support for multiple timeframes.
"""

import logging
import pandas as pd
from datetime import datetime, timedelta
from typing import Optional, List, Dict, Any
import time
from binance.client import Client
from .binance_client import BinanceClient

logger = logging.getLogger(__name__)


class BinanceDataFetcher:
    """
    High-level data fetcher for Binance cryptocurrency data.

    Provides convenient methods for fetching OHLCV data with proper
    error handling, rate limiting, and data validation.
    """

    # Binance timeframe mapping
    TIMEFRAME_MAPPING = {
        '1m': Client.KLINE_INTERVAL_1MINUTE,
        '3m': Client.KLINE_INTERVAL_3MINUTE,
        '5m': Client.KLINE_INTERVAL_5MINUTE,
        '15m': Client.KLINE_INTERVAL_15MINUTE,
        '30m': Client.KLINE_INTERVAL_30MINUTE,
        '1h': Client.KLINE_INTERVAL_1HOUR,
        '2h': Client.KLINE_INTERVAL_2HOUR,
        '4h': Client.KLINE_INTERVAL_4HOUR,
        '6h': Client.KLINE_INTERVAL_6HOUR,
        '8h': Client.KLINE_INTERVAL_8HOUR,
        '12h': Client.KLINE_INTERVAL_12HOUR,
        '1d': Client.KLINE_INTERVAL_1DAY,
        '3d': Client.KLINE_INTERVAL_3DAY,
        '1w': Client.KLINE_INTERVAL_1WEEK,
        '1M': Client.KLINE_INTERVAL_1MONTH,
    }

    def __init__(self, client: Optional[BinanceClient] = None):
        """
        Initialize data fetcher.

        Args:
            client: BinanceClient instance (creates new if None)
        """
        self.client = client or BinanceClient()

        # Test connectivity on initialization
        if not self.client.test_connectivity():
            raise ConnectionError("Failed to connect to Binance API")

        logger.info("BinanceDataFetcher initialized successfully")

    def fetch_ohlcv(self, symbol: str, timeframe: str, start_date: Optional[str] = None,
                   end_date: Optional[str] = None, limit: int = 1000) -> pd.DataFrame:
        """
        Fetch OHLCV data for a symbol.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            timeframe: Timeframe ('1h', '4h', '1d', etc.)
            start_date: Start date string (YYYY-MM-DD or ISO format)
            end_date: End date string (YYYY-MM-DD or ISO format)
            limit: Maximum number of candles to fetch

        Returns:
            DataFrame with OHLCV data and timestamp index
        """
        # Validate inputs
        if timeframe not in self.TIMEFRAME_MAPPING:
            raise ValueError(f"Invalid timeframe: {timeframe}. "
                           f"Supported: {list(self.TIMEFRAME_MAPPING.keys())}")

        if not self.client.validate_symbol(symbol):
            raise ValueError(f"Invalid or non-tradeable symbol: {symbol}")

        try:
            # Get kline data from Binance
            interval = self.TIMEFRAME_MAPPING[timeframe]
            klines = self.client.get_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_date,
                end_str=end_date,
                limit=limit
            )

            if not klines:
                raise ValueError(f"No data returned for {symbol}")

            # Convert to DataFrame
            df = self._klines_to_dataframe(klines)

            # Validate data quality
            self._validate_ohlcv_data(df, symbol, timeframe)

            logger.info(f"Fetched {len(df)} candles for {symbol} ({timeframe})")
            return df

        except Exception as e:
            logger.error(f"Failed to fetch OHLCV data for {symbol}: {e}")
            raise

    def fetch_btc_usdt_data(self, timeframe: str = '4h', days: int = 365) -> pd.DataFrame:
        """
        Convenience method to fetch BTC/USDT data.

        Args:
            timeframe: Timeframe ('1h', '4h', '1d')
            days: Number of days of historical data

        Returns:
            DataFrame with BTC/USDT OHLCV data
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        return self.fetch_ohlcv(
            symbol='BTCUSDT',
            timeframe=timeframe,
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

    def get_current_price(self, symbol: str) -> float:
        """
        Get current price for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Current price as float
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol)
            return float(ticker['price'])
        except Exception as e:
            logger.error(f"Failed to get current price for {symbol}: {e}")
            raise

    def get_24h_stats(self, symbol: str) -> Dict[str, float]:
        """
        Get 24-hour statistics for symbol.

        Args:
            symbol: Trading pair symbol

        Returns:
            Dictionary with 24h stats (price_change, volume, etc.)
        """
        try:
            ticker = self.client.get_symbol_ticker(symbol)
            return {
                'price': float(ticker['lastPrice']),
                'price_change': float(ticker['priceChange']),
                'price_change_percent': float(ticker['priceChangePercent']),
                'high': float(ticker['highPrice']),
                'low': float(ticker['lowPrice']),
                'volume': float(ticker['volume']),
                'quote_volume': float(ticker['quoteVolume']),
                'open_price': float(ticker['openPrice']),
                'prev_close': float(ticker['prevClosePrice']),
                'bid_price': float(ticker['bidPrice']),
                'ask_price': float(ticker['askPrice']),
                'weighted_avg_price': float(ticker['weightedAvgPrice'])
            }
        except Exception as e:
            logger.error(f"Failed to get 24h stats for {symbol}: {e}")
            raise

    def get_supported_symbols(self, quote_asset: str = 'USDT') -> List[str]:
        """
        Get list of supported trading symbols.

        Args:
            quote_asset: Quote asset to filter by (e.g., 'USDT', 'BTC')

        Returns:
            List of symbol strings
        """
        try:
            exchange_info = self.client.get_exchange_info()
            symbols = []

            for symbol_info in exchange_info['symbols']:
                if (symbol_info['status'] == 'TRADING' and
                    symbol_info['quoteAsset'] == quote_asset):
                    symbols.append(symbol_info['symbol'])

            return sorted(symbols)
        except Exception as e:
            logger.error(f"Failed to get supported symbols: {e}")
            raise

    def get_market_depth(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get market depth (order book) for symbol.

        Args:
            symbol: Trading pair symbol
            limit: Depth limit

        Returns:
            Dictionary with bids and asks
        """
        try:
            order_book = self.client.get_order_book(symbol, limit)
            return {
                'bids': [[float(price), float(qty)] for price, qty in order_book['bids']],
                'asks': [[float(price), float(qty)] for price, qty in order_book['asks']],
                'last_update_id': order_book['lastUpdateId']
            }
        except Exception as e:
            logger.error(f"Failed to get market depth for {symbol}: {e}")
            raise

    def _klines_to_dataframe(self, klines: List[List]) -> pd.DataFrame:
        """
        Convert Binance klines data to pandas DataFrame.

        Args:
            klines: Raw klines data from Binance API

        Returns:
            DataFrame with OHLCV data
        """
        # Binance kline format:
        # [timestamp, open, high, low, close, volume, close_time, quote_asset_volume,
        #  number_of_trades, taker_buy_base_asset_volume, taker_buy_quote_asset_volume, ignore]

        df = pd.DataFrame(klines, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])

        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)

        # Convert OHLCV columns to numeric
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

        # Keep only OHLCV columns for simplicity
        df = df[['open', 'high', 'low', 'close', 'volume']]

        return df

    def _validate_ohlcv_data(self, df: pd.DataFrame, symbol: str, timeframe: str) -> None:
        """
        Validate OHLCV data quality.

        Args:
            df: DataFrame to validate
            symbol: Symbol name for logging
            timeframe: Timeframe for logging

        Raises:
            ValueError: If data validation fails
        """
        if df.empty:
            raise ValueError(f"Empty dataset for {symbol} {timeframe}")

        # Check for missing values
        if df.isnull().any().any():
            logger.warning(f"Found missing values in {symbol} {timeframe} data")

        # Check OHLC relationships
        invalid_ohlc = (
            (df['high'] < df['open']) |
            (df['high'] < df['close']) |
            (df['low'] > df['open']) |
            (df['low'] > df['close']) |
            (df['high'] < df['low'])
        )

        if invalid_ohlc.any():
            logger.warning(f"Found {invalid_ohlc.sum()} invalid OHLC relationships in {symbol} data")

        # Check for negative values
        if (df[['open', 'high', 'low', 'close', 'volume']] < 0).any().any():
            raise ValueError(f"Found negative values in {symbol} data")

        # Check for zero prices (volume can be zero)
        price_cols = ['open', 'high', 'low', 'close']
        if (df[price_cols] == 0).any().any():
            raise ValueError(f"Found zero prices in {symbol} data")

        logger.debug(f"Data validation passed for {symbol} {timeframe}")

    def fetch_multiple_timeframes(self, symbol: str, timeframes: List[str],
                                days: int = 365) -> Dict[str, pd.DataFrame]:
        """
        Fetch data for multiple timeframes.

        Args:
            symbol: Trading pair symbol
            timeframes: List of timeframes to fetch
            days: Number of days of historical data

        Returns:
            Dictionary mapping timeframes to DataFrames
        """
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)

        results = {}
        for timeframe in timeframes:
            try:
                df = self.fetch_ohlcv(
                    symbol=symbol,
                    timeframe=timeframe,
                    start_date=start_date.strftime('%Y-%m-%d'),
                    end_date=end_date.strftime('%Y-%m-%d')
                )
                results[timeframe] = df

                # Rate limiting to avoid API limits
                time.sleep(0.1)

            except Exception as e:
                logger.error(f"Failed to fetch {symbol} {timeframe}: {e}")
                # Continue with other timeframes
                continue

        return results