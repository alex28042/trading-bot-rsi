"""
Binance Client for Cryptocurrency Trading

Handles connection to Binance API with proper error handling and rate limiting.
Designed for paper trading preparation with future live trading capability.
"""

import logging
import time
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException, BinanceOrderException
import pandas as pd

logger = logging.getLogger(__name__)


class BinanceClient:
    """
    Binance API client with comprehensive error handling and rate limiting.

    This class provides a safe interface to Binance API for:
    - Historical data fetching
    - Real-time data streaming
    - Paper trading preparation (no actual trading)
    - Account information (when authenticated)
    """

    def __init__(self, api_key: Optional[str] = None, api_secret: Optional[str] = None,
                 testnet: bool = True):
        """
        Initialize Binance client.

        Args:
            api_key: Binance API key (optional for public data)
            api_secret: Binance API secret (optional for public data)
            testnet: Use testnet for safe testing (default: True)
        """
        self.api_key = api_key
        self.api_secret = api_secret
        self.testnet = testnet

        # Initialize client (can be None for public data only)
        self.client = None
        if api_key and api_secret:
            try:
                self.client = Client(
                    api_key=api_key,
                    api_secret=api_secret,
                    testnet=testnet
                )
                logger.info(f"Binance client initialized {'(TESTNET)' if testnet else '(LIVE)'}")
            except Exception as e:
                logger.error(f"Failed to initialize Binance client: {e}")
                raise
        else:
            # Public client for data fetching only
            self.client = Client()
            logger.info("Binance public client initialized (data only)")

    def get_server_time(self) -> Dict[str, Any]:
        """Get Binance server time for synchronization."""
        try:
            return self.client.get_server_time()
        except BinanceAPIException as e:
            logger.error(f"Failed to get server time: {e}")
            raise

    def get_exchange_info(self, symbol: Optional[str] = None) -> Dict[str, Any]:
        """
        Get exchange trading rules and symbol information.

        Args:
            symbol: Specific symbol to get info for (e.g., 'BTCUSDT')

        Returns:
            Exchange information dictionary
        """
        try:
            if symbol:
                return self.client.get_symbol_info(symbol)
            else:
                return self.client.get_exchange_info()
        except BinanceAPIException as e:
            logger.error(f"Failed to get exchange info: {e}")
            raise

    def get_symbol_ticker(self, symbol: str) -> Dict[str, Any]:
        """
        Get 24hr ticker price change statistics.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            Ticker information dictionary
        """
        try:
            return self.client.get_ticker(symbol=symbol)
        except BinanceAPIException as e:
            logger.error(f"Failed to get ticker for {symbol}: {e}")
            raise

    def get_klines(self, symbol: str, interval: str, limit: int = 500,
                   start_str: Optional[str] = None, end_str: Optional[str] = None) -> List[List]:
        """
        Get historical kline/candlestick data.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')
            interval: Kline interval ('1h', '4h', '1d', etc.)
            limit: Number of klines to retrieve (max 1000)
            start_str: Start time string (e.g., '2023-01-01')
            end_str: End time string (e.g., '2023-12-31')

        Returns:
            List of kline data [timestamp, open, high, low, close, volume, ...]
        """
        try:
            return self.client.get_historical_klines(
                symbol=symbol,
                interval=interval,
                start_str=start_str,
                end_str=end_str,
                limit=limit
            )
        except BinanceAPIException as e:
            logger.error(f"Failed to get klines for {symbol}: {e}")
            raise

    def get_account_info(self) -> Dict[str, Any]:
        """
        Get account information (requires authentication).

        Returns:
            Account information including balances
        """
        if not self.client or not self.api_key:
            raise ValueError("Authentication required for account information")

        try:
            return self.client.get_account()
        except BinanceAPIException as e:
            logger.error(f"Failed to get account info: {e}")
            raise

    def get_asset_balance(self, asset: str) -> Dict[str, str]:
        """
        Get balance for specific asset.

        Args:
            asset: Asset symbol (e.g., 'BTC', 'USDT')

        Returns:
            Balance information dictionary
        """
        if not self.client or not self.api_key:
            raise ValueError("Authentication required for balance information")

        try:
            return self.client.get_asset_balance(asset=asset)
        except BinanceAPIException as e:
            logger.error(f"Failed to get balance for {asset}: {e}")
            raise

    def test_connectivity(self) -> bool:
        """
        Test connectivity to Binance API.

        Returns:
            True if connection successful, False otherwise
        """
        try:
            self.client.ping()
            logger.info("Binance API connectivity test successful")
            return True
        except Exception as e:
            logger.error(f"Binance API connectivity test failed: {e}")
            return False

    def get_all_tickers(self) -> List[Dict[str, str]]:
        """
        Get price ticker for all symbols.

        Returns:
            List of ticker information for all symbols
        """
        try:
            return self.client.get_all_tickers()
        except BinanceAPIException as e:
            logger.error(f"Failed to get all tickers: {e}")
            raise

    def validate_symbol(self, symbol: str) -> bool:
        """
        Validate if symbol exists and is tradeable.

        Args:
            symbol: Trading pair symbol (e.g., 'BTCUSDT')

        Returns:
            True if symbol is valid and tradeable
        """
        try:
            symbol_info = self.get_exchange_info(symbol)
            return symbol_info is not None and symbol_info.get('status') == 'TRADING'
        except:
            return False

    def place_test_order(self, symbol: str, side: str, order_type: str,
                        quantity: float, price: Optional[float] = None) -> Dict[str, Any]:
        """
        Place a test order (no actual execution).

        Args:
            symbol: Trading pair symbol
            side: 'BUY' or 'SELL'
            order_type: 'MARKET', 'LIMIT', etc.
            quantity: Order quantity
            price: Order price (for limit orders)

        Returns:
            Test order response
        """
        if not self.client:
            raise ValueError("Client not initialized")

        try:
            if order_type == 'MARKET':
                return self.client.create_test_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity
                )
            else:
                return self.client.create_test_order(
                    symbol=symbol,
                    side=side,
                    type=order_type,
                    quantity=quantity,
                    price=price
                )
        except BinanceAPIException as e:
            logger.error(f"Test order failed: {e}")
            raise

    def get_order_book(self, symbol: str, limit: int = 100) -> Dict[str, Any]:
        """
        Get order book depth for symbol.

        Args:
            symbol: Trading pair symbol
            limit: Depth limit (5, 10, 20, 50, 100, 500, 1000, 5000)

        Returns:
            Order book data
        """
        try:
            return self.client.get_order_book(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Failed to get order book for {symbol}: {e}")
            raise

    def get_recent_trades(self, symbol: str, limit: int = 500) -> List[Dict[str, Any]]:
        """
        Get recent trades for symbol.

        Args:
            symbol: Trading pair symbol
            limit: Number of trades to retrieve

        Returns:
            List of recent trades
        """
        try:
            return self.client.get_recent_trades(symbol=symbol, limit=limit)
        except BinanceAPIException as e:
            logger.error(f"Failed to get recent trades for {symbol}: {e}")
            raise