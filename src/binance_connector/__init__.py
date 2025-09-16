"""
Binance Connector Module

This module provides integration with Binance API for cryptocurrency data fetching
and paper trading preparation. It supports both REST API and WebSocket connections
for real-time data.
"""

from .binance_client import BinanceClient
from .data_fetcher import BinanceDataFetcher

__all__ = ['BinanceClient', 'BinanceDataFetcher']