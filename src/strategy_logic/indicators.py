"""
Technical indicators module for the trading strategy.

This module implements various technical indicators including RSI and ADX
that are used in the momentum strategy.
"""

import pandas as pd
import numpy as np
import logging
from typing import Union

logger = logging.getLogger(__name__)


class TechnicalIndicators:
    """Class containing implementations of various technical indicators."""

    @staticmethod
    def rsi(data: Union[pd.Series, pd.DataFrame], period: int = 14, column: str = 'close') -> pd.Series:
        """
        Calculate Relative Strength Index (RSI).

        RSI is a momentum oscillator that measures the speed and magnitude
        of price changes. It oscillates between 0 and 100.

        Args:
            data: Price data (Series or DataFrame)
            period: Number of periods for RSI calculation (default: 14)
            column: Column name if data is DataFrame (default: 'close')

        Returns:
            Series with RSI values
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data

        if len(prices) < period + 1:
            logger.warning(f"Insufficient data for RSI calculation. Need at least {period + 1} periods.")
            return pd.Series(index=prices.index, dtype=float)

        # Calculate price changes
        delta = prices.diff()

        # Separate gains and losses
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)

        # Calculate average gain and loss using exponential moving average
        avg_gain = gain.ewm(span=period, adjust=False).mean()
        avg_loss = loss.ewm(span=period, adjust=False).mean()

        # Calculate relative strength and RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))

        return rsi

    @staticmethod
    def adx(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average Directional Index (ADX).

        ADX measures the strength of a trend regardless of direction.
        Values above 25 typically indicate a strong trend.

        Args:
            data: OHLC data DataFrame
            period: Number of periods for ADX calculation (default: 14)

        Returns:
            Series with ADX values
        """
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        if len(data) < period + 1:
            logger.warning(f"Insufficient data for ADX calculation. Need at least {period + 1} periods.")
            return pd.Series(index=data.index, dtype=float)

        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        # Calculate True Range (TR)
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate Directional Movement (+DM and -DM)
        high_diff = high - high.shift(1)
        low_diff = low.shift(1) - low

        plus_dm = np.where((high_diff > low_diff) & (high_diff > 0), high_diff, 0)
        minus_dm = np.where((low_diff > high_diff) & (low_diff > 0), low_diff, 0)

        plus_dm = pd.Series(plus_dm, index=data.index)
        minus_dm = pd.Series(minus_dm, index=data.index)

        # Calculate smoothed averages
        atr = true_range.ewm(span=period, adjust=False).mean()
        plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
        minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.ewm(span=period, adjust=False).mean()

        return adx

    @staticmethod
    def sma(data: Union[pd.Series, pd.DataFrame], period: int, column: str = 'close') -> pd.Series:
        """
        Calculate Simple Moving Average (SMA).

        Args:
            data: Price data (Series or DataFrame)
            period: Number of periods for SMA calculation
            column: Column name if data is DataFrame (default: 'close')

        Returns:
            Series with SMA values
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data

        return prices.rolling(window=period).mean()

    @staticmethod
    def ema(data: Union[pd.Series, pd.DataFrame], period: int, column: str = 'close') -> pd.Series:
        """
        Calculate Exponential Moving Average (EMA).

        Args:
            data: Price data (Series or DataFrame)
            period: Number of periods for EMA calculation
            column: Column name if data is DataFrame (default: 'close')

        Returns:
            Series with EMA values
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data

        return prices.ewm(span=period, adjust=False).mean()

    @staticmethod
    def atr(data: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        Calculate Average True Range (ATR).

        ATR measures market volatility by analyzing the range of price movement.

        Args:
            data: OHLC data DataFrame
            period: Number of periods for ATR calculation (default: 14)

        Returns:
            Series with ATR values
        """
        required_columns = ['high', 'low', 'close']
        if not all(col in data.columns for col in required_columns):
            raise ValueError(f"DataFrame must contain columns: {required_columns}")

        high = data['high']
        low = data['low']
        close = data['close']
        prev_close = close.shift(1)

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - prev_close)
        tr3 = abs(low - prev_close)
        true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR
        atr = true_range.ewm(span=period, adjust=False).mean()

        return atr

    @staticmethod
    def bollinger_bands(data: Union[pd.Series, pd.DataFrame], period: int = 20,
                       std_dev: float = 2, column: str = 'close') -> dict:
        """
        Calculate Bollinger Bands.

        Args:
            data: Price data (Series or DataFrame)
            period: Number of periods for calculation (default: 20)
            std_dev: Number of standard deviations (default: 2)
            column: Column name if data is DataFrame (default: 'close')

        Returns:
            Dictionary with 'upper', 'middle', and 'lower' bands
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data

        middle = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)

        return {
            'upper': upper,
            'middle': middle,
            'lower': lower
        }

    @staticmethod
    def macd(data: Union[pd.Series, pd.DataFrame], fast_period: int = 12,
             slow_period: int = 26, signal_period: int = 9, column: str = 'close') -> dict:
        """
        Calculate MACD (Moving Average Convergence Divergence).

        Args:
            data: Price data (Series or DataFrame)
            fast_period: Fast EMA period (default: 12)
            slow_period: Slow EMA period (default: 26)
            signal_period: Signal line EMA period (default: 9)
            column: Column name if data is DataFrame (default: 'close')

        Returns:
            Dictionary with 'macd', 'signal', and 'histogram'
        """
        if isinstance(data, pd.DataFrame):
            prices = data[column]
        else:
            prices = data

        fast_ema = prices.ewm(span=fast_period, adjust=False).mean()
        slow_ema = prices.ewm(span=slow_period, adjust=False).mean()

        macd_line = fast_ema - slow_ema
        signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
        histogram = macd_line - signal_line

        return {
            'macd': macd_line,
            'signal': signal_line,
            'histogram': histogram
        }


def add_all_indicators(data: pd.DataFrame, rsi_period: int = 14,
                      adx_period: int = 14) -> pd.DataFrame:
    """
    Add all relevant technical indicators to the dataset.

    Args:
        data: OHLC data DataFrame
        rsi_period: RSI calculation period
        adx_period: ADX calculation period

    Returns:
        DataFrame with added technical indicators
    """
    df = data.copy()

    # Add technical indicators
    df['rsi'] = TechnicalIndicators.rsi(df, period=rsi_period)
    df['adx'] = TechnicalIndicators.adx(df, period=adx_period)
    df['sma_20'] = TechnicalIndicators.sma(df, period=20)
    df['ema_20'] = TechnicalIndicators.ema(df, period=20)
    df['atr'] = TechnicalIndicators.atr(df, period=14)

    # Add Bollinger Bands
    bb = TechnicalIndicators.bollinger_bands(df)
    df['bb_upper'] = bb['upper']
    df['bb_middle'] = bb['middle']
    df['bb_lower'] = bb['lower']

    # Add MACD
    macd = TechnicalIndicators.macd(df)
    df['macd'] = macd['macd']
    df['macd_signal'] = macd['signal']
    df['macd_histogram'] = macd['histogram']

    return df