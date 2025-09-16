"""
RSI-ADX Momentum Strategy implementation.

This module implements the core trading strategy logic including
entry and exit signals based on RSI and ADX indicators.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Optional
from .indicators import TechnicalIndicators

logger = logging.getLogger(__name__)


class RSIADXStrategy:
    """
    RSI-ADX Momentum Trading Strategy.

    Strategy Rules:
    - Long entry: RSI crosses above 50 AND ADX > 25
    - Exit: RSI crosses below 50 OR stop-loss/take-profit hit
    """

    def __init__(self, strategy_config):
        """
        Initialize strategy with configuration parameters.

        Args:
            strategy_config: StrategyConfig object with parameters
        """
        self.config = strategy_config
        self.data = None
        self.signals = None

    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate technical indicators for the strategy.

        Args:
            data: OHLC data DataFrame

        Returns:
            DataFrame with added technical indicators
        """
        df = data.copy()

        # Calculate RSI
        df['rsi'] = TechnicalIndicators.rsi(
            df, period=self.config.rsi_period, column='close'
        )

        # Calculate ADX
        df['adx'] = TechnicalIndicators.adx(
            df, period=self.config.adx_period
        )

        # Calculate previous RSI for crossover detection
        df['rsi_prev'] = df['rsi'].shift(1)

        logger.info("Technical indicators calculated successfully")
        return df

    def generate_signals(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Generate trading signals based on strategy rules.

        Args:
            data: DataFrame with OHLC data and indicators

        Returns:
            DataFrame with trading signals
        """
        df = data.copy()

        # Initialize signal columns
        df['signal'] = 0  # 0: no signal, 1: buy, -1: sell
        df['position'] = 0  # 0: no position, 1: long position
        df['entry_signal'] = False
        df['exit_signal'] = False

        # Entry conditions: RSI crosses above 50 AND ADX > 25
        rsi_cross_up = (
            (df['rsi'] > self.config.rsi_entry_threshold) &
            (df['rsi_prev'] <= self.config.rsi_entry_threshold)
        )
        adx_strong = df['adx'] > self.config.adx_minimum_threshold

        df['entry_signal'] = rsi_cross_up & adx_strong

        # Exit conditions: RSI crosses below 50
        rsi_cross_down = (
            (df['rsi'] < self.config.rsi_exit_threshold) &
            (df['rsi_prev'] >= self.config.rsi_exit_threshold)
        )

        df['exit_signal'] = rsi_cross_down

        # Generate position signals
        position = 0
        for i in range(len(df)):
            if df.iloc[i]['entry_signal'] and position == 0:
                df.iloc[i, df.columns.get_loc('signal')] = 1  # Buy signal
                position = 1
            elif df.iloc[i]['exit_signal'] and position == 1:
                df.iloc[i, df.columns.get_loc('signal')] = -1  # Sell signal
                position = 0

            df.iloc[i, df.columns.get_loc('position')] = position

        self.signals = df
        logger.info(f"Generated {(df['signal'] != 0).sum()} trading signals")

        return df

    def get_entry_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Get boolean series indicating entry signals.

        Args:
            data: DataFrame with signals

        Returns:
            Boolean Series for entry signals
        """
        return data['signal'] == 1

    def get_exit_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Get boolean series indicating exit signals.

        Args:
            data: DataFrame with signals

        Returns:
            Boolean Series for exit signals
        """
        return data['signal'] == -1

    def analyze_signals(self, data: pd.DataFrame) -> Dict:
        """
        Analyze generated signals and provide statistics.

        Args:
            data: DataFrame with signals

        Returns:
            Dictionary with signal analysis
        """
        entry_signals = self.get_entry_signals(data)
        exit_signals = self.get_exit_signals(data)

        total_entries = entry_signals.sum()
        total_exits = exit_signals.sum()

        # Calculate time in market
        time_in_market = (data['position'] == 1).sum()
        total_periods = len(data)

        analysis = {
            'total_entry_signals': int(total_entries),
            'total_exit_signals': int(total_exits),
            'time_in_market_periods': int(time_in_market),
            'time_in_market_percentage': round(time_in_market / total_periods * 100, 2),
            'signal_frequency': {
                'entries_per_year': round(total_entries * 252 / total_periods, 2),
                'avg_holding_period': round(time_in_market / max(total_entries, 1), 2) if total_entries > 0 else 0
            }
        }

        # Get signal dates for review
        entry_dates = data[entry_signals].index.tolist()
        exit_dates = data[exit_signals].index.tolist()

        analysis['signal_dates'] = {
            'entries': [str(date.date()) for date in entry_dates[:10]],  # First 10
            'exits': [str(date.date()) for date in exit_dates[:10]]      # First 10
        }

        return analysis

    def validate_strategy_parameters(self) -> bool:
        """
        Validate strategy parameters for logical consistency.

        Returns:
            True if parameters are valid, False otherwise
        """
        errors = []

        # Check RSI parameters
        if not (2 <= self.config.rsi_period <= 50):
            errors.append("RSI period should be between 2 and 50")

        if not (0 <= self.config.rsi_entry_threshold <= 100):
            errors.append("RSI entry threshold should be between 0 and 100")

        if not (0 <= self.config.rsi_exit_threshold <= 100):
            errors.append("RSI exit threshold should be between 0 and 100")

        # Check ADX parameters
        if not (2 <= self.config.adx_period <= 50):
            errors.append("ADX period should be between 2 and 50")

        if not (0 <= self.config.adx_minimum_threshold <= 100):
            errors.append("ADX minimum threshold should be between 0 and 100")

        # Check risk management parameters
        if not (0 < self.config.stop_loss_percentage <= 1):
            errors.append("Stop loss percentage should be between 0 and 1")

        if not (0 < self.config.take_profit_percentage <= 1):
            errors.append("Take profit percentage should be between 0 and 1")

        if not (0 < self.config.position_size <= 1):
            errors.append("Position size should be between 0 and 1")

        if errors:
            for error in errors:
                logger.error(f"Parameter validation error: {error}")
            return False

        logger.info("All strategy parameters validated successfully")
        return True

    def get_strategy_summary(self) -> Dict:
        """
        Get a summary of the strategy configuration.

        Returns:
            Dictionary with strategy summary
        """
        return {
            'strategy_name': 'RSI-ADX Momentum Strategy',
            'parameters': {
                'rsi_period': self.config.rsi_period,
                'rsi_entry_threshold': self.config.rsi_entry_threshold,
                'rsi_exit_threshold': self.config.rsi_exit_threshold,
                'adx_period': self.config.adx_period,
                'adx_minimum_threshold': self.config.adx_minimum_threshold,
                'stop_loss_percentage': self.config.stop_loss_percentage,
                'take_profit_percentage': self.config.take_profit_percentage,
                'position_size': self.config.position_size
            },
            'entry_rules': [
                f"RSI crosses above {self.config.rsi_entry_threshold}",
                f"ADX > {self.config.adx_minimum_threshold}"
            ],
            'exit_rules': [
                f"RSI crosses below {self.config.rsi_exit_threshold}",
                f"Stop loss: {self.config.stop_loss_percentage * 100}%",
                f"Take profit: {self.config.take_profit_percentage * 100}%"
            ]
        }