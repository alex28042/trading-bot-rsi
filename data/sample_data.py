"""
Sample data generation script for the trading system.

This script generates realistic sample OHLC data that can be used
for testing the RSI-ADX momentum strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import sys
import os

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data_ingestion.data_loader import create_sample_data


def generate_aapl_sample_data(days: int = 1000, save_to_csv: bool = True) -> pd.DataFrame:
    """
    Generate sample AAPL-like data with realistic price movements.

    Args:
        days: Number of trading days to generate
        save_to_csv: Whether to save data to CSV file

    Returns:
        DataFrame with OHLC data
    """
    np.random.seed(42)  # For reproducible results

    # Generate date range (business days only)
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=int(days * 1.4))  # Account for weekends
    dates = pd.bdate_range(start=start_date, end=end_date)[:days]

    # AAPL-like parameters
    initial_price = 150.0
    annual_drift = 0.08  # 8% annual growth
    annual_volatility = 0.25  # 25% annual volatility

    # Generate daily returns with some realistic patterns
    daily_drift = annual_drift / 252
    daily_vol = annual_volatility / np.sqrt(252)

    # Generate correlated returns (momentum and mean reversion)
    returns = np.random.normal(daily_drift, daily_vol, len(dates))

    # Add momentum effect
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]  # Small momentum

    # Add occasional volatility clusters
    for i in range(50, len(returns)):
        if abs(returns[i-1]) > 2 * daily_vol:
            returns[i] *= 1.5  # Increase volatility after large moves

    # Calculate prices
    log_prices = np.cumsum(returns)
    prices = initial_price * np.exp(log_prices)

    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic intraday data
        daily_range_pct = np.random.uniform(0.008, 0.035)  # 0.8% to 3.5% daily range
        daily_range = close * daily_range_pct

        # Generate high and low
        high_offset = np.random.uniform(0.3, 0.8) * daily_range
        low_offset = np.random.uniform(0.3, 0.8) * daily_range

        high = close + high_offset
        low = close - low_offset

        # Generate open (biased towards previous close for realistic gaps)
        if i == 0:
            open_price = close + np.random.normal(0, close * 0.005)
        else:
            prev_close = data[i-1]['close']
            gap = np.random.normal(0, prev_close * 0.003)
            open_price = prev_close + gap

        # Ensure logical consistency
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume (higher volume on larger price moves)
        price_change_pct = abs(returns[i]) if i > 0 else 0
        base_volume = 50_000_000  # 50M base volume
        volume_multiplier = 1 + (price_change_pct * 10)  # Higher volume on big moves
        volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))

        data.append({
            'open': round(open_price, 2),
            'high': round(high, 2),
            'low': round(low, 2),
            'close': round(close, 2),
            'volume': volume
        })

    df = pd.DataFrame(data, index=dates)

    if save_to_csv:
        df.to_csv('/Users/alexalonso/trading-bot-rsi/data/AAPL.csv')
        print(f"Sample AAPL data saved to data/AAPL.csv")
        print(f"Data range: {df.index[0].date()} to {df.index[-1].date()}")
        print(f"Total trading days: {len(df)}")
        print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    return df


def generate_multiple_symbols_data():
    """Generate sample data for multiple symbols."""
    symbols = {
        'AAPL': {'initial_price': 150.0, 'drift': 0.08, 'vol': 0.25},
        'MSFT': {'initial_price': 300.0, 'drift': 0.12, 'vol': 0.22},
        'TSLA': {'initial_price': 200.0, 'drift': 0.15, 'vol': 0.45},
        'SPY': {'initial_price': 400.0, 'drift': 0.07, 'vol': 0.18}
    }

    for symbol, params in symbols.items():
        print(f"Generating data for {symbol}...")

        np.random.seed(42 + len(symbol))  # Different seed for each symbol

        days = 1000
        end_date = datetime(2023, 12, 31)
        start_date = end_date - timedelta(days=int(days * 1.4))
        dates = pd.bdate_range(start=start_date, end=end_date)[:days]

        # Generate returns
        daily_drift = params['drift'] / 252
        daily_vol = params['vol'] / np.sqrt(252)
        returns = np.random.normal(daily_drift, daily_vol, len(dates))

        # Add momentum
        for i in range(1, len(returns)):
            returns[i] += 0.05 * returns[i-1]

        # Calculate prices
        log_prices = np.cumsum(returns)
        prices = params['initial_price'] * np.exp(log_prices)

        # Generate OHLC
        data = []
        for i, (date, close) in enumerate(zip(dates, prices)):
            daily_range_pct = np.random.uniform(0.008, 0.035)
            daily_range = close * daily_range_pct

            high_offset = np.random.uniform(0.3, 0.8) * daily_range
            low_offset = np.random.uniform(0.3, 0.8) * daily_range

            high = close + high_offset
            low = close - low_offset

            if i == 0:
                open_price = close + np.random.normal(0, close * 0.005)
            else:
                prev_close = data[i-1]['close']
                gap = np.random.normal(0, prev_close * 0.003)
                open_price = prev_close + gap

            high = max(high, open_price, close)
            low = min(low, open_price, close)

            base_volume = 30_000_000 if symbol != 'TSLA' else 80_000_000
            volume_multiplier = 1 + (abs(returns[i]) * 10)
            volume = int(base_volume * volume_multiplier * np.random.uniform(0.5, 1.5))

            data.append({
                'open': round(open_price, 2),
                'high': round(high, 2),
                'low': round(low, 2),
                'close': round(close, 2),
                'volume': volume
            })

        df = pd.DataFrame(data, index=dates)
        df.to_csv(f'/Users/alexalonso/trading-bot-rsi/data/{symbol}.csv')
        print(f"  {symbol} data saved: {len(df)} days, ${df['close'].iloc[0]:.2f} -> ${df['close'].iloc[-1]:.2f}")


if __name__ == "__main__":
    print("Generating sample market data...")

    # Generate main AAPL data for testing
    generate_aapl_sample_data(days=1000, save_to_csv=True)

    print("\nGenerating additional symbols...")
    generate_multiple_symbols_data()

    print("\nSample data generation completed!")
    print("\nAvailable data files:")
    data_dir = '/Users/alexalonso/trading-bot-rsi/data'
    for file in os.listdir(data_dir):
        if file.endswith('.csv'):
            print(f"  - {file}")