"""
Simple sample data generation script without external dependencies.

This script generates realistic sample OHLC data that can be used
for testing the RSI-ADX momentum strategy.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os


def generate_sample_data(symbol: str = "AAPL", days: int = 1000,
                        initial_price: float = 150.0,
                        annual_drift: float = 0.08,
                        annual_volatility: float = 0.25) -> pd.DataFrame:
    """
    Generate sample OHLC data with realistic price movements.

    Args:
        symbol: Stock symbol
        days: Number of trading days to generate
        initial_price: Starting price
        annual_drift: Annual return drift
        annual_volatility: Annual volatility

    Returns:
        DataFrame with OHLC data
    """
    np.random.seed(42)  # For reproducible results

    # Generate date range (business days only)
    end_date = datetime(2023, 12, 31)
    start_date = end_date - timedelta(days=int(days * 1.4))  # Account for weekends

    # Create business day range
    dates = []
    current_date = start_date
    while len(dates) < days and current_date <= end_date:
        if current_date.weekday() < 5:  # Monday=0, Friday=4
            dates.append(current_date)
        current_date += timedelta(days=1)

    dates = dates[:days]  # Ensure exact number of days

    # Parameters for realistic price movements
    daily_drift = annual_drift / 252
    daily_vol = annual_volatility / np.sqrt(252)

    # Generate daily returns with some realistic patterns
    returns = np.random.normal(daily_drift, daily_vol, len(dates))

    # Add momentum effect (small autocorrelation)
    for i in range(1, len(returns)):
        returns[i] += 0.05 * returns[i-1]

    # Add occasional volatility clusters
    for i in range(50, len(returns)):
        if abs(returns[i-1]) > 2 * daily_vol:
            returns[i] *= 1.5

    # Calculate prices using geometric Brownian motion
    log_prices = np.cumsum(returns)
    prices = initial_price * np.exp(log_prices)

    # Generate OHLC data
    data = []
    for i, (date, close) in enumerate(zip(dates, prices)):
        # Generate realistic intraday range
        daily_range_pct = np.random.uniform(0.008, 0.035)  # 0.8% to 3.5% daily range
        daily_range = close * daily_range_pct

        # Generate high and low around close
        high_offset = np.random.uniform(0.3, 0.8) * daily_range
        low_offset = np.random.uniform(0.3, 0.8) * daily_range

        high = close + high_offset
        low = close - low_offset

        # Generate open (with small gap from previous close)
        if i == 0:
            open_price = close + np.random.normal(0, close * 0.005)
        else:
            prev_close = data[i-1]['close']
            gap = np.random.normal(0, prev_close * 0.003)
            open_price = prev_close + gap

        # Ensure logical consistency: high >= max(open, close), low <= min(open, close)
        high = max(high, open_price, close)
        low = min(low, open_price, close)

        # Generate volume (correlated with price volatility)
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

    # Create DataFrame with date index
    df = pd.DataFrame(data)
    df.index = pd.DatetimeIndex(dates)
    df.index.name = 'Date'

    return df


def main():
    """Generate sample data files."""
    print("Generating sample market data...")

    # Create data directory if it doesn't exist
    data_dir = '/Users/alexalonso/trading-bot-rsi/data'
    os.makedirs(data_dir, exist_ok=True)

    # Generate AAPL data (main test data)
    print("Generating AAPL data...")
    aapl_data = generate_sample_data(
        symbol="AAPL",
        days=1000,
        initial_price=150.0,
        annual_drift=0.08,
        annual_volatility=0.25
    )
    aapl_data.to_csv(f'{data_dir}/AAPL.csv')
    print(f"  AAPL: {len(aapl_data)} days, {aapl_data.index[0].date()} to {aapl_data.index[-1].date()}")
    print(f"  Price range: ${aapl_data['close'].min():.2f} - ${aapl_data['close'].max():.2f}")

    # Generate additional symbols for testing
    symbols_config = {
        'MSFT': {'initial_price': 300.0, 'drift': 0.12, 'vol': 0.22},
        'TSLA': {'initial_price': 200.0, 'drift': 0.15, 'vol': 0.45},
        'SPY': {'initial_price': 400.0, 'drift': 0.07, 'vol': 0.18},
        'QQQ': {'initial_price': 350.0, 'drift': 0.10, 'vol': 0.20}
    }

    for symbol, config in symbols_config.items():
        print(f"Generating {symbol} data...")

        # Use different seed for each symbol
        np.random.seed(42 + len(symbol))

        data = generate_sample_data(
            symbol=symbol,
            days=1000,
            initial_price=config['initial_price'],
            annual_drift=config['drift'],
            annual_volatility=config['vol']
        )

        data.to_csv(f'{data_dir}/{symbol}.csv')
        print(f"  {symbol}: ${data['close'].iloc[0]:.2f} -> ${data['close'].iloc[-1]:.2f}")

    print("\nSample data generation completed!")
    print(f"\nGenerated files in {data_dir}:")

    # List generated files
    for file in sorted(os.listdir(data_dir)):
        if file.endswith('.csv'):
            file_path = os.path.join(data_dir, file)
            file_size = os.path.getsize(file_path) / 1024  # KB
            print(f"  - {file} ({file_size:.1f} KB)")

    # Show sample of AAPL data
    print(f"\nSample of AAPL data (first 5 rows):")
    print(aapl_data.head())

    print(f"\nStatistics for AAPL data:")
    print(f"  Total Return: {(aapl_data['close'].iloc[-1] / aapl_data['close'].iloc[0] - 1) * 100:.1f}%")
    print(f"  Max Price: ${aapl_data['close'].max():.2f}")
    print(f"  Min Price: ${aapl_data['close'].min():.2f}")
    print(f"  Avg Volume: {aapl_data['volume'].mean():,.0f}")


if __name__ == "__main__":
    main()