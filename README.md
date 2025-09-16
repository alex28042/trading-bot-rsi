# RSI-ADX Momentum Trading Strategy

A professional-grade algorithmic trading system implementing a momentum strategy based on RSI (Relative Strength Index) and ADX (Average Directional Index) indicators with comprehensive backtesting and performance analysis capabilities. Enhanced with full cryptocurrency trading support via Binance API integration.

## Strategy Overview

### Trading Rules
- **Long Entry**: RSI crosses above 50 AND ADX > 25
- **Exit**: RSI crosses below 50 OR stop-loss/take-profit is hit
- **Risk Management**: Configurable stop-loss and take-profit percentages
- **Position Sizing**: Fixed percentage allocation of portfolio

### Key Features
- ✅ **Modular Architecture**: Clean separation of concerns with dedicated modules
- ✅ **Comprehensive Risk Management**: Stop-loss, take-profit, and position sizing
- ✅ **Advanced Performance Metrics**: Sharpe ratio, maximum drawdown, profit factor, and more
- ✅ **Professional Visualizations**: Equity curves, drawdown analysis, trade statistics
- ✅ **Multiple Data Sources**: CSV files, Yahoo Finance, and Binance API integration
- ✅ **Cryptocurrency Trading**: Full BTC/USDT support with real-time Binance data
- ✅ **BTC Position Sizing**: Trade in BTC fractions with crypto-specific risk management
- ✅ **Multi-Timeframe Support**: 1h, 4h, 1d timeframes for cryptocurrency trading
- ✅ **Configurable Parameters**: Easy customization of all strategy parameters
- ✅ **Detailed Logging**: Comprehensive logging and error handling

## Project Structure

```
trading-bot-rsi/
├── config/                     # Configuration management
│   ├── __init__.py
│   └── settings.py            # Strategy and system parameters
├── src/                       # Source code modules
│   ├── data_ingestion/        # Data loading and preprocessing
│   │   ├── __init__.py
│   │   └── data_loader.py
│   ├── strategy_logic/        # Technical indicators and trading signals
│   │   ├── __init__.py
│   │   ├── indicators.py      # RSI, ADX, and other indicators
│   │   └── strategy.py        # Strategy implementation
│   ├── risk_management/       # Position sizing and risk controls
│   │   ├── __init__.py
│   │   └── risk_manager.py
│   ├── backtesting_engine/    # Backtesting simulation
│   │   ├── __init__.py
│   │   └── backtest.py
│   ├── performance_reporting/ # Analysis and visualization
│   │   ├── __init__.py
│   │   └── reporter.py
│   └── binance_connector/     # Binance API integration
│       ├── __init__.py
│       ├── binance_client.py  # API client wrapper
│       └── data_fetcher.py    # Crypto data fetching
├── data/                      # Market data files
│   ├── generate_sample_data.py
│   └── *.csv                  # OHLC data files
├── results/                   # Output directory for results
├── logs/                      # Log files
├── main.py                    # Main execution script
├── crypto_backtest_example.py # Cryptocurrency trading example
├── requirements.txt           # Python dependencies
├── setup.py                   # Installation script
└── README.md                  # This file
```

## Installation

### Prerequisites
- Python 3.8 or higher
- pip (Python package manager)

### Quick Setup

1. **Clone or download the project**:
   ```bash
   cd trading-bot-rsi
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv trading_env
   source trading_env/bin/activate  # On Windows: trading_env\\Scripts\\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Generate sample data** (for testing):
   ```bash
   python data/generate_sample_data.py
   ```

### Alternative Installation
You can also install the package in development mode:
```bash
pip install -e .
```

## Usage

### Cryptocurrency Trading (Default)

Run a BTC/USDT backtest with Binance data (default configuration):
```bash
python main.py
```

### Quick Cryptocurrency Examples

```bash
# BTC/USDT with 1-hour timeframe
python main.py --symbol BTCUSDT --timeframe 1h

# Ethereum trading with custom position size
python main.py --symbol ETHUSDT --btc-position-size 1.0

# Different crypto pairs
python main.py --symbol ADAUSDT --timeframe 4h
python main.py --symbol DOTUSDT --timeframe 1d
```

### Stock Trading

Run traditional stock backtests:
```bash
# Apple stock with Yahoo Finance data
python main.py --symbol AAPL --data-source yahoo

# Custom stock with CSV data
python main.py --symbol MSFT --data-source csv
```

### Comprehensive Cryptocurrency Example

For a complete cryptocurrency trading demonstration:
```bash
python crypto_backtest_example.py
```

### Advanced Usage Examples

**Test different symbols:**
```bash
python main.py --symbol MSFT
python main.py --symbol TSLA
```

**Customize date range:**
```bash
python main.py --symbol AAPL --start-date 2022-01-01 --end-date 2023-12-31
```

**Adjust strategy parameters:**
```bash
python main.py --rsi-period 21 --adx-period 20 --rsi-entry 55 --adx-min 30
```

**Modify risk management:**
```bash
python main.py --stop-loss 0.03 --take-profit 0.06 --position-size 0.15
```

**Custom capital and output:**
```bash
python main.py --initial-capital 250000 --output-dir custom_results
```

**Skip chart generation:**
```bash
python main.py --no-plots
```

### Command Line Options

| Parameter | Description | Default |
|-----------|-------------|---------|
| `--symbol` | Symbol to backtest (crypto/stock) | BTCUSDT |
| `--data-source` | Data source (binance/yahoo/csv) | binance |
| `--timeframe` | Crypto timeframe (1h/4h/1d) | 4h |
| `--crypto-mode` | Enable crypto trading mode | Auto-detected |
| `--start-date` | Start date (YYYY-MM-DD) | 2020-01-01 |
| `--end-date` | End date (YYYY-MM-DD) | 2023-12-31 |
| `--rsi-period` | RSI calculation period | 14 |
| `--adx-period` | ADX calculation period | 14 |
| `--rsi-entry` | RSI entry threshold | 50.0 |
| `--adx-min` | Minimum ADX for trend | 25.0 |
| `--stop-loss` | Stop loss percentage | 0.02 (2%) |
| `--take-profit` | Take profit percentage | 0.04 (4%) |
| `--position-size` | Position size fraction | 0.10 (10%) |
| `--btc-position-size` | BTC position size for crypto | 0.1 BTC |
| `--initial-capital` | Starting capital | 100000 |
| `--output-dir` | Results directory | results |
| `--no-plots` | Skip chart generation | False |
| `--verbose` | Enable debug logging | False |

## Sample Data

The project includes a data generation script that creates realistic sample data for testing:

**Generated symbols:**
- **AAPL**: Apple Inc. (primary test data)
- **MSFT**: Microsoft Corporation
- **TSLA**: Tesla Inc.
- **SPY**: SPDR S&P 500 ETF
- **QQQ**: Invesco QQQ ETF

**Data characteristics:**
- 1000 trading days (approximately 4 years)
- Realistic price movements with momentum and volatility clustering
- Proper OHLC relationships
- Volume correlated with price volatility

## Output and Results

### Console Output
The system provides real-time progress updates and a comprehensive summary including:
- Strategy performance metrics
- Trade statistics
- Risk analysis

### Generated Files
Results are saved to the specified output directory:

- `backtest_SYMBOL_TIMESTAMP_summary.txt` - Text summary report
- `backtest_SYMBOL_TIMESTAMP_trades.csv` - Detailed trade log
- `backtest_SYMBOL_TIMESTAMP_equity_curve.csv` - Portfolio value history
- `equity_curve.png` - Portfolio performance chart
- `drawdown_analysis.png` - Drawdown visualization
- `monthly_returns_heatmap.png` - Monthly performance heatmap
- `trade_analysis.png` - Trade distribution analysis
- `risk_metrics.png` - Risk analysis charts
- `performance_comparison.png` - Strategy vs. buy-and-hold

### Key Performance Metrics

**Return Metrics:**
- Total Return
- Annualized Return
- Sharpe Ratio
- Sortino Ratio
- Calmar Ratio

**Risk Metrics:**
- Maximum Drawdown
- Volatility
- Value at Risk (VaR)
- Conditional VaR

**Trade Metrics:**
- Total Trades
- Win Rate
- Profit Factor
- Average P&L per trade
- Best/Worst trades

## Configuration

### Strategy Parameters (config/settings.py)

```python
@dataclass
class StrategyConfig:
    # Technical indicators
    rsi_period: int = 14
    adx_period: int = 14

    # Entry/exit thresholds
    rsi_entry_threshold: float = 50.0
    rsi_exit_threshold: float = 50.0
    adx_minimum_threshold: float = 25.0

    # Risk management
    stop_loss_percentage: float = 0.02  # 2%
    take_profit_percentage: float = 0.04  # 4%
    position_size: float = 0.10  # 10%

    # Capital and costs
    initial_capital: float = 100000.0
    commission_per_trade: float = 5.0
```

## Cryptocurrency Trading

### Binance Integration

The system features comprehensive Binance API integration for cryptocurrency trading:

- **Real-time Data**: Fetch live OHLCV data from Binance
- **Multiple Timeframes**: Support for 1h, 4h, and 1d candles
- **No API Keys Required**: Public data access for backtesting
- **Professional Error Handling**: Robust connection management

### Supported Cryptocurrency Pairs

Popular trading pairs automatically supported:
- **BTC/USDT**: Bitcoin (default)
- **ETH/USDT**: Ethereum
- **BNB/USDT**: Binance Coin
- **ADA/USDT**: Cardano
- **DOT/USDT**: Polkadot
- **MATIC/USDT**: Polygon
- **SOL/USDT**: Solana

### BTC Position Sizing

Unlike traditional percentage-based position sizing, the crypto mode uses fixed BTC amounts:

```python
# Traditional: 10% of portfolio
position_size = 0.10  # 10% of $100k = $10k

# Crypto: Fixed BTC amount
btc_position_size = 0.1  # Always trade 0.1 BTC regardless of price
```

This approach provides:
- **Consistent exposure** in BTC terms
- **Simplified risk management**
- **Better comparison** across different price levels

### Crypto-Specific Risk Management

Enhanced risk management for cryptocurrency volatility:

```python
crypto_stop_loss = 2%      # Tighter stops for volatile markets
crypto_take_profit = 4%    # Higher profit targets
trading_fees = 0.1%        # Binance spot trading fees
```

### Data Sources
The system supports multiple data sources:
- **Binance API**: Real-time cryptocurrency data (default)
- **CSV files**: Local market data files
- **Yahoo Finance**: Real-time data download (requires yfinance)
- **Alpha Vantage**: API integration (requires API key)

## Technical Indicators

### RSI (Relative Strength Index)
- **Purpose**: Momentum oscillator (0-100)
- **Signal**: Values > 50 indicate upward momentum
- **Implementation**: Exponential moving average of gains/losses

### ADX (Average Directional Index)
- **Purpose**: Trend strength indicator (0-100)
- **Signal**: Values > 25 indicate strong trend
- **Implementation**: Smoothed directional movement indicators

### Additional Indicators
The system also calculates (for analysis):
- Simple/Exponential Moving Averages
- Bollinger Bands
- MACD
- Average True Range (ATR)

## Strategy Logic

### Entry Conditions
1. **RSI Crossover**: RSI crosses above the entry threshold (default: 50)
2. **Trend Strength**: ADX is above minimum threshold (default: 25)
3. **Position Availability**: No existing open position

### Exit Conditions
1. **RSI Crossover**: RSI crosses below the exit threshold (default: 50)
2. **Stop Loss**: Price drops below stop-loss level
3. **Take Profit**: Price rises above take-profit level

### Risk Management
- **Position Sizing**: Fixed percentage of portfolio value
- **Stop Loss**: Percentage-based protective stop
- **Take Profit**: Percentage-based profit target
- **Commission Costs**: Realistic transaction costs included

## Development and Testing

### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-cov

# Run tests (when test suite is available)
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Code Quality
```bash
# Install development tools
pip install black flake8 mypy

# Format code
black src/ config/ main.py

# Check linting
flake8 src/ config/ main.py

# Type checking
mypy src/ config/ main.py
```

### Adding New Indicators
1. Implement indicator in `src/strategy_logic/indicators.py`
2. Add to strategy logic in `src/strategy_logic/strategy.py`
3. Update configuration parameters if needed

### Adding New Data Sources
1. Extend `DataLoader` class in `src/data_ingestion/data_loader.py`
2. Add configuration parameters in `config/settings.py`
3. Update main script to handle new source

## Performance Considerations

### Optimization Tips
1. **Data Efficiency**: Use vectorized pandas operations
2. **Memory Management**: Process data in chunks for large datasets
3. **Caching**: Cache calculated indicators to avoid recomputation
4. **Parallel Processing**: Consider multiprocessing for multiple symbols

### Scalability
- The system can handle datasets with thousands of data points
- Memory usage scales linearly with data size
- Computational complexity is O(n) for most operations

## Limitations and Considerations

### Strategy Limitations
- **Look-ahead Bias**: Indicators use only historical data
- **Transaction Costs**: Fixed commission model (not percentage-based)
- **Slippage**: Not explicitly modeled (assumes fill at exact prices)
- **Market Hours**: Assumes continuous trading
- **Survivorship Bias**: Sample data doesn't account for delisted stocks

### Technical Limitations
- **Data Quality**: Depends on input data accuracy
- **Real-time Trading**: Not designed for live trading (backtesting only)
- **Multiple Assets**: Currently handles one symbol at a time
- **Short Selling**: Only long positions implemented

## Live Trading Preparation

### Binance API Setup (Optional)

The system is designed for easy transition to live trading:

1. **Create Binance Account**: Sign up at binance.com
2. **Generate API Keys**: Enable spot trading permissions
3. **Configure API Access**:
   ```python
   config.data.binance_api_key = "your_api_key"
   config.data.binance_api_secret = "your_secret"
   config.data.binance_testnet = True  # Use testnet first
   ```

### Paper Trading Ready

The infrastructure supports paper trading with:
- **Test Orders**: Validate order placement without real execution
- **Portfolio Tracking**: Monitor positions and balances
- **Risk Controls**: Position limits and exposure management
- **Performance Monitoring**: Real-time P&L tracking

### Live Trading Considerations

Before going live:
1. **Extensive Backtesting**: Validate strategy performance
2. **Paper Trading**: Test with live data, simulated execution
3. **Risk Management**: Set maximum position sizes and drawdown limits
4. **Monitoring**: Implement alerts and logging
5. **Gradual Scaling**: Start small and increase position sizes gradually

## Future Enhancements

### Potential Improvements
1. **Multi-asset Backtesting**: Portfolio-level strategies
2. **Walk-forward Analysis**: Rolling optimization windows
3. **Monte Carlo Simulation**: Robust statistical testing
4. **Advanced Order Types**: Stop-limit, OCO, and trailing stops
5. **Machine Learning**: Adaptive parameters
6. **Alternative Risk Models**: VaR-based position sizing

### Contributing
To contribute to this project:
1. Fork the repository
2. Create a feature branch
3. Implement changes with tests
4. Submit a pull request

## License

This project is licensed under the MIT License. See LICENSE file for details.

## Disclaimer

**Important**: This software is for educational and research purposes only. It is not intended for live trading or investment advice.

**Cryptocurrency Warning**: Cryptocurrency trading involves substantial risk of loss and is not suitable for all investors. Digital assets are highly volatile and can fluctuate significantly in value. Only trade with funds you can afford to lose.

**Risk Disclosure**:
- Always conduct thorough backtesting and paper trading before live implementation
- Past performance does not guarantee future results
- Market conditions change and strategies may become ineffective
- Consult with qualified financial professionals before making investment decisions
- Start with small position sizes when transitioning to live trading

**API Security**: When using live trading APIs, protect your credentials and use appropriate security measures including IP restrictions and withdrawal limitations.

## Support

For questions, issues, or feature requests:
- Create an issue on the project repository
- Review the code documentation
- Check the logs directory for detailed execution information

## Acknowledgments

This project implements established financial indicators and backtesting methodologies from academic and industry sources. The RSI indicator was developed by J. Welles Wilder Jr., and the ADX indicator is also attributed to Wilder's work in technical analysis.