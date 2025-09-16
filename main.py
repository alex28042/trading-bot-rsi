"""
Main execution script for the RSI-ADX Momentum Trading Strategy.

This script provides the entry point for running backtests and generating
performance reports for the momentum strategy. Enhanced with cryptocurrency
trading support via Binance API integration.

Usage:
    # Default BTC/USDT crypto trading
    python main.py

    # Custom crypto pair with different timeframe
    python main.py --symbol ETHUSDT --timeframe 1h

    # Traditional stock trading
    python main.py --symbol AAPL --data-source yahoo --crypto-mode false

    # Custom parameters
    python main.py --symbol BTCUSDT --start-date 2022-01-01 --btc-position-size 0.2
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from config.settings import config
from src.data_ingestion.data_loader import DataLoader
from src.backtesting_engine.backtest import BacktestEngine
from src.performance_reporting.reporter import PerformanceReporter


def setup_directories():
    """Create necessary directories if they don't exist."""
    directories = ['logs', 'results', 'data']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='RSI-ADX Momentum Strategy Backtesting System',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Data parameters
    parser.add_argument(
        '--symbol', '-s',
        type=str,
        default='BTCUSDT',
        help='Symbol to backtest (e.g., BTCUSDT for crypto, AAPL for stocks)'
    )

    parser.add_argument(
        '--start-date',
        type=str,
        default='2020-01-01',
        help='Start date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--end-date',
        type=str,
        default='2023-12-31',
        help='End date for backtest (YYYY-MM-DD)'
    )

    parser.add_argument(
        '--data-source',
        type=str,
        choices=['csv', 'yahoo', 'binance'],
        default='binance',
        help='Data source for market data'
    )

    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['1h', '4h', '1d'],
        default='4h',
        help='Timeframe for cryptocurrency data (Binance only)'
    )

    parser.add_argument(
        '--crypto-mode',
        action='store_true',
        help='Enable cryptocurrency trading mode with BTC position sizing'
    )

    # Strategy parameters
    parser.add_argument(
        '--rsi-period',
        type=int,
        default=14,
        help='RSI calculation period'
    )

    parser.add_argument(
        '--adx-period',
        type=int,
        default=14,
        help='ADX calculation period'
    )

    parser.add_argument(
        '--rsi-entry',
        type=float,
        default=50.0,
        help='RSI entry threshold'
    )

    parser.add_argument(
        '--adx-min',
        type=float,
        default=25.0,
        help='Minimum ADX for trend strength'
    )

    # Risk management parameters
    parser.add_argument(
        '--stop-loss',
        type=float,
        default=0.02,
        help='Stop loss percentage (0.02 = 2%)'
    )

    parser.add_argument(
        '--take-profit',
        type=float,
        default=0.04,
        help='Take profit percentage (0.04 = 4%)'
    )

    parser.add_argument(
        '--position-size',
        type=float,
        default=0.10,
        help='Position size as fraction of portfolio (0.10 = 10%) or BTC amount for crypto'
    )

    parser.add_argument(
        '--btc-position-size',
        type=float,
        default=0.1,
        help='Position size in BTC for cryptocurrency trading (default: 0.1 BTC)'
    )

    parser.add_argument(
        '--initial-capital',
        type=float,
        default=100000.0,
        help='Initial capital for backtesting'
    )

    # Output parameters
    parser.add_argument(
        '--no-plots',
        action='store_true',
        help='Skip generating plots'
    )

    parser.add_argument(
        '--output-dir',
        type=str,
        default='results',
        help='Output directory for results'
    )

    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )

    return parser.parse_args()


def update_config_from_args(args):
    """Update configuration based on command line arguments."""
    # Update strategy parameters
    config.strategy.symbol = args.symbol
    config.strategy.start_date = args.start_date
    config.strategy.end_date = args.end_date
    config.strategy.rsi_period = args.rsi_period
    config.strategy.adx_period = args.adx_period
    config.strategy.rsi_entry_threshold = args.rsi_entry
    config.strategy.adx_minimum_threshold = args.adx_min
    config.strategy.stop_loss_percentage = args.stop_loss
    config.strategy.take_profit_percentage = args.take_profit
    config.strategy.position_size = args.position_size
    config.strategy.initial_capital = args.initial_capital

    # Update data parameters
    config.data.data_source = args.data_source
    config.data.results_directory = args.output_dir
    config.data.timeframe = args.timeframe

    # Update crypto parameters
    config.crypto.use_btc_position_sizing = args.crypto_mode
    config.crypto.position_size_btc = args.btc_position_size

    # Auto-enable crypto mode for crypto symbols or Binance data source
    if args.symbol.endswith('USDT') or args.data_source == 'binance':
        config.crypto.use_btc_position_sizing = True
        logger.info("Auto-enabled cryptocurrency trading mode")

    # Update backtest parameters
    config.backtest.generate_plots = not args.no_plots
    config.backtest.verbose_logging = args.verbose

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)


def load_market_data(symbol: str, start_date: str, end_date: str):
    """Load market data using the data loader."""
    logger.info(f"Loading market data for {symbol}")

    data_loader = DataLoader(config.data)

    try:
        data = data_loader.load_data(symbol, start_date, end_date)
        logger.info(f"Successfully loaded {len(data)} days of data")

        # Print data summary
        summary = data_loader.get_data_summary(data)
        logger.info(f"Data summary: {summary}")

        return data

    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise


def run_backtest(data):
    """Run the backtest using the backtesting engine."""
    logger.info("Starting backtest execution")

    try:
        # Initialize backtesting engine
        backtest_engine = BacktestEngine(config)

        # Run backtest
        results = backtest_engine.run_backtest(data)

        logger.info("Backtest completed successfully")
        return backtest_engine, results

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        raise


def generate_reports(backtest_engine, results):
    """Generate performance reports and visualizations."""
    logger.info("Generating performance reports")

    try:
        # Initialize performance reporter
        reporter = PerformanceReporter(config)

        # Generate comprehensive report
        report = reporter.generate_full_report(
            results,
            save_plots=config.backtest.generate_plots
        )

        # Print summary to console
        print_performance_summary(backtest_engine)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"{config.data.results_directory}/backtest_{config.strategy.symbol}_{timestamp}"
        backtest_engine.save_results(output_prefix)

        logger.info(f"Reports generated and saved to {config.data.results_directory}")
        return report

    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        raise


def print_performance_summary(backtest_engine):
    """Print a formatted performance summary to console."""
    summary = backtest_engine.get_performance_summary()

    print("\n" + "="*60)
    print("RSI-ADX MOMENTUM STRATEGY BACKTEST RESULTS")
    print("="*60)

    for section, metrics in summary.items():
        print(f"\n{section.upper()}:")
        print("-" * len(section))
        for metric, value in metrics.items():
            print(f"  {metric:.<25} {value}")

    print("\n" + "="*60)


def main():
    """Main execution function."""
    try:
        # Setup
        setup_directories()
        args = parse_arguments()
        update_config_from_args(args)

        logger.info("Starting RSI-ADX Momentum Strategy Backtest")
        logger.info(f"Configuration: {config.to_dict()}")

        # Load data
        data = load_market_data(
            args.symbol,
            args.start_date,
            args.end_date
        )

        # Run backtest
        backtest_engine, results = run_backtest(data)

        # Generate reports
        report = generate_reports(backtest_engine, results)

        logger.info("Backtest execution completed successfully")

        # Display final summary
        print(f"\nBacktest completed for {args.symbol}")
        print(f"Period: {args.start_date} to {args.end_date}")
        print(f"Results saved to: {config.data.results_directory}")

        if config.backtest.generate_plots:
            print("Charts and visualizations have been generated")

        return 0

    except KeyboardInterrupt:
        logger.info("Backtest interrupted by user")
        return 1

    except Exception as e:
        logger.error(f"Backtest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)