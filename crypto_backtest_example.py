#!/usr/bin/env python3
"""
BTC/USDT Cryptocurrency Trading Strategy Example

This script demonstrates how to run a complete backtest for BTC/USDT
using real data from Binance API with the RSI-ADX momentum strategy.

Features:
- Real-time data fetching from Binance
- Cryptocurrency-specific risk management (BTC position sizing)
- Multiple timeframe support (1h, 4h, 1d)
- Crypto-aware performance reporting
"""

import sys
import os
import logging
from datetime import datetime, timedelta

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/crypto_backtest.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

# Import project modules
from config.settings import Config
from src.data_ingestion.data_loader import DataLoader
from src.backtesting_engine.backtest import BacktestEngine
from src.performance_reporting.reporter import PerformanceReporter


def setup_crypto_config():
    """Set up configuration specifically for cryptocurrency trading."""
    config = Config()

    # Crypto-specific data settings
    config.data.data_source = "binance"
    config.data.timeframe = "4h"  # 4-hour timeframe for swing trading

    # Strategy settings optimized for crypto volatility
    config.strategy.symbol = "BTCUSDT"
    config.strategy.rsi_period = 14
    config.strategy.adx_period = 14
    config.strategy.rsi_entry_threshold = 50.0
    config.strategy.adx_minimum_threshold = 25.0

    # Crypto risk management
    config.crypto.use_btc_position_sizing = True
    config.crypto.position_size_btc = 0.1  # Trade 0.1 BTC per signal
    config.crypto.crypto_stop_loss_percentage = 0.02  # 2% stop loss
    config.crypto.crypto_take_profit_percentage = 0.04  # 4% take profit

    # Backtesting parameters
    config.strategy.initial_capital = 100000.0  # $100k starting capital
    config.strategy.start_date = "2023-01-01"
    config.strategy.end_date = "2024-01-01"

    # Output settings
    config.data.results_directory = "results/crypto"
    config.backtest.generate_plots = True
    config.backtest.verbose_logging = True

    return config


def run_crypto_backtest_example():
    """Run a comprehensive cryptocurrency backtest example."""
    print("="*70)
    print("BTC/USDT CRYPTOCURRENCY TRADING STRATEGY BACKTEST")
    print("="*70)
    print()

    try:
        # Setup directories
        os.makedirs('logs', exist_ok=True)
        os.makedirs('results/crypto', exist_ok=True)

        # Initialize configuration
        logger.info("Setting up cryptocurrency trading configuration...")
        config = setup_crypto_config()

        print(f"Strategy Configuration:")
        print(f"  Symbol: {config.strategy.symbol}")
        print(f"  Timeframe: {config.data.timeframe}")
        print(f"  Data Source: {config.data.data_source}")
        print(f"  Period: {config.strategy.start_date} to {config.strategy.end_date}")
        print(f"  Position Size: {config.crypto.position_size_btc} BTC per trade")
        print(f"  Stop Loss: {config.crypto.crypto_stop_loss_percentage:.1%}")
        print(f"  Take Profit: {config.crypto.crypto_take_profit_percentage:.1%}")
        print()

        # Load market data
        logger.info("Loading BTC/USDT market data from Binance...")
        data_loader = DataLoader(config.data)

        try:
            data = data_loader.load_data(
                config.strategy.symbol,
                config.strategy.start_date,
                config.strategy.end_date
            )
            print(f"Loaded {len(data)} candles of {config.data.timeframe} data")
            print(f"Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")
            print()

        except Exception as e:
            logger.error(f"Failed to load data from Binance: {e}")
            print("Note: This example requires internet connection and Binance API access.")
            print("If Binance API is unavailable, consider using CSV data with --data-source csv")
            return False

        # Run backtest
        logger.info("Running cryptocurrency backtest...")
        backtest_engine = BacktestEngine(config)
        results = backtest_engine.run_backtest(data)

        print("Backtest completed successfully!")
        print()

        # Generate performance report
        logger.info("Generating performance reports...")
        reporter = PerformanceReporter(config)
        report = reporter.generate_full_report(results, save_plots=True)

        # Display results
        display_crypto_results(backtest_engine, config)

        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_prefix = f"results/crypto/btc_backtest_{timestamp}"
        backtest_engine.save_results(output_prefix)

        print(f"\nDetailed results saved to: {output_prefix}_*")
        print("Charts and visualizations saved to results/crypto/")

        return True

    except Exception as e:
        logger.error(f"Backtest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def display_crypto_results(backtest_engine, config):
    """Display formatted cryptocurrency backtest results."""
    risk_metrics = backtest_engine.risk_manager.get_risk_metrics()

    print("PERFORMANCE SUMMARY")
    print("-" * 50)

    # Portfolio Performance
    initial_capital = config.strategy.initial_capital
    final_value = risk_metrics.get('current_portfolio_value', initial_capital)
    total_return = (final_value - initial_capital) / initial_capital

    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Portfolio Value: ${final_value:,.2f}")
    print(f"Total Return: {total_return:.2%}")
    print(f"Total P&L: ${final_value - initial_capital:,.2f}")
    print()

    # Trade Statistics
    print("TRADE STATISTICS")
    print("-" * 50)
    print(f"Total Trades: {risk_metrics.get('total_trades', 0)}")
    print(f"Winning Trades: {risk_metrics.get('winning_trades', 0)}")
    print(f"Losing Trades: {risk_metrics.get('losing_trades', 0)}")
    print(f"Win Rate: {risk_metrics.get('win_rate', 0):.1%}")
    print(f"Average P&L per Trade: ${risk_metrics.get('average_pnl', 0):.2f}")
    print(f"Best Trade: ${risk_metrics.get('best_trade', 0):.2f}")
    print(f"Worst Trade: ${risk_metrics.get('worst_trade', 0):.2f}")

    profit_factor = risk_metrics.get('profit_factor', 0)
    if profit_factor != float('inf'):
        print(f"Profit Factor: {profit_factor:.2f}")
    else:
        print("Profit Factor: ∞ (no losing trades)")
    print()

    # Crypto-specific metrics
    print("CRYPTOCURRENCY METRICS")
    print("-" * 50)
    btc_traded = risk_metrics.get('total_trades', 0) * config.crypto.position_size_btc
    print(f"BTC Position Size: {config.crypto.position_size_btc} BTC per trade")
    print(f"Total BTC Traded: {btc_traded:.2f} BTC")
    print(f"Trading Fees: {config.crypto.taker_fee:.3%} per trade")
    print()

    # Risk Metrics
    print("RISK ANALYSIS")
    print("-" * 50)
    max_consecutive_losses = risk_metrics.get('max_consecutive_losses', 0)
    print(f"Maximum Consecutive Losses: {max_consecutive_losses}")

    # Calculate additional metrics if we have trade data
    trades_df = backtest_engine.risk_manager.get_trade_summary()
    if not trades_df.empty and 'pnl_percentage' in trades_df.columns:
        returns = trades_df['pnl_percentage'].dropna()
        if len(returns) > 1:
            volatility = returns.std() * (252 ** 0.5)  # Annualized volatility
            if returns.mean() > 0:
                sharpe = returns.mean() / returns.std() * (252 ** 0.5) if returns.std() > 0 else 0
                print(f"Volatility (annualized): {volatility:.1%}")
                print(f"Sharpe Ratio: {sharpe:.2f}")


def run_multiple_timeframe_example():
    """Example of running backtests across multiple timeframes."""
    print("\n" + "="*70)
    print("MULTIPLE TIMEFRAME ANALYSIS")
    print("="*70)

    timeframes = ['1h', '4h', '1d']
    results_summary = []

    for timeframe in timeframes:
        print(f"\nTesting {timeframe} timeframe...")

        config = setup_crypto_config()
        config.data.timeframe = timeframe

        try:
            # Load data
            data_loader = DataLoader(config.data)
            data = data_loader.load_data(
                config.strategy.symbol,
                config.strategy.start_date,
                config.strategy.end_date
            )

            # Run backtest
            backtest_engine = BacktestEngine(config)
            results = backtest_engine.run_backtest(data)

            # Get performance metrics
            risk_metrics = backtest_engine.risk_manager.get_risk_metrics()
            initial_capital = config.strategy.initial_capital
            final_value = risk_metrics.get('current_portfolio_value', initial_capital)
            total_return = (final_value - initial_capital) / initial_capital

            results_summary.append({
                'timeframe': timeframe,
                'total_return': total_return,
                'total_trades': risk_metrics.get('total_trades', 0),
                'win_rate': risk_metrics.get('win_rate', 0),
                'profit_factor': risk_metrics.get('profit_factor', 0)
            })

            print(f"  Return: {total_return:.2%}, Trades: {risk_metrics.get('total_trades', 0)}")

        except Exception as e:
            print(f"  Failed: {e}")
            continue

    # Display summary
    print("\nTIMEFRAME COMPARISON")
    print("-" * 50)
    print(f"{'Timeframe':<10} {'Return':<10} {'Trades':<8} {'Win Rate':<10} {'Profit Factor':<12}")
    print("-" * 50)

    for result in results_summary:
        pf = result['profit_factor']
        pf_str = "∞" if pf == float('inf') else f"{pf:.2f}"
        print(f"{result['timeframe']:<10} {result['total_return']:<9.1%} {result['total_trades']:<8} "
              f"{result['win_rate']:<9.1%} {pf_str:<12}")


if __name__ == "__main__":
    print("Starting BTC/USDT cryptocurrency trading backtest example...")

    # Run main backtest
    success = run_crypto_backtest_example()

    if success:
        print("\n" + "="*70)
        print("Backtest completed successfully!")
        print("\nNext steps:")
        print("1. Review the generated charts in results/crypto/")
        print("2. Analyze the detailed trade log CSV files")
        print("3. Experiment with different parameters:")
        print("   - Position size (--btc-position-size)")
        print("   - Timeframe (--timeframe 1h/4h/1d)")
        print("   - RSI/ADX periods (--rsi-period, --adx-period)")
        print("   - Stop loss/take profit levels")
        print("\n4. Try different cryptocurrency pairs:")
        print("   python main.py --symbol ETHUSDT")
        print("   python main.py --symbol ADAUSDT")

        # Optional: Run multiple timeframe analysis
        try:
            try_multiple = input("\nWould you like to run multiple timeframe analysis? (y/N): ")
            if try_multiple.lower().startswith('y'):
                run_multiple_timeframe_example()
        except KeyboardInterrupt:
            pass

    else:
        print("\nBacktest failed. Please check the logs for details.")
        print("\nTroubleshooting:")
        print("1. Ensure you have internet connection for Binance API")
        print("2. Install required dependencies: pip install -r requirements.txt")
        print("3. Check that the Binance API is accessible from your location")
        print("4. Try using CSV data instead: python main.py --data-source csv")

    print("\nFor more information, see README.md")