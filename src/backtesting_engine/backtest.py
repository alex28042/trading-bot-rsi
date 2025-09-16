"""
Backtesting engine for the RSI-ADX momentum strategy.

This module implements a custom backtesting framework that simulates
trading with realistic market conditions and transaction costs.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import sys
import os

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from src.strategy_logic.strategy import RSIADXStrategy
from src.risk_management.risk_manager import RiskManager, Trade

logger = logging.getLogger(__name__)


class BacktestEngine:
    """
    Custom backtesting engine for the RSI-ADX momentum strategy.

    This engine simulates realistic trading conditions including:
    - Transaction costs
    - Slippage
    - Stop-loss and take-profit execution
    - Position sizing
    """

    def __init__(self, config):
        """
        Initialize backtesting engine.

        Args:
            config: Configuration object with all parameters
        """
        self.config = config
        self.strategy = RSIADXStrategy(config.strategy)

        # Initialize risk manager with crypto config if available
        crypto_config = getattr(config, 'crypto', None)
        self.risk_manager = RiskManager(config.strategy, crypto_config)

        # Backtest results
        self.results = None
        self.equity_curve = None
        self.trades = None
        self.performance_metrics = None

    def run_backtest(self, data: pd.DataFrame) -> Dict:
        """
        Run the complete backtest simulation.

        Args:
            data: OHLC data DataFrame

        Returns:
            Dictionary with backtest results
        """
        logger.info("Starting backtest simulation...")

        # Validate data
        if not self._validate_data(data):
            raise ValueError("Invalid data for backtesting")

        # Validate strategy parameters
        if not self.strategy.validate_strategy_parameters():
            raise ValueError("Invalid strategy parameters")

        # Calculate technical indicators
        data_with_indicators = self.strategy.calculate_indicators(data)

        # Generate trading signals
        data_with_signals = self.strategy.generate_signals(data_with_indicators)

        # Simulate trading
        simulation_results = self._simulate_trading(data_with_signals)

        # Calculate performance metrics
        self.performance_metrics = self._calculate_performance_metrics(simulation_results)

        # Store results
        self.results = {
            'data': data_with_signals,
            'simulation': simulation_results,
            'performance': self.performance_metrics,
            'trades': self.risk_manager.get_trade_summary(),
            'risk_metrics': self.risk_manager.get_risk_metrics()
        }

        logger.info("Backtest completed successfully")
        return self.results

    def _validate_data(self, data: pd.DataFrame) -> bool:
        """Validate input data for backtesting."""
        required_columns = ['open', 'high', 'low', 'close', 'volume']

        if not all(col in data.columns for col in required_columns):
            logger.error(f"Missing required columns. Need: {required_columns}")
            return False

        if len(data) < 50:
            logger.error("Insufficient data points. Need at least 50 periods.")
            return False

        if data.isnull().any().any():
            logger.warning("Data contains null values")

        return True

    def _simulate_trading(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Simulate trading based on signals and risk management.

        Args:
            data: DataFrame with OHLC data and signals

        Returns:
            DataFrame with simulation results
        """
        df = data.copy()

        # Initialize tracking columns
        df['portfolio_value'] = 0.0
        df['cash'] = 0.0
        df['position_value'] = 0.0
        df['trade_pnl'] = 0.0
        df['cumulative_pnl'] = 0.0
        df['drawdown'] = 0.0
        df['entry_executed'] = False
        df['exit_executed'] = False
        df['exit_reason'] = ''

        # Track portfolio values
        portfolio_values = []
        peak_value = self.config.strategy.initial_capital

        for i, (date, row) in enumerate(df.iterrows()):
            current_price = row['close']
            high_price = row['high']
            low_price = row['low']

            # Check for exit conditions first
            if self.risk_manager.current_trade:
                exit_reason = self.risk_manager.check_exit_conditions(
                    date, high_price, low_price, current_price, row['signal'] == -1
                )

                if exit_reason:
                    # Determine exit price based on reason
                    if exit_reason == "stop_loss":
                        exit_price = self.risk_manager.current_trade.stop_loss_price
                    elif exit_reason == "take_profit":
                        exit_price = self.risk_manager.current_trade.take_profit_price
                    else:
                        exit_price = current_price

                    trade = self.risk_manager.exit_position(date, exit_price, exit_reason)

                    if trade:
                        df.loc[date, 'exit_executed'] = True
                        df.loc[date, 'exit_reason'] = exit_reason
                        df.loc[date, 'trade_pnl'] = trade.pnl

            # Check for entry conditions
            elif row['signal'] == 1:
                trade = self.risk_manager.enter_position(date, current_price)
                if trade:
                    df.loc[date, 'entry_executed'] = True

            # Update portfolio value
            self.risk_manager.update_portfolio_value(current_price)

            # Record values
            df.loc[date, 'portfolio_value'] = self.risk_manager.portfolio_value
            df.loc[date, 'cash'] = self.risk_manager.cash

            if self.risk_manager.current_trade:
                df.loc[date, 'position_value'] = (
                    self.risk_manager.current_trade.quantity * current_price
                )

            # Calculate cumulative P&L
            total_pnl = self.risk_manager.portfolio_value - self.config.strategy.initial_capital
            df.loc[date, 'cumulative_pnl'] = total_pnl

            # Calculate drawdown
            portfolio_values.append(self.risk_manager.portfolio_value)
            peak_value = max(peak_value, self.risk_manager.portfolio_value)
            current_drawdown = (peak_value - self.risk_manager.portfolio_value) / peak_value
            df.loc[date, 'drawdown'] = current_drawdown

        self.equity_curve = df[['portfolio_value', 'cumulative_pnl', 'drawdown']].copy()

        return df

    def _calculate_performance_metrics(self, simulation_data: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.

        Args:
            simulation_data: DataFrame with simulation results

        Returns:
            Dictionary with performance metrics
        """
        initial_capital = self.config.strategy.initial_capital
        final_value = simulation_data['portfolio_value'].iloc[-1]

        # Basic returns
        total_return = (final_value - initial_capital) / initial_capital

        # Time-based calculations
        start_date = simulation_data.index[0]
        end_date = simulation_data.index[-1]
        total_days = (end_date - start_date).days
        total_years = total_days / 365.25

        # Annualized return
        annualized_return = (final_value / initial_capital) ** (1 / total_years) - 1 if total_years > 0 else 0

        # Calculate daily returns for risk metrics
        daily_returns = simulation_data['portfolio_value'].pct_change().dropna()

        # Risk metrics
        volatility = daily_returns.std() * np.sqrt(252)  # Annualized volatility
        sharpe_ratio = (annualized_return - self.config.backtest.risk_free_rate) / volatility if volatility > 0 else 0

        # Maximum drawdown
        max_drawdown = simulation_data['drawdown'].max()

        # Calmar ratio
        calmar_ratio = annualized_return / max_drawdown if max_drawdown > 0 else 0

        # Win rate and trade statistics
        risk_metrics = self.risk_manager.get_risk_metrics()

        # Sortino ratio (downside deviation)
        negative_returns = daily_returns[daily_returns < 0]
        downside_deviation = negative_returns.std() * np.sqrt(252)
        sortino_ratio = (annualized_return - self.config.backtest.risk_free_rate) / downside_deviation if len(negative_returns) > 0 and downside_deviation > 0 else 0

        # Value at Risk (95% confidence)
        var_95 = daily_returns.quantile(0.05)

        # Maximum consecutive losses
        portfolio_values = simulation_data['portfolio_value'].values
        consecutive_losses = 0
        max_consecutive_losses = 0

        for i in range(1, len(portfolio_values)):
            if portfolio_values[i] < portfolio_values[i-1]:
                consecutive_losses += 1
                max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
            else:
                consecutive_losses = 0

        metrics = {
            # Return metrics
            'total_return': round(total_return, 4),
            'annualized_return': round(annualized_return, 4),
            'initial_capital': initial_capital,
            'final_value': round(final_value, 2),
            'total_pnl': round(final_value - initial_capital, 2),

            # Risk metrics
            'volatility': round(volatility, 4),
            'sharpe_ratio': round(sharpe_ratio, 4),
            'sortino_ratio': round(sortino_ratio, 4),
            'calmar_ratio': round(calmar_ratio, 4),
            'max_drawdown': round(max_drawdown, 4),
            'var_95': round(var_95, 4),

            # Trade metrics
            'total_trades': risk_metrics.get('total_trades', 0),
            'win_rate': round(risk_metrics.get('win_rate', 0), 4),
            'profit_factor': round(risk_metrics.get('profit_factor', 0), 4),
            'average_pnl': round(risk_metrics.get('average_pnl', 0), 2),
            'best_trade': round(risk_metrics.get('best_trade', 0), 2),
            'worst_trade': round(risk_metrics.get('worst_trade', 0), 2),
            'max_consecutive_losses': max_consecutive_losses,

            # Time metrics
            'backtest_period': f"{start_date.date()} to {end_date.date()}",
            'total_days': total_days,
            'total_years': round(total_years, 2),

            # Strategy-specific metrics
            'time_in_market': round(
                (simulation_data['position_value'] > 0).sum() / len(simulation_data), 4
            ),
            'average_holding_period': round(
                risk_metrics.get('total_trades', 1) / max(1, risk_metrics.get('total_trades', 1)), 2
            )
        }

        return metrics

    def get_equity_curve(self) -> pd.DataFrame:
        """Return the equity curve DataFrame."""
        return self.equity_curve

    def get_trade_summary(self) -> pd.DataFrame:
        """Return summary of all trades."""
        return self.risk_manager.get_trade_summary()

    def get_performance_summary(self) -> Dict:
        """Return formatted performance summary."""
        if not self.performance_metrics:
            return {"error": "No backtest results available"}

        summary = {
            "Strategy Performance": {
                "Total Return": f"{self.performance_metrics['total_return']:.2%}",
                "Annualized Return": f"{self.performance_metrics['annualized_return']:.2%}",
                "Sharpe Ratio": f"{self.performance_metrics['sharpe_ratio']:.2f}",
                "Maximum Drawdown": f"{self.performance_metrics['max_drawdown']:.2%}",
                "Calmar Ratio": f"{self.performance_metrics['calmar_ratio']:.2f}",
            },
            "Trade Statistics": {
                "Total Trades": self.performance_metrics['total_trades'],
                "Win Rate": f"{self.performance_metrics['win_rate']:.2%}",
                "Profit Factor": f"{self.performance_metrics['profit_factor']:.2f}",
                "Average P&L": f"${self.performance_metrics['average_pnl']:.2f}",
                "Best Trade": f"${self.performance_metrics['best_trade']:.2f}",
                "Worst Trade": f"${self.performance_metrics['worst_trade']:.2f}",
            },
            "Risk Metrics": {
                "Volatility": f"{self.performance_metrics['volatility']:.2%}",
                "Sortino Ratio": f"{self.performance_metrics['sortino_ratio']:.2f}",
                "VaR (95%)": f"{self.performance_metrics['var_95']:.2%}",
                "Time in Market": f"{self.performance_metrics['time_in_market']:.2%}",
            }
        }

        return summary

    def save_results(self, filepath: str):
        """Save backtest results to file."""
        if not self.results:
            logger.error("No results to save")
            return

        # Save main results
        with open(f"{filepath}_summary.txt", 'w') as f:
            f.write("RSI-ADX Momentum Strategy Backtest Results\n")
            f.write("=" * 50 + "\n\n")

            summary = self.get_performance_summary()
            for section, metrics in summary.items():
                f.write(f"{section}:\n")
                for metric, value in metrics.items():
                    f.write(f"  {metric}: {value}\n")
                f.write("\n")

        # Save detailed trades
        trades_df = self.get_trade_summary()
        if not trades_df.empty:
            trades_df.to_csv(f"{filepath}_trades.csv", index=False)

        # Save equity curve
        if self.equity_curve is not None:
            self.equity_curve.to_csv(f"{filepath}_equity_curve.csv")

        logger.info(f"Results saved to {filepath}_*.csv and {filepath}_summary.txt")