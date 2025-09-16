"""
Performance reporting module for backtesting results.

This module provides comprehensive performance analysis including
charts, metrics, and detailed reports.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Rectangle
import seaborn as sns
import logging
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import os

logger = logging.getLogger(__name__)

# Set style for better looking plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class PerformanceReporter:
    """Class for generating comprehensive performance reports and visualizations."""

    def __init__(self, config):
        """
        Initialize performance reporter.

        Args:
            config: Configuration object
        """
        self.config = config
        self.results_dir = config.data.results_directory

        # Ensure results directory exists
        os.makedirs(self.results_dir, exist_ok=True)

    def generate_full_report(self, backtest_results: Dict, save_plots: bool = True) -> Dict:
        """
        Generate comprehensive performance report.

        Args:
            backtest_results: Results from backtesting engine
            save_plots: Whether to save plots to files

        Returns:
            Dictionary with report data
        """
        logger.info("Generating comprehensive performance report...")

        report = {
            'summary': self._generate_summary_report(backtest_results),
            'detailed_metrics': self._calculate_detailed_metrics(backtest_results),
            'trade_analysis': self._analyze_trades(backtest_results),
            'risk_analysis': self._analyze_risk(backtest_results),
            'monthly_returns': self._calculate_monthly_returns(backtest_results)
        }

        if save_plots:
            self.create_visualizations(backtest_results)

        return report

    def _generate_summary_report(self, results: Dict) -> Dict:
        """Generate executive summary of performance."""
        metrics = results['performance']
        risk_metrics = results['risk_metrics']

        summary = {
            'strategy_name': 'RSI-ADX Momentum Strategy',
            'backtest_period': metrics['backtest_period'],
            'total_return': f"{metrics['total_return']:.2%}",
            'annualized_return': f"{metrics['annualized_return']:.2%}",
            'volatility': f"{metrics['volatility']:.2%}",
            'sharpe_ratio': metrics['sharpe_ratio'],
            'max_drawdown': f"{metrics['max_drawdown']:.2%}",
            'total_trades': metrics['total_trades'],
            'win_rate': f"{metrics['win_rate']:.2%}",
            'profit_factor': metrics['profit_factor'],
            'final_portfolio_value': f"${metrics['final_value']:,.2f}"
        }

        return summary

    def _calculate_detailed_metrics(self, results: Dict) -> Dict:
        """Calculate additional detailed performance metrics."""
        simulation_data = results['simulation']
        portfolio_values = simulation_data['portfolio_value']
        daily_returns = portfolio_values.pct_change().dropna()

        # Additional risk metrics
        metrics = {
            'skewness': float(daily_returns.skew()),
            'kurtosis': float(daily_returns.kurtosis()),
            'information_ratio': self._calculate_information_ratio(daily_returns),
            'ulcer_index': self._calculate_ulcer_index(portfolio_values),
            'recovery_factor': self._calculate_recovery_factor(results),
            'expectancy': self._calculate_expectancy(results),
            'kelly_criterion': self._calculate_kelly_criterion(results)
        }

        return metrics

    def _analyze_trades(self, results: Dict) -> Dict:
        """Analyze individual trade performance."""
        trades_df = results['trades']

        if trades_df.empty:
            return {'error': 'No trades to analyze'}

        # Filter closed trades
        closed_trades = trades_df[trades_df['exit_date'].notna()]

        if closed_trades.empty:
            return {'error': 'No closed trades to analyze'}

        analysis = {
            'trade_distribution': {
                'total_trades': len(closed_trades),
                'winning_trades': len(closed_trades[closed_trades['pnl'] > 0]),
                'losing_trades': len(closed_trades[closed_trades['pnl'] < 0]),
                'break_even_trades': len(closed_trades[closed_trades['pnl'] == 0])
            },
            'pnl_statistics': {
                'total_pnl': float(closed_trades['pnl'].sum()),
                'average_pnl': float(closed_trades['pnl'].mean()),
                'median_pnl': float(closed_trades['pnl'].median()),
                'std_pnl': float(closed_trades['pnl'].std()),
                'min_pnl': float(closed_trades['pnl'].min()),
                'max_pnl': float(closed_trades['pnl'].max())
            },
            'holding_periods': {
                'average_days': self._calculate_average_holding_period(closed_trades),
                'min_days': self._calculate_min_holding_period(closed_trades),
                'max_days': self._calculate_max_holding_period(closed_trades)
            },
            'exit_reasons': dict(closed_trades['exit_reason'].value_counts()),
            'monthly_trade_frequency': self._calculate_monthly_trade_frequency(closed_trades)
        }

        return analysis

    def _analyze_risk(self, results: Dict) -> Dict:
        """Perform detailed risk analysis."""
        simulation_data = results['simulation']
        portfolio_values = simulation_data['portfolio_value']
        drawdowns = simulation_data['drawdown']

        risk_analysis = {
            'drawdown_analysis': {
                'max_drawdown': float(drawdowns.max()),
                'average_drawdown': float(drawdowns[drawdowns > 0].mean()) if (drawdowns > 0).any() else 0,
                'drawdown_periods': self._analyze_drawdown_periods(drawdowns),
                'time_underwater': self._calculate_time_underwater(drawdowns)
            },
            'value_at_risk': {
                'var_95': float(portfolio_values.pct_change().quantile(0.05)),
                'var_99': float(portfolio_values.pct_change().quantile(0.01)),
                'cvar_95': self._calculate_conditional_var(portfolio_values, 0.05),
                'cvar_99': self._calculate_conditional_var(portfolio_values, 0.01)
            },
            'tail_ratio': self._calculate_tail_ratio(portfolio_values),
            'gain_to_pain_ratio': self._calculate_gain_to_pain_ratio(portfolio_values)
        }

        return risk_analysis

    def _calculate_monthly_returns(self, results: Dict) -> pd.DataFrame:
        """Calculate monthly returns for analysis."""
        simulation_data = results['simulation']
        portfolio_values = simulation_data['portfolio_value']

        # Resample to monthly and calculate returns
        monthly_values = portfolio_values.resample('M').last()
        monthly_returns = monthly_values.pct_change().dropna()

        # Create detailed monthly analysis
        monthly_df = pd.DataFrame({
            'return': monthly_returns,
            'cumulative_return': (1 + monthly_returns).cumprod() - 1,
            'value': monthly_values[1:]  # Skip first NaN from pct_change
        })

        return monthly_df

    def create_visualizations(self, results: Dict):
        """Create comprehensive visualization suite."""
        logger.info("Creating performance visualizations...")

        # 1. Equity curve chart
        self._plot_equity_curve(results)

        # 2. Drawdown chart
        self._plot_drawdown(results)

        # 3. Monthly returns heatmap
        self._plot_monthly_returns_heatmap(results)

        # 4. Trade analysis charts
        self._plot_trade_analysis(results)

        # 5. Risk metrics visualization
        self._plot_risk_metrics(results)

        # 6. Performance comparison
        self._plot_performance_comparison(results)

        logger.info(f"Visualizations saved to {self.results_dir}")

    def _plot_equity_curve(self, results: Dict):
        """Plot equity curve with key statistics."""
        simulation_data = results['simulation']
        portfolio_values = simulation_data['portfolio_value']

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        # Main equity curve
        ax1.plot(portfolio_values.index, portfolio_values.values, linewidth=2, label='Portfolio Value')
        ax1.axhline(y=self.config.strategy.initial_capital, color='gray', linestyle='--', alpha=0.7, label='Initial Capital')

        # Highlight trades
        trades_df = results['trades']
        if not trades_df.empty:
            entry_dates = trades_df['entry_date'].dropna()
            exit_dates = trades_df['exit_date'].dropna()

            for date in entry_dates:
                if date in portfolio_values.index:
                    ax1.axvline(x=date, color='green', alpha=0.3, linewidth=1)

            for date in exit_dates:
                if date in portfolio_values.index:
                    ax1.axvline(x=date, color='red', alpha=0.3, linewidth=1)

        ax1.set_title('Portfolio Equity Curve', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Returns subplot
        returns = portfolio_values.pct_change().dropna()
        ax2.plot(returns.index, returns.values, linewidth=1, alpha=0.7, label='Daily Returns')
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.set_title('Daily Returns', fontsize=14)
        ax2.set_ylabel('Daily Return (%)', fontsize=12)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.grid(True, alpha=0.3)

        # Format x-axis
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
            ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/equity_curve.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_drawdown(self, results: Dict):
        """Plot drawdown analysis."""
        simulation_data = results['simulation']
        drawdowns = simulation_data['drawdown']

        fig, ax = plt.subplots(figsize=(14, 6))

        # Plot drawdown
        ax.fill_between(drawdowns.index, drawdowns.values * 100, 0, alpha=0.7, color='red', label='Drawdown')
        ax.plot(drawdowns.index, drawdowns.values * 100, color='darkred', linewidth=1)

        ax.set_title('Portfolio Drawdown Analysis', fontsize=16, fontweight='bold')
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.set_xlabel('Date', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend()

        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
        plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/drawdown_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_monthly_returns_heatmap(self, results: Dict):
        """Create monthly returns heatmap."""
        monthly_returns = self._calculate_monthly_returns(results)

        if len(monthly_returns) < 2:
            logger.warning("Insufficient data for monthly returns heatmap")
            return

        # Prepare data for heatmap
        monthly_returns['year'] = monthly_returns.index.year
        monthly_returns['month'] = monthly_returns.index.month

        heatmap_data = monthly_returns.pivot_table(
            values='return', index='year', columns='month', aggfunc='mean'
        )

        # Create heatmap
        fig, ax = plt.subplots(figsize=(12, 8))

        sns.heatmap(
            heatmap_data * 100,  # Convert to percentage
            annot=True,
            fmt='.1f',
            cmap='RdYlGn',
            center=0,
            ax=ax,
            cbar_kws={'label': 'Monthly Return (%)'}
        )

        ax.set_title('Monthly Returns Heatmap', fontsize=16, fontweight='bold')
        ax.set_xlabel('Month', fontsize=12)
        ax.set_ylabel('Year', fontsize=12)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/monthly_returns_heatmap.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_trade_analysis(self, results: Dict):
        """Plot trade analysis charts."""
        trades_df = results['trades']

        if trades_df.empty:
            logger.warning("No trades data for plotting")
            return

        closed_trades = trades_df[trades_df['exit_date'].notna()]

        if closed_trades.empty:
            logger.warning("No closed trades for plotting")
            return

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. P&L distribution
        ax1.hist(closed_trades['pnl'], bins=20, alpha=0.7, edgecolor='black')
        ax1.axvline(x=0, color='red', linestyle='--', alpha=0.7)
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L ($)')
        ax1.set_ylabel('Frequency')
        ax1.grid(True, alpha=0.3)

        # 2. Cumulative P&L
        cumulative_pnl = closed_trades['pnl'].cumsum()
        ax2.plot(range(len(cumulative_pnl)), cumulative_pnl, linewidth=2)
        ax2.set_title('Cumulative P&L by Trade', fontweight='bold')
        ax2.set_xlabel('Trade Number')
        ax2.set_ylabel('Cumulative P&L ($)')
        ax2.grid(True, alpha=0.3)

        # 3. Exit reasons pie chart
        exit_reasons = closed_trades['exit_reason'].value_counts()
        ax3.pie(exit_reasons.values, labels=exit_reasons.index, autopct='%1.1f%%')
        ax3.set_title('Exit Reasons Distribution', fontweight='bold')

        # 4. Win/Loss streaks
        wins_losses = (closed_trades['pnl'] > 0).astype(int)
        streaks = self._calculate_streaks(wins_losses)
        ax4.bar(['Win Streaks', 'Loss Streaks'], [streaks['max_win_streak'], streaks['max_loss_streak']])
        ax4.set_title('Maximum Win/Loss Streaks', fontweight='bold')
        ax4.set_ylabel('Number of Consecutive Trades')

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/trade_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_risk_metrics(self, results: Dict):
        """Plot risk metrics visualization."""
        simulation_data = results['simulation']
        returns = simulation_data['portfolio_value'].pct_change().dropna()

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

        # 1. Returns distribution
        ax1.hist(returns * 100, bins=50, alpha=0.7, edgecolor='black', density=True)
        ax1.axvline(x=returns.mean() * 100, color='red', linestyle='--', label='Mean')
        ax1.set_title('Daily Returns Distribution', fontweight='bold')
        ax1.set_xlabel('Daily Return (%)')
        ax1.set_ylabel('Density')
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2. Rolling volatility
        rolling_vol = returns.rolling(window=30).std() * np.sqrt(252) * 100
        ax2.plot(rolling_vol.index, rolling_vol.values)
        ax2.set_title('30-Day Rolling Volatility', fontweight='bold')
        ax2.set_xlabel('Date')
        ax2.set_ylabel('Annualized Volatility (%)')
        ax2.grid(True, alpha=0.3)

        # 3. Q-Q plot for normality test
        from scipy import stats
        stats.probplot(returns, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot (Normality Test)', fontweight='bold')
        ax3.grid(True, alpha=0.3)

        # 4. Rolling Sharpe ratio
        rolling_sharpe = (returns.rolling(window=60).mean() * 252 - self.config.backtest.risk_free_rate) / (
            returns.rolling(window=60).std() * np.sqrt(252)
        )
        ax4.plot(rolling_sharpe.index, rolling_sharpe.values)
        ax4.axhline(y=0, color='red', linestyle='--', alpha=0.7)
        ax4.set_title('60-Day Rolling Sharpe Ratio', fontweight='bold')
        ax4.set_xlabel('Date')
        ax4.set_ylabel('Sharpe Ratio')
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/risk_metrics.png", dpi=300, bbox_inches='tight')
        plt.close()

    def _plot_performance_comparison(self, results: Dict):
        """Plot performance comparison with buy-and-hold."""
        simulation_data = results['simulation']
        data = results['data']

        # Calculate buy and hold performance
        initial_price = data['close'].iloc[0]
        final_price = data['close'].iloc[-1]
        buy_hold_return = (final_price / initial_price - 1)

        # Calculate strategy performance
        strategy_return = results['performance']['total_return']

        # Create comparison chart
        fig, ax = plt.subplots(figsize=(10, 6))

        categories = ['Strategy', 'Buy & Hold']
        returns = [strategy_return * 100, buy_hold_return * 100]
        colors = ['green' if r > 0 else 'red' for r in returns]

        bars = ax.bar(categories, returns, color=colors, alpha=0.7)
        ax.axhline(y=0, color='black', linestyle='-', alpha=0.5)

        # Add value labels on bars
        for bar, value in zip(bars, returns):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + (1 if height > 0 else -3),
                   f'{value:.1f}%', ha='center', va='bottom' if height > 0 else 'top')

        ax.set_title('Strategy vs Buy & Hold Performance', fontsize=16, fontweight='bold')
        ax.set_ylabel('Total Return (%)', fontsize=12)
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{self.results_dir}/performance_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()

    # Helper methods for calculations
    def _calculate_information_ratio(self, returns: pd.Series) -> float:
        """Calculate information ratio."""
        excess_returns = returns - self.config.backtest.risk_free_rate / 252
        return float(excess_returns.mean() / excess_returns.std() * np.sqrt(252)) if excess_returns.std() > 0 else 0

    def _calculate_ulcer_index(self, portfolio_values: pd.Series) -> float:
        """Calculate Ulcer Index."""
        running_max = portfolio_values.expanding().max()
        drawdowns = (portfolio_values - running_max) / running_max
        return float(np.sqrt((drawdowns ** 2).mean()))

    def _calculate_recovery_factor(self, results: Dict) -> float:
        """Calculate recovery factor."""
        total_return = results['performance']['total_return']
        max_drawdown = results['performance']['max_drawdown']
        return total_return / max_drawdown if max_drawdown > 0 else 0

    def _calculate_expectancy(self, results: Dict) -> float:
        """Calculate expectancy per trade."""
        trades_df = results['trades']
        closed_trades = trades_df[trades_df['exit_date'].notna()]

        if closed_trades.empty:
            return 0

        win_rate = (closed_trades['pnl'] > 0).mean()
        avg_win = closed_trades[closed_trades['pnl'] > 0]['pnl'].mean() if win_rate > 0 else 0
        avg_loss = closed_trades[closed_trades['pnl'] < 0]['pnl'].mean() if win_rate < 1 else 0

        return float(win_rate * avg_win + (1 - win_rate) * avg_loss)

    def _calculate_kelly_criterion(self, results: Dict) -> float:
        """Calculate Kelly Criterion optimal position size."""
        trades_df = results['trades']
        closed_trades = trades_df[trades_df['exit_date'].notna()]

        if closed_trades.empty:
            return 0

        win_rate = (closed_trades['pnl_percentage'] > 0).mean()
        avg_win = closed_trades[closed_trades['pnl_percentage'] > 0]['pnl_percentage'].mean() if win_rate > 0 else 0
        avg_loss = abs(closed_trades[closed_trades['pnl_percentage'] < 0]['pnl_percentage'].mean()) if win_rate < 1 else 0

        if avg_loss == 0:
            return 0

        return float(win_rate - (1 - win_rate) / (avg_win / avg_loss))

    def _calculate_average_holding_period(self, trades: pd.DataFrame) -> float:
        """Calculate average holding period in days."""
        if trades.empty:
            return 0

        holding_periods = (trades['exit_date'] - trades['entry_date']).dt.days
        return float(holding_periods.mean())

    def _calculate_min_holding_period(self, trades: pd.DataFrame) -> float:
        """Calculate minimum holding period in days."""
        if trades.empty:
            return 0

        holding_periods = (trades['exit_date'] - trades['entry_date']).dt.days
        return float(holding_periods.min())

    def _calculate_max_holding_period(self, trades: pd.DataFrame) -> float:
        """Calculate maximum holding period in days."""
        if trades.empty:
            return 0

        holding_periods = (trades['exit_date'] - trades['entry_date']).dt.days
        return float(holding_periods.max())

    def _calculate_monthly_trade_frequency(self, trades: pd.DataFrame) -> Dict:
        """Calculate monthly trade frequency."""
        if trades.empty:
            return {}

        trades['month'] = trades['entry_date'].dt.month
        monthly_counts = trades['month'].value_counts().sort_index()
        return monthly_counts.to_dict()

    def _analyze_drawdown_periods(self, drawdowns: pd.Series) -> Dict:
        """Analyze drawdown periods."""
        periods = []
        current_period = None

        for date, dd in drawdowns.items():
            if dd > 0:  # In drawdown
                if current_period is None:
                    current_period = {'start': date, 'max_dd': dd, 'end': None}
                else:
                    current_period['max_dd'] = max(current_period['max_dd'], dd)
            else:  # Not in drawdown
                if current_period is not None:
                    current_period['end'] = date
                    periods.append(current_period)
                    current_period = None

        # Handle case where drawdown period extends to end
        if current_period is not None:
            current_period['end'] = drawdowns.index[-1]
            periods.append(current_period)

        if not periods:
            return {'count': 0, 'avg_duration': 0, 'max_duration': 0}

        durations = [(p['end'] - p['start']).days for p in periods]

        return {
            'count': len(periods),
            'avg_duration': np.mean(durations),
            'max_duration': max(durations),
            'avg_depth': np.mean([p['max_dd'] for p in periods])
        }

    def _calculate_time_underwater(self, drawdowns: pd.Series) -> float:
        """Calculate percentage of time underwater (in drawdown)."""
        underwater_periods = (drawdowns > 0).sum()
        total_periods = len(drawdowns)
        return underwater_periods / total_periods if total_periods > 0 else 0

    def _calculate_conditional_var(self, portfolio_values: pd.Series, alpha: float) -> float:
        """Calculate Conditional Value at Risk (Expected Shortfall)."""
        returns = portfolio_values.pct_change().dropna()
        var_threshold = returns.quantile(alpha)
        return float(returns[returns <= var_threshold].mean())

    def _calculate_tail_ratio(self, portfolio_values: pd.Series) -> float:
        """Calculate tail ratio (95th percentile / 5th percentile)."""
        returns = portfolio_values.pct_change().dropna()
        return float(returns.quantile(0.95) / abs(returns.quantile(0.05))) if returns.quantile(0.05) != 0 else 0

    def _calculate_gain_to_pain_ratio(self, portfolio_values: pd.Series) -> float:
        """Calculate gain to pain ratio."""
        returns = portfolio_values.pct_change().dropna()
        total_return = (portfolio_values.iloc[-1] / portfolio_values.iloc[0] - 1)
        pain = returns[returns < 0].sum()
        return float(total_return / abs(pain)) if pain < 0 else float('inf')

    def _calculate_streaks(self, wins_losses: pd.Series) -> Dict:
        """Calculate win/loss streaks."""
        streaks = {'max_win_streak': 0, 'max_loss_streak': 0, 'current_streak': 0}

        current_win_streak = 0
        current_loss_streak = 0
        max_win_streak = 0
        max_loss_streak = 0

        for win in wins_losses:
            if win == 1:  # Win
                current_win_streak += 1
                current_loss_streak = 0
                max_win_streak = max(max_win_streak, current_win_streak)
            else:  # Loss
                current_loss_streak += 1
                current_win_streak = 0
                max_loss_streak = max(max_loss_streak, current_loss_streak)

        return {
            'max_win_streak': max_win_streak,
            'max_loss_streak': max_loss_streak
        }