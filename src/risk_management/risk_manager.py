"""
Risk management module for the trading strategy.

This module handles position sizing, stop-loss, take-profit,
and other risk management features. Enhanced for cryptocurrency trading
with BTC fraction position sizing and crypto-specific risk parameters.
"""

import pandas as pd
import numpy as np
import logging
from typing import Dict, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Trade:
    """Class to represent a single trade."""
    entry_date: pd.Timestamp
    entry_price: float
    exit_date: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    quantity: float = 0
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    exit_reason: Optional[str] = None
    pnl: Optional[float] = None
    pnl_percentage: Optional[float] = None

    def is_open(self) -> bool:
        """Check if trade is still open."""
        return self.exit_date is None

    def close_trade(self, exit_date: pd.Timestamp, exit_price: float, exit_reason: str):
        """Close the trade and calculate P&L."""
        self.exit_date = exit_date
        self.exit_price = exit_price
        self.exit_reason = exit_reason

        self.pnl = (exit_price - self.entry_price) * self.quantity
        self.pnl_percentage = (exit_price - self.entry_price) / self.entry_price


class RiskManager:
    """Risk management system for trading strategy."""

    def __init__(self, risk_config, crypto_config=None):
        """
        Initialize risk manager with configuration.

        Args:
            risk_config: Configuration object with risk parameters
            crypto_config: Optional crypto-specific configuration
        """
        self.config = risk_config
        self.crypto_config = crypto_config
        self.trades = []
        self.current_trade = None
        self.portfolio_value = risk_config.initial_capital
        self.cash = risk_config.initial_capital

        # Determine if using crypto mode
        self.is_crypto = crypto_config is not None and getattr(crypto_config, 'use_btc_position_sizing', False)

    def calculate_position_size(self, entry_price: float, current_portfolio_value: float) -> float:
        """
        Calculate position size based on fixed percentage allocation or BTC fractions.

        Args:
            entry_price: Entry price for the position
            current_portfolio_value: Current portfolio value

        Returns:
            Number of shares/units to buy (float for crypto fractions)
        """
        if self.is_crypto and self.crypto_config:
            # Use fixed BTC position sizing for crypto
            btc_quantity = self.crypto_config.position_size_btc

            # Account for trading fees (Binance fees)
            fee_amount = btc_quantity * entry_price * self.crypto_config.taker_fee
            total_cost = (btc_quantity * entry_price) + fee_amount

            # Check if we have enough balance
            if total_cost > self.cash:
                # Scale down to available cash
                available_for_trade = self.cash * 0.95  # Leave 5% buffer
                btc_quantity = available_for_trade / (entry_price * (1 + self.crypto_config.taker_fee))

            logger.debug(f"Crypto position size: {btc_quantity:.6f} BTC at ${entry_price:.2f} = ${btc_quantity * entry_price:.2f}")
            return max(0, btc_quantity)
        else:
            # Traditional percentage-based position sizing
            allocation_amount = current_portfolio_value * self.config.position_size

            # Account for commission
            allocation_amount -= self.config.commission_per_trade

            # Calculate number of shares (rounded down to avoid over-allocation)
            shares = int(allocation_amount / entry_price)

            logger.debug(f"Traditional position size: ${allocation_amount:.2f} / ${entry_price:.2f} = {shares} shares")
            return max(0, shares)

    def calculate_stop_loss_price(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate stop-loss price based on percentage (crypto-aware).

        Args:
            entry_price: Entry price of the position
            is_long: True for long positions, False for short

        Returns:
            Stop-loss price
        """
        # Use crypto-specific stop loss if available
        if self.is_crypto and self.crypto_config:
            stop_loss_pct = self.crypto_config.crypto_stop_loss_percentage
        else:
            stop_loss_pct = self.config.stop_loss_percentage

        if is_long:
            stop_loss = entry_price * (1 - stop_loss_pct)
        else:
            stop_loss = entry_price * (1 + stop_loss_pct)

        # Use more precision for crypto prices
        decimals = 2 if not self.is_crypto else 6
        return round(stop_loss, decimals)

    def calculate_take_profit_price(self, entry_price: float, is_long: bool = True) -> float:
        """
        Calculate take-profit price based on percentage (crypto-aware).

        Args:
            entry_price: Entry price of the position
            is_long: True for long positions, False for short

        Returns:
            Take-profit price
        """
        # Use crypto-specific take profit if available
        if self.is_crypto and self.crypto_config:
            take_profit_pct = self.crypto_config.crypto_take_profit_percentage
        else:
            take_profit_pct = self.config.take_profit_percentage

        if is_long:
            take_profit = entry_price * (1 + take_profit_pct)
        else:
            take_profit = entry_price * (1 - take_profit_pct)

        # Use more precision for crypto prices
        decimals = 2 if not self.is_crypto else 6
        return round(take_profit, decimals)

    def enter_position(self, date: pd.Timestamp, price: float) -> Optional[Trade]:
        """
        Enter a new position with risk management parameters.

        Args:
            date: Entry date
            price: Entry price

        Returns:
            Trade object if position entered, None otherwise
        """
        if self.current_trade is not None and self.current_trade.is_open():
            logger.warning(f"Cannot enter new position - existing position is open")
            return None

        # Calculate position size
        quantity = self.calculate_position_size(price, self.portfolio_value)

        if quantity <= 0:
            logger.warning(f"Cannot enter position - insufficient capital")
            return None

        # Calculate risk management levels
        stop_loss_price = self.calculate_stop_loss_price(price)
        take_profit_price = self.calculate_take_profit_price(price)

        # Calculate total cost including commission/fees
        if self.is_crypto and self.crypto_config:
            # Use percentage-based trading fees for crypto
            trading_fee = quantity * price * self.crypto_config.taker_fee
            total_cost = (quantity * price) + trading_fee
        else:
            # Use fixed commission for traditional assets
            total_cost = (quantity * price) + self.config.commission_per_trade

        if total_cost > self.cash:
            logger.warning(f"Insufficient cash: need ${total_cost:.2f}, have ${self.cash:.2f}")
            return None

        # Create trade
        trade = Trade(
            entry_date=date,
            entry_price=price,
            quantity=quantity,
            stop_loss_price=stop_loss_price,
            take_profit_price=take_profit_price
        )

        # Update cash and portfolio
        self.cash -= total_cost
        self.current_trade = trade
        self.trades.append(trade)

        # Crypto-friendly logging
        if self.is_crypto:
            logger.info(f"Entered position: {quantity:.6f} BTC at ${price:.2f} on {date.date()}")
            logger.info(f"Stop loss: ${stop_loss_price:.2f}, Take profit: ${take_profit_price:.2f}")
        else:
            logger.info(f"Entered position: {quantity} shares at ${price:.2f} on {date.date()}")
            logger.info(f"Stop loss: ${stop_loss_price:.2f}, Take profit: ${take_profit_price:.2f}")

        return trade

    def check_exit_conditions(self, date: pd.Timestamp, high: float, low: float,
                            close: float, signal_exit: bool = False) -> Optional[str]:
        """
        Check if any exit conditions are met.

        Args:
            date: Current date
            high: High price of the period
            low: Low price of the period
            close: Close price of the period
            signal_exit: True if strategy signal indicates exit

        Returns:
            Exit reason if exit conditions met, None otherwise
        """
        if self.current_trade is None or not self.current_trade.is_open():
            return None

        trade = self.current_trade

        # Check stop loss (use low price for conservative approach)
        if low <= trade.stop_loss_price:
            return "stop_loss"

        # Check take profit (use high price for conservative approach)
        if high >= trade.take_profit_price:
            return "take_profit"

        # Check strategy signal
        if signal_exit:
            return "strategy_signal"

        return None

    def exit_position(self, date: pd.Timestamp, price: float, exit_reason: str) -> Optional[Trade]:
        """
        Exit the current position.

        Args:
            date: Exit date
            price: Exit price
            exit_reason: Reason for exit

        Returns:
            Closed trade object
        """
        if self.current_trade is None or not self.current_trade.is_open():
            logger.warning("No open position to exit")
            return None

        trade = self.current_trade

        # Adjust exit price based on exit reason
        if exit_reason == "stop_loss":
            exit_price = trade.stop_loss_price
        elif exit_reason == "take_profit":
            exit_price = trade.take_profit_price
        else:
            exit_price = price

        # Close the trade
        trade.close_trade(date, exit_price, exit_reason)

        # Calculate proceeds (subtract commission/fees)
        if self.is_crypto and self.crypto_config:
            # Use percentage-based trading fees for crypto
            exit_fee = trade.quantity * exit_price * self.crypto_config.taker_fee
            proceeds = (trade.quantity * exit_price) - exit_fee
        else:
            # Use fixed commission for traditional assets
            proceeds = (trade.quantity * exit_price) - self.config.commission_per_trade

        self.cash += proceeds

        # Update portfolio value
        self.portfolio_value = self.cash

        # Crypto-friendly logging
        if self.is_crypto:
            logger.info(f"Exited position: {trade.quantity:.6f} BTC at ${exit_price:.2f} on {date.date()}")
        else:
            logger.info(f"Exited position: {trade.quantity} shares at ${exit_price:.2f} on {date.date()}")

        logger.info(f"Exit reason: {exit_reason}, P&L: ${trade.pnl:.2f} ({trade.pnl_percentage:.2%})")

        self.current_trade = None
        return trade

    def update_portfolio_value(self, current_price: float):
        """
        Update portfolio value with current market price.

        Args:
            current_price: Current market price
        """
        if self.current_trade and self.current_trade.is_open():
            position_value = self.current_trade.quantity * current_price
            self.portfolio_value = self.cash + position_value
        else:
            self.portfolio_value = self.cash

    def get_current_exposure(self) -> float:
        """
        Get current portfolio exposure as percentage.

        Returns:
            Exposure percentage (0.0 to 1.0)
        """
        if self.current_trade and self.current_trade.is_open():
            position_value = self.current_trade.quantity * self.current_trade.entry_price
            return position_value / self.portfolio_value
        return 0.0

    def get_risk_metrics(self) -> Dict:
        """
        Calculate risk metrics for all trades.

        Returns:
            Dictionary with risk metrics
        """
        if not self.trades:
            return {"error": "No trades to analyze"}

        closed_trades = [t for t in self.trades if not t.is_open()]

        if not closed_trades:
            return {"error": "No closed trades to analyze"}

        pnls = [t.pnl for t in closed_trades]
        pnl_percentages = [t.pnl_percentage for t in closed_trades]

        winning_trades = [t for t in closed_trades if t.pnl > 0]
        losing_trades = [t for t in closed_trades if t.pnl < 0]

        total_pnl = sum(pnls)
        total_return = (self.portfolio_value - self.config.initial_capital) / self.config.initial_capital

        metrics = {
            'total_trades': len(closed_trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(closed_trades) if closed_trades else 0,
            'total_pnl': total_pnl,
            'total_return': total_return,
            'average_pnl': np.mean(pnls),
            'average_return': np.mean(pnl_percentages),
            'best_trade': max(pnls) if pnls else 0,
            'worst_trade': min(pnls) if pnls else 0,
            'profit_factor': sum([t.pnl for t in winning_trades]) / abs(sum([t.pnl for t in losing_trades])) if losing_trades else float('inf'),
            'max_consecutive_losses': self._calculate_max_consecutive_losses(closed_trades),
            'current_portfolio_value': self.portfolio_value,
            'current_cash': self.cash
        }

        return metrics

    def _calculate_max_consecutive_losses(self, trades) -> int:
        """Calculate maximum consecutive losing trades."""
        max_consecutive = 0
        current_consecutive = 0

        for trade in trades:
            if trade.pnl < 0:
                current_consecutive += 1
                max_consecutive = max(max_consecutive, current_consecutive)
            else:
                current_consecutive = 0

        return max_consecutive

    def get_trade_summary(self) -> pd.DataFrame:
        """
        Get summary of all trades as DataFrame.

        Returns:
            DataFrame with trade details
        """
        if not self.trades:
            return pd.DataFrame()

        trade_data = []
        for trade in self.trades:
            trade_data.append({
                'entry_date': trade.entry_date,
                'entry_price': trade.entry_price,
                'exit_date': trade.exit_date,
                'exit_price': trade.exit_price,
                'quantity': trade.quantity,
                'exit_reason': trade.exit_reason,
                'pnl': trade.pnl,
                'pnl_percentage': trade.pnl_percentage,
                'stop_loss_price': trade.stop_loss_price,
                'take_profit_price': trade.take_profit_price
            })

        return pd.DataFrame(trade_data)