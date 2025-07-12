import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report

class Backtester:
    def __init__(self, slippage_ticks=1, commission=0.0, initial_capital=100000):
        """
        Initialize the backtester.
        
        Args:
            slippage_ticks: Number of ticks for slippage
            commission: Commission per trade
            initial_capital: Initial capital
        """
        self.slippage_ticks = slippage_ticks
        self.commission = commission
        self.initial_capital = initial_capital
        
    def calculate_slippage(self, prices, signal):
        """
        Calculate slippage impact on trade execution.
        
        Args:
            prices: Price series
            signal: Trading signal (-1, 0, 1)
        
        Returns:
            Execution prices with slippage
        """
        # Estimate tick size (assume $0.01 for SPY)
        tick_size = 0.01
        
        execution_prices = prices.copy()
        
        # Apply slippage for buy/sell signals
        buy_mask = signal == 1
        sell_mask = signal == -1
        
        # Buy orders execute at higher price (slippage up)
        execution_prices[buy_mask] = prices[buy_mask] + (self.slippage_ticks * tick_size)
        
        # Sell orders execute at lower price (slippage down)
        execution_prices[sell_mask] = prices[sell_mask] - (self.slippage_ticks * tick_size)
        
        return execution_prices
    
    def calculate_returns(self, prices, signals):
        """
        Calculate trade returns with slippage and commissions.
        
        Args:
            prices: Price series
            signals: Trading signal series
        
        Returns:
            Returns series
        """
        # Calculate execution prices with slippage
        execution_prices = self.calculate_slippage(prices, signals)
        
        # Calculate position changes
        position_changes = signals.diff()
        
        # Calculate returns
        returns = pd.Series(0.0, index=prices.index)
        
        # For each trade
        for i in range(1, len(prices)):
            if position_changes.iloc[i] != 0:  # Position change
                if position_changes.iloc[i] > 0:  # Buy
                    # Cost of buying
                    cost = position_changes.iloc[i] * execution_prices.iloc[i] * (1 + self.commission)
                    returns.iloc[i] = -cost
                else:  # Sell
                    # Proceeds from selling
                    proceeds = abs(position_changes.iloc[i]) * execution_prices.iloc[i] * (1 - self.commission)
                    returns.iloc[i] = proceeds
        
        # Add unrealized P&L for current position
        positions = signals.cumsum()
        price_changes = prices.pct_change()
        unrealized_pnl = positions.shift(1) * price_changes
        
        returns = returns + unrealized_pnl
        
        return returns
    
    def calculate_metrics(self, returns, signals, true_labels=None):
        """
        Calculate performance metrics.
        
        Args:
            returns: Returns series
            signals: Trading signal series
            true_labels: True labels for accuracy calculation
        
        Returns:
            Dictionary of metrics
        """
        # Basic return metrics
        total_return = returns.sum()
        cumulative_return = (1 + returns).cumprod()
        final_value = self.initial_capital * cumulative_return.iloc[-1]
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 288)  # Annualized (288 5-min periods per day)
        sharpe_ratio = (returns.mean() * 252 * 288) / volatility if volatility > 0 else 0
        
        # Sortino ratio (downside deviation)
        downside_returns = returns[returns < 0]
        downside_deviation = downside_returns.std() * np.sqrt(252 * 288) if len(downside_returns) > 0 else 0
        sortino_ratio = (returns.mean() * 252 * 288) / downside_deviation if downside_deviation > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Hit rate
        winning_trades = returns[returns > 0]
        losing_trades = returns[returns < 0]
        total_trades = len(winning_trades) + len(losing_trades)
        hit_rate = len(winning_trades) / total_trades if total_trades > 0 else 0
        
        # Average win/loss
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades.mean() if len(losing_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        
        # Signal statistics
        signal_counts = signals.value_counts()
        signal_ratio = signal_counts[1] / (signal_counts[1] + signal_counts[-1]) if (1 in signal_counts and -1 in signal_counts) else 0
        
        # Accuracy metrics (if true labels provided)
        accuracy_metrics = {}
        if true_labels is not None:
            # Filter to non-zero signals for accuracy calculation
            signal_mask = signals != 0
            if signal_mask.sum() > 0:
                signal_accuracy = accuracy_score(true_labels[signal_mask], signals[signal_mask])
                signal_balanced_accuracy = balanced_accuracy_score(true_labels[signal_mask], signals[signal_mask])
                accuracy_metrics = {
                    'signal_accuracy': signal_accuracy,
                    'signal_balanced_accuracy': signal_balanced_accuracy
                }
        
        metrics = {
            'total_return': total_return,
            'final_value': final_value,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'win_loss_ratio': win_loss_ratio,
            'total_trades': total_trades,
            'signal_ratio': signal_ratio,
            'avg_win': avg_win,
            'avg_loss': avg_loss
        }
        
        metrics.update(accuracy_metrics)
        
        return metrics, cumulative_return, drawdown
    
    def run_backtest(self, prices, signals, true_labels=None, save_plots=True):
        """
        Run complete backtest.
        
        Args:
            prices: Price series
            signals: Trading signal series
            true_labels: True labels for accuracy calculation
            save_plots: Whether to save plots
        
        Returns:
            Dictionary with backtest results
        """
        print("Running backtest...")
        
        # Calculate returns
        returns = self.calculate_returns(prices, signals)
        
        # Calculate metrics
        metrics, cumulative_return, drawdown = self.calculate_metrics(returns, signals, true_labels)
        
        # Print results
        print("\n=== Backtest Results ===")
        print(f"Total Return: {metrics['total_return']:.4f}")
        print(f"Final Value: ${metrics['final_value']:,.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"Hit Rate: {metrics['hit_rate']:.4f}")
        print(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.4f}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        if 'signal_accuracy' in metrics:
            print(f"Signal Accuracy: {metrics['signal_accuracy']:.4f}")
            print(f"Signal Balanced Accuracy: {metrics['signal_balanced_accuracy']:.4f}")
        
        # Create plots
        if save_plots:
            self.create_plots(prices, signals, returns, cumulative_return, drawdown)
        
        return {
            'metrics': metrics,
            'returns': returns,
            'cumulative_return': cumulative_return,
            'drawdown': drawdown,
            'signals': signals
        }
    
    def create_plots(self, prices, signals, returns, cumulative_return, drawdown):
        """
        Create and save backtest plots.
        
        Args:
            prices: Price series
            signals: Trading signal series
            returns: Returns series
            cumulative_return: Cumulative return series
            drawdown: Drawdown series
        """
        print("Creating plots...")
        
        # Create figures directory
        os.makedirs('./figs', exist_ok=True)
        
        # Plot 1: Equity Curve
        plt.figure(figsize=(12, 8))
        
        plt.subplot(2, 2, 1)
        plt.plot(cumulative_return.index, cumulative_return.values, 'b-', linewidth=2)
        plt.title('Equity Curve')
        plt.ylabel('Cumulative Return')
        plt.grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        plt.subplot(2, 2, 2)
        plt.fill_between(drawdown.index, drawdown.values, 0, color='red', alpha=0.3)
        plt.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        plt.title('Drawdown')
        plt.ylabel('Drawdown')
        plt.grid(True, alpha=0.3)
        
        # Plot 3: Price and Signals
        plt.subplot(2, 2, 3)
        plt.plot(prices.index, prices.values, 'k-', linewidth=1, alpha=0.7, label='Price')
        
        # Plot buy signals
        buy_signals = signals[signals == 1]
        if len(buy_signals) > 0:
            plt.scatter(buy_signals.index, prices.loc[buy_signals.index], 
                       color='green', marker='^', s=50, label='Buy Signal')
        
        # Plot sell signals
        sell_signals = signals[signals == -1]
        if len(sell_signals) > 0:
            plt.scatter(sell_signals.index, prices.loc[sell_signals.index], 
                       color='red', marker='v', s=50, label='Sell Signal')
        
        plt.title('Price and Trading Signals')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot 4: Returns Distribution
        plt.subplot(2, 2, 4)
        plt.hist(returns.values, bins=50, alpha=0.7, color='blue', edgecolor='black')
        plt.title('Returns Distribution')
        plt.xlabel('Return')
        plt.ylabel('Frequency')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('./figs/backtest_results.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Plots saved to ./figs/backtest_results.png")
    
    def save_report(self, results, filename='backtest_report.json'):
        """
        Save backtest results to JSON file.
        
        Args:
            results: Backtest results dictionary
            filename: Output filename
        """
        import json
        
        # Convert numpy types to native Python types for JSON serialization
        report = {}
        for key, value in results['metrics'].items():
            if isinstance(value, (np.integer, np.floating)):
                report[key] = float(value)
            else:
                report[key] = value
        
        # Add metadata
        report['metadata'] = {
            'slippage_ticks': self.slippage_ticks,
            'commission': self.commission,
            'initial_capital': self.initial_capital,
            'backtest_date': pd.Timestamp.now().isoformat()
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")

if __name__ == "__main__":
    # Test the backtester
    import pandas as pd
    import numpy as np
    
    # Create sample data
    dates = pd.date_range('2024-01-01', periods=1000, freq='5T')
    prices = pd.Series(100 + np.cumsum(np.random.randn(1000) * 0.1), index=dates)
    signals = pd.Series(np.random.choice([-1, 0, 1], 1000, p=[0.1, 0.8, 0.1]), index=dates)
    
    # Run backtest
    backtester = Backtester()
    results = backtester.run_backtest(prices, signals)
    
    print("Backtest completed successfully!")