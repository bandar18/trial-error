"""
Backtesting module for trading strategy.
Handles signal generation, P&L calculation, and performance metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

class TradingBacktester:
    """Backtesting engine for trading strategies."""
    
    def __init__(self, data, model, initial_capital=100000, slippage_ticks=1, commission=0.0):
        self.data = data.copy()
        self.model = model
        self.initial_capital = initial_capital
        self.slippage_ticks = slippage_ticks
        self.commission = commission
        self.results = None
        
    def run_backtest(self, test_days=7, meta_threshold=0.55):
        """Run the complete backtest."""
        print(f"Running backtest on final {test_days} days (OOS)...")
        
        # Get out-of-sample data
        total_periods = len(self.data)
        periods_per_day = total_periods // 56
        test_size = int(test_days * periods_per_day)
        
        test_data = self.data.iloc[-test_size:].copy()
        
        # Prepare features
        feature_cols = [col for col in test_data.columns if col != 'label']
        X_test = test_data[feature_cols]
        y_test = test_data['label']
        
        print(f"Test set: {len(X_test)} samples")
        
        # Generate signals
        signals, base_pred, base_probs, meta_probs = self.model.generate_signals(
            X_test, meta_threshold=meta_threshold
        )
        
        # Calculate P&L
        pnl_results = self._calculate_pnl(test_data, signals)
        
        # Calculate performance metrics
        metrics = self._calculate_metrics(pnl_results, signals, y_test, base_pred)
        
        # Store results
        self.results = {
            'test_data': test_data,
            'signals': signals,
            'base_predictions': base_pred,
            'base_probabilities': base_probs,
            'meta_probabilities': meta_probs,
            'pnl_results': pnl_results,
            'metrics': metrics,
            'parameters': {
                'test_days': test_days,
                'meta_threshold': meta_threshold,
                'initial_capital': self.initial_capital,
                'slippage_ticks': self.slippage_ticks,
                'commission': self.commission
            }
        }
        
        # Print results
        self._print_results()
        
        return self.results
    
    def _calculate_pnl(self, data, signals):
        """Calculate P&L with slippage and commission."""
        # Get price data
        price_col = 'SPY_Close'
        if price_col not in data.columns:
            raise ValueError(f"Price column {price_col} not found")
        
        prices = data[price_col].values
        
        # Calculate tick size (approximate)
        tick_size = 0.01  # $0.01 for SPY
        
        # Initialize tracking variables
        position = 0
        cash = self.initial_capital
        equity = [self.initial_capital]
        trades = []
        
        for i in range(len(signals)):
            signal = signals[i]
            price = prices[i]
            
            # Calculate slippage
            slippage_cost = self.slippage_ticks * tick_size
            
            # Position changes
            if signal != position:
                # Close existing position
                if position != 0:
                    exit_price = price + (slippage_cost if position > 0 else -slippage_cost)
                    trade_pnl = position * (exit_price - entry_price)
                    cash += trade_pnl
                    
                    trades.append({
                        'entry_time': entry_time,
                        'exit_time': data.index[i],
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'position': position,
                        'pnl': trade_pnl,
                        'duration': i - entry_idx
                    })
                
                # Open new position
                if signal != 0:
                    entry_price = price + (slippage_cost if signal > 0 else -slippage_cost)
                    entry_time = data.index[i]
                    entry_idx = i
                
                position = signal
            
            # Calculate current equity
            if position != 0:
                current_price = price
                unrealized_pnl = position * (current_price - entry_price)
                current_equity = cash + unrealized_pnl
            else:
                current_equity = cash
            
            equity.append(current_equity)
        
        # Close final position if any
        if position != 0:
            exit_price = prices[-1] + (slippage_cost if position > 0 else -slippage_cost)
            trade_pnl = position * (exit_price - entry_price)
            cash += trade_pnl
            
            trades.append({
                'entry_time': entry_time,
                'exit_time': data.index[-1],
                'entry_price': entry_price,
                'exit_price': exit_price,
                'position': position,
                'pnl': trade_pnl,
                'duration': len(signals) - entry_idx
            })
        
        return {
            'equity_curve': pd.Series(equity[1:], index=data.index),
            'trades': trades,
            'final_cash': cash,
            'total_return': (cash - self.initial_capital) / self.initial_capital
        }
    
    def _calculate_metrics(self, pnl_results, signals, y_true, y_pred):
        """Calculate comprehensive performance metrics."""
        equity_curve = pnl_results['equity_curve']
        trades = pnl_results['trades']
        
        # Returns
        returns = equity_curve.pct_change().dropna()
        
        # Basic metrics
        total_return = pnl_results['total_return']
        annualized_return = (1 + total_return) ** (252 / len(equity_curve)) - 1
        
        # Risk metrics
        volatility = returns.std() * np.sqrt(252 * 288)  # Annualized (288 5-min periods per day)
        sharpe_ratio = annualized_return / volatility if volatility > 0 else 0
        
        # Sortino ratio
        downside_returns = returns[returns < 0]
        downside_volatility = downside_returns.std() * np.sqrt(252 * 288)
        sortino_ratio = annualized_return / downside_volatility if downside_volatility > 0 else 0
        
        # Maximum drawdown
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Trade-based metrics
        if trades:
            winning_trades = [t for t in trades if t['pnl'] > 0]
            losing_trades = [t for t in trades if t['pnl'] < 0]
            
            hit_rate = len(winning_trades) / len(trades)
            avg_win = np.mean([t['pnl'] for t in winning_trades]) if winning_trades else 0
            avg_loss = np.mean([t['pnl'] for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win * len(winning_trades) / (avg_loss * len(losing_trades))) if losing_trades else np.inf
        else:
            hit_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Prediction accuracy
        from sklearn.metrics import accuracy_score, balanced_accuracy_score
        
        # Only consider non-zero predictions for accuracy
        non_zero_mask = y_pred != 0
        if non_zero_mask.sum() > 0:
            prediction_accuracy = accuracy_score(y_true[non_zero_mask], y_pred[non_zero_mask])
            balanced_accuracy = balanced_accuracy_score(y_true[non_zero_mask], y_pred[non_zero_mask])
        else:
            prediction_accuracy = 0
            balanced_accuracy = 0
        
        # Signal statistics
        signal_stats = pd.Series(signals).value_counts()
        
        metrics = {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'max_drawdown': max_drawdown,
            'hit_rate': hit_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'num_trades': len(trades),
            'prediction_accuracy': prediction_accuracy,
            'balanced_accuracy': balanced_accuracy,
            'signal_distribution': signal_stats.to_dict()
        }
        
        return metrics
    
    def _print_results(self):
        """Print backtest results."""
        if not self.results:
            return
        
        metrics = self.results['metrics']
        
        print("\n" + "="*60)
        print("BACKTEST RESULTS")
        print("="*60)
        
        print(f"Total Return: {metrics['total_return']:.2%}")
        print(f"Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"Volatility: {metrics['volatility']:.2%}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.2%}")
        
        print(f"\nTrading Metrics:")
        print(f"Number of Trades: {metrics['num_trades']}")
        print(f"Hit Rate: {metrics['hit_rate']:.2%}")
        print(f"Average Win: ${metrics['avg_win']:.2f}")
        print(f"Average Loss: ${metrics['avg_loss']:.2f}")
        print(f"Profit Factor: {metrics['profit_factor']:.2f}")
        
        print(f"\nPrediction Accuracy:")
        print(f"Accuracy: {metrics['prediction_accuracy']:.2%}")
        print(f"Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
        
        print(f"\nSignal Distribution:")
        for signal, count in metrics['signal_distribution'].items():
            print(f"  {signal}: {count} ({count/sum(metrics['signal_distribution'].values()):.1%})")
    
    def plot_results(self, save_path='./figs'):
        """Plot backtest results."""
        if not self.results:
            print("No results to plot")
            return
        
        # Create figures directory
        os.makedirs(save_path, exist_ok=True)
        
        # Plot equity curve
        plt.figure(figsize=(12, 8))
        
        # Equity curve
        plt.subplot(2, 1, 1)
        equity_curve = self.results['pnl_results']['equity_curve']
        plt.plot(equity_curve.index, equity_curve.values, 'b-', linewidth=2)
        plt.title('Equity Curve')
        plt.ylabel('Portfolio Value ($)')
        plt.grid(True, alpha=0.3)
        
        # Drawdown
        plt.subplot(2, 1, 2)
        returns = equity_curve.pct_change().dropna()
        cumulative = (1 + returns).cumprod()
        rolling_max = cumulative.expanding().max()
        drawdown = (cumulative - rolling_max) / rolling_max
        
        plt.fill_between(drawdown.index, drawdown.values, 0, alpha=0.3, color='red')
        plt.plot(drawdown.index, drawdown.values, 'r-', linewidth=1)
        plt.title('Drawdown')
        plt.ylabel('Drawdown (%)')
        plt.xlabel('Time')
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'equity_and_drawdown.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        # Plot signals
        plt.figure(figsize=(12, 6))
        
        test_data = self.results['test_data']
        signals = self.results['signals']
        
        # Price and signals
        prices = test_data['SPY_Close']
        plt.plot(prices.index, prices.values, 'k-', linewidth=1, label='SPY Price')
        
        # Long signals
        long_mask = signals == 1
        if long_mask.sum() > 0:
            plt.scatter(prices.index[long_mask], prices.values[long_mask], 
                       color='green', marker='^', s=50, label='Long Signal')
        
        # Short signals
        short_mask = signals == -1
        if short_mask.sum() > 0:
            plt.scatter(prices.index[short_mask], prices.values[short_mask], 
                       color='red', marker='v', s=50, label='Short Signal')
        
        plt.title('Trading Signals')
        plt.ylabel('Price ($)')
        plt.xlabel('Time')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(save_path, 'signals.png'), dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plots saved to {save_path}")
    
    def save_report(self, filename='backtest_report.json'):
        """Save backtest report as JSON."""
        if not self.results:
            print("No results to save")
            return
        
        # Prepare report data
        report = {
            'timestamp': datetime.now().isoformat(),
            'parameters': self.results['parameters'],
            'metrics': self.results['metrics'],
            'trades': self.results['pnl_results']['trades']
        }
        
        # Convert numpy types to native Python types
        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        report = convert_numpy(report)
        
        # Save to file
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"Report saved to {filename}")

def run_backtest(data, model, test_days=7, meta_threshold=0.55, save_plots=True, save_report=True):
    """Convenience function to run complete backtest."""
    backtester = TradingBacktester(data, model)
    results = backtester.run_backtest(test_days=test_days, meta_threshold=meta_threshold)
    
    if save_plots:
        backtester.plot_results()
    
    if save_report:
        backtester.save_report()
    
    return results

if __name__ == "__main__":
    # Test the backtester
    from data import download_market_data
    from features import prepare_features_and_labels
    from model import train_trading_model
    
    print("Testing backtester...")
    
    # Download and prepare data
    data, spreads = download_market_data(days=21)
    labeled_data = prepare_features_and_labels(data, spreads)
    
    # Train model
    model, training_results = train_trading_model(labeled_data, train_days=10, val_days=3)
    
    # Run backtest
    backtest_results = run_backtest(labeled_data, model, test_days=3)
    
    print("\nBacktest completed!")