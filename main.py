#!/usr/bin/env python3
"""
Trading System CLI Entry Point

Usage:
    python main.py --days 56 --horizon 30 --eta 0.0005
    python main.py --days 28 --horizon 15 --eta 0.001
"""

import argparse
import sys
import os
import warnings
warnings.filterwarnings('ignore')

# Import our modules
from data import download_data, get_spread_estimate
from features import prepare_features_and_labels
from model import TradingModel
from backtest import Backtester

def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description='End-to-end trading system with triple-barrier labeling and meta-labeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --days 56 --horizon 30 --eta 0.0005
  python main.py --days 28 --horizon 15 --eta 0.001 --symbol SPY
  python main.py --days 42 --horizon 20 --eta auto
        """
    )
    
    parser.add_argument(
        '--days', 
        type=int, 
        default=56,
        help='Number of days to download (default: 56)'
    )
    
    parser.add_argument(
        '--horizon', 
        type=int, 
        default=30,
        help='Prediction horizon in 5-minute periods (default: 30)'
    )
    
    parser.add_argument(
        '--eta', 
        type=str, 
        default='auto',
        help='Triple-barrier threshold (default: auto, or specify float like 0.0005)'
    )
    
    parser.add_argument(
        '--symbol', 
        type=str, 
        default='SPY',
        help='Trading symbol (default: SPY)'
    )
    
    parser.add_argument(
        '--train-days', 
        type=int, 
        default=42,
        help='Number of days for training (default: 42)'
    )
    
    parser.add_argument(
        '--val-days', 
        type=int, 
        default=7,
        help='Number of days for validation (default: 7)'
    )
    
    parser.add_argument(
        '--slippage', 
        type=int, 
        default=1,
        help='Slippage in ticks (default: 1)'
    )
    
    parser.add_argument(
        '--commission', 
        type=float, 
        default=0.0,
        help='Commission per trade (default: 0.0)'
    )
    
    parser.add_argument(
        '--capital', 
        type=float, 
        default=100000,
        help='Initial capital (default: 100000)'
    )
    
    parser.add_argument(
        '--meta-threshold', 
        type=float, 
        default=0.55,
        help='Meta-model confidence threshold (default: 0.55)'
    )
    
    parser.add_argument(
        '--no-plots', 
        action='store_true',
        help='Skip generating plots'
    )
    
    parser.add_argument(
        '--output', 
        type=str, 
        default='backtest_report.json',
        help='Output JSON report filename (default: backtest_report.json)'
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("TRADING SYSTEM - Triple-Barrier + Meta-Labeling")
    print("=" * 60)
    print(f"Configuration:")
    print(f"  Days: {args.days}")
    print(f"  Horizon: {args.horizon} periods")
    print(f"  Eta: {args.eta}")
    print(f"  Symbol: {args.symbol}")
    print(f"  Train days: {args.train_days}")
    print(f"  Val days: {args.val_days}")
    print(f"  Slippage: {args.slippage} ticks")
    print(f"  Commission: {args.commission}")
    print(f"  Capital: ${args.capital:,.0f}")
    print(f"  Meta threshold: {args.meta_threshold}")
    print("=" * 60)
    
    try:
        # Step 1: Download data
        print("\n1. Downloading data...")
        data = download_data(symbols=[args.symbol, '^VIX', 'DX-Y.NYB'], days=args.days)
        
        # Step 2: Calculate eta if auto
        if args.eta == 'auto':
            spread_estimate = get_spread_estimate(data, args.symbol)
            eta = max(0.0005, 5 * spread_estimate)
            print(f"Auto-calculated eta: {eta:.6f} (based on spread estimate: {spread_estimate:.6f})")
        else:
            eta = float(args.eta)
        
        # Step 3: Engineer features and create labels
        print("\n2. Engineering features and creating labels...")
        features, labels, prices = prepare_features_and_labels(
            data, 
            symbol=args.symbol, 
            horizon=args.horizon, 
            eta=eta
        )
        
        # Step 4: Train models
        print("\n3. Training models...")
        model = TradingModel(
            train_days=args.train_days,
            val_days=args.val_days,
            n_splits=5
        )
        
        training_results = model.train(features, labels, prices)
        
        # Step 5: Generate signals for out-of-sample testing
        print("\n4. Generating trading signals...")
        
        # Use the last validation period for OOS testing
        total_samples = len(features)
        val_periods = args.val_days * 24 * 12  # 12 5-minute periods per hour
        oos_start = total_samples - val_periods
        
        if oos_start < 0:
            print("Warning: Not enough data for OOS testing, using all data")
            oos_features = features
            oos_labels = labels
            oos_prices = prices
        else:
            oos_features = features.iloc[oos_start:]
            oos_labels = labels.iloc[oos_start:]
            oos_prices = prices.iloc[oos_start:]
        
        # Create meta-features for OOS data
        base_probs = model.base_model.predict(oos_features)
        oos_meta_features = model.create_meta_features(base_probs, oos_features, oos_labels, oos_prices)
        
        # Generate signals
        signals = model.generate_signals(oos_features, oos_meta_features, threshold=args.meta_threshold)
        
        # Step 6: Run backtest
        print("\n5. Running backtest...")
        backtester = Backtester(
            slippage_ticks=args.slippage,
            commission=args.commission,
            initial_capital=args.capital
        )
        
        backtest_results = backtester.run_backtest(
            oos_prices, 
            signals, 
            true_labels=oos_labels,
            save_plots=not args.no_plots
        )
        
        # Step 7: Save report
        print("\n6. Saving results...")
        backtester.save_report(backtest_results, filename=args.output)
        
        # Step 8: Print final summary
        print("\n" + "=" * 60)
        print("FINAL RESULTS SUMMARY")
        print("=" * 60)
        metrics = backtest_results['metrics']
        
        print(f"Out-of-Sample Period: {oos_prices.index[0]} to {oos_prices.index[-1]}")
        print(f"Total Return: {metrics['total_return']:.4f}")
        print(f"Final Portfolio Value: ${metrics['final_value']:,.2f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Sortino Ratio: {metrics['sortino_ratio']:.4f}")
        print(f"Maximum Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"Hit Rate: {metrics['hit_rate']:.4f}")
        print(f"Win/Loss Ratio: {metrics['win_loss_ratio']:.4f}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        if 'signal_accuracy' in metrics:
            print(f"Signal Accuracy: {metrics['signal_accuracy']:.4f}")
            print(f"Signal Balanced Accuracy: {metrics['signal_balanced_accuracy']:.4f}")
        
        print(f"\nFiles generated:")
        if not args.no_plots:
            print(f"  - ./figs/backtest_results.png")
        print(f"  - {args.output}")
        
        print("\nTrading system completed successfully!")
        
    except Exception as e:
        print(f"\nError: {e}")
        print("Trading system failed. Please check your parameters and try again.")
        sys.exit(1)

if __name__ == "__main__":
    main()