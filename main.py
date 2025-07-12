#!/usr/bin/env python3
"""
Main CLI entry point for the quantitative trading system.
Orchestrates the complete pipeline from data download to backtesting.
"""

import argparse
import sys
import warnings
from datetime import datetime
import os

# Import our modules
from data import download_market_data
from features import prepare_features_and_labels
from model import train_trading_model
from backtest import run_backtest

warnings.filterwarnings('ignore')

def main():
    """Main function to run the complete trading pipeline."""
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Quantitative Trading System with LightGBM + Meta-Labeling',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --days 56 --horizon 30 --eta 0.001
  python main.py --days 42 --horizon 20 --eta auto
  python main.py --help
        """
    )
    
    parser.add_argument('--days', type=int, default=56,
                       help='Number of days to download (default: 56)')
    
    parser.add_argument('--horizon', type=int, default=30,
                       help='Triple barrier horizon in minutes (default: 30)')
    
    parser.add_argument('--eta', type=str, default='auto',
                       help='Triple barrier threshold (default: auto, or specify float)')
    
    parser.add_argument('--train-days', type=int, default=42,
                       help='Training period in days (default: 42)')
    
    parser.add_argument('--val-days', type=int, default=7,
                       help='Validation period in days (default: 7)')
    
    parser.add_argument('--test-days', type=int, default=7,
                       help='Test period in days (default: 7)')
    
    parser.add_argument('--meta-threshold', type=float, default=0.55,
                       help='Meta-labeling threshold (default: 0.55)')
    
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    parser.add_argument('--no-report', action='store_true',
                       help='Skip generating JSON report')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Verbose output')
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.days < args.train_days + args.val_days + args.test_days:
        print(f"Error: Total days ({args.days}) must be >= train + val + test days "
              f"({args.train_days + args.val_days + args.test_days})")
        sys.exit(1)
    
    # Parse eta parameter
    if args.eta == 'auto':
        eta = None
    else:
        try:
            eta = float(args.eta)
        except ValueError:
            print(f"Error: eta must be 'auto' or a float, got '{args.eta}'")
            sys.exit(1)
    
    # Print configuration
    print("="*80)
    print("QUANTITATIVE TRADING SYSTEM")
    print("="*80)
    print(f"Configuration:")
    print(f"  Data days: {args.days}")
    print(f"  Horizon: {args.horizon} minutes")
    print(f"  Eta: {args.eta}")
    print(f"  Training: {args.train_days} days")
    print(f"  Validation: {args.val_days} days")
    print(f"  Testing: {args.test_days} days")
    print(f"  Meta threshold: {args.meta_threshold}")
    print(f"  Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    try:
        # Step 1: Download and prepare data
        print("STEP 1: Downloading market data...")
        print("-" * 40)
        data, spreads = download_market_data(days=args.days)
        
        if args.verbose:
            print(f"Downloaded data shape: {data.shape}")
            print(f"Date range: {data.index.min()} to {data.index.max()}")
            print(f"Spreads: {spreads}")
        
        # Step 2: Feature engineering and labeling
        print("\nSTEP 2: Feature engineering and labeling...")
        print("-" * 40)
        labeled_data = prepare_features_and_labels(
            data, spreads, horizon_minutes=args.horizon, eta=eta
        )
        
        if args.verbose:
            print(f"Labeled data shape: {labeled_data.shape}")
            print(f"Feature columns: {len([c for c in labeled_data.columns if c != 'label'])}")
            print(f"Label distribution: {labeled_data['label'].value_counts().to_dict()}")
        
        # Step 3: Model training
        print("\nSTEP 3: Training models...")
        print("-" * 40)
        model, training_results = train_trading_model(
            labeled_data, 
            train_days=args.train_days, 
            val_days=args.val_days
        )
        
        if args.verbose:
            print("Base model feature importance (top 10):")
            base_importance = training_results['base_model'].get_feature_importance()
            print(base_importance.head(10))
            
            print("\nMeta-model feature importance (top 10):")
            meta_importance = training_results['meta_model'].get_feature_importance()
            print(meta_importance.head(10))
        
        # Step 4: Backtesting
        print("\nSTEP 4: Running backtest...")
        print("-" * 40)
        backtest_results = run_backtest(
            labeled_data, 
            model, 
            test_days=args.test_days,
            meta_threshold=args.meta_threshold,
            save_plots=not args.no_plots,
            save_report=not args.no_report
        )
        
        # Step 5: Summary
        print("\nSTEP 5: Summary...")
        print("-" * 40)
        
        metrics = backtest_results['metrics']
        
        print(f"Strategy Performance:")
        print(f"  Total Return: {metrics['total_return']:.2%}")
        print(f"  Annualized Return: {metrics['annualized_return']:.2%}")
        print(f"  Sharpe Ratio: {metrics['sharpe_ratio']:.3f}")
        print(f"  Sortino Ratio: {metrics['sortino_ratio']:.3f}")
        print(f"  Max Drawdown: {metrics['max_drawdown']:.2%}")
        print(f"  Hit Rate: {metrics['hit_rate']:.2%}")
        print(f"  Number of Trades: {metrics['num_trades']}")
        print(f"  Prediction Accuracy: {metrics['prediction_accuracy']:.2%}")
        print(f"  Balanced Accuracy: {metrics['balanced_accuracy']:.2%}")
        
        # Check if files were created
        if not args.no_plots:
            if os.path.exists('./figs'):
                print(f"\nPlots saved to ./figs/")
            
        if not args.no_report:
            if os.path.exists('backtest_report.json'):
                print(f"Report saved to backtest_report.json")
        
        print(f"\nPipeline completed successfully!")
        print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        
    except Exception as e:
        print(f"\nError occurred: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)

def run_demo():
    """Run a quick demo with minimal data."""
    print("Running demo with minimal data...")
    
    # Override default parameters for demo
    sys.argv = ['main.py', '--days', '14', '--train-days', '7', '--val-days', '3', '--test-days', '3', '--verbose']
    main()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # If no arguments provided, show help
        print("No arguments provided. Use --help for usage information.")
        print("Or run with default parameters:")
        print("  python main.py --days 56 --horizon 30 --eta auto")
        print()
        
        # Ask if user wants to run demo
        response = input("Would you like to run a quick demo? (y/n): ").lower().strip()
        if response in ['y', 'yes']:
            run_demo()
        else:
            print("Exiting. Use --help for usage information.")
    else:
        main()