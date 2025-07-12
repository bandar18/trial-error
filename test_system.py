#!/usr/bin/env python3
"""
Quick test script for the trading system.
This runs a smaller version to verify everything works.
"""

import warnings
warnings.filterwarnings('ignore')

from data import download_data, get_spread_estimate
from features import prepare_features_and_labels
from model import TradingModel
from backtest import Backtester

def test_system():
    """Test the complete trading system with smaller parameters."""
    print("=" * 50)
    print("TRADING SYSTEM TEST")
    print("=" * 50)
    
    try:
        # Step 1: Download data (smaller dataset for testing)
        print("1. Downloading test data...")
        data = download_data(symbols=['SPY', '^VIX', 'DX-Y.NYB'], days=14)
        print(f"Downloaded data shape: {data.shape}")
        
        # Step 2: Calculate eta
        spread_estimate = get_spread_estimate(data, 'SPY')
        eta = max(0.0005, 5 * spread_estimate)
        print(f"Calculated eta: {eta:.6f}")
        
        # Step 3: Engineer features and create labels
        print("2. Engineering features and creating labels...")
        features, labels, prices = prepare_features_and_labels(
            data, 
            symbol='SPY', 
            horizon=10,  # Shorter horizon for testing
            eta=eta
        )
        print(f"Features shape: {features.shape}")
        print(f"Labels distribution: {labels.value_counts().to_dict()}")
        
        # Step 4: Train models (smaller training set)
        print("3. Training models...")
        model = TradingModel(
            train_days=7,
            val_days=3,
            n_splits=2
        )
        
        training_results = model.train(features, labels, prices)
        print("Model training completed!")
        
        # Step 5: Generate signals
        print("4. Generating signals...")
        base_probs = model.base_model.predict(features)
        meta_features = model.create_meta_features(base_probs, features, labels, prices)
        signals = model.generate_signals(features, meta_features, threshold=0.55)
        print(f"Signal distribution: {signals.value_counts().to_dict()}")
        
        # Step 6: Run backtest
        print("5. Running backtest...")
        backtester = Backtester(
            slippage_ticks=1,
            commission=0.0,
            initial_capital=100000
        )
        
        backtest_results = backtester.run_backtest(
            prices, 
            signals, 
            true_labels=labels,
            save_plots=True
        )
        
        # Step 7: Print results
        print("\n" + "=" * 50)
        print("TEST RESULTS")
        print("=" * 50)
        metrics = backtest_results['metrics']
        
        print(f"Total Return: {metrics['total_return']:.4f}")
        print(f"Sharpe Ratio: {metrics['sharpe_ratio']:.4f}")
        print(f"Max Drawdown: {metrics['max_drawdown']:.4f}")
        print(f"Hit Rate: {metrics['hit_rate']:.4f}")
        print(f"Total Trades: {metrics['total_trades']}")
        
        if 'signal_accuracy' in metrics:
            print(f"Signal Accuracy: {metrics['signal_accuracy']:.4f}")
        
        print("\nTest completed successfully!")
        print("Check ./figs/backtest_results.png for plots")
        
    except Exception as e:
        print(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_system()