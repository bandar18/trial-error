# Trading System - Triple-Barrier + Meta-Labeling

A complete end-to-end Python trading system that implements triple-barrier labeling and meta-labeling for algorithmic trading.

## Features

- **Data Download**: Downloads 1-minute SPY, VIX, and DXY data via yfinance with automatic stitching of 7-day windows
- **Feature Engineering**: Comprehensive technical indicators including returns ladder, volatility, ATR, RSI, MACD, volume z-score, and cross-asset features
- **Triple-Barrier Labeling**: Implements triple-barrier method with configurable horizon and thresholds
- **Meta-Labeling**: Two-stage model with LightGBM base model and RandomForest meta-model
- **Time Series Validation**: Proper time series cross-validation to prevent look-ahead bias
- **Backtesting**: Complete backtesting with slippage, commissions, and comprehensive risk metrics
- **Visualization**: Automatic generation of equity curves, drawdown charts, and performance plots

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

```bash
# Run with default parameters (56 days, 30-period horizon, auto eta)
python main.py

# Run with custom parameters
python main.py --days 42 --horizon 20 --eta 0.001

# Run with specific symbol and meta-threshold
python main.py --symbol SPY --meta-threshold 0.6 --slippage 2
```

### Command Line Arguments

- `--days`: Number of days to download (default: 56)
- `--horizon`: Prediction horizon in 5-minute periods (default: 30)
- `--eta`: Triple-barrier threshold (default: auto, or specify float like 0.0005)
- `--symbol`: Trading symbol (default: SPY)
- `--train-days`: Number of days for training (default: 42)
- `--val-days`: Number of days for validation (default: 7)
- `--slippage`: Slippage in ticks (default: 1)
- `--commission`: Commission per trade (default: 0.0)
- `--capital`: Initial capital (default: 100000)
- `--meta-threshold`: Meta-model confidence threshold (default: 0.55)
- `--no-plots`: Skip generating plots
- `--output`: Output JSON report filename (default: backtest_report.json)

## System Architecture

### 1. Data Module (`data.py`)
- Downloads 1-minute data for SPY, VIX, and DXY
- Stitches 7-day windows to overcome Yahoo Finance limits
- Handles data alignment and preprocessing

### 2. Features Module (`features.py`)
- Engineers technical indicators at 1-minute granularity
- Resamples to 5-minute data
- Implements triple-barrier labeling with configurable parameters
- Features include:
  - Price returns ladder (r₁, r₅, r₁₀, r₃₀)
  - Rolling volatility (10-minute window)
  - ATR (20-period)
  - RSI-14
  - MACD with signal and histogram
  - Volume z-score
  - Cross-asset features (ΔVIX%, ΔDXY%)

### 3. Model Module (`model.py`)
- LightGBM base model for triple-class classification
- Time series cross-validation to prevent overfitting
- RandomForest meta-model for signal confidence
- Meta-labeling: predicts whether base model predictions are correct

### 4. Backtest Module (`backtest.py`)
- Signal generation with meta-model confidence filtering
- P&L calculation with slippage and commissions
- Comprehensive risk metrics:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
  - Hit rate
  - Win/loss ratio
  - Balanced accuracy

## Output Files

- `./figs/backtest_results.png`: Performance visualization
- `backtest_report.json`: Detailed performance metrics and metadata

## Example Output

```
============================================================
TRADING SYSTEM - Triple-Barrier + Meta-Labeling
============================================================
Configuration:
  Days: 56
  Horizon: 30 periods
  Eta: auto
  Symbol: SPY
  Train days: 42
  Val days: 7
  Slippage: 1 ticks
  Commission: 0.0
  Capital: $100,000
  Meta threshold: 0.55
============================================================

=== Backtest Results ===
Total Return: 0.0234
Final Value: $102,340.00
Sharpe Ratio: 1.2345
Sortino Ratio: 1.8765
Max Drawdown: -0.0456
Hit Rate: 0.6234
Win/Loss Ratio: 1.3456
Total Trades: 156
Signal Accuracy: 0.6123
Signal Balanced Accuracy: 0.5987
```

## Technical Details

### Triple-Barrier Labeling
- Horizon: 30 minutes (6 periods at 5-minute granularity)
- Thresholds: η = max(0.0005, 5×avg-spread)
- Labels: y ∈ {-1, 0, +1} for short, flat, long

### Meta-Labeling
- Base model: LightGBM with multiclass objective
- Meta-model: RandomForest predicting base model correctness
- Signal generation: Only trade when meta-model confidence > threshold

### Risk Management
- 1-tick slippage simulation
- Configurable commission structure
- Proper position sizing and P&L tracking

## Dependencies

- pandas: Data manipulation and analysis
- numpy: Numerical computing
- yfinance: Yahoo Finance data download
- lightgbm: Gradient boosting framework
- scikit-learn: Machine learning utilities
- matplotlib: Plotting and visualization

## License

This project is for educational and research purposes. Use at your own risk for actual trading.

## Disclaimer

This trading system is for educational purposes only. Past performance does not guarantee future results. Always conduct thorough testing and risk management before using any trading strategy with real money.
