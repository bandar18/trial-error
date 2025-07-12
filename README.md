# Quantitative Trading System

A comprehensive end-to-end Python trading system that implements LightGBM with meta-labeling for SPY trading using cross-asset features from VIX and DXY.

## 🚀 Features

- **Multi-Asset Data Download**: Downloads 56 days of 1-minute data for SPY, ^VIX, and DXY via yfinance
- **Advanced Feature Engineering**: Creates price returns ladder, technical indicators, and cross-asset features
- **Triple-Barrier Labeling**: Implements sophisticated labeling with dynamic thresholds
- **Machine Learning Pipeline**: LightGBM base model with RandomForest meta-labeling
- **Comprehensive Backtesting**: Full P&L calculation with slippage, performance metrics, and visualization
- **Professional Reporting**: Generates plots and JSON reports for analysis

## 📁 Project Structure

```
├── data.py        # Data downloading and stitching
├── features.py    # Feature engineering + triple-barrier labeling
├── model.py       # LightGBM training + meta-labeling
├── backtest.py    # Signal generation + backtesting
├── main.py        # CLI entry point
├── requirements.txt
└── README.md
```

## 📊 Strategy Overview

1. **Data Collection**: Downloads 1-minute OHLCV data for SPY, ^VIX, DXY (8×7-day windows)
2. **Feature Engineering**: 
   - Price returns ladder (r₁, r₅, r₁₀, r₃₀)
   - Technical indicators (RSI-14, MACD, ATR-20, volatility)
   - Cross-asset features (ΔVIX%, ΔDXY%)
   - Resamples to 5-minute intervals
3. **Labeling**: Triple-barrier method with 30-minute horizon
4. **Model Training**: 
   - Base LightGBM model (directional prediction)
   - Meta-labeling RandomForest (prediction confidence)
5. **Signal Generation**: 
   - Long: base==+1 & meta P(correct)>0.55
   - Short: base==-1 & meta P(correct)>0.55
   - Flat: otherwise
6. **Backtesting**: 7-day out-of-sample test with realistic costs

## 🛠️ Installation

```bash
# Clone or download the files
# Install dependencies
pip install -r requirements.txt
```

## 🔧 Usage

### Command Line Interface

```bash
# Basic usage with default parameters
python main.py --days 56 --horizon 30 --eta auto

# Custom parameters
python main.py --days 42 --horizon 20 --eta 0.001 --meta-threshold 0.6

# Quick demo with minimal data
python main.py --days 14 --train-days 7 --val-days 3 --test-days 3 --verbose
```

### Available Arguments

- `--days`: Number of days to download (default: 56)
- `--horizon`: Triple barrier horizon in minutes (default: 30)
- `--eta`: Triple barrier threshold ('auto' or float, default: auto)
- `--train-days`: Training period in days (default: 42)
- `--val-days`: Validation period in days (default: 7)
- `--test-days`: Test period in days (default: 7)
- `--meta-threshold`: Meta-labeling threshold (default: 0.55)
- `--no-plots`: Skip generating plots
- `--no-report`: Skip generating JSON report
- `--verbose`: Verbose output

### Programmatic Usage

```python
from data import download_market_data
from features import prepare_features_and_labels
from model import train_trading_model
from backtest import run_backtest

# Download data
data, spreads = download_market_data(days=56)

# Prepare features and labels
labeled_data = prepare_features_and_labels(data, spreads, horizon_minutes=30)

# Train model
model, results = train_trading_model(labeled_data, train_days=42, val_days=7)

# Run backtest
backtest_results = run_backtest(labeled_data, model, test_days=7)
```

## 📈 Output

The system generates:

1. **Console Output**: Real-time progress and performance metrics
2. **Plots** (saved to `./figs/`):
   - Equity curve and drawdown chart
   - Trading signals on price chart
3. **JSON Report** (`backtest_report.json`): Detailed metrics and trade log

### Sample Performance Metrics

```
Strategy Performance:
  Total Return: 2.45%
  Annualized Return: 15.8%
  Sharpe Ratio: 1.23
  Sortino Ratio: 1.67
  Max Drawdown: -1.2%
  Hit Rate: 58.3%
  Number of Trades: 24
  Prediction Accuracy: 61.2%
  Balanced Accuracy: 59.8%
```

## 🔍 Technical Details

### Data Processing
- Handles Yahoo Finance 7-day limitation by stitching multiple windows
- Forward-fills missing values and removes gaps
- Calculates dynamic spread-based thresholds

### Feature Engineering
- **Price Features**: Multiple return periods, log returns
- **Technical Indicators**: RSI, MACD, ATR, rolling volatility
- **Cross-Asset**: VIX and DXY percentage changes
- **Volume**: Z-score normalization

### Triple-Barrier Labeling
- Threshold: η = max(0.0005, 5×avg_spread)
- Horizon: 30 minutes (6 periods at 5-min frequency)
- Labels: {-1, 0, +1} for {down, flat, up}

### Model Architecture
- **Base Model**: LightGBM multiclass classifier
- **Meta-Model**: RandomForest binary classifier
- **Training**: TimeSeriesSplit with 42/7 day train/val split

### Backtesting
- **Costs**: 1 tick slippage per trade
- **Position Sizing**: Unit exposure (1 share equivalent)
- **Risk Management**: Flat position when confidence low

## ⚠️ Important Notes

1. **Data Limitations**: Yahoo Finance may have gaps or delays
2. **Overfitting Risk**: Models trained on limited historical data
3. **Transaction Costs**: Simplified cost model
4. **Market Conditions**: Strategy may not work in all market regimes
5. **Educational Purpose**: This is for research/educational use only

## 🤝 Contributing

Feel free to submit issues, feature requests, or pull requests to improve the system.

## 📜 License

This project is for educational and research purposes. Use at your own risk.

---

**Disclaimer**: This system is for educational purposes only. Past performance does not guarantee future results. Trading involves risk of loss.
