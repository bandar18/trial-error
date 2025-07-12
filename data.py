import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def download_data(symbols=['SPY', '^VIX', 'DX-Y.NYB'], days=56):
    """
    Download 1-minute data for multiple symbols by stitching 7-day windows.
    
    Args:
        symbols: List of symbols to download
        days: Total number of days to download
    
    Returns:
        DataFrame with multi-level columns (symbol, OHLCV)
    """
    print(f"Downloading {days} days of 1-minute data for {symbols}...")
    
    # Calculate date ranges for 7-day windows
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    # Create 7-day windows
    windows = []
    current_start = start_date
    
    while current_start < end_date:
        current_end = min(current_start + timedelta(days=7), end_date)
        windows.append((current_start, current_end))
        current_start = current_end
    
    all_data = {}
    
    for symbol in symbols:
        print(f"Processing {symbol}...")
        symbol_data = []
        
        for i, (start, end) in enumerate(windows):
            try:
                # Download data for this window
                ticker = yf.Ticker(symbol)
                data = ticker.history(
                    start=start,
                    end=end,
                    interval='1m',
                    prepost=True
                )
                
                if not data.empty:
                    symbol_data.append(data)
                    print(f"  Window {i+1}/{len(windows)}: {len(data)} records")
                else:
                    print(f"  Window {i+1}/{len(windows)}: No data")
                    
            except Exception as e:
                print(f"  Error downloading {symbol} for window {i+1}: {e}")
                continue
        
        if symbol_data:
            # Concatenate all windows for this symbol
            combined_data = pd.concat(symbol_data, axis=0)
            combined_data = combined_data[~combined_data.index.duplicated(keep='first')]
            combined_data = combined_data.sort_index()
            all_data[symbol] = combined_data
        else:
            print(f"Warning: No data downloaded for {symbol}")
    
    # Create multi-level DataFrame
    if all_data:
        # Align all symbols to common index
        common_index = None
        for symbol, data in all_data.items():
            if common_index is None:
                common_index = data.index
            else:
                common_index = common_index.intersection(data.index)
        
        # Filter all data to common index
        aligned_data = {}
        for symbol, data in all_data.items():
            aligned_data[symbol] = data.loc[common_index]
        
        # Create multi-level columns
        result_data = {}
        for symbol, data in aligned_data.items():
            for col in data.columns:
                result_data[(symbol, col)] = data[col]
        
        result_df = pd.DataFrame(result_data, index=common_index)
        result_df.index.name = 'datetime'
        
        print(f"Final dataset shape: {result_df.shape}")
        print(f"Date range: {result_df.index.min()} to {result_df.index.max()}")
        
        return result_df
    else:
        raise ValueError("No data could be downloaded for any symbol")

def get_spread_estimate(data, symbol='SPY'):
    """
    Estimate average spread for threshold calculation.
    
    Args:
        data: Multi-level DataFrame
        symbol: Symbol to calculate spread for
    
    Returns:
        Average spread estimate
    """
    if (symbol, 'High') in data.columns and (symbol, 'Low') in data.columns:
        spread = (data[(symbol, 'High')] - data[(symbol, 'Low')]) / data[(symbol, 'Close')]
        return spread.mean()
    else:
        return 0.0001  # Default small spread

if __name__ == "__main__":
    # Test the data download
    data = download_data()
    print("\nSample data:")
    print(data.head())
    print(f"\nColumns: {data.columns.tolist()}")