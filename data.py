"""
Data downloading and stitching module for financial data.
Downloads 56 days of 1-minute data for SPY, ^VIX, DXY by stitching 8×7-day windows.
"""

import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
import warnings
warnings.filterwarnings('ignore')

class DataDownloader:
    """Downloads and stitches financial data from Yahoo Finance."""
    
    def __init__(self, symbols=['SPY', '^VIX', 'DXY'], days=56):
        self.symbols = symbols
        self.days = days
        self.window_size = 7  # Yahoo Finance limit
        self.num_windows = (days + self.window_size - 1) // self.window_size
        
    def download_data(self):
        """Download and stitch data for all symbols."""
        print(f"Downloading {self.days} days of 1-minute data for {self.symbols}")
        
        all_data = {}
        end_date = datetime.now()
        
        for symbol in self.symbols:
            print(f"Downloading {symbol}...")
            symbol_data = []
            
            # Download in 7-day windows
            for i in range(self.num_windows):
                window_end = end_date - timedelta(days=i * self.window_size)
                window_start = window_end - timedelta(days=self.window_size)
                
                try:
                    data = yf.download(
                        symbol,
                        start=window_start,
                        end=window_end,
                        interval='1m',
                        progress=False
                    )
                    
                    if not data.empty:
                        symbol_data.append(data)
                        
                except Exception as e:
                    print(f"Error downloading {symbol} for window {i}: {e}")
                    continue
            
            # Stitch windows together
            if symbol_data:
                stitched_data = pd.concat(symbol_data, axis=0)
                stitched_data = stitched_data.sort_index()
                
                # Remove duplicates
                stitched_data = stitched_data[~stitched_data.index.duplicated(keep='first')]
                
                # Keep only the last 56 days
                cutoff_date = end_date - timedelta(days=self.days)
                stitched_data = stitched_data[stitched_data.index >= cutoff_date]
                
                all_data[symbol] = stitched_data
                print(f"  {symbol}: {len(stitched_data)} records")
            else:
                print(f"  {symbol}: No data downloaded")
        
        return self._merge_symbols(all_data)
    
    def _merge_symbols(self, all_data):
        """Merge data from all symbols into a single DataFrame."""
        if not all_data:
            raise ValueError("No data downloaded for any symbol")
        
        # Start with the first symbol's data
        merged_data = None
        
        for symbol, data in all_data.items():
            # Select OHLCV columns
            cols = ['Open', 'High', 'Low', 'Close', 'Volume']
            available_cols = [col for col in cols if col in data.columns]
            
            if not available_cols:
                print(f"Warning: No OHLCV data for {symbol}")
                continue
            
            symbol_data = data[available_cols].copy()
            
            # Rename columns with symbol prefix
            symbol_data.columns = [f"{symbol}_{col}" for col in symbol_data.columns]
            
            if merged_data is None:
                merged_data = symbol_data
            else:
                merged_data = merged_data.join(symbol_data, how='outer')
        
        if merged_data is None:
            raise ValueError("Failed to merge any symbol data")
        
        # Forward fill missing values
        merged_data = merged_data.fillna(method='ffill')
        
        # Drop rows with any remaining NaN values
        merged_data = merged_data.dropna()
        
        print(f"Merged dataset: {len(merged_data)} records, {len(merged_data.columns)} columns")
        return merged_data
    
    def calculate_spread(self, data):
        """Calculate average spread for each symbol."""
        spreads = {}
        
        for symbol in self.symbols:
            high_col = f"{symbol}_High"
            low_col = f"{symbol}_Low"
            close_col = f"{symbol}_Close"
            
            if all(col in data.columns for col in [high_col, low_col, close_col]):
                # Calculate spread as (High - Low) / Close
                spread = (data[high_col] - data[low_col]) / data[close_col]
                spreads[symbol] = spread.mean()
            else:
                spreads[symbol] = 0.001  # Default spread
        
        return spreads

def download_market_data(days=56):
    """Convenience function to download market data."""
    downloader = DataDownloader(days=days)
    return downloader.download_data(), downloader.calculate_spread(downloader.download_data())

if __name__ == "__main__":
    # Test the data downloader
    data, spreads = download_market_data(days=56)
    print("\nData shape:", data.shape)
    print("Columns:", list(data.columns))
    print("Date range:", data.index.min(), "to", data.index.max())
    print("Average spreads:", spreads)