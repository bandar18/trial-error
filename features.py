"""
Feature engineering and triple-barrier labeling module.
Engineers features at 1-min granularity, resamples to 5-min, and applies triple-barrier labeling.
"""

import pandas as pd
import numpy as np
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

class FeatureEngineer:
    """Feature engineering and labeling for financial data."""
    
    def __init__(self, data, primary_symbol='SPY'):
        self.data = data.copy()
        self.primary_symbol = primary_symbol
        self.features = None
        
    def engineer_features(self):
        """Engineer all features at 1-minute granularity."""
        print("Engineering features...")
        
        # Start with original data
        df = self.data.copy()
        
        # Price-based features for primary symbol
        df = self._add_price_features(df, self.primary_symbol)
        
        # Technical indicators
        df = self._add_technical_indicators(df, self.primary_symbol)
        
        # Cross-asset features
        df = self._add_cross_asset_features(df)
        
        # Resample to 5-minute intervals
        df_5min = self._resample_to_5min(df)
        
        print(f"Features engineered: {len(df_5min.columns)} columns, {len(df_5min)} rows")
        self.features = df_5min
        return df_5min
    
    def _add_price_features(self, df, symbol):
        """Add price return ladder features."""
        close_col = f"{symbol}_Close"
        
        if close_col not in df.columns:
            print(f"Warning: {close_col} not found")
            return df
        
        # Calculate returns at different lags
        for lag in [1, 5, 10, 30]:
            df[f"{symbol}_return_{lag}m"] = df[close_col].pct_change(lag)
        
        # Log returns
        df[f"{symbol}_log_return"] = np.log(df[close_col] / df[close_col].shift(1))
        
        return df
    
    def _add_technical_indicators(self, df, symbol):
        """Add technical indicators."""
        close_col = f"{symbol}_Close"
        high_col = f"{symbol}_High"
        low_col = f"{symbol}_Low"
        volume_col = f"{symbol}_Volume"
        
        if close_col not in df.columns:
            return df
        
        # Rolling volatility (10-minute window)
        df[f"{symbol}_volatility_10m"] = df[f"{symbol}_log_return"].rolling(10).std()
        
        # ATR (20-period)
        if all(col in df.columns for col in [high_col, low_col, close_col]):
            df = self._calculate_atr(df, symbol, 20)
        
        # RSI (14-period)
        df = self._calculate_rsi(df, symbol, 14)
        
        # MACD
        df = self._calculate_macd(df, symbol)
        
        # Volume z-score (if volume data available)
        if volume_col in df.columns:
            df[f"{symbol}_volume_zscore"] = stats.zscore(df[volume_col].rolling(60).mean(), nan_policy='omit')
        
        return df
    
    def _calculate_atr(self, df, symbol, period):
        """Calculate Average True Range."""
        high_col = f"{symbol}_High"
        low_col = f"{symbol}_Low"
        close_col = f"{symbol}_Close"
        
        # True Range components
        tr1 = df[high_col] - df[low_col]
        tr2 = abs(df[high_col] - df[close_col].shift(1))
        tr3 = abs(df[low_col] - df[close_col].shift(1))
        
        # True Range
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        
        # ATR
        df[f"{symbol}_ATR_{period}"] = tr.rolling(period).mean()
        
        return df
    
    def _calculate_rsi(self, df, symbol, period):
        """Calculate Relative Strength Index."""
        close_col = f"{symbol}_Close"
        
        delta = df[close_col].diff()
        gain = (delta.where(delta > 0, 0)).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        
        rs = gain / loss
        df[f"{symbol}_RSI_{period}"] = 100 - (100 / (1 + rs))
        
        return df
    
    def _calculate_macd(self, df, symbol, fast=12, slow=26, signal=9):
        """Calculate MACD indicator."""
        close_col = f"{symbol}_Close"
        
        ema_fast = df[close_col].ewm(span=fast).mean()
        ema_slow = df[close_col].ewm(span=slow).mean()
        
        macd = ema_fast - ema_slow
        signal_line = macd.ewm(span=signal).mean()
        
        df[f"{symbol}_MACD"] = macd
        df[f"{symbol}_MACD_signal"] = signal_line
        df[f"{symbol}_MACD_histogram"] = macd - signal_line
        
        return df
    
    def _add_cross_asset_features(self, df):
        """Add cross-asset features (VIX and DXY changes)."""
        # VIX change
        vix_close = "^VIX_Close"
        if vix_close in df.columns:
            df["VIX_change_1m"] = df[vix_close].pct_change(1)
            df["VIX_change_5m"] = df[vix_close].pct_change(5)
        
        # DXY change
        dxy_close = "DXY_Close"
        if dxy_close in df.columns:
            df["DXY_change_1m"] = df[dxy_close].pct_change(1)
            df["DXY_change_5m"] = df[dxy_close].pct_change(5)
        
        return df
    
    def _resample_to_5min(self, df):
        """Resample 1-minute data to 5-minute intervals."""
        # Define aggregation rules
        agg_rules = {}
        
        for col in df.columns:
            if any(x in col.lower() for x in ['open']):
                agg_rules[col] = 'first'
            elif any(x in col.lower() for x in ['high']):
                agg_rules[col] = 'max'
            elif any(x in col.lower() for x in ['low']):
                agg_rules[col] = 'min'
            elif any(x in col.lower() for x in ['close', 'rsi', 'macd']):
                agg_rules[col] = 'last'
            elif any(x in col.lower() for x in ['volume']):
                agg_rules[col] = 'sum'
            else:
                agg_rules[col] = 'mean'  # Default for other features
        
        # Resample
        df_5min = df.resample('5T').agg(agg_rules)
        
        # Drop rows with NaN values
        df_5min = df_5min.dropna()
        
        return df_5min

class TripleBarrierLabeler:
    """Triple barrier labeling for financial data."""
    
    def __init__(self, data, symbol='SPY', horizon_minutes=30, eta=None, avg_spreads=None):
        self.data = data.copy()
        self.symbol = symbol
        self.horizon_minutes = horizon_minutes
        self.eta = eta
        self.avg_spreads = avg_spreads or {}
        
    def apply_triple_barrier(self):
        """Apply triple barrier labeling."""
        print(f"Applying triple barrier labeling (horizon={self.horizon_minutes}min)...")
        
        close_col = f"{self.symbol}_Close"
        if close_col not in self.data.columns:
            raise ValueError(f"Close price column {close_col} not found")
        
        # Calculate threshold
        threshold = self._calculate_threshold()
        print(f"Using threshold η = {threshold:.6f}")
        
        # Apply barriers
        labels = self._calculate_labels(close_col, threshold)
        
        # Add labels to data
        labeled_data = self.data.copy()
        labeled_data['label'] = labels
        
        # Remove rows where we can't calculate future returns
        horizon_periods = self.horizon_minutes // 5  # Convert to 5-min periods
        labeled_data = labeled_data.iloc[:-horizon_periods]
        
        print(f"Label distribution: {labeled_data['label'].value_counts().to_dict()}")
        return labeled_data
    
    def _calculate_threshold(self):
        """Calculate threshold η = max(0.0005, 5×avg_spread)."""
        if self.eta is not None:
            return self.eta
        
        # Get average spread for the symbol
        avg_spread = self.avg_spreads.get(self.symbol, 0.001)
        threshold = max(0.0005, 5 * avg_spread)
        
        return threshold
    
    def _calculate_labels(self, close_col, threshold):
        """Calculate triple barrier labels."""
        prices = self.data[close_col].values
        n = len(prices)
        horizon_periods = self.horizon_minutes // 5  # Convert to 5-min periods
        
        labels = np.zeros(n)
        
        for i in range(n - horizon_periods):
            entry_price = prices[i]
            future_prices = prices[i+1:i+1+horizon_periods]
            
            if len(future_prices) == 0:
                continue
            
            # Calculate returns
            returns = (future_prices - entry_price) / entry_price
            
            # Check barriers
            upper_breach = np.where(returns >= threshold)[0]
            lower_breach = np.where(returns <= -threshold)[0]
            
            if len(upper_breach) > 0 and len(lower_breach) > 0:
                # Both barriers breached - take the first one
                if upper_breach[0] < lower_breach[0]:
                    labels[i] = 1  # Upper barrier hit first
                else:
                    labels[i] = -1  # Lower barrier hit first
            elif len(upper_breach) > 0:
                labels[i] = 1  # Only upper barrier hit
            elif len(lower_breach) > 0:
                labels[i] = -1  # Only lower barrier hit
            else:
                # No barrier hit - use final return
                final_return = returns[-1]
                if final_return > 0:
                    labels[i] = 1
                elif final_return < 0:
                    labels[i] = -1
                else:
                    labels[i] = 0
        
        return labels

def prepare_features_and_labels(data, avg_spreads, horizon_minutes=30, eta=None):
    """Convenience function to prepare features and labels."""
    # Engineer features
    engineer = FeatureEngineer(data)
    features = engineer.engineer_features()
    
    # Apply triple barrier labeling
    labeler = TripleBarrierLabeler(features, horizon_minutes=horizon_minutes, eta=eta, avg_spreads=avg_spreads)
    labeled_data = labeler.apply_triple_barrier()
    
    return labeled_data

if __name__ == "__main__":
    # Test feature engineering
    from data import download_market_data
    
    data, spreads = download_market_data(days=14)  # Test with smaller dataset
    labeled_data = prepare_features_and_labels(data, spreads)
    
    print("\nFeature columns:", [col for col in labeled_data.columns if col != 'label'])
    print("Data shape:", labeled_data.shape)
    print("Label distribution:", labeled_data['label'].value_counts())