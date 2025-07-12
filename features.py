import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def calculate_returns(prices, periods=[1, 5, 10, 30]):
    """
    Calculate price returns for different periods.
    
    Args:
        prices: Price series
        periods: List of periods to calculate returns for
    
    Returns:
        DataFrame with return columns
    """
    returns = pd.DataFrame(index=prices.index)
    
    for period in periods:
        returns[f'r_{period}'] = prices.pct_change(period)
    
    return returns

def calculate_volatility(prices, window=10):
    """
    Calculate rolling volatility.
    
    Args:
        prices: Price series
        window: Rolling window size in minutes
    
    Returns:
        Volatility series
    """
    returns = prices.pct_change()
    return returns.rolling(window).std()

def calculate_atr(high, low, close, window=20):
    """
    Calculate Average True Range.
    
    Args:
        high: High prices
        low: Low prices
        close: Close prices
        window: Rolling window size
    
    Returns:
        ATR series
    """
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window).mean()
    
    return atr

def calculate_rsi(prices, window=14):
    """
    Calculate Relative Strength Index.
    
    Args:
        prices: Price series
        window: Rolling window size
    
    Returns:
        RSI series
    """
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD.
    
    Args:
        prices: Price series
        fast: Fast EMA period
        slow: Slow EMA period
        signal: Signal line period
    
    Returns:
        DataFrame with MACD, signal, and histogram
    """
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    histogram = macd - signal_line
    
    return pd.DataFrame({
        'macd': macd,
        'macd_signal': signal_line,
        'macd_histogram': histogram
    }, index=prices.index)

def calculate_volume_zscore(volume, window=20):
    """
    Calculate volume z-score.
    
    Args:
        volume: Volume series
        window: Rolling window size
    
    Returns:
        Volume z-score series
    """
    rolling_mean = volume.rolling(window).mean()
    rolling_std = volume.rolling(window).std()
    zscore = (volume - rolling_mean) / rolling_std
    
    return zscore

def engineer_features(data, symbol='SPY'):
    """
    Engineer all features for the given symbol.
    
    Args:
        data: Multi-level DataFrame
        symbol: Symbol to engineer features for
    
    Returns:
        DataFrame with all engineered features
    """
    print(f"Engineering features for {symbol}...")
    
    # Extract price data
    close = data[(symbol, 'Close')]
    high = data[(symbol, 'High')]
    low = data[(symbol, 'Low')]
    volume = data[(symbol, 'Volume')]
    
    features = pd.DataFrame(index=data.index)
    
    # Price returns ladder
    returns = calculate_returns(close)
    features = pd.concat([features, returns], axis=1)
    
    # Volatility
    features[f'{symbol}_volatility'] = calculate_volatility(close, window=10)
    
    # ATR
    features[f'{symbol}_atr'] = calculate_atr(high, low, close, window=20)
    
    # RSI
    features[f'{symbol}_rsi'] = calculate_rsi(close, window=14)
    
    # MACD
    macd_data = calculate_macd(close)
    features = pd.concat([features, macd_data], axis=1)
    
    # Volume z-score
    features[f'{symbol}_volume_zscore'] = calculate_volume_zscore(volume, window=20)
    
    # Cross-asset features
    if symbol == 'SPY':
        # VIX changes
        if ('^VIX', 'Close') in data.columns:
            vix_close = data[('^VIX', 'Close')]
            features['vix_change'] = vix_close.pct_change()
            features['vix_change_5'] = vix_close.pct_change(5)
        
        # DXY changes
        if ('DX-Y.NYB', 'Close') in data.columns:
            dxy_close = data[('DX-Y.NYB', 'Close')]
            features['dxy_change'] = dxy_close.pct_change()
            features['dxy_change_5'] = dxy_close.pct_change(5)
    
    # Clean up infinite and NaN values
    features = features.replace([np.inf, -np.inf], np.nan)
    features = features.fillna(method='ffill').fillna(0)
    
    print(f"Engineered {len(features.columns)} features")
    return features

def resample_to_5min(data):
    """
    Resample 1-minute data to 5-minute data.
    
    Args:
        data: 1-minute DataFrame
    
    Returns:
        5-minute DataFrame
    """
    print("Resampling to 5-minute data...")
    
    # For OHLCV data, use appropriate aggregation
    resampled = {}
    
    for col in data.columns:
        if isinstance(col, tuple):
            symbol, field = col
            if field == 'Volume':
                resampled[col] = data[col].resample('5T').sum()
            else:
                resampled[col] = data[col].resample('5T').ohlc()
        else:
            # For engineered features, use mean
            resampled[col] = data[col].resample('5T').mean()
    
    # Handle OHLC columns
    final_data = {}
    for col, values in resampled.items():
        if isinstance(col, tuple):
            symbol, field = col
            if field in ['Open', 'High', 'Low', 'Close']:
                if isinstance(values, pd.DataFrame):
                    # Extract the appropriate OHLC value
                    if field == 'Open':
                        final_data[col] = values['open']
                    elif field == 'High':
                        final_data[col] = values['high']
                    elif field == 'Low':
                        final_data[col] = values['low']
                    elif field == 'Close':
                        final_data[col] = values['close']
                else:
                    final_data[col] = values
            else:
                final_data[col] = values
        else:
            final_data[col] = values
    
    result = pd.DataFrame(final_data)
    result = result.fillna(method='ffill').fillna(0)
    
    print(f"Resampled data shape: {result.shape}")
    return result

def triple_barrier_labeling(prices, horizon=30, eta=0.0005, upper_barrier=None, lower_barrier=None):
    """
    Implement triple-barrier labeling.
    
    Args:
        prices: Price series
        horizon: Prediction horizon in periods
        eta: Threshold for barriers
        upper_barrier: Upper barrier multiplier (optional)
        lower_barrier: Lower barrier multiplier (optional)
    
    Returns:
        Series with labels {-1, 0, 1}
    """
    print(f"Applying triple-barrier labeling (horizon={horizon}, eta={eta})...")
    
    labels = pd.Series(0, index=prices.index)
    
    for i in range(len(prices) - horizon):
        current_price = prices.iloc[i]
        future_prices = prices.iloc[i:i+horizon+1]
        
        # Calculate barriers
        if upper_barrier is None:
            upper_barrier_val = current_price * (1 + eta)
        else:
            upper_barrier_val = current_price * (1 + upper_barrier)
            
        if lower_barrier is None:
            lower_barrier_val = current_price * (1 - eta)
        else:
            lower_barrier_val = current_price * (1 - lower_barrier)
        
        # Check if barriers are hit
        upper_hit = (future_prices > upper_barrier_val).any()
        lower_hit = (future_prices < lower_barrier_val).any()
        
        if upper_hit and lower_hit:
            # Both barriers hit, check which one first
            upper_idx = (future_prices > upper_barrier_val).idxmax()
            lower_idx = (future_prices < lower_barrier_val).idxmax()
            
            if upper_idx <= lower_idx:
                labels.iloc[i] = 1
            else:
                labels.iloc[i] = -1
        elif upper_hit:
            labels.iloc[i] = 1
        elif lower_hit:
            labels.iloc[i] = -1
        # else remains 0 (no barrier hit)
    
    # Remove last horizon periods (no future data)
    labels.iloc[-horizon:] = np.nan
    
    print(f"Label distribution: {labels.value_counts().to_dict()}")
    return labels

def prepare_features_and_labels(data, symbol='SPY', horizon=30, eta=0.0005):
    """
    Complete feature engineering and labeling pipeline.
    
    Args:
        data: Raw multi-level DataFrame
        symbol: Symbol to process
        horizon: Prediction horizon for labeling
        eta: Threshold for triple-barrier
    
    Returns:
        Tuple of (features, labels, prices)
    """
    # Engineer features
    features = engineer_features(data, symbol)
    
    # Resample to 5-minute
    features_5min = resample_to_5min(features)
    
    # Get prices for labeling
    prices_5min = resample_to_5min(data[[(symbol, 'Close')]])
    prices = prices_5min[(symbol, 'Close')]
    
    # Apply triple-barrier labeling
    labels = triple_barrier_labeling(prices, horizon=horizon, eta=eta)
    
    # Align features and labels
    common_index = features_5min.index.intersection(labels.index)
    features_aligned = features_5min.loc[common_index]
    labels_aligned = labels.loc[common_index]
    prices_aligned = prices.loc[common_index]
    
    # Remove rows with NaN labels
    valid_mask = ~labels_aligned.isna()
    features_final = features_aligned[valid_mask]
    labels_final = labels_aligned[valid_mask]
    prices_final = prices_aligned[valid_mask]
    
    print(f"Final dataset: {len(features_final)} samples, {len(features_final.columns)} features")
    
    return features_final, labels_final, prices_final

if __name__ == "__main__":
    # Test feature engineering
    from data import download_data
    
    data = download_data(days=7)  # Test with smaller dataset
    features, labels, prices = prepare_features_and_labels(data)
    
    print("\nSample features:")
    print(features.head())
    print("\nSample labels:")
    print(labels.head())