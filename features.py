from typing import Tuple

import numpy as np
import pandas as pd


def _rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Wilder's RSI."""
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    ma_up = up.ewm(alpha=1 / period, min_periods=period).mean()
    ma_down = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = ma_up / ma_down
    return 100 - (100 / (1 + rs))


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def _macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    macd_hist = macd - macd_signal
    return pd.DataFrame({"macd": macd, "macd_signal": macd_signal, "macd_hist": macd_hist})


def _triple_barrier(price: pd.Series, horizon: int, eta: float) -> pd.Series:
    """Vectorised triple-barrier labeling.

    price is expected at 5-minute resolution. horizon in bars (e.g. 6 == 30min).
    """
    arr = price.values
    up_mat = (arr.reshape(-1, 1) * (1 + eta))
    dn_mat = (arr.reshape(-1, 1) * (1 - eta))

    # Build matrix of forward prices within horizon using numpy broadcasting
    idx = np.arange(len(arr))
    forward_indices = idx[:, None] + np.arange(1, horizon + 1)
    forward_indices[forward_indices >= len(arr)] = len(arr) - 1
    fwd_prices = arr[forward_indices]

    up_hit = (fwd_prices >= up_mat).argmax(axis=1)
    dn_hit = (fwd_prices <= dn_mat).argmax(axis=1)

    up_hit_mask = (fwd_prices >= up_mat).any(axis=1)
    dn_hit_mask = (fwd_prices <= dn_mat).any(axis=1)

    label = np.zeros(len(arr), dtype="int8")
    # cases where both barriers never hit remain 0
    for i in range(len(arr)):
        if not up_hit_mask[i] and not dn_hit_mask[i]:
            continue
        elif up_hit_mask[i] and dn_hit_mask[i]:
            if up_hit[i] < dn_hit[i]:
                label[i] = 1
            elif dn_hit[i] < up_hit[i]:
                label[i] = -1
            else:
                label[i] = 0
        elif up_hit_mask[i]:
            label[i] = 1
        elif dn_hit_mask[i]:
            label[i] = -1
    return pd.Series(label, index=price.index)


def build_dataset(
    data: pd.DataFrame,
    resample_rule: str = "5T",
    horizon_min: int = 30,
    eta: float = 0.0005,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Generate (X, y, price) DataFrames.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix at 5-minute resolution.
    y : pd.Series
        Labels {-1,0,1} aligned with X.
    price : pd.Series
        Resampled SPY close prices matching X/y index.
    """
    # Extract key series
    close_spy = data["SPY"]["Close"]
    volume_spy = data["SPY"]["Volume"]

    vix_close = data["^VIX"]["Close"]
    dxy_ticker = [t for t in data.columns.levels[0] if "DX" in t][0]
    dxy_close = data[dxy_ticker]["Close"]

    # Feature engineering @1-min
    feats = pd.DataFrame(index=close_spy.index)
    feats["r1"] = close_spy.pct_change(1)
    feats["r5"] = close_spy.pct_change(5)
    feats["r10"] = close_spy.pct_change(10)
    feats["r30"] = close_spy.pct_change(30)

    feats["sigma10"] = close_spy.pct_change().rolling(10).std()
    feats["atr20"] = _atr(data["SPY"]["High"], data["SPY"]["Low"], close_spy, 20)
    feats["rsi14"] = _rsi(close_spy, 14)
    feats = feats.join(_macd(close_spy))

    feats["vol_z"] = (volume_spy - volume_spy.rolling(30).mean()) / volume_spy.rolling(30).std()
    feats["delta_vix"] = vix_close.pct_change(1)
    feats["delta_dxy"] = dxy_close.pct_change(1)

    # Resample to 5-minute bars (last observation)
    agg = {c: "last" for c in feats.columns}
    feats_5 = feats.resample(resample_rule).agg(agg).dropna()

    price_5 = close_spy.resample(resample_rule).last().loc[feats_5.index]

    # Labeling
    horizon_bars = max(1, horizon_min // 5)
    y = _triple_barrier(price_5, horizon_bars, eta)

    # Align & drop NaNs caused by indicators
    X = feats_5.loc[y.index]
    valid = X.dropna().index.intersection(y.index)
    return X.loc[valid], y.loc[valid], price_5.loc[valid]