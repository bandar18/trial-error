import datetime
from typing import List

import pandas as pd
import yfinance as yf


def download_data(tickers: List[str], days: int = 56) -> pd.DataFrame:
    """Download 1-minute OHLCV for `tickers` going back `days` days.

    Yahoo Finance limits intraday (\u22641m) queries to 7 days, so the function
    stitches successive 7-day windows moving backwards from *now* into a single
    DataFrame.

    Parameters
    ----------
    tickers : list[str]
        Symbols understood by Yahoo Finance (e.g. "SPY", "^VIX").
    days : int, default 56
        Calendar days to retrieve (must be multiple of 7).

    Returns
    -------
    pd.DataFrame
        Concatenated intraday OHLCV with a DateTimeIndex (UTC) and a two-level
        column index (level-0 ticker, level-1 field).
    """
    now = pd.Timestamp.utcnow().floor("T")
    window = datetime.timedelta(days=7)
    n_chunks = -(-days // 7)  # ceiling division

    frames: list[pd.DataFrame] = []
    for i in range(n_chunks):
        chunk_end = now - i * window
        chunk_start = chunk_end - window

        df = yf.download(
            tickers=tickers,
            start=chunk_start,
            end=chunk_end,
            interval="1m",
            group_by="ticker",
            progress=False,
            threads=True,
            auto_adjust=False,
        )
        if df.empty:
            continue
        frames.append(df)

    if not frames:
        raise ValueError("No data downloaded – verify tickers / internet connectivity.")

    data = pd.concat(frames).sort_index()
    data = data[~data.index.duplicated(keep="first")]

    # Ensure timezone awareness (UTC)
    if data.index.tz is None:
        data.index = data.index.tz_localize("UTC")
    else:
        data.index = data.index.tz_convert("UTC")

    return data


if __name__ == "__main__":
    df = download_data(["SPY", "^VIX", "DX-Y.NYB"], 14)
    print(df.head())