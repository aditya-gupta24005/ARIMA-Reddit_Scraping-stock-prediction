# stock_data.py
"""
Download historical stock prices using yfinance, with simple retry logic.
"""

import time
import yfinance as yf


def get_stock_data(symbol, start_date, end_date):
    """
    Returns a DataFrame of OHLC data for `symbol` between start_date and end_date.
    Retries up to 3 times on transient errors.
    """
    attempts = 3
    while attempts:
        try:
            df = yf.download(symbol, start=start_date, end=end_date)
            if df.empty:
                raise ValueError("No data was returned")
            return df
        except Exception as e:
            attempts -= 1
            print(f"Warning: failed to download {symbol} ({attempts} retries left): {e}")
            time.sleep(1)
    # if we reach here, all attempts failed
    raise RuntimeError(f"Could not fetch stock data for {symbol}")
