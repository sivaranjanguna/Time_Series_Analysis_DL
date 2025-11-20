"""
data_acquisition.py
Programmatically downloads historical market data (S&P 500 ^GSPC) using yfinance.
If yfinance is unavailable or no internet, the script creates a placeholder CSV so the rest of
the pipeline can run offline.
Output: multivariate_timeseries.csv
"""
import os
import pandas as pd

try:
    import yfinance as yf
except Exception:
    yf = None

def fetch_sp500(start_date='2010-01-01', end_date=None, out_csv='multivariate_timeseries.csv'):
    """
    Fetch S&P 500 (^GSPC) data and save to CSV.
    If yfinance is not available, save a synthetic placeholder CSV.
    """
    if yf is None:
        print('yfinance not available; creating placeholder CSV.')
        import numpy as np
        n = 1200
        t = pd.date_range(start=start_date, periods=n, freq='D')
        df = pd.DataFrame({
            'time': t,
            'Open': 1000 + 0.1 * np.arange(n),
            'High': 1000 + 0.1 * np.arange(n) + 1.0,
            'Low': 1000 + 0.1 * np.arange(n) - 1.0,
            'Close': 1000 + 0.1 * np.arange(n) + 0.2,
            'Volume': 1000000 + (np.arange(n) % 50) * 1000
        })
        df.to_csv(out_csv, index=False)
        print('Placeholder dataset saved to', out_csv)
        return out_csv
    else:
        print('Downloading S&P 500 (^GSPC) from Yahoo Finance...')
        ticker = '^GSPC'
        data = yf.download(ticker, start=start_date, end=end_date)
        if data is None or data.empty:
            raise RuntimeError('Failed to download data. Check internet or ticker.')
        data = data.reset_index().rename(columns={'Date': 'time'})
        data.to_csv(out_csv, index=False)
        print('Saved to', out_csv)
        return out_csv

if __name__ == '__main__':
    fetch_sp500()
