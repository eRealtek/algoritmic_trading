import pandas as pd
from glob import glob
import os
from pandas import Timestamp, Timedelta
import numpy as np
import pandas as pd

class FuturesReader:
    def __init__(self, filepath):
        """
        Initialize the custom trade data reader.

        :param filepath: Path to the semicolon-separated CSV file
        """
        self.filepath = filepath
        self.df = None

    def read(self):
        """
        Reads and parses a single CSV file containing trade data.

        Expected columns:
        - 'Aggressor flag', 'Price', 'Volume', 'Time left'

        Returns:
        - DataFrame with ['timestamp', 'aggressor_flag', 'price', 'volume']
        """
        df = pd.read_csv(
            self.filepath,
            sep=";",
            parse_dates=["Time left"],
            dtype={
                "Aggressor flag": "string",
                "Price": "Float64",
                "Volume": "Int64"
            }
        )

        df.rename(columns={
            "Time left": "timestamp",
            "Aggressor flag": "aggressor_flag",
            "Price": "price",
            "Volume": "volume"
        }, inplace=True)

        df.dropna(subset=["price", "volume", "timestamp"], inplace=True)

        df = df.sort_values("timestamp").reset_index(drop=True)

        self.df = df
        return self.df
    
    def aggregate_to_candles(trades_df, freq='1min', save_csv: bool = False, output_path: str = "Data/Futures/OHLCV/NQ_ohlcv.csv", date_format: str = "%Y-%m-%d %H:%M:%S%z"):
        """
        Aggregate trades to OHLCV in UTC, without any non-UTC timezone conversions.

        - The DataFrame index is ensured to be tz-aware UTC and remains UTC throughout
        - Resamples the entire dataset into OHLCV with bar-open timestamps

        :param trades_df: DataFrame with DatetimeIndex and columns ['price','volume']
                          If tz-naive, it's assumed to be UTC.
        :param freq: Resample frequency like '1min', '15min', '1h'
        :param save_csv: If True, save OHLCV to CSV with a numeric UTC 'timestamp' (epoch ms) column
        :param output_path: Destination CSV path when save_csv is True
        :param date_format: Kept for backward compatibility; ignored when saving timestamp column
        :return: OHLCV DataFrame (UTC tz-aware index)
        """
        # Ensure required columns
        if not {'price'}.issubset(trades_df.columns):
            raise ValueError("DataFrame must contain 'price' column.")

        # Ensure index is UNIX timestamp in milliseconds
        if not np.issubdtype(trades_df.index.dtype, np.integer):
            raise ValueError("Index must contain integer timestamps in milliseconds.")

        # Optional sanity check: are they really milliseconds?
        ts_min, ts_max = trades_df.index.min(), trades_df.index.max()
        if ts_min < 1e12 or ts_max > 1e13:
            raise ValueError("Index values do not look like millisecond UNIX timestamps.")
        
        df = trades_df.sort_index().copy()

        print(df.head())

        # Resample to OHLCV in UTC, needs to be converted to datetime first due to resampling logic
        df.index = pd.to_datetime(df.index, unit="ms")
        ohlcv = df.resample(freq, label='left', closed='left').agg({
            'price': ['first', 'max', 'min', 'last'],
            'volume': 'sum'
        })

        # Convert index back to milliseconds
        ohlcv.index = ohlcv.index.astype("int64") // 10**6

        if ohlcv.empty:
            return pd.DataFrame(columns=['open','high','low','close','volume'])

        ohlcv.columns = ['open', 'high', 'low', 'close', 'volume']
        ohlcv.dropna(inplace=True)
        ohlcv.index.name = 'timestamp'

        if save_csv:
            ohlcv.to_csv(output_path, index=True)

        return ohlcv
