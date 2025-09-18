import pandas as pd
import numpy as np
from collections import deque

class IndicatorFormulas_iterative:
    def __init__(self, df: pd.DataFrame):
        """
        Initialize with a DataFrame that includes at least 'close'.
        Optionally 'open', 'high', 'low', 'volume' for other indicators.
        """
        self.df = df.copy()

    # === Incremental/iterative helpers ===
    def sma_update(
        self,
        *,
        prev_sma: float | None,
        new_close: float,
        window: int,
        prev_count: int | None = None,
        old_close_out: float | None = None,
    ) -> tuple[float, int]:
        """
        Incrementally update SMA using previous SMA and the latest close.

        Parameters
        - prev_sma: previous SMA value (None if not available)
        - new_close: latest close price
        - window: SMA window length
        - prev_count: number of observations contributing to prev_sma (<= window).
                      If None and prev_sma is not None, assumes window (steady-state).
                      If None and prev_sma is None, assumes 0 (starting state).
        - old_close_out: close price leaving the window (required once count >= window)

        Returns (new_sma, new_count)
        """
        w = max(1, int(window))
        if prev_sma is None:
            # starting state
            count_prev = int(prev_count) if prev_count is not None else 0
            sum_prev = 0.0
        else:
            count_prev = int(prev_count) if prev_count is not None else w
            eff_n = min(max(0, count_prev), w)
            sum_prev = float(prev_sma) * float(eff_n)

        if count_prev < w:
            # growing window
            sum_new = sum_prev + float(new_close)
            count_new = count_prev + 1
            sma_new = sum_new / float(count_new)
        else:
            # steady-state rolling window; subtract the value that drops out
            drop = float(old_close_out) if old_close_out is not None else 0.0
            sum_new = sum_prev + float(new_close) - drop
            count_new = w
            sma_new = sum_new / float(w)

        return float(sma_new), int(count_new)

    def vwap_update(
        self,
        *,
        prev_vwap: float | None,
        prev_cum_vol: float | None,
        new_high: float,
        new_low: float,
        new_close: float,
        new_volume: float,
    ) -> tuple[float, float]:
        """
        Incrementally update VWAP using previous VWAP, cumulative volume, and latest OHLCV.

        We maintain cumulative (price*volume) via: prev_cum_pv = prev_vwap * prev_cum_vol
        Then: vwap_new = (prev_cum_pv + tp*vol) / (prev_cum_vol + vol)

        Returns (vwap_new, cum_vol_new)
        """
        vol = float(new_volume)
        tp = (float(new_high) + float(new_low) + float(new_close)) / 3.0
        pv_add = tp * vol

        if prev_vwap is None or prev_cum_vol is None or prev_cum_vol <= 0:
            cum_vol_new = vol
            vwap_new = tp if vol > 0 else float('nan')
        else:
            cum_pv_prev = float(prev_vwap) * float(prev_cum_vol)
            cum_vol_new = float(prev_cum_vol) + vol
            vwap_new = (cum_pv_prev + pv_add) / cum_vol_new if cum_vol_new > 0 else float('nan')

        return float(vwap_new)


    def atr_update(
        self,
        *,
        prev_atr: float | None,
        prev_close: float | None,
        new_high: float,
        new_low: float,
        new_close: float,
        period: int = 14,
    ) -> float:
        """
        Incrementally update ATR (Wilder's) using previous ATR and latest OHLC.

        TR_t = max(high - low, |high - prev_close|, |low - prev_close|)
        If prev_atr is None: initialize with TR_t
        Else: ATR_t = (prev_atr * (period - 1) + TR_t) / period
        """
        p = max(1, int(period))
        high = float(new_high)
        low = float(new_low)
        close_prev = float(prev_close) if prev_close is not None else None

        if close_prev is None:
            tr = max(0.0, high - low)
        else:
            tr = max(high - low, abs(high - close_prev), abs(low - close_prev))

        if prev_atr is None:
            return float(tr)

        atr_new = (float(prev_atr) * (p - 1) + float(tr)) / float(p)
        return float(atr_new)


    def detect_fractals(self, look_left: int = 2, look_right: int = 2):

        high = self.df['high']
        low = self.df['low']
        length = len(self.df)

        self.df["fractal_high"] = False
        self.df["fractal_low"] = False

        for i in range(length):
            center_high = high.iloc[i]
            center_low = low.iloc[i]

            # Bounds for left/right slices
            left_start = max(0, i - look_left)
            right_end = min(length, i + look_right + 1)

            left_highs = high.iloc[left_start:i]
            right_highs = high.iloc[i + 1:right_end]

            left_lows = low.iloc[left_start:i]
            right_lows = low.iloc[i + 1:right_end]

            # Fractal high
            if (len(left_highs) == look_left or i < look_left) and \
            (len(right_highs) == look_right):
                if all(center_high > h for h in left_highs) and all(center_high > h for h in right_highs):
                    self.df.at[self.df.index[i], "fractal_high"] = True

            # Fractal low
            if (len(left_lows) == look_left or i < look_left) and \
            (len(right_lows) == look_right):
                if all(center_low < l for l in left_lows) and all(center_low < l for l in right_lows):
                    self.df.at[self.df.index[i], "fractal_low"] = True

        # print(self.df.shape)
        # print(self.df.head())

        return self.df

    def detect_series_fractals(
        self,
        series: pd.Series,
        look_left: int = 2,
        look_right: int = 2,
    ):
        """
        Detect fractal highs/lows on a single numeric Series using the same
        left/right strict inequality criteria as price fractals.

        Returns:
        - fractal_high: Boolean Series where local maxima are detected
        - fractal_low: Boolean Series where local minima are detected
        """
        length = len(series)
        fractal_high = pd.Series(False, index=series.index)
        fractal_low = pd.Series(False, index=series.index)

        for i in range(length):
            center_val = series.iloc[i]
            left_start = max(0, i - look_left)
            right_end = min(length, i + look_right + 1)

            left_vals = series.iloc[left_start:i]
            right_vals = series.iloc[i + 1:right_end]

            # Fractal high
            if (len(left_vals) == look_left or i < look_left) and (len(right_vals) == look_right):
                if all(center_val > v for v in left_vals) and all(center_val > v for v in right_vals):
                    fractal_high.iloc[i] = True

            # Fractal low
            if (len(left_vals) == look_left or i < look_left) and (len(right_vals) == look_right):
                if all(center_val < v for v in left_vals) and all(center_val < v for v in right_vals):
                    fractal_low.iloc[i] = True

        return fractal_high, fractal_low

    def fractal_update(
        self,
        session_candles: pd.DataFrame,
        *,
        look_left: int = 2,
        look_right: int = 2,
    ) -> tuple[bool, bool]:
        """
        Check if the last confirmable candle in the provided session is a fractal.

        The confirmable candle is at index len(session) - 1 - look_right, so it has
        exactly look_right candles to its right. Returns only when that candle is a
        fractal high or low.

        Returns (timestamp | None, is_high, is_low).
        If not enough candles or it's not a fractal, returns (None, False, False).
        """
        if session_candles is None or len(session_candles) == 0:
            return False, False

        L = max(0, int(look_left))
        R = max(0, int(look_right))
        # At minimum we need the center candle and exactly R candles to the right.
        # We allow fewer than L candles on the left at the start of session.
        if len(session_candles) < (1 + R):
            return False, False

        # Determine the center/confirmable candle
        center_idx = len(session_candles) - 1 - R
        try:
            center_high = float(session_candles['high'].iloc[center_idx])
            center_low = float(session_candles['low'].iloc[center_idx])
        except Exception:
            return False, False

        # Left and right windows around center
        left_slice = slice(max(0, center_idx - L), center_idx)
        right_slice = slice(center_idx + 1, center_idx + 1 + R)

        try:
            left_highs = session_candles['high'].iloc[left_slice]
            right_highs = session_candles['high'].iloc[right_slice]
            left_lows = session_candles['low'].iloc[left_slice]
            right_lows = session_candles['low'].iloc[right_slice]
        except Exception:
            return False, False

        # Require full right window; allow shorter left window at session start
        if len(right_highs) != R or len(right_lows) != R:
            return False, False

        # For the left side, if the window is shorter than L (session start), we still apply the same strict inequality
        # over what is available. Right side must be full R and is strictly enforced above.
        is_high = bool(all(center_high > h for h in left_highs) and all(center_high > h for h in right_highs))
        is_low = bool(all(center_low < l for l in left_lows) and all(center_low < l for l in right_lows))

        if not (is_high or is_low):
            return False, False

        return is_high, is_low

    