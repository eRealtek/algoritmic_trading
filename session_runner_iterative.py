import pandas as pd
import numpy as np
import mplfinance as mpf
import matplotlib.pyplot as plt
from Indicators.indicator_formulas_iterative import IndicatorFormulas_iterative
from Strategy.ib_vwap_formulas_iterative import StrategyFormulasIterative
from Strategy.trade_analysis import TradeAnalysis
from collections import deque

class NYVWAPSessionRunner:
    def __init__(self, *, session_open_utc: str = "16:00", session_hours: float = 3.0, max_sessions: int | None = 10):
        self.session_open_utc = str(session_open_utc)
        self.session_hours = float(session_hours)
        self.max_sessions = max_sessions if (max_sessions is None or int(max_sessions) > 0) else None

    def find_session_starts(self, candles_all: pd.DataFrame, *, start_date: pd.Timestamp | None = None) -> pd.DatetimeIndex:
        starts = candles_all.between_time(self.session_open_utc, self.session_open_utc).index.unique()
        if start_date is not None:
            starts = starts[starts >= start_date]
        return starts

    def run_and_plot_sessions_iterative(
        self,
        candles_all: pd.DataFrame,
        *,
        start_date: pd.Timestamp | None = None,
        open_retrace: float = 0.0,
        cwc_retrace: float = 0.0,
        rr: float = 1.0,
        atr_multiplier: float = 1.0,
        show_plot: bool = False,
    ) -> list[dict]:
        """
        Iterative variant using three DataFrames only: indicators_df, trend_df, trades_df.
        """
        session_starts = self.find_session_starts(candles_all, start_date=start_date)
        session_count = 0
        losing_trade_timeout = 0
        all_trades: list[dict] = []

         # Print number of trading days/sessions being processed
        print("--------------------------------")
        print(f"Number of trading days being processed: {len(session_starts)}")
        print("--------------------------------")

        for session_start in session_starts:
            if self.max_sessions is not None and session_count >= self.max_sessions:
                break

            start_time = session_start
            end_time = start_time + pd.Timedelta(hours=float(self.session_hours))

            session_candles = candles_all.loc[start_time:end_time]
            if len(session_candles) < 10:
                continue

            session_count += 1
            idx_full = session_candles.index

            # 1) indicators_df: vwap, atr, fractal flags, highlight
            indicators_df = pd.DataFrame(index=idx_full, data={
                'vwap': np.nan,
                'atr': np.nan,
                'fractal_high': False,
                'fractal_low': False,
                'highlight': np.nan,
            })

            # 2) trend_df: store only trend state
            trend_df = pd.DataFrame(index=idx_full, data={
                'state': None,
            })


            # Incremental helpers
            ind_inc = IndicatorFormulas_iterative(session_candles)
            prev_vwap: float | None = None
            prev_cum_vol: float = 0.0
            prev_atr: float | None = None
            prev_close: float | None = None

            # Minimal state at API boundary
            trend_state_scalar = None
            trade_update_state = {'open_trade': None, 'pending_signal': None}
            last_fractal_high_price: float | None = None
            last_fractal_low_price: float | None = None
            fractal_structure = deque(maxlen=3)

            # 3 Candle Structure
            is_breach_candle = False
            cwc_memory_max = 2
            cwc_memory = 0
            trade_pending = False
            trade_open = False

            for ts in idx_full:
                current_candle = session_candles.loc[ts]
                high_v = float(current_candle['high']) ; low_v = float(current_candle['low']) ; close_v = float(current_candle['close']) ; vol_v = float(current_candle.get('volume', 0.0) or 0.0)

                # VWAP
                vwap_val = ind_inc.vwap_update(
                    prev_vwap=prev_vwap,
                    prev_cum_vol=prev_cum_vol,
                    new_high=high_v,
                    new_low=low_v,
                    new_close=close_v,
                    new_volume=vol_v,
                )
                indicators_df.loc[ts, 'vwap'] = vwap_val
                prev_vwap = vwap_val
                prev_cum_vol += vol_v

                # ATR
                prev_atr = ind_inc.atr_update(
                    prev_atr=prev_atr,
                    prev_close=prev_close,
                    new_high=high_v,
                    new_low=low_v,
                    new_close=close_v,
                    period=14,
                )
                prev_close = close_v
                indicators_df.loc[ts, 'atr'] = prev_atr

                # Fractals (confirmed with delay)
                look_left_val = 4
                look_right_val = 4
                is_high, is_low = ind_inc.fractal_update(
                    session_candles=session_candles.loc[:ts],
                    look_left=look_left_val,
                    look_right=look_right_val,
                )
                # If a fractal is confirmed, align it to the center candle (ts - look_right)
                if is_high or is_low:
                    window_index = session_candles.loc[:ts].index
                    if len(window_index) >= (1 + look_right_val):
                        confirmed_ts = window_index[-1 - look_right_val]
                        if is_high:
                            indicators_df.loc[confirmed_ts, 'fractal_high'] = True
                            fractal_structure.append('H')
                            try:
                                last_fractal_high_price = float(session_candles.loc[confirmed_ts, 'high'])
                            except Exception:
                                pass
                        if is_low:
                            indicators_df.loc[confirmed_ts, 'fractal_low'] = True
                            fractal_structure.append('L')
                            try:
                                last_fractal_low_price = float(session_candles.loc[confirmed_ts, 'low'])
                            except Exception:
                                pass

                if len(fractal_structure) < 2:
                    continue
                elif len(fractal_structure) > 2:
                    fractal_structure.popleft()

                # Trend update: consume output and save
                trend_df.loc[ts, 'state'] = trend_state_scalar

                # Po dokončení svíčky uděláme update trendu
                last_close_val = float(session_candles.loc[ts, 'close'])
                last_open_val = float(session_candles.loc[ts, 'open'])
                last_vwap_val = float(indicators_df.loc[ts, 'vwap']) if pd.notna(indicators_df.loc[ts, 'vwap']) else np.nan
                trend_state_scalar, losing_trade_timeout, break_fractal_structure = StrategyFormulasIterative.detect_trend_zones_update(
                    state=trend_state_scalar,
                    last_close=last_close_val,
                    last_open=last_open_val,
                    last_vwap=last_vwap_val,
                    fractal_low_price_low=last_fractal_low_price,
                    fractal_high_price_high=last_fractal_high_price,
                    losing_trade_timeout=losing_trade_timeout,
                    atr_multiplier=float(atr_multiplier),
                    atr=prev_atr,
                    fractal_structure=fractal_structure,
                )
                
                # Reset fractal structure if invalidated
                if break_fractal_structure:
                    last_fractal_high_price = None
                    last_fractal_low_price = None

                if not trade_open and not trade_pending:

                    # --- Detect 3 candle structure --- #

                    # 1. Breach VWAP candle
                    if cwc_memory == 0:
                        is_breach_candle = StrategyFormulasIterative.detect_vwap_breach_candle(
                            candle=current_candle,
                            vwap=indicators_df.loc[ts, 'vwap']
                        )

                        if is_breach_candle:
                            cwc_memory = cwc_memory_max

                    if cwc_memory > 0:

                        # 2. Look for CWC candle
                        is_cwc_candle = StrategyFormulasIterative.detect_cwc_candle(
                            candle=current_candle,
                            vwap=indicators_df.loc[ts, 'vwap'],
                            trend=trend_df.loc[ts, 'state'],
                            cwc_candle_retrace=float(cwc_retrace)
                        )

                        if is_cwc_candle:

                            # 3. Place limit order
                            trade = StrategyFormulasIterative.place_limit_order(
                                cwc_candle=current_candle,
                                vwap=indicators_df.loc[ts, 'vwap'],
                                atr=indicators_df.loc[ts, 'atr'],
                                trend=trend_df.loc[ts, 'state'],
                                entry_candle_retrace=float(open_retrace),
                                rr=float(rr)
                            )

                            cwc_memory = 0
                            trade_pending = True

                            # We place the limit order and continue to the next candle
                            continue

                        # CWC candle did not happen, decrement memory
                        cwc_memory -= 1
                    
                # Confirm limit order is actually filled and trade is opened
                if trade_pending:
                    
                    # Check if limit order is filled
                    if current_candle['high'] > trade['entry_price'] or current_candle['low'] < trade['entry_price']:
                        trade_open = True
                        trade_pending = False

                        # Add time information to trade
                        trade['entry_idx'] = ts
                        trade['session_date'] = start_time.date()

                    else:
                        # Cancel limit order and delete trade
                        trade_pending = False   
                        trade = None                       

                if trade_open:

                    # 4. Close trade
                    if trade['trade_type'] == 'long':
                        if current_candle['low'] < trade['sl']:
                            trade['exit_idx'] = ts
                            trade['exit_price'] = trade['sl']
                            trade['result'] = 'SL'
                            trade_open = False
                            all_trades.append(dict(trade))
                            trade = None

                        elif current_candle['high'] > trade['tp']:
                            trade['exit_idx'] = ts
                            trade['exit_price'] = trade['tp']
                            trade['result'] = 'TP'
                            trade_open = False
                            all_trades.append(dict(trade))
                            trade = None

                    elif trade['trade_type'] == 'short':
                        if current_candle['low'] < trade['tp']:
                            trade['exit_idx'] = ts
                            trade['exit_price'] = trade['tp']
                            trade['result'] = 'TP'
                            trade_open = False
                            all_trades.append(dict(trade))
                            trade = None
                        
                        elif current_candle['high'] > trade['sl']:
                            trade['exit_idx'] = ts
                            trade['exit_price'] = trade['sl']
                            trade['result'] = 'SL'
                            trade_open = False
                            all_trades.append(dict(trade))
                            trade = None

            # If a trade remains open at the end of the session, store it as OPEN
            '''
            if trade_open and isinstance(trade, dict):
                trade_copy = dict(trade)
                trade_copy.setdefault('exit_idx', None)
                trade_copy.setdefault('exit_price', None)
                trade_copy.setdefault('session_date', start_time.date())
                trade_copy['result'] = 'OPEN'
                all_trades.append(trade_copy)
            '''
            
            # -------- PLOT SESSION -------- #
            
            # Build addplots from three DFs
            apds: list = []
            
            # Build color-specific VWAP series on the fly using state only
            vwap_series_full = indicators_df['vwap']
            green_mask = trend_df['state'] == 'uptrend'
            red_mask = trend_df['state'] == 'downtrend'
            gray_mask = ~green_mask & ~red_mask
            vwap_green = vwap_series_full.where(green_mask)
            vwap_red = vwap_series_full.where(red_mask)
            vwap_gray = vwap_series_full.where(gray_mask)
            if vwap_gray.notna().any():
                apds.append(mpf.make_addplot(vwap_gray, panel=0, color='gray'))
            if vwap_green.notna().any():
                apds.append(mpf.make_addplot(vwap_green, panel=0, color='green'))
            if vwap_red.notna().any():
                apds.append(mpf.make_addplot(vwap_red, panel=0, color='red'))

            # No-trend ATR band around VWAP (alpha orange dotted)
            if indicators_df['atr'].notna().any() and vwap_series_full.notna().any():
                # Use vwap_gray to ensure bands only appear during gray (no-trend) periods
                atr_band_upper = (vwap_gray + float(atr_multiplier) * indicators_df['atr']).where(gray_mask)
                atr_band_lower = (vwap_gray - float(atr_multiplier) * indicators_df['atr']).where(gray_mask)
                if atr_band_upper.notna().any():
                    apds.append(mpf.make_addplot(atr_band_upper, panel=0, color='orange', linestyle=':', alpha=0.9))
                if atr_band_lower.notna().any():
                    apds.append(mpf.make_addplot(atr_band_lower, panel=0, color='orange', linestyle=':', alpha=0.9))

            # Fractal markers
            high_marker = session_candles['high'].where(indicators_df['fractal_high']) + 1.0
            low_marker = session_candles['low'].where(indicators_df['fractal_low']) - 1.0
            if not high_marker.dropna().empty:
                apds.append(mpf.make_addplot(high_marker, type='scatter', markersize=30, marker='v', color='lime', panel=0))
            if not low_marker.dropna().empty:
                apds.append(mpf.make_addplot(low_marker, type='scatter', markersize=30, marker='^', color='red', panel=0))

            # Highlights
            if indicators_df['highlight'].notna().any():
                apds.append(mpf.make_addplot(indicators_df['highlight'], type='scatter', marker='s', markersize=50, color='#8A2BE2', panel=0))

            # Entries
            entries_for_session = [t for t in all_trades if t.get('session_date') == start_time.date()]
            if entries_for_session:
                entry_long = pd.Series(np.nan, index=session_candles.index)
                entry_short = pd.Series(np.nan, index=session_candles.index)
                for tr in entries_for_session:
                    ts_e = tr.get('entry_idx')
                    price_e = tr.get('entry_price')
                    side = tr.get('trade_type')
                    if ts_e is None or price_e is None or side not in {'long', 'short'}:
                        continue
                    if ts_e in entry_long.index:
                        if side == 'long':
                            entry_long.loc[ts_e] = float(price_e)
                        else:
                            entry_short.loc[ts_e] = float(price_e)
                if entry_long.notna().any():
                    apds.append(mpf.make_addplot(entry_long, type='scatter', marker='^', markersize=80, color='yellow', panel=0))
                if entry_short.notna().any():
                    apds.append(mpf.make_addplot(entry_short, type='scatter', marker='v', markersize=80, color='yellow', panel=0))

                # TP/SL dotted lines for each trade
                for tr in entries_for_session:
                    ent_ts = tr.get('entry_idx')
                    exit_ts = tr.get('exit_idx')
                    tp_val = tr.get('tp')
                    sl_val = tr.get('sl')
                    if ent_ts is None or (tp_val is None and sl_val is None):
                        continue
                    start_idx = ent_ts
                    end_idx = exit_ts if exit_ts is not None else session_candles.index[-1]
                    # Build mask over session index
                    mask = (session_candles.index >= start_idx) & (session_candles.index <= end_idx)
                    if bool(mask.any()):
                        # TP line
                        if tp_val is not None:
                            tp_series = pd.Series(np.nan, index=session_candles.index)
                            tp_series.loc[mask] = float(tp_val)
                            tp_color = 'green' if tr.get('result') == 'TP' else 'orange'
                            apds.append(mpf.make_addplot(tp_series, panel=0, color=tp_color, linestyle=':'))
                        # SL line (red dotted)
                        if sl_val is not None:
                            sl_series = pd.Series(np.nan, index=session_candles.index)
                            sl_series.loc[mask] = float(sl_val)
                            sl_color = 'red' if tr.get('result') == 'SL' else 'orange'
                            apds.append(mpf.make_addplot(sl_series, panel=0, color=sl_color, linestyle=':'))

            fig, axes = mpf.plot(
                session_candles,
                type='candle',
                style='charles',
                volume=False,
                addplot=apds,
                title=(
                    f"Session {start_time.date()} iterative "
                    f"(open_retrace={float(open_retrace):.2f}, "
                    f"cwc_retrace={float(cwc_retrace):.2f}, "
                    f"RR={float(rr):.2f})"
                ),
                ylabel='Price',
                returnfig=True,
            )

            if show_plot:
                plt.show(block=True)
            else:
                try:
                    plt.close(fig)
                except Exception:
                    pass

        return all_trades

