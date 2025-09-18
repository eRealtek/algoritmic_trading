import pandas as pd
import numpy as np
from Indicators.indicator_formulas_iterative import IndicatorFormulas_iterative
from NY_VWAP_TEST.pythonVersion.indicators import atr
from collections import deque


class StrategyFormulasIterative:
    """
    Iterative VWAP strategy formulas that operate per-candle, without lookahead.

    Exposes a step-wise trade detector via detect_trades_iterative which processes
    candles in chronological order and emits trades when entries/exits are confirmed.
    """

    @staticmethod
    def detect_trend_zones_update(
        *,
        state: dict | None,
        last_close: float,
        last_open: float,
        last_vwap: float,
        fractal_low_price_low: float | None,
        fractal_high_price_high: float | None,
        losing_trade_timeout: int = 0,
        atr_multiplier: float = 1.0,
        atr: float | None = None,
        fractal_structure: deque | None = None,
    ) -> tuple[dict, int, bool]:

        trend_invalidated = False
        trend_confirmed = False
        break_fractal_structure = False

        # Nechceme tradit po ztrátovém tradu    
        if losing_trade_timeout > 0:
            losing_trade_timeout -= 1
            return state, losing_trade_timeout, break_fractal_structure
       
        # Invalidation check
        if state == 'uptrend' and fractal_low_price_low is not None:
            # if fractal_low_price_low < last_vwap:  # TODO: remove this check if we want to have more trades
                deactivation_price = float(fractal_low_price_low)
                trend_invalidated = float(last_close) < deactivation_price

        elif state == 'downtrend' and fractal_high_price_high is not None:
            # if fractal_high_price_high > last_vwap:  # TODO: remove this check if we want to have more trades
                deactivation_price = float(fractal_high_price_high)
                trend_invalidated = float(last_close) > deactivation_price
        
        # Existuje high a low fractal a ATR je dostupný
        if state is None and (fractal_high_price_high is not None) and (fractal_low_price_low is not None) and (atr is not None):

            # Confirm structure HLH and uptrend if we broke above the last fractal high
            if (float(last_close) > float(fractal_high_price_high)) and (fractal_structure[0] == 'H' and fractal_structure[1] == 'L'):
                
                # Přidáme kontrolní range ATR kolem VWAP a open a close jsou nad VWAP
                if (float(last_close) > float(last_vwap)) and (float(last_open) > float(last_vwap)) and (float(last_close) > float(last_vwap + atr_multiplier * atr)):

                    trend_confirmed = True
                    new_state = 'uptrend'
                
                else:
                    break_fractal_structure = True

            # Confirm structure LHL and downtrend if we broke below the last fractal low
            elif (float(last_close) < float(fractal_low_price_low)) and (fractal_structure[0] == 'L' and fractal_structure[1] == 'H'):
                
                # Přidáme kontrolní range ATR kolem VWAP a open a close jsou pod VWAP
                if (float(last_close) < float(last_vwap)) and (float(last_open) < float(last_vwap)) and (float(last_close) < float(last_vwap - atr_multiplier * atr)):

                    trend_confirmed = True
                    new_state = 'downtrend'

                else:
                    break_fractal_structure = True

        if trend_invalidated:  ## ano/ne? or break_fractal_structure:
            new_state = None
        elif trend_confirmed:
            new_state = new_state

        # Pokud nedošlo k začátku nového trendu ani invalidaci trendu, tak se trend nezmění
        else:
            new_state = state

        return new_state, losing_trade_timeout, break_fractal_structure

    # 3 Candle Structure

    @staticmethod
    def detect_vwap_breach_candle(candle: pd.Series, vwap: float) -> bool:
        """
        Detect if the candle breaches the VWAP.
        """
        return candle['low'] < vwap  and candle['high'] > vwap 

    @staticmethod
    def detect_cwc_candle(candle: pd.Series, vwap: float, trend: str, cwc_candle_retrace: float) -> bool:
        """
        Detect if the candle is a CWC candle.
        """
        if trend == 'uptrend':
            retrace_ratio = (candle['close'] - candle['low']) / (candle['high'] - candle['low']) if (candle['high'] - candle['low']) != 0 else 0

            if retrace_ratio >= float(cwc_candle_retrace) and candle['close'] > vwap:
                return True

        elif trend == 'downtrend':
            retrace_ratio = (candle['high'] - candle['close']) / (candle['high'] - candle['low']) if (candle['high'] - candle['low']) != 0 else 0

            if retrace_ratio >= float(cwc_candle_retrace) and candle['close'] < vwap:
                return True

        return False

    @staticmethod
    def place_limit_order(cwc_candle: pd.Series, vwap: float, atr: float, trend: str, entry_candle_retrace: float, rr: float) -> bool:
        """
        Place limit order in a candle
        """

        # Determine trade type
        if trend == 'uptrend':
            trade_type = 'long'
        elif trend == 'downtrend':
            trade_type = 'short'
        
        # Determine entry price and SL and TP
        if trade_type == 'long':
            entry_price = cwc_candle['close'] - entry_candle_retrace * (cwc_candle['high'] - cwc_candle['low'])
            sl = entry_price - atr
            tp = entry_price + rr * (entry_price - sl)

        elif trade_type == 'short':
            entry_price = cwc_candle['close'] + entry_candle_retrace * (cwc_candle['high'] - cwc_candle['low'])
            sl = entry_price + atr
            tp = entry_price - rr * (sl - entry_price)

        # Trade size TBD

        trade = {
                    'trade_type': trade_type,
                    'entry_price': float(entry_price),
                    'sl': float(sl),
                    'tp': float(tp),
                    'contracts': 1,
                }

        return trade


    @staticmethod
    def detect_3_candle_structure_window(df: pd.DataFrame, vwap_series: pd.Series, trend_flags: pd.DataFrame, i: int, cwc_candle_retrace: float) -> tuple[bool, float]:
        """
        Inspect a 3-candle window ending at i (i>=2): [i-2]=init, [i-1]=cwc, [i]=entry.
        Returns (is_valid_cwc_at_i_minus_1, fitness_score_for_cwc_or_0).
        """
        init = i - 2
        cwc = i - 1
        entry = i
        if init < 0:
            return False, 0.0

        init_open = df['open'].iloc[init]
        init_low = df['low'].iloc[init]
        init_close = df['close'].iloc[init]
        init_high = df['high'].iloc[init]
        init_vwap = vwap_series.iloc[init]
        if not (init_low < init_vwap and init_high > init_vwap):
            return False, 0.0

        open_cwc = df['open'].iloc[cwc]
        close_cwc = df['close'].iloc[cwc]
        low_cwc = df['low'].iloc[cwc]
        high_cwc = df['high'].iloc[cwc]
        vwap_cwc = vwap_series.iloc[cwc]

        if bool(trend_flags['uptrend'].iloc[init]):
            if not (init_close < init_open):
                return False, 0.0
            if open_cwc < close_cwc and close_cwc > vwap_cwc:
                retrace_ratio = (close_cwc - low_cwc) / (high_cwc - low_cwc) if (high_cwc - low_cwc) != 0 else 0
                fitness_score = StrategyFormulasIterative.fitness_2_candle_structure(
                    init_open, init_high, init_low, init_close,
                    open_cwc, high_cwc, low_cwc, close_cwc,
                    init_vwap, 'bull'
                )
                if retrace_ratio >= float(cwc_candle_retrace) / 100.0:
                    return True, float(fitness_score)
            return False, 0.0
        elif bool(trend_flags['downtrend'].iloc[init]):
            if not (init_close > init_open):
                return False, 0.0
            retrace_init = (init_close - init_low) / (init_high - init_low) if (init_high - init_low) != 0 else 0
            if (retrace_init >= float(cwc_candle_retrace) / 100.0) and (init_close < init_vwap):
                return True, float(retrace_init)
            if open_cwc > close_cwc and close_cwc < vwap_cwc:
                retrace_ratio = (high_cwc - close_cwc) / (high_cwc - low_cwc) if (high_cwc - low_cwc) != 0 else 0
                fitness_score = StrategyFormulasIterative.fitness_2_candle_structure(
                    init_open, init_high, init_low, init_close,
                    open_cwc, high_cwc, low_cwc, close_cwc,
                    init_vwap, 'bear'
                )
                if retrace_ratio >= float(cwc_candle_retrace) / 100.0:
                    return True, float(fitness_score)
            return False, 0.0
        else:
            return False, 0.0

    @staticmethod
    def fitness_2_candle_structure(first_open, first_high, first_low, first_close,
                                   second_open, second_high, second_low, second_close,
                                   line_value, trend_direction='bull') -> float:
        fitness_score = 0.0
        first_crosses_line = (first_low < line_value < first_high)
        if not first_crosses_line:
            return 0.0
        if trend_direction == 'bull':
            if first_close < first_open and second_close > second_open:
                if second_high != second_low:
                    high_close_distance = (second_high - second_close) / (second_high - second_low)
                    factor1 = 1.0 - high_close_distance
                else:
                    factor1 = 0.5
                factor2 = 1.0 if second_close > line_value else 0.0
                second_body_size = abs(second_close - second_open)
                first_body_size = abs(first_close - first_open)
                factor3 = (min(second_body_size / first_body_size, 2.0) / 2.0) if first_body_size > 0 else 0.5
                line_break = max(0, line_value - second_low) / (second_high - second_low) if (second_high - second_low) > 0 else 0
                factor4 = 1.0 - min(line_break, 1.0)
                weights = [0.6, 0.15, 0.15, 0.10]
                fitness_score = (factor1 * weights[0] + factor2 * weights[1] + factor3 * weights[2] + factor4 * weights[3])
        elif trend_direction == 'bear':
            if first_close > first_open and second_close < second_open:
                if second_high != second_low:
                    low_close_distance = (second_close - second_low) / (second_high - second_low)
                    factor1 = 1.0 - low_close_distance
                else:
                    factor1 = 0.5
                factor2 = 1.0 if second_close < line_value else 0.0
                second_body_size = abs(second_close - second_open)
                first_body_size = abs(first_close - first_open)
                factor3 = (min(second_body_size / first_body_size, 2.0) / 2.0) if first_body_size > 0 else 0.5
                line_break = max(0, second_high - line_value) / (second_high - second_low) if (second_high - second_low) > 0 else 0
                factor4 = 1.0 - min(line_break, 1.0)
                weights = [0.6, 0.15, 0.15, 0.10]
                fitness_score = (factor1 * weights[0] + factor2 * weights[1] + factor3 * weights[2] + factor4 * weights[3])
        return float(fitness_score)

    @staticmethod
    def detect_trades_iterative(
        df: pd.DataFrame,
        vwap_series: pd.Series,
        *,
        rr: float = 1.0,
        open_candle_retrace: float = 0.0,
        cwc_retrace_pct: float = 0.0,
        atr_period: int = 14,
        df_fractals_confirmed: pd.DataFrame | None = None,
    ) -> list[dict]:
        """
        Iterative trade detection over df, using only available info at each step:
        - Confirms CWC using a 3-candle rolling window [i-2, i-1, i] when i arrives.
        - Entries are evaluated on candle i (next after CWC) at a retrace level.
        - Exits (SL/TP) are evaluated on subsequent candles, one by one.
        - ATR is updated incrementally per candle via Wilder's smoothing.
        - Trend flags are derived iteratively from confirmed fractals if provided; else neutral.
        """
        trades: list[dict] = []
        n = len(df)
        if n < 3:
            return trades

        # Incremental ATR
        ind_inc = IndicatorFormulas_iterative(df)
        prev_atr: float | None = None
        prev_close: float | None = None
        atr_series = pd.Series(np.nan, index=df.index)

        # Trend flags from confirmed fractals if provided
        if df_fractals_confirmed is not None:
            trend_flags = StrategyFormulasIterative.detect_trend_zones_iterative(df_fractals_confirmed, vwap_series)
        else:
            trend_flags = pd.DataFrame(index=df.index, data={
                'uptrend': pd.Series(False, index=df.index),
                'downtrend': pd.Series(False, index=df.index),
            })

        open_trade: dict | None = None

        for i, ts in enumerate(df.index):
            high_i = float(df['high'].iloc[i])
            low_i = float(df['low'].iloc[i])
            close_i = float(df['close'].iloc[i])

            # ATR update
            prev_atr = ind_inc.atr_update(
                prev_atr=prev_atr,
                prev_close=prev_close,
                new_high=high_i,
                new_low=low_i,
                new_close=close_i,
                period=int(atr_period),
            )
            atr_series.iloc[i] = prev_atr
            prev_close = close_i

            # Check exit for open trade first
            if open_trade is not None:
                bar_high = high_i
                bar_low = low_i
                if open_trade['trade_type'] == 'long':
                    if bar_low <= open_trade['sl']:
                        trades.append({**open_trade, 'exit_idx': ts, 'exit_price': float(open_trade['sl']), 'result': 'SL'})
                        open_trade = None
                    elif open_trade.get('tp') is not None and bar_high >= open_trade['tp']:
                        trades.append({**open_trade, 'exit_idx': ts, 'exit_price': float(open_trade['tp']), 'result': 'TP'})
                        open_trade = None
                else:
                    if bar_high >= open_trade['sl']:
                        trades.append({**open_trade, 'exit_idx': ts, 'exit_price': float(open_trade['sl']), 'result': 'SL'})
                        open_trade = None
                    elif open_trade.get('tp') is not None and bar_low <= open_trade['tp']:
                        trades.append({**open_trade, 'exit_idx': ts, 'exit_price': float(open_trade['tp']), 'result': 'TP'})
                        open_trade = None

            # If a trade is still open, do not process new signals
            if open_trade is not None:
                continue

            # Confirm potential CWC at i-1 when i arrives
            if i >= 2:
                is_cwc, fitness = StrategyFormulasIterative.detect_3_candle_structure_window(
                    df, vwap_series, trend_flags, i, cwc_candle_retrace=float(cwc_retrace_pct)
                )
                if is_cwc:
                    cwc_pos = i - 1
                    entry_pos = i
                    entry_idx = df.index[entry_pos]
                    entry_open = float(df['open'].iloc[entry_pos])
                    prev_high = float(df['high'].iloc[cwc_pos])
                    prev_low = float(df['low'].iloc[cwc_pos])
                    prev_range = max(prev_high - prev_low, 0.0)
                    retrace_pct = max(0.0, float(open_candle_retrace))

                    close_cwc = float(df['close'].iloc[cwc_pos])
                    open_cwc = float(df['open'].iloc[cwc_pos])
                    vwap_cwc = float(vwap_series.iloc[cwc_pos])
                    if close_cwc > open_cwc and close_cwc > vwap_cwc:
                        trade_type = 'long'
                        target_entry = entry_open - retrace_pct * prev_range
                        entry_hit = (float(df['low'].iloc[entry_pos]) <= target_entry <= float(df['high'].iloc[entry_pos]))
                    elif close_cwc < open_cwc and close_cwc < vwap_cwc:
                        trade_type = 'short'
                        target_entry = entry_open + retrace_pct * prev_range
                        entry_hit = (float(df['low'].iloc[entry_pos]) <= target_entry <= float(df['high'].iloc[entry_pos]))
                    else:
                        trade_type = None
                        entry_hit = False

                    if trade_type and entry_hit:
                        entry_price = float(target_entry)
                        vwap_at_entry = float(vwap_series.iloc[entry_pos])
                        atr_at_entry = float(atr_series.iloc[entry_pos]) if pd.notna(atr_series.iloc[entry_pos]) else 0.0
                        rr_val = max(0.0, float(rr))
                        if trade_type == 'long':
                            sl = vwap_at_entry - atr_at_entry
                            tp = entry_price + rr_val * atr_at_entry
                        else:
                            sl = vwap_at_entry + atr_at_entry
                            tp = entry_price - rr_val * atr_at_entry

                        open_trade = {
                            'trade_type': trade_type,
                            'entry_idx': entry_idx,
                            'entry_price': float(entry_price),
                            'sl': float(sl),
                            'tp': float(tp),
                            'contracts': 1,
                        }

        return trades

    @staticmethod
    def detect_trades_update(
        *,
        state: dict | None,
        i: int,
        index: pd.DatetimeIndex,
        candles: pd.DataFrame,
        vwap_series: pd.Series,
        atr_series: pd.Series | None,
        rr: float = 1.0,
        open_candle_retrace: float = 0.0,
        cwc_retrace_pct: float = 0.0,
        trend_flags: pd.DataFrame | None = None,
        on_sl_cooldown_bars: int = 0,
    ) -> tuple[dict, list[dict], int]:
        """
        Update-style trade detection that respects non-anticipation:
        - When a CWC is confirmed at i-1 using window ending at i, schedule an entry for i (next bar after CWC confirmation).
        - Entries are placed only on the bar following the decision point (no past placement).
        - Exits (SL/TP) are evaluated per subsequent bar.

        State schema:
          {
            'open_trade': Optional[dict],
            'pending_signal': Optional[dict],  # {'cwc_pos': int, 'direction': 'long'|'short', 'entry_pos': int, 'prev_range': float}
          }

        Returns updated (state, events, losing_trade_timeout)
        """
        if state is None:
            state = {'open_trade': None, 'pending_signal': None}
        events: list[dict] = []
        losing_trade_timeout = 0

        n = len(index)
        if i < 0 or i >= n:
            return state, events, losing_trade_timeout

        # 1) Manage open trade exits on current bar i
        open_trade = state.get('open_trade')
        if open_trade is not None:
            bar_high = float(candles['high'].iloc[i])
            bar_low = float(candles['low'].iloc[i])
            ts = index[i]
            if open_trade['trade_type'] == 'long':
                if bar_low <= open_trade['sl']:
                    trade = dict(open_trade)
                    trade.update({'exit_idx': ts, 'exit_price': float(open_trade['sl']), 'result': 'SL'})
                    events.append({'event': 'close', 'trade': trade})
                    open_trade = None
                    losing_trade_timeout = int(on_sl_cooldown_bars)
                elif open_trade.get('tp') is not None and bar_high >= open_trade['tp']:
                    trade = dict(open_trade)
                    trade.update({'exit_idx': ts, 'exit_price': float(open_trade['tp']), 'result': 'TP'})
                    events.append({'event': 'close', 'trade': trade})
                    open_trade = None
            else:  # short
                if bar_high >= open_trade['sl']:
                    trade = dict(open_trade)
                    trade.update({'exit_idx': ts, 'exit_price': float(open_trade['sl']), 'result': 'SL'})
                    events.append({'event': 'close', 'trade': trade})
                    open_trade = None
                    losing_trade_timeout = int(on_sl_cooldown_bars)
                elif open_trade.get('tp') is not None and bar_low <= open_trade['tp']:
                    trade = dict(open_trade)
                    trade.update({'exit_idx': ts, 'exit_price': float(open_trade['tp']), 'result': 'TP'})
                    events.append({'event': 'close', 'trade': trade})
                    open_trade = None
        state['open_trade'] = open_trade

        # 2) If a pending signal exists and current i is its entry bar, try to open
        pending = state.get('pending_signal')
        if pending is not None and int(pending.get('entry_pos', -1)) == i and state.get('open_trade') is None:
            trade_type = pending.get('direction')
            entry_open = float(candles['open'].iloc[i])
            prev_range = float(pending.get('prev_range', 0.0))
            retrace_pct = max(0.0, float(open_candle_retrace))
            target_entry = entry_open - retrace_pct * prev_range if trade_type == 'long' else entry_open + retrace_pct * prev_range
            entry_low = float(candles['low'].iloc[i])
            entry_high = float(candles['high'].iloc[i])
            if entry_low <= target_entry <= entry_high:
                entry_idx = index[i]
                vwap_at_entry = float(vwap_series.iloc[i])
                atr_at_entry = float(atr_series.iloc[i]) if (atr_series is not None and pd.notna(atr_series.iloc[i])) else 0.0
                rr_val = max(0.0, float(rr))
                if trade_type == 'long':
                    sl = vwap_at_entry - atr_at_entry
                    tp = target_entry + rr_val * atr_at_entry
                else:
                    sl = vwap_at_entry + atr_at_entry
                    tp = target_entry - rr_val * atr_at_entry
                open_trade = {
                    'trade_type': trade_type,
                    'entry_idx': entry_idx,
                    'entry_price': float(target_entry),
                    'sl': float(sl),
                    'tp': float(tp),
                    'contracts': 1,
                }
                state['open_trade'] = open_trade
                events.append({'event': 'open', 'trade': dict(open_trade)})
            # Clear pending whether filled or not; if not filled, signal expires
            state['pending_signal'] = None

        # 3) If no open trade and no pending, check for new CWC confirmation using window ending at i
        if state.get('open_trade') is None and state.get('pending_signal') is None and i >= 2:
            # require trend flags
            if trend_flags is None:
                trend_flags = pd.DataFrame(index=index, data={'uptrend': False, 'downtrend': False})
            is_cwc, fitness = StrategyFormulasIterative.detect_3_candle_structure_window(
                candles, vwap_series, trend_flags, i, cwc_candle_retrace=float(cwc_retrace_pct)
            )
            if is_cwc:
                cwc_pos = i - 1
                close_cwc = float(candles['close'].iloc[cwc_pos])
                open_cwc = float(candles['open'].iloc[cwc_pos])
                vwap_cwc = float(vwap_series.iloc[cwc_pos])
                prev_high = float(candles['high'].iloc[cwc_pos])
                prev_low = float(candles['low'].iloc[cwc_pos])
                prev_range = max(prev_high - prev_low, 0.0)
                if close_cwc > open_cwc and close_cwc > vwap_cwc:
                    direction = 'long'
                elif close_cwc < open_cwc and close_cwc < vwap_cwc:
                    direction = 'short'
                else:
                    direction = None
                if direction is not None and (i + 1) < n:
                    state['pending_signal'] = {
                        'cwc_pos': cwc_pos,
                        'direction': direction,
                        'entry_pos': i + 1,
                        'prev_range': prev_range,
                    }

        return state, events, losing_trade_timeout

    


