import pandas as pd
from typing import List, Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt
import math

from Indicators.indicator_formulas import IndicatorFormulas
from Strategy.ib_vwap_formulas import StrategyFormulas
from Strategy.session_runner_iterative import NYVWAPSessionRunner
from Strategy.double_sma_formulas import DoubleSMAStrategy


class ParameterOptimizer:
    """
    Grid-test optimizer for R:R parameter on a train/validation split.
    """

    @staticmethod
    def train_validate_split(df: pd.DataFrame, train_ratio: float = 0.7) -> Tuple[pd.DataFrame, pd.DataFrame]:
        train_ratio = max(0.0, min(1.0, float(train_ratio)))
        split_idx = int(len(df) * train_ratio)
        return df.iloc[:split_idx], df.iloc[split_idx:]

    @staticmethod
    def print_results_table(results: Dict[str, Dict[str, float]]):
        if not results:
            print("No optimization results.")
            return

        rr_cols = sorted(results.keys(), key=lambda x: float(x))
        rows = ['total_pnl', 'win_rate_pct']

        # Header
        header = [f"RR={c:>5}" for c in rr_cols]
        print("Optimization Results (Validation)")
        print("=" * (len(header) * 12))

    @staticmethod
    def backtest_session_runner(
        df: pd.DataFrame,
        vwap: pd.Series,
        open_retrace_values: List[float],
        cwc_retrace_values: List[float],
        rr_values: List[float],
        train_ratio: float = 0.7,
        *,
        session_open_utc: str = "16:00",
        session_hours: float = 1.5,
        atr_multiplier: float = 1.0,
        show_plot: bool = False,
    ) -> Dict[str, Dict[str, Dict[str, Dict[str, float]]]]:
        """
        Optimize combinations of (open_retrace, cwc_retrace, rr) on the train split,
        then evaluate on the validation split.

        """
        train_df, val_df = ParameterOptimizer.train_validate_split(df, train_ratio=train_ratio)
        vwap_train = vwap.loc[train_df.index]
        vwap_val = vwap.loc[val_df.index]

        results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}

        total = max(1, len(open_retrace_values) * len(cwc_retrace_values) * len(rr_values))
        progress = 0

        # 1) TRAIN pass: compute metrics and pick best by train_total_pnl
        print('Starting TRAIN optimization pass...')
        train_metrics: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
        train_progress = 0
        for open_retrace in open_retrace_values:
            open_key = str(float(open_retrace))
            if open_key not in train_metrics:
                train_metrics[open_key] = {}
            for cwc_retrace in cwc_retrace_values:
                cwc_key = str(float(cwc_retrace))
                if cwc_key not in train_metrics[open_key]:
                    train_metrics[open_key][cwc_key] = {}
                for rr in rr_values:
                    train_progress += 1
                    # Use iterative session runner to generate trades on TRAIN split
                    runner_tr = NYVWAPSessionRunner(session_open_utc=session_open_utc, session_hours=float(session_hours), max_sessions=None)
                    trades_tr = runner_tr.run_and_plot_sessions_iterative(
                        train_df,
                        start_date=None,
                        open_retrace=float(open_retrace),
                        cwc_retrace=float(cwc_retrace),
                        rr=float(rr),
                        atr_multiplier=float(atr_multiplier),
                        show_plot=bool(show_plot),
                    )
                    pnls_tr = []
                    wins_tr = 0
                    count_tr = 0
                    if trades_tr:
                        for t in trades_tr:
                            count_tr += 1
                            pnl = None
                            side = t.get('trade_type')
                            ep = t.get('entry_price')
                            xp = t.get('exit_price')
                            if side in {'long','short'} and ep is not None and xp is not None:
                                try:
                                    ep = float(ep); xp = float(xp)
                                    pnl = (xp - ep) if side == 'long' else (ep - xp)
                                except Exception:
                                    pnl = None
                            pnl = float(pnl) if pnl is not None else 0.0
                            pnls_tr.append(pnl)
                            if pnl > 0:
                                wins_tr += 1

                    train_total = float(sum(pnls_tr)) if pnls_tr else 0.0
                    train_win = (wins_tr / count_tr * 100.0) if count_tr > 0 else 0.0
                    train_metrics[open_key][cwc_key][str(rr)] = {
                        'train_total_pnl': train_total,
                        'train_win_rate_pct': train_win,
                        'trades_train': int(count_tr),
                    }

                    print(
                        f"[TRAIN] ({train_progress}/{total}) open={open_retrace:.2f}, "
                        f"cwc={cwc_retrace:.1f}, RR={rr:.2f} -> Train PnL {train_total:,.2f} (Trades {count_tr})"
                    )

        # Select best by train_total_pnl and rank all combos
        best_open_key = None
        best_cwc_key = None
        best_rr_key = None
        best_train_total = -float('inf')
        train_rankings: list[tuple[str, str, str, float]] = []
        for open_key, lvl1 in train_metrics.items():
            for cwc_key, lvl2 in lvl1.items():
                for rr_key, metrics in lvl2.items():
                    tr_tot = float(metrics.get('train_total_pnl', 0.0))
                    train_rankings.append((open_key, cwc_key, rr_key, tr_tot))
                    if tr_tot > best_train_total:
                        best_train_total = tr_tot
                        best_open_key = open_key
                        best_cwc_key = cwc_key
                        best_rr_key = rr_key

        # Keep only top-K combinations for validation
        TOP_K_VALIDATE = 5
        train_rankings.sort(key=lambda x: x[3], reverse=True)
        top_combos: list[tuple[str, str, str, float]] = train_rankings[:min(TOP_K_VALIDATE, len(train_rankings))]

        # 2) VALIDATION pass: compute metrics for all combos (compatibility) and attach combined + is_best
        print('Starting VALIDATION evaluation pass...')
        total_val = max(1, len(top_combos))
        for open_key, cwc_key, rr_key, _ in top_combos:
            if open_key not in results:
                results[open_key] = {}
            if cwc_key not in results[open_key]:
                results[open_key][cwc_key] = {}

            open_retrace = float(open_key)
            cwc_retrace = float(cwc_key)
            rr = float(rr_key)

            progress += 1
            # Use iterative session runner to generate trades on VALIDATION split
            runner_val = NYVWAPSessionRunner(session_open_utc=session_open_utc, session_hours=float(session_hours), max_sessions=None)
            trades_val = runner_val.run_and_plot_sessions_iterative(
                val_df,
                start_date=None,
                open_retrace=float(open_retrace),
                cwc_retrace=float(cwc_retrace),
                rr=float(rr),
                atr_multiplier=float(atr_multiplier),
                show_plot=bool(show_plot),
            )
            pnls_val = []
            wins_val = 0
            count_val = 0
            if trades_val:
                for t in trades_val:
                    count_val += 1
                    pnl = None
                    side = t.get('trade_type')
                    ep = t.get('entry_price')
                    xp = t.get('exit_price')
                    if side in {'long','short'} and ep is not None and xp is not None:
                        try:
                            ep = float(ep); xp = float(xp)
                            pnl = (xp - ep) if side == 'long' else (ep - xp)
                        except Exception:
                            pnl = None
                    pnl = float(pnl) if pnl is not None else 0.0
                    pnls_val.append(pnl)
                    if pnl > 0:
                        wins_val += 1

            val_total = float(sum(pnls_val)) if pnls_val else 0.0
            val_win = (wins_val / count_val * 100.0) if count_val > 0 else 0.0

            trm = train_metrics.get(open_key, {}).get(cwc_key, {}).get(str(rr), {
                'train_total_pnl': 0.0, 'train_win_rate_pct': 0.0, 'trades_train': 0
            })
            combined_total = float(trm.get('train_total_pnl', 0.0)) + float(val_total)
            is_best = bool(open_key == best_open_key and cwc_key == best_cwc_key and str(rr_key) == best_rr_key)

            results[open_key][cwc_key][str(rr)] = {
                'total_pnl': float(val_total),
                'win_rate_pct': float(val_win),
                'trades': int(count_val),
                'train_total_pnl': float(trm.get('train_total_pnl', 0.0)),
                'train_win_rate_pct': float(trm.get('train_win_rate_pct', 0.0)),
                'trades_train': int(trm.get('trades_train', 0)),
                'val_total_pnl': float(val_total),
                'val_win_rate_pct': float(val_win),
                'trades_val': int(count_val),
                'combined_total_pnl': float(combined_total),
                'is_best': is_best,
            }

            print(
                f"[VAL]   ({progress}/{total_val}) open={open_retrace:.2f}, "
                f"cwc={cwc_retrace:.1f}, RR={rr:.2f} -> "
                f"Train {trm.get('train_total_pnl', 0.0):,.2f} (Trades {int(trm.get('trades_train', 0))}), "
                f"Val {val_total:,.2f} (Trades {count_val}), Combined {combined_total:,.2f}"
            )

        return results

    @staticmethod
    def print_results_top(results: Dict[str, Dict[str, Dict[str, Dict[str, float]]]], top_k: int = 10):
        """
        Print top-k parameter combinations by total_pnl.
        """
        flat: List[Tuple[float, float, float, float, float, int]] = []
        # (open_retrace, cwc_retrace, rr, total_pnl, win_rate_pct, trades)
        for open_key, lvl1 in results.items():
            for cwc_key, lvl2 in lvl1.items():
                for rr_key, metrics in lvl2.items():
                    try:
                        flat.append(
                            (
                                float(open_key),
                                float(cwc_key),
                                float(rr_key),
                                float(metrics.get('total_pnl', 0.0)),
                                float(metrics.get('win_rate_pct', 0.0)),
                                int(metrics.get('trades', 0)),
                            )
                        )
                    except Exception:
                        continue

        if not flat:
            print('No optimization results.')
            return

        flat.sort(key=lambda x: x[3], reverse=True)
        print('Top parameter combinations by total PnL:')
        print('=' * 78)

    # ===== Double SMA optimizer =====
    @staticmethod
    def optimize_double_sma(
        df_1m: pd.DataFrame,
        *,
        sma_short_values: list[int],
        sma_long_values: list[int],
        trend_slope_values: list[float],
        rr_values: list[float],
        train_ratio: float = 0.7,
        session_only: bool = True,
        base_usd_allocation: float = 100.0,
    ) -> list[dict]:
        """
        Grid-search optimizer for the Double SMA strategy on validation split.

        Returns list of dict rows with metrics for each parameter combo on validation set:
          {
            'sma_short', 'sma_long', 'trend_slope', 'rr',
            'trades', 'total_usd_pnl', 'avg_usd_per_trade', 'win_rate_pct'
          }
        """
        if df_1m is None or df_1m.empty:
            return []

        # Optionally filter to NY session hours first (keeps only 09:30–13:30 ET)
        df = df_1m.copy()
        if session_only:
            try:
                mask = (df.index.time >= pd.Timestamp('09:30').time()) & (df.index.time < pd.Timestamp('13:30').time())
                df = df.loc[mask]
            except Exception:
                pass

        # Train/validation split
        train_df, val_df = ParameterOptimizer.train_validate_split(df, train_ratio=train_ratio)

        results: list[dict] = []
        total = max(1, len(sma_short_values) * len(sma_long_values) * len(trend_slope_values) * len(rr_values))
        progress = 0

        for s_short in sma_short_values:
            for s_long in sma_long_values:
                for slope in trend_slope_values:
                    # Compute signals on validation subset for given params
                    strat = DoubleSMAStrategy(val_df)
                    signals = strat.compute_signals(
                        short_period=int(s_short),
                        long_period=int(s_long),
                        use_trend_filter=True,
                        trend_slope=float(slope),
                    )
                    for rr in rr_values:
                        progress += 1
                        trades = DoubleSMAStrategy.generate_trades(
                            signals_df=signals,
                            rr=float(rr),
                            atr_window=14,
                            ohlcv_1m=val_df,
                        )
                        count = len(trades) if trades else 0
                        wins = 0
                        total_usd = 0.0
                        # Build cumulative equity curve (USD) across validation index
                        equity_curve = None
                        if trades:
                            for t in trades:
                                ret_pct = t.get('return_pct')
                                pnl_usd = 0.0
                                try:
                                    if ret_pct is not None:
                                        pnl_usd = float(base_usd_allocation) * float(ret_pct)
                                except Exception:
                                    pnl_usd = 0.0
                                total_usd += pnl_usd
                                # win if pnl_usd > 0
                                if pnl_usd > 0:
                                    wins += 1

                            # Compute equity curve using provided helper (falls back to 0s if needed)
                            try:
                                equity_curve = DoubleSMAStrategy.cumulative_pnl_usd_series(
                                    trades,
                                    val_df.index,
                                    base_allocation_usd=float(base_usd_allocation),
                                )
                            except Exception:
                                equity_curve = pd.Series(0.0, index=val_df.index)
                        else:
                            equity_curve = pd.Series(0.0, index=val_df.index)

                        avg_usd = (total_usd / count) if count > 0 else 0.0
                        win_rate = (wins / count * 100.0) if count > 0 else 0.0
                        results.append({
                            'sma_short': int(s_short),
                            'sma_long': int(s_long),
                            'trend_slope': float(slope),
                            'rr': float(rr),
                            'trades': int(count),
                            'total_usd_pnl': float(total_usd),
                            'avg_usd_per_trade': float(avg_usd),
                            'win_rate_pct': float(win_rate),
                            # Store equity curve for plotting
                            'equity_curve': equity_curve,
                        })

                        print(
                            f"Opt ({progress}/{total}) short={s_short} long={s_long} slope={slope:.4f} RR={rr} -> $ {total_usd:,.2f}, win {win_rate:.2f}% ({count})"
                        )

        # Sort best first by total_usd_pnl desc
        results.sort(key=lambda r: r.get('total_usd_pnl', 0.0), reverse=True)
        return results

    @staticmethod
    def print_double_sma_top(results: list[dict], top_k: int = 10):
        if not results:
            print('No optimization results.')
            return
        print('Top Double SMA parameter combinations by total USD PnL:')
        print('=' * 92)
        print(f"{'rank':>4}  {'short':>5}  {'long':>5}  {'slope':>8}  {'RR':>5}  {'PnL$':>12}  {'Win%':>7}  {'Trades':>6}  {'Avg$':>10}")
        print('-' * 92)
        for rank, row in enumerate(results[:max(1, int(top_k))], start=1):
            print(
                f"{rank:>4}  {row['sma_short']:5d}  {row['sma_long']:5d}  {row['trend_slope']:8.4f}  {row['rr']:5.2f}  "
                f"{row['total_usd_pnl']:12,.2f}  {row['win_rate_pct']:7.2f}  {row['trades']:6d}  {row['avg_usd_per_trade']:10.2f}"
            )
        print('=' * 92)

    @staticmethod
    def permute_candles_log(
        df: pd.DataFrame,
        *,
        seed: int | None = None,
        columns: tuple[str, str, str, str] = ('open', 'high', 'low', 'close'),
        shuffle_volume: bool = False,
        num_permutations: int = 1,
    ) -> pd.DataFrame | list[pd.DataFrame]:
        """
        Create a permuted OHLC series by shuffling candle-relative log movements.

        Logic:
        - Keep the first OHLC candle unchanged
        - For each subsequent candle i: compute
            open_rel  = log(open_i / close_{i-1})
            high_rel  = log(high_i / open_i)
            low_rel   = log(low_i / open_i)
            close_rel = log(close_i / open_i)
        - Randomly permute the tuples (open_rel, high_rel, low_rel, close_rel) for i >= 1
        - Reconstruct the OHLC path by sequentially applying the permuted log moves,
          with each new open based on the previous synthetic close.

        Parameters:
        - df: DataFrame with at least columns ['open','high','low','close'] and a DatetimeIndex
        - seed: optional RNG seed for reproducibility
        - columns: tuple specifying the OHLC column names in df
        - shuffle_volume: if True and a 'volume' column exists, volume for i>=1 will be
          permuted according to the same permutation; otherwise volume is left as-is

        Returns:
        - If num_permutations == 1: a new DataFrame with the same index and columns
        - If num_permutations  > 1: a list of DataFrames, one per permutation
        """
        o_col, h_col, l_col, c_col = columns

        if df is None or df.empty:
            return df.copy()
        for col in [o_col, h_col, l_col, c_col]:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        n = len(df)
        if n < 2:
            return df.copy()

        if int(num_permutations) < 1:
            raise ValueError("num_permutations must be >= 1")

        df_float = df.copy()
        # Ensure float types for safe numeric ops
        for col in [o_col, h_col, l_col, c_col]:
            df_float[col] = pd.to_numeric(df_float[col], errors='coerce').astype(float)

        # Compute relative log components for i>=1
        prev_close = df_float[c_col].shift(1).iloc[1:].values
        opens = df_float[o_col].iloc[1:].values
        highs = df_float[h_col].iloc[1:].values
        lows = df_float[l_col].iloc[1:].values
        closes = df_float[c_col].iloc[1:].values

        with np.errstate(divide='ignore', invalid='ignore'):
            open_rel_log = np.log(opens / prev_close)
            high_rel_log = np.log(highs / opens)
            low_rel_log = np.log(lows / opens)
            close_rel_log = np.log(closes / opens)

        m = n - 1
        # For small m, compute exact maximum unique permutations and validate request
        if m <= 12:
            max_perms = math.factorial(m)
            if num_permutations > max_perms:
                raise ValueError(
                    f"Requested num_permutations={num_permutations} exceeds maximum unique permutations={max_perms} for m={m}"
                )

        rng = np.random.default_rng(seed) if hasattr(np.random, 'default_rng') else np.random.RandomState(seed)

        def build_from_perm(perm_idx: np.ndarray) -> pd.DataFrame:
            open_rel_perm = open_rel_log[perm_idx]
            high_rel_perm = high_rel_log[perm_idx]
            low_rel_perm = low_rel_log[perm_idx]
            close_rel_perm = close_rel_log[perm_idx]

            # Reconstruct synthetic OHLC
            open_new = np.empty(n, dtype=float)
            high_new = np.empty(n, dtype=float)
            low_new = np.empty(n, dtype=float)
            close_new = np.empty(n, dtype=float)

            # First candle unchanged
            open_new[0] = float(df_float.iloc[0][o_col])
            high_new[0] = float(df_float.iloc[0][h_col])
            low_new[0] = float(df_float.iloc[0][l_col])
            close_new[0] = float(df_float.iloc[0][c_col])

            prev_c = close_new[0]
            for i in range(1, n):
                j = i - 1  # index into perm arrays
                o_i = prev_c * float(np.exp(open_rel_perm[j]))
                h_i = o_i * float(np.exp(high_rel_perm[j]))
                l_i = o_i * float(np.exp(low_rel_perm[j]))
                c_i = o_i * float(np.exp(close_rel_perm[j]))

                # Ensure OHLC consistency
                hi_bound = max(h_i, o_i, c_i)
                lo_bound = min(l_i, o_i, c_i)

                open_new[i] = o_i
                high_new[i] = hi_bound
                low_new[i] = lo_bound
                close_new[i] = c_i

                prev_c = c_i

            out = df.copy()
            out[o_col] = open_new
            out[h_col] = high_new
            out[l_col] = low_new
            out[c_col] = close_new

            # Optionally permute volume aligned to the same permutation
            if shuffle_volume and 'volume' in df.columns:
                vol = out['volume'].values.copy()
                if len(vol) == n:
                    vol_perm = vol.copy()
                    vol_perm[0] = vol[0]
                    vol_perm[1:] = vol[1:][perm_idx]
                    out['volume'] = vol_perm

            return out

        if num_permutations == 1:
            perm = rng.permutation(m)
            return build_from_perm(perm)
        else:
            out_list: list[pd.DataFrame] = []
            for _ in range(int(num_permutations)):
                perm = rng.permutation(m)
                out_list.append(build_from_perm(perm))
            return out_list

    @staticmethod
    def plot_double_sma_equity_curves(
        results: list[dict],
        *,
        alpha_other_min: float = 0.15,
        linewidth_other: float = 1.0,
        linewidth_best: float = 2.5,
        title: str | None = None,
    ) -> None:
        """
        Plot equity curves for all optimization runs on the same axes.

        - Uses alpha scaling to fade less profitable curves.
        - The most profitable (by total_usd_pnl) is highlighted with alpha=1.0 and thicker line.
        """
        if not results:
            print('No optimization results to plot.')
            return

        # Keep only rows with equity curves
        curves = [r for r in results if isinstance(r.get('equity_curve'), pd.Series)]
        if not curves:
            print('No equity curves recorded in results.')
            return

        # Rank by total_usd_pnl
        curves_sorted = sorted(curves, key=lambda r: r.get('total_usd_pnl', 0.0))
        min_pnl = curves_sorted[0].get('total_usd_pnl', 0.0)
        max_pnl = curves_sorted[-1].get('total_usd_pnl', 0.0)
        span = max(1e-9, float(max_pnl - min_pnl))

        fig, ax = plt.subplots(figsize=(10, 5))

        best_row = None
        for row in curves_sorted:
            equity: pd.Series = row['equity_curve']
            pnl = float(row.get('total_usd_pnl', 0.0))
            score = (pnl - min_pnl) / span
            alpha = alpha_other_min + (1.0 - alpha_other_min) * score
            lw = linewidth_other
            color = 'tab:gray'

            # Defer plotting best until end to overlay on top
            if pnl >= max_pnl - 1e-12:
                best_row = row
                continue

            try:
                ax.plot(equity.index, equity.values, color=color, alpha=alpha, linewidth=lw)
            except Exception:
                # Best effort; skip malformed series
                continue

        # Plot the best on top
        if best_row is not None:
            eq_best: pd.Series = best_row['equity_curve']
            label = (
                f"best: s={best_row['sma_short']}, l={best_row['sma_long']}, "
                f"slope={best_row['trend_slope']:.4f}, RR={best_row['rr']:.2f}, "
                f"PnL=$ {best_row['total_usd_pnl']:,.2f}"
            )
            ax.plot(eq_best.index, eq_best.values, color='tab:blue', alpha=1.0, linewidth=linewidth_best, label=label)
            ax.legend(loc='upper left')

        ax.set_title(title or 'Double SMA Optimization Equity Curves (Validation)')
        ax.set_ylabel('Cumulative PnL (USD)')
        ax.grid(True, linestyle='--', alpha=0.3)
        plt.tight_layout()
        plt.show(block=True)


    @staticmethod
    def plot_permutation_histogram(
        real_value: float,
        permuted_values: list[float],
        *,
        bins: int = 30,
        title: str | None = None,
        xlabel: str = 'Combined total PnL',
        ylabel: str = 'Frequency',
    ) -> None:
        """
        Plot histogram of permuted scenario scores with a vertical line for the real scenario.

        Parameters
        - real_value: scalar metric for the real (non-permuted) data
        - permuted_values: list of scalar metrics for each permutation
        - bins: histogram bins
        - title, xlabel, ylabel: plot labels
        """
        try:
            fig, ax = plt.subplots(figsize=(9, 5))
            data = [float(v) for v in permuted_values if v is not None]
            if len(data) == 0:
                ax.text(0.5, 0.5, 'No permutation results to plot', ha='center', va='center', transform=ax.transAxes)
            else:
                ax.hist(data, bins=int(max(5, bins)), color='tab:gray', alpha=0.7, label='Permutations')
                ax.axvline(float(real_value), color='tab:blue', linestyle='--', linewidth=2.0, label=f"Real: {real_value:,.2f}")
            ax.set_title(title or 'Permutation distribution vs real')
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.legend(loc='upper right')
            ax.grid(True, linestyle='--', alpha=0.3)
            plt.tight_layout()
            plt.show(block=True)
        except Exception:
            # Best-effort; don't fail the run due to plotting errors
            pass

