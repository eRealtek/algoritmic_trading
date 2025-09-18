import pandas as pd
import numpy as np
import mplfinance as mpf
from Data.futures_reader import FuturesReader
import matplotlib.pyplot as plt
from Indicators.indicator_formulas import IndicatorFormulas
from Strategy.ib_vwap_formulas_iterative import StrategyFormulasIterative
from Strategy.trade_analysis import TradeAnalysis
from Strategy.optimizer import ParameterOptimizer
from Strategy.session_runner_iterative import NYVWAPSessionRunner

def main(filepath, freq, start: str | None = None, optimize: bool = False, num_permutation: int = 1, permute_seed: int | None = 42, atr_multiplier: float = 1.0):

    # --- USER SETUP --- #

    max_sessions = 50
    open_retrace_pct = 0  # default; will be set by best params
    cwc_retrace_pct = 0.3   # default; will be set by best params
    rr = 1.0                # default; will be set by best params
    atr_multiplier = float(atr_multiplier)
    trades_data = False # Trade or OHLCV data
    show_plot = False

    ### --- Read and prepare data --- ###

    if trades_data:
        reader = FuturesReader(filepath)
        df = reader.read()
        df.set_index("timestamp", inplace=True)
        df.index = df.index.astype("int64") // 10**6    # Convert to timestamp numerical format in milliseconds

        # Agregate trades to OHLCV candles
        candles_all = FuturesReader.aggregate_to_candles(df, freq, save_csv = True)
    else:
        candles_file_path = "Data/Futures/OHLCV/NQ_ohlcv.csv"
        candles_all = pd.read_csv(candles_file_path, index_col=0, parse_dates=True)

    candles_all = candles_all.astype({"open": float, "high": float, "low": float, "close": float, "volume": float})

    # --- Convert to New York time for session logic ---
    candles_all.index = pd.to_datetime(candles_all.index, unit="ms", utc=True)
    ny_df = candles_all.tz_convert("America/New_York")
    session_starts = ny_df.between_time("09:30", "09:30").index.normalize().unique()

    # Optional: filter sessions to start on/after provided start date
    start_date = start

    if start_date is not None:
        # start_date = pd.Timestamp(start_ts.date()).tz_localize(start_ts.tz)
        session_starts = session_starts[session_starts >= start_date]

    # ----- Loop 1: choose data scenario by permutation count ----- #
    
    if int(num_permutation) <= 1:
        permutations = [candles_all]
    else:
        perms = ParameterOptimizer.permute_candles_log(
            candles_all,
            seed=permute_seed,
            num_permutations=int(num_permutation),
        )
        permutations = perms if isinstance(perms, list) else [perms]

    # Parameter grids (Loop 2)
    open_retrace_values = [0, 0.1, 0.2]
    cwc_retrace_values = [0.2, 0.3]
    rr_values = [1.0] #, 2.0, 3.0]

    # Collect scenario scores for permutation histogram
    real_combined_scores: list[float] = []
    permuted_combined_scores: list[float] = []

    for idx, scenario in enumerate(permutations):
        try:
            total_perms = len(permutations)
            label = "real" if idx == 0 else "perm"
            print(f"[PERM] Running permutation {idx + 1}/{total_perms} ({label})")
        except Exception:
            pass
        # Compute VWAP for this scenario
        vwap_all = IndicatorFormulas(scenario).vwap()

        # Loop 2: grid-search for best parameters, store PnL for each combo
        best_tuple = (None, None, None)

        if optimize:
            grid_results = ParameterOptimizer.backtest_open_cwc_rr(
            scenario,
                vwap_all,
                open_retrace_values=open_retrace_values,
                cwc_retrace_values=cwc_retrace_values,
                rr_values=rr_values,
                train_ratio=0.7,
            )

            # Pick best params by highest combined_total_pnl (fallback to total_pnl)
            best_score = -float('inf')
            for open_key, lvl1 in grid_results.items():
                for cwc_key, lvl2 in lvl1.items():
                    for rr_key, metrics in lvl2.items():
                        score = float(metrics.get('combined_total_pnl', metrics.get('total_pnl', 0.0)))
                        if score > best_score:
                            best_score = score
                            best_tuple = (float(open_key), float(cwc_key), float(rr_key))

            if all(v is not None for v in best_tuple):
                open_retrace_pct, cwc_retrace_pct, rr = best_tuple

        # Optional: print top results for this scenario
        try:
            ParameterOptimizer.print_results_top(grid_results, top_k=10)
        except Exception:
            pass
        
        # Loop 3: run and plot sessions day-by-day using the best params
        runner = NYVWAPSessionRunner(session_open_utc="16:00", session_hours=1.5, max_sessions=max_sessions)
        runner.run_and_plot_sessions_iterative(
            scenario,
            start_date=start_date,
            open_retrace=float(open_retrace_pct),
            cwc_retrace=float(cwc_retrace_pct),
            rr=float(rr),
            atr_multiplier=float(atr_multiplier),
            show_plot=show_plot
        )

        # Record a scalar score for histogram (use combined_total_pnl of the best tuple)
        if optimize:
            best_metrics = grid_results.get(str(open_retrace_pct), {}).get(str(cwc_retrace_pct), {}).get(str(rr), {})
            combined_best = float(best_metrics.get('combined_total_pnl', best_metrics.get('total_pnl', 0.0)))
            if idx == 0:
                real_combined_scores.append(combined_best)
            else:
                permuted_combined_scores.append(combined_best)

    # After all permutations, plot histogram of permutation distribution vs real
    if len(real_combined_scores) > 0 and len(permuted_combined_scores) > 0:
        ParameterOptimizer.plot_permutation_histogram(
            real_value=float(real_combined_scores[0]),
            permuted_values=permuted_combined_scores,
            bins=30,
            title='Permutation distribution vs real (combined total PnL)',
            xlabel='Combined total PnL',
            ylabel='Frequency',
        )
   
if __name__ == "__main__":
 
    filepath =  "Data/Futures/NQ Rithmic, Tick - Tick - Last, 1_1_2025 120000 AM-2_15_2025 120000 AM_fb31943b-fc76-46a7-8e7e-9ec372ee249d.csv"
    freq = "1min"  # Example frequency, can be 
    start = None # "2025-06-30"  # e.g., "2025-03-01" or "2025-03-01 09:30"; None starts from file begin
    optimize = True
    num_permutation = 1  # 1 -> real data; >1 -> number of randomized permutations
    permute_seed = 42

    main(filepath, freq, start=start, optimize=optimize, num_permutation=num_permutation, permute_seed=permute_seed, atr_multiplier=1.0)
