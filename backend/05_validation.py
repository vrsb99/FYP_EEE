import warnings
import os
import traceback
from datetime import datetime
from typing import Tuple
import argparse
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from portfolio import compute_portfolio
from vectorbt.portfolio.enums import SizeType, Direction
from vectorbt.portfolio.nb import order_nb, sort_call_seq_nb
import vectorbt as vbt
import concurrent.futures
from risk_measures import rm_names, MODELS
import logging

logging.basicConfig(
    filename="../logs/valildation.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    filemode="w",
)

DEBUG = True

if DEBUG:
    MODELS = [
        # "Classic",
        # "Classic 3-Factors",
        # "Classic 5-Factors",
        # "Risk Parity",
        "Risk Parity 3-Factors",
        # "Risk Parity 5-Factors",
        # "Black-Litterman",
        # "Black-Litterman 3-Factors",
        # "Black-Litterman 5-Factors",
        # "Hierarchical Risk Parity",
        # "Monte Carlo",
    ]

    rm_names = rm_names


def main(
    start_date: str, end_date: str, lookback_period: int, backtest_years: int
) -> None:
    """
    Perform backtesting for different financial models over a set period with rebalancing at each quarter start.
    This function configures the rebalance dates to align with trading days and processes futures for each model's backtest.

    Args:
        start_date (str): The start date for the backtesting period in 'YYYY-MM-DD' format.
        end_date (str): The end date for the backtesting period in 'YYYY-MM-DD' format.
        lookback_period (int): Number of years to look back for calculating returns.
        backtest_years (int): Number of years over which the backtesting is performed.

    Returns:
        None
    """

    three_factor_returns, five_factor_returns, rf_rates = get_factors_and_rf()
    data = get_equities_data()
    asset_classes = get_asset_classes()

    rebalance_dates = pd.date_range(start=start_date, end=end_date, freq="QS")

    adjusted_rebalance_dates = []
    length = len(rebalance_dates)
    for idx, date in enumerate(rebalance_dates):
        if idx < length - 1:
            start_of_period, _ = trading_days_start_end(
                str(date), str(rebalance_dates[idx + 1])
            )
        else:
            _, start_of_period = trading_days_start_end(
                str(rebalance_dates[idx - 1]), str(date)
            )
        adjusted_rebalance_dates.append(start_of_period)

    rebalance_dates = pd.to_datetime(adjusted_rebalance_dates)

    rebalance_dates = rebalance_dates[rebalance_dates.isin(data.index)]
    rebalance_indices = [data.index.get_loc(date) for date in rebalance_dates]

    futures = []
    with concurrent.futures.ProcessPoolExecutor() as executor:
        for model in MODELS:
            if model == "Monte Carlo":
                risk_measures = ["Standard Deviation"]
            elif "Risk Parity" in model and "Hierarchical" not in model:
                excluded = [
                    "Worst Realization",
                    "Average Drawdown",
                    "Max Drawdown",
                    "Range",
                ]
                risk_measures = [risk for risk in rm_names if risk not in excluded]
            else:
                risk_measures = rm_names

            factor_returns = None
            if "3-Factors" in model:
                factor_returns = three_factor_returns
            elif "5-Factors" in model:
                factor_returns = five_factor_returns

            for rm in risk_measures:
                if (
                    not os.path.exists(
                        f"../data/backtest/redo/{model}_{rm}_B{lookback_period}_Y{backtest_years}.pkl"
                    )
                    or DEBUG
                ):
                    futures.append(
                        executor.submit(
                            compute_model,
                            data,
                            rebalance_indices,
                            rm,
                            lookback_period,
                            asset_classes,
                            model,
                            factor_returns,
                            rf_rates,
                            backtest_years,
                        )
                    )
                else:
                    print(f"Backtest already exists for {model}, {rm}")

    concurrent.futures.wait(futures)


def compute_model(
    data: pd.DataFrame,
    rebalance_indices: np.ndarray,
    rm: str,
    lookback_period: int,
    asset_classes: pd.DataFrame,
    model: str,
    factor_returns: pd.DataFrame,
    rf_rates: pd.DataFrame,
    backtest_years: int,
) -> None:
    """
    Execute backtesting on financial models using given parameters and save the results. Handles various portfolio configurations
    and adjusts for trading specifics like fees and slippage.

    Args:
        data (pd.DataFrame): The DataFrame containing historical stock prices.
        rebalance_indices (np.ndarray): Array of indices where rebalancing should occur.
        rm (str): The risk measure used in the model.
        lookback_period (int): The period (in years) used to calculate backtest data.
        asset_classes (pd.DataFrame): DataFrame containing asset class information.
        model (str): The financial model to use for portfolio optimization.
        factor_returns (pd.DataFrame): DataFrame containing factor returns used in the model.
        rf_rates (pd.DataFrame): DataFrame containing risk-free rates.
        backtest_years (int): The number of years over which the backtest is performed.

    Returns:
        None
    """

    # Parameters
    vbt.settings.returns["year_freq"] = "252 days"
    vbt.settings.portfolio["fees"] = 0.001  # 0.1%
    vbt.settings.portfolio["slippage"] = 0.005  # 0.5%
    assets = data.columns.tolist()
    dates = data.index

    print(f"Starting backtest for {model}, {rm}")
    try:
        backtest_data = vbt.Portfolio.from_order_func(
            data,
            order_func_nb,
            pre_sim_func_nb=pre_sim_func_nb,
            pre_sim_args=(rebalance_indices,),
            pre_segment_func_nb=pre_segment_func_nb,
            pre_segment_args=(
                opt_weights,
                rm,
                lookback_period,
                assets,
                dates,
                asset_classes,
                model,
                30,
                factor_returns,
                rf_rates,
            ),
            cash_sharing=True,
            group_by=True,
            use_numba=False,
            freq="D",
            init_cash=1e5,
        )

        backtest_data.save(
            f"../data/backtest/redo/{model}_{rm}_B{lookback_period}_Y{backtest_years}.pkl"
        )
        print(f"Backtest complete for {model}, {rm}")

    except Exception as e:
        print(f"Backtest not complete for {model}, {rm}: {e}")
        traceback.print_exc()


# order_func_nb: Function defining the order behavior in the backtest
def order_func_nb(oc: vbt.agg.OrderContext, weights: np.ndarray) -> vbt.agg.Order:
    """
    A numba-compiled function defining the order behavior during the backtesting simulation based on the weights.

    Args:
        oc (OrderContext): The context providing access to various data during the backtest.
        weights (np.ndarray): Array of target weights for the portfolio assets.

    Returns:
        Order (Generated by order_nb function): An order object configured to adjust portfolio to target weights.
    """

    col_i = oc.call_seq_now[oc.call_idx]
    return order_nb(
        weights[col_i],
        oc.close[oc.i, oc.col],
        size_type=SizeType.TargetPercent,
    )


# pre_sim_func_nb: Function to define rebalancing days
def pre_sim_func_nb(
    sc: vbt.agg.SimulationContext, rebalance_indices: np.ndarray
) -> Tuple:
    """
    A numba-compiled function to define the rebalancing days in the simulation, marking the segments where trades should occur.

    Args:
        sc (SimulationContext): The simulation context holding the state throughout the backtest.
        rebalance_indices (np.ndarray): Indices where rebalancing should occur within the simulation period.

    Returns:
        Tuple: Returns an empty tuple after updating the segment mask in the simulation context.
    """

    sc.segment_mask[:, :] = False

    length = len(sc.segment_mask)
    for i in range(length):
        if i in rebalance_indices:
            sc.segment_mask[i, :] = True

    return ()


# pre_segment_func_nb: Function to prepare each segment of the simulation
def pre_segment_func_nb(
    sc,
    find_weights_nb,
    rm,
    lookback_period,
    assets,
    dates,
    asset_classes,
    model,
    top: int = 30,
    factor_returns: pd.DataFrame = None,
    rf_rates: pd.DataFrame = None,
):
    """
    A function to prepare each segment of the simulation, computing the optimal weights for the portfolio.

    Args:
        sc (SegmentContext): The segment context containing data for a specific part of the backtest.
        find_weights_nb (callable): Function to calculate optimal weights.
        rm (str): The risk measure used.
        lookback_period (int): The lookback period for the data.
        assets (np.ndarray): Array of asset identifiers.
        dates (pd.DatetimeIndex): Index of dates for the data.
        asset_classes (pd.DataFrame): DataFrame with asset class information.
        model (str): The financial model to use for optimization.
        top (int): The number of top assets to include.
        factor_returns (pd.DataFrame): DataFrame containing factor returns.
        rf_rates (pd.DataFrame): DataFrame containing risk-free rates.

    Returns:
        Tuple (Weights and other order parameters): Configures weights and order parameters for the trading segment.
    """

    current_date = dates[sc.i]
    historical_start_date = current_date - pd.DateOffset(years=lookback_period)

    start_date_adj, _ = trading_days_start_end(
        str(historical_start_date), str(current_date)
    )
    try:
        start_date_adj = pd.to_datetime(start_date_adj)
        start_idx = dates.get_loc(start_date_adj)
    except KeyError:
        start_idx = 0

        return (np.full(sc.group_len, np.nan),)

    end_idx = dates.get_loc(current_date)

    close_df = sc.close[start_idx : end_idx + 1, :]
    # print(f"Start date: {dates[start_idx]} and end date: {dates[end_idx]}")

    # Filter out the dates of start and end
    close_df = pd.DataFrame(
        close_df, columns=assets, index=dates[start_idx : end_idx + 1]
    )

    factor_returns = pd.DataFrame(
        factor_returns, index=dates[start_idx + 1 : end_idx + 1]
    )

    factor_returns = factor_returns.ffill()

    close_df = close_df.ffill().dropna(axis=1, how="any")

    # Filter out stocks with low prices to avoid penny stocks
    close_df = close_df.loc[:, (close_df.mean() > 0.2)]

    # Find optimal weights
    weights = find_weights_nb(
        rm,
        close_df,
        assets,
        asset_classes,
        model,
        top,
        factor_returns,
        rf_rates,
    )

    weights = np.asarray(weights, dtype=np.float64).flatten()

    # Update valuation price and reorder orders
    size_type = np.full(sc.group_len, SizeType.TargetPercent, dtype=np.int_)
    direction = np.full(sc.group_len, Direction.LongOnly, dtype=np.int_)
    temp_float_arr = np.empty(sc.group_len, dtype=np.float_)

    for k in range(sc.group_len):
        col = sc.from_col + k
        sc.last_val_price[col] = sc.close[sc.i, col]

    sort_call_seq_nb(sc, weights, size_type, direction, temp_float_arr)

    return (weights,)


# opt_weights: Function to compute optimal weights for portfolio
def opt_weights(
    rm: str,
    close_df: pd.DataFrame,
    assets: np.ndarray,
    asset_classes: pd.DataFrame,
    model: str,
    top: int = 30,
    factor_returns: pd.DataFrame = None,
    rf_rates: pd.DataFrame = None,
) -> np.ndarray:
    """
    Compute optimal portfolio weights based on a specified risk measure, financial model, and additional parameters.

    Args:
        rm (str): The risk measure to apply.
        close_df (pd.DataFrame): DataFrame of closing prices.
        assets (np.ndarray): Array of asset identifiers.
        asset_classes (pd.DataFrame): DataFrame with asset class information.
        model (str): Financial model to determine the weighting strategy.
        top (int): Number of top assets to consider for the portfolio.
        factor_returns (pd.DataFrame): Factor returns for the model.
        rf_rates (pd.DataFrame): Risk-free rates during the period.

    Returns:
        np.ndarray: Array of calculated weights for the portfolio.
    """

    returns = np.log(close_df / close_df.shift(1)).dropna().copy()
    returns = returns[np.isfinite(returns)].dropna()
    returns = returns.loc[:, (returns.mean() != 0.0)]

    relevant_rf_rates = rf_rates.loc[returns.index]
    avg_rf_rate = round(relevant_rf_rates.mean(), 4)

    asset_classes = asset_classes[asset_classes["Code"].isin(returns.columns)]

    w = compute_portfolio(
        returns=returns,
        selected_portfolio=model,
        risk_measure=rm,
        asset_classes=asset_classes,
        top=top,
        factor_returns=factor_returns,
        rf_rate=avg_rf_rate,
    )

    w_new = pd.DataFrame(0, columns=["weights"], index=assets).astype(float)
    w_new.loc[w.index] = w.values
    w = w_new.copy()

    weights = np.ravel(w.to_numpy())
    return weights


def process_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Processes factor data to compute the percent change of returns, converting the index to datetime and handling infinities.

    Args:
        factors_df (pd.DataFrame): DataFrame containing factor data with the last column typically being the risk-free rate.

    Returns:
        pd.DataFrame: Processed factor returns with infinities and NaN values handled, indexed by datetime.
    """

    factors_df.index = pd.to_datetime(factors_df.index)

    factor_returns = factors_df.iloc[:, :-1].pct_change()[1:]

    processed_returns = (
        factor_returns.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    )

    return processed_returns


def get_factors_and_rf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Retrieves and processes factor data along with risk-free rates from stored parquet files.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: Three-factor returns, five-factor returns, and risk-free rates respectively.
    """

    five_factors_and_rf = pd.read_parquet("../data/factors_data/five_factors.parquet")
    three_factors_and_rf = pd.read_parquet("../data/factors_data/three_factors.parquet")

    three_factor_returns = process_factors(three_factors_and_rf)
    five_factor_returns = process_factors(five_factors_and_rf)

    rf_rates = five_factors_and_rf.iloc[:, -1]

    return three_factor_returns, five_factor_returns, rf_rates


def get_equities_data() -> pd.DataFrame:
    """
    Loads and processes equities data from parquet files, handling any potential errors and forward filling missing "^STI" data.

    Returns:
        pd.DataFrame: DataFrame containing equities data indexed by date, with columns named after each equity's code.
    """

    equities = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet", columns=["Code"]
    )

    data = {}
    for code in equities["Code"]:
        try:
            file_name = f"stock_{code[:3]}" if code != "^STI" else "STI"
            df = pd.read_parquet(
                f"../data/obtain_data/{file_name}.parquet", columns=["Date", "Close"]
            )
            df.rename(columns={"Close": code}, inplace=True)
            data[code] = df
        except Exception as e:
            print(f"Error in {code}: {e}")

    equities_data = pd.concat(data.values(), axis=1)
    equities_data["^STI"] = equities_data["^STI"].ffill()
    equities_data.index = pd.to_datetime(equities_data.index)
    return equities_data


def get_asset_classes() -> pd.DataFrame:
    """
    Retrieves asset class information from a parquet file, which includes equity codes and their corresponding sectors.

    Returns:
        pd.DataFrame: DataFrame containing the codes and sectors of assets.
    """

    asset_classes = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet", columns=["Code", "Sector"]
    )
    return asset_classes


def trading_days_start_end(
    start_date: str, end_date: str, exchange: str = "XSES"
) -> Tuple[datetime, datetime]:
    """
    Determines the start and end trading days within a specified date range for a given stock exchange.

    Args:
        start_date (str): The starting date of the period in 'YYYY-MM-DD' format.
        end_date (str): The ending date of the period in 'YYYY-MM-DD' format.
        exchange (str): Stock exchange identifier, e.g., 'XSES' for the Singapore Stock Exchange.

    Returns:
        Tuple[datetime, datetime]: Start and end trading dates as datetime objects.
    """

    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return schedule.iloc[0].market_open.date(), schedule.iloc[-1].market_close.date()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process quarterly data.")
    parser.add_argument(
        "--lookback_period",
        "-l",
        type=int,
        choices=[1, 2, 3, 4, 5, 6],
        default=3,
        help="The backtesting number of years (default: %(default)s)",
    )
    parser.add_argument(
        "--backtest_years",
        "-b",
        type=int,
        choices=[21, 15, 10],
        default=21,
        help="The number of years to backtest (default: %(default)s)",
    )
    args = parser.parse_args()
    lookback_period = args.lookback_period
    backtest_years = args.backtest_years

    fixed_current_year = 2024
    end_date = "2024-01-01"

    if backtest_years == 21:
        start_year = fixed_current_year - backtest_years
        start_date = f"{start_year}-01-01"
        main(start_date, end_date, lookback_period, backtest_years)
    elif backtest_years == 15:
        start_year = fixed_current_year - backtest_years
        start_date = f"{start_year}-01-01"
        main(start_date, end_date, lookback_period, backtest_years)
    elif backtest_years == 10:
        start_year = fixed_current_year - backtest_years
        start_date = f"{start_year}-01-01"
        main(start_date, end_date, lookback_period, backtest_years)
