import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from typing import Tuple
from portfolio import compute_portfolio, compute_acc_returns
import os
import concurrent.futures
import warnings
from risk_measures import rm_names, MODELS
import logging

# logging.basicConfig(filename='../logs/test_models.txt', level=logging.INFO, format='%(asctime)s %(levelname)s:%(message)s', filemode='w')

DEBUG = False

if DEBUG:
    MODELS = [
        "Black-Litterman",
    ]

    rm_names = rm_names


def process_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the index of the DataFrame to datetime, calculate the percent change of returns, and handle infinities.

    Args:
        factors_df (pd.DataFrame): DataFrame containing factor data with risk-free rates as the last column.

    Returns:
        pd.DataFrame: DataFrame with processed returns excluding the last column which contains risk-free rates.
    """

    factors_df.index = pd.to_datetime(factors_df.index)

    factor_returns = factors_df.iloc[:, :-1].pct_change()[1:]

    processed_returns = (
        factor_returns.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    )

    return processed_returns


def get_factors_and_rf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Retrieve and process factor data along with risk-free rates from stored parquet files.

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
    Load equities data from stored parquet files, handling errors and filling forward missing values.

    Returns:
        pd.DataFrame: DataFrame with equities data indexed by date and including a special handling for "^STI".
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
    Load asset class information from a parquet file which includes equities and their associated sectors.

    Returns:
        pd.DataFrame: DataFrame containing asset class information for equities.
    """

    asset_classes = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet", columns=["Code", "Sector"]
    )
    return asset_classes


def filter_returns_and_factors(
    start_date: datetime,
    end_date: datetime,
    start_date_oos: datetime,
    end_date_oos: datetime,
    three_factors: pd.DataFrame,
    five_factors: pd.DataFrame,
    equities_data: pd.DataFrame,
    mean_return_threshold: float = 0.2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter equity and factor data for in-sample and out-of-sample analysis based on provided dates and return thresholds.

    Args:
        start_date (datetime): Start date for the in-sample period.
        end_date (datetime): End date for the in-sample period.
        start_date_oos (datetime): Start date for the out-of-sample period.
        end_date_oos (datetime): End date for the out-of-sample period.
        three_factors (pd.DataFrame): DataFrame of three-factor data.
        five_factors (pd.DataFrame): DataFrame of five-factor data.
        equities_data (pd.DataFrame): DataFrame of equities data.
        mean_return_threshold (float): Threshold for filtering equities based on their mean returns.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: Filtered three-factor data, five-factor data,
        returns for in-sample, and returns for out-of-sample periods respectively.
    """

    # Convert dates to pandas datetime for consistency
    start_pd = pd.to_datetime(start_date)
    end_pd = pd.to_datetime(end_date)
    start_date_oos_pd = pd.to_datetime(start_date_oos)
    end_date_oos_pd = pd.to_datetime(end_date_oos)

    # 1. Filter equities data for the broader date range
    equities_data_filtered = equities_data.loc[start_pd:end_date_oos_pd]

    # 2. Drop columns (equities) with any NA values
    equities_data_filtered = equities_data_filtered.ffill().dropna(axis=1, how="any")

    # 3. Filter equities based on mean return from start_pd to end_pd
    mean_returns = equities_data_filtered.loc[start_pd:end_pd].mean()
    equities_to_keep = mean_returns[mean_returns > mean_return_threshold].index
    equities_data_filtered = equities_data_filtered[equities_to_keep]

    # Calculate returns
    returns_all = (
        np.log(equities_data_filtered / equities_data_filtered.shift(1)).dropna().copy()
    )
    returns_all = returns_all[np.isfinite(returns_all)].dropna()

    # Separate returns into in-sample and out-of-sample periods
    returns = returns_all.loc[start_pd:end_pd]
    returns = returns.loc[:, (returns.mean() != 0.0)]
    returns_oos = returns_all.loc[start_date_oos_pd:end_date_oos_pd]
    returns_oos = returns_oos.reindex(returns.columns, axis=1)

    # 4. Ensure three, five factors have the same dates as the returns
    three_factors_filtered = three_factors.loc[start_pd:end_pd].reindex(returns.index)
    five_factors_filtered = five_factors.loc[start_pd:end_pd].reindex(returns.index)

    return three_factors_filtered, five_factors_filtered, returns, returns_oos


def main():
    """
    The main execution function that orchestrates the entire process from data retrieval to model computation across specified rebalance dates.

    Returns:
        None
    """

    three_factor_returns, five_factor_returns, rf_rates = get_factors_and_rf()
    equities_data = get_equities_data()
    asset_classes = get_asset_classes()

    rebalance_dates = pd.date_range(start="2002-12-31", end="2024-03-01", freq="QS")

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

    rebalance_dates = rebalance_dates[rebalance_dates.isin(equities_data.index)]
    futures = []

    with concurrent.futures.ProcessPoolExecutor() as executor:
        for idx, date in enumerate(rebalance_dates[:-1]):
            start_date = str(date - pd.DateOffset(years=3))
            end_date = str(date)
            start_date_adjusted, end_date_adjusted = trading_days_start_end(
                start_date, end_date
            )

            start_date_oos_dt = date + pd.DateOffset(days=1)
            start_date_oos = str(start_date_oos_dt)
            end_date_oos = (
                str(rebalance_dates[idx + 1] - pd.DateOffset(days=1))
                if idx + 1 < len(rebalance_dates)
                else str(date + pd.DateOffset(months=3))
            )
            start_date_adjusted_oos, end_date_adjusted_oos = trading_days_start_end(
                start_date_oos, end_date_oos
            )

            print(
                f"Computing model parameters for {start_date_adjusted} to {end_date_adjusted}"
            )

            rf_rate = round(
                rf_rates.loc[start_date_adjusted:end_date_adjusted].mean(), 4
            )
            print(f"Risk free rate: {rf_rate}")

            (
                three_factors_filtered,
                five_factors_filtered,
                returns,
                returns_oos,
            ) = filter_returns_and_factors(
                start_date_adjusted,
                end_date_adjusted,
                start_date_adjusted_oos,
                end_date_adjusted_oos,
                three_factor_returns,
                five_factor_returns,
                equities_data,
            )

            asset_classes_modified = asset_classes[
                asset_classes["Code"].isin(returns.columns)
            ]

            if not DEBUG:
                MODELS.append("Frontier")

            for model in MODELS:
                if model in ["Monte Carlo", "Frontier"]:
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

                year = start_date_oos_dt.year
                month = start_date_oos_dt.month
                quarter = (month - 1) // 3 + 1

                factor_returns = None
                if "3-Factors" in model:
                    factor_returns = three_factors_filtered
                elif "5-Factors" in model:
                    factor_returns = five_factors_filtered

                for risk_measure in risk_measures:

                    if (
                        not os.path.exists(
                            f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
                        )
                        or DEBUG
                    ):
                        future = executor.submit(
                            compute_model_parameters,
                            returns,
                            returns_oos,
                            factor_returns,
                            rf_rate,
                            asset_classes_modified,
                            model,
                            risk_measure,
                            start_date_oos_dt,
                        )
                        futures.append(future)
                        # compute_model_parameters(
                        #     returns,
                        #     returns_oos,
                        #     factor_returns,
                        #     rf_rate,
                        #     asset_classes_modified,
                        #     model,
                        #     risk_measure,
                        #     start_date_oos_dt,
                        # )
                    else:
                        df = pd.read_parquet(
                            f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
                        )
                        contains_na = df.isna().any().any()
                        # Contains only 1 weight

                        if contains_na or df.shape[0] == 1:
                            print(df)
                            print(
                                f"Recomputing {model}_{risk_measure}_{year}_Q{quarter}"
                            )
                            future = executor.submit(
                                compute_model_parameters,
                                returns,
                                returns_oos,
                                factor_returns,
                                rf_rate,
                                asset_classes_modified,
                                model,
                                risk_measure,
                                start_date_oos_dt,
                            )
                            futures.append(future)

                PATH = (
                    f"../data/studying_models/input_returns/{year}_Q{quarter}.parquet"
                )
                if not os.path.exists(PATH):
                    returns.to_parquet(
                        f"../data/studying_models/input_returns/{year}_Q{quarter}.parquet"
                    )

    concurrent.futures.wait(futures)

    for future in futures:
        if future.exception() is not None:
            logging.error(future.exception())
            print(future.exception())


def compute_model_parameters(
    quarterly_returns_data: pd.DataFrame,
    oos_returns_data: pd.DataFrame,
    factor_returns: pd.DataFrame,
    rf_rate: float,
    asset_classes: pd.DataFrame,
    model: str,
    risk_measure: str,
    date: datetime,
) -> None:
    """
    Perform portfolio optimization and risk management calculations for specified asset classes over a given period.
    This function logs and saves optimized portfolio weights and optionally computes out-of-sample returns.

    Args:
        quarterly_returns_data (pd.DataFrame): DataFrame containing quarterly returns data.
        oos_returns_data (pd.DataFrame): DataFrame containing out-of-sample returns data.
        factor_returns (pd.DataFrame): DataFrame containing factor returns relevant to the model.
        rf_rate (float): The risk-free rate for the period.
        asset_classes (pd.DataFrame): DataFrame containing asset class information for filtering and grouping.
        model (str): The portfolio model to be used for optimization.
        risk_measure (str): The risk measure to be applied within the model.
        date (datetime): The starting date for the period considered in the calculations.

    Returns:
        None
    """

    print(f"Optimization method: {model} - Risk measure: {risk_measure}")
    year = date.year
    month = date.month
    quarter = (month - 1) // 3 + 1
    try:
        # logging.info(f"{year} - Q{quarter}")
        # logging.info(len(quarterly_returns_data.columns))
        # logging.info(f"Start date: {quarterly_returns_data.index[0]} - End date: {quarterly_returns_data.index[-1]}")

        w = compute_portfolio(
            returns=quarterly_returns_data,
            selected_portfolio=model,
            risk_measure=risk_measure,
            asset_classes=asset_classes,
            top=30,
            factor_returns=factor_returns,
            rf_rate=rf_rate,
        )

        if model != "Frontier" and oos_returns_data is not None:
            acc_ret = compute_acc_returns(oos_returns_data[w.index], w)
            strat_ret = acc_ret.pct_change().dropna()
            strat_ret = pd.DataFrame(strat_ret, columns=["returns"])

            strat_ret.to_parquet(
                f"../data/studying_models/output_returns/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
            )

        contains_na = w.isna().any().any()
        if contains_na:
            print(w)
            raise ValueError(f"Contains NA {model}_{risk_measure}_{year}_Q{quarter}")

        w.to_parquet(
            f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
        )

        print(f"Saved {model}_{risk_measure}_{year}_Q{quarter}")
    except Exception as e:
        print(f"Error in {model}_{risk_measure}_{year}: {e}")
        raise e


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
    main()
