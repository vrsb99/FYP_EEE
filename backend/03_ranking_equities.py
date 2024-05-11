import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from datetime import datetime
from typing import Tuple
import riskfolio as rp
import quantstats as qs


def main() -> None:
    """
    Perform rebalancing on equity portfolios using trading factors and price data.
    This function determines the rebalancing dates and adjusts them to the nearest trading days.
    It filters returns and factors for each rebalance period and computes different types of Sharpe Ratios.

    Returns:
        None
    """

    three_factor_returns, five_factor_returns, rf_rates = get_factors_and_rf()
    equities_data = get_equities_data()

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

    for idx, date in enumerate(rebalance_dates[:-1]):
        start_date = str(date - pd.DateOffset(years=3))
        end_date = str(date)
        start_date_adjusted, end_date_adjusted = trading_days_start_end(
            start_date, end_date
        )

        (
            three_factors_filtered,
            five_factors_filtered,
            price,
        ) = filter_returns_and_factors(
            start_date_adjusted,
            end_date_adjusted,
            three_factor_returns,
            five_factor_returns,
            equities_data,
        )

        print(f"Computing rankings for {start_date_adjusted} to {end_date_adjusted}")
        (
            ordinary_ratio,
            three_factors_ratio,
            five_factors_ratio,
        ) = compute_ratio(
            start_date_adjusted,
            end_date_adjusted,
            three_factors_filtered,
            five_factors_filtered,
            price,
            rf_rates,
        )

        year = rebalance_dates[idx + 1].year
        quarter = rebalance_dates[idx + 1].quarter
        print(f"Saving rankings for {year} Q{quarter}")
        ordinary_ratio.to_parquet(
            f"../data/ranking_equities/ordinary/{year}_Q{quarter}.parquet"
        )
        three_factors_ratio.to_parquet(
            f"../data/ranking_equities/expected_three/{year}_Q{quarter}.parquet"
        )
        five_factors_ratio.to_parquet(
            f"../data/ranking_equities/expected_five/{year}_Q{quarter}.parquet"
        )


def process_factors(factors_df: pd.DataFrame) -> pd.DataFrame:
    """
    Process financial factor data by converting the index to datetime, computing daily returns,
    and handling infinite values.

    Args:
        factors_df (pd.DataFrame): DataFrame containing factor data including returns and possibly risk-free rates.

    Returns:
        pd.DataFrame: Processed DataFrame with daily returns of the factors.
    """

    factors_df.index = pd.to_datetime(factors_df.index)

    factor_returns = factors_df.iloc[:, :-1].pct_change()[1:]

    processed_returns = (
        factor_returns.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    )

    return processed_returns


def get_factors_and_rf() -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    """
    Load factor data and the risk-free rate from parquet files and process them for use.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.Series]: A tuple containing DataFrame for three factors returns,
        DataFrame for five factors returns, and Series for risk-free rates.
    """

    five_factors_and_rf = pd.read_parquet("../data/factors_data/five_factors.parquet")
    three_factors_and_rf = pd.read_parquet("../data/factors_data/three_factors.parquet")

    three_factor_returns = process_factors(three_factors_and_rf)
    five_factor_returns = process_factors(five_factors_and_rf)

    rf_rates = five_factors_and_rf.iloc[:, -1]

    return three_factor_returns, five_factor_returns, rf_rates


def get_equities_data() -> pd.DataFrame:
    """
    Load equities data from parquet files, filtering and organizing them into a single DataFrame.

    Returns:
        pd.DataFrame: DataFrame with equities data indexed by date.
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
    equities_data.index = pd.to_datetime(equities_data.index)

    return equities_data


def filter_returns_and_factors(
    start_date: datetime,
    end_date: datetime,
    three_factors_data: pd.DataFrame,
    five_factors_data: pd.DataFrame,
    equities_data: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Filter the returns and factor data within a specified date range and based on specific conditions such as non-NA values.

    Args:
        start_date (datetime): The starting date of the period.
        end_date (datetime): The ending date of the period.
        three_factors_data (pd.DataFrame): DataFrame containing three factors data.
        five_factors_data (pd.DataFrame): DataFrame containing five factors data.
        equities_data (pd.DataFrame): DataFrame containing equities data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing filtered three factors data,
        filtered five factors data, and filtered equities returns.
    """

    # 1. Filter out dates for equities data
    start_pd = pd.to_datetime(start_date)
    end_pd = pd.to_datetime(end_date)

    equities_data_filtered = equities_data.loc[start_pd:end_pd]
    # 2. Filter out equities with NA values
    equities_data_filtered = equities_data_filtered.dropna(axis=1, how="any")
    # 3. Filter out equities with a mean return of < 0.2
    equities_data_filtered = equities_data_filtered.loc[
        :, (equities_data_filtered.mean() > 0.2)
    ]

    # Calculate returns
    returns = (
        np.log(equities_data_filtered / equities_data_filtered.shift(1)).dropna().copy()
    )
    returns = returns[np.isfinite(returns)].dropna()
    returns = returns.loc[:, (returns.mean() != 0.0)]

    # 4. Ensure that five factors and equities data have the same dates
    three_factors_filtered = three_factors_data[start_pd:end_pd].reindex(returns.index)
    five_factors_filtered = five_factors_data[start_pd:end_pd].reindex(returns.index)

    return three_factors_filtered, five_factors_filtered, returns


def trading_days_start_end(
    start_date: str, end_date: str, exchange: str = "XSES"
) -> Tuple[datetime, datetime]:
    """
    Determine the start and end trading days within a given date range for a specified exchange.

    Args:
        start_date (str): The starting date string of the period.
        end_date (str): The ending date string of the period.
        exchange (str): The stock exchange code.

    Returns:
        Tuple[datetime, datetime]: A tuple containing the actual start and end trading days as datetime objects.
    """

    cal = mcal.get_calendar(exchange)
    schedule = cal.schedule(start_date=start_date, end_date=end_date)
    return schedule.iloc[0].market_open.date(), schedule.iloc[-1].market_close.date()


def compute_ratio(
    start_date: datetime,
    end_date: datatime,
    three_factors_data: pd.DataFrame,
    five_factors_data: pd.DataFrame,
    price: pd.DataFrame,
    rf_rates: pd.Series,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute historical and expected Sharpe ratios for equities based on specified factors over a given period.

    Args:
        start_date (str): The starting date string of the period.
        end_date (str): The ending date string of the period.
        three_factors_data (pd.DataFrame): DataFrame containing filtered data for three factors.
        five_factors_data (pd.DataFrame): DataFrame containing filtered data for five factors.
        price (pd.DataFrame): DataFrame containing price data of equities.
        rf_rates (pd.Series): Series containing the risk-free rates over the period.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing DataFrames for historical Sharpe ratios,
        Sharpe ratios based on three factors, and Sharpe ratios based on five factors.
    """

    periods = 63

    # Compute risk-free rate
    rf_rate = round(rf_rates.loc[start_date:end_date].mean(), 4)
    print(f"Risk-free rate: {rf_rate}")

    # Values used for expected sharpe ratio
    returns = qs.stats._utils._prepare_returns(price, rf_rate, periods)
    divisor = returns.std(ddof=1)
    divisor = divisor * qs.stats.autocorr_penalty(returns)

    mu_three_fm, cov_three_fm, ret_three_est = compute_risk_factors(
        returns, three_factors_data
    )
    mu_five_fm, cov_fm, ret_five_est = compute_risk_factors(returns, five_factors_data)

    # Compute historical Sharpe Ratios
    sharpe_ratios_hist = returns.mean() / divisor
    sharpe_ratios_hist = sharpe_ratios_hist * np.sqrt(periods)

    sharpe_ratios_hist = pd.DataFrame(sharpe_ratios_hist, columns=["Sharpe Ratio"])
    sharpe_ratios_hist = sharpe_ratios_hist.sort_values(
        by="Sharpe Ratio", ascending=False
    )

    # Prepare returns for expected Sharpe Ratios
    returns_three_est = qs.stats._utils._prepare_returns(
        ret_three_est, rf_rate, periods
    )
    returns_five_est = qs.stats._utils._prepare_returns(ret_five_est, rf_rate, periods)

    # Compute expected Sharpe Ratios
    sharpe_ratios_three_factors = returns_three_est.mean() / divisor
    sharpe_ratios_three_factors = sharpe_ratios_three_factors * np.sqrt(periods)

    sharpe_ratios_five_factors = returns_five_est.mean() / divisor
    sharpe_ratios_five_factors = sharpe_ratios_five_factors * np.sqrt(periods)

    sharpe_ratios_three_factors = pd.DataFrame(
        sharpe_ratios_three_factors, columns=["Sharpe Ratio"]
    )
    sharpe_ratios_three_factors = sharpe_ratios_three_factors.sort_values(
        by="Sharpe Ratio", ascending=False
    )

    sharpe_ratios_five_factors = pd.DataFrame(
        sharpe_ratios_five_factors, columns=["Sharpe Ratio"]
    )
    sharpe_ratios_five_factors = sharpe_ratios_five_factors.sort_values(
        by="Sharpe Ratio", ascending=False
    )

    return sharpe_ratios_hist, sharpe_ratios_three_factors, sharpe_ratios_five_factors


def compute_risk_factors(
    returns: pd.DataFrame, factors_data: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Compute mu (expected returns), covariance, and the adjusted returns using PCA on the given returns and factor data.

    Args:
        returns (pd.DataFrame): DataFrame containing returns data.
        factors_data (pd.DataFrame): DataFrame containing factor data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing DataFrames for mu, covariance, and adjusted returns.
    """

    returns.index = pd.to_datetime(returns.index)
    factors_data.index = pd.to_datetime(factors_data.index)

    X = factors_data.resample("D").last()
    X = X[~np.isinf(X)].dropna()

    Y = returns.resample("D").last()
    Y = Y[~np.isinf(Y)].dropna()

    idx_common = X.index.intersection(Y.index)

    X = X.loc[idx_common]
    Y = Y.loc[idx_common]

    # print(f"Common dates: {len(X)}")
    comp = X.shape[1]
    mu, cov, ret, _ = rp.risk_factors(X, Y, feature_selection="PCR", n_components=comp)

    return mu, cov, ret


if __name__ == "__main__":
    main()
