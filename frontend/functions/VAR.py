import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List
import pickle
from scipy.stats import norm
import pandas_market_calendars as mcal
from functions.database import get_ticker_data, filter_tickers

# For 1 to 3 year historic VAR:
# Use the mean_return, and portfolio_volatility from the weights of the optimal portfolio based on 1, 2, or 3 year historic data.


def main(
    run_id: int,
    start_date: str,
    portfolio_value: int,
    optimal_weights: np.ndarray,
    confidence_levels: List[float] = [0.95, 0.97, 0.99],
    days_to_simulate: int = 63,
) -> Tuple[
    List[float], List[float], List[float], List[float], List[float], List[float]
]:
    """
    1. Downloads historical stock data for the tickers in the portfolio.
    2. Calculates the log returns, mean returns, and volatility for the portfolio.
    3. Calculates Value at Risk (VaR) using Monte Carlo simulation, historical data, and parametric method.
    """
    # 1.
    tickers = get_ticker_data()

    filtered_tickers, filtered_weights = filter_tickers(tickers, optimal_weights)
    data = download_data()
    print(filtered_tickers)
    print(filtered_weights)
    print(data)
    print(filtered_weights.shape)
    print(data.shape)
    # 2.
    log_returns = calculate_returns(data)
    yearly_mean_returns, yearly_volatilities = calculate_returns_volatility(
        log_returns, filtered_weights, start_date
    )
    # 3.
    monte_carlo_var, simulated_portfolio_val = monte_carlo_method(
        portfolio_value,
        yearly_mean_returns,
        yearly_volatilities,
        confidence_levels,
        days_to_simulate=days_to_simulate,
    )
    historical_var_1yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=365,
        days_to_simulate=days_to_simulate,
    )
    historical_var_2yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=730,
        days_to_simulate=days_to_simulate,
    )
    historical_var_3yr = calculate_historical_var(
        portfolio_value,
        confidence_levels,
        log_returns,
        filtered_weights,
        start_date,
        days=1095,
        days_to_simulate=days_to_simulate,
    )
    parametric_var = calculate_parametric_var(
        portfolio_value,
        confidence_levels,
        yearly_mean_returns,
        yearly_volatilities,
        days_to_simulate=days_to_simulate,
    )

    return (
        monte_carlo_var,
        historical_var_1yr,
        historical_var_2yr,
        historical_var_3yr,
        parametric_var,
        simulated_portfolio_val,
    )


def download_data() -> pd.DataFrame:

    with open("../data/old_data/all_data.pkl", "rb") as f:
        stock_data = pickle.load(f)

    return pd.DataFrame(stock_data)


def calculate_returns(data: pd.DataFrame) -> pd.DataFrame:
    return np.log(data / data.shift(1)).dropna()


# Calculate portfolio statistics for multiple periods
def calculate_returns_volatility(
    log_returns: pd.DataFrame, weights: np.ndarray, start_date: str, days: int = 365
) -> Tuple[float, float]:
    new_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=days)
    trading_days = trading_days_between_dates(new_start_date, start_date)
    filtered_returns = log_returns.tail(trading_days)
    mean_return = np.dot(weights, filtered_returns.mean())
    volatility = np.sqrt(np.dot(weights.T, np.dot(filtered_returns.cov(), weights)))
    print(mean_return, volatility)
    return mean_return, volatility


# Calculate Parametric VaR for multiple periods and confidence levels
def calculate_parametric_var(
    portfolio_value: int,
    confidence_levels: List[float],
    mean_returns: float,
    volatilities: float,
    days_to_simulate: int = 63,
) -> List[List[float]]:
    var_results = []
    for conf in confidence_levels:
        var = portfolio_value * (
            mean_returns * days_to_simulate
            - volatilities * np.sqrt(days_to_simulate) * norm.ppf(1 - conf)
        )
        var_results.append(var)
    return var_results


# Calculate VaR for multiple periods and confidence levels
def calculate_historical_var(
    portfolio_value: int,
    confidence_levels: List[float],
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: str,
    days: int = 365,
    days_to_simulate: int = 63,
) -> List[List[float]]:
    new_start_date = datetime.strptime(start_date, "%Y-%m-%d") - timedelta(days=days)
    trading_days = trading_days_between_dates(new_start_date, start_date)
    var_results = []
    portfolio_log_returns = log_returns.tail(trading_days).dot(weights)
    rolling_window_returns = portfolio_log_returns.rolling(
        window=days_to_simulate
    ).apply(lambda x: np.prod(1 + x) - 1)

    for conf in confidence_levels:
        var_value = portfolio_value * np.percentile(
            rolling_window_returns.dropna(), (1 - conf) * 100
        )
        var_results.append(abs(var_value))

    return var_results


# Calculate VaR using Monte Carlo simulation
def monte_carlo_method(
    portfolio_value: int,
    mean_return: float,
    volatility: float,
    confidence_levels: List[float],
    days_to_simulate: int = 63,
    iterations: int = 10000,
) -> List[float]:
    random_numbers = np.random.normal(0, 1, [1, iterations])
    simulated_portfolio_value = portfolio_value * np.exp(
        days_to_simulate * (mean_return - 0.5 * volatility**2)
        + volatility * np.sqrt(days_to_simulate) * random_numbers
    )
    simulated_portfolio_value = np.sort(simulated_portfolio_value)

    var_results = []
    for conf in confidence_levels:
        percentile = np.percentile(simulated_portfolio_value, (1 - conf) * 100)
        var_value = portfolio_value - percentile
        var_results.append(var_value)

    return var_results, simulated_portfolio_value


def trading_days_between_dates(
    start_date: datetime, end_date: datetime, exchange: str = "XSES"
) -> int:
    cal = mcal.get_calendar(exchange)
    trading_days = cal.schedule(start_date=start_date, end_date=end_date)
    return len(trading_days)


if __name__ == "__main__":
    # Testing
    main(4, "2022-01-01", 1000000)
