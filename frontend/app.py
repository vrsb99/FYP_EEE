import streamlit as st
import pandas as pd
from typing import Tuple, List
import numpy as np
import riskfolio as rp
import plotly.express as px
from st_pages import Page, Section, show_pages, show_pages, add_page_title
import quantstats as qs
import glob
import os
import threading
from risk_measures import MODELS, rm_names
from datetime import datetime, timedelta
import pandas_market_calendars as mcal
import plotly.graph_objects as go
from plotting import plot_efficient_frontier
from descriptions import (
    main_description_new,
    scenario_analysis_info,
)

add_page_title()

show_pages(
    [
        Page("app.py", "Portfolio Optimisation", "💵"),
        Page("old.py", "Interim Progress (Old)", "🔨"),
        Section("How its made", "🛠"),
        Page("pages/ranking_equities.py", "1 Ranking Equities", "🔍"),
        Page("pages/studying_models.py", "2 Studying Models", "📊"),
        Page("pages/validation.py", "3 Validation", "📈"),
    ]
)


def download_data() -> None:
    """
    Checks if data directory exists, if not, executes a script to download the data.

    Returns:
        None: This function does not return any values.
    """

    data_exists = os.path.exists("../data")
    if not data_exists:
        os.system("python download_data.py")


data_download_thread = threading.Thread(target=download_data)
data_download_thread.start()

financial_crisis = {
    "European Debt Crisis": ("2011-07-01", "2012-06-30"),
    "Chinese Stock Market Crash": ("2015-06-12", "2015-08-31"),
    "US-China Trade War Impact": ("2018-06-15", "2019-12-13"),
    "2019 Singapore Economic Slowdown": ("2019-01-01", "2019-12-31"),
    "COVID-19 Impact": ("2020-02-20", "2020-03-23"),
}


def get_equities_data() -> pd.DataFrame:
    """
    Retrieves equities data from parquet files, specifically the stock codes and their corresponding closing prices.

    Returns:
        pd.DataFrame: DataFrame containing equities data with stock codes as column headers.
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
            continue

    equities_data = pd.concat(data.values(), axis=1)
    equities_data.index = pd.to_datetime(equities_data.index)
    return equities_data


def calculate_profit_and_loss(
    data: pd.DataFrame, benchmark: pd.DataFrame, weights: pd.DataFrame
) -> Tuple[float, float, float]:
    """
    Calculates the total return of the portfolio and the benchmark, and the profit and loss by comparing the two.

    Args:
        data (pd.DataFrame): DataFrame containing the portfolio returns.
        benchmark (pd.DataFrame): DataFrame containing the benchmark returns.
        weights (pd.DataFrame): DataFrame containing weights for each stock in the portfolio.

    Returns:
        Tuple[float, float, float]: Portfolio return, benchmark return, and the profit and loss.
    """

    benchmark_return = calculate_total_return(benchmark)
    portfolio_return = calculate_total_return(data, weights)
    profit_and_loss = portfolio_return - benchmark_return
    return portfolio_return, benchmark_return, profit_and_loss


def calculate_total_return(data: pd.DataFrame, weights: pd.DataFrame = None) -> float:
    """
    Calculates the total return of a portfolio or a single asset based on provided data and weights.

    Args:
        data (pd.DataFrame): DataFrame containing the price data.
        weights (pd.DataFrame, optional): DataFrame containing weights for each asset, defaults to None for single asset calculation.

    Returns:
        float: The total return as a percentage.
    """

    if weights is not None:
        total_return = 0
        for ticker, row in weights.iterrows():
            weight = row["weights"]
            if ticker in data.columns:
                stock_df = data[ticker].dropna()
                if not stock_df.empty:
                    start_price = stock_df.iloc[0]
                    end_price = stock_df.iloc[-1]
                    total_return += ((end_price - start_price) / start_price) * weight
                else:
                    print(f"No data for {ticker}")
        total_return *= 100
    else:
        start_price = data.iloc[0]
        end_price = data.iloc[-1]
        total_return = ((end_price - start_price) / start_price) * 100
    return total_return


def calculate_stock_return(df: pd.DataFrame) -> float:
    """
    Calculates the return of a single stock based on provided data.

    Args:
        df (pd.DataFrame): DataFrame containing the price data.

    Returns:
        float: The return as a percentage.
    """
    start_price = df.iloc[0, 0]
    end_price = df.iloc[-1, 0]
    return (end_price - start_price) / start_price * 100


def get_asset_classes() -> pd.DataFrame:
    """
    Retrieves asset classes from a parquet file which includes the stock codes, company names, and sectors.

    Returns:
        pd.DataFrame: DataFrame containing asset classes.
    """

    asset_classes = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet",
        columns=["Code", "Company Name", "Sector"],
    )
    return asset_classes


def get_stock_and_sector_returns(
    data: pd.DataFrame, tickers: list, crisis_name: str, portfolio_value: int
) -> pd.DataFrame:
    """
    Calculates returns for specific stocks and sectors during defined crisis periods and computes dollar returns based on a given portfolio value.

    Args:
        data (pd.DataFrame): DataFrame containing stock data.
        tickers (list): List of ticker symbols to analyze.
        crisis_name (str): Name of the crisis period to reference in output.
        portfolio_value (int): Total value of the portfolio to calculate dollar returns.

    Returns:
        pd.DataFrame: DataFrame with indexed results by sector, stock name, and ticker with returns data.
    """

    asset_classes = get_asset_classes()

    result_list = []

    for ticker in tickers:
        name = asset_classes.loc[
            asset_classes["Code"] == ticker, "Company Name"
        ].values[0]
        sector = asset_classes.loc[asset_classes["Code"] == ticker, "Sector"].values[0]

        stock_df = data[[ticker]].dropna()
        if stock_df.empty:
            print(f"No data for {ticker}")
            continue

        stock_return = calculate_stock_return(stock_df)
        crisis_name_intials = "".join([name[0] for name in crisis_name.split(" ")])
        result_list.append(
            {
                "Ticker": ticker,
                "Name": name,
                "Sector": sector,
                f"{crisis_name} Returns %": stock_return,
                f"{crisis_name_intials} Returns $": stock_return
                * portfolio_value
                / 100,
            }
        )

    result_df = pd.DataFrame(result_list)
    grouped_df = result_df.set_index(["Sector", "Name", "Ticker"])

    return grouped_df


def fetch_portfolio_weights(
    year: int, quarter: str, selected_model: str, selected_risk_measure: str
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Fetches portfolio weights and returns data for a specific year, quarter, model, and risk measure from parquet files.

    Args:
        year (int): Year of the data.
        quarter (str): Quarter of the year.
        selected_model (str): The model used in the analysis.
        selected_risk_measure (str): The risk measure associated with the model.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: DataFrames containing portfolio weights and a series of returns.
    """

    weights_filename_pattern = f"../data/studying_models/port_weights/{selected_model}_{selected_risk_measure}_{year}_{quarter}.parquet"
    returns_filename_pattern = f"../data/studying_models/output_returns/{selected_model}_{selected_risk_measure}_{year}_{quarter}.parquet"

    weights_files = glob.glob(weights_filename_pattern)
    returns_files = glob.glob(returns_filename_pattern)

    if not weights_files or not returns_files:
        raise FileNotFoundError("The specified files do not exist.")

    weights_df = pd.read_parquet(weights_files[0])
    returns_df = pd.read_parquet(returns_files[0])

    returns_series = pd.Series(returns_df.squeeze())
    returns_series.index = pd.to_datetime(returns_series.index)

    return weights_df, returns_series


# Calculate VaR for multiple periods and confidence levels
def calculate_historical_var(
    portfolio_value: int,
    confidence_levels: List[float],
    log_returns: pd.DataFrame,
    weights: np.ndarray,
    start_date: datetime,
    days: int = 365,
    type_of_var: str = "historic",
) -> dict:
    new_start_date = start_date - timedelta(days=days)
    trading_days = trading_days_between_dates(new_start_date, start_date)
    simple_returns = np.exp(log_returns) - 1
    port_returns = simple_returns.tail(trading_days).dot(weights)

    var_results = {}
    for conf in confidence_levels:
        if type_of_var == "Conditional":
            var_value = rp.RiskFunctions.CVaR_Hist(port_returns, alpha=1 - conf)
        elif type_of_var == "Entropic":
            var_value, z = rp.RiskFunctions.EVaR_Hist(port_returns, alpha=1 - conf)
        else:
            var_value = rp.RiskFunctions.VaR_Hist(port_returns, alpha=1 - conf)

        var_results[conf] = var_value * portfolio_value

    return var_results


# Calculate VaR using Monte Carlo simulation
@st.cache_resource
def monte_carlo_var(
    log_returns, portfolio_value, weights, confidence_levels, no_of_simulations
):
    # Convert log returns to simple returns
    simple_returns = np.exp(log_returns) - 1

    # Calculate mean and standard deviation of portfolio returns
    port_returns = simple_returns.dot(weights)
    mean = port_returns.mean()
    std_dev = port_returns.std()

    # Simulate future returns using geometric Brownian motion
    np.random.seed(42)
    simulated_returns = np.random.normal(
        loc=mean, scale=std_dev, size=(no_of_simulations)
    )

    # Calculate simulated future portfolio values
    simulated_end_portfolio_values = portfolio_value * np.exp(simulated_returns)

    # Calculate the VaR at the desired confidence level
    var_results = {}
    for conf in confidence_levels:
        var_value = np.percentile(simulated_end_portfolio_values, (1 - conf) * 100)
        var_results[conf] = abs(var_value - portfolio_value)

    return var_results, simulated_end_portfolio_values


def trading_days_between_dates(
    start_date: datetime, end_date: datetime, exchange: str = "XSES"
) -> int:
    cal = mcal.get_calendar(exchange)
    trading_days = cal.schedule(start_date=start_date, end_date=end_date)
    return len(trading_days)


def show_monte_carlo_simulations(
    simulation_data: np.ndarray,
    portfolio_value: int,
    var_values: list,
    confidence_levels: list,
):
    simulation_data = sorted(simulation_data)
    # Create the base line chart for simulated portfolio values
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(simulation_data))),
            y=simulation_data,
            mode="lines",
            name="Simulated Portfolio Value",
        )
    )
    length = len(simulation_data)

    # Add VaR lines to the plot
    for var_value, conf_level in zip(var_values, confidence_levels):
        fig.add_shape(
            go.layout.Shape(
                type="line",
                x0=0,
                x1=length,
                y0=portfolio_value - var_value,
                y1=portfolio_value - var_value,
                line=dict(width=2, dash="dash"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[0, length],
                y=[portfolio_value - var_value, portfolio_value - var_value],
                mode="lines",
                name=f"VaR at {conf_level*100}% Confidence Level",
            )
        )
        st.write(
            f"VaR at {conf_level*100}% confidence level: :red[${round(portfolio_value - var_value, 2)}] which is a loss of :red[{round(var_value / np.mean(simulation_data[0]) * 100, 2)}%] of the portfolio value"
        )

        # Add labels and title
        fig.update_layout(
            title=f"Monte Carlo Simulation for Portfolio",
            xaxis_title="Iteration",
            yaxis_title="Simulated Portfolio Value",
        )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


def main():
    # With the research done, Risk Parity, Risk Parity 3 Factors are the best models
    # Best risk measures are Standard
    # Hence, append "Preferred" to the model names which contain these keywords
    models_org = MODELS
    rm_names_org = rm_names
    models = [
        (
            model + " (Preferred)"
            if "Risk Parity" in model and "Hierarchical" not in model
            else model
        )
        for model in MODELS
    ]
    preferred_models = [
        model
        for model in MODELS
        if "Risk Parity" in model and "Hierarchical" not in model
    ]

    quarters = ["Q1", "Q2", "Q3", "Q4"]
    year = st.sidebar.selectbox("Select Year", range(2003, 2024), index=19)
    quarter = st.sidebar.selectbox("Select Quarter", quarters, index=2)
    portfolio_value = st.sidebar.number_input("Portfolio Value", value=1000000)
    confidence_levels_slider = st.sidebar.slider(
        "Confidence Levels", 0.90, 0.99, (0.95, 0.99), 0.01, format="%.2f"
    )
    three_confidences = [
        confidence_levels_slider[0],
        (confidence_levels_slider[0] + confidence_levels_slider[1]) / 2,
        confidence_levels_slider[1],
    ]

    st.markdown(f"### Overview for {year} {quarter}")

    st.markdown("### 1. Exploring Investment Strategies")
    selected_model_modified = st.selectbox(
        "Choose an Investment Strategy", models, index=5
    )
    selected_model = models_org[models.index(selected_model_modified)]

    if selected_model == preferred_models[0]:
        rm_names_modified = [
            (
                f"{rm} (Preferred)"
                if rm
                in [
                    "Mean Absolute Deviation",
                    "Semi Standard Deviation",
                ]
                else rm
            )
            for rm in rm_names_org
        ]
        rm_names_modified = [
            rm
            for rm in rm_names_modified
            if rm
            not in ["Worst Realization", "Average Drawdown", "Max Drawdown", "Range"]
        ]
    elif selected_model == preferred_models[1]:
        rm_names_modified = [
            "Mean Absolute Deviation (Preferred)",
            "Semi Standard Deviation (Preferred)",
        ]
    elif selected_model == "Monte Carlo":
        rm_names_modified = ["Standard Deviation"]
    else:
        rm_names_modified = rm_names_org.copy()

    st.markdown("### 2. Understanding Risk Measures")
    selected_risk_measure_modifed = st.selectbox(
        "Select a Measure of Risk", rm_names_modified
    )
    selected_risk_measure = rm_names_org[
        rm_names_modified.index(selected_risk_measure_modifed)
    ]

    st.markdown(main_description_new)

    equities = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet",
        columns=["Code", "Company Name", "Sector"],
    )
    w, strat_ret = fetch_portfolio_weights(
        year, quarter, selected_model, selected_risk_measure
    )

    if w.shape[0] == 1:
        st.error(
            f"The analysis for {selected_model} on {year} {quarter} is unavailable. Please adjust your selection"
        )

    returns = pd.read_parquet(
        f"../data/studying_models/input_returns/{year}_{quarter}.parquet"
    )
    results = {}

    # Start date for historical VaR calculation
    start_date = returns.index.max().date()

    weights_series = w.squeeze()
    weights_series = weights_series.reindex(returns.columns, fill_value=0.0)

    if not strat_ret.empty:
        # Display dataframe with stats
        pie_tab, table_tab, efficient_frontier_tab = st.tabs(
            ["Pie Chart", "Table", "Efficient Frontier"]
        )
        with pie_tab:
            st.plotly_chart(plot_pie(w, equities, title=selected_model))

        with table_tab:
            weights = w.index.to_list()
            ticker_weights_df = equities[equities["Code"].isin(weights)].copy()
            ticker_weights_df["Weight (%)"] = np.round(
                ticker_weights_df["Code"].map(weights_series).fillna(0.0) * 100, 3
            )
            ticker_weights_df.rename(
                columns={"Company Name": "Name", "Code": "Ticker"}, inplace=True
            )
            total_row_df = pd.DataFrame(
                [
                    {
                        "Name": "Total",
                        "Ticker": "",
                        "Weight (%)": ticker_weights_df["Weight (%)"].sum(),
                    }
                ]
            )

            ticker_weights_df = pd.concat(
                [ticker_weights_df, total_row_df], ignore_index=True
            )

            st.dataframe(
                ticker_weights_df.set_index("Ticker"), use_container_width=True
            )

        with efficient_frontier_tab:
            models_to_plot = [
                "Risk Parity",
                "Risk Parity 3-Factors",
                "Hierarchical Risk Parity",
                selected_model,
            ]
            models_to_plot = np.unique(models_to_plot)
            plot_efficient_frontier(
                year, quarter, models_to_plot, [selected_risk_measure], returns
            )

        types = ["Historic", "Conditional", "Entropic"]
        years = [1, 2, 3]
        results["Monte Carlo VaR"], simulated_portfolio_val = monte_carlo_var(
            returns, portfolio_value, weights_series, three_confidences, 10000
        )

        for type_of_var in types:
            for year in years:
                results[f"{type_of_var} VaR {year}Yr"] = calculate_historical_var(
                    portfolio_value,
                    three_confidences,
                    returns,
                    weights_series,
                    start_date,
                    365 * year,
                    type_of_var,
                )

        value_at_risk_tab, monte_carlo_sim_tab = st.tabs(
            ["Value at Risk", "Monte Carlo Simulation"]
        )

        with value_at_risk_tab:
            # Display the VaR results
            st.markdown("### 3. Value at Risk")
            st.markdown(
                f"VaR is a measure of the :red[losses] that a portfolio may experience at given :red[confidence levels]."
            )
            all_var_types = types + ["Monte Carlo"]
            var_types = st.multiselect(
                "Select VaR Type", all_var_types, default=all_var_types
            )
            years_selected = st.multiselect(
                "Select Number of Years to Calculate VaR", years, default=years[0]
            )

            results_pd = pd.DataFrame(results)
            results = results_pd.loc[
                :, results_pd.columns.str.contains("|".join(var_types))
            ]
            years_selected += ["Monte Carlo"]
            results = results.loc[
                :, results.columns.str.contains("|".join(map(str, years_selected)))
            ]

            results.index = [f"{int(conf*100)}%" for conf in three_confidences]
            results.index.name = "Confidence Level"
            results_T = results.T
            st.dataframe(results_T, use_container_width=True)
        with monte_carlo_sim_tab:
            st.markdown("### Monte Carlo Simulation")
            show_monte_carlo_simulations(
                simulated_portfolio_val,
                portfolio_value,
                list(results_pd["Monte Carlo VaR"]),
                three_confidences,
            )

        st.markdown("### 4. Scenario Analysis")
        st.markdown(
            "Scenario analysis is a technique used to :red[analyze decisions] through :red[speculation] of various possible outcomes in financial investments."
        )

        scenario_summary_data = []
        sector_returns = []
        equities_data = get_equities_data()
        sti_data = equities_data["^STI"].ffill(limit=5).bfill(limit=5)
        # Calculate profit and loss for each scenario
        for crisis_name, (start_date, end_date) in financial_crisis.items():
            start_date = datetime.strptime(start_date, "%Y-%m-%d")
            end_date = datetime.strptime(end_date, "%Y-%m-%d")

            equities_data_crisis = equities_data.loc[start_date:end_date]
            sti_data_crisis = sti_data.loc[start_date:end_date]

            (
                portfolio_return,
                benchmark_return,
                profit_and_loss,
            ) = calculate_profit_and_loss(equities_data_crisis, sti_data_crisis, w)

            stock_and_sector_returns = get_stock_and_sector_returns(
                equities_data_crisis, w.index.tolist(), crisis_name, portfolio_value
            )
            scenario_summary_data.append(
                {
                    "Scenario": crisis_name,
                    "P/L ($)": profit_and_loss * portfolio_value / 100,
                    "P/L (%)": profit_and_loss,
                    "P. Return ($)": portfolio_return * portfolio_value / 100,
                    "P. Return (%)": portfolio_return,
                    "B. Return ($)": benchmark_return * portfolio_value / 100,
                    "B. Return (%)": benchmark_return,
                }
            )
            sector_returns.append(stock_and_sector_returns)

        scenario_summary_df = pd.DataFrame(scenario_summary_data)
        scenario_summary_df = (
            scenario_summary_df.sort_values(
                by=scenario_summary_df.columns[3], ascending=False
            )
            .reset_index(drop=True)
            .set_index("Scenario")
        )

        combined_sector_returns = pd.concat(sector_returns, axis=1)

        fontname = "DejaVu Sans"
        scenario_summary_tab, affected_sector_tab = st.tabs(
            ["Scenario Summary", "Affected Sectors"]
        )
        with scenario_summary_tab:
            st.dataframe(scenario_summary_data, use_container_width=True)
        with affected_sector_tab:
            st.dataframe(combined_sector_returns, use_container_width=True)
        with st.expander("Show More Information About Scenario Analysis"):
            st.markdown(scenario_analysis_info)

        with st.expander("View Metrics"):
            metrics = qs.reports.metrics(strat_ret, display=False)
            st.dataframe(metrics, use_container_width=True)

        with st.expander("View Cumulative Returns"):
            ret_plot = qs.plots.returns(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(ret_plot, clear_figure=True)
            log_ret_plot = qs.plots.log_returns(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(log_ret_plot, clear_figure=True)

        with st.expander("View YOY Returns"):
            eoy_ret = qs.plots.yearly_returns(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(eoy_ret, clear_figure=True)

        with st.expander("View Rolling Metrics (Volatility, Sharpe, Sortino)"):
            length_of_returns = len(strat_ret)
            win_year, win_half_year = qs.reports._get_trading_periods(length_of_returns)
            roll_vol = qs.plots.rolling_volatility(
                returns=strat_ret,
                show=False,
                period_label=f"{win_half_year} days",
                period=win_half_year,
                periods_per_year=win_year,
                fontname=fontname,
            )
            st.pyplot(roll_vol, clear_figure=True)

            roll_sharpe = qs.plots.rolling_sharpe(
                returns=strat_ret,
                show=False,
                period_label=f"{win_half_year} days",
                period=win_half_year,
                periods_per_year=win_year,
                fontname=fontname,
            )
            st.pyplot(roll_sharpe, clear_figure=True)

            roll_sortino = qs.plots.rolling_sortino(
                returns=strat_ret,
                show=False,
                period_label=f"{win_half_year} days",
                period=win_half_year,
                periods_per_year=win_year,
                fontname=fontname,
            )
            st.pyplot(roll_sortino, clear_figure=True)

        with st.expander(
            "View Drawdowns, Earnings, Monthly Returns, and Returns Distribution"
        ):
            dd_periods = qs.plots.drawdowns_periods(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(dd_periods, clear_figure=True)
            dd = qs.plots.drawdown(returns=strat_ret, show=False, fontname=fontname)
            st.pyplot(dd, clear_figure=True)
            earnings = qs.plots.earnings(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(earnings, clear_figure=True)
            month_ret = qs.plots.monthly_heatmap(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(month_ret, clear_figure=True)
            distr_ret = qs.plots.distribution(
                returns=strat_ret, show=False, fontname=fontname
            )
            st.pyplot(distr_ret, clear_figure=True)
    else:
        st.error(
            "No data available for the selected parameters. Please adjust your selections and try again."
        )


def plot_pie(w: pd.DataFrame, equities: pd.DataFrame, title: str | None = None):
    company_names = (
        equities.set_index("Code")["Company Name"].reindex(w.index).fillna("Unknown")
    )
    fig = px.pie(w, values="weights", names=company_names, title=title)
    fig.update_traces(textinfo="percent+label")
    return fig


if __name__ == "__main__":
    main()
