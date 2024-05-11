# Standard Libraries
import numpy as np
import pandas as pd
from datetime import datetime
from typing import List, Tuple
import pickle

# Third-Party Libraries
import plotly.graph_objects as go
import pandas_market_calendars as mcal
import streamlit as st

# Local Modules
from functions.VAR import main as var_main
from functions.backtesting import main as backtesting_main
from database import (
    get_date_ranges,
    get_portfolio_weights,
    get_optimal_weights,
    get_efficient_frontier,
    get_ticker_data,
    get_run_id,
    get_ticker_names,
)
from descriptions import (
    more_info_portfolios,
    more_info_sharpe_ratio,
    main_description,
    risk_free_info,
    scenario_analysis_info,
)
from st_pages import Page, Section, show_pages, show_pages, add_page_title

add_page_title()

show_pages(
    [
        Page("app.py", "Portfolio Optimization", "💵"),
        Page("old.py", "Interim Progress (Old)", "🔨"),
        Section("How its made", "🛠"),
        Page("pages/ranking_equities.py", "1 Ranking Equities", "🔍"),
        Page("pages/studying_models.py", "2 Studying Models", "📊"),
        Page("pages/validation.py", "3 Validation", "📈"),
    ]
)

# Constants
DATE_FORMAT = "%Y-%m-%d"


# ----------------- Utility Functions ----------------- #
def trading_days_between_dates(
    start_date: datetime, end_date: datetime, exchange: str = "XSES"
) -> int:
    cal = mcal.get_calendar(exchange)
    trading_days = cal.schedule(start_date=start_date, end_date=end_date)
    return len(trading_days)


@st.cache_data(ttl="30d")
def get_three_month_yield() -> float:
    # yield_rates = pd.read_html(
    #     "http://www.worldgovernmentbonds.com/country/singapore/"
    # )[1]
    # three_mnth_yield = float(yield_rates.iloc[5, 2].replace("%", ""))
    three_mnth_yield = 0.0468
    return three_mnth_yield


def get_selected_date_range(st: st, date_options: list) -> Tuple[str, str]:
    length_date_options = len(date_options)
    selected_date_option = st.selectbox(
        "Select Date Range:",
        date_options,
        index=length_date_options - 1,
        help="Select the date range to generate the optimal portfolio.",
    )
    selected_start_date, selected_end_date = selected_date_option.split(" to ")
    selected_start_date = selected_start_date.split("From ")[1].strip()
    selected_end_date = selected_end_date.strip()
    return selected_start_date, selected_end_date


def get_risk_free_rate(st: st, three_mnth_yield: float) -> float:
    risk_free_rate = (
        st.number_input(
            "Risk Free Rate (%)",
            value=three_mnth_yield,
            step=0.1,
            format="%.3f",
            min_value=0.0,
            max_value=10.0,
            help=risk_free_info,
        )
        / 100
    )
    return risk_free_rate


# ----------------- Streamlit Components ----------------- #
@st.cache_resource
def show_portfolios(
    portfolio_returns: np.ndarray,
    portfolio_volatilities: np.ndarray,
    optimal_returns: np.ndarray,
    optimal_volatilities: np.ndarray,
    efficient_returns: np.ndarray = None,
    efficient_volatilities: np.ndarray = None,
    trading_days: int = 1,
) -> None:
    sharpe_ratios = [
        np.float64(portfolio_return) / np.float64(portfolio_volatility)
        for portfolio_return, portfolio_volatility in zip(
            portfolio_returns, portfolio_volatilities
        )
    ]

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=portfolio_volatilities,
            y=portfolio_returns,
            mode="markers",
            marker=dict(size=5, color=sharpe_ratios, colorscale="Viridis"),
            name="Portfolios",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=optimal_volatilities,
            y=optimal_returns,
            mode="markers",
            marker=dict(size=15, color="green"),
            name="Optimal Portfolio",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=efficient_volatilities,
            y=efficient_returns,
            mode="lines+markers",
            marker=dict(size=5, color="magenta"),
            name="Efficient Frontier",
        )
    )

    fig.update_layout(
        title="Portfolio Weights Scatter Plot",
        xaxis_title="Expected Volatility",
        yaxis_title="Expected Return",
        showlegend=True,
    )
    st.plotly_chart(fig, use_container_width=True)


@st.cache_resource
def show_monte_carlo_simulations(
    simulation_data: np.ndarray,
    days: int,
    portfolio_value: int,
    var_values: list,
    confidence_levels: list,
):
    # Create the base line chart for simulated portfolio values
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=list(range(len(simulation_data[0]))),
            y=simulation_data[0],
            mode="lines",
            name="Simulated Portfolio Value",
        )
    )
    length = len(simulation_data[0])

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
        title=f"Monte Carlo Simulation for Portfolio over {days} Days",
        xaxis_title="Iteration",
        yaxis_title="Simulated Portfolio Value",
    )

    # Display the plot in Streamlit
    st.plotly_chart(fig, use_container_width=True)


@st.cache_data
def get_summary_data_and_simulation(
    run_id: int,
    selected_start_date: str,
    portfolio_value: int,
    optimal_weights: np.ndarray,
    confidence_levels: List[float],
    number_of_days_to_simulate: int,
) -> Tuple[pd.DataFrame, np.ndarray, List[float]]:
    portfolios_and_simulation = var_main(
        run_id,
        selected_start_date,
        portfolio_value,
        optimal_weights,
        confidence_levels,
        number_of_days_to_simulate,
    )
    portfolios, simuation = (
        portfolios_and_simulation[:-1],
        portfolios_and_simulation[-1],
    )
    summary_data = pd.DataFrame()
    for i, conf in enumerate(confidence_levels):
        summary_data[f"{conf*100} % VaR"] = [
            portfolios[0][i],  # Monte Carlo VaR
            portfolios[1][i],  # 1-year historical VaR
            portfolios[2][i],  # 2-year historical VaR
            portfolios[3][i],  # 3-year historical VaR
            portfolios[4][i],  # Parametric VaR
        ]
    summary_data.index = [
        "Monte Carlo VaR",
        "1 Year Historical VaR",
        "2 Year Historical VaR",
        "3 Year Historical VaR",
        "Parametric VaR",
    ]
    summary_data.index.name = "Methodology"
    summary_data = summary_data.round(2)

    return summary_data, simuation, portfolios[0]


# @st.cache_data
def backtesting(
    run_id: int,
    portfolio_value: int,
    optimal_weights: np.ndarray,
    start_date: datetime,
    end_date: datetime,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    return backtesting_main(
        run_id, portfolio_value, optimal_weights, start_date, end_date
    )


def main():
    st.title("Portfolio Optimizer", ":chart_with_upwards_trend:")

    # Get available date ranges
    date_ranges = get_date_ranges()
    date_options = [f"From {start} to {end}" for start, end in date_ranges]

    with st.sidebar:
        st.header("Parameters")

        # Get selected date range
        selected_start_date, selected_end_date = get_selected_date_range(
            st, date_options
        )

        # Get run_id, trading_days and other date related calculations
        run_id = get_run_id()
        start_date_obj = datetime.strptime(selected_start_date, DATE_FORMAT)
        end_date_obj = datetime.strptime(selected_end_date, DATE_FORMAT)
        trading_days = trading_days_between_dates(start_date_obj, end_date_obj)

        # Get risk-free rate
        three_mnth_yield = get_three_month_yield()
        risk_free_rate = get_risk_free_rate(st, three_mnth_yield)

        portfolio_value = st.number_input(
            "Enter Portfolio Value ($)",
            value=10000.0,
            step=1000.0,
            format="%.2f",
            min_value=1000.0,
            max_value=1000000.0,
        )

        max_confidence_level = st.slider(
            "Select Maximum Confidence Level",
            min_value=0.75,
            max_value=0.99,
            value=0.99,
            step=0.01,
            help="Select the maximum confidence level to calculate VaR for.",
        )
        confidence_levels = np.arange(
            max_confidence_level - 0.06, max_confidence_level, 0.02
        )
        confidence_levels = np.round(confidence_levels, 2)

        number_of_days_to_simulate = st.number_input(
            "Enter Number of Days to Simulate",
            value=trading_days,
            step=1,
            format="%d",
            min_value=1,
            max_value=trading_days,
            help="Select the number of days to simulate VaR for.",
        )

    # Get portfolio and optimal weights for selected dates
    (
        # portfolio_weights,
        portfolio_returns,
        portfolio_volatilities,
    ) = get_portfolio_weights()
    optimal_returns, optimal_volatilities = get_optimal_weights()
    (
        efficient_weights,
        efficient_returns,
        efficient_volatilities,
    ) = get_efficient_frontier()
    tickers = get_ticker_data()

    st.subheader(":chart_with_upwards_trend: Optimal Portfolio Allocation")
    st.subheader(
        f"for {start_date_obj.strftime('%d %B %Y')} to {end_date_obj.strftime('%d %B %Y')}"
    )
    st.markdown(main_description(len(portfolio_returns), len(tickers)))
    with st.expander("Show More Information About Sharpe Ratio"):
        st.markdown(more_info_sharpe_ratio)
    with st.expander("Show More Information About How Portfolios Were Generated"):
        st.markdown(more_info_portfolios)
    st.markdown(
        "Portfolios contain :green[4 to 20 stocks] with a :green[minimum weight of 5%]."
    )

    st.subheader("Portfolio Picker")
    st.markdown(
        "By default, the portfolio with the :green[best return] for the :blue[lowest volatility] is selected, which has the :green[highest Sharpe Ratio]."
    )
    st.markdown(
        "Would you like a portfolio with a :green[higher return] or :blue[lower volatility]?"
    )
    option = [
        f"{np.round(efficient_return*100,2)}% Return, {np.round(efficient_volatilitiy*100,2)}% Volatility"
        for efficient_return, efficient_volatilitiy in zip(
            efficient_returns, efficient_volatilities
        )
    ]

    sharpe_ratios = np.round(
        (efficient_returns - risk_free_rate) / (efficient_volatilities),
        3,
    )
    max_sharpe = np.max(sharpe_ratios)
    max_sharpe_indices = np.where(sharpe_ratios == max_sharpe)[0]
    default_index = max_sharpe_indices[np.argmax(efficient_returns[max_sharpe_indices])]
    # Added due to shutdown of database, only one option is selectable now
    option = [option[default_index]]
    selected_portfolio = st.selectbox("Select Portfolio", option, index=0)
    # selected_portfolio = st.selectbox(
    #     "Select Portfolio", option, index=int(default_index)
    # )

    selected_portfolio_index = option.index(selected_portfolio)
    selected_portfolio_weights = efficient_weights[selected_portfolio_index]
    selected_portfolio_returns = efficient_returns[selected_portfolio_index]
    selected_portfolio_volatility = efficient_volatilities[selected_portfolio_index]

    calculated_return = selected_portfolio_returns
    st.subheader(f"Allocation and Statistics for :blue[{trading_days}] Trading Days")

    left_col, right_col = st.columns((1, 1))
    with left_col:
        # Creating DataFrame to hold the ticker and weights
        ticker_weights_df = pd.DataFrame(
            {
                "Ticker": tickers,
                "Weight (%)": np.round(selected_portfolio_weights * 100, 2),
            }
        )

        # Remove rows with weights equal to zero
        ticker_weights_df = ticker_weights_df[ticker_weights_df["Weight (%)"] > 0]

        # Fetching the names corresponding to the tickers
        ticker_names = get_ticker_names()
        ticker_weights_df["Name"] = ticker_weights_df["Ticker"].map(ticker_names)
        # Appending the total weight
        total_weight_row = pd.DataFrame(
            {
                "Name": ["Total"],
                "Ticker": [""],
                "Weight (%)": [np.round(selected_portfolio_weights.sum() * 100)],
            },
            index=["Total"],
        )

        ticker_weights_df = pd.concat([ticker_weights_df, total_weight_row])

        # Displaying the DataFrame with Name as the index
        st.dataframe(ticker_weights_df.set_index("Name"), use_container_width=True)
    with right_col:
        st.write(f"Expected Return: {np.round(np.exp(calculated_return)-1,2)*100}%")
        st.write(f"Log Return: {np.round(calculated_return*100,2)}%")
        st.write(f"Volatility: {np.round(selected_portfolio_volatility*100, 2)}%")
        sharpe_ratio = (selected_portfolio_returns - risk_free_rate) / (
            selected_portfolio_volatility
        )
        st.write(f"Sharpe Ratio: {np.round(sharpe_ratio, 3)}")
        sortino_ratio = (selected_portfolio_returns - risk_free_rate) / (
            np.std(portfolio_returns[portfolio_returns < 0])
        )
        st.write(f"Sortino Ratio: {np.round(sortino_ratio, 3)}")

    st.write(
        "Below is the scatter plot of the :blue[Risk Adjusted Returns] of each portfolio generated and the :green[Optimal Portfolio]."
    )

    show_portfolios(
        portfolio_returns,
        portfolio_volatilities,
        optimal_returns,
        optimal_volatilities,
        efficient_returns,
        efficient_volatilities,
        trading_days,
    )

    # st.subheader("Value At Risk (VaR)")
    # st.markdown(
    #     f"VaR is a measure of the :red[losses] that a portfolio may experience over a :blue[{trading_days}-day] period at given :red[confidence levels]."
    # )

    # summary_data, simulation, portfolios = get_summary_data_and_simulation(
    #     run_id,
    #     selected_start_date,
    #     portfolio_value,
    #     selected_portfolio_weights,
    #     confidence_levels,
    #     number_of_days_to_simulate,
    # )
    # tab1, tab2 = st.tabs(["VAR Summary", "Monte Carlo Simulations"])
    # with tab1:
    #     st.dataframe(summary_data, use_container_width=True)
    # with tab2:
    #     show_monte_carlo_simulations(
    #         simulation,
    #         number_of_days_to_simulate,
    #         portfolio_value,
    #         portfolios,
    #         confidence_levels,
    #     )

    st.subheader("Scenario Analysis")
    st.markdown(
        "Scenario analysis is a technique used to :red[analyze decisions] through :red[speculation] of various possible outcomes in financial investments."
    )

    scenario_summary_data, affected_sectors_data = backtesting(
        run_id,
        portfolio_value,
        selected_portfolio_weights,
        start_date_obj,
        end_date_obj,
    )

    tab1, tab2 = st.tabs(["Scenario Summary", "Affected Sectors"])
    with tab1:
        st.dataframe(scenario_summary_data, use_container_width=True)
    with tab2:
        st.dataframe(affected_sectors_data, use_container_width=True)
    with st.expander("Show More Information About Scenario Analysis"):
        st.markdown(scenario_analysis_info)


if __name__ == "__main__":
    main()
