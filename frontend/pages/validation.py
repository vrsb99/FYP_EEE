import streamlit as st
import pandas as pd
import quantstats as qs
import altair as alt
from typing import Tuple, List
from st_pages import Page, Section, show_pages, show_pages, add_page_title
import numpy as np
from risk_measures import MODELS, rm_names

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


@st.cache_data()
def fetch_backtest_data(backtest_period: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Fetches the backtest results and monthly cumulative metrics from parquet files for a given backtest period.

    Args:
        backtest_period (int): The backtest period in years for which to fetch data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containing two DataFrames: one with all models' backtest results and another with monthly cumulative metrics.
    """

    backtest_all_results = pd.read_parquet(
        f"../data/backtest/validation/all_models_values_Y{backtest_period}.parquet",
        engine="pyarrow",
    )
    monthly_cumulative_metrics = pd.read_parquet(
        f"../data/backtest/validation/monthly_cumulative_metrics_Y{backtest_period}.parquet",
        engine="pyarrow",
    )

    return backtest_all_results, monthly_cumulative_metrics


def display_metric_charts(
    metrics_df: pd.DataFrame,
    group_by_column: str,
    selected_items: List[str],
    ratio_types: List[str],
) -> None:
    """
    Displays metric charts for selected items and ratio types, grouped by a specified column. It visualizes median values of metrics over time.

    Args:
        metrics_df (pd.DataFrame): DataFrame containing metrics data including dates and various ratios.
        group_by_column (str): The column by which to group data (e.g., 'Model' or 'RiskMeasure').
        selected_items (List[str]): The items (models or risk measures) to include in the visualization.
        ratio_types (List[str]): The types of ratios to visualize (e.g., Sharpe Ratio, Sortino Ratio).

    Returns:
        None: This function renders visualizations directly in a Streamlit interface and does not return any value.
    """

    tabs = st.tabs(ratio_types)

    for tab, ratio in zip(tabs, ratio_types):
        with tab:
            chart_data = pd.DataFrame()

            for item in selected_items:
                if group_by_column == "Model":
                    filtered_data = metrics_df[metrics_df["Model"] == item]
                else:
                    filtered_data = metrics_df[metrics_df["RiskMeasure"] == item]

                median_values = (
                    filtered_data.groupby("Date")[ratio].median().reset_index()
                )
                median_values[group_by_column] = item
                median_values["Value"] = median_values[ratio]
                chart_data = pd.concat([chart_data, median_values], ignore_index=True)

            line_chart = (
                alt.Chart(chart_data)
                .mark_line()
                .encode(
                    x="Date:T",
                    y=alt.Y("Value:Q", title=f"Median {ratio}"),
                    color=f"{group_by_column}:N",
                    tooltip=["Date:T", "Value:Q", f"{group_by_column}:N"],
                )
                .properties(
                    title=f"Median {ratio} Over Time by {group_by_column}",
                    width=700,
                    height=400,
                )
                .interactive()
            )

            st.altair_chart(line_chart, use_container_width=True)


def main() -> None:
    """
    Main function to execute the Streamlit app for backtest analysis. It provides interactive options to select the backtest period,
    models, and risk measures, and displays the results in various visual formats including line charts and summary tables.

    Returns:
        None: The function sets up the Streamlit interface and does not return any values but performs data fetching, processing, and visualization.
    """

    st.markdown("### 1. Select the backtest period")
    backtest_periods_avail = [21, 15]
    start_date_values = {
        21: pd.to_datetime("2003-01-02"),
        15: pd.to_datetime("2009-01-02"),
    }
    backtest_period = st.selectbox(
        "Select Backtest Period", backtest_periods_avail, index=0
    )

    st.markdown(
        f"## Viewing Backtest Results for all allocation models over {backtest_period} years"
    )
    st.markdown("### 2. Select the start and end date for the backtest")

    start_date = st.date_input(
        "Start Date",
        start_date_values[backtest_period],
        min_value=start_date_values[backtest_period],
        max_value=pd.to_datetime("2023-12-31"),
    )
    end_date = st.date_input(
        "End Date",
        pd.to_datetime("2024-03-01"),
        min_value=start_date_values[backtest_period] + pd.DateOffset(month=3),
        max_value=pd.to_datetime("2024-03-01"),
    )

    st.markdown("### 3. Select the models to compare")
    models_selected = st.multiselect(
        "Select Allocation Models",
        MODELS,
        default=["Risk Parity", "Risk Parity 3-Factors"],
    )
    risk_measures_selected = st.multiselect(
        "Select Risk Measures", rm_names, default=rm_names[:2]
    )
    backtest_all_results, monthly_cumulative_metrics = fetch_backtest_data(
        backtest_period
    )

    start_date_pd = pd.to_datetime(start_date)
    end_date_pd = pd.to_datetime(end_date)

    # Filter results according to the selected inputs
    benchmark = "STI_Standard Deviation"

    backtest_filtered_dates = backtest_all_results.loc[start_date_pd:end_date_pd]

    st.markdown("### 4. Backtest Results")
    if models_selected and risk_measures_selected:
        columns_of_interest = [
            col
            for col in backtest_filtered_dates.columns
            if any(col.split("_")[0] == model for model in models_selected)
            and any(col.split("_")[1] == rm for rm in risk_measures_selected)
        ]

        columns_of_interest.append(benchmark)

        backtest_filtered_model_rm = backtest_filtered_dates[columns_of_interest]

        data_long = backtest_filtered_model_rm.reset_index().melt(
            "index", var_name="Model & Risk Measure", value_name="Value"
        )
        data_long["Model"], data_long["RiskMeasure"] = zip(
            *data_long["Model & Risk Measure"].apply(lambda x: x.split("_"))
        )
        data_long.loc[data_long["Model"] == "STI", "RiskMeasure"] = None
        data_long.loc[
            data_long["Model & Risk Measure"] == benchmark, "Model & Risk Measure"
        ] = "STI"

        tooltip = [
            alt.Tooltip("index:T", title="Date"),
            alt.Tooltip("Value:Q", title="Value", format="$,.2f"),
            alt.Tooltip("Model:N", title="Model"),
            alt.Tooltip("RiskMeasure:N", title="Risk Measure"),
        ]

        # Create an interactive line chart
        line_chart = (
            alt.Chart(data_long)
            .mark_line()
            .encode(
                x=alt.X("index:T", title="Date"),
                y=alt.Y("Value:Q", title="Value in $"),
                color="Model & Risk Measure:N",
                tooltip=tooltip,
            )
            .properties(width=800, height=400)
            .configure_legend(
                orient="top",
                columns=4,
            )
            .interactive()
        )

        st.altair_chart(line_chart, use_container_width=True)

        # Calculate metrics
        ratio_types = ["Sharpe Ratio", "Sortino Ratio", "Calmar Ratio", "Volatility"]
        sharpes = (
            qs.stats.sharpe(backtest_filtered_model_rm)
            .sort_values(ascending=False)
            .rename("Sharpe Ratio")
        )
        volatilities = (
            qs.stats.volatility(backtest_filtered_model_rm)
            .sort_values(ascending=True)
            .rename("Volatility")
        )
        calmars = (
            qs.stats.calmar(backtest_filtered_model_rm)
            .sort_values(ascending=False)
            .rename("Calmar Ratio")
        )
        sortinos = (
            qs.stats.sortino(backtest_filtered_model_rm)
            .sort_values(ascending=False)
            .rename("Sortino Ratio")
        )

        st.markdown(
            "### 5. Trend of Median Sharpe Ratios, Sortino Ratios, Calmar Ratios and Volatility Metrics"
        )
        st.markdown(
            "#### Choose to view the trend of metrics over time for the median of each model or the median of each risk measure"
        )
        model_tab, rm_tab = st.tabs(["Model", "Risk Measure"])

        models_selected += [benchmark.split("_")[0]]
        filtered_df = monthly_cumulative_metrics[
            monthly_cumulative_metrics["Model"].isin(models_selected)
            & monthly_cumulative_metrics["RiskMeasure"].isin(risk_measures_selected)
            & (monthly_cumulative_metrics["Date"] >= start_date_pd)
            & (monthly_cumulative_metrics["Date"] <= end_date_pd)
        ]
        benchmark_prefix = benchmark.split("_")[0]
        matching_rows = monthly_cumulative_metrics["Model"].str.startswith(
            benchmark_prefix
        )
        monthly_cumulative_metrics.loc[matching_rows, "RiskMeasure"] = None
        filtered_df = pd.concat(
            [
                filtered_df,
                monthly_cumulative_metrics[
                    monthly_cumulative_metrics["Model"] == benchmark_prefix
                ],
            ]
        )
        with model_tab:
            display_metric_charts(filtered_df, "Model", models_selected, ratio_types)

        with rm_tab:
            display_metric_charts(
                filtered_df, "RiskMeasure", risk_measures_selected, ratio_types
            )

        st.markdown("### 6. Sharpe, Sortino, Calmar Ratios and Volatility Metrics")
        sharpe_tab, sortino_tab, calmar_tab, vol_tab = st.tabs(ratio_types)

        with sharpe_tab:
            st.markdown(
                "#### Sharpe Ratios\nSharpe ratio is the average return earned per unit of volatility."
            )
            sharpes = sharpes.to_frame()
            sharpes["Model"], sharpes["Risk Measure"] = zip(
                *sharpes.index.str.split("_")
            )
            sharpes.loc[sharpes["Model"] == "STI", "Risk Measure"] = None
            sharpes = sharpes[["Model", "Risk Measure", "Sharpe Ratio"]]

            st.dataframe(sharpes, hide_index=True)
        with sortino_tab:
            st.markdown(
                "#### Sortino Ratios\nSortino ratio is the average return earned per unit of downside volatility."
            )
            sortinos = sortinos.to_frame()
            sortinos["Model"], sortinos["Risk Measure"] = zip(
                *sortinos.index.str.split("_")
            )
            sortinos.loc[sortinos["Model"] == "STI", "Risk Measure"] = None
            sortinos = sortinos[["Model", "Risk Measure", "Sortino Ratio"]]

            st.dataframe(sortinos, hide_index=True)
        with calmar_tab:
            st.markdown(
                "#### Calmar Ratios\nCalmar ratio is the average return earned per unit of drawdown."
            )
            calmars = calmars.to_frame()
            calmars["Model"], calmars["Risk Measure"] = zip(
                *calmars.index.str.split("_")
            )
            calmars.loc[calmars["Model"] == "STI", "Risk Measure"] = None
            calmars = calmars[["Model", "Risk Measure", "Calmar Ratio"]]

            st.dataframe(calmars, hide_index=True)
        with vol_tab:
            st.markdown(
                "#### Volatility\nVolatility is a measure of the dispersion of returns for a given asset or portfolio."
            )
            volatilities = volatilities.to_frame()
            volatilities["Model"], volatilities["Risk Measure"] = zip(
                *volatilities.index.str.split("_")
            )
            volatilities.loc[volatilities["Model"] == "STI", "Risk Measure"] = None
            volatilities = volatilities[["Model", "Risk Measure", "Volatility"]]

            st.dataframe(volatilities, hide_index=True)


if __name__ == "__main__":
    main()
