import streamlit as st
import pandas as pd
import altair as alt
from st_pages import Page, Section, show_pages, show_pages, add_page_title
import glob
from typing import Tuple

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


def fetch_ratios_data(
    year: str, quarter: str
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Fetches Sharpe ratio data for specified year and quarter from parquet files. The function loads data for ordinary,
    expected three-factor, and expected five-factor Sharpe ratios, along with equity metadata.

    Args:
        year (str): The year of the data.
        quarter (str): The quarter of the data.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: A tuple containing dataframes for
        ordinary Sharpe ratios, expected three-factor Sharpe ratios, expected five-factor Sharpe ratios, and equities metadata.
    """

    ordinary_ratio_filename_pattern = (
        f"../data/ranking_equities/ordinary/{year}_{quarter}.parquet"
    )
    expected_five_filename_pattern = (
        f"../data/ranking_equities/expected_five/{year}_{quarter}.parquet"
    )
    expected_three_filename_pattern = (
        f"../data/ranking_equities/expected_three/{year}_{quarter}.parquet"
    )

    ordinary_files = glob.glob(ordinary_ratio_filename_pattern)
    expected_five_files = glob.glob(expected_five_filename_pattern)
    expected_three_files = glob.glob(expected_three_filename_pattern)

    if not ordinary_files:
        raise FileNotFoundError("The specified ordinary files do not exist.")

    ordinary_ratio = pd.read_parquet(ordinary_files[0]).reset_index()
    ordinary_ratio.columns = ["Ticker", "Historical Sharpe Ratio"]

    if expected_three_files:
        expected_three_ratio = pd.read_parquet(expected_three_files[0]).reset_index()
        expected_three_ratio.columns = ["Ticker", "Expected 3-Factor Sharpe Ratio"]
    else:
        expected_three_ratio = pd.DataFrame(
            columns=["Ticker", "Expected 3-Factor Sharpe Ratio"]
        )

    if expected_five_files:
        expected_five_ratio = pd.read_parquet(expected_five_files[0]).reset_index()
        expected_five_ratio.columns = ["Ticker", "Expected 5-Factor Sharpe Ratio"]
    else:
        expected_five_ratio = pd.DataFrame(
            columns=["Ticker", "Expected 5-Factor Sharpe Ratio"]
        )

    equities = pd.read_parquet(
        "../data/obtain_tickers/equities.parquet",
        columns=["Code", "Company Name"],
    )

    return ordinary_ratio, expected_three_ratio, expected_five_ratio, equities


def visualize_sharpe_ratios(
    data: pd.DataFrame, title: str, sharpe_ratio_column: str
) -> None:
    """
    Visualizes Sharpe ratios using Altair bar charts. The function sorts the data to display the top 30 entries with the highest Sharpe ratios.

    Args:
        data (pd.DataFrame): DataFrame containing the data to visualize which includes Tickers and their respective Sharpe Ratios.
        title (str): Title for the visualization.
        sharpe_ratio_column (str): Column name in `data` containing Sharpe ratio values.

    Returns:
        None: This function does not return a value but renders a chart within a Streamlit container.
    """

    sorted_data = data.sort_values(sharpe_ratio_column, ascending=False)[:30]

    st.subheader(title)

    chart = (
        alt.Chart(sorted_data)
        .mark_bar()
        .encode(
            x=alt.X("Ticker", sort="-y"),
            y=alt.Y(sharpe_ratio_column, title="Sharpe Ratio"),
            color="Ticker",
            tooltip=data.columns.tolist(),
        )
    )

    st.altair_chart(chart, use_container_width=True)


def main() -> None:
    """
    Main function to run the application. It sets up a Streamlit interface to select year and quarter,
    fetches Sharpe ratios data, merges and prepares the data for display, and renders visualizations for different Sharpe ratio metrics.

    Returns:
        None: The function sets up the Streamlit interface and does not return any values.
    """

    year = st.sidebar.selectbox("Select Year", range(2003, 2024), index=20)
    quarter = st.sidebar.selectbox("Select Quarter", ["Q1", "Q2", "Q3", "Q4"])

    ordinary_ratio, expected_three, expected_five, equities = fetch_ratios_data(
        year, quarter
    )

    all_ratios = pd.merge(ordinary_ratio, expected_three, on="Ticker")
    all_ratios = pd.merge(all_ratios, expected_five, on="Ticker")
    all_ratios = pd.merge(equities, all_ratios, left_on="Code", right_on="Ticker")
    all_ratios = all_ratios.drop(columns=["Code", "Ticker"])
    all_ratios = all_ratios.round(3)
    all_ratios = all_ratios.sort_values("Historical Sharpe Ratio", ascending=False)

    st.header(f"Sharpe Ratio Rankings for {quarter} {year}")
    st.dataframe(all_ratios, use_container_width=True)

    visualize_sharpe_ratios(
        ordinary_ratio,
        f"Sharpe Ratios Historical Returns for {quarter} {year}",
        "Historical Sharpe Ratio",
    )

    with st.expander("Explanation on Expected Sharpe Ratios"):
        st.markdown(
            """The 3-Factor and 5-Factor Sharpe Ratios are :green[equal] as they are calculated using :rainbow[Principal Component Regression] which consist
                    of :orange[Principal Component Analysis (PCA)] and :orange[Linear Regression]."""
        )

    three_factor_tab, five_factor_tab = st.tabs(
        ["Expected 3-Factor Sharpe Ratios", "Expected 5-Factor Sharpe Ratios"]
    )

    with three_factor_tab:
        visualize_sharpe_ratios(
            expected_three,
            f"Sharpe Ratios on 3-Factor Returns for {quarter} {year}",
            "Expected 3-Factor Sharpe Ratio",
        )
    with five_factor_tab:
        visualize_sharpe_ratios(
            expected_five,
            f"Sharpe Ratios on 5-Factor Returns for {quarter} {year}",
            "Expected 5-Factor Sharpe Ratio",
        )


if __name__ == "__main__":
    main()
