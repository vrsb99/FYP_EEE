import streamlit as st
import pandas as pd
from typing import List
from st_pages import Page, Section, show_pages, show_pages, add_page_title
import altair as alt

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

MODELS = [
    "Classic",
    "Black-Litterman",
    "Black-Litterman 3-Factors",
    "Risk Parity",
    "Risk Parity 3-Factors",
    "Hierarchical Risk Parity",
    "Monte Carlo",
]

RM_NAMES = ["Standard Deviation", "Mean Absolute Deviation"]

YEARS = list(range(1, 7))


def create_chart_for_model(
    sharpe_df: pd.DataFrame,
    model: str,
    risk_measures: List[str],
    default_color: str = "lightblue",
) -> alt.Chart:
    """
    Creates a bar chart for Sharpe ratios filtered by specified model and risk measures using Altair visualization library.

    Args:
        sharpe_df (pd.DataFrame): DataFrame containing Sharpe ratios and model information.
        model (str): The specific model to filter Sharpe ratios for visualization.
        risk_measures (List[str]): List of risk measures to include in the filter.
        default_color (str): Default color for the bars in the chart.

    Returns:
        alt.Chart: An Altair Chart object representing the Sharpe ratios visualization.
    """

    model_df = sharpe_df[
        sharpe_df["Model & Risk Measure"].str.contains(model)
        & sharpe_df["Model & Risk Measure"].str.contains("|".join(risk_measures))
    ]

    color_conditions = [
        (model_df["Model & Risk Measure"].str.contains("5-Factors"), "lightcoral"),
        (model_df["Model & Risk Measure"].str.contains("3-Factors"), "lightsalmon"),
    ]

    model_df["color"] = default_color
    for condition, color in color_conditions:
        model_df.loc[condition, "color"] = color

    chart = (
        alt.Chart(model_df)
        .mark_bar()
        .encode(
            x=alt.X("Model & Risk Measure:N", sort="-y", axis=alt.Axis(labelAngle=-45)),
            y=alt.Y("Sharpe Ratio:Q"),
            color=alt.Color("color:N", scale=None),
            tooltip=["Model & Risk Measure", "Sharpe Ratio"],
        )
        .properties(title=f"Sharpe Ratios for {model}", width=700, height=400)
    )

    return chart


def main() -> None:
    """
    Main function to execute the Streamlit app, which provides interactive analysis of Sharpe ratios across different years,
    models, and risk measures. It enables users to select various models and measures, visualize Sharpe ratios trends,
    and explore the effects of lookback periods and factor models on the performance of allocation models.

    Returns:
        None: This function does not return any values but performs data fetching, processing, and rendering visualizations in Streamlit.
    """

    st.markdown("## 1. Investigating the effect of lookback period")
    st.write(
        "Studying the effect of :orange[lookback period] on the performance of different allocation models and risk measures by comparing the :orange[Sharpe Ratios] across different years."
    )
    selected_models = st.multiselect("Select Models", MODELS, default=MODELS)
    selected_risk_measures = st.multiselect(
        "Select Risk Measures", RM_NAMES, default=RM_NAMES
    )

    sharpe_ratios = pd.read_parquet(
        "../data/backtest/validation/sharpe_ratios_all_lookback.parquet"
    )
    sharpe_data = []

    for year in YEARS:
        for model in selected_models:
            if model == "Monte Carlo":
                if "Mean Absolute Deviation" in selected_risk_measures:
                    current_risk_measures = ["Standard Deviation"]
                else:
                    current_risk_measures = selected_risk_measures
            else:
                current_risk_measures = selected_risk_measures

            for rm in current_risk_measures:
                try:
                    # Row with year and column with model_rm
                    sharpe_ratio = sharpe_ratios.loc[year, f"{model}_{rm}"]
                    sharpe_data.append(
                        {
                            "Year": year,
                            "Model & Risk Measure": f"{model} - {rm}",
                            "Sharpe Ratio": sharpe_ratio,
                        }
                    )
                except FileNotFoundError:
                    st.warning(f"File for {model} with {rm} in year {year} not found.")

    sharpe_df = pd.DataFrame(sharpe_data)

    if not sharpe_df.empty:
        chart = (
            alt.Chart(sharpe_df)
            .mark_line(point=True)
            .encode(
                x="Year:N",
                y=alt.Y("Sharpe Ratio:Q", scale=alt.Scale(zero=False)),
                color="Model & Risk Measure:N",
                tooltip=["Year", "Sharpe Ratio", "Model & Risk Measure"],
            )
            .properties(
                title="Sharpe Ratios by Model and Risk Measure Across Years",
                width=700,
                height=400,
            )
            .interactive()
        )

        st.altair_chart(chart, use_container_width=True)
    else:
        st.write("No data to display. Please adjust your selections.")

    average_sharpe_df = sharpe_df.groupby("Year")["Sharpe Ratio"].mean().reset_index()

    average_chart = (
        alt.Chart(average_sharpe_df)
        .mark_line(point=True)
        .encode(
            x=alt.X(
                "Year:O", title="Lookback Period (Years)", axis=alt.Axis(labelAngle=0)
            ),
            y=alt.Y("Sharpe Ratio:Q", title="Average Sharpe Ratio"),
            tooltip=["Year", "Sharpe Ratio"],
        )
        .properties(
            title="Average Sharpe Ratio Over Lookback Period (Years)",
            width=700,
            height=400,
        )
        .interactive()
    )

    st.altair_chart(average_chart, use_container_width=True)

    st.markdown("## 2. Investigating the effect of factor models on allocation models")
    st.write(
        "Studying the effect of different :orange[factor models] on the performance of different allocation models by comparing the :orange[Sharpe Ratios]."
    )

    sharpe_for_factors = pd.read_parquet(
        "../data/backtest/validation/sharpe_ratios_factors.parquet"
    )

    models = [
        "Classic",
        "Black-Litterman",
        "Risk Parity",
    ]

    selected_models_for_factors = st.multiselect(
        "Select Models for Factors", models, default=models
    )
    selected_risk_measures_for_factors = st.multiselect(
        "Select Risk Measures for Factors", RM_NAMES, default=RM_NAMES[1]
    )

    sharpe_for_factors = sharpe_for_factors.reset_index()
    sharpe_for_factors.rename(columns={"index": "Model & Risk Measure"}, inplace=True)

    for model in selected_models_for_factors:
        if model == "Classic":
            st.write(
                "It can be observed that the performance of the original model is identical to both the :blue[3-Factor] and :blue[5-Factor] models.\n\
                    This could be due to the :green[historical method] used to :orange[estimate returns and the covariance matrices]"
            )
        elif model == "Risk Parity":
            st.write(
                "It can be observed that the performance of the :blue[5-Factor] model is identical to the :blue[3-Factor] model.\n\
                     This is probably due to :orange[Principal Component Regression (PCR)] used to estimate returns.\
                     "
            )
        else:
            st.write(
                "It can be observed that the performance of the :blue[5-Factor] differs from the :blue[3-Factor] mode.\n\
                     This is probably due to the Black-Litterman :green[integrating subjective views] into the model.\
                     "
            )

        model_chart = create_chart_for_model(
            sharpe_for_factors, model, selected_risk_measures_for_factors
        )
        st.altair_chart(model_chart, use_container_width=True)


if __name__ == "__main__":
    main()
