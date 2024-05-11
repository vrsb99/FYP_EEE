import pandas as pd
import plotly.graph_objects as go
import numpy as np
import streamlit as st
from typing import Tuple


def compute_portfolio_stats(
    returns: pd.DataFrame, weights: pd.Series, annualization_factor: int = 252
) -> Tuple[float, float]:
    """
    Computes the expected annual return and annual volatility of a portfolio given its returns and weights.

    Args:
        returns (pd.DataFrame): DataFrame containing the historical returns of assets in the portfolio.
        weights (pd.Series): Series containing the weights of the respective assets in the portfolio.
        annualization_factor (int): The factor used to annualize the returns, typically 252 for daily returns.

    Returns:
        Tuple[float, float]: A tuple containing the expected annual return and annual volatility of the portfolio.
    """

    weights /= weights.sum()

    expected_return = np.dot(weights, returns.mean()) * annualization_factor

    cov_matrix = returns.cov() * annualization_factor
    volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    return expected_return, volatility


def plot_efficient_frontier(
    year: int, quarter: int, models: list, rm_names: list, returns: pd.DataFrame
) -> None:
    """
    Generates a Plotly graph object representing the efficient frontier for a selection of portfolio models
    and risk measures based on the provided returns data. The function plots each model and risk measure combination
    as a unique point on the graph and connects the points representing the efficient frontier.

    Args:
        year (int): The year for which the data is plotted.
        quarter (int): The quarter for which the data is plotted.
        models (list): List of portfolio models to include in the frontier plot.
        rm_names (list): List of risk measures associated with the models.
        returns (pd.DataFrame): DataFrame containing the returns data used to compute the efficient frontier.

    Returns:
        None: This function directly plots the efficient frontier using Plotly within a Streamlit application and does not return any values.
    """

    model_markers = {
        "Classic": "circle",
        "Classic 3-Factors": "square",
        "Black-Litterman": "triangle-up",
        "Black-Litterman 3-Factors": "diamond",
        "Black-Litterman 5-Factors": "pentagon",
        "Risk Parity": "triangle-left",
        "Risk Parity 3-Factors": "triangle-right",
        "Hierarchical Risk Parity": "hexagram",
        "Monte Carlo": "star",
    }

    model_colours = {
        "Classic": "green",
        "Classic 3-Factors": "red",
        "Black-Litterman": "cyan",
        "Black-Litterman 3-Factors": "magenta",
        "Black-Litterman 5-Factors": "yellow",
        "Risk Parity": "greenyellow",
        "Risk Parity 3-Factors": "deepskyblue",
        "Hierarchical Risk Parity": "lightcoral",
        "Monte Carlo": "lightseagreen",
    }

    fig = go.Figure()

    portfolios = {}
    for model in models:
        if model in ["Monte Carlo"]:
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

        for risk_measure in risk_measures:
            portfolios[model, risk_measure] = pd.read_parquet(
                f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_{quarter}.parquet"
            )

    model = "Frontier"
    risk_measure = "Standard Deviation"
    frontier_pd = pd.read_parquet(
        f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_{quarter}.parquet"
    )
    frontier_returns = []
    frontier_volatility = []
    annualization_factor = 252

    for col in frontier_pd.columns:
        weights = frontier_pd[col]
        expected_return, volatility = compute_portfolio_stats(
            returns, weights, annualization_factor
        )
        frontier_returns.append(expected_return)
        frontier_volatility.append(volatility)

    for (model, risk_measure), weights in portfolios.items():
        weights = weights.squeeze()
        weights = weights.reindex(returns.columns, fill_value=0)

        expected_return, volatility = compute_portfolio_stats(
            returns, weights, annualization_factor
        )

        # print(expected_return, volatility)
        fig.add_trace(
            go.Scatter(
                x=[volatility],
                y=[expected_return],
                mode="markers",
                marker=dict(
                    symbol=model_markers[model],
                    color=model_colours[model],
                    size=10,
                ),
                name=f"{model}, {risk_measure}",
            )
        )

    # print(frontier_returns)
    # print(frontier_volatility)
    fig.add_trace(
        go.Scatter(
            x=frontier_volatility,
            y=frontier_returns,
            mode="markers+lines",
            line=dict(color="orange", width=2),
            name="Efficient Frontier",
        )
    )

    # Update layout
    fig.update_layout(
        title=f"Efficient Frontier for {quarter} {year}",
        xaxis_title="Volatility (Standard Deviation)",
        yaxis_title="Expected Return",
        legend=dict(yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    st.plotly_chart(fig, use_container_width=True)
