import streamlit as st
import numpy as np
from typing import List
import pickle
import os


def get_date_ranges() -> list:

    with open("../data/old_data/date_ranges.pkl", "rb") as f:
        date_ranges = pickle.load(f)

    return date_ranges


@st.cache_data(ttl="30d")
def get_portfolio_weights() -> list:

    with open("../data/old_data/portfolio_weights.pkl", "rb") as f:
        portfolio_returns = pickle.load(f)
        portfolio_volatilities = pickle.load(f)

    return np.array(portfolio_returns), np.array(portfolio_volatilities)


@st.cache_data(ttl="30d")
def get_optimal_weights() -> list:

    with open("../data/old_data/optimal_weights.pkl", "rb") as f:
        optimal_returns = pickle.load(f)
        optimal_volatilities = pickle.load(f)

    return (
        # np.array(optimal_weights),
        np.array(optimal_returns),
        np.array(optimal_volatilities),
    )


@st.cache_data(ttl="30d")
def get_efficient_frontier() -> list:

    with open("../data/old_data/efficient_frontier.pkl", "rb") as f:
        efficient_weights = pickle.load(f)
        portfolio_returns = pickle.load(f)
        portfolio_volatilities = pickle.load(f)

    return (
        np.array(efficient_weights),
        np.array(portfolio_returns),
        np.array(portfolio_volatilities),
    )


@st.cache_data(ttl="30d")
def get_ticker_data() -> List[str]:

    with open("../data/old_data/tickers_data.pkl", "rb") as f:
        tickers = pickle.load(f)

    return tickers


@st.cache_data(ttl="30d")
def get_run_id() -> int:

    with open("../data/old_data/run_id.pkl", "rb") as f:
        run_id = pickle.load(f)

    return run_id


@st.cache_data(ttl="30d")
def get_ticker_names() -> dict:

    with open("../data/old_data/ticker_names.pkl", "rb") as f:
        ticker_names = pickle.load(f)

    return ticker_names
