import numpy as np
import pandas as pd
from typing import List, Tuple
import pickle


def get_ticker_data() -> List[str]:

    with open("../data/old_data/tickers_data.pkl", "rb") as f:
        tickers = pickle.load(f)

    return tickers


def filter_tickers(
    tickers: List[str], weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.DataFrame({"Ticker": tickers, "Weight": weights})
    df = df[df["Weight"] > 0]
    return df["Ticker"].to_numpy(), df["Weight"].to_numpy()
