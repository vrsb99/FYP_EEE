import pandas as pd
import yfinance as yf
import urllib.request


def main() -> None:
    """
    Fetch and filter the tickers of all equities in the Singapore stock market.

    Returns:
        None
    """
    ticker_source = fetch_tickers()
    # print(ticker_source)
    filtered_stocks = ticker_source
    filtered_stocks = filter_stocks(ticker_source)
    print(f"{len(filtered_stocks)} stocks remaining")
    next_index = len(filtered_stocks)
    sti_index = [
        next_index + 1,
        "Straight Times Index",
        "^STI",
        "Benchmark",
    ]  # Adding STI as a benchmark
    filtered_stocks.loc[next_index] = sti_index

    filtered_stocks.to_parquet("../data/obtain_tickers/equities.parquet")


def fetch_tickers() -> pd.DataFrame:
    """
    Fetch the tickers of all equities from topforeignstocks.com.

    Returns:
        pd.DataFrame: A DataFrame containing the tickers of all equities.
    """
    url = "https://topforeignstocks.com/listed-companies-lists/the-complete-list-of-listed-companies-in-singapore/"
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    return pd.read_html(urllib.request.urlopen(req))[0]


def filter_stocks(ticker_source: pd.DataFrame) -> pd.DataFrame:
    """
    Filter stocks based on specific criteria such as being delisted, not equities, insufficient data, or insufficient market cap.

    Args:
        ticker_source (pd.DataFrame): The DataFrame containing the initial list of tickers.

    Returns:
        pd.DataFrame: A DataFrame containing the filtered list of tickers.
    """
    counter = 0
    delisted = []
    non_equity = []
    insufficient_data = []
    insufficient_market_cap = []

    for idx, ticker in enumerate(ticker_source["Code"]):
        try:
            data = yf.Ticker(ticker)
            quote_type = data.fast_info["quoteType"]
        except Exception as e:
            print(f"{ticker} is delisted")
            delisted.append(idx)
            continue

        if quote_type != "EQUITY":
            print(f"{ticker} is not an equity. Data not retrieved.")
            non_equity.append(idx)
            continue

        info = data.info
        keys = ["marketCap", "averageVolume"]
        if not all(key in info for key in keys):
            print(f"{ticker} has insufficient data. Data not retrieved.")
            insufficient_data.append(idx)
            continue

        market_cap = info["marketCap"]
        avg_volume = info["averageVolume"]

        if market_cap < 10e6 or avg_volume < 50e3:
            print(
                f"{ticker} has insufficient market cap of {market_cap} or avg volume of {avg_volume}. Data not retrieved."
            )
            insufficient_market_cap.append(idx)
        else:
            print(f"{ticker} has sufficient market cap and average volume.")
            counter += 1
            ticker_source.loc[idx, "Sector"] = info["sector"]

    print(f"{counter} tickers have sufficient market cap and average volume.")
    to_drop = delisted + non_equity + insufficient_data + insufficient_market_cap
    ticker_source.drop(to_drop, inplace=True)
    return ticker_source


if __name__ == "__main__":
    main()
