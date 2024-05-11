import pandas as pd
import yfinance as yf
import datetime
import os


def main() -> None:
    """
    Update or initialize the historical data of equities from the Singapore stock market.
    The function reads a parquet file containing equity tickers, checks if the data for each ticker
    is up to date in the local storage, and downloads new data if necessary using the yfinance API.

    Returns:
        None
    """

    equities = pd.read_parquet("../data/obtain_tickers/equities.parquet")

    for ticker in equities["Code"]:
        # Create a new table for each stock if it doesn't exist
        file_name = (
            f"stock_{ticker[:3]}" if ticker != "^STI" else "STI"
        )  # Remove the .SI suffix

        # Get the latest date from parquet
        latest_date_in_db = None
        if os.path.exists(f"../data/obtain_data/{file_name}.parquet"):
            df = pd.read_parquet(f"../data/obtain_data/{file_name}.parquet")
            latest_date_in_db = df.index.max().strftime("%Y-%m-%d")

        if latest_date_in_db:
            # Start from the day after the latest date in the database
            yesterday_date = (
                datetime.datetime.today() - datetime.timedelta(days=1)
            ).strftime("%Y-%m-%d")
            if latest_date_in_db >= yesterday_date:
                # If the latest date is today, skip
                print(f"Data for {ticker} is up to date.")
                continue
            else:
                print(f"Data for {ticker} is not up to date. Updating...")
                start_date = (
                    datetime.datetime.strptime(latest_date_in_db, "%Y-%m-%d")
                    + datetime.timedelta(days=1)
                ).strftime("%Y-%m-%d")
                historical_data = yf.download(
                    ticker, start=start_date, interval="1d", threads=True
                )
                historical_data = pd.concat([df, historical_data])
                historical_data.to_parquet(f"../data/obtain_data/{file_name}.parquet")
        else:
            # Starts from 2010-01-01 if the table is empty
            print(f"Data for {ticker} is empty. Downloading...")
            start_date = "2000-01-01"
            historical_data = yf.download(
                ticker, start=start_date, interval="1d", threads=True
            )
            historical_data.to_parquet(f"../data/obtain_data/{file_name}.parquet")


if __name__ == "__main__":
    main()
