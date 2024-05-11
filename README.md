# Optimal Portfolio Allocation

## Overview

This repository contains the codebase for the Optimal Portfolio Allocation project, which aims to provide an efficient and automated way to allocate assets in a financial portfolio using machine learning and mathematical optimization techniques. The project is divided into two main components:

- **Backend**: Handles data collection, preprocessing, and optimization algorithms.
- **Frontend**: Provides a user interface for interacting with the system.

## Live Demo

You can access the live demo of the project [here](https://vrsbfyp.me/).

## Components

### Backend

- `00_download_all_files.py`: Downloading all generated data from DigitalOcean Spaces.
- `01_obtain_tickers.py`: To download and store ticker symbols from [TopForeignStocks.com](https://topforeignstocks.com/listed-companies-lists/the-complete-list-of-listed-companies-in-singapore/)
- `02_obtain_data.py`: To download and store technical data from Yahoo Finance.
- `03_ranking_equities.py`: Ranking equities based on Sharpe Ratio.
- `04_test_models.py`: Testing different allocation models for a quarterly period.
- `05_validation.py`: Backtesting each model on a rolling basis, over a given period.
- `06_upload_app.py`: Upload all generated data to DigitalOcean Spaces.
- `07_review.ipynb`: Reviewing the results of the models and generate graphs for report
- `08_other_backtest.ipynb`: Reviewing the results of other backtests and generate graphs for report
- `bucket.py`: Get environment variables.
- `plotting.py`: Helper functions for plotting the efficient frontier.
- `portfolio.py`: Helper functions for each allocation model.
- `risk_measures.py`: List of types of risk measures calculated for.

### Frontend

#### Components currently in use

- `app.py`: Main file for the Streamlit app.
- `pages/ranking_equities.py`: Streamlit page for ranking equities based on Sharpe Ratio.
- `pages/studying_models.py`: Streamlit page for studying different allocation models.
- `pages/validation.py`: Streamlit page for validating models on a rolling basis.
- `pages/other_backtest.py`: Streamlit page for validating other backtests.
- `bucket.py`: Get environment variables.
- `description.py`: Description for frontend.
- `plotting.py`: Helper functions for plotting the efficient frontier.
- `risk_measures.py`: List of types of risk measures calculated for.

#### Components not in use (From interim progress report)

- `old.py`: Streamlit app, Frontend.
- `functions/backtest.py`: Backtesting for Scenario Analysis.
- `functions/VAR.py`: Value at Risk.
- `database.py`: Database functions for the frontend.
- `functions/backtest.py`: Backtesting for Scenario Analysis.

## Installation

1. Install the required packages using the following command:

```bash
pip install -r requirements.txt
```

2. Set up the environment variables by creating a `.env` file in the root directory and adding the following variables:

```bash
ACCESS_KEY=<YOUR_ACCESS_KEY>
SECRET_KEY=<YOUR_SECRET_KEY>
REGION_NAME=<YOUR_REGION_NAME>
ENDPOINT_URL=<YOUR_ENDPOINT_URL>
BUCKET_NAME=<YOUR_BUCKET_NAME>
```

3. Run the Streamlit app using the following command:

Windows:

```powershell
cd frontend; streamlit run app.py
```

Linux:

```bash
cd frontend && streamlit run app.py
```
