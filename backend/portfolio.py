import pandas as pd
import numpy as np
from typing import Tuple
import quantstats as qs
import riskfolio as rp
from risk_measures import risk_measures
import logging

logging.basicConfig(
    filename="../logs/portfolio.txt",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(message)s",
    filemode="w",
)


def compute_acc_returns(returns: pd.DataFrame, w: pd.DataFrame) -> pd.Series:
    """
    Computes accumulated returns for a given set of asset returns and corresponding weights.

    Args:
        returns (pd.DataFrame): Asset returns data.
        w (pd.DataFrame): Corresponding weights for each asset.

    Returns:
        pd.Series: A series representing the accumulated returns over time.
    """

    X = w.columns.tolist()
    index = returns.index.tolist()
    for i, _ in enumerate(X):
        a = np.array(returns, ndmin=2) @ np.array(w[X[i]], ndmin=2).T
        prices = 1 + np.insert(a, 0, 0, axis=0)
        prices = np.cumprod(prices, axis=0)
        prices = np.ravel(prices).tolist()
        del prices[0]
    return pd.Series(prices, index=index)


def compute_portfolio(
    returns: pd.DataFrame,
    selected_portfolio: str,
    risk_measure: str,
    asset_classes: pd.DataFrame | None = None,
    top: int = 30,
    factor_returns: pd.DataFrame | None = None,
    rf_rate: float = 0,
) -> pd.DataFrame:
    """
    Computes the weights of the portfolio using various strategies such as Monte Carlo, Black-Litterman,
    Hierarchical Risk Parity, and others based on the specified risk measures and portfolio models.

    Args:
        returns (pd.DataFrame): DataFrame containing the returns of assets.
        selected_portfolio (str): The portfolio model to use (e.g., 'Monte Carlo', 'Black-Litterman').
        risk_measure (str): The risk measure to use (e.g., 'MV', 'CDaR').
        asset_classes (pd.DataFrame, optional): DataFrame containing asset class information.
        top (int): Number of top assets to consider for the portfolio.
        factor_returns (pd.DataFrame, optional): DataFrame containing factor returns.
        rf_rate (float): The risk-free rate.

    Returns:
        pd.DataFrame: DataFrame containing the portfolio weights.
    """

    rm = risk_measures.loc[risk_measure].loc["rm"]

    portfolio_dict = {
        "Monte Carlo": compute_mc_weights,
        "Classic": compute_classic_weights,
        "Classic 3-Factors": compute_classic_weights,
        "Classic 5-Factors": compute_classic_weights,
        "Black-Litterman": compute_bl_weights,
        "Black-Litterman 3-Factors": compute_bl_weights,
        "Black-Litterman 5-Factors": compute_bl_weights,
        "Hierarchical Risk Parity": compute_hierarchical_weights,
        "Risk Parity": compute_risk_parity_weights,
        "Risk Parity 3-Factors": compute_risk_parity_weights,
        "Risk Parity 5-Factors": compute_risk_parity_weights,
        "Frontier": compute_frontier_weights,
    }
    if selected_portfolio == "Black-Litterman":
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            asset_classes=asset_classes,
            rm=rm,
            rf_rate=rf_rate,
        )
    elif selected_portfolio == "Monte Carlo":
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            rf_rate=rf_rate,
        )

    elif selected_portfolio == "Hierarchical Risk Parity":
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            rm=rm,
            rf_rate=rf_rate,
            top=top,
        )
    elif selected_portfolio in ["Classic", "Frontier", "Risk Parity"]:
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            rm=rm,
            rf_rate=rf_rate,
        )
    elif selected_portfolio in ["Classic 3-Factors", "Classic 5-Factors"]:
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            model="FM",
            rm=rm,
            factor_returns=factor_returns,
            rf_rate=rf_rate,
        )
    elif selected_portfolio in [
        "Black-Litterman 3-Factors",
        "Black-Litterman 5-Factors",
    ]:
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            model="BL_FM",
            asset_classes=asset_classes,
            rm=rm,
            factor_returns=factor_returns,
            rf_rate=rf_rate,
        )
    elif selected_portfolio in ["Risk Parity 3-Factors", "Risk Parity 5-Factors"]:
        w = portfolio_dict[selected_portfolio](
            returns=returns,
            model="FM",
            rm=rm,
            factor_returns=factor_returns,
            rf_rate=rf_rate,
        )

    if w is None:
        sorted_returns = qs.stats.sharpe(returns, smart=True)
        top_sorted_returns = sorted_returns.nlargest(top)
        w = pd.DataFrame(
            [1 / len(top_sorted_returns)] * len(top_sorted_returns),
            index=top_sorted_returns.index,
        )
        print(f"Portfolio {selected_portfolio} failed to compute, using equal weights")

        return w

    if selected_portfolio != "Frontier":
        w[w < 0.001] = 0
        w = w / w.sum()
        w = w[w > 0.001].dropna()

    if len(w) > top and selected_portfolio != "Frontier":
        print(f"Number of assets before filtering: {len(w)}")
        sorted_weights = w.sort_values(ascending=False, by="weights")[:top]

        if "Black-Litterman" in selected_portfolio:
            filtered_assets = (
                sorted_weights.index.tolist()
                if "^STI" in sorted_weights.index.tolist()
                else sorted_weights.index.tolist()[:-1] + ["^STI"]
            )
        else:
            filtered_assets = sorted_weights.index.tolist()

        returns = returns[filtered_assets]

        asset_classes_filtered = (
            asset_classes[asset_classes["Code"].isin(filtered_assets)]
            if asset_classes is not None
            else None
        )
        # Recompute the portfolio weights for these top assets
        updated_w = compute_portfolio(
            returns=returns,
            selected_portfolio=selected_portfolio,
            risk_measure=risk_measure,
            asset_classes=asset_classes_filtered,
            rf_rate=rf_rate,
            factor_returns=factor_returns,
            top=top,
        )
        print(f"Number of assets after filtering: {len(updated_w)}")
        equal_weights = pd.DataFrame(
            [1 / len(updated_w.columns)] * len(updated_w.columns),
            index=updated_w.columns,
            columns=["weights"],
        )

        if updated_w.equals(equal_weights):
            return w
        else:
            return updated_w

    return w


def compute_mc_weights(
    returns: pd.DataFrame, num_simulations: int = 10000, rf_rate: float = 0
) -> pd.DataFrame:
    """
    Computes portfolio weights using the Monte Carlo simulation method to maximize the Sharpe ratio.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        num_simulations (int): The number of simulation runs for the Monte Carlo method.
        rf_rate (float): The risk-free rate used in the Sharpe ratio calculation.

    Returns:
        pd.DataFrame: A DataFrame containing the optimal portfolio weights.
    """

    np.random.seed(1)

    cov_matrix = returns.cov()

    # Define the number of stocks in the portfolio
    num_stocks = len(returns.columns)

    # Define a function to calculate portfolio returns and risks
    def portfolio_performance(weights, returns, cov_matrix):
        portfolio_return = np.sum(returns.mean() * weights) * 252
        portfolio_risk = np.sqrt(
            np.dot(weights.T, np.dot(cov_matrix, weights))
        ) * np.sqrt(252)
        return portfolio_return, portfolio_risk

    # Define the Monte Carlo simulation
    def monte_carlo_simulation(returns, cov_matrix, num_simulations):
        results = np.zeros((3 + num_stocks, num_simulations))
        weights_record = []
        for i in range(num_simulations):
            # Generate random weights
            weights = np.random.random(num_stocks)
            weights /= np.sum(weights)
            weights_record.append(weights)

            # Calculate portfolio performance
            portfolio_return, portfolio_risk = portfolio_performance(
                weights, returns, cov_matrix
            )

            # Save results
            results[0, i] = portfolio_return
            results[1, i] = portfolio_risk
            results[2, i] = (portfolio_return - rf_rate) / portfolio_risk
            for j, _ in enumerate(weights):
                results[j + 3, i] = weights[j]

        return results, weights_record

    # Run the Monte Carlo simulation
    simulation_results, weights_record = monte_carlo_simulation(
        returns, cov_matrix, num_simulations
    )

    # Find the index of the portfolio with the maximum Sharpe ratio
    max_sharpe_idx = np.argmax(simulation_results[2])

    # Get the optimal weights of the portfolio with the maximum Sharpe ratio
    optimal_weights = weights_record[max_sharpe_idx]

    # # Print the optimal weights
    # for i, stock in enumerate(returns.columns):
    #     print(f"Optimal weight for {stock}: {optimal_weights[i]:.2%}")

    w_mc = pd.DataFrame([optimal_weights], columns=returns.columns, index=["weights"]).T
    return w_mc


def compute_frontier_weights(
    returns: pd.DataFrame,
    model: str = "Classic",
    rm: str = "MV",
    method_mu: str = "hist",
    method_cov: str = "hist",
    factor_returns: pd.DataFrame | None = None,
    rf_rate: float = 0,
) -> pd.DataFrame:
    """
    Computes portfolio weights by finding the efficient frontier using the specified risk measure.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        model (str): The portfolio model used (e.g., 'Classic').
        rm (str): The risk measure (e.g., 'MV' for Mean-Variance).
        method_mu (str): The method for calculating expected returns.
        method_cov (str): The method for calculating covariance.
        factor_returns (pd.DataFrame, optional): Additional factor returns data.
        rf_rate (float): The risk-free rate.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio weights.
    """

    # Building the portfolio object
    port = rp.Portfolio(returns=returns, factors=factor_returns)

    # Calculating optimal portfolio
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    # Estimate optimal portfolio:
    port.alpha = 0.05
    hist = True

    w_frontier = port.efficient_frontier(model=model, rm=rm, hist=hist, points=50)

    return w_frontier


def compute_classic_weights(
    returns: pd.DataFrame,
    model: str = "Classic",
    rm: str = "MV",
    obj: str = "Sharpe",
    method_mu: str = "hist",
    method_cov: str = "ledoit",
    factor_returns: pd.DataFrame | None = None,
    rf_rate: float = 0,
) -> pd.DataFrame:
    """
    Computes portfolio weights using traditional methods like Mean-Variance, based on historical returns.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        model (str): Portfolio optimization model (e.g., 'Classic').
        rm (str): The risk measure used in optimization.
        obj (str): The objective function (e.g., 'Sharpe').
        method_mu (str): Method for estimating means.
        method_cov (str): Method for estimating covariance.
        factor_returns (pd.DataFrame, optional): Factor returns data.
        rf_rate (float): The risk-free rate.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio weights.
    """

    # Building the portfolio object
    port = rp.Portfolio(returns=returns, factors=factor_returns)

    # Calculating optimal portfolio
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    if model == "FM":
        comp = factor_returns.shape[1]
        port.factors_stats(feature_selection="PCR", dict_risk=dict(n_components=comp))

    # Estimate optimal portfolio:
    port.alpha = 0.05

    w_mv = port.optimization(model=model, rm=rm, obj=obj, hist=True)

    return w_mv


def compute_risk_parity_weights(
    returns: pd.DataFrame,
    model: str = "Classic",
    rm: str = "MV",
    method_mu: str = "ewma1",
    method_cov: str = "ledoit",
    factor_returns: pd.DataFrame | None = None,
    rf_rate: float = 0,
) -> pd.DataFrame:
    """
    Computes portfolio weights using traditional methods like Mean-Variance, based on historical returns.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        model (str): Portfolio optimization model (e.g., 'Classic').
        rm (str): The risk measure used in optimization.
        obj (str): The objective function (e.g., 'Sharpe').
        method_mu (str): Method for estimating means.
        method_cov (str): Method for estimating covariance.
        factor_returns (pd.DataFrame, optional): Factor returns data.
        rf_rate (float): The risk-free rate.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio weights.
    """

    # Building the portfolio object
    port = rp.Portfolio(returns=returns, factors=factor_returns)

    # Calculating optimal portfolio
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    if model == "FM":
        comp = factor_returns.shape[1]
        port.factors_stats(feature_selection="PCR", dict_risk=dict(n_components=comp))

    # Estimate optimal portfolio:
    port.alpha = 0.05

    w_mv = port.rp_optimization(model=model, rm=rm, hist=True, rf=rf_rate)

    return w_mv


def compute_hierarchical_weights(
    returns: pd.DataFrame,
    model: str = "HRP",
    codependence: str = "distance",
    rm: str = "MV",
    rf_rate: float = 0,
    obj: str = "Sharpe",
    linkage: str = "centroid",
    top: int = 30,
) -> pd.DataFrame:
    """
    Computes portfolio weights using the Hierarchical Risk Parity (HRP) method, incorporating a clustering approach based on asset returns.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        model (str): HRP model type.
        codependence (str): The measure of codependence among assets.
        rm (str): Risk measure.
        rf_rate (float): The risk-free rate.
        obj (str): The objective for optimization.
        linkage (str): The linkage method used in clustering.
        top (int): The number of top assets to include in the portfolio.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio weights.
    """

    # Sort by sharpe
    sharpe = qs.stats.sharpe(returns, smart=True)
    returns = returns[sharpe.sort_values(ascending=False).index.tolist()[:top]]

    # Building the portfolio object
    port_hrp = rp.HCPortfolio(returns=returns)

    # If you want to limit the number of effective assets in the portfolio
    # # First we need to set a solver that support Mixed Integer Programming
    # port_hrp.solvers = ['MOSEK']

    # # Then we need to set the cardinality constraint (maximum number of assets or effective assets)
    # port_hrp.card = top
    # port_hrp.nea = top

    # Estimate optimal portfolio:

    max_k = 10  # Max number of clusters used in two difference gap statistic, only for HERC model
    leaf_order = True  # Consider optimal order of leafs in dendrogram

    w_hrp = port_hrp.optimization(
        model=model,
        codependence=codependence,
        rm=rm,
        rf=rf_rate,
        linkage=linkage,
        max_k=max_k,
        leaf_order=leaf_order,
        obj=obj,
        covariance="hist",
    )
    return w_hrp


def compute_bl_weights(
    returns: pd.DataFrame,
    model: str = "BL",
    asset_classes: pd.DataFrame | None = None,
    rm: str = "MV",
    obj: str = "Sharpe",
    method_mu: str = "ewma1",
    method_cov: str = "ledoit",
    factor_returns: pd.DataFrame | None = None,
    rf_rate: float = 0,
) -> pd.DataFrame:
    """
    Computes portfolio weights using the Black-Litterman model, which incorporates investor views
    into the asset allocation process alongside the market equilibrium.

    Args:
        returns (pd.DataFrame): The returns data for assets.
        model (str): The Black-Litterman model variant.
        asset_classes (pd.DataFrame, optional): DataFrame containing asset class information.
        rm (str): The risk measure used.
        obj (str): The objective, such as maximizing the Sharpe ratio.
        method_mu (str): The method for calculating expected returns.
        method_cov (str): The method for calculating covariance.
        factor_returns (pd.DataFrame, optional): Additional factor returns data.
        rf_rate (float): The risk-free rate.

    Returns:
        pd.DataFrame: A DataFrame containing the portfolio weights according to the Black-Litterman model.
    """

    P, Q = get_bl_P_Q(returns, asset_classes)

    weights = compute_classic_weights(returns)

    # Building the portfolio object
    port = rp.Portfolio(returns=returns, factors=factor_returns)
    port.alpha = 0.05
    # Calculating optimal portfolio
    port.assets_stats(method_mu=method_mu, method_cov=method_cov)

    if model == "BL_FM":
        comp = factor_returns.shape[1]
        loadings = rp.loadings_matrix(
            factor_returns,
            returns,
            feature_selection="PCR",
            n_components=comp,
            criterion="AIC",
        ).drop("const", axis=1)
        P_f, Q_f = get_bl_P_f_Q_f(factor_returns, loadings)

        port.blfactors_stats(
            P_f=P_f,
            Q_f=Q_f / len(returns.index),
            P=P,
            Q=Q / len(returns.index),
            B=loadings,
            rf=rf_rate,
            eq=True,
            w=weights,
            method_mu=method_mu,
            method_cov=method_cov,
            diag=True,
            delta=0.95,
        )
    else:
        port.blacklitterman_stats(
            P,
            Q / len(returns.index),
            rf=rf_rate,
            eq=True,
            w=weights,
            method_mu=method_mu,
            method_cov=method_cov,
            delta=0.95,
        )

    w_bl = port.optimization(model=model, rm=rm, obj=obj, hist=True)
    return w_bl


def get_bl_P_Q(
    returns: pd.DataFrame, asset_classes: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates matrices P and Q for the Black-Litterman model based on views derived from the sector performance.
    This function computes views on expected returns relative to a benchmark (typically the market index).

    Args:
        returns (pd.DataFrame): The returns data for assets.
        asset_classes (pd.DataFrame): DataFrame containing asset class information for grouping and views.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The matrices P (pick matrix of views) and Q (views on the returns), essential for the BL model.
    """

    sharpe = qs.stats.sharpe(returns, smart=True)
    cols = sharpe.reset_index().T.iloc[0, :].tolist()
    sharpe = sharpe.reset_index().T.iloc[1:, :]
    sharpe.columns = cols

    # Function to get the mean Sharpe ratio for a sector
    def get_sector_sharpe(sector, asset_classes):
        stocks = get_stocks_in_sector(sector, asset_classes)
        return sharpe[stocks].mean(axis=1)[0]

    # Method to determine the sign and difference
    def get_sign(sec_est, sec_ref):
        diff = sec_est - sec_ref
        sig = "<=" if diff <= 0 else ">="
        return sig, round(diff, 2)

    # Sector estimates
    sectors = [
        "Benchmark",
        "Real Estate",
        "Healthcare",
        "Basic Materials",
        "Energy",
        "Industrials",
        "Consumer Cyclical",
        "Utilities",
        "Consumer Defensive",
        "Technology",
        "Financial Services",
        "Communication Services",
    ]
    sector_estimates = {
        sector: get_sector_sharpe(sector, asset_classes) for sector in sectors
    }

    # Views construction
    views_list = []
    for sector in sectors:
        if sector == "Benchmark":
            continue
        sig, diff = get_sign(sector_estimates[sector], sector_estimates["Benchmark"])
        views_list.append(
            {
                "Disabled": False,
                "Type": "Classes",
                "Set": "Sector",
                "Position": sector,
                "Sign": sig,
                "Return": diff,
                "Type Relative": "Classes",
                "Relative Set": "Sector",
                "Relative": "Benchmark",
            }
        )

    views = pd.DataFrame(views_list)
    P, Q = rp.assets_views(views, asset_classes)
    return P, Q


def get_bl_P_f_Q_f(
    factor_returns: pd.DataFrame, loadings: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generates matrices P and Q for factor-based views in the Black-Litterman model, adjusting based on specific market conditions.
    These matrices allow for the integration of factor-specific views into the Black-Litterman framework.

    Args:
        factor_returns (pd.DataFrame): Factor returns data used to derive views based on market conditions.
        loadings (pd.DataFrame): Loadings matrix linking factor returns to asset returns.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The matrices P and Q for factor-based views, used in the BL model.
    """

    # Helper function to calculate conditional statistics
    def calculate_conditional_stats(factor, condition):
        # Conditional mean returns
        conditional_mean = factor_returns.loc[condition, factor].mean()
        # Conditional volatility
        conditional_volatility = factor_returns.loc[condition, factor].std()
        return conditional_mean, conditional_volatility

    # Determine market condition: Above or below the median 'Mkt-RF' return
    market_median = factor_returns["Mkt-RF"].median()
    above_median_condition = factor_returns["Mkt-RF"] > market_median

    factors = factor_returns.columns.tolist()
    views_list = []

    for factor in factors:
        if factor == "Mkt-RF":
            continue  # Skip 'Mkt-RF'

        # Calculate mean returns and volatility under above median market condition
        mean_above, vol_above = calculate_conditional_stats(
            factor, above_median_condition
        )
        # Calculate mean returns and volatility under below median market condition
        mean_below, vol_below = calculate_conditional_stats(
            factor, ~above_median_condition
        )

        # Determine which condition (above or below median 'Mkt-RF') the factor performs better risk-adjusted
        sharpe_above = (mean_above - factor_returns["Mkt-RF"].mean()) / vol_above
        sharpe_below = (mean_below - factor_returns["Mkt-RF"].mean()) / vol_below

        # Construct views based on better risk-adjusted performance
        if sharpe_above > sharpe_below:
            sign = ">="
            value = sharpe_above - sharpe_below
        else:
            sign = "<="
            value = sharpe_below - sharpe_above

        views_list.append(
            {
                "Disabled": False,
                "Factor": factor,
                "Sign": sign,
                "Value": round(value, 4),
                "Relative Factor": "Mkt-RF",
            }
        )

    views = pd.DataFrame(views_list)

    # Assuming rp.factors_views is correctly defined in your context to process views into P_f and Q_f
    P_f, Q_f = rp.factors_views(views, loadings, const=False)

    return P_f, Q_f


def get_stocks_in_sector(sector: str, asset_classes: pd.DataFrame) -> pd.Series:
    """
    Retrieves the list of stocks within a specific sector from the asset classes DataFrame.

    Args:
        sector (str): The sector to filter by.
        asset_classes (pd.DataFrame): DataFrame containing asset class information

    Returns:
        pd.Series: A Series containing the stock codes within the specified sector.
    """
    assets_in_sec = asset_classes[asset_classes["Sector"] == sector]["Code"]
    # print(f"Assets in {sector}: {assets_in_sec.shape[0]}")
    return assets_in_sec
