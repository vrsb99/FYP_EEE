import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import riskfolio as rp
import numpy as np
import riskfolio.src.RiskFunctions as rk
from risk_measures import rmeasures, rm_names, MODELS


def plot_pie(w: pd.DataFrame, title: str | None = None):
    ax = rp.plot_pie(
        w=w,
        title=title,
        others=0.05,
        nrow=25,
        cmap="tab20",
        height=6 * 1.5,
        width=10 * 1.5,
        ax=None,
    )

    return ax


DEBUG = True

if DEBUG:
    MODELS = [
        "Classic",
        "Risk Parity",
        "Risk Parity 3-Factors",
        "Hierarchical Risk Parity",
    ]

    rm_names = [
        "Standard Deviation",
        # "Mean Absolute Deviation",
    ]


def plot_efficient_frontier(year: int, quarter: int) -> None:
    """
    Plots the efficient frontier for a given year and quarter.

    Args:
        year (int): Year for which the efficient frontier is plotted.
        quarter (int): Quarter for which the efficient frontier is plotted.
    Returns:
        None
    """

    model_markers = {
        "Classic": "o",
        "Classic 3-Factors": "s",
        "Black-Litterman": "^",
        "Black-Litterman 3-Factors": "D",
        "Black-Litterman 5-Factors": "p",
        "Risk Parity": "<",
        "Risk Parity 3-Factors": ">",
        "Hierarchical Risk Parity": "H",
        "Monte Carlo": "*",
    }

    risk_measure_colors = {
        "Standard Deviation": "blue",
        "Mean Absolute Deviation": "green",
        "Semi Standard Deviation": "red",
        "First Lower Partial Moment": "cyan",
        "Second Lower Partial Moment": "magenta",
        "Entropic Value at Risk": "yellow",
        "Worst Realization": "black",
        "Range": "#a52a2a",  # Brown
        "Max Drawdown": "#ff8c00",  # DarkOrange
        "Average Drawdown": "#adff2f",  # GreenYellow
        "Conditional Drawdown at Risk": "#ff1493",  # DeepPink
        "Entropic Drawdown at Risk": "#20b2aa",  # LightSeaGreen
        "Ulcer Index": "#778899",  # LightSlateGray
    }

    portfolios = {}
    for model in MODELS:
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
                f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
            )

    ws = pd.concat(portfolios, axis=1)

    model = "Frontier"
    risk_measure = "Standard Deviation"
    frontier_pd = pd.read_parquet(
        f"../data/studying_models/port_weights/{model}_{risk_measure}_{year}_Q{quarter}.parquet"
    )
    returns = pd.read_parquet(
        f"../data/studying_models/input_returns/{year}_Q{quarter}.parquet"
    )
    # port.assets_stats(method_mu=method_mu, method_cov=method_cov)
    cov = returns.cov()
    mu = returns.mean().to_frame().T

    # Plotting the Efficient Frontier
    ax = rp.plot_frontier(
        w_frontier=frontier_pd,
        mu=mu,
        cov=cov,
        returns=returns,
        alpha=0.05,
        cmap="viridis",
        s=16,
        c="black",
        marker="*",
        label="Efficient Frontier",
        t_factor=63,
    )

    for (model, risk_measure), weights in portfolios.items():
        # Calculate expected return and volatility for each portfolio
        weights = weights.squeeze()
        weights = weights.reindex(returns.columns, fill_value=0)
        port_returns = returns.dot(weights)

        expected_return = port_returns.mean() * 63
        volatility = port_returns.std() * np.sqrt(63)

        model_marker = model_markers.get(model, "o")
        color = risk_measure_colors.get(risk_measure, "grey")

        plt.scatter(
            volatility,
            expected_return,
            marker=model_marker,
            color=color,
            label=f"{model}, {risk_measure}",
            s=50,
        )

    plt.title("Efficient Frontier for Year {} Quarter {}".format(year, quarter))
    plt.xlabel("Volatility (Standard Deviation)")
    plt.ylabel("Expected Return")

    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.1),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    plt.grid(True)
    plt.tight_layout(rect=[0, 0.1, 1, 1])

    plt.show()


def plot_frontier(
    w_frontier,
    mu,
    cov=None,
    returns=None,
    rm="MV",
    kelly=False,
    rf=0,
    alpha=0.05,
    a_sim=100,
    beta=None,
    b_sim=None,
    kappa=0.30,
    solver="CLARABEL",
    cmap="viridis",
    w=None,
    label="Portfolio",
    marker="*",
    s=16,
    c="r",
    height=6,
    width=10,
    t_factor=252,
    ax=None,
):
    r"""
    Creates a plot of the efficient frontier for a risk measure specified by
    the user.

    Parameters
    ----------
    w_frontier : DataFrame
        Portfolio weights of some points in the efficient frontier.
    mu : DataFrame of shape (1, n_assets)
        Vector of expected returns, where n_assets is the number of assets.
    cov : DataFrame of shape (n_features, n_features)
        Covariance matrix, where n_features is the number of features.
    returns : DataFrame of shape (n_samples, n_features)
        Features matrix, where n_samples is the number of samples and
        n_features is the number of features.
    rm : str, optional
        The risk measure used to estimate the frontier.
        The default is 'MV'. Possible values are:

        - 'MV': Standard Deviation.
        - 'KT': Square Root Kurtosis.
        - 'MAD': Mean Absolute Deviation.
        - 'MSV': Semi Standard Deviation.
        - 'SKT': Square Root Semi Kurtosis.
        - 'FLPM': First Lower Partial Moment (Omega Ratio).
        - 'SLPM': Second Lower Partial Moment (Sortino Ratio).
        - 'CVaR': Conditional Value at Risk.
        - 'TG': Tail Gini.
        - 'EVaR': Entropic Value at Risk.
        - 'RLVaR': Relativistic Value at Risk.
        - 'WR': Worst Realization (Minimax).
        - 'CVRG': CVaR range of returns.
        - 'TGRG': Tail Gini range of returns.
        - 'RG': Range of returns.
        - 'MDD': Maximum Drawdown of uncompounded returns (Calmar Ratio).
        - 'ADD': Average Drawdown of uncompounded cumulative returns.
        - 'DaR': Drawdown at Risk of uncompounded cumulative returns.
        - 'CDaR': Conditional Drawdown at Risk of uncompounded cumulative returns.
        - 'EDaR': Entropic Drawdown at Risk of uncompounded cumulative returns.
        - 'RLDaR': Relativistic Drawdown at Risk of uncompounded cumulative returns.
        - 'UCI': Ulcer Index of uncompounded cumulative returns.

    kelly : bool, optional
        Method used to calculate mean return. Possible values are False for
        arithmetic mean return and True for mean logarithmic return. The default
        is False.
    rf : float, optional
        Risk free rate or minimum acceptable return. The default is 0.
    alpha : float, optional
        Significance level of VaR, CVaR, EVaR, RLVaR, DaR, CDaR, EDaR, RLDaR and Tail Gini of losses.
        The default is 0.05.
    a_sim : float, optional
        Number of CVaRs used to approximate Tail Gini of losses. The default is 100.
    beta : float, optional
        Significance level of CVaR and Tail Gini of gains. If None it duplicates alpha value.
        The default is None.
    b_sim : float, optional
        Number of CVaRs used to approximate Tail Gini of gains. If None it duplicates a_sim value.
        The default is None.
    kappa : float, optional
        Deformation parameter of RLVaR and RLDaR, must be between 0 and 1. The default is 0.30.
    solver: str, optional
        Solver available for CVXPY that supports power cone programming. Used to calculate RLVaR and RLDaR.
        The default value is 'CLARABEL'.
    cmap : cmap, optional
        Colorscale that represents the risk adjusted return ratio.
        The default is 'viridis'.
    w : DataFrame of shape (n_assets, 1), optional
        A portfolio specified by the user. The default is None.
    label : str or list, optional
        Name or list of names of portfolios that appear on plot legend.
        The default is 'Portfolio'.
    marker : str, optional
        Marker of w. The default is "*".
    s : float, optional
        Size of marker. The default is 16.
    c : str, optional
        Color of marker. The default is 'r'.
    height : float, optional
        Height of the image in inches. The default is 6.
    width : float, optional
        Width of the image in inches. The default is 10.
    t_factor : float, optional
        Factor used to annualize expected return and expected risks for
        risk measures based on returns (not drawdowns). The default is 252.

        .. math::

            \begin{align}
            \text{Annualized Return} & = \text{Return} \, \times \, \text{t_factor} \\
            \text{Annualized Risk} & = \text{Risk} \, \times \, \sqrt{\text{t_factor}}
            \end{align}

    ax : matplotlib axis, optional
        If provided, plot on this axis. The default is None.

    Raises
    ------
    ValueError
        When the value cannot be calculated.

    Returns
    -------
    ax : matplotlib Axes
        Returns the Axes object with the plot for further tweaking.

    Example
    -------
    ::

        label = 'Max Risk Adjusted Return Portfolio'
        mu = port.mu
        cov = port.cov
        returns = port.returns

        ax = rp.plot_frontier(w_frontier=ws,
                              mu=mu,
                              cov=cov,
                              returns=returns,
                              rm=rm,
                              rf=0,
                              alpha=0.05,
                              cmap='viridis',
                              w=w1,
                              label=label,
                              marker='*',
                              s=16,
                              c='r',
                              height=6,
                              width=10,
                              t_factor=252,
                              ax=None)

    .. image:: images/MSV_Frontier.png


    """

    if not isinstance(w_frontier, pd.DataFrame):
        raise ValueError("w_frontier must be a DataFrame")

    if not isinstance(mu, pd.DataFrame):
        raise ValueError("mu must be a DataFrame")

    if not isinstance(cov, pd.DataFrame):
        raise ValueError("cov must be a DataFrame")

    if not isinstance(returns, pd.DataFrame):
        raise ValueError("returns must be a DataFrame")

    if returns.shape[1] != w_frontier.shape[0]:
        a1 = str(returns.shape)
        a2 = str(w_frontier.shape)
        raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")

    if w is not None:
        if not isinstance(w, pd.DataFrame):
            raise ValueError("w must be a DataFrame")

        if w.shape[1] > 1 and w.shape[0] == 1:
            w = w.T

        if returns.shape[1] != w.shape[0]:
            a1 = str(returns.shape)
            a2 = str(w.shape)
            raise ValueError("shapes " + a1 + " and " + a2 + " not aligned")

    if beta is None:
        beta = alpha
    if b_sim is None:
        b_sim = a_sim

    if ax is None:
        fig = plt.gcf()
        ax = fig.gca()
        fig.set_figwidth(width)
        fig.set_figheight(height)
    else:
        fig = ax.get_figure()

    mu_ = np.array(mu, ndmin=2)

    if kelly == False:
        ax.set_ylabel("Expected Arithmetic Return")
    elif kelly == True:
        ax.set_ylabel("Expected Logarithmic Return")

    item = rmeasures.index(rm)
    if rm in ["CVaR", "TG", "EVaR", "RLVaR", "CVRG", "TGRG", "CDaR", "EDaR", "RLDaR"]:
        x_label = (
            rm_names[item] + " (" + rm + ")" + " $\\alpha = $" + "{0:.2%}".format(alpha)
        )
    else:
        x_label = rm_names[item] + " (" + rm + ")"
    if rm in ["CVRG", "TGRG"]:
        x_label += ", $\\beta = $" + "{0:.2%}".format(beta)
    if rm in ["RLVaR", "RLDaR"]:
        x_label += ", $\\kappa = $" + "{0:.2}".format(kappa)
    ax.set_xlabel("Expected Risk - " + x_label)

    title = "Efficient Frontier Mean - " + x_label
    ax.set_title(title)

    X1 = []
    Y1 = []
    Z1 = []

    for i in range(w_frontier.shape[1]):
        try:
            weights = np.array(w_frontier.iloc[:, i], ndmin=2).T
            risk = rk.Sharpe_Risk(
                weights,
                cov=cov,
                returns=returns,
                rm=rm,
                rf=rf,
                alpha=alpha,
                a_sim=a_sim,
                beta=beta,
                b_sim=b_sim,
                kappa=kappa,
                solver=solver,
            )

            if kelly == False:
                ret = mu_ @ weights
            elif kelly == True:
                ret = 1 / returns.shape[0] * np.sum(np.log(1 + returns @ weights))
            ret = ret.item() * t_factor

            if rm not in ["MDD", "ADD", "CDaR", "EDaR", "RLDaR", "UCI"]:
                risk = risk * t_factor**0.5

            ratio = (ret - rf) / risk

            X1.append(risk)
            Y1.append(ret)
            Z1.append(ratio)
        except:
            pass

    ax1 = ax.scatter(X1, Y1, c=Z1, cmap=cmap)

    if w is not None:
        if isinstance(label, str):
            label = [label]

        if label is None:
            label = w.columns.tolist()

        if w.shape[1] != len(label):
            label = w.columns.tolist()

        label = [
            v + " " + str(label[:i].count(v) + 1) if label.count(v) > 1 else v
            for i, v in enumerate(label)
        ]

        if isinstance(c, str):
            colormap = np.array(colors.to_rgba(c)).reshape(1, -1)
        elif c is None:
            colormap = np.array(colors.to_rgba("red")).reshape(1, -1)

        elif isinstance(c, list):
            colormap = [list(colors.to_rgba(i)) for i in c]
            colormap = np.array(colormap)

        if len(label) != colormap.shape[0]:
            colormap = cm.get_cmap("tab20")
            colormap = colormap(np.linspace(0, 1, 20))
            colormap = np.vstack(
                [colormap[6:8], colormap[2:6], colormap[8:], colormap[0:2]]
            )

        n_repeats = int(len(label) // 20 + 1)
        if n_repeats > 1:
            colormap = np.vstack([colormap] * n_repeats)

        for i in range(w.shape[1]):
            weights = w.iloc[:, i].to_numpy().reshape(-1, 1)
            risk = rk.Sharpe_Risk(
                weights,
                cov=cov,
                returns=returns,
                rm=rm,
                rf=rf,
                alpha=alpha,
                a_sim=a_sim,
                beta=beta,
                b_sim=b_sim,
                kappa=kappa,
                solver=solver,
            )
            if kelly == False:
                ret = mu_ @ weights
            elif kelly == True:
                ret = 1 / returns.shape[0] * np.sum(np.log(1 + returns @ weights))
            ret = ret.item() * t_factor

            if rm not in ["MDD", "ADD", "CDaR", "EDaR", "RLDaR", "UCI"]:
                risk = risk * t_factor**0.5

            color = colormap[i].reshape(1, -1)
            ax.scatter(risk, ret, marker=marker, s=s**2, c=color, label=label[i])

        ax.legend(loc="upper left")
        # Hide legend
        # ax.legend().set_visible(False)

    xmin = np.min(X1) - np.abs(np.max(X1) - np.min(X1)) * 0.1
    xmax = np.max(X1) + np.abs(np.max(X1) - np.min(X1)) * 0.1
    ymin = np.min(Y1) - np.abs(np.max(Y1) - np.min(Y1)) * 0.1
    ymax = np.max(Y1) + np.abs(np.max(Y1) - np.min(Y1)) * 0.1

    ax.set_ylim(ymin, ymax)
    ax.set_xlim(xmin, xmax)

    ax.xaxis.set_major_locator(plt.AutoLocator())

    ticks_loc = ax.get_yticks().tolist()
    ax.set_yticks(ax.get_yticks().tolist())
    ax.set_yticklabels(["{:.2%}".format(x) for x in ticks_loc])
    ticks_loc = ax.get_xticks().tolist()
    ax.set_xticks(ax.get_xticks().tolist())
    ax.set_xticklabels(["{:.2%}".format(x) for x in ticks_loc])

    ax.tick_params(axis="y", direction="in")
    ax.tick_params(axis="x", direction="in")

    ax.grid(linestyle=":")

    # colorbar = ax.figure.colorbar(ax1)
    # colorbar.set_label("Risk Adjusted Return Ratio")

    try:
        fig.tight_layout()
    except:
        pass

    return ax, ax1
