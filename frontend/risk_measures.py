import pandas as pd

MODELS = [
    "Classic",
    # "Classic 3-Factors",
    # "Classic 5-Factors",
    "Black-Litterman",
    "Black-Litterman 3-Factors",
    "Black-Litterman 5-Factors",
    "Risk Parity",
    "Risk Parity 3-Factors",
    # "Risk Parity 5-Factors",
    "Hierarchical Risk Parity",
    "Monte Carlo",
]

rm_names = [
    "Standard Deviation",
    "Mean Absolute Deviation",
    # "Gini Mean Difference",
    "Semi Standard Deviation",
    "First Lower Partial Moment",
    "Second Lower Partial Moment",
    # "Conditional Value at Risk",
    # "Tail Gini",
    "Entropic Value at Risk",
    # "Relativistic Value at Risk",
    "Worst Realization",  # Not for RP
    # "Conditional Value at Risk Range",
    # "Tail Gini Range",
    "Range",  # Not for RP
    "Max Drawdown",  # Not for RP
    "Average Drawdown",  # Not for RP
    "Conditional Drawdown at Risk",
    "Entropic Drawdown at Risk",
    # "Relativistic Drawdown at Risk",
    "Ulcer Index",
]

rmeasures = [
    "MV",
    "MAD",
    # "GMD",
    "MSV",
    "FLPM",
    "SLPM",
    # "CVaR",
    # "TG",
    "EVaR",
    # "RLVaR",
    "WR",
    # "CVRG",
    # "TGRG",
    "RG",
    "MDD",
    "ADD",
    "CDaR",
    "EDaR",
    # "RLDaR",
    "UCI",
]

risk_measures_ = {"name": rm_names, "rm": rmeasures}

risk_measures = pd.DataFrame(risk_measures_).set_index("name")
