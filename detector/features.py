import pandas as pd
import numpy as np


def compute_features(prices: pd.Series, vol_window: int = 20,
                     momentum_window: int = 60) -> pd.DataFrame:
    """
    Feature engineering for regime detection.
      returns      — daily log returns
      vol_20       — 20-day rolling annualised volatility
      vol_60       — 60-day rolling annualised volatility
      momentum_60  — 60-day log return
      vol_ratio    — short-term vol / long-term vol
    """
    df = pd.DataFrame(index=prices.index)
    df["returns"]     = np.log(prices / prices.shift(1))
    df["vol_20"]      = df["returns"].rolling(vol_window).std() * np.sqrt(252)
    df["vol_60"]      = df["returns"].rolling(momentum_window).std() * np.sqrt(252)
    df["momentum_60"] = np.log(prices / prices.shift(momentum_window))
    df["vol_ratio"]   = df["vol_20"] / df["vol_60"]
    return df.dropna()
