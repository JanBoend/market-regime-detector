"""
Gaussian HMM for market regime detection.
3 states mapped to bull/bear/sideways by mean daily return — highest = bull, lowest = bear.
"""
import pickle
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from detector.features import compute_features


class RegimeDetector:

    def __init__(self, n_states: int = 3, n_iter: int = 200, random_state: int = 42):
        self.n_states   = n_states
        self.model      = GaussianHMM(
            n_components=n_states, covariance_type="full",
            n_iter=n_iter, random_state=random_state
        )
        self._label_map = {}

    def fit(self, prices: pd.Series) -> "RegimeDetector":
        features = compute_features(prices)
        X        = features[["returns", "vol_20", "momentum_60"]].values
        self.model.fit(X)

        states      = self.model.predict(X)
        state_means = {s: features["returns"].values[states == s].mean()
                       for s in range(self.n_states)}
        sorted_states = sorted(state_means, key=state_means.get)
        labels = ["bear", "sideways", "bull"] if self.n_states == 3 else \
                 ["bear"] + ["sideways"] * (self.n_states - 2) + ["bull"]
        self._label_map = {sorted_states[i]: labels[i] for i in range(self.n_states)}
        return self

    def predict(self, prices: pd.Series) -> pd.Series:
        features = compute_features(prices)
        X        = features[["returns", "vol_20", "momentum_60"]].values
        states   = self.model.predict(X)
        labels   = pd.Series(
            [self._label_map[s] for s in states],
            index=features.index,
            name="regime"
        )
        # Reindex to full price index; forward-fill warm-up NaNs then back-fill
        return labels.reindex(prices.index).ffill().bfill()

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump({
                "model":     self.model,
                "label_map": self._label_map,
                "n_states":  self.n_states,
            }, f)

    @classmethod
    def load(cls, path: str) -> "RegimeDetector":
        with open(path, "rb") as f:
            d = pickle.load(f)
        obj = cls(n_states=d["n_states"])
        obj.model      = d["model"]
        obj._label_map = d["label_map"]
        return obj
