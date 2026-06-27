import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import pandas as pd
import numpy as np
import pytest
from detector.features import compute_features
from detector.hmm_detector import RegimeDetector


def sample_prices(n=500):
    rng = np.random.default_rng(42)
    idx = pd.date_range("2020-01-01", periods=n, freq="B")
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, n)))
    return pd.Series(prices, index=idx, name="Close")


def test_compute_features_returns_dataframe():
    prices   = sample_prices()
    features = compute_features(prices)
    assert isinstance(features, pd.DataFrame)
    assert "returns" in features.columns
    assert "vol_20" in features.columns
    assert not features.isnull().all().any()


def test_detector_predict_returns_valid_labels():
    prices  = sample_prices()
    det     = RegimeDetector(n_states=3)
    det.fit(prices)
    labels  = det.predict(prices)
    valid   = {"bull", "bear", "sideways"}
    assert set(labels.unique()).issubset(valid)


def test_detector_predict_same_length_as_input():
    prices  = sample_prices()
    det     = RegimeDetector(n_states=3)
    det.fit(prices)
    labels  = det.predict(prices)
    assert len(labels) == len(prices)


def test_detector_save_load(tmp_path):
    prices  = sample_prices()
    det     = RegimeDetector(n_states=3)
    det.fit(prices)
    path    = str(tmp_path / "model.pkl")
    det.save(path)
    det2    = RegimeDetector.load(path)
    labels1 = det.predict(prices)
    labels2 = det2.predict(prices)
    pd.testing.assert_series_equal(labels1, labels2)
