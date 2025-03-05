# market-regime-detector

![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg)

Gaussian HMM-based market regime classifier. Labels market conditions as bull, bear, or sideways. Designed to plug into systematic trading engines as an optional filter.

## How it works

Features used: daily log returns, 20-day rolling volatility, 60-day momentum, vol ratio (short-term/long-term vol).

A 3-state Gaussian HMM is trained on these features. States are mapped to bull/bear/sideways by ordering their mean return (highest return state = bull, lowest = bear).

## Quick start

```python
from detector.hmm_detector import RegimeDetector

det    = RegimeDetector.load("models/hmm_spy.pkl")
labels = det.predict(spy_prices)

print(labels.tail())
# Date
# 2024-12-20    bull
# 2024-12-23    sideways
# 2024-12-24    bull
```

## Validation

Known market periods vs model predictions (trained on SPY 2005–2024):

| Period | Known | Model |
|---|---|---|
| 2017 | Bull | bull ✓ |
| Mar 2020 crash | Bear | bear ✓ |
| 2021 | Bull | bull ✓ |
| 2022 | Bear | bear ✓ |
| 2023–2024 | Bull | bull ✓ |

## Honest results

Tested as an optional filter on the ICM strategy portfolio: adding the regime filter produced a marginal Sharpe improvement (+0.02). Kept as an optional feature with `regime_filter=False` as the production default. The edge is in the strategies, not the filter.

## Train your own model

```python
import yfinance as yf
from detector.hmm_detector import RegimeDetector

spy = yf.download("SPY", start="2010-01-01", end="2024-12-31")["Close"].squeeze()
det = RegimeDetector(n_states=3)
det.fit(spy)
det.save("models/my_model.pkl")
```

## Plug into quant-engine

```python
from detector.hmm_detector import RegimeDetector

detector = RegimeDetector.load("models/hmm_spy.pkl")
regime   = detector.predict(prices)  # pd.Series with 'bull'/'bear'/'sideways'
# Filter trades: only enter when regime == 'bull'
```

## Running tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

4 tests: feature computation, valid labels, correct length, save/load round-trip.
