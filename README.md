# market-regime-detector

[![CI](https://github.com/JanBoend/market-regime-detector/actions/workflows/ci.yml/badge.svg)](https://github.com/JanBoend/market-regime-detector/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.10+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

A Gaussian HMM that labels market conditions bull/bear/sideways. Meant to sit in front of a strategy as an optional filter, not as a strategy on its own.

Features: daily log returns, 20-day rolling volatility, 60-day momentum, short/long vol ratio. A 3-state HMM is trained on these; states get mapped to bull/bear/sideways by ordering mean return.

## Try it

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

## Does it actually work

Checked against known periods, trained on SPY 2005–2024:

| Period | Known | Model |
|---|---|---|
| 2017 | Bull | bull ✓ |
| Mar 2020 crash | Bear | bear ✓ |
| 2021 | Bull | bull ✓ |
| 2022 | Bear | bear ✓ |
| 2023–2024 | Bull | bull ✓ |

Tested as a filter on a diversified multi-strategy portfolio and the improvement was marginal, so it ships disabled by default. The edge lives in the strategies, this is a nice-to-have.

## Train your own

```python
import yfinance as yf
from detector.hmm_detector import RegimeDetector

spy = yf.download("SPY", start="2010-01-01", end="2024-12-31")["Close"].squeeze()
det = RegimeDetector(n_states=3)
det.fit(spy)
det.save("models/my_model.pkl")
```

## Wiring it into a strategy

```python
from detector.hmm_detector import RegimeDetector

detector = RegimeDetector.load("models/hmm_spy.pkl")
regime   = detector.predict(prices)  # pd.Series with 'bull'/'bear'/'sideways'
# only enter when regime == 'bull'
```

## Tests

```bash
pip install -r requirements.txt
pytest tests/ -v
```

4 tests: feature computation, valid label set, output length, save/load round-trip.
