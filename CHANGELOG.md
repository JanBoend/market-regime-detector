# Changelog

## [1.1.0] - 2025-03-05
- Pre-train HMM on SPY 2005–2024 and commit model weights
- Add `predict` length fix: reindex to full price index with ffill/bfill
- Validate: 2022 = 89% bear, 2021 = 91% bull

## [1.0.1] - 2025-01-18
- Fix warm-up period: features.dropna() was causing index length mismatch
- Add 3-state label mapping by mean return (bear/sideways/bull)

## [1.0.0] - 2024-10-15
- Initial release: Gaussian HMM with 3 hidden states
- Features: daily returns, 20-day volatility, 60-day momentum
- 4 unit tests
