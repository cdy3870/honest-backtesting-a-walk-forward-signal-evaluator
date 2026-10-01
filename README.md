# Honest Backtesting: A Walk-Forward Signal Evaluator

Build a backtester whose job is to talk you out of your own signal. Start with causal features and a look-ahead detector, replace k-fold with purged walk-forward splits, charge realistic transaction costs, then test the result against shuffled returns and deflate it for every variant you tried. The output is a verdict you can defend.

## How to run

```bash
python scaffold.py
```

## Steps

- [x] **1.** to_log_returns
- [x] **2.** rolling_zscore
- [x] **3.** momentum_feature
- [x] **4.** has_lookahead
- [x] **5.** audit_features
- [x] **6.** purged_walk_forward_splits
- [x] **7.** fit_ols
- [x] **8.** predict_ols
- [x] **9.** r2_score
- [x] **10.** information_coefficient
- [x] **11.** sharpe_ratio

---

Built on Deep-ML.
