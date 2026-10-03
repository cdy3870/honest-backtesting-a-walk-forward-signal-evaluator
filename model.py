"""
Honest Backtesting: A Walk-Forward Signal Evaluator

Assembled from your step-by-step solutions.
"""

import numpy as np

# Step 1 - to_log_returns
import numpy as np

def to_log_returns(prices):
    # TODO: return ln(p_t) - ln(p_{t-1}) as an array of length len(prices) - 1
    
    result = np.empty_like(prices)
    result[:1] = 0
    result[1:] = np.array(prices)[:-1]


    return (np.log(np.array(prices)) - np.log(result))[1:]

# Step 2 - rolling_zscore
import numpy as np

def rolling_zscore(x, window):
    # TODO: causal rolling z-score; first window-1 entries are np.nan
    # TODO: a window with zero standard deviation scores 0.0
    
    z = np.full((len(x),), np.nan)
    
    for i in range(window - 1, len(x), 1):
        if x[i - window + 1: i + 1].std() == 0:
            z[i] = 0
        else:
            z[i] = (x[i] - x[i - window + 1: i + 1].mean()) / x[i - window + 1: i + 1].std()


    return z

# Step 3 - momentum_feature
import numpy as np

def momentum_feature(prices, lookback):
    # TODO: ln(p_t / p_{t-lookback}); entries before index lookback are np.nan
    
    prices = np.array(prices)

    z = np.full((len(prices),), np.nan)
    
    # for i in range(lookback - 1, len(prices), 1):

    #     z[i] = np.log((prices[i]) / (prices[i - lookback]))

    z[lookback:] = np.log(prices[lookback:] / prices[:-lookback])

    return z

# Step 4 - has_lookahead
import numpy as np

def has_lookahead(feature_fn, x):
    # TODO: perturb only the last element of a copy of x
    # TODO: return True if any earlier output value changed (NaN counts as equal to NaN)
    
    base = feature_fn(x)
    perturbed = x.copy(); 
    perturbed[-1] += 1000.0
    changed = not np.allclose(base[:-1], feature_fn(perturbed)[:-1], equal_nan=True)

    return changed

# Step 5 - audit_features
import numpy as np

def audit_features(feature_fns, x):
    # TODO: return {name: has_lookahead(fn, x)} for every entry
    d = {}
    for k, v in feature_fns.items():
        d[k] = has_lookahead(v, x)

    return d

# Step 6 - purged_walk_forward_splits
import numpy as np

def purged_walk_forward_splits(n_samples, n_splits, embargo):
    # TODO: expanding-window splits with an embargo gap before each test block
    
    fold_size = n_samples // (n_splits + 1)

    test_samples = []

    for k in range(1, n_splits, 1):
        test_block = [k * fold_size, (k + 1) * fold_size]
        [0, k * fold_size - embargo]

# Step 7 - fit_ols
import numpy as np

def fit_ols(X, y):
    # TODO: prepend an intercept column and return the least-squares coefficients
    n, p = X.shape[0], X.shape[1]
    X = np.hstack((np.ones((n, 1)), X))

    x, residuals, rank, s = np.linalg.lstsq(X, y)

    return x

# Step 8 - predict_ols
import numpy as np

def predict_ols(X, coef):
    # TODO: intercept is coef[0]; slopes are coef[1:]
    
    return coef[0] + X @ coef[1:]

# Step 9 - r2_score
import numpy as np

def r2_score(y_true, y_pred):
    # TODO: 1 - SS_res / SS_tot, returning 0.0 when SS_tot is 0
    
    SS_res = np.square(y_true - y_pred).sum() 
    SS_tot = np.square(y_true - y_true.mean()).sum()

    if SS_tot == 0:
        return 0.0

    r_squared = 1 - (SS_res/SS_tot)

    return r_squared

# Step 10 - information_coefficient
import numpy as np

def information_coefficient(y_true, y_pred):
    # TODO: Pearson correlation; return 0.0 if either input has zero variance
    den = (np.sqrt(np.square(y_true - y_true.mean()).sum()) * np.sqrt(np.square(y_pred - y_pred.mean()).sum()))
    if den == 0:
        return 0.0
    return ((y_true - y_true.mean()) * (y_pred - y_pred.mean())).sum() / den

# Step 11 - sharpe_ratio
import numpy as np

def sharpe_ratio(returns, periods_per_year):
    # TODO: mean / std(ddof=1) * sqrt(periods_per_year)
    # TODO: return 0.0 for zero volatility or fewer than two observations
    mean = returns.mean()
    if len(returns) == 1:
        return 0.0
    std = np.std(returns, ddof=1) # n - 1, corrects bias when sample used to estimate large
    if std == 0 or periods_per_year == 0:
        return 0.0
    return mean / std * periods_per_year ** 0.5

# Step 13 - positions_from_predictions
import numpy as np
import math

def positions_from_predictions(preds, threshold):
    # TODO: sign(pred) where abs(pred) >= threshold, else 0.0; NaN maps to 0.0
    return np.where(abs(preds) >= threshold, np.sign(preds), 0)

# Step 15 - permutation_pvalue
import numpy as np

def permutation_pvalue(positions, forward_returns, n_permutations, seed):
    # TODO: observed = mean(positions * forward_returns)
    # TODO: permute forward_returns with default_rng(seed), count >= observed
    # TODO: return (count + 1) / (n_permutations + 1)
    observed = (positions * forward_returns).mean()
    rng = np.random.default_rng(seed)

    means = []
    count = 0
    
    for n in range(n_permutations):
        permuted = rng.permutation(forward_returns)
        if (positions * permuted).mean() >= observed:
            count += 1

        
    return (count + 1) / (n_permutations + 1)

# Step 16 - deflate_pvalue
import numpy as np

def deflate_pvalue(p_value, n_trials):
    # TODO: Bonferroni adjustment, capped at 1.0
    
    return min(1.0, p_value * n_trials)

