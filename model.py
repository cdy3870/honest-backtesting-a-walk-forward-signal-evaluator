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

