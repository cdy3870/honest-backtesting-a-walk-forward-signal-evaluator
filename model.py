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

