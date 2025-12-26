from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Sequence, Tuple, List

def naive_predictions(y: pd.Series, val_index: pd.Index, horizon: int = 1) -> pd.Series:
    # persistence baseline for direct h-step target: y_hat(s) = y(s-h)
    pred = y.shift(horizon).reindex(val_index)
    return pred

# seasonal naive for direct h-step target: uses last year's same month
def seasonal_naive_predictions(y: pd.Series, val_index: pd.Index, season: int = 12) -> pd.Series:
    pred = y.shift(season).reindex(val_index)
    return pred
