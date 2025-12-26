from __future__ import annotations
import pandas as pd
from typing import List

def make_lagged_features(df: pd.DataFrame, lags: List[int], rolling_windows: List[int], forecast_horizon: int = 1) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    if "cpi_yoy" not in df.columns:
        raise ValueError("cpi_yoy column is required")

    feats = pd.DataFrame(index=df.index)
    # lags for all variables
    for col in df.columns:
        for L in lags:
            feats[f"{col}_lag{L}"] = df[col].shift(L)

    # rolling means for CPI only (based on past)
    for w in rolling_windows:
        feats[f"cpi_yoy_roll{w}"] = df["cpi_yoy"].shift(1).rolling(window=w, min_periods=w).mean()

    # target: direct h-step ahead (h=forecast_horizon). For h=1, it's next-month.
    y = df["cpi_yoy"].shift(-forecast_horizon).copy()

    # align and drop rows only where target is NA; allow NaNs in X (imputed in pipelines)
    data = feats.join(y.rename("target"), how="inner")
    data = data.dropna(subset=["target"])  # keep rows even if features have NaNs
    X = data.drop(columns=["target"]).dropna(axis=1, how="all")  # drop all-NaN columns
    y = data["target"]
    return X, y, list(X.columns)