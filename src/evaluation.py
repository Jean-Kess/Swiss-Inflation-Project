from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .baselines import naive_predictions, seasonal_naive_predictions

def rmse(y_true, y_pred) -> float:
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

def mae(y_true, y_pred) -> float:
    return float(mean_absolute_error(y_true, y_pred))

def mase(y_true: pd.Series, y_pred: pd.Series, y_naive: pd.Series) -> float:
    """Mean Absolute Scaled Error using the Naive forecast as benchmark.

    MASE < 1 indicates that the model improves on the Naive baseline
    on average over the evaluation window.
    """
    # Align on common index and drop missing values
    common_idx = y_true.index.intersection(y_pred.index).intersection(y_naive.index)
    if len(common_idx) == 0:
        return float("nan")

    y_true_c = y_true.loc[common_idx]
    y_pred_c = y_pred.loc[common_idx]
    y_naive_c = y_naive.loc[common_idx]

    mae_model = mean_absolute_error(y_true_c, y_pred_c)
    mae_naive = mean_absolute_error(y_true_c, y_naive_c)
    if mae_naive == 0:
        return float("nan")
    return float(mae_model / mae_naive)

def backtest_models(X: pd.DataFrame, y: pd.Series, model_specs, splits: int = 5, horizon: int = 1) -> Tuple[pd.DataFrame, Dict[str, pd.Series]]:
    tscv = TimeSeriesSplit(n_splits=splits)
    metrics_rows = []
    oof_preds: Dict[str, pd.Series] = {}

    # Prepare OOF containers
    idx_all = y.index.copy()
    for spec in model_specs:
        oof_preds[spec.name] = pd.Series(index=idx_all, dtype=float)

    # Baselines
    oof_preds["Naive"] = pd.Series(index=idx_all, dtype=float)
    oof_preds["SeasonalNaive"] = pd.Series(index=idx_all, dtype=float)

    for fold, (tr, va) in enumerate(tscv.split(X, y), 1):
        tr_idx, va_idx = X.index[tr], X.index[va]

        X_tr, y_tr = X.loc[tr_idx], y.loc[tr_idx]
        X_va, y_va = X.loc[va_idx], y.loc[va_idx]

        # Fit/predict models
        for spec in model_specs:
            spec.pipeline.fit(X_tr, y_tr)
            pred = spec.pipeline.predict(X_va)
            oof_preds[spec.name].loc[va_idx] = pred

        # Baselines
        oof_preds["Naive"].loc[va_idx] = naive_predictions(y, va_idx, horizon=horizon)
        oof_preds["SeasonalNaive"].loc[va_idx] = seasonal_naive_predictions(y, va_idx, season=12)

    # Compute metrics
    naive_series = oof_preds.get("Naive")
    for name, preds in oof_preds.items():
        valid = preds.dropna()
        common_idx = y.index.intersection(valid.index)
        y_true_common = y.loc[common_idx]
        y_pred_common = valid.loc[common_idx]

        row = {
            "model": name,
            "RMSE": rmse(y_true_common, y_pred_common),
            "MAE": mae(y_true_common, y_pred_common),
        }

        # MASE only defined if Naive predictions are available
        if naive_series is not None:
            row["MASE"] = mase(y_true_common, y_pred_common, naive_series)

        # Always record number of samples last
        row["n_samples"] = len(common_idx)

        metrics_rows.append(row)

    metrics = pd.DataFrame(metrics_rows).sort_values(["RMSE", "MAE"]).reset_index(drop=True)

    # Ensure consistent column order: RMSE, MAE, MASE, n_samples at the end
    desired_cols = ["model", "RMSE", "MAE", "MASE", "n_samples"]
    existing_cols = [c for c in desired_cols if c in metrics.columns]
    metrics = metrics[existing_cols]
    return metrics, oof_preds

def save_oof_plot(y: pd.Series, preds: pd.Series, title: str, outfile):
    common = y.dropna().to_frame("y").join(preds.rename("yhat"), how="inner")
    plt.figure(figsize=(10,4))
    plt.plot(common.index, common["y"], label="Actual", lw=2)
    plt.plot(common.index, common["yhat"], label="OOF Pred", lw=1)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outfile, dpi=150)
    plt.close()

def save_oof_csv(y: pd.Series, preds: pd.Series, outfile):
    common = y.dropna().to_frame("y").join(preds.rename("yhat"), how="inner")
    common.to_csv(outfile, index=True)