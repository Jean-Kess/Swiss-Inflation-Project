from __future__ import annotations
from pathlib import Path
import argparse
import os
import random
import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_regression

from src.config import DATA_DIR, RESULTS_DIR, START_DATE, DEFAULT_LAGS, DEFAULT_ROLLING, DEFAULT_SPLITS
from src.data_loader import load_all_series
from src.features import make_lagged_features
from src.models import get_models
from src.evaluation import backtest_models, save_oof_plot, save_oof_csv
from src.diagnostics import generate_permutation_importance, generate_corr_heatmap

def parse_args():
    p = argparse.ArgumentParser(description="Swiss CPI forecasting. Use --months to set forecast horizon.")
    p.add_argument("--months", type=int, default=1, help="Forecast horizon in months (default: 1)")
    p.add_argument("--3months", action="store_true", help="Shortcut to set --months 3")
    return p.parse_args()

MIN_COVERAGE = 0.7  # default coverage threshold for features after lagging/rolling

def main():
    args = parse_args()
    months = 3 if getattr(args, "3months", False) else max(1, int(args.months))
    RESULTS_DIR.mkdir(exist_ok=True)
    suffix = f"_h{months}"

    # Reproducibility: fix seeds and hash behavior
    os.environ["PYTHONHASHSEED"] = "0"
    random.seed(42)
    np.random.seed(42)

    print(f"Loading datasets from: {DATA_DIR}")
    df = load_all_series(DATA_DIR)

    # ensure start date
    df = df.loc[df.index >= pd.Timestamp(START_DATE)]
    print("Columns:", list(df.columns))
    print("Data range:", df.index.min().date(), "->", df.index.max().date(), f"({len(df)} months)")

    # Use defaults from config
    lags = DEFAULT_LAGS
    rolls = DEFAULT_ROLLING

    # Always build features for forecasting; direct horizon = months
    X, y, feat_names = make_lagged_features(df, lags=lags, rolling_windows=rolls, forecast_horizon=months)
    print(f"Feature matrix: {X.shape}, Target: {y.shape}")

    if len(X) == 0:
        print("[ERROR] No samples after feature engineering. Try reducing lags/rolling or check CPI parsing.")
        return

    # --- Feature relevance summary & coverage filtering ---
    coverage = X.notna().mean(axis=0)
    means = X.mean(axis=0, skipna=True)
    stds = X.std(axis=0, skipna=True)
    # Correlation with target (pairwise non-null)
    abs_corr = []
    for col in X.columns:
        try:
            c = X[col].corr(y)
            abs_corr.append(abs(c) if pd.notna(c) else 0.0)
        except Exception:
            abs_corr.append(0.0)
    # Mutual information (impute NaNs with column median for diagnostic only)
    X_mi = X.copy()
    X_mi = X_mi.fillna(X_mi.median())
    try:
        mi = mutual_info_regression(X_mi.values, y.values, random_state=42)
    except Exception:

        mi = np.zeros(X.shape[1])

    feat_summary = pd.DataFrame({
        "feature": X.columns,
        "coverage": coverage.values,
        "mean": means.values,
        "std": stds.values,
        "abs_corr": abs_corr,
        "mutual_info": mi,
    })
    feat_summary_path = RESULTS_DIR / f"feature_relevance{suffix}.csv"
    feat_summary_sorted = feat_summary.sort_values(["coverage", "abs_corr", "mutual_info"], ascending=[False, False, False])
    feat_summary_sorted.to_csv(feat_summary_path, index=False)
    # Apply coverage filter
    before_cols = X.shape[1]
    keep_cols = feat_summary_sorted.loc[feat_summary_sorted["coverage"] >= MIN_COVERAGE, "feature"].tolist()
    drop_cols = [c for c in X.columns if c not in keep_cols]
    X = X[keep_cols]
    after_cols = X.shape[1]
    print(f"Coverage filter: kept {after_cols}/{before_cols} features (min_coverage={MIN_COVERAGE}).")
    # Show it in console summary only 
    if drop_cols:
        print(f"Dropped {len(drop_cols)} low-coverage features.")

    model_specs = get_models()
    metrics, oof = backtest_models(X, y, model_specs, splits=DEFAULT_SPLITS, horizon=months)

    # Save outputs
    out_csv = RESULTS_DIR / f"metrics{suffix}.csv"
    metrics.to_csv(out_csv, index=False)
    print()
    print(metrics)

    # Comparison figures: save two separate images (RMSE+MAE) and (MASE)
    try:
        import matplotlib.pyplot as plt
        plot_df = metrics.sort_values("RMSE").reset_index(drop=True)
        x = np.arange(len(plot_df["model"]))

        has_mase = "MASE" in plot_df.columns

        # Figure 1: RMSE bars + MAE line (same units)
        fig1, ax1 = plt.subplots(figsize=(9, 5))
        ax1.bar(x, plot_df["RMSE"], color="#4C78A8", label="RMSE")
        ax1.plot(x, plot_df["MAE"], color="#F58518", marker="o", label="MAE")
        ax1.set_ylabel("Error")
        ax1.set_title("Model comparison (OOF): RMSE and MAE")
        ax1.set_xticks(x)
        ax1.set_xticklabels(plot_df["model"], rotation=30, ha="right")
        ax1.legend(frameon=False)
        plt.tight_layout()
        comp_errors_png = RESULTS_DIR / f"model_comparison_errors{suffix}.png"
        fig1.savefig(comp_errors_png, dpi=150)
        plt.close(fig1)

        # Figure 2: MASE bars (unitless ratio)
        if has_mase:
            fig2, ax2 = plt.subplots(figsize=(9, 4))
            ax2.bar(x, plot_df["MASE"], color="#54A24B", label="MASE")
            ax2.axhline(1.0, color="black", linewidth=1.0, alpha=0.7)
            ax2.set_ylabel("MASE")
            ax2.set_title("Model comparison (OOF): MASE (relative to Naive)")
            ax2.set_xticks(x)
            ax2.set_xticklabels(plot_df["model"], rotation=30, ha="right")
            ax2.legend(frameon=False)
            plt.tight_layout()
            comp_mase_png = RESULTS_DIR / f"model_comparison_mase{suffix}.png"
            fig2.savefig(comp_mase_png, dpi=150)
            plt.close(fig2)
    except Exception as e:
        print("Warning: could not save comparison figure:", e)

    # Save OOF plots and CSVs for every model
    for name, preds in oof.items():
        png = RESULTS_DIR / f"oof_{name}{suffix}.png"
        csv = RESULTS_DIR / f"oof_{name}{suffix}.csv"
        save_oof_plot(y, preds, f"OOF — {name}", png)
        save_oof_csv(y, preds, csv)
    # Quiet: no path spam for saved OOF artifacts

    # Diagnostics: permutation importance
    generate_permutation_importance(model_specs, metrics, X, y, RESULTS_DIR, suffix)

    # Diagnostics: correlation heatmap
    generate_corr_heatmap(X, y, RESULTS_DIR, suffix)

    # Next-month forecast using the full dataset 
    print(f"\nGenerating {months}-month ahead forecasts for each model...")
    last_X = X.tail(1)
    forecast_rows = []
    for spec in model_specs:
        spec.pipeline.fit(X, y)
        next_pred = float(spec.pipeline.predict(last_X)[0])
        forecast_rows.append({"model": spec.name, "next_month_cpi_yoy_pred": next_pred})
    # Baselines
    naive_next = float(y.iloc[-months]) if len(y) >= months else float(y.iloc[-1])
    forecast_rows.append({"model": "Naive", "next_month_cpi_yoy_pred": naive_next})
    seasonal_next = float(y.iloc[-12]) if len(y) >= 12 else naive_next
    forecast_rows.append({"model": "SeasonalNaive", "next_month_cpi_yoy_pred": seasonal_next})
    df_fc = pd.DataFrame(forecast_rows)
    fc_csv = RESULTS_DIR / f"forecasts{suffix}.csv"
    df_fc.to_csv(fc_csv, index=False)

    # Forecast horizon info: next month after last observed date (from raw df)
    try:
        last_obs = df.index.max()
        target_month = (pd.Period(last_obs, freq="M") + months).to_timestamp()
        print("Forecast horizon:", target_month.strftime("%Y-%m"), f"({months} months after last data)")
    except Exception:
        pass

    # Write a concise forecast report summarizing the rankings
    rep_path = RESULTS_DIR / f"forecast_report{suffix}.txt"
    best_rmse = metrics.sort_values(["RMSE", "MAE"]).iloc[0]
    best_mae = metrics.sort_values(["MAE", "RMSE"]).iloc[0]
    lines = []
    lines.append(f"Forecast Summary (h={months})\n")
    lines.append(f"Data range: {y.index.min().date()} -> {y.index.max().date()}  (n={len(y)})\n")
    lines.append("\nTop by RMSE:\n")
    if "MASE" in metrics.columns:
        lines.append(f"- {best_rmse['model']}: RMSE={best_rmse['RMSE']:.4f}, MAE={best_rmse['MAE']:.4f}, MASE={best_rmse['MASE']:.3f}, n={int(best_rmse['n_samples'])}\n")
    else:
        lines.append(f"- {best_rmse['model']}: RMSE={best_rmse['RMSE']:.4f}, MAE={best_rmse['MAE']:.4f}, n={int(best_rmse['n_samples'])}\n")

    lines.append("Top by MAE:\n")
    if "MASE" in metrics.columns:
        lines.append(f"- {best_mae['model']}: RMSE={best_mae['RMSE']:.4f}, MAE={best_mae['MAE']:.4f}, MASE={best_mae['MASE']:.3f}, n={int(best_mae['n_samples'])}\n")
    else:
        lines.append(f"- {best_mae['model']}: RMSE={best_mae['RMSE']:.4f}, MAE={best_mae['MAE']:.4f}, n={int(best_mae['n_samples'])}\n")
    pred_label = "1-month-ahead predictions" if months == 1 else f"{months}-month-ahead predictions"
    lines.append(f"\n{pred_label}:\n")
    for row in forecast_rows:
        lines.append(f"- {row['model']}: {row['next_month_cpi_yoy_pred']:.4f}\n")
    lines.append("\nFiles:\n")
    lines.append(f"- Metrics: {out_csv}\n")
    lines.append(f"- Forecasts: {fc_csv}\n")
    with open(rep_path, "w", encoding="utf-8") as fh:
        fh.writelines(lines)

if __name__ == "__main__":
    main()