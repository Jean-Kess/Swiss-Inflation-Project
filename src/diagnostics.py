from __future__ import annotations
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

# Permutation Importance
def generate_permutation_importance(model_specs, metrics: pd.DataFrame, X: pd.DataFrame, y: pd.Series, results_dir: os.PathLike, suffix: str) -> None:
    try:
        from sklearn.inspection import permutation_importance
        # pick best non-baseline model
        non_baselines = metrics[~metrics["model"].isin(["Naive", "SeasonalNaive"])].sort_values(["RMSE", "MAE"])
        if non_baselines.empty:
            return
        best_name = str(non_baselines.iloc[0]["model"])
        best_spec = next((s for s in model_specs if s.name == best_name), None)
        if best_spec is None:
            return
        best_spec.pipeline.fit(X, y)
        pi = permutation_importance(best_spec.pipeline, X, y, scoring="r2", n_repeats=10, random_state=42)
        pi_df = pd.DataFrame({
            "feature": X.columns,
            "importance_mean": pi.importances_mean,
            "importance_std": pi.importances_std,
        }).sort_values("importance_mean", ascending=False)
        pi_csv = os.path.join(results_dir, f"perm_importance_{best_name}{suffix}.csv")
        pi_df.to_csv(pi_csv, index=False)
        # Plot top 20 features
        top_pi = pi_df.head(20)
        plt.figure(figsize=(8, 6))
        plt.barh(top_pi["feature"], top_pi["importance_mean"], color="#4C78A8")
        plt.gca().invert_yaxis()
        plt.xlabel("Permutation importance (Δ R²)")
        plt.title(f"Permutation Importance — {best_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"perm_importance_{best_name}{suffix}.png"), dpi=150)
        plt.close()
    except Exception:
        # silent skip on error as per prior behavior
        pass

# Correlation Heatmap
def generate_corr_heatmap(X: pd.DataFrame, y: pd.Series, results_dir: os.PathLike, suffix: str) -> None:
    try:
        # Compute correlation to target
        corr_to_y = []
        for col in X.columns:
            try:
                c = X[col].corr(y)
            except Exception:
                c = np.nan
            corr_to_y.append((col, c))
        corr_to_y = pd.DataFrame(corr_to_y, columns=["feature", "corr"]).assign(abs_corr=lambda d: d["corr"].abs().fillna(0.0))
        top_cols = corr_to_y.sort_values("abs_corr", ascending=False)["feature"].head(20).tolist()
        if not top_cols:
            return
        corr_mat = X[top_cols].corr()
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_mat, cmap="coolwarm", center=0, xticklabels=True, yticklabels=True)
        plt.title("Feature Correlation Heatmap (Top 20 by |corr with target|)")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"corr_heatmap{suffix}.png"), dpi=150)
        plt.close()
    except Exception:
        # silent skip on error
        pass
