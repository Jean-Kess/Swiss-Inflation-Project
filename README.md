# Swiss Inflation Forecasting: Benchmarks & Predictions

Author: Benoît Guignard  
Date: December 2025

## Research Question
Can regularized regression and ML models outperform Naive benchmarks in forecasting Swiss CPI YoY inflation at 1 and 3 month horizons?

## Summary of Approach
This project builds lagged and rolling features from monthly macro/market series, applies a coverage filter to reduce sparsity, and evaluates models with time aware backtesting. The pipeline always produces an h month ahead forecast (default h=1) and saves reproducible artifacts for analysis.

## Summary of Findings
Across both 1 and 3 month horizons, the ML models don't outperform the Naive persistence baseline in out of fold evaluation. 

## Methods
- Backtesting: `TimeSeriesSplit (k=5)`, no shuffling; OOF RMSE/MAE
- Direct horizon modeling: target is `cpi_yoy` shifted by `-h`
- Features: lags for all columns; CPI rolling means from past data
- Baselines: Naive (persistence), SeasonalNaive (12 month seasonal persistence)
- Models: LinearRegression, Ridge, Lasso, RandomForest

## Project Structure
```
SwissInflationProject/
├─ main.py                         # Entry point: features, backtests, forecast, artifacts
├─ project_report.md               # Report source 
├─ project_report.pdf              # PDF report
├─ README.md					   # Setup & Usage
├─ requirements.txt                # Python dependencies
├─ data/                            # Monthly CSVs (target + predictors)
├─ results/                         # Metrics, OOF, diagnostics, forecasts
└─ src/
   ├─ __init__.py
   ├─ baselines.py                  # Naive & SeasonalNaive baselines
   ├─ config.py                     # Paths and default parameters
   ├─ data_loader.py                # Robust CSV ingestion + CPI target selection
   ├─ evaluation.py                 # TimeSeriesSplit backtest, metrics, OOF artifacts
   ├─ features.py                   # Lag/rolling features and h-step target
   ├─ models.py                     # Model pipelines (imputer, scaler, estimators)
   └─ diagnostics.py                # Permutation importance and correlation heatmap
```

## Setup

Windows (no activation needed):
```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install --upgrade pip
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

macOS/Linux:
```bash
python3 -m venv .venv
./.venv/bin/python -m pip install --upgrade pip
./.venv/bin/python -m pip install -r requirements.txt
```

## Usage
Run the forecast (default horizon is 1 month):

Windows:
```powershell
.\.venv\Scripts\python.exe .\main.py
```

And then run the forecast 3 months ahead:

```powershell
.\.venv\Scripts\python.exe .\main.py --3months
```

macOS/Linux:
```bash
./.venv/bin/python main.py
```

And then run the forecast 3 months ahead:

```bash
./.venv/bin/python main.py --3months
```

This executes: data load → feature build → backtests → diagnostics → h‑ahead forecast.

## Expected Outputs
All artifacts are saved under `results/` 

- Metrics: `metrics_h{m}.csv` — RMSE and MAE for each model/baseline
- OOF predictions per model: `oof_<Model>_h{m}.csv` and `oof_<Model>_h{m}.png`
- Model comparison: `model_comparison_errors_h{m}.png` — RMSE bars + MAE line
- Model comparison: `model_comparison_mase_h{m}.png` — MASE bars (relative to Naive)
- Feature relevance: `feature_relevance_h{m}.csv` — coverage, mean, std, |corr|, mutual information
- Correlation heatmap: `corr_heatmap_h{m}.png` — top 20 features by |corr| with target
- Permutation importance: `perm_importance_<BestModel>_h{m}.csv/.png` — best non baseline only
- Forecast report: `forecast_report_h{m}.txt` — horizon, top models, and h‑ahead predictions
- Forecast values: `forecasts_h{m}.csv` — one line per model/baseline

## Reproducibility Notes
- Python 3.11 recommended, seeds fixed for determinism:
	- `PYTHONHASHSEED=0`, `random.seed(42)`, `numpy.random.seed(42)`
- Plots saved with a non‑GUI backend (`Agg`) to avoid display issues

## Requirements
- Python 3.11
- Core libraries: pandas, numpy, scikit-learn, matplotlib, seaborn, statsmodels, joblib, python-dateutil
- Data sourcing: requests, pytrends

## Next Extensions (Optional)
- Statistical comparison vs baselines (e.g., Diebold–Mariano test)
- Additional horizons (e.g., h=6) and auto‑tuning of coverage threshold
- Expand feature set and data coverage