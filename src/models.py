from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer

@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
#We create a function to return a list of model specifications for benchmarking.
def get_models() -> List[ModelSpec]:
    models: List[ModelSpec] = []

    models.append(ModelSpec(
        "LinearRegression",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("model", LinearRegression())
        ])
    ))

    models.append(ModelSpec(
        "Ridge",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("model", Ridge(alpha=1.0, random_state=42))
        ])
    ))

    models.append(ModelSpec(
        "Lasso",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=True)),
            ("model", Lasso(alpha=0.01, max_iter=20000, random_state=42))
        ])
    ))

    models.append(ModelSpec(
        "RandomForest",
        Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("model", RandomForestRegressor(
                n_estimators=600, random_state=42, n_jobs=-1, max_depth=None, min_samples_leaf=1
            ))
        ])
    ))

    return models