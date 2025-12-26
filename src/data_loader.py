from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path
import csv
from typing import Optional, Tuple, List

DATE_ALIASES = ["date", "month", "time", "period", "time_period"]

def _read_csv_smart(path: Path) -> pd.DataFrame:
    # First, detect if the file contains a header line like SNB exports ("Date";"D0";"Value")
    lines: list[str] = []
    try:
        with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
            lines = [next(f) for _ in range(60)]
    except StopIteration:
        pass
    except Exception:
        lines = []

    header_row = None
    if lines:
        for i, ln in enumerate(lines):
            low = ln.strip().lower()
            if ("date" in low) and ("value" in low):
                header_row = i
                break

    if header_row is not None:
        # Parse as SNB-like with semicolon; keep the header line as the first row
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(
                    path,
                    sep=";",
                    engine="python",
                    skiprows=header_row,  # leave header line as first row after skipping
                    header=0,
                    na_values=["", "NA", "NaN", "-", ".."],
                    on_bad_lines="skip",
                    encoding=enc,
                    quotechar='"',
                    doublequote=True,
                    skip_blank_lines=True,
                )
                if df.shape[1] >= 2:
                    return df
            except Exception:
                continue

    # Fallbacks: try common CSV variants with robust settings
    attempts = [
        {"sep": ",", "decimal": "."},
        {"sep": ";", "decimal": ","},
        {"sep": ";", "decimal": "."},
        {"sep": ",", "decimal": ","},
    ]

    for opts in attempts:
        for enc in ("utf-8-sig", "latin-1"):
            try:
                df = pd.read_csv(
                    path,
                    sep=opts["sep"],
                    decimal=opts["decimal"],
                    engine="python",
                    comment="#",
                    na_values=["", "NA", "NaN", "-", ".."],
                    encoding=enc,
                    on_bad_lines="skip",
                    quotechar='"',
                    doublequote=True,
                    escapechar="\\",
                )
                if df.shape[1] >= 2:
                    return df
            except Exception:
                # Try again with quoting disabled (treat quotes as regular chars)
                try:
                    df = pd.read_csv(
                        path,
                        sep=opts["sep"],
                        decimal=opts["decimal"],
                        engine="python",
                        comment="#",
                        na_values=["", "NA", "NaN", "-", ".."],
                        encoding=enc,
                        on_bad_lines="skip",
                        quoting=csv.QUOTE_NONE,
                        escapechar="\\",
                    )
                    if df.shape[1] >= 2:
                        return df
                except Exception:
                    continue

    # Last resort: read with python engine and skip bad lines
    return pd.read_csv(path, engine="python", on_bad_lines="skip", encoding="utf-8-sig")

def _find_date_column(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if str(c).strip().lower() in DATE_ALIASES:
            return c
    # try first column if datetime-like strings
    c0 = df.columns[0]
    try:
        pd.to_datetime(df[c0], errors="raise")
        return c0
    except Exception:
        return None

def _coerce_numeric_columns(df: pd.DataFrame, exclude: List[str]) -> pd.DataFrame:
    for c in df.columns:
        if c in exclude:
            continue
        if df[c].dtype.kind not in "biufc":
            df[c] = (
                df[c]
                .astype(str)
                .str.replace('"', "", regex=False)
                .str.replace("'", "", regex=False)
                .str.replace("\u202f", "", regex=False)  # thin space
                .str.replace(" ", "", regex=False)
            )
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def load_all_series(data_dir: Path) -> pd.DataFrame:
    data_dir = Path(data_dir)
    all_files = sorted([p for p in data_dir.glob("*.csv")])
    if not all_files:
        raise FileNotFoundError(f"No CSV files found in {data_dir}")

    merged: Optional[pd.DataFrame] = None

    for path in all_files:
        df = _read_csv_smart(path)
        # drop empty columns
        df = df.loc[:, ~df.columns.astype(str).str.fullmatch(r"\s*")]
        date_col = _find_date_column(df)
        if date_col is None:
            # If a single column file, skip
            if df.shape[1] < 2:
                continue
            date_col = df.columns[0]

        df = df.rename(columns={date_col: "date"})
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df = df.dropna(subset=["date"])
        df["date"] = df["date"].dt.to_period("M").dt.to_timestamp()

        # Keep only numeric columns (after coercion); exclude columns that are entirely NaN
        df = _coerce_numeric_columns(df, exclude=["date"])
        num_cols = [
            c for c in df.columns
            if c != "date" and df[c].dtype.kind in "biufc" and df[c].notna().any()
        ]
        if not num_cols:
            continue

        # CPI naming heuristic
        stem = path.stem.lower()
        if "cpi" in stem or "consumer" in stem:
            # take the first numeric column as CPI YoY
            cpi_col = num_cols[0]
            df = df[["date", cpi_col]].rename(columns={cpi_col: "cpi_yoy"})
        else:
            # prefix columns by file stem
            rename_map = {c: f"{path.stem}_{c}" for c in num_cols}
            df = df[["date"] + num_cols].rename(columns=rename_map)

        df = df.set_index("date").sort_index()

        merged = df if merged is None else merged.join(df, how="outer")

    if merged is None or "cpi_yoy" not in merged.columns:
        raise ValueError("Could not find CPI series. Ensure a CSV with 'cpi' in its filename exists and contains a numeric column.")

    merged = merged.sort_index()
    return merged