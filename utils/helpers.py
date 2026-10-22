"""
Helper Utilities for Urban Road Analytics System.
"""

import pandas as pd
import numpy as np
import re
from config import DATASET_SIGNATURES

def normalize_column_name(col: str) -> str:
    """Standardize column names to lower_snake_case for internal matching."""
    cleaned = re.sub(r'[^\w\s]', '_', str(col).strip())
    cleaned = re.sub(r'\s+', '_', cleaned)
    cleaned = re.sub(r'_+', '_', cleaned)
    return cleaned.lower()

def detect_dataset_type(df: pd.DataFrame) -> str:
    """
    Automatically detect the category of uploaded dataset
    based on matching known column signatures.
    """
    if df is None or df.empty:
        return "unknown"
    
    col_set = {normalize_column_name(c) for c in df.columns}
    
    # Calculate score for each dataset signature
    scores = {}
    for dtype, sig_cols in DATASET_SIGNATURES.items():
        matches = sum(1 for sc in sig_cols if any(sc in c for c in col_set))
        scores[dtype] = matches / len(sig_cols)
    
    best_match = max(scores, key=scores.get)
    if scores[best_match] >= 0.4:
        return best_match
    return "unknown"

def safe_to_numeric(series: pd.Series, default_val=np.nan) -> pd.Series:
    """Safely convert series to numeric, replacing unparseable values."""
    return pd.to_numeric(series, errors='coerce').fillna(default_val)

def safe_to_datetime(series: pd.Series) -> pd.Series:
    """Safely convert series to datetime."""
    return pd.to_datetime(series, errors='coerce')

def safe_strip_strings(df: pd.DataFrame) -> pd.DataFrame:
    """Strip leading and trailing whitespace from all string columns."""
    df_clean = df.copy()
    for col in df_clean.select_dtypes(include=['object', 'string']).columns:
        df_clean[col] = df_clean[col].astype(str).str.strip()
        # Replace 'nan', 'None', 'null' with actual NaN
        df_clean[col] = df_clean[col].replace({'nan': np.nan, 'None': np.nan, 'null': np.nan, '': np.nan})
    return df_clean

def standardize_yes_no(series: pd.Series) -> pd.Series:
    """Standardize Yes/No categorical values."""
    mapping = {
        'yes': 'Yes', 'y': 'Yes', '1': 'Yes', 'true': 'Yes', 't': 'Yes', 1: 'Yes', True: 'Yes',
        'no': 'No', 'n': 'No', '0': 'No', 'false': 'No', 'f': 'No', 0: 'No', False: 'No'
    }
    return series.astype(str).str.strip().str.lower().map(mapping).fillna('No')

def min_max_scale(series: pd.Series, target_min: float = 0.0, target_max: float = 100.0) -> pd.Series:
    """
    Transparent Min-Max Normalization scaling series between target_min and target_max.
    Handles constant series without division by zero.
    """
    s = pd.to_numeric(series, errors='coerce').fillna(0)
    s_min = s.min()
    s_max = s.max()
    if s_max == s_min:
        return pd.Series(0.0, index=s.index)
    return ((s - s_min) / (s_max - s_min)) * (target_max - target_min) + target_min
