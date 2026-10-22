"""
Data Loader Module for Urban Road Analytics and Maintenance Assessment System.
Handles CSV/Excel file loading, dataset auto-detection, and metadata extraction.
"""

import os
import pandas as pd
from typing import Dict, Tuple, Optional
from utils.helpers import detect_dataset_type
from config import DEFAULT_FILES

def load_file_from_path_or_buffer(file_or_path, file_name: str = "") -> pd.DataFrame:
    """
    Load a CSV or Excel file safely into a pandas DataFrame.
    """
    if file_or_path is None:
        return None
        
    try:
        # If it's a file path string
        if isinstance(file_or_path, str):
            if not os.path.exists(file_or_path):
                return None
            if file_or_path.endswith('.csv'):
                return pd.read_csv(file_or_path)
            elif file_or_path.endswith(('.xlsx', '.xls')):
                return pd.read_excel(file_or_path)
            else:
                try:
                    return pd.read_csv(file_or_path)
                except Exception:
                    return pd.read_excel(file_or_path)
        
        # If it's an uploaded file buffer (Streamlit UploadedFile)
        name = getattr(file_or_path, 'name', file_name).lower()
        if name.endswith('.csv'):
            return pd.read_csv(file_or_path)
        elif name.endswith(('.xlsx', '.xls')):
            return pd.read_excel(file_or_path)
        else:
            try:
                return pd.read_csv(file_or_path)
            except Exception:
                return pd.read_excel(file_or_path)
    except Exception as e:
        print(f"Error loading {file_name}: {e}")
        return None

def load_default_datasets() -> Dict[str, pd.DataFrame]:
    """
    Load default sample datasets from project data/raw directory.
    """
    datasets = {}
    for key, path in DEFAULT_FILES.items():
        if os.path.exists(path):
            df = load_file_from_path_or_buffer(path, os.path.basename(path))
            if df is not None:
                datasets[key] = df
    return datasets

def extract_dataset_metadata(df: pd.DataFrame, dataset_name: str = "Dataset") -> Dict:
    """
    Extract key metadata about a DataFrame (shape, types, columns, nulls).
    """
    if df is None or df.empty:
        return {
            "name": dataset_name,
            "rows": 0,
            "columns": 0,
            "column_names": [],
            "dtypes": {},
            "null_count": 0,
            "memory_mb": 0.0
        }
    
    return {
        "name": dataset_name,
        "rows": len(df),
        "columns": len(df.columns),
        "column_names": df.columns.tolist(),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
        "null_count": int(df.isnull().sum().sum()),
        "memory_mb": round(df.memory_usage(deep=True).sum() / (1024 * 1024), 2)
    }
