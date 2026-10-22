"""
Data Validation Module for Urban Road Analytics and Maintenance Assessment System.
Performs 18+ systematic data quality and referential integrity checks.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

def validate_road_master(df: pd.DataFrame) -> List[Dict]:
    """Validate Road Master dataset integrity and bounds."""
    results = []
    if df is None or df.empty:
        return results
    
    total = len(df)
    
    # 1. Missing Values
    null_count = df.isnull().sum().sum()
    results.append({
        "Module": "Road Master",
        "Check": "Missing Cell Values",
        "Total Records": total,
        "Invalid Records": int(null_count),
        "Status": "PASS" if null_count == 0 else ("WARNING" if null_count < total * 0.05 else "FAIL"),
        "Explanation": f"Found {null_count} null/missing cells across all road master columns."
    })
    
    # 2. Duplicate Rows
    dups = df.duplicated().sum()
    results.append({
        "Module": "Road Master",
        "Check": "Exact Duplicate Rows",
        "Total Records": total,
        "Invalid Records": int(dups),
        "Status": "PASS" if dups == 0 else "FAIL",
        "Explanation": f"Identified {dups} exact duplicate rows."
    })
    
    # 3. Duplicate Road_ID
    if "Road_ID" in df.columns:
        dup_ids = df["Road_ID"].duplicated().sum()
        results.append({
            "Module": "Road Master",
            "Check": "Duplicate Primary Key (Road_ID)",
            "Total Records": total,
            "Invalid Records": int(dup_ids),
            "Status": "PASS" if dup_ids == 0 else "FAIL",
            "Explanation": f"Detected {dup_ids} duplicated Road_ID keys in master catalog."
        })
    
    # 4. Numeric Bounds: Length & Width
    if "Road_Length_KM" in df.columns:
        inv_len = (pd.to_numeric(df["Road_Length_KM"], errors='coerce') <= 0).sum()
        results.append({
            "Module": "Road Master",
            "Check": "Road Length Range (> 0 km)",
            "Total Records": total,
            "Invalid Records": int(inv_len),
            "Status": "PASS" if inv_len == 0 else "WARNING",
            "Explanation": f"{inv_len} records have non-positive or unparseable road lengths."
        })
        
    if "Road_Width_M" in df.columns:
        inv_w = (pd.to_numeric(df["Road_Width_M"], errors='coerce') <= 0).sum()
        results.append({
            "Module": "Road Master",
            "Check": "Road Width Range (> 0 m)",
            "Total Records": total,
            "Invalid Records": int(inv_w),
            "Status": "PASS" if inv_w == 0 else "WARNING",
            "Explanation": f"{inv_w} records have non-positive or unparseable road widths."
        })
        
    return results

def validate_traffic_data(df: pd.DataFrame, road_master_df: pd.DataFrame = None) -> List[Dict]:
    """Validate Traffic dataset values, speeds, counts, and referential integrity."""
    results = []
    if df is None or df.empty:
        return results
    
    total = len(df)
    
    # 1. Missing Values
    null_count = df.isnull().sum().sum()
    results.append({
        "Module": "Traffic Data",
        "Check": "Missing Cell Values",
        "Total Records": total,
        "Invalid Records": int(null_count),
        "Status": "PASS" if null_count == 0 else "WARNING",
        "Explanation": f"Found {null_count} null cells in traffic records."
    })
    
    # 2. Negative Vehicle Counts
    if "Vehicle_Count" in df.columns:
        neg_veh = (pd.to_numeric(df["Vehicle_Count"], errors='coerce') < 0).sum()
        results.append({
            "Module": "Traffic Data",
            "Check": "Non-Negative Vehicle Counts",
            "Total Records": total,
            "Invalid Records": int(neg_veh),
            "Status": "PASS" if neg_veh == 0 else "FAIL",
            "Explanation": f"{neg_veh} records contain negative total vehicle counts."
        })
        
    # 3. Negative Speed KMH
    if "Average_Speed_KMH" in df.columns:
        neg_spd = (pd.to_numeric(df["Average_Speed_KMH"], errors='coerce') < 0).sum()
        results.append({
            "Module": "Traffic Data",
            "Check": "Non-Negative Speed (>= 0 km/h)",
            "Total Records": total,
            "Invalid Records": int(neg_spd),
            "Status": "PASS" if neg_spd == 0 else "FAIL",
            "Explanation": f"{neg_spd} records contain negative average speed observations."
        })
        
    # 4. Road Occupancy Range (0 - 100%)
    if "Road_Occupancy_Percent" in df.columns:
        occ = pd.to_numeric(df["Road_Occupancy_Percent"], errors='coerce')
        inv_occ = ((occ < 0) | (occ > 100)).sum()
        results.append({
            "Module": "Traffic Data",
            "Check": "Road Occupancy Range (0–100%)",
            "Total Records": total,
            "Invalid Records": int(inv_occ),
            "Status": "PASS" if inv_occ == 0 else "FAIL",
            "Explanation": f"{inv_occ} records have occupancy values outside the logical 0-100% boundary."
        })
        
    # 5. Referential Integrity: Traffic Road_ID in Road Master
    if road_master_df is not None and "Road_ID" in df.columns and "Road_ID" in road_master_df.columns:
        master_ids = set(road_master_df["Road_ID"].dropna().unique())
        orphan_ids = (~df["Road_ID"].isin(master_ids)).sum()
        results.append({
            "Module": "Traffic Data",
            "Check": "Referential Integrity (Road_ID in Master)",
            "Total Records": total,
            "Invalid Records": int(orphan_ids),
            "Status": "PASS" if orphan_ids == 0 else "FAIL",
            "Explanation": f"{orphan_ids} traffic records reference Road_IDs not found in Road Master."
        })
        
    # 6. Date Format Validation
    if "Date" in df.columns:
        inv_dates = pd.to_datetime(df["Date"], errors='coerce').isnull().sum()
        results.append({
            "Module": "Traffic Data",
            "Check": "Date Parse Consistency",
            "Total Records": total,
            "Invalid Records": int(inv_dates),
            "Status": "PASS" if inv_dates == 0 else "WARNING",
            "Explanation": f"{inv_dates} records have unparseable traffic date values."
        })
        
    return results

def validate_accident_data(df: pd.DataFrame, road_master_df: pd.DataFrame = None) -> List[Dict]:
    """Validate Accident dataset counts, casualties, coordinates, and foreign keys."""
    results = []
    if df is None or df.empty:
        return results
        
    total = len(df)
    
    # 1. Missing Values (Excluding optional road defect)
    null_count = df.drop(columns=['Road_Surface_Defect'], errors='ignore').isnull().sum().sum()
    results.append({
        "Module": "Accidents",
        "Check": "Mandatory Fields Completeness",
        "Total Records": total,
        "Invalid Records": int(null_count),
        "Status": "PASS" if null_count == 0 else "WARNING",
        "Explanation": f"{null_count} null values in mandatory accident fields."
    })
    
    # 2. Negative Casualties or Fatalities
    if "Number_of_Casualties" in df.columns:
        neg_cas = (pd.to_numeric(df["Number_of_Casualties"], errors='coerce') < 0).sum()
        results.append({
            "Module": "Accidents",
            "Check": "Non-Negative Casualties",
            "Total Records": total,
            "Invalid Records": int(neg_cas),
            "Status": "PASS" if neg_cas == 0 else "FAIL",
            "Explanation": f"{neg_cas} records have negative casualty counts."
        })
        
    if "Fatalities" in df.columns:
        neg_fat = (pd.to_numeric(df["Fatalities"], errors='coerce') < 0).sum()
        results.append({
            "Module": "Accidents",
            "Check": "Non-Negative Fatalities",
            "Total Records": total,
            "Invalid Records": int(neg_fat),
            "Status": "PASS" if neg_fat == 0 else "FAIL",
            "Explanation": f"{neg_fat} records have negative fatality counts."
        })
        
    # 3. Geographic Coordinates Validity
    if "Latitude" in df.columns and "Longitude" in df.columns:
        lat = pd.to_numeric(df["Latitude"], errors='coerce')
        lon = pd.to_numeric(df["Longitude"], errors='coerce')
        inv_coords = ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180) | lat.isnull() | lon.isnull()).sum()
        results.append({
            "Module": "Accidents",
            "Check": "Geographic Coordinates Range (WGS84)",
            "Total Records": total,
            "Invalid Records": int(inv_coords),
            "Status": "PASS" if inv_coords == 0 else "FAIL",
            "Explanation": f"{inv_coords} accident records contain invalid or out-of-bound Lat/Lon coordinates."
        })
        
    # 4. Referential Integrity
    if road_master_df is not None and "Road_ID" in df.columns and "Road_ID" in road_master_df.columns:
        master_ids = set(road_master_df["Road_ID"].dropna().unique())
        orphan_acc = (~df["Road_ID"].isin(master_ids)).sum()
        results.append({
            "Module": "Accidents",
            "Check": "Referential Integrity (Road_ID in Master)",
            "Total Records": total,
            "Invalid Records": int(orphan_acc),
            "Status": "PASS" if orphan_acc == 0 else "FAIL",
            "Explanation": f"{orphan_acc} accidents reference Road_IDs missing from Road Master."
        })
        
    return results

def validate_pothole_data(df: pd.DataFrame, road_master_df: pd.DataFrame = None) -> List[Dict]:
    """Validate Road Condition/Pothole dataset damage percentages and coordinates."""
    results = []
    if df is None or df.empty:
        return results
        
    total = len(df)
    
    # 1. Pothole Counts Non-Negative
    if "Pothole_Count" in df.columns:
        neg_poth = (pd.to_numeric(df["Pothole_Count"], errors='coerce') < 0).sum()
        results.append({
            "Module": "Road Condition",
            "Check": "Non-Negative Pothole Counts",
            "Total Records": total,
            "Invalid Records": int(neg_poth),
            "Status": "PASS" if neg_poth == 0 else "FAIL",
            "Explanation": f"{neg_poth} complaint records contain negative pothole counts."
        })
        
    # 2. Road Damage Percentage (0 - 100%)
    if "Road_Damage_Percent" in df.columns:
        dmg = pd.to_numeric(df["Road_Damage_Percent"], errors='coerce')
        inv_dmg = ((dmg < 0) | (dmg > 100)).sum()
        results.append({
            "Module": "Road Condition",
            "Check": "Road Damage Percentage Range (0–100%)",
            "Total Records": total,
            "Invalid Records": int(inv_dmg),
            "Status": "PASS" if inv_dmg == 0 else "FAIL",
            "Explanation": f"{inv_dmg} records have road damage outside logical 0-100% bounds."
        })
        
    # 3. Geographic Coordinates
    if "Latitude" in df.columns and "Longitude" in df.columns:
        lat = pd.to_numeric(df["Latitude"], errors='coerce')
        lon = pd.to_numeric(df["Longitude"], errors='coerce')
        inv_coords = ((lat < -90) | (lat > 90) | (lon < -180) | (lon > 180) | lat.isnull() | lon.isnull()).sum()
        results.append({
            "Module": "Road Condition",
            "Check": "Geographic Coordinates Range (WGS84)",
            "Total Records": total,
            "Invalid Records": int(inv_coords),
            "Status": "PASS" if inv_coords == 0 else "FAIL",
            "Explanation": f"{inv_coords} pothole records contain invalid Lat/Lon coordinates."
        })
        
    # 4. Referential Integrity
    if road_master_df is not None and "Road_ID" in df.columns and "Road_ID" in road_master_df.columns:
        master_ids = set(road_master_df["Road_ID"].dropna().unique())
        orphan_poth = (~df["Road_ID"].isin(master_ids)).sum()
        results.append({
            "Module": "Road Condition",
            "Check": "Referential Integrity (Road_ID in Master)",
            "Total Records": total,
            "Invalid Records": int(orphan_poth),
            "Status": "PASS" if orphan_poth == 0 else "FAIL",
            "Explanation": f"{orphan_poth} complaints reference Road_IDs missing from Road Master."
        })
        
    return results

def run_all_validations(
    road_master_df: pd.DataFrame = None,
    traffic_df: pd.DataFrame = None,
    accidents_df: pd.DataFrame = None,
    potholes_df: pd.DataFrame = None
) -> pd.DataFrame:
    """
    Run comprehensive 18-point validation across all available datasets
    and compile into a structured evaluation table.
    """
    all_results = []
    
    if road_master_df is not None:
        all_results.extend(validate_road_master(road_master_df))
    if traffic_df is not None:
        all_results.extend(validate_traffic_data(traffic_df, road_master_df))
    if accidents_df is not None:
        all_results.extend(validate_accident_data(accidents_df, road_master_df))
    if potholes_df is not None:
        all_results.extend(validate_pothole_data(potholes_df, road_master_df))
        
    if not all_results:
        return pd.DataFrame(columns=["Module", "Check", "Total Records", "Invalid Records", "Status", "Explanation"])
        
    return pd.DataFrame(all_results)
