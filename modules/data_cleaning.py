"""
Data Cleaning Module for Urban Road Analytics and Maintenance Assessment System.
Applies safe, transparent, and non-fabricating transformations.
"""

import pandas as pd
import numpy as np
from utils.helpers import safe_strip_strings, standardize_yes_no, safe_to_numeric, safe_to_datetime

def clean_road_master(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Road Master dataset."""
    if df is None or df.empty:
        return df
    
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    df_clean = safe_strip_strings(df_clean)
    
    # Standardize Road_ID
    if "Road_ID" in df_clean.columns:
        df_clean["Road_ID"] = df_clean["Road_ID"].astype(str).str.strip().str.upper()
    
    # Numeric conversions
    num_cols = ["Road_Length_KM", "Number_of_Lanes", "Road_Width_M", "Construction_Year", "Road_Age_Years"]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
            
    # Dates
    if "Last_Repair_Date" in df_clean.columns:
        df_clean["Last_Repair_Date"] = pd.to_datetime(df_clean["Last_Repair_Date"], errors='coerce')
        
    # Standardize Yes/No fields
    yes_no_cols = ["Drainage_Available", "Street_Lighting"]
    for col in yes_no_cols:
        if col in df_clean.columns:
            df_clean[col] = standardize_yes_no(df_clean[col])
            
    # Deduplicate exact rows
    df_clean = df_clean.drop_duplicates(subset=["Road_ID"] if "Road_ID" in df_clean.columns else None)
    
    return df_clean

def clean_traffic_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean Traffic observations dataset."""
    if df is None or df.empty:
        return df
        
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    df_clean = safe_strip_strings(df_clean)
    
    if "Road_ID" in df_clean.columns:
        df_clean["Road_ID"] = df_clean["Road_ID"].astype(str).str.strip().str.upper()
        
    # Convert dates
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors='coerce')
        
    # Convert numerics
    count_cols = ["Vehicle_Count", "Cars_Count", "Two_Wheeler_Count", "Bus_Count", "Truck_Count", "Auto_Count"]
    for col in count_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).clip(lower=0)
            
    if "Average_Speed_KMH" in df_clean.columns:
        df_clean["Average_Speed_KMH"] = pd.to_numeric(df_clean["Average_Speed_KMH"], errors='coerce').clip(lower=0)
        
    if "Road_Occupancy_Percent" in df_clean.columns:
        df_clean["Road_Occupancy_Percent"] = pd.to_numeric(df_clean["Road_Occupancy_Percent"], errors='coerce').clip(0, 100)
        
    if "Travel_Time_Min" in df_clean.columns:
        df_clean["Travel_Time_Min"] = pd.to_numeric(df_clean["Travel_Time_Min"], errors='coerce').clip(lower=0)
        
    if "Peak_Hour_Flag" in df_clean.columns:
        df_clean["Peak_Hour_Flag"] = standardize_yes_no(df_clean["Peak_Hour_Flag"])
        
    return df_clean

def clean_accident_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean recorded accident events dataset."""
    if df is None or df.empty:
        return df
        
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    df_clean = safe_strip_strings(df_clean)
    
    if "Road_ID" in df_clean.columns:
        df_clean["Road_ID"] = df_clean["Road_ID"].astype(str).str.strip().str.upper()
        
    if "Date" in df_clean.columns:
        df_clean["Date"] = pd.to_datetime(df_clean["Date"], errors='coerce')
        
    # Numeric casualty columns
    num_cols = ["Number_of_Vehicles", "Number_of_Casualties", "Fatalities", "Injuries", "Speed_Limit_KMH"]
    for col in num_cols:
        if col in df_clean.columns:
            df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).clip(lower=0)
            
    # Coordinates
    if "Latitude" in df_clean.columns:
        df_clean["Latitude"] = pd.to_numeric(df_clean["Latitude"], errors='coerce')
    if "Longitude" in df_clean.columns:
        df_clean["Longitude"] = pd.to_numeric(df_clean["Longitude"], errors='coerce')
        
    # Standardize Yes/No fields
    for col in ["Pedestrian_Involved", "Two_Wheeler_Involved", "Police_Attended"]:
        if col in df_clean.columns:
            df_clean[col] = standardize_yes_no(df_clean[col])
            
    # Standardize Accident Severity
    if "Accident_Severity" in df_clean.columns:
        df_clean["Accident_Severity"] = df_clean["Accident_Severity"].astype(str).str.strip().str.title()
        
    return df_clean

def clean_pothole_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean road condition and pothole complaints dataset."""
    if df is None or df.empty:
        return df
        
    df_clean = df.copy()
    df_clean.columns = [c.strip() for c in df_clean.columns]
    df_clean = safe_strip_strings(df_clean)
    
    if "Road_ID" in df_clean.columns:
        df_clean["Road_ID"] = df_clean["Road_ID"].astype(str).str.strip().str.upper()
        
    for date_col in ["Complaint_Date", "Reported_Date", "Resolved_Date"]:
        if date_col in df_clean.columns:
            df_clean[date_col] = pd.to_datetime(df_clean[date_col], errors='coerce')
            
    # Numeric columns
    if "Pothole_Count" in df_clean.columns:
        df_clean["Pothole_Count"] = pd.to_numeric(df_clean["Pothole_Count"], errors='coerce').fillna(0).clip(lower=0)
        
    if "Pothole_Size_M2" in df_clean.columns:
        df_clean["Pothole_Size_M2"] = pd.to_numeric(df_clean["Pothole_Size_M2"], errors='coerce').clip(lower=0)
        
    if "Road_Damage_Percent" in df_clean.columns:
        df_clean["Road_Damage_Percent"] = pd.to_numeric(df_clean["Road_Damage_Percent"], errors='coerce').clip(0, 100)
        
    if "Resolution_Days" in df_clean.columns:
        df_clean["Resolution_Days"] = pd.to_numeric(df_clean["Resolution_Days"], errors='coerce').clip(lower=0)
        
    # Coordinates
    if "Latitude" in df_clean.columns:
        df_clean["Latitude"] = pd.to_numeric(df_clean["Latitude"], errors='coerce')
    if "Longitude" in df_clean.columns:
        df_clean["Longitude"] = pd.to_numeric(df_clean["Longitude"], errors='coerce')
        
    # Yes/No fields
    for col in ["Waterlogging_Present", "Drainage_Issue", "Traffic_Obstruction"]:
        if col in df_clean.columns:
            df_clean[col] = standardize_yes_no(df_clean[col])
            
    if "Severity" in df_clean.columns:
        df_clean["Severity"] = df_clean["Severity"].astype(str).str.strip().str.title()
        
    return df_clean
