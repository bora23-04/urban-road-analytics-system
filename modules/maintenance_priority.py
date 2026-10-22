"""
Maintenance Priority Assessment Module.
Calculates a transparent, rule-based, multi-criteria Maintenance Priority Index (0-100).
Strictly deterministic, explainable, and non-predictive.
"""

import pandas as pd
import numpy as np
from typing import Dict, Tuple
from utils.helpers import min_max_scale
from config import DEFAULT_WEIGHTS, ACCIDENT_WEIGHTS, DEFAULT_PRIORITY_THRESHOLDS

def calculate_maintenance_priority_index(
    summary_df: pd.DataFrame,
    weights: Dict[str, float] = None,
    accident_multipliers: Dict[str, float] = None,
    thresholds: Dict[str, float] = None
) -> pd.DataFrame:
    """
    Calculates the multi-criteria Maintenance Priority Index for each road.
    Formula:
      Maintenance_Score = w_trf * Traffic_Score + w_acc * Accident_Score + w_dmg * Damage_Score + w_cmp * Complaint_Score + w_drn * Drainage_Score
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
        
    df = summary_df.copy()
    
    # Use defaults if not supplied
    w = weights if weights else DEFAULT_WEIGHTS
    acc_m = accident_multipliers if accident_multipliers else ACCIDENT_WEIGHTS
    th = thresholds if thresholds else DEFAULT_PRIORITY_THRESHOLDS
    
    # 1. Traffic Burden Component (0 - 100)
    # Higher vehicle count and higher occupancy increase traffic burden
    raw_traffic = (
        df.get("Avg_Vehicle_Count", pd.Series(0, index=df.index)) * 0.7 +
        df.get("Avg_Occupancy", pd.Series(0, index=df.index)) * 0.3
    )
    df["Traffic_Score"] = min_max_scale(raw_traffic, 0, 100).round(2)
    
    # 2. Accident Burden Component (0 - 100)
    # Combines raw accident counts with fatality and casualty multipliers
    raw_accident = (
        df.get("Accident_Count", pd.Series(0, index=df.index)) * acc_m.get("base_count", 1.0) +
        df.get("Fatalities", pd.Series(0, index=df.index)) * acc_m.get("fatalities_multiplier", 3.0) +
        df.get("Total_Casualties", pd.Series(0, index=df.index)) * acc_m.get("casualties_multiplier", 0.5)
    )
    df["Accident_Score"] = min_max_scale(raw_accident, 0, 100).round(2)
    
    # 3. Road Damage Component (0 - 100)
    # Direct road damage percentage
    raw_damage = df.get("Avg_Road_Damage_Percent", pd.Series(0, index=df.index))
    df["Damage_Score"] = min_max_scale(raw_damage, 0, 100).round(2)
    
    # 4. Complaint / Pothole Burden Component (0 - 100)
    # Combines complaint frequency and total pothole counts
    raw_complaints = (
        df.get("Complaint_Count", pd.Series(0, index=df.index)) * 0.5 +
        df.get("Total_Pothole_Count", pd.Series(0, index=df.index)) * 0.3 +
        df.get("High_Severity_Complaints", pd.Series(0, index=df.index)) * 0.2
    )
    df["Complaint_Score"] = min_max_scale(raw_complaints, 0, 100).round(2)
    
    # 5. Drainage & Waterlogging Vulnerability Component (0 - 100)
    raw_drainage = (
        df.get("Waterlogging_Complaints", pd.Series(0, index=df.index)) * 0.5 +
        df.get("Drainage_Issues", pd.Series(0, index=df.index)) * 0.5
    )
    # If Drainage_Condition column exists in road master (Poor / Moderate)
    if "Drainage_Condition" in df.columns:
        drain_cond_bonus = df["Drainage_Condition"].apply(lambda x: 20 if str(x).lower() == 'poor' else (10 if str(x).lower() == 'moderate' else 0))
        raw_drainage = raw_drainage + drain_cond_bonus
        
    df["Drainage_Score"] = min_max_scale(raw_drainage, 0, 100).round(2)
    
    # Normalize weights sum to 1.0
    total_w = sum(w.values())
    w_trf = w.get("traffic_weight", 0.25) / total_w
    w_acc = w.get("accident_weight", 0.25) / total_w
    w_dmg = w.get("damage_weight", 0.25) / total_w
    w_cmp = w.get("complaint_weight", 0.15) / total_w
    w_drn = w.get("drainage_weight", 0.10) / total_w
    
    # Composite Maintenance Score (0 - 100)
    df["Maintenance_Score"] = (
        df["Traffic_Score"] * w_trf +
        df["Accident_Score"] * w_acc +
        df["Damage_Score"] * w_dmg +
        df["Complaint_Score"] * w_cmp +
        df["Drainage_Score"] * w_drn
    ).round(2)
    
    # Assign Priority Level
    low_cut = th.get("low_cutoff", 33.33)
    med_cut = th.get("medium_cutoff", 66.66)
    
    def assign_priority(score):
        if score >= med_cut:
            return "High"
        elif score >= low_cut:
            return "Medium"
        else:
            return "Low"
            
    df["Priority_Level"] = df["Maintenance_Score"].apply(assign_priority)
    
    # Generate transparent explanations of contributing factors
    def explain_road_priority(row):
        factors = []
        if row["Damage_Score"] >= 50:
            factors.append(f"Severe pavement damage ({row['Damage_Score']:.1f}/100)")
        if row["Accident_Score"] >= 40:
            factors.append(f"Recorded crash history ({row['Accident_Score']:.1f}/100, {int(row.get('Accident_Count', 0))} crashes)")
        if row["Traffic_Score"] >= 50:
            factors.append(f"High traffic demand ({row['Traffic_Score']:.1f}/100)")
        if row["Complaint_Score"] >= 40:
            factors.append(f"Frequent pothole complaints ({row['Complaint_Score']:.1f}/100)")
        if row["Drainage_Score"] >= 40:
            factors.append(f"Drainage / waterlogging vulnerability ({row['Drainage_Score']:.1f}/100)")
            
        if not factors:
            factors.append("Low distress and manageable operational traffic levels across all sub-indices.")
            
        return "; ".join(factors)
        
    df["Contributing_Factors"] = df.apply(explain_road_priority, axis=1)
    
    return df
