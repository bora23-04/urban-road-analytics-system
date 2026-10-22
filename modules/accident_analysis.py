"""
Accident Analysis Module for Urban Road Analytics System.
Evaluates recorded crash frequencies, severity distributions, vulnerable road users, and environmental factors.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict

def get_top_accident_roads(summary_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return top N roads by recorded accident count and casualties."""
    if summary_df is None or "Accident_Count" not in summary_df.columns:
        return pd.DataFrame()
        
    cols = ["Road_ID", "Road_Name", "Road_Type", "Road_Category", "Accident_Count", "Fatalities", "Total_Casualties", "Injuries"]
    avail_cols = [c for c in cols if c in summary_df.columns]
    return summary_df[avail_cols].sort_values(by=["Accident_Count", "Fatalities"], ascending=[False, False]).head(n)

def plot_accident_causes(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot recorded primary cause categories for accidents."""
    if accidents_df is None or "Cause_Category" not in accidents_df.columns:
        return None, "Cause category data not available."
        
    counts = accidents_df["Cause_Category"].value_counts().reset_index()
    counts.columns = ["Cause_Category", "Count"]
    
    fig = px.bar(
        counts,
        x="Count",
        y="Cause_Category",
        orientation="h",
        color="Count",
        color_continuous_scale="Reds",
        title="Recorded Primary Contributing Causes in Accident Reports",
        labels={"Count": "Recorded Incident Count", "Cause_Category": "Recorded Cause Category"},
        text="Count"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), height=420)
    
    top_cause = counts.iloc[0]["Cause_Category"]
    top_cnt = counts.iloc[0]["Count"]
    interp = f"The most prevalent recorded contributing factor is '{top_cause}' ({top_cnt:,} incidents, {(top_cnt/len(accidents_df))*100:.1f}% of documented crash events)."
    return fig, interp

def plot_vehicle_involvement(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot vehicle type involvement in recorded collisions."""
    if accidents_df is None or "Vehicle_Type" not in accidents_df.columns:
        return None, "Vehicle type data not available."
        
    counts = accidents_df["Vehicle_Type"].value_counts().reset_index()
    counts.columns = ["Vehicle_Type", "Count"]
    
    fig = px.bar(
        counts,
        x="Vehicle_Type",
        y="Count",
        color="Vehicle_Type",
        title="Vehicle Class Involvement in Documented Traffic Incidents",
        labels={"Vehicle_Type": "Primary Vehicle Type Involved", "Count": "Accident Count"},
        text="Count",
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=400)
    
    interp = f"Vehicle class distribution shows substantial involvement of '{counts.iloc[0]['Vehicle_Type']}', accounting for {counts.iloc[0]['Count']:,} recorded incident occurrences."
    return fig, interp

def plot_vulnerable_road_users(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot pedestrian and two-wheeler involvement proportions."""
    if accidents_df is None:
        return None, "Accident dataset not available."
        
    ped_count = (accidents_df["Pedestrian_Involved"].astype(str).str.title() == "Yes").sum() if "Pedestrian_Involved" in accidents_df.columns else 0
    tw_count = (accidents_df["Two_Wheeler_Involved"].astype(str).str.title() == "Yes").sum() if "Two_Wheeler_Involved" in accidents_df.columns else 0
    total = len(accidents_df)
    
    df_vru = pd.DataFrame([
        {"Category": "Two-Wheeler Involved", "Count": tw_count, "Percentage": (tw_count / total * 100) if total else 0},
        {"Category": "Pedestrian Involved", "Count": ped_count, "Percentage": (ped_count / total * 100) if total else 0}
    ])
    
    fig = px.bar(
        df_vru,
        x="Category",
        y="Count",
        color="Category",
        title="Vulnerable Road User (VRU) Involvement in Collisions",
        labels={"Count": "Documented Incidents", "Category": "Vulnerable User Group"},
        text_auto=True,
        color_discrete_map={"Two-Wheeler Involved": "#f59e0b", "Pedestrian Involved": "#ef4444"}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=380)
    
    interp = f"Two-wheelers were involved in {tw_count:,} incidents ({(tw_count/total*100):.1f}%), while pedestrians were involved in {ped_count:,} incidents ({(ped_count/total*100):.1f}%), highlighting vulnerable road user exposure."
    return fig, interp

def plot_accidents_by_road_type(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot accident counts categorized by road classification."""
    if accidents_df is None or "Road_Type" not in accidents_df.columns:
        return None, "Road type data missing in accidents dataset."
        
    counts = accidents_df.groupby(["Road_Type", "Accident_Severity"]).size().reset_index(name="Count")
    
    fig = px.bar(
        counts,
        x="Road_Type",
        y="Count",
        color="Accident_Severity",
        title="Observed Accident Frequency by Road Classification & Severity",
        labels={"Road_Type": "Road Classification", "Count": "Accident Count", "Accident_Severity": "Severity"},
        barmode="stack",
        color_discrete_map={"Fatal": "#d9534f", "Serious": "#f0ad4e", "Minor": "#5bc0de"}
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-30, height=420)
    
    interp = "Higher-speed and multi-lane arterial and highway classifications exhibit distinct severity compositions compared to lower-order local access roads."
    return fig, interp

def plot_accidents_by_environmental_factors(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot accident frequency by lighting and weather conditions."""
    if accidents_df is None:
        return None, "Accident dataset not available."
        
    col = "Light_Condition" if "Light_Condition" in accidents_df.columns else ("Weather_Condition" if "Weather_Condition" in accidents_df.columns else None)
    if not col:
        return None, "Environmental condition columns not found."
        
    counts = accidents_df[col].value_counts().reset_index()
    counts.columns = [col, "Count"]
    
    fig = px.bar(
        counts,
        x=col,
        y="Count",
        color=col,
        title=f"Accident Occurrence by Environmental Factor: {col.replace('_', ' ')}",
        labels={"Count": "Recorded Accident Events", col: col.replace('_', ' ')},
        text="Count",
        color_discrete_sequence=px.colors.sequential.Turbo
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=380)
    
    interp = f"Observed crash frequencies show notable concentrations under '{counts.iloc[0][col]}' environmental conditions ({counts.iloc[0]['Count']:,} events)."
    return fig, interp
