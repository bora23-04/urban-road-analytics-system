"""
Road Condition Analysis Module for Urban Road Analytics System.
Evaluates physical surface defects, pothole severity, waterlogging, drainage distress, and repair latency.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict

def get_top_damaged_roads(summary_df: pd.DataFrame, n: int = 15, sort_by: str = "Avg_Road_Damage_Percent") -> pd.DataFrame:
    """Return top N roads sorted by specified condition indicator."""
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
        
    cols = [
        "Road_ID", "Road_Name", "Road_Type", "Road_Category",
        "Complaint_Count", "Total_Pothole_Count", "Avg_Road_Damage_Percent",
        "High_Severity_Complaints", "Waterlogging_Complaints", "Drainage_Issues"
    ]
    avail_cols = [c for c in cols if c in summary_df.columns]
    sort_col = sort_by if sort_by in summary_df.columns else "Avg_Road_Damage_Percent"
    
    return summary_df[avail_cols].sort_values(by=sort_col, ascending=False).head(n)

def plot_issue_categories(potholes_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot distribution of reported road defect/complaint issue categories."""
    if potholes_df is None or "Issue_Category" not in potholes_df.columns:
        return None, "Issue Category data missing."
        
    counts = potholes_df["Issue_Category"].value_counts().reset_index()
    counts.columns = ["Issue_Category", "Count"]
    
    fig = px.bar(
        counts,
        x="Count",
        y="Issue_Category",
        orientation="h",
        color="Count",
        color_continuous_scale="Oranges",
        title="Distribution of Observed Road Condition Distress Categories",
        labels={"Count": "Complaint Count", "Issue_Category": "Reported Distress Category"},
        text="Count"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", yaxis=dict(autorange="reversed"), height=400)
    
    top_issue = counts.iloc[0]["Issue_Category"]
    top_cnt = counts.iloc[0]["Count"]
    interp = f"The primary defect reported across urban road assets is '{top_issue}' ({top_cnt:,} instances, {(top_cnt/len(potholes_df))*100:.1f}% of total reported distress)."
    return fig, interp

def plot_road_damage_by_category(road_master_df: pd.DataFrame, potholes_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot average road damage percentage across administrative agencies."""
    if road_master_df is None or potholes_df is None:
        return None, "Data insufficient for administrative breakdown."
        
    merged = potholes_df.merge(road_master_df[["Road_ID", "Road_Category", "Road_Type"]], on="Road_ID", how="inner")
    if merged.empty or "Road_Category" not in merged.columns or "Road_Damage_Percent" not in merged.columns:
        return None, "Cannot merge road category with distress metrics."
        
    cat_dmg = merged.groupby("Road_Category")["Road_Damage_Percent"].mean().reset_index()
    
    fig = px.bar(
        cat_dmg,
        x="Road_Category",
        y="Road_Damage_Percent",
        color="Road_Damage_Percent",
        color_continuous_scale="Viridis",
        title="Average Pavement Damage Percentage by Managing Agency",
        labels={"Road_Category": "Managing Agency / Category", "Road_Damage_Percent": "Mean Surface Damage (%)"},
        text_auto=".1f"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", height=380)
    
    interp = f"Observed surface degradation varies across administrative jurisdictions, with '{cat_dmg.loc[cat_dmg['Road_Damage_Percent'].idxmax()]['Road_Category']}' recording the highest mean damage rate ({cat_dmg['Road_Damage_Percent'].max():.1f}%)."
    return fig, interp

def plot_resolution_days_distribution(potholes_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot distribution of complaint resolution turnaround times (days)."""
    if potholes_df is None or "Resolution_Days" not in potholes_df.columns:
        return None, "Resolution days data not available."
        
    valid_res = potholes_df["Resolution_Days"].dropna()
    if valid_res.empty:
        return None, "No resolved complaint records found."
        
    fig = px.histogram(
        potholes_df,
        x="Resolution_Days",
        nbins=30,
        title="Turnaround Time for Road Defect Repair & Resolution (Days)",
        labels={"Resolution_Days": "Resolution Latency (Days)", "count": "Completed Repair Actions"},
        marginal="box",
        color_discrete_sequence=["#8b5cf6"]
    )
    fig.update_layout(template="plotly_white", height=380)
    
    mean_days = valid_res.mean()
    median_days = valid_res.median()
    interp = f"Municipal maintenance turnaround times average {mean_days:.1f} days (median {median_days:.0f} days) between initial defect reporting and documented resolution."
    return fig, interp

def plot_environmental_distress_factors(potholes_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot occurrence of waterlogging, drainage issues, and traffic obstruction."""
    if potholes_df is None:
        return None, "Pothole dataset not available."
        
    total = len(potholes_df)
    wl_count = (potholes_df["Waterlogging_Present"].astype(str).str.title() == "Yes").sum() if "Waterlogging_Present" in potholes_df.columns else 0
    dr_count = (potholes_df["Drainage_Issue"].astype(str).str.title() == "Yes").sum() if "Drainage_Issue" in potholes_df.columns else 0
    ob_count = (potholes_df["Traffic_Obstruction"].astype(str).str.title() == "Yes").sum() if "Traffic_Obstruction" in potholes_df.columns else 0
    
    factors_df = pd.DataFrame([
        {"Factor": "Waterlogging Present", "Count": wl_count, "Percentage": (wl_count / total * 100) if total else 0},
        {"Factor": "Drainage Issue Identified", "Count": dr_count, "Percentage": (dr_count / total * 100) if total else 0},
        {"Factor": "Traffic Flow Obstructed", "Count": ob_count, "Percentage": (ob_count / total * 100) if total else 0}
    ])
    
    fig = px.bar(
        factors_df,
        x="Factor",
        y="Count",
        color="Factor",
        title="Associated Drainage and Traffic Impact Factors in Road Distress Reports",
        labels={"Count": "Reported Instances", "Factor": "Impact Factor"},
        text_auto=True,
        color_discrete_sequence=["#0ea5e9", "#6366f1", "#f97316"]
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=380)
    
    interp = f"Waterlogging accompanied {wl_count:,} defect reports ({(wl_count/total*100):.1f}%), while drainage deficiencies were flagged in {dr_count:,} locations ({(dr_count/total*100):.1f}%)."
    return fig, interp
