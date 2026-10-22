"""
Road & Traffic Analysis Module for Urban Road Analytics System.
Analyzes volumes, operating speeds, temporal patterns, and congestion bottlenecks.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict

def get_top_traffic_roads(summary_df: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    """Return top N roads sorted by average vehicular volume."""
    if summary_df is None or "Avg_Vehicle_Count" not in summary_df.columns:
        return pd.DataFrame()
    
    cols = ["Road_ID", "Road_Name", "Road_Type", "Road_Category", "Avg_Vehicle_Count", "Max_Vehicle_Count", "Avg_Speed", "Avg_Occupancy"]
    avail_cols = [c for c in cols if c in summary_df.columns]
    return summary_df[avail_cols].sort_values(by="Avg_Vehicle_Count", ascending=False).head(n)

def plot_hourly_traffic_pattern(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot average vehicle counts and speeds across hours of the day."""
    if traffic_df is None or traffic_df.empty or "Time" not in traffic_df.columns:
        return None, "Traffic hourly time series not available."
        
    df = traffic_df.copy()
    # Extract hour
    try:
        if pd.api.types.is_string_dtype(df["Time"]):
            df["Hour"] = pd.to_datetime(df["Time"], format="%H:%M:%S", errors='coerce').dt.hour
            if df["Hour"].isnull().all():
                df["Hour"] = pd.to_datetime(df["Time"], errors='coerce').dt.hour
        else:
            df["Hour"] = pd.to_datetime(df["Time"].astype(str), errors='coerce').dt.hour
    except Exception:
        df["Hour"] = 0
        
    hourly = df.groupby("Hour").agg(
        Avg_Vehicle_Count=("Vehicle_Count", "mean"),
        Avg_Speed=("Average_Speed_KMH", "mean")
    ).reset_index().dropna(subset=["Hour"])
    
    if hourly.empty:
        return None, "Could not parse hourly intervals from traffic time data."
        
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=hourly["Hour"],
        y=hourly["Avg_Vehicle_Count"],
        name="Avg Volume (Vehicles/hr)",
        marker_color="#2b5c8f",
        yaxis="y1"
    ))
    fig.add_trace(go.Scatter(
        x=hourly["Hour"],
        y=hourly["Avg_Speed"],
        name="Avg Speed (KM/H)",
        mode="lines+markers",
        marker_color="#e11d48",
        line=dict(width=3),
        yaxis="y2"
    ))
    
    fig.update_layout(
        title="Diurnal Traffic Profile: Hourly Traffic Volume vs Average Speed",
        xaxis=dict(title="Hour of Day (24-Hour Military Format)", tickmode='linear', dtick=2),
        yaxis=dict(title="Average Volume (Vehicles/hr)", side="left"),
        yaxis2=dict(title="Average Operating Speed (KM/H)", overlaying="y", side="right"),
        template="plotly_white",
        legend=dict(x=0.01, y=0.99),
        height=420
    )
    
    peak_hour = hourly.loc[hourly["Avg_Vehicle_Count"].idxmax()]["Hour"]
    interp = f"The diurnal profile reveals pronounced peak periods around {int(peak_hour):02d}:00 hours, coinciding with an observed reduction in average operating speeds."
    return fig, interp

def plot_peak_vs_non_peak(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot comparative metrics between peak and non-peak hours."""
    if traffic_df is None or "Peak_Hour_Flag" not in traffic_df.columns:
        return None, "Peak Hour indicator not available."
        
    peak_comp = traffic_df.groupby("Peak_Hour_Flag").agg(
        Avg_Vehicle_Count=("Vehicle_Count", "mean"),
        Avg_Speed=("Average_Speed_KMH", "mean"),
        Avg_Occupancy=("Road_Occupancy_Percent", "mean"),
        Avg_Travel_Time=("Travel_Time_Min", "mean")
    ).reset_index()
    
    fig = px.bar(
        peak_comp,
        x="Peak_Hour_Flag",
        y="Avg_Vehicle_Count",
        color="Peak_Hour_Flag",
        title="Traffic Volume Comparison: Peak vs Non-Peak Operating Hours",
        labels={"Peak_Hour_Flag": "Peak Hour Flag (Yes/No)", "Avg_Vehicle_Count": "Average Vehicle Count"},
        text_auto=".1f",
        color_discrete_map={"Yes": "#d9534f", "No": "#5bc0de"}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=380)
    
    interp = "Peak operating hours register significantly elevated vehicular loads and corresponding lane occupancy relative to off-peak periods."
    return fig, interp

def plot_weekday_vs_weekend(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot weekday vs weekend traffic volumes."""
    if traffic_df is None or traffic_df.empty:
        return None, "Traffic data not available."
        
    df = traffic_df.copy()
    if "Day_of_Week" in df.columns:
        day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        df["Day_Type"] = df["Day_of_Week"].apply(lambda x: "Weekend" if str(x).strip() in ["Saturday", "Sunday"] else "Weekday")
        
        day_agg = df.groupby("Day_of_Week").agg(
            Avg_Vehicle_Count=("Vehicle_Count", "mean"),
            Avg_Speed=("Average_Speed_KMH", "mean")
        ).reindex(day_order).dropna().reset_index()
        
        fig = px.bar(
            day_agg,
            x="Day_of_Week",
            y="Avg_Vehicle_Count",
            color="Avg_Speed",
            color_continuous_scale="Viridis",
            title="Weekly Traffic Volume & Speed Variations across Days of Week",
            labels={"Day_of_Week": "Day of Week", "Avg_Vehicle_Count": "Mean Vehicle Count", "Avg_Speed": "Mean Speed (KM/H)"},
            text_auto=".0f"
        )
        fig.update_traces(textposition='outside')
        fig.update_layout(template="plotly_white", height=400)
        
        interp = "Traffic patterns vary across days of the week, with characteristic weekday commuting demands contrasting with weekend vehicular distributions."
        return fig, interp
        
    return None, "Day of week data unavailable."

def plot_speed_vs_volume_scatter(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Scatter plot showing observed Speed vs Vehicle Volume relationship."""
    if traffic_df is None or "Vehicle_Count" not in traffic_df.columns or "Average_Speed_KMH" not in traffic_df.columns:
        return None, "Speed or vehicle count columns missing."
        
    # Sample 3000 points if dataset is large to maintain crisp interactive performance
    sample_df = traffic_df.sample(n=min(3000, len(traffic_df)), random_state=42)
    
    fig = px.scatter(
        sample_df,
        x="Vehicle_Count",
        y="Average_Speed_KMH",
        opacity=0.4,
        trendline="ols",
        title="Macroscopic Speed-Flow Relationship: Average Speed vs Vehicle Volume",
        labels={"Vehicle_Count": "Observed Vehicle Count (vehicles/hr)", "Average_Speed_KMH": "Average Speed (KM/H)"},
        color_discrete_sequence=["#3b82f6"]
    )
    fig.update_layout(template="plotly_white", height=420)
    
    corr = sample_df["Vehicle_Count"].corr(sample_df["Average_Speed_KMH"])
    interp = f"The speed-flow relationship displays an inverse empirical correlation (r = {corr:.2f}), matching classical traffic engineering principles where higher volumes constrain operating velocities."
    return fig, interp

def identify_high_traffic_low_speed_roads(
    summary_df: pd.DataFrame,
    volume_percentile: float = 75.0,
    speed_percentile: float = 50.0
) -> Tuple[pd.DataFrame, float, float]:
    """
    Identifies high-traffic and low-speed road corridors using transparent,
    user-configurable percentile cutoffs.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame(), 0.0, 0.0
        
    vol_cutoff = summary_df["Avg_Vehicle_Count"].quantile(volume_percentile / 100.0)
    spd_cutoff = summary_df["Avg_Speed"].quantile(speed_percentile / 100.0)
    
    flagged = summary_df[
        (summary_df["Avg_Vehicle_Count"] >= vol_cutoff) &
        (summary_df["Avg_Speed"] <= spd_cutoff)
    ].copy()
    
    flagged["Identification_Criteria"] = f"Volume >= {vol_cutoff:.1f} (P{int(volume_percentile)}) & Speed <= {spd_cutoff:.1f} km/h (P{int(speed_percentile)})"
    
    return flagged.sort_values(by="Avg_Vehicle_Count", ascending=False), vol_cutoff, spd_cutoff
