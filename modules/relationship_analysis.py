"""
Relationship Analysis Module for Urban Road Analytics System.
Evaluates observed empirical associations and correlations among traffic, distress, and safety metrics.
Strictly non-causal, academic diagnostic interpretations.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Tuple, Dict

def analyze_bivariate_relationship(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    x_label: str,
    y_label: str,
    title: str,
    color_col: str = None
) -> Tuple[go.Figure, float, int, str]:
    """
    Computes Pearson correlation coefficient and renders an interactive scatter plot
    with OLS trendline and non-causal diagnostic summary.
    """
    if df is None or x_col not in df.columns or y_col not in df.columns:
        return None, 0.0, 0, f"Attributes '{x_col}' or '{y_col}' are missing."
        
    valid_df = df.dropna(subset=[x_col, y_col]).copy()
    valid_df = valid_df[(pd.to_numeric(valid_df[x_col], errors='coerce').notnull()) & 
                        (pd.to_numeric(valid_df[y_col], errors='coerce').notnull())]
    
    # Exclude all-zero pairs if desired, or sample for smooth rendering if huge
    obs_count = len(valid_df)
    if obs_count < 3:
        return None, 0.0, 0, "Insufficient paired observations to evaluate relationship."
        
    sample_df = valid_df.sample(n=min(4000, obs_count), random_state=42)
    
    corr = float(valid_df[x_col].corr(valid_df[y_col]))
    
    # Classify strength and direction
    direction = "positive" if corr > 0 else "negative"
    abs_corr = abs(corr)
    if abs_corr >= 0.7:
        strength = "strong"
    elif abs_corr >= 0.4:
        strength = "moderate"
    elif abs_corr >= 0.15:
        strength = "mild / weak"
    else:
        strength = "negligible / minimal"
        
    fig = px.scatter(
        sample_df,
        x=x_col,
        y=y_col,
        color=color_col if color_col in sample_df.columns else None,
        hover_data=["Road_ID", "Road_Name"] if "Road_Name" in sample_df.columns else ["Road_ID"],
        trendline="ols",
        opacity=0.45,
        title=f"{title} (r = {corr:.2f}, N = {obs_count:,})",
        labels={x_col: x_label, y_col: y_label},
        color_discrete_sequence=["#2563eb"]
    )
    fig.update_layout(template="plotly_white", height=420)
    
    interp = (
        f"Diagnostic Assessment: These variables demonstrate a {strength} {direction} association "
        f"(Pearson r = {corr:.2f}, based on {obs_count:,} road observations). "
        f"Statistical co-variation reflects empirical concurrency within the network and does NOT establish direct causality."
    )
    
    return fig, corr, obs_count, interp

def get_all_relationship_analyses(summary_df: pd.DataFrame) -> Dict[str, Dict]:
    """Execute all 5 core bivariate relationship evaluations."""
    relationships = {
        "traffic_vs_accidents": {
            "x_col": "Avg_Vehicle_Count",
            "y_col": "Accident_Count",
            "x_label": "Average Observed Vehicle Count (vehicles/hr)",
            "y_label": "Recorded Accident Count",
            "title": "Observed Association: Traffic Volume vs Recorded Accident Frequency"
        },
        "traffic_vs_damage": {
            "x_col": "Avg_Vehicle_Count",
            "y_col": "Avg_Road_Damage_Percent",
            "x_label": "Average Observed Vehicle Count (vehicles/hr)",
            "y_label": "Mean Surface Damage (%)",
            "title": "Observed Association: Traffic Volume vs Pavement Damage Rate"
        },
        "damage_vs_accidents": {
            "x_col": "Avg_Road_Damage_Percent",
            "y_col": "Accident_Count",
            "x_label": "Mean Surface Damage (%)",
            "y_label": "Recorded Accident Count",
            "title": "Observed Association: Pavement Damage Rate vs Recorded Accident Frequency"
        },
        "potholes_vs_damage": {
            "x_col": "Total_Pothole_Count",
            "y_col": "Avg_Road_Damage_Percent",
            "x_label": "Total Reported Potholes",
            "y_label": "Mean Surface Damage (%)",
            "title": "Observed Association: Pothole Frequency vs Road Damage Percentage"
        },
        "speed_vs_volume": {
            "x_col": "Avg_Vehicle_Count",
            "y_col": "Avg_Speed",
            "x_label": "Average Observed Vehicle Count (vehicles/hr)",
            "y_label": "Average Operating Speed (KM/H)",
            "title": "Observed Association: Traffic Volume vs Average Operating Speed"
        }
    }
    
    results = {}
    for key, cfg in relationships.items():
        fig, corr, n, interp = analyze_bivariate_relationship(
            summary_df,
            x_col=cfg["x_col"],
            y_col=cfg["y_col"],
            x_label=cfg["x_label"],
            y_label=cfg["y_label"],
            title=cfg["title"],
            color_col="Road_Category" if "Road_Category" in summary_df.columns else None
        )
        results[key] = {
            "fig": fig,
            "corr": corr,
            "n": n,
            "interp": interp,
            "title": cfg["title"]
        }
    return results
