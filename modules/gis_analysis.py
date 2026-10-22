"""
GIS Analysis Module for Urban Road Analytics and Maintenance Assessment System.
Combines GeoPandas geospatial vector processing (with safe fallback) and
interactive Plotly + Folium mapping engines.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional

# Safe import for GeoPandas & Shapely
try:
    import geopandas as gpd
    from shapely.geometry import Point

    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Safe import for Folium
try:
    import folium
    from folium.plugins import MarkerCluster, Fullscreen

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from maps.map_utils import create_base_map, create_priority_popup_html, add_priority_legend_to_map
from config import MAP_DEFAULT_CENTER, PRIORITY_COLORS, SEVERITY_COLORS


def convert_to_geodataframe(df: pd.DataFrame, lat_col="Latitude", lon_col="Longitude"):
    """
    Safely converts a DataFrame into a GeoDataFrame (if geopandas available)
    or a validated coordinate DataFrame.
    """
    if df is None or df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return None

    valid_pts = df[
        pd.to_numeric(df[lat_col], errors='coerce').notnull() &
        pd.to_numeric(df[lon_col], errors='coerce').notnull() &
        (df[lat_col] >= -90) & (df[lat_col] <= 90) &
        (df[lon_col] >= -180) & (df[lon_col] <= 180)
        ].copy()

    if valid_pts.empty:
        return None

    if HAS_GEOPANDAS:
        try:
            return gpd.GeoDataFrame(
                valid_pts,
                geometry=gpd.points_from_xy(valid_pts[lon_col], valid_pts[lat_col]),
                crs="EPSG:4326"
            )
        except Exception:
            return valid_pts
    return valid_pts


def get_geospatial_summary(df: pd.DataFrame) -> Dict:
    """Calculates spatial bounding box, centroid, and coverage metrics."""
    if df is None or df.empty or "Latitude" not in df.columns or "Longitude" not in df.columns:
        return {}

    valid = df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return {}

    min_lat = float(valid["Latitude"].min())
    max_lat = float(valid["Latitude"].max())
    min_lon = float(valid["Longitude"].min())
    max_lon = float(valid["Longitude"].max())

    center_lat = (min_lat + max_lat) / 2.0
    center_lon = (min_lon + max_lon) / 2.0
    area_approx = (max_lat - min_lat) * (max_lon - min_lon)

    return {
        "point_count": len(valid),
        "min_longitude": min_lon,
        "max_longitude": max_lon,
        "min_latitude": min_lat,
        "max_latitude": max_lat,
        "center_lat": center_lat,
        "center_lon": center_lon,
        "spatial_extent_degrees": round(area_approx, 4)
    }


# ----------------------------------------------------------------------
# 1. NATIVE PLOTLY INTERACTIVE MAPS (100% RELIABLE, ZERO IFRAME ISSUES)
# ----------------------------------------------------------------------

def build_plotly_priority_map(priority_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds a rich interactive Plotly GIS scatter map for Maintenance Priority."""
    if priority_df is None or priority_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No priority data available.", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig

    valid = priority_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No geographic coordinates available in dataset.", xref="paper", yref="paper", x=0.5,
                           y=0.5)
        return fig

    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid

    # Render with open-street-map tiles
    try:
        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="Priority_Level",
            size="Maintenance_Score",
            hover_name="Road_Name",
            hover_data={
                "Road_ID": True,
                "Maintenance_Score": ":.1f",
                "Traffic_Score": ":.1f",
                "Accident_Score": ":.1f",
                "Damage_Score": ":.1f",
                "Contributing_Factors": True,
                "Latitude": False,
                "Longitude": False
            },
            color_discrete_map=PRIORITY_COLORS,
            zoom=10.5,
            center={"lat": float(sample_df["Latitude"].mean()), "lon": float(sample_df["Longitude"].mean())},
            mapbox_style="open-street-map",
            title="🗺️ Urban Road Network: Rule-Based Maintenance Priority Assessment"
        )
    except Exception:
        # Fallback to standard Cartesian coordinate scatter
        fig = px.scatter(
            sample_df,
            x="Longitude",
            y="Latitude",
            color="Priority_Level",
            size="Maintenance_Score",
            hover_name="Road_Name",
            hover_data=["Road_ID", "Maintenance_Score", "Contributing_Factors"],
            color_discrete_map=PRIORITY_COLORS,
            title="🗺️ Urban Road Network: Maintenance Priority (Coordinate Scatter)"
        )

    fig.update_layout(
        height=560,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)")
    )
    return fig


def build_plotly_traffic_map(summary_df: pd.DataFrame, metric: str = "Avg_Vehicle_Count",
                             max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly GIS scatter map for Traffic metrics."""
    if summary_df is None or summary_df.empty:
        return go.Figure()

    valid = summary_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()

    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid

    try:
        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color=metric,
            size=metric,
            hover_name="Road_Name",
            hover_data=["Road_ID", "Road_Category", "Avg_Speed", "Avg_Occupancy"],
            color_continuous_scale="Viridis",
            zoom=10.5,
            center={"lat": float(sample_df["Latitude"].mean()), "lon": float(sample_df["Longitude"].mean())},
            mapbox_style="open-street-map",
            title=f"🚦 Traffic Operational Flow: {metric.replace('_', ' ')}"
        )
    except Exception:
        fig = px.scatter(
            sample_df,
            x="Longitude",
            y="Latitude",
            color=metric,
            title=f"Traffic Distribution: {metric.replace('_', ' ')}"
        )

    fig.update_layout(height=560, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def build_plotly_accident_map(accidents_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly GIS scatter map for Accidents."""
    if accidents_df is None or accidents_df.empty:
        return go.Figure()

    valid = accidents_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()

    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid

    try:
        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="Accident_Severity",
            hover_name="Accident_ID",
            hover_data=["Road_ID", "Fatalities", "Number_of_Casualties", "Cause_Category", "Vehicle_Type"],
            color_discrete_map=SEVERITY_COLORS,
            zoom=10.5,
            center={"lat": float(sample_df["Latitude"].mean()), "lon": float(sample_df["Longitude"].mean())},
            mapbox_style="open-street-map",
            title="⚠️ Recorded Accident Locations by Severity Grade"
        )
    except Exception:
        fig = px.scatter(
            sample_df,
            x="Longitude",
            y="Latitude",
            color="Accident_Severity",
            title="Accident Locations by Severity"
        )

    fig.update_layout(height=560, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


def build_plotly_pothole_map(potholes_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly GIS scatter map for Potholes & Road Damage."""
    if potholes_df is None or potholes_df.empty:
        return go.Figure()

    valid = potholes_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()

    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid

    try:
        fig = px.scatter_mapbox(
            sample_df,
            lat="Latitude",
            lon="Longitude",
            color="Severity",
            size="Road_Damage_Percent" if "Road_Damage_Percent" in sample_df.columns else "Pothole_Count",
            hover_name="Complaint_ID",
            hover_data=["Road_ID", "Issue_Category", "Pothole_Count", "Road_Damage_Percent"],
            color_discrete_map=SEVERITY_COLORS,
            zoom=10.5,
            center={"lat": float(sample_df["Latitude"].mean()), "lon": float(sample_df["Longitude"].mean())},
            mapbox_style="open-street-map",
            title="🕳️ Road Surface Defects & Pothole Distress"
        )
    except Exception:
        fig = px.scatter(
            sample_df,
            x="Longitude",
            y="Latitude",
            color="Severity",
            title="Road Distress / Potholes"
        )

    fig.update_layout(height=560, margin={"r": 0, "t": 40, "l": 0, "b": 0})
    return fig


# ----------------------------------------------------------------------
# 2. FOLIUM MAP BUILDERS
# ----------------------------------------------------------------------

def build_road_location_map(summary_df: pd.DataFrame, max_points: int = 1500):
    if not HAS_FOLIUM or summary_df is None:
        return None
    valid = summary_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta["center_lat"], meta["center_lon"]], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    fg = folium.FeatureGroup(name="Road Master Assets")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        name = row.get("Road_Name", "Road")
        r_id = row.get("Road_ID", "")
        folium.CircleMarker(
            location=[lat, lon], radius=4, color="#2563eb", fill=True, fill_color="#3b82f6", fill_opacity=0.7,
            popup=folium.Popup(f"<b>{name}</b><br>ID: {r_id}", max_width=250), tooltip=f"{name} ({r_id})"
        ).add_to(fg)
    fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m


def build_traffic_spatial_map(summary_df: pd.DataFrame, metric: str = "Avg_Vehicle_Count", max_points: int = 1500):
    if not HAS_FOLIUM or summary_df is None:
        return None
    valid = summary_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta["center_lat"], meta["center_lon"]], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    val_series = pd.to_numeric(plot_df.get(metric, 0), errors='coerce').fillna(0)
    v_min, v_max = val_series.min(), val_series.max()
    fg = folium.FeatureGroup(name=f"Traffic: {metric.replace('_', ' ')}")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row.get(metric, 0)
        radius = 3 + ((val - v_min) / (v_max - v_min)) * 9 if v_max > v_min else 5
        color = "#ef4444" if val > val_series.quantile(0.75) else (
            "#f59e0b" if val > val_series.quantile(0.5) else "#3b82f6")
        folium.CircleMarker(
            location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(f"<b>{row.get('Road_Name', 'Road')}</b><br>{metric}: {val:.1f}", max_width=250),
            tooltip=f"{row.get('Road_Name')}: {val:.1f}"
        ).add_to(fg)
    fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m


def build_accident_spatial_map(accidents_df: pd.DataFrame, severity_filter: list = None, max_points: int = 2500):
    if not HAS_FOLIUM or accidents_df is None:
        return None
    valid = accidents_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    if severity_filter:
        valid = valid[valid["Accident_Severity"].isin(severity_filter)]
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta.get("center_lat", 28.6275), meta.get("center_lon", 77.1709)], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    cluster = MarkerCluster(name="Recorded Accidents Cluster").add_to(m)
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        sev = row.get("Accident_Severity", "Minor")
        color = SEVERITY_COLORS.get(sev, "#3b82f6")
        folium.CircleMarker(
            location=[lat, lon], radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.8,
            popup=folium.Popup(f"<b>{row.get('Accident_ID')}</b><br>Severity: {sev}", max_width=260),
            tooltip=f"{sev} Accident"
        ).add_to(cluster)
    folium.LayerControl().add_to(m)
    return m


def build_road_condition_spatial_map(potholes_df: pd.DataFrame, max_points: int = 2500):
    if not HAS_FOLIUM or potholes_df is None:
        return None
    valid = potholes_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta.get("center_lat", 28.6275), meta.get("center_lon", 77.1709)], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    cluster = MarkerCluster(name="Road Distress / Potholes").add_to(m)
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        sev = str(row.get("Severity", "Medium")).title()
        color = SEVERITY_COLORS.get(sev, "#f0ad4e")
        folium.CircleMarker(
            location=[lat, lon], radius=5, color=color, fill=True, fill_color=color, fill_opacity=0.75,
            popup=folium.Popup(f"<b>{row.get('Complaint_ID')}</b><br>Damage: {row.get('Road_Damage_Percent', 0):.1f}%",
                               max_width=260),
            tooltip=f"{sev} Distress"
        ).add_to(cluster)
    folium.LayerControl().add_to(m)
    return m


def build_maintenance_priority_map(priority_df: pd.DataFrame, max_points: int = 2000):
    if not HAS_FOLIUM or priority_df is None:
        return None
    valid = priority_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta.get("center_lat", 28.6275), meta.get("center_lon", 77.1709)], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    fg_high = folium.FeatureGroup(name="🔴 High Priority")
    fg_med = folium.FeatureGroup(name="🟠 Medium Priority")
    fg_low = folium.FeatureGroup(name="🟢 Low Priority")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        prio = row.get("Priority_Level", "Medium")
        color = PRIORITY_COLORS.get(prio, "#f0ad4e")
        popup_html = create_priority_popup_html(row)
        radius = 7 if prio == "High" else (5 if prio == "Medium" else 3.5)
        marker = folium.CircleMarker(
            location=[lat, lon], radius=radius, color=color, fill=True, fill_color=color,
            fill_opacity=0.85 if prio == "High" else 0.7,
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{prio} Priority ({row.get('Maintenance_Score', 0):.1f}) - {row.get('Road_Name', 'Road')}"
        )
        if prio == "High":
            marker.add_to(fg_high)
        elif prio == "Medium":
            marker.add_to(fg_med)
        else:
            marker.add_to(fg_low)
    fg_high.add_to(m)
    fg_med.add_to(m)
    fg_low.add_to(m)
    add_priority_legend_to_map(m)
    folium.LayerControl(collapsed=False).add_to(m)
    return m


def aggregate_by_administrative_zone(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df is None or "Zone" not in summary_df.columns:
        return pd.DataFrame()
    zone_agg = summary_df.groupby("Zone").agg(
        Total_Roads=("Road_ID", "count"),
        Mean_Maintenance_Score=("Maintenance_Score", "mean"),
        High_Priority_Roads=("Priority_Level", lambda s: (s == "High").sum()),
        Avg_Traffic_Volume=("Avg_Vehicle_Count", "mean"),
        Total_Accidents=("Accident_Count", "sum"),
        Total_Complaints=("Complaint_Count", "sum"),
        Avg_Damage_Percent=("Avg_Road_Damage_Percent", "mean")
    ).reset_index()
    return zone_agg.sort_values(by="Mean_Maintenance_Score", ascending=False)
