"""
GIS Analysis Module for Urban Road Analytics and Maintenance Assessment System.
Combines GeoPandas geospatial vector processing (with safe fallback) and
interactive Plotly OpenStreetMap + Folium mapping engines.
"""

import os
import sys
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple, Optional, Any

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

# Safe import for Map utilities with embedded fallback
try:
    from maps.map_utils import create_base_map, create_priority_popup_html, add_priority_legend_to_map
except Exception:
    def create_base_map(center=None, zoom=11, tiles="OpenStreetMap"):
        if not HAS_FOLIUM:
            return None
        c = center if center is not None else [28.6275, 77.1709]
        return folium.Map(location=c, zoom_start=zoom, tiles=tiles)

    def create_priority_popup_html(row):
        name = row.get("Road_Name", "Road")
        score = row.get("Maintenance_Score", 0)
        prio = row.get("Priority_Level", "Medium")
        return f"<b>{name}</b><br>ID: {row.get('Road_ID', '')}<br>Priority: {prio}<br>Score: {score:.1f}"

    def add_priority_legend_to_map(m):
        pass

from config import MAP_DEFAULT_CENTER, PRIORITY_COLORS, SEVERITY_COLORS

# ----------------------------------------------------------------------
# VECTOR & GEODATAFRAME CONVERSION
# ----------------------------------------------------------------------

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


def get_geospatial_summary(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> Dict[str, Any]:
    """Computes bounding box and geospatial metadata for the urban network."""
    if df is None or df.empty or lat_col not in df.columns or lon_col not in df.columns:
        return {}
        
    valid = df.dropna(subset=[lat_col, lon_col])
    if valid.empty:
        return {}
        
    min_lat, max_lat = float(valid[lat_col].min()), float(valid[lat_col].max())
    min_lon, max_lon = float(valid[lon_col].min()), float(valid[lon_col].max())
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
# 1. NATIVE PLOTLY OPENSTREETMAP INTERACTIVE GIS MAPS
# ----------------------------------------------------------------------

def _render_safe_osm_map(
    df: pd.DataFrame,
    lat_col: str,
    lon_col: str,
    color_col: str = None,
    hover_name: str = None,
    hover_cols: list = None,
    color_map: dict = None,
    color_scale: str = None,
    title: str = "GIS Infrastructure Map"
) -> go.Figure:
    """Helper ensuring OpenStreetMap tiles are rendered in any Plotly version without failing to Cartesian scatter."""
    lat_center = float(df[lat_col].mean())
    lon_center = float(df[lon_col].mean())
    h_cols = [c for c in (hover_cols or []) if c in df.columns]

    # Strategy 1: Plotly 6.0+ scatter_map
    if hasattr(px, "scatter_map"):
        try:
            kwargs = {
                "data_frame": df,
                "lat": lat_col,
                "lon": lon_col,
                "zoom": 10.5,
                "center": {"lat": lat_center, "lon": lon_center},
                "map_style": "open-street-map",
                "title": title
            }
            if color_col and color_col in df.columns:
                kwargs["color"] = color_col
            if hover_name and hover_name in df.columns:
                kwargs["hover_name"] = hover_name
            if h_cols:
                kwargs["hover_data"] = h_cols
            if color_map:
                kwargs["color_discrete_map"] = color_map
            if color_scale:
                kwargs["color_continuous_scale"] = color_scale
                
            fig = px.scatter_map(**kwargs)
            fig.update_traces(marker=dict(size=8, opacity=0.85))
            fig.update_layout(
                height=560,
                margin={"r": 0, "t": 40, "l": 0, "b": 0},
                legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)")
            )
            return fig
        except Exception:
            pass

    # Strategy 2: Plotly 5.x scatter_mapbox
    try:
        kwargs = {
            "data_frame": df,
            "lat": lat_col,
            "lon": lon_col,
            "zoom": 10.5,
            "center": {"lat": lat_center, "lon": lon_center},
            "mapbox_style": "open-street-map",
            "title": title
        }
        if color_col and color_col in df.columns:
            kwargs["color"] = color_col
        if hover_name and hover_name in df.columns:
            kwargs["hover_name"] = hover_name
        if h_cols:
            kwargs["hover_data"] = h_cols
        if color_map:
            kwargs["color_discrete_map"] = color_map
        if color_scale:
            kwargs["color_continuous_scale"] = color_scale

        fig = px.scatter_mapbox(**kwargs)
        fig.update_traces(marker=dict(size=8, opacity=0.85))
        fig.update_layout(
            height=560,
            margin={"r": 0, "t": 40, "l": 0, "b": 0},
            legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)")
        )
        return fig
    except Exception:
        pass

    # Strategy 3: Direct Graph Objects with OpenStreetMap tiles
    fig = go.Figure()
    if color_map and color_col in df.columns:
        for val, col_hex in color_map.items():
            sub = df[df[color_col] == val]
            if not sub.empty:
                hover_txt = [
                    f"<b>{row.get(hover_name, '')}</b><br>" + "<br>".join([f"{c}: {row.get(c, '')}" for c in h_cols])
                    for _, row in sub.iterrows()
                ]
                fig.add_trace(go.Scattermapbox(
                    lat=sub[lat_col],
                    lon=sub[lon_col],
                    mode="markers",
                    marker=dict(size=8, color=col_hex, opacity=0.85),
                    text=hover_txt,
                    hoverinfo="text",
                    name=str(val)
                ))
    else:
        hover_txt = [
            f"<b>{row.get(hover_name, '')}</b><br>" + "<br>".join([f"{c}: {row.get(c, '')}" for c in h_cols])
            for _, row in df.iterrows()
        ]
        fig.add_trace(go.Scattermapbox(
            lat=df[lat_col],
            lon=df[lon_col],
            mode="markers",
            marker=dict(size=8, color="#2563eb", opacity=0.85),
            text=hover_txt,
            hoverinfo="text",
            name="Road Assets"
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=lat_center, lon=lon_center),
            zoom=10.5
        ),
        title=title,
        height=560,
        margin={"r": 0, "t": 40, "l": 0, "b": 0},
        legend=dict(yanchor="top", y=0.98, xanchor="left", x=0.01, bgcolor="rgba(255,255,255,0.85)")
    )
    return fig


def build_plotly_priority_map(priority_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds a rich interactive Plotly OpenStreetMap GIS for Maintenance Priority."""
    if priority_df is None or priority_df.empty:
        fig = go.Figure()
        fig.add_annotation(text="No priority data available.", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
        
    valid = priority_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        fig = go.Figure()
        fig.add_annotation(text="No geographic coordinates available in dataset.", xref="paper", yref="paper", x=0.5, y=0.5)
        return fig
        
    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    
    return _render_safe_osm_map(
        df=sample_df,
        lat_col="Latitude",
        lon_col="Longitude",
        color_col="Priority_Level",
        hover_name="Road_Name",
        hover_cols=["Road_ID", "Priority_Level", "Maintenance_Score", "Contributing_Factors"],
        color_map=PRIORITY_COLORS,
        title="🗺️ Urban Road Network: Rule-Based Maintenance Priority Assessment (GIS Map)"
    )


def build_plotly_traffic_map(summary_df: pd.DataFrame, metric: str = "Avg_Vehicle_Count", max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly OpenStreetMap for Traffic metrics."""
    if summary_df is None or summary_df.empty:
        return go.Figure()
        
    valid = summary_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()
        
    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    
    return _render_safe_osm_map(
        df=sample_df,
        lat_col="Latitude",
        lon_col="Longitude",
        color_col=metric if metric in sample_df.columns else None,
        hover_name="Road_Name",
        hover_cols=["Road_ID", "Road_Category", metric, "Avg_Speed", "Avg_Occupancy"],
        color_scale="Viridis",
        title=f"🚦 Traffic Operational Flow: {metric.replace('_', ' ')} (GIS Map)"
    )


def build_plotly_accident_map(accidents_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly OpenStreetMap for Accidents."""
    if accidents_df is None or accidents_df.empty:
        return go.Figure()
        
    valid = accidents_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()
        
    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    
    return _render_safe_osm_map(
        df=sample_df,
        lat_col="Latitude",
        lon_col="Longitude",
        color_col="Accident_Severity" if "Accident_Severity" in sample_df.columns else None,
        hover_name="Accident_ID",
        hover_cols=["Road_ID", "Accident_Severity", "Fatalities", "Number_of_Casualties", "Cause_Category"],
        color_map=SEVERITY_COLORS,
        title="⚠️ Recorded Accident Locations by Severity Grade (GIS Map)"
    )


def build_plotly_pothole_map(potholes_df: pd.DataFrame, max_points: int = 4000) -> go.Figure:
    """Builds native interactive Plotly OpenStreetMap for Potholes & Road Damage."""
    if potholes_df is None or potholes_df.empty:
        return go.Figure()
        
    valid = potholes_df.dropna(subset=["Latitude", "Longitude"]).copy()
    if valid.empty:
        return go.Figure()
        
    sample_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    
    return _render_safe_osm_map(
        df=sample_df,
        lat_col="Latitude",
        lon_col="Longitude",
        color_col="Severity" if "Severity" in sample_df.columns else None,
        hover_name="Complaint_ID",
        hover_cols=["Road_ID", "Issue_Category", "Pothole_Count", "Road_Damage_Percent"],
        color_map=SEVERITY_COLORS,
        title="🕳️ Road Surface Defects & Pothole Distress (GIS Map)"
    )

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
    m = create_base_map(center=[meta.get("center_lat", 28.6275), meta.get("center_lon", 77.1709)], zoom=11)
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
    m = create_base_map(center=[meta.get("center_lat", 28.6275), meta.get("center_lon", 77.1709)], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    val_series = pd.to_numeric(plot_df.get(metric, 0), errors='coerce').fillna(0)
    v_min, v_max = val_series.min(), val_series.max()
    fg = folium.FeatureGroup(name=f"Traffic: {metric.replace('_', ' ')}")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row.get(metric, 0)
        radius = 3 + ((val - v_min) / (v_max - v_min)) * 9 if v_max > v_min else 5
        color = "#ef4444" if val > val_series.quantile(0.75) else ("#f59e0b" if val > val_series.quantile(0.5) else "#3b82f6")
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
            popup=folium.Popup(f"<b>{row.get('Complaint_ID')}</b><br>Damage: {row.get('Road_Damage_Percent', 0):.1f}%", max_width=260),
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
    """Aggregates maintenance priority scores and counts across administrative zones."""
    if summary_df is None or "Zone" not in summary_df.columns:
        return pd.DataFrame()
    grouped = summary_df.groupby("Zone").agg(
        Total_Roads=("Road_ID", "count"),
        High_Priority_Roads=("Priority_Level", lambda x: (x == "High").sum()),
        Medium_Priority_Roads=("Priority_Level", lambda x: (x == "Medium").sum()),
        Low_Priority_Roads=("Priority_Level", lambda x: (x == "Low").sum()),
        Mean_Maintenance_Score=("Maintenance_Score", "mean"),
        Max_Maintenance_Score=("Maintenance_Score", "max")
    ).reset_index()
    grouped["High_Priority_Pct"] = (grouped["High_Priority_Roads"] / grouped["Total_Roads"] * 100).round(1)
    grouped["Mean_Maintenance_Score"] = grouped["Mean_Maintenance_Score"].round(2)
    return grouped.sort_values(by="High_Priority_Roads", ascending=False)
