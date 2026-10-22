"""
GIS and Geospatial Infrastructure Analysis Module.
Utilizes GeoPandas, Shapely, and Plotly OpenStreetMap engines for geospatial analysis.
Completely deterministic and explainable GIS analytics.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Any, List, Optional

# Optional GeoPandas / Shapely integration
try:
    import geopandas as gpd
    from shapely.geometry import Point, LineString
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False

# Optional Folium
try:
    import folium
    from folium.plugins import MarkerCluster, HeatMap
    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from config import PRIORITY_COLORS, SEVERITY_COLORS, MAP_DEFAULT_CENTER, MAP_DEFAULT_ZOOM
from maps.map_utils import create_base_map, add_legend_to_map, add_tile_layers

# ----------------------------------------------------------------------
# VECTOR & GEODATAFRAME CONVERSION
# ----------------------------------------------------------------------

def convert_to_geodataframe(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude"):
    """
    Converts a pandas DataFrame with lat/long columns into a GeoPandas GeoDataFrame with WGS84 CRS (EPSG:4326).
    """
    if not HAS_GEOPANDAS or df is None or df.empty:
        return df
        
    valid_coords = df.dropna(subset=[lat_col, lon_col]).copy()
    if valid_coords.empty:
        return valid_coords
        
    geometry = [Point(xy) for xy in zip(valid_coords[lon_col], valid_coords[lat_col])]
    gdf = gpd.GeoDataFrame(valid_coords, geometry=geometry, crs="EPSG:4326")
    return gdf


def get_geospatial_summary(df: pd.DataFrame, lat_col: str = "Latitude", lon_col: str = "Longitude") -> Dict[str, Any]:
    """Computes bounding box and geospatial metadata for the urban network."""
    if df is None or df.empty:
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
# 1. NATIVE PLOTLY INTERACTIVE MAPS (100% RELIABLE OPENSTREETMAP TILES)
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
    max_val = plot_df[metric].max() if metric in plot_df.columns and plot_df[metric].max() > 0 else 1.0
    fg = folium.FeatureGroup(name=f"Traffic: {metric}")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        val = row.get(metric, 0)
        norm_val = val / max_val
        if norm_val > 0.66:
            c = "#d9534f"
        elif norm_val > 0.33:
            c = "#f0ad4e"
        else:
            c = "#5cb85c"
        folium.CircleMarker(
            location=[lat, lon], radius=4 + int(norm_val * 6), color=c, fill=True, fill_color=c, fill_opacity=0.75,
            popup=folium.Popup(f"<b>{row.get('Road_Name', 'Road')}</b><br>{metric}: {val:,.1f}", max_width=250),
            tooltip=f"{row.get('Road_Name', '')}: {val:,.1f}"
        ).add_to(fg)
    fg.add_to(m)
    folium.LayerControl().add_to(m)
    return m

def build_accident_heatmap(accidents_df: pd.DataFrame, max_points: int = 2500):
    if not HAS_FOLIUM or accidents_df is None or accidents_df.empty:
        return None
    valid = accidents_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta["center_lat"], meta["center_lon"]], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    heat_data = [[row["Latitude"], row["Longitude"], float(row.get("Fatalities", 0) * 2 + 1)] for _, row in plot_df.iterrows()]
    HeatMap(heat_data, radius=12, blur=15, max_zoom=13).add_to(m)
    return m

def build_pothole_cluster_map(potholes_df: pd.DataFrame, max_points: int = 2500):
    if not HAS_FOLIUM or potholes_df is None or potholes_df.empty:
        return None
    valid = potholes_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta["center_lat"], meta["center_lon"]], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    cluster = MarkerCluster(name="Reported Defect Clusters").add_to(m)
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        folium.CircleMarker(
            location=[lat, lon], radius=5, color="#d9534f" if row.get("Severity") == "Critical" else "#f0ad4e",
            fill=True, fill_opacity=0.8,
            popup=f"<b>ID:</b> {row.get('Complaint_ID')}<br><b>Damage:</b> {row.get('Road_Damage_Percent', 0)}%"
        ).add_to(cluster)
    folium.LayerControl().add_to(m)
    return m

def build_maintenance_priority_map(priority_df: pd.DataFrame, max_points: int = 2000):
    if not HAS_FOLIUM or priority_df is None or priority_df.empty:
        return None
    valid = priority_df.dropna(subset=["Latitude", "Longitude"])
    if valid.empty:
        return create_base_map()
    meta = get_geospatial_summary(valid)
    m = create_base_map(center=[meta["center_lat"], meta["center_lon"]], zoom=11)
    plot_df = valid.sample(n=min(max_points, len(valid)), random_state=42) if len(valid) > max_points else valid
    fg_high = folium.FeatureGroup(name="High Priority (Red)")
    fg_med = folium.FeatureGroup(name="Medium Priority (Amber)")
    fg_low = folium.FeatureGroup(name="Low Priority (Green)")
    for _, row in plot_df.iterrows():
        lat, lon = row["Latitude"], row["Longitude"]
        prio = row.get("Priority_Level", "Low")
        score = row.get("Maintenance_Score", 0.0)
        factors = row.get("Contributing_Factors", "N/A")
        color = PRIORITY_COLORS.get(prio, "#5cb85c")
        popup_html = f"""<div style="font-family: Arial, sans-serif; font-size: 12px; width: 200px;">
            <b style="color: {color}; font-size: 14px;">{prio} Priority</b><br>
            <b>Road:</b> {row.get('Road_Name', 'N/A')}<br>
            <b>ID:</b> {row.get('Road_ID', 'N/A')}<br>
            <b>Score:</b> {score:.1f} / 100<br>
            <hr style="margin: 4px 0;">
            <b>Rationale:</b> {factors}
        </div>"""
        marker = folium.CircleMarker(
            location=[lat, lon], radius=5 if prio == "Low" else (7 if prio == "Medium" else 9),
            color=color, fill=True, fill_color=color, fill_opacity=0.85,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{row.get('Road_Name', 'Road')} - {prio} ({score:.1f})"
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
    folium.LayerControl().add_to(m)
    return m

def aggregate_by_administrative_zone(priority_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregates maintenance priority scores and counts across administrative zones."""
    if priority_df is None or priority_df.empty or "Zone" not in priority_df.columns:
        return pd.DataFrame()
    grouped = priority_df.groupby("Zone").agg(
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
