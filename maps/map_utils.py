"""
Map Utilities for Folium GIS Maps.
Generates styled interactive Folium map layers, legends, popups, and clustered overlays.
"""

import os
import sys

# Ensure root directory is always on python sys.path
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

try:
    import folium
    from folium.plugins import MarkerCluster, Fullscreen

    HAS_FOLIUM = True
except ImportError:
    HAS_FOLIUM = False

from config import MAP_DEFAULT_CENTER, MAP_DEFAULT_ZOOM, PRIORITY_COLORS, SEVERITY_COLORS


def create_base_map(center=None, zoom=None, tiles="OpenStreetMap"):
    """Create a standardized Folium Map instance with controls."""
    if not HAS_FOLIUM:
        return None

    c = center if center is not None else MAP_DEFAULT_CENTER
    z = zoom if zoom is not None else MAP_DEFAULT_ZOOM

    m = folium.Map(
        location=c,
        zoom_start=z,
        tiles="OpenStreetMap",
        control_scale=True,
        prefer_canvas=True
    )
    try:
        Fullscreen(position="topright").add_to(m)
    except Exception:
        pass
    return m


def create_priority_popup_html(row: dict) -> str:
    """Construct an informative HTML popup for maintenance priority inspection."""
    r_id = row.get("Road_ID", "N/A")
    name = row.get("Road_Name", "Unknown Road")
    prio = str(row.get("Priority_Level", "Medium"))
    score = float(row.get("Maintenance_Score", 0.0))
    traffic_score = float(row.get("Traffic_Score", 0.0))
    acc_score = float(row.get("Accident_Score", 0.0))
    dmg_score = float(row.get("Damage_Score", 0.0))
    cmp_score = float(row.get("Complaint_Score", 0.0))
    drn_score = float(row.get("Drainage_Score", 0.0))
    factors = row.get("Contributing_Factors", "Standard maintenance cycle.")

    prio_color = PRIORITY_COLORS.get(prio, "#3b82f6")

    html = f"""
    <div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif; font-size: 12px; width: 280px; line-height: 1.4;">
        <div style="background-color: {prio_color}; color: white; padding: 6px 10px; font-weight: 700; border-radius: 4px 4px 0 0;">
            {name}
        </div>
        <div style="padding: 10px; border: 1px solid #e5e7eb; border-top: none; background: white; border-radius: 0 0 4px 4px;">
            <div style="margin-bottom: 6px;"><b>Road ID:</b> {r_id}</div>
            <div style="margin-bottom: 6px;">
                <b>Maintenance Score:</b> <span style="font-size: 14px; font-weight: bold; color: {prio_color};">{score:.1f} / 100</span> 
                ({prio} Priority)
            </div>
            <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 8px 0;">
            <div style="font-size: 11px; color: #374151; margin-bottom: 4px;"><b>Component Sub-Scores:</b></div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 4px; font-size: 11px;">
                <div>🚦 Traffic: <b>{traffic_score:.1f}</b></div>
                <div>⚠️ Safety: <b>{acc_score:.1f}</b></div>
                <div>🕳️ Damage: <b>{dmg_score:.1f}</b></div>
                <div>📋 Complaints: <b>{cmp_score:.1f}</b></div>
            </div>
            <div style="margin-top: 4px; font-size: 11px;">💧 Drainage: <b>{drn_score:.1f}</b></div>
            <hr style="border: 0; border-top: 1px solid #e5e7eb; margin: 8px 0;">
            <div style="font-size: 11px; color: #4b5563;">
                <b>Main Contributing Factors:</b><br>
                <span style="color: #1f2937;">{factors}</span>
            </div>
        </div>
    </div>
    """
    return html


def add_priority_legend_to_map(m):
    """Add floating HTML color legend for Maintenance Priority."""
    if not HAS_FOLIUM or m is None:
        return

    legend_html = """
     <div style="
     position: fixed; 
     bottom: 25px; left: 25px; width: 180px; height: 115px; 
     background-color: white; border:2px solid #bbb; z-index:9999; 
     font-size:12px; padding: 10px; border-radius: 6px; box-shadow: 2px 2px 6px rgba(0,0,0,0.25);">
     <b>Maintenance Priority</b><br>
     <div style="margin-top:4px;"><span style="color:#d9534f; font-size:16px;">●</span> High (>= 66.7)</div>
     <div><span style="color:#f0ad4e; font-size:16px;">●</span> Medium (33.3 - 66.6)</div>
     <div><span style="color:#5cb85c; font-size:16px;">●</span> Low (< 33.3)</div>
     </div>
     """
    try:
        m.get_root().html.add_child(folium.Element(legend_html))
    except Exception:
        pass
