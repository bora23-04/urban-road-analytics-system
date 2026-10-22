"""
Export Utilities for Urban Road Analytics and Maintenance Assessment System.
"""

import io
import json
import pandas as pd
import geopandas as gpd

def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """Convert a DataFrame to CSV bytes for Streamlit download."""
    if df is None or df.empty:
        return b""
    return df.to_csv(index=False).encode('utf-8')

def dataframes_to_excel_bytes(dfs_dict: dict) -> bytes:
    """
    Export multiple DataFrames into sheets of a single Excel (.xlsx) workbook.
    """
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet_name, df in dfs_dict.items():
            if df is not None and not df.empty:
                # Excel sheet name limit is 31 chars
                clean_sheet_name = str(sheet_name)[:30]
                df.to_excel(writer, sheet_name=clean_sheet_name, index=False)
    output.seek(0)
    return output.getvalue()

def geodataframe_to_geojson_str(gdf: gpd.GeoDataFrame) -> str:
    """Convert a GeoDataFrame to GeoJSON string."""
    if gdf is None or gdf.empty:
        return "{}"
    # Convert datetime columns to string before exporting GeoJSON to avoid serializing errors
    export_gdf = gdf.copy()
    for col in export_gdf.select_dtypes(include=['datetime64[ns]', 'datetime64']).columns:
        export_gdf[col] = export_gdf[col].dt.strftime('%Y-%m-%d %H:%M:%S')
    return export_gdf.to_json(drop_id=True)

def generate_text_summary_report(
    road_master_df: pd.DataFrame,
    traffic_df: pd.DataFrame,
    accidents_df: pd.DataFrame,
    potholes_df: pd.DataFrame,
    priority_df: pd.DataFrame,
    weights_dict: dict,
    thresholds_dict: dict
) -> str:
    """Generate a clean, academic-grade text assessment report."""
    total_roads = len(road_master_df) if road_master_df is not None else 0
    total_traffic_records = len(traffic_df) if traffic_df is not None else 0
    total_accidents = len(accidents_df) if accidents_df is not None else 0
    total_complaints = len(potholes_df) if potholes_df is not None else 0
    
    high_prio = (priority_df['Priority_Level'] == 'High').sum() if priority_df is not None and 'Priority_Level' in priority_df.columns else 0
    med_prio = (priority_df['Priority_Level'] == 'Medium').sum() if priority_df is not None and 'Priority_Level' in priority_df.columns else 0
    low_prio = (priority_df['Priority_Level'] == 'Low').sum() if priority_df is not None and 'Priority_Level' in priority_df.columns else 0

    lines = [
        "=" * 80,
        "URBAN ROAD ANALYTICS AND MAINTENANCE ASSESSMENT SYSTEM",
        "FINAL ACADEMIC TECHNICAL SUMMARY REPORT",
        "=" * 80,
        "",
        "1. EXECUTIVE PROJECT OVERVIEW",
        f"- Project Type: B.Tech Engineering Major Project Deliverable",
        f"- Analysis Scope: Urban Road Infrastructure Condition & Maintenance Prioritization",
        f"- Methodological Classification: Descriptive Analytics, Geospatial Aggregation, and Rule-Based Indexing",
        f"- Methodological Disclaimer: This system DOES NOT employ predictive/ML models. All metrics are computed",
        f"  from observed historical infrastructure datasets.",
        "",
        "2. DATA INVENTORY & CORPUS SUMMARY",
        f"- Total Road Segments in Master Catalog: {total_roads:,}",
        f"- Total Traffic Observations Analyzed: {total_traffic_records:,}",
        f"- Total Recorded Accident Events: {total_accidents:,}",
        f"- Total Road Condition / Pothole Complaints: {total_complaints:,}",
        "",
        "3. MAINTENANCE PRIORITY CRITERIA & CONFIGURATION",
        f"- Traffic Burden Weight: {weights_dict.get('traffic_weight', 0.25)*100:.1f}%",
        f"- Recorded Accident Burden Weight: {weights_dict.get('accident_weight', 0.25)*100:.1f}%",
        f"- Road Damage Percentage Weight: {weights_dict.get('damage_weight', 0.25)*100:.1f}%",
        f"- Pothole & Complaint Frequency Weight: {weights_dict.get('complaint_weight', 0.15)*100:.1f}%",
        f"- Drainage & Waterlogging Vulnerability Weight: {weights_dict.get('drainage_weight', 0.10)*100:.1f}%",
        f"- Classification Thresholds: Low (< {thresholds_dict.get('low_cutoff', 33.3):.1f}), "
        f"Medium ({thresholds_dict.get('low_cutoff', 33.3):.1f} - {thresholds_dict.get('medium_cutoff', 66.6):.1f}), "
        f"High (>= {thresholds_dict.get('medium_cutoff', 66.6):.1f})",
        "",
        "4. MAINTENANCE ASSESSMENT SUMMARY",
        f"- High Maintenance Priority Roads: {high_prio:,} ({(high_prio/total_roads*100) if total_roads else 0:.1f}%)",
        f"- Medium Maintenance Priority Roads: {med_prio:,} ({(med_prio/total_roads*100) if total_roads else 0:.1f}%)",
        f"- Low Maintenance Priority Roads: {low_prio:,} ({(low_prio/total_roads*100) if total_roads else 0:.1f}%)",
        "",
        "5. TOP 10 HIGHEST MAINTENANCE PRIORITY ROADS",
        "-" * 80
    ]
    
    if priority_df is not None and not priority_df.empty:
        top10 = priority_df.sort_values(by="Maintenance_Score", ascending=False).head(10)
        lines.append(f"{'Road ID':<15} | {'Road Name':<32} | {'Score':<6} | {'Priority':<8} | {'Traffic':<6} | {'Accidents':<6} | {'Damage%':<6}")
        lines.append("-" * 80)
        for _, r in top10.iterrows():
            r_id = str(r.get('Road_ID', 'N/A'))
            r_name = str(r.get('Road_Name', 'N/A'))[:30]
            score = f"{float(r.get('Maintenance_Score', 0)):.1f}"
            prio = str(r.get('Priority_Level', 'N/A'))
            trf = f"{float(r.get('Avg_Vehicle_Count', 0)):.0f}"
            acc = f"{int(r.get('Accident_Count', 0))}"
            dmg = f"{float(r.get('Avg_Road_Damage_Percent', 0)):.1f}%"
            lines.append(f"{r_id:<15} | {r_name:<32} | {score:<6} | {prio:<8} | {trf:<6} | {acc:<6} | {dmg:<6}")
    
    lines.extend([
        "-" * 80,
        "",
        "6. ACADEMIC INTEGRITY & METHODOLOGICAL NOTE",
        "- All scoring rules and normalizations are deterministic, reproducible, and explainable.",
        "- No synthetic or fabricated data points were added beyond the verified source datasets.",
        "=" * 80
    ])
    
    return "\n".join(lines)
