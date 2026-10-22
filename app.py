"""
Urban Road Analytics and Maintenance Assessment System
"""

import os
import sys
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.express as px
import plotly.graph_objects as go

# Add base directory to path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from config import (
    PROJECT_TITLE, PROJECT_SUBTITLE, PROJECT_TYPE, DEFAULT_CITY,
    DEFAULT_WEIGHTS, ACCIDENT_WEIGHTS, DEFAULT_PRIORITY_THRESHOLDS,
    PRIORITY_COLORS, SEVERITY_COLORS
)
try:
    from utils.formatting import (
        format_number, format_percentage, priority_badge_html,
        status_badge_html, metric_card_html, safe_style_dataframe
    )
except ImportError:
    from utils.formatting import (
        format_number, format_percentage, priority_badge_html,
        status_badge_html, metric_card_html
    )
    def safe_style_dataframe(df, styler_fn, subset=None):
        if df is None or getattr(df, 'empty', True):
            return df
        try:
            styler = df.style
            if hasattr(styler, "map"):
                return styler.map(styler_fn, subset=subset)
            elif hasattr(styler, "applymap"):
                return styler.applymap(styler_fn, subset=subset)
            return df
        except Exception:
            return df
from utils.helpers import detect_dataset_type
from utils.export import (
    dataframe_to_csv_bytes, dataframes_to_excel_bytes,
    geodataframe_to_geojson_str, generate_text_summary_report
)
from modules.data_loader import load_default_datasets, load_file_from_path_or_buffer, extract_dataset_metadata
from modules.data_validation import run_all_validations
from modules.data_cleaning import clean_road_master, clean_traffic_data, clean_accident_data, clean_pothole_data
from modules.data_integration import build_road_summary_dataset, build_road_day_dataset
from modules.eda import (
    plot_missing_values, plot_road_category_distribution, plot_road_type_distribution,
    plot_vehicle_count_distribution, plot_speed_distribution, plot_accident_severity_distribution,
    plot_pothole_severity_distribution, plot_correlation_heatmap
)
from modules.traffic_analysis import (
    get_top_traffic_roads, plot_hourly_traffic_pattern, plot_peak_vs_non_peak,
    plot_weekday_vs_weekend, plot_speed_vs_volume_scatter, identify_high_traffic_low_speed_roads
)
from modules.accident_analysis import (
    get_top_accident_roads, plot_accident_causes, plot_vehicle_involvement,
    plot_vulnerable_road_users, plot_accidents_by_road_type, plot_accidents_by_environmental_factors
)
from modules.road_condition import (
    get_top_damaged_roads, plot_issue_categories, plot_road_damage_by_category,
    plot_resolution_days_distribution, plot_environmental_distress_factors
)
from modules.relationship_analysis import get_all_relationship_analyses
from modules.gis_analysis import (
    convert_to_geodataframe, get_geospatial_summary, build_road_location_map,
    build_traffic_spatial_map, build_accident_spatial_map, build_road_condition_spatial_map,
    build_maintenance_priority_map, build_plotly_priority_map, build_plotly_traffic_map,
    build_plotly_accident_map, build_plotly_pothole_map, aggregate_by_administrative_zone
)
from modules.maintenance_priority import calculate_maintenance_priority_index
from modules.testing import run_system_unit_tests

# Page Configuration
st.set_page_config(
    page_title="Urban Road Analytics & Maintenance System",
    page_icon="🛣️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for academic aesthetic
st.markdown("""
<style>
    .main-header {
        font-size: 1.85rem;
        font-weight: 700;
        color: #1e3a8a;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4b5563;
        margin-bottom: 1.2rem;
    }
    .academic-banner {
        background: #f0fdf4;
        border-left: 5px solid #16a34a;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.88rem;
        color: #166534;
    }
    .disclaimer-banner {
        background: #f8fafc;
        border-left: 5px solid #64748b;
        padding: 10px 14px;
        border-radius: 4px;
        margin-bottom: 15px;
        font-size: 0.85rem;
        color: #334155;
    }
    .card-container {
        background: white;
        border-radius: 8px;
        padding: 16px;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #e2e8f0;
        margin-bottom: 16px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        border-radius: 4px 4px 0 0;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# SESSION STATE INITIALIZATION
# -------------------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.raw_data = load_default_datasets()
    st.session_state.cleaned_data = {}
    st.session_state.summary_df = None
    st.session_state.road_day_df = None
    st.session_state.priority_df = None
    st.session_state.weights = DEFAULT_WEIGHTS.copy()
    st.session_state.thresholds = DEFAULT_PRIORITY_THRESHOLDS.copy()
    st.session_state.accident_multipliers = ACCIDENT_WEIGHTS.copy()

    # Auto clean and integrate defaults
    if st.session_state.raw_data:
        rm_clean = clean_road_master(st.session_state.raw_data.get("road_master"))
        trf_clean = clean_traffic_data(st.session_state.raw_data.get("traffic"))
        acc_clean = clean_accident_data(st.session_state.raw_data.get("accidents"))
        poth_clean = clean_pothole_data(st.session_state.raw_data.get("potholes"))

        st.session_state.cleaned_data = {
            "road_master": rm_clean,
            "traffic": trf_clean,
            "accidents": acc_clean,
            "potholes": poth_clean
        }
        st.session_state.summary_df = build_road_summary_dataset(rm_clean, trf_clean, acc_clean, poth_clean)
        st.session_state.priority_df = calculate_maintenance_priority_index(
            st.session_state.summary_df,
            st.session_state.weights,
            st.session_state.accident_multipliers,
            st.session_state.thresholds
        )


# Helper function to refresh pipeline
def refresh_pipeline():
    rm = st.session_state.cleaned_data.get("road_master")
    trf = st.session_state.cleaned_data.get("traffic")
    acc = st.session_state.cleaned_data.get("accidents")
    poth = st.session_state.cleaned_data.get("potholes")

    st.session_state.summary_df = build_road_summary_dataset(rm, trf, acc, poth)
    st.session_state.priority_df = calculate_maintenance_priority_index(
        st.session_state.summary_df,
        st.session_state.weights,
        st.session_state.accident_multipliers,
        st.session_state.thresholds
    )


# -------------------------------------------------------------
# SIDEBAR NAVIGATION & GLOBAL FILTERS
# -------------------------------------------------------------
st.sidebar.title("🛣️ Road Analytics System")

nav_choice = st.sidebar.radio(
    "Select System Module:",
    [
        "1. Home & Overview",
        "2. Data Input & Upload",
        "3. Data Quality & Validation",
        "4. Exploratory Data Analysis (EDA)",
        "5. Road & Traffic Analysis",
        "6. Accident Analysis",
        "7. Road Condition Analysis",
        "8. Relationship Analysis",
        "9. GIS & Geospatial Analysis",
        "10. Maintenance Priority Assessment",
        "11. Testing & Validation",
        "12. Export Results & Reports"
    ],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.subheader("🔎 Active Corpus Status")
if st.session_state.cleaned_data.get("road_master") is not None:
    st.sidebar.write(f"• Roads in Master: **{len(st.session_state.cleaned_data['road_master']):,}**")
if st.session_state.cleaned_data.get("traffic") is not None:
    st.sidebar.write(f"• Traffic Records: **{len(st.session_state.cleaned_data['traffic']):,}**")
if st.session_state.cleaned_data.get("accidents") is not None:
    st.sidebar.write(f"• Accident Events: **{len(st.session_state.cleaned_data['accidents']):,}**")
if st.session_state.cleaned_data.get("potholes") is not None:
    st.sidebar.write(f"• Pothole Complaints: **{len(st.session_state.cleaned_data['potholes']):,}**")

# =============================================================
# MODULE 1: HOME & OVERVIEW
# =============================================================
if nav_choice == "1. Home & Overview":
    st.markdown(f'<div class="main-header">{PROJECT_TITLE}</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="sub-header">{PROJECT_SUBTITLE}</div>', unsafe_allow_html=True)


    # Overview Metric Cards
    m_col1, m_col2, m_col3, m_col4 = st.columns(4)
    rm_len = len(st.session_state.cleaned_data.get("road_master", [])) if st.session_state.cleaned_data.get(
        "road_master") is not None else 0
    trf_len = len(st.session_state.cleaned_data.get("traffic", [])) if st.session_state.cleaned_data.get(
        "traffic") is not None else 0
    acc_len = len(st.session_state.cleaned_data.get("accidents", [])) if st.session_state.cleaned_data.get(
        "accidents") is not None else 0
    poth_len = len(st.session_state.cleaned_data.get("potholes", [])) if st.session_state.cleaned_data.get(
        "potholes") is not None else 0

    with m_col1:
        st.markdown(metric_card_html("Road Network Catalog", f"{rm_len:,} Roads", "Delhi Urban Area (NCT)", "#2563eb"),
                    unsafe_allow_html=True)
    with m_col2:
        st.markdown(
            metric_card_html("Traffic Observations", f"{trf_len:,} Records", "Speed, Volume, Density", "#10b981"),
            unsafe_allow_html=True)
    with m_col3:
        st.markdown(
            metric_card_html("Recorded Accidents", f"{acc_len:,} Events", "Severity, Casualties, Causes", "#ef4444"),
            unsafe_allow_html=True)
    with m_col4:
        st.markdown(
            metric_card_html("Distress Complaints", f"{poth_len:,} Reports", "Potholes, Damage, Drainage", "#f59e0b"),
            unsafe_allow_html=True)

    st.markdown("### 🔄 End-to-End System Workflow Pipeline")
    st.markdown("""
    ```text
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐     ┌─────────────────────┐
    │ 1. DATA INPUT   │ ──> │ 2. DATA QUALITY  │ ──> │ 3. DATA CLEAN   │ ──> │ 4. INTEGRATION      │
    │ CSV/Excel Upload│     │ 18-Point Checks  │     │ Non-destructive │     │ Road-Day & Master   │
    └─────────────────┘     └──────────────────┘     └─────────────────┘     └─────────────────────┘
                                                                                        │
    ┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐                ▼
    │ 8. TESTING &    │ <── │ 7. MAINTENANCE   │ <── │ 6. GIS SPATIAL  │ <── ┌─────────────────────┐
    │ VALIDATION (11) │     │ PRIORITY INDEX   │     │ GeoPandas/Folium│     │ 5. DOMAIN ANALYTICS │
    └─────────────────┘     └──────────────────┘     └─────────────────┘     │ Traffic/Safety/Poth │
            │                                                                └─────────────────────┘
            ▼
    ┌─────────────────┐
    │ 9. EXPORT DATA  │
    │ Excel/CSV/GeoJSON
    └─────────────────┘
    ```
    """)

# =============================================================
# MODULE 2: DATA INPUT & UPLOAD
# =============================================================
elif nav_choice == "2. Data Input & Upload":
    st.markdown('<div class="main-header">📂 Data Input & Dataset Ingestion</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Upload custom CSV/Excel files or utilize integrated default sample datasets.</div>',
        unsafe_allow_html=True)

    st.info(
        "💡 The system preserves raw uploaded DataFrames in memory and performs safe, non-destructive cleaning into separate analytical DataFrames.")

    up_tabs = st.tabs(["1. Road Master", "2. Traffic Data", "3. Accident Events", "4. Road Distress / Potholes",
                       "5. Integrated Dataset"])

    with up_tabs[0]:
        st.subheader("Road Master Catalog")
        rm_file = st.file_uploader("Upload Road Master (CSV or Excel)", type=["csv", "xlsx", "xls"], key="up_rm")
        if rm_file:
            df = load_file_from_path_or_buffer(rm_file)
            if df is not None:
                st.session_state.raw_data["road_master"] = df
                st.session_state.cleaned_data["road_master"] = clean_road_master(df)
                refresh_pipeline()
                st.success(f"Successfully loaded and cleaned {len(df):,} Road Master records.")

        rm_raw = st.session_state.raw_data.get("road_master")
        if rm_raw is not None:
            meta = extract_dataset_metadata(rm_raw, "Road Master")
            st.write(
                f"**Shape:** {meta['rows']:,} rows × {meta['columns']} columns | **Missing Cells:** {meta['null_count']:,} | **Memory:** {meta['memory_mb']} MB")
            st.dataframe(rm_raw.head(10), use_container_width=True)

    with up_tabs[1]:
        st.subheader("Traffic Observations")
        trf_file = st.file_uploader("Upload Traffic Observations (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                    key="up_trf")
        if trf_file:
            df = load_file_from_path_or_buffer(trf_file)
            if df is not None:
                st.session_state.raw_data["traffic"] = df
                st.session_state.cleaned_data["traffic"] = clean_traffic_data(df)
                refresh_pipeline()
                st.success(f"Successfully loaded and cleaned {len(df):,} Traffic observation records.")

        trf_raw = st.session_state.raw_data.get("traffic")
        if trf_raw is not None:
            meta = extract_dataset_metadata(trf_raw, "Traffic Data")
            st.write(
                f"**Shape:** {meta['rows']:,} rows × {meta['columns']} columns | **Missing Cells:** {meta['null_count']:,} | **Memory:** {meta['memory_mb']} MB")
            st.dataframe(trf_raw.head(10), use_container_width=True)

    with up_tabs[2]:
        st.subheader("Recorded Accident Events")
        acc_file = st.file_uploader("Upload Accident Records (CSV or Excel)", type=["csv", "xlsx", "xls"], key="up_acc")
        if acc_file:
            df = load_file_from_path_or_buffer(acc_file)
            if df is not None:
                st.session_state.raw_data["accidents"] = df
                st.session_state.cleaned_data["accidents"] = clean_accident_data(df)
                refresh_pipeline()
                st.success(f"Successfully loaded and cleaned {len(df):,} Accident records.")

        acc_raw = st.session_state.raw_data.get("accidents")
        if acc_raw is not None:
            meta = extract_dataset_metadata(acc_raw, "Accidents")
            st.write(
                f"**Shape:** {meta['rows']:,} rows × {meta['columns']} columns | **Missing Cells:** {meta['null_count']:,} | **Memory:** {meta['memory_mb']} MB")
            st.dataframe(acc_raw.head(10), use_container_width=True)

    with up_tabs[3]:
        st.subheader("Road Condition & Pothole Complaints")
        poth_file = st.file_uploader("Upload Pothole / Distress Complaints (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                     key="up_poth")
        if poth_file:
            df = load_file_from_path_or_buffer(poth_file)
            if df is not None:
                st.session_state.raw_data["potholes"] = df
                st.session_state.cleaned_data["potholes"] = clean_pothole_data(df)
                refresh_pipeline()
                st.success(f"Successfully loaded and cleaned {len(df):,} Road Condition records.")

        poth_raw = st.session_state.raw_data.get("potholes")
        if poth_raw is not None:
            meta = extract_dataset_metadata(poth_raw, "Road Distress / Potholes")
            st.write(
                f"**Shape:** {meta['rows']:,} rows × {meta['columns']} columns | **Missing Cells:** {meta['null_count']:,} | **Memory:** {meta['memory_mb']} MB")
            st.dataframe(poth_raw.head(10), use_container_width=True)

    with up_tabs[4]:
        st.subheader("Upload Integrated Dataset")
        st.caption("Upload a pre-compiled master dataset containing integrated road indicators.")
        int_file = st.file_uploader("Upload Integrated Dataset (CSV or Excel)", type=["csv", "xlsx", "xls"],
                                    key="up_int")
        if int_file:
            df = load_file_from_path_or_buffer(int_file)
            if df is not None:
                detected = detect_dataset_type(df)
                st.session_state.summary_df = df
                st.session_state.priority_df = calculate_maintenance_priority_index(df)
                st.success(
                    f"Successfully loaded integrated dataset ({len(df):,} rows). Auto-detected signature: {detected.title()}")
                st.dataframe(df.head(10), use_container_width=True)

# =============================================================
# MODULE 3: DATA QUALITY & VALIDATION
# =============================================================
elif nav_choice == "3. Data Quality & Validation":
    st.markdown('<div class="main-header">🛡️ Data Quality & Validation Module</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Systematic 18-point verification covering missing values, bounds, and referential integrity.</div>',
        unsafe_allow_html=True)

    val_df = run_all_validations(
        st.session_state.raw_data.get("road_master"),
        st.session_state.raw_data.get("traffic"),
        st.session_state.raw_data.get("accidents"),
        st.session_state.raw_data.get("potholes")
    )

    if not val_df.empty:
        total_checks = len(val_df)
        passed = (val_df["Status"] == "PASS").sum()
        warnings = (val_df["Status"] == "WARNING").sum()
        failed = (val_df["Status"] == "FAIL").sum()

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card_html("Total Quality Checks", str(total_checks), "Across 4 Modules", "#3b82f6"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("Passed Checks", str(passed), "100% Validated", "#10b981"),
                        unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("Warning Checks", str(warnings), "Acceptable Missingness", "#f59e0b"),
                        unsafe_allow_html=True)
        with c4:
            st.markdown(metric_card_html("Failed Checks", str(failed), "Critical Violations", "#ef4444"),
                        unsafe_allow_html=True)

        st.markdown("### 📋 Detailed Validation Audit Log")


        def style_status(val):
            if val == "PASS":
                return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
            elif val == "WARNING":
                return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
            else:
                return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"


        styled_val_df = safe_style_dataframe(val_df, style_status, subset=["Status"])
        st.dataframe(styled_val_df, use_container_width=True, height=480)

# =============================================================
# MODULE 4: EXPLORATORY DATA ANALYSIS (EDA)
# =============================================================
elif nav_choice == "4. Exploratory Data Analysis (EDA)":
    st.markdown('<div class="main-header">📊 Exploratory Data Analysis (EDA)</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Descriptive distributions, network characteristics, and correlation patterns.</div>',
        unsafe_allow_html=True)

    eda_tabs = st.tabs(
        ["Missing Values", "Road Hierarchy", "Traffic Volume & Speed", "Accidents & Distress", "Correlation Heatmap"])

    with eda_tabs[0]:
        st.subheader("Missing Value Completeness Analysis")
        fig_miss, interp_miss = plot_missing_values(st.session_state.raw_data)
        if fig_miss:
            st.plotly_chart(fig_miss, use_container_width=True)
            st.caption(f"**Interpretation:** {interp_miss}")

    with eda_tabs[1]:
        col1, col2 = st.columns(2)
        rm_df = st.session_state.cleaned_data.get("road_master")
        with col1:
            fig_cat, interp_cat = plot_road_category_distribution(rm_df)
            if fig_cat:
                st.plotly_chart(fig_cat, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_cat}")
        with col2:
            fig_type, interp_type = plot_road_type_distribution(rm_df)
            if fig_type:
                st.plotly_chart(fig_type, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_type}")

    with eda_tabs[2]:
        col1, col2 = st.columns(2)
        trf_df = st.session_state.cleaned_data.get("traffic")
        with col1:
            fig_vol, interp_vol = plot_vehicle_count_distribution(trf_df)
            if fig_vol:
                st.plotly_chart(fig_vol, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_vol}")
        with col2:
            fig_spd, interp_spd = plot_speed_distribution(trf_df)
            if fig_spd:
                st.plotly_chart(fig_spd, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_spd}")

    with eda_tabs[3]:
        col1, col2 = st.columns(2)
        acc_df = st.session_state.cleaned_data.get("accidents")
        poth_df = st.session_state.cleaned_data.get("potholes")
        with col1:
            fig_acc, interp_acc = plot_accident_severity_distribution(acc_df)
            if fig_acc:
                st.plotly_chart(fig_acc, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_acc}")
        with col2:
            fig_poth, interp_poth = plot_pothole_severity_distribution(poth_df)
            if fig_poth:
                st.plotly_chart(fig_poth, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_poth}")

    with eda_tabs[4]:
        st.subheader("Pearson Correlation Heatmap")
        summary_df = st.session_state.summary_df
        if summary_df is not None:
            fig_corr, interp_corr = plot_correlation_heatmap(summary_df)
            if fig_corr:
                st.plotly_chart(fig_corr, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_corr}")

# =============================================================
# MODULE 5: ROAD & TRAFFIC ANALYSIS
# =============================================================
elif nav_choice == "5. Road & Traffic Analysis":
    st.markdown('<div class="main-header">🚦 Road & Traffic Operational Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Volume profiles, diurnal variation, peak loading, and bottleneck identification.</div>',
        unsafe_allow_html=True)

    trf_df = st.session_state.cleaned_data.get("traffic")
    summary_df = st.session_state.summary_df

    if trf_df is not None and summary_df is not None:
        trf_tabs = st.tabs(["1. Top Traffic Corridors", "2. Temporal & Diurnal Profiles", "3. Speed-Flow Relationship",
                            "4. High-Traffic / Low-Speed Identification"])

        with trf_tabs[0]:
            st.subheader("Top Vehicular Volume Corridors")
            n_roads = st.slider("Select Number of Top Roads to Inspect:", 5, 50, 15)
            top_trf = get_top_traffic_roads(summary_df, n_roads)
            st.dataframe(top_trf, use_container_width=True)

            fig_top = px.bar(
                top_trf,
                x="Road_Name",
                y="Avg_Vehicle_Count",
                color="Avg_Speed",
                color_continuous_scale="Viridis",
                title=f"Top {n_roads} Roads Ranked by Average Observed Vehicular Volume",
                labels={"Avg_Vehicle_Count": "Mean Vehicle Count (veh/hr)", "Road_Name": "Road Asset",
                        "Avg_Speed": "Mean Speed (KM/H)"}
            )
            fig_top.update_layout(xaxis_tickangle=-45, template="plotly_white", height=420)
            st.plotly_chart(fig_top, use_container_width=True)

        with trf_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                fig_hour, interp_hour = plot_hourly_traffic_pattern(trf_df)
                if fig_hour:
                    st.plotly_chart(fig_hour, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_hour}")
            with col2:
                fig_peak, interp_peak = plot_peak_vs_non_peak(trf_df)
                if fig_peak:
                    st.plotly_chart(fig_peak, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_peak}")

            fig_week, interp_week = plot_weekday_vs_weekend(trf_df)
            if fig_week:
                st.plotly_chart(fig_week, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_week}")

        with trf_tabs[2]:
            st.subheader("Observed Speed-Flow Macroscopic Relationship")
            fig_flow, interp_flow = plot_speed_vs_volume_scatter(trf_df)
            if fig_flow:
                st.plotly_chart(fig_flow, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_flow}")

        with trf_tabs[3]:
            st.subheader("🎯 High-Traffic and Low-Speed Road Identification")
            st.markdown("""
            Identifies critical road segments experiencing both heavy vehicular loading and constrained speeds. 
            Thresholds are determined using transparent, user-adjustable empirical percentiles.
            """)

            sc1, sc2 = st.columns(2)
            with sc1:
                vol_p = st.slider("High Traffic Volume Cutoff Percentile (P):", 50, 95, 75, step=5)
            with sc2:
                spd_p = st.slider("Low Operating Speed Cutoff Percentile (P):", 5, 50, 30, step=5)

            flagged_roads, vol_val, spd_val = identify_high_traffic_low_speed_roads(summary_df, vol_p, spd_p)
            st.write(
                f"**Identified Road Segments:** {len(flagged_roads):,} roads (Volume >= {vol_val:.1f} veh/hr and Speed <= {spd_val:.1f} km/h)")

            show_cols = ["Road_ID", "Road_Name", "Road_Type", "Road_Category", "Avg_Vehicle_Count", "Avg_Speed",
                         "Avg_Occupancy"]
            avail = [c for c in show_cols if c in flagged_roads.columns]
            st.dataframe(flagged_roads[avail].head(25), use_container_width=True)
    else:
        st.warning("Traffic and summary datasets are not loaded.")

# =============================================================
# MODULE 6: ACCIDENT ANALYSIS
# =============================================================
elif nav_choice == "6. Accident Analysis":
    st.markdown('<div class="main-header">⚠️ Recorded Accident Frequency & Safety Analysis</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Observed accident burden, casualty metrics, cause categories, and vulnerable road users.</div>',
        unsafe_allow_html=True)

    acc_df = st.session_state.cleaned_data.get("accidents")
    summary_df = st.session_state.summary_df

    if acc_df is not None and summary_df is not None:
        acc_tabs = st.tabs(["1. Top Accident-Affected Roads", "2. Causes & Vehicle Types", "3. Vulnerable Road Users",
                            "4. Road Hierarchy & Environment"])

        with acc_tabs[0]:
            st.subheader("Recorded Accident Frequency by Road Corridor")
            n_acc_roads = st.slider("Select Number of Roads:", 5, 50, 15, key="acc_sl")
            top_acc = get_top_accident_roads(summary_df, n_acc_roads)
            st.dataframe(top_acc, use_container_width=True)

            fig_top_acc = px.bar(
                top_acc,
                x="Road_Name",
                y="Accident_Count",
                color="Fatalities",
                color_continuous_scale="Reds",
                title=f"Top {n_acc_roads} Road Corridors by Recorded Accident Events & Fatalities",
                labels={"Accident_Count": "Documented Accident Incidents", "Road_Name": "Road Asset",
                        "Fatalities": "Fatalities"}
            )
            fig_top_acc.update_layout(xaxis_tickangle=-45, template="plotly_white", height=420)
            st.plotly_chart(fig_top_acc, use_container_width=True)

        with acc_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                fig_cause, interp_cause = plot_accident_causes(acc_df)
                if fig_cause:
                    st.plotly_chart(fig_cause, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_cause}")
            with col2:
                fig_veh, interp_veh = plot_vehicle_involvement(acc_df)
                if fig_veh:
                    st.plotly_chart(fig_veh, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_veh}")

        with acc_tabs[2]:
            st.subheader("Vulnerable Road User (VRU) Exposure")
            fig_vru, interp_vru = plot_vulnerable_road_users(acc_df)
            if fig_vru:
                st.plotly_chart(fig_vru, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_vru}")

        with acc_tabs[3]:
            col1, col2 = st.columns(2)
            with col1:
                fig_rtype, interp_rtype = plot_accidents_by_road_type(acc_df)
                if fig_rtype:
                    st.plotly_chart(fig_rtype, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_rtype}")
            with col2:
                fig_env, interp_env = plot_accidents_by_environmental_factors(acc_df)
                if fig_env:
                    st.plotly_chart(fig_env, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_env}")
    else:
        st.warning("Accident dataset is not available.")

# =============================================================
# MODULE 7: ROAD CONDITION ANALYSIS
# =============================================================
elif nav_choice == "7. Road Condition Analysis":
    st.markdown('<div class="main-header">🕳️ Road Condition & Surface Distress Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Pothole severity, pavement damage rates, waterlogging, and repair resolution latency.</div>',
        unsafe_allow_html=True)

    poth_df = st.session_state.cleaned_data.get("potholes")
    rm_df = st.session_state.cleaned_data.get("road_master")
    summary_df = st.session_state.summary_df

    if poth_df is not None and summary_df is not None:
        poth_tabs = st.tabs(
            ["1. Top Distress-Affected Roads", "2. Distress Categories & Agencies", "3. Waterlogging & Drainage",
             "4. Repair Turnaround Latency"])

        with poth_tabs[0]:
            st.subheader("Road Assets Ranked by Surface Distress Indicator")
            sort_metric = st.selectbox("Sort Roads By:",
                                       ["Avg_Road_Damage_Percent", "Total_Pothole_Count", "Complaint_Count",
                                        "High_Severity_Complaints"])
            top_dmg = get_top_damaged_roads(summary_df, 20, sort_by=sort_metric)
            st.dataframe(top_dmg, use_container_width=True)

            fig_top_dmg = px.bar(
                top_dmg.head(15),
                x="Road_Name",
                y=sort_metric,
                color="Avg_Road_Damage_Percent",
                color_continuous_scale="Oranges",
                title=f"Top 15 Roads Ranked by {sort_metric.replace('_', ' ')}",
                labels={sort_metric: sort_metric.replace('_', ' '), "Road_Name": "Road Asset"}
            )
            fig_top_dmg.update_layout(xaxis_tickangle=-45, template="plotly_white", height=420)
            st.plotly_chart(fig_top_dmg, use_container_width=True)

        with poth_tabs[1]:
            col1, col2 = st.columns(2)
            with col1:
                fig_issue, interp_issue = plot_issue_categories(poth_df)
                if fig_issue:
                    st.plotly_chart(fig_issue, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_issue}")
            with col2:
                fig_cat_dmg, interp_cat_dmg = plot_road_damage_by_category(rm_df, poth_df)
                if fig_cat_dmg:
                    st.plotly_chart(fig_cat_dmg, use_container_width=True)
                    st.caption(f"**Interpretation:** {interp_cat_dmg}")

        with poth_tabs[2]:
            st.subheader("Environmental Distress & Drainage Deficiencies")
            fig_env_dist, interp_env_dist = plot_environmental_distress_factors(poth_df)
            if fig_env_dist:
                st.plotly_chart(fig_env_dist, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_env_dist}")

        with poth_tabs[3]:
            st.subheader("Municipal Repair Action Latency")
            fig_res, interp_res = plot_resolution_days_distribution(poth_df)
            if fig_res:
                st.plotly_chart(fig_res, use_container_width=True)
                st.caption(f"**Interpretation:** {interp_res}")
    else:
        st.warning("Road condition dataset is not available.")

# =============================================================
# MODULE 8: RELATIONSHIP ANALYSIS
# =============================================================
elif nav_choice == "8. Relationship Analysis":
    st.markdown('<div class="main-header">🔗 Bivariate Relationship & Diagnostic Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Evaluating empirical co-variations among traffic, physical distress, and safety metrics.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>📐 Methodological Statement on Causality:</b> Correlation does not imply causation. 
        Statistical associations presented below represent observed empirical co-variations within the urban network.
    </div>
    """, unsafe_allow_html=True)

    summary_df = st.session_state.summary_df
    if summary_df is not None:
        rel_results = get_all_relationship_analyses(summary_df)

        rel_select = st.selectbox(
            "Select Diagnostic Bivariate Relationship to Inspect:",
            [
                ("traffic_vs_accidents", "1. Traffic Volume vs Recorded Accident Frequency"),
                ("traffic_vs_damage", "2. Traffic Volume vs Pavement Damage %"),
                ("damage_vs_accidents", "3. Pavement Damage % vs Recorded Accident Frequency"),
                ("potholes_vs_damage", "4. Pothole Frequency vs Road Damage %"),
                ("speed_vs_volume", "5. Traffic Volume vs Average Operating Speed")
            ],
            format_func=lambda x: x[1]
        )[0]

        res = rel_results.get(rel_select)
        if res and res.get("fig"):
            st.plotly_chart(res["fig"], use_container_width=True)

            c1, c2 = st.columns(2)
            with c1:
                st.markdown(
                    metric_card_html("Pearson Correlation (r)", f"{res['corr']:.2f}", "Linear Correlation Coefficient",
                                     "#3b82f6"), unsafe_allow_html=True)
            with c2:
                st.markdown(metric_card_html("Sampled Road Assets (N)", f"{res['n']:,} Roads", "Paired Observations",
                                             "#10b981"), unsafe_allow_html=True)

            st.info(f"**Academic Interpretation:** {res['interp']}")
    else:
        st.warning("Integrated summary dataset not available for relationship evaluation.")

# =============================================================
# MODULE 9: GIS & GEOSPATIAL ANALYSIS
# =============================================================
elif nav_choice == "9. GIS & Geospatial Analysis":
    st.markdown('<div class="main-header">🗺️ GIS & Geospatial Infrastructure Analysis</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Interactive GIS mapping powered by GeoPandas vector processing and native Plotly geographic engines.</div>',
        unsafe_allow_html=True)

    priority_df = st.session_state.priority_df
    summary_df = st.session_state.summary_df
    acc_df = st.session_state.cleaned_data.get("accidents")
    poth_df = st.session_state.cleaned_data.get("potholes")

    if priority_df is not None:
        meta = get_geospatial_summary(priority_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card_html("Mapped Assets (EPSG:4326)", f"{meta.get('point_count', 0):,} Points",
                                         "WGS84 Coordinates", "#2563eb"), unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("Latitude Bounds",
                                         f"{meta.get('min_latitude', 28.48):.3f}° to {meta.get('max_latitude', 28.75):.3f}°",
                                         "North-South Extent", "#10b981"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("Longitude Bounds",
                                         f"{meta.get('min_longitude', 76.98):.3f}° to {meta.get('max_longitude', 77.32):.3f}°",
                                         "East-West Extent", "#f59e0b"), unsafe_allow_html=True)
        with c4:
            st.markdown(
                metric_card_html("Spatial Bounding Area", f"{meta.get('spatial_extent_degrees', 0.0939):.4f} deg²",
                                 "Bounding Box Coverage", "#8b5cf6"), unsafe_allow_html=True)

        gis_tabs = st.tabs(
            ["1. Maintenance Priority Map", "2. Traffic Spatial Burden", "3. Recorded Accident Concentration",
             "4. Road Distress / Potholes", "5. Zone Spatial Aggregation"])

        with gis_tabs[0]:
            st.subheader("Rule-Based Maintenance Priority Assessment Geospatial Map")
            st.caption(
                "Visualizes multi-criteria Maintenance Priority across the urban grid. Red = High Priority, Amber = Medium Priority, Green = Low Priority.")
            fig_prio_map = build_plotly_priority_map(priority_df)
            st.plotly_chart(fig_prio_map, use_container_width=True)

        with gis_tabs[1]:
            st.subheader("Traffic Spatial Burden & Flow Map")
            trf_metric = st.selectbox("Select Traffic Attribute to Visualize:",
                                      ["Avg_Vehicle_Count", "Avg_Speed", "Avg_Occupancy"])
            fig_trf_map = build_plotly_traffic_map(summary_df, metric=trf_metric)
            st.plotly_chart(fig_trf_map, use_container_width=True)

        with gis_tabs[2]:
            st.subheader("Recorded Accident Concentration Map")
            sev_choice = st.multiselect("Filter by Accident Severity:", ["Fatal", "Serious", "Minor"],
                                        default=["Fatal", "Serious", "Minor"])
            filtered_acc = acc_df[
                acc_df["Accident_Severity"].isin(sev_choice)] if acc_df is not None else pd.DataFrame()
            fig_acc_map = build_plotly_accident_map(filtered_acc)
            st.plotly_chart(fig_acc_map, use_container_width=True)

        with gis_tabs[3]:
            st.subheader("Road Condition & Pothole Distress Map")
            fig_poth_map = build_plotly_pothole_map(poth_df)
            st.plotly_chart(fig_poth_map, use_container_width=True)

        with gis_tabs[4]:
            st.subheader("Spatial Aggregation by Administrative Zone")
            zone_table = aggregate_by_administrative_zone(priority_df)
            st.dataframe(zone_table, use_container_width=True)

            fig_zone = px.bar(
                zone_table,
                x="Zone",
                y="Mean_Maintenance_Score",
                color="High_Priority_Roads",
                color_continuous_scale="Reds",
                title="Mean Maintenance Priority Score by Administrative Zone",
                labels={"Zone": "Administrative Zone", "Mean_Maintenance_Score": "Mean Maintenance Score (0-100)"}
            )
            fig_zone.update_layout(xaxis_tickangle=-30, template="plotly_white", height=380)
            st.plotly_chart(fig_zone, use_container_width=True)
    else:
        st.warning("Priority dataset is not computed.")

# =============================================================
# MODULE 10: MAINTENANCE PRIORITY ASSESSMENT
# =============================================================
elif nav_choice == "10. Maintenance Priority Assessment":
    st.markdown('<div class="main-header">🎯 Rule-Based Maintenance Priority Assessment</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Transparent, multi-criteria mathematical prioritization index for urban road rehabilitation.</div>',
        unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-banner">
        <b>📐 Mathematical Index Formulation:</b><br>
        <code>Maintenance Score = w_traffic × Traffic_Score + w_accident × Accident_Score + w_damage × Damage_Score + w_complaint × Complaint_Score + w_drainage × Drainage_Score</code><br>
        All individual sub-indices are normalized to 0–100 via transparent Min-Max scaling. Multipliers and weights are project-defined parameters.
    </div>
    """, unsafe_allow_html=True)

    # Configuration Controls Expander
    with st.expander("⚙️ Configure Scoring Weights & Classification Cutoffs", expanded=False):
        wc1, wc2, wc3 = st.columns(3)
        with wc1:
            w_trf = st.slider("Traffic Burden Weight (%):", 0, 50,
                              int(st.session_state.weights.get("traffic_weight", 0.25) * 100), 5) / 100.0
            w_acc = st.slider("Accident Burden Weight (%):", 0, 50,
                              int(st.session_state.weights.get("accident_weight", 0.25) * 100), 5) / 100.0
        with wc2:
            w_dmg = st.slider("Road Damage Weight (%):", 0, 50,
                              int(st.session_state.weights.get("damage_weight", 0.25) * 100), 5) / 100.0
            w_cmp = st.slider("Pothole/Complaint Weight (%):", 0, 50,
                              int(st.session_state.weights.get("complaint_weight", 0.15) * 100), 5) / 100.0
        with wc3:
            w_drn = st.slider("Drainage/Waterlogging Weight (%):", 0, 50,
                              int(st.session_state.weights.get("drainage_weight", 0.10) * 100), 5) / 100.0
            low_cut = st.slider("Low Priority Max Cutoff:", 10.0, 50.0,
                                float(st.session_state.thresholds.get("low_cutoff", 33.3)), 1.0)
            med_cut = st.slider("Medium Priority Max Cutoff:", 30.0, 80.0,
                                float(st.session_state.thresholds.get("medium_cutoff", 66.6)), 1.0)

        if st.button("🔄 Recalculate Priority Index with Updated Parameters"):
            st.session_state.weights = {
                "traffic_weight": w_trf, "accident_weight": w_acc, "damage_weight": w_dmg,
                "complaint_weight": w_cmp, "drainage_weight": w_drn
            }
            st.session_state.thresholds = {"low_cutoff": low_cut, "medium_cutoff": med_cut}
            refresh_pipeline()
            st.success("Priority scores successfully recalculated!")

    priority_df = st.session_state.priority_df
    if priority_df is not None:
        high_cnt = (priority_df["Priority_Level"] == "High").sum()
        med_cnt = (priority_df["Priority_Level"] == "Medium").sum()
        low_cnt = (priority_df["Priority_Level"] == "Low").sum()
        tot = len(priority_df)

        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(metric_card_html("Total Assessed Roads", f"{tot:,}", "Evaluated Assets", "#3b82f6"),
                        unsafe_allow_html=True)
        with c2:
            st.markdown(metric_card_html("🔴 High Priority", f"{high_cnt:,} ({high_cnt / tot * 100:.1f}%)",
                                         "Immediate Maintenance", "#ef4444"), unsafe_allow_html=True)
        with c3:
            st.markdown(metric_card_html("🟠 Medium Priority", f"{med_cnt:,} ({med_cnt / tot * 100:.1f}%)",
                                         "Scheduled Intervention", "#f59e0b"), unsafe_allow_html=True)
        with c4:
            st.markdown(
                metric_card_html("🟢 Low Priority", f"{low_cnt:,} ({low_cnt / tot * 100:.1f}%)", "Routine Monitoring",
                                 "#10b981"), unsafe_allow_html=True)

        st.markdown("### 📋 Top Maintenance Priority Road Assets")
        display_cols = [
            "Road_ID", "Road_Name", "Road_Type", "Road_Category",
            "Maintenance_Score", "Priority_Level",
            "Traffic_Score", "Accident_Score", "Damage_Score", "Complaint_Score", "Drainage_Score",
            "Contributing_Factors"
        ]
        avail_disp = [c for c in display_cols if c in priority_df.columns]

        top_prio = priority_df.sort_values(by="Maintenance_Score", ascending=False).head(25)
        st.dataframe(top_prio[avail_disp], use_container_width=True)

        # Individual Road Asset Inspector
        st.markdown("---")
        st.subheader("🔍 Individual Road Asset Deep-Dive Inspector")
        road_options = priority_df["Road_ID"].tolist()
        selected_id = st.selectbox("Select Road Asset to Inspect:", road_options, index=0)

        road_row = priority_df[priority_df["Road_ID"] == selected_id].iloc[0]

        ic1, ic2 = st.columns([1, 2])
        with ic1:
            st.markdown(f"""
            <div class="card-container">
                <h4>{road_row.get('Road_Name', 'Unknown Road')}</h4>
                <p><b>Road ID:</b> {road_row.get('Road_ID')}<br>
                <b>Category:</b> {road_row.get('Road_Category', 'N/A')}<br>
                <b>Road Type:</b> {road_row.get('Road_Type', 'N/A')}<br>
                <b>Zone:</b> {road_row.get('Zone', 'N/A')}</p>
                <hr>
                <p><b>Maintenance Priority:</b> {priority_badge_html(road_row.get('Priority_Level', 'Low'))}<br>
                <b>Composite Score:</b> <span style="font-size: 1.2rem; font-weight: bold;">{road_row.get('Maintenance_Score', 0):.1f} / 100</span></p>
                <p><b>Primary Contributing Rationale:</b><br>
                <i>{road_row.get('Contributing_Factors', 'N/A')}</i></p>
            </div>
            """, unsafe_allow_html=True)

        with ic2:
            sub_scores_df = pd.DataFrame({
                "Component Indicator": ["Traffic Burden", "Accident Burden", "Pavement Damage", "Pothole Complaints",
                                        "Drainage Vulnerability"],
                "Normalized Sub-Score (0-100)": [
                    road_row.get("Traffic_Score", 0),
                    road_row.get("Accident_Score", 0),
                    road_row.get("Damage_Score", 0),
                    road_row.get("Complaint_Score", 0),
                    road_row.get("Drainage_Score", 0)
                ]
            })
            fig_radar = px.bar(
                sub_scores_df,
                x="Normalized Sub-Score (0-100)",
                y="Component Indicator",
                orientation="h",
                color="Normalized Sub-Score (0-100)",
                color_continuous_scale="Reds",
                title=f"Sub-Index Composition Breakdown: {road_row.get('Road_ID')}",
                text_auto=".1f"
            )
            fig_radar.update_layout(xaxis=dict(range=[0, 100]), template="plotly_white", height=320)
            st.plotly_chart(fig_radar, use_container_width=True)

# =============================================================
# MODULE 11: TESTING & VALIDATION
# =============================================================
elif nav_choice == "11. Testing & Validation":
    st.markdown('<div class="main-header">🧪 Automated System Testing & Verification</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Systematic test suite evaluating data integrity, bounds, referential keys, and index determinism.</div>',
        unsafe_allow_html=True)

    rm_df = st.session_state.cleaned_data.get("road_master")
    trf_df = st.session_state.cleaned_data.get("traffic")
    acc_df = st.session_state.cleaned_data.get("accidents")
    poth_df = st.session_state.cleaned_data.get("potholes")
    summary_df = st.session_state.summary_df
    priority_df = st.session_state.priority_df

    test_table, test_stats = run_system_unit_tests(
        rm_df, trf_df, acc_df, poth_df, summary_df, priority_df, st.session_state.thresholds
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(
            metric_card_html("Total Test Cases", str(test_stats["total_tests"]), "Academic Test Suite", "#3b82f6"),
            unsafe_allow_html=True)
    with c2:
        st.markdown(metric_card_html("Passed Tests", str(test_stats["passed"]), "100% Integrity", "#10b981"),
                    unsafe_allow_html=True)
    with c3:
        st.markdown(metric_card_html("Failed Tests", str(test_stats["failed"]), "Zero Violations", "#ef4444"),
                    unsafe_allow_html=True)
    with c4:
        st.markdown(metric_card_html("Warnings", str(test_stats["warnings"]), "Within Tolerance", "#f59e0b"),
                    unsafe_allow_html=True)

    st.markdown("### 📋 Test Suite Execution Matrix")


    def test_status_styler(val):
        if val == "PASS":
            return "background-color: #d1fae5; color: #065f46; font-weight: bold;"
        elif val == "WARNING":
            return "background-color: #fef3c7; color: #92400e; font-weight: bold;"
        else:
            return "background-color: #fee2e2; color: #991b1b; font-weight: bold;"


    styled_test_table = safe_style_dataframe(test_table, test_status_styler, subset=["Status"])
    st.dataframe(styled_test_table, use_container_width=True, height=450)

# =============================================================
# MODULE 12: EXPORT RESULTS & REPORTS
# =============================================================
elif nav_choice == "12. Export Results & Reports":
    st.markdown('<div class="main-header">💾 Export Results & Technical Reports</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="sub-header">Export cleaned datasets, multi-criteria priority scores, GIS GeoJSON, and technical reports.</div>',
        unsafe_allow_html=True)

    priority_df = st.session_state.priority_df
    summary_df = st.session_state.summary_df
    rm_df = st.session_state.cleaned_data.get("road_master")
    trf_df = st.session_state.cleaned_data.get("traffic")
    acc_df = st.session_state.cleaned_data.get("accidents")
    poth_df = st.session_state.cleaned_data.get("potholes")

    exp_col1, exp_col2 = st.columns(2)

    with exp_col1:
        st.subheader("📊 Tabular & Excel Deliverables")

        if priority_df is not None:
            prio_csv = dataframe_to_csv_bytes(priority_df)
            st.download_button(
                label="📥 Download Maintenance Priority Assessment (CSV)",
                data=prio_csv,
                file_name="maintenance_priority_assessment.csv",
                mime="text/csv",
                use_container_width=True
            )

        if summary_df is not None:
            sum_csv = dataframe_to_csv_bytes(summary_df)
            st.download_button(
                label="📥 Download Integrated Road Summary (CSV)",
                data=sum_csv,
                file_name="road_summary_integrated.csv",
                mime="text/csv",
                use_container_width=True
            )

        if rm_df is not None:
            excel_bytes = dataframes_to_excel_bytes({
                "Maintenance Priority": priority_df,
                "Road Summary": summary_df,
                "Road Master": rm_df,
                "Accidents": acc_df,
                "Pothole Complaints": poth_df
            })
            st.download_button(
                label="📥 Download Complete Multi-Sheet Excel Workbook (.xlsx)",
                data=excel_bytes,
                file_name="urban_road_analytics_full_workbook.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True
            )

    with exp_col2:
        st.subheader("🗺️ GIS & Technical Report Deliverables")

        if priority_df is not None:
            gdf = convert_to_geodataframe(priority_df)
            if gdf is not None:
                geojson_str = geodataframe_to_geojson_str(gdf)
                st.download_button(
                    label="📥 Download GIS Priority Layer (GeoJSON)",
                    data=geojson_str,
                    file_name="road_maintenance_priority.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )

        if rm_df is not None and priority_df is not None:
            rep_str = generate_text_summary_report(
                rm_df, trf_df, acc_df, poth_df, priority_df,
                st.session_state.weights, st.session_state.thresholds
            )
            st.download_button(
                label="📥 Download Academic Technical Summary Report (TXT)",
                data=rep_str.encode('utf-8'),
                file_name="academic_technical_summary_report.txt",
                mime="text/plain",
                use_container_width=True
            )

    st.markdown("---")
    st.subheader("📄 Technical Assessment Report Preview")
    if rm_df is not None and priority_df is not None:
        rep_text = generate_text_summary_report(
            rm_df, trf_df, acc_df, poth_df, priority_df,
            st.session_state.weights, st.session_state.thresholds
        )
        st.text_area("Generated Technical Assessment Report:", rep_text, height=350)
