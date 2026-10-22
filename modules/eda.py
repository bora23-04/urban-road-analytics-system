"""
Exploratory Data Analysis (EDA) Module.
Generates Plotly analytical visualizations and academic textual summaries.
"""

import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from typing import Dict, Tuple

def plot_missing_values(df_dict: Dict[str, pd.DataFrame]) -> Tuple[go.Figure, str]:
    """Plot missing values across all active datasets."""
    data = []
    for name, df in df_dict.items():
        if df is not None and not df.empty:
            nulls = df.isnull().sum()
            for col, count in nulls.items():
                if count > 0:
                    pct = (count / len(df)) * 100
                    data.append({"Dataset": name.title().replace("_", " "), "Column": col, "Missing_Count": count, "Missing_Pct": pct})
                    
    if not data:
        fig = go.Figure()
        fig.add_annotation(text="No missing values detected across datasets (100% complete).", 
                           xref="paper", yref="paper", x=0.5, y=0.5, showarrow=False, font=dict(size=14))
        fig.update_layout(title="Missing Value Completeness Analysis", height=300)
        return fig, "All loaded datasets exhibit zero missing values across observed records."
        
    m_df = pd.DataFrame(data)
    fig = px.bar(
        m_df,
        x="Column",
        y="Missing_Pct",
        color="Dataset",
        title="Missing Values by Field (%)",
        labels={"Missing_Pct": "Missing Records (%)", "Column": "Attribute Column"},
        hover_data=["Missing_Count"],
        text_auto=".1f"
    )
    fig.update_layout(template="plotly_white", xaxis_tickangle=-45, height=400)
    
    interp = f"Missing values are observed in {len(m_df)} specific fields, with maximum missingness in '{m_df.loc[m_df['Missing_Pct'].idxmax()]['Column']}' ({m_df['Missing_Pct'].max():.1f}%). Safe imputation and non-destructive filtering are applied."
    return fig, interp

def plot_road_category_distribution(road_master_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot distribution of Road Categories (e.g. MCD, PWD, NHAI)."""
    if road_master_df is None or "Road_Category" not in road_master_df.columns:
        return None, "Road Category attribute not available."
        
    cat_counts = road_master_df["Road_Category"].value_counts().reset_index()
    cat_counts.columns = ["Road_Category", "Count"]
    
    fig = px.bar(
        cat_counts,
        x="Road_Category",
        y="Count",
        color="Road_Category",
        title="Distribution of Road Master Network by Administrative Category",
        labels={"Road_Category": "Managing Agency / Category", "Count": "Number of Roads"},
        text="Count",
        color_discrete_sequence=px.colors.qualitative.Safe
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", showlegend=False, height=400)
    
    top_cat = cat_counts.iloc[0]["Road_Category"]
    top_cnt = cat_counts.iloc[0]["Count"]
    interp = f"The road inventory is dominated by '{top_cat}' managed roads ({top_cnt:,} segments, {(top_cnt/len(road_master_df))*100:.1f}% of total catalog)."
    return fig, interp

def plot_road_type_distribution(road_master_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot distribution of Road Types (e.g. Arterial, Collector, Residential)."""
    if road_master_df is None or "Road_Type" not in road_master_df.columns:
        return None, "Road Type attribute not available."
        
    type_counts = road_master_df["Road_Type"].value_counts().reset_index()
    type_counts.columns = ["Road_Type", "Count"]
    
    fig = px.bar(
        type_counts,
        x="Road_Type",
        y="Count",
        color="Count",
        color_continuous_scale="Viridis",
        title="Distribution of Road Classifications across Urban Grid",
        labels={"Road_Type": "Road Classification", "Count": "Number of Roads"},
        text="Count"
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", height=400)
    
    interp = f"The network spans {len(type_counts)} distinct road hierarchy types, led by {type_counts.iloc[0]['Road_Type']} ({type_counts.iloc[0]['Count']:,} segments)."
    return fig, interp

def plot_vehicle_count_distribution(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Histogram of traffic vehicle counts."""
    if traffic_df is None or "Vehicle_Count" not in traffic_df.columns:
        return None, "Traffic Vehicle Count data not available."
        
    fig = px.histogram(
        traffic_df,
        x="Vehicle_Count",
        nbins=50,
        title="Distribution of Observed Traffic Volume (Vehicle Count per Hour)",
        labels={"Vehicle_Count": "Observed Vehicle Count (vehicles/hr)", "count": "Observation Frequency"},
        marginal="box",
        color_discrete_sequence=["#2b5c8f"]
    )
    fig.update_layout(template="plotly_white", height=400)
    
    median_val = traffic_df["Vehicle_Count"].median()
    q75 = traffic_df["Vehicle_Count"].quantile(0.75)
    interp = f"Traffic observations show a median volume of {median_val:.0f} vehicles/hour, with the 75th percentile at {q75:.0f} vehicles/hour. Most observations concentrate in the moderate volume corridor."
    return fig, interp

def plot_speed_distribution(traffic_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Histogram of average speed in KM/H."""
    if traffic_df is None or "Average_Speed_KMH" not in traffic_df.columns:
        return None, "Average Speed data not available."
        
    fig = px.histogram(
        traffic_df,
        x="Average_Speed_KMH",
        nbins=40,
        title="Distribution of Observed Average Speeds (KM/H)",
        labels={"Average_Speed_KMH": "Average Speed (KM/H)", "count": "Frequency"},
        marginal="box",
        color_discrete_sequence=["#10b981"]
    )
    fig.update_layout(template="plotly_white", height=400)
    
    mean_speed = traffic_df["Average_Speed_KMH"].mean()
    interp = f"Observed vehicular speeds average {mean_speed:.1f} KM/H across monitoring intervals, with typical corridor speeds ranging between {traffic_df['Average_Speed_KMH'].quantile(0.25):.1f} and {traffic_df['Average_Speed_KMH'].quantile(0.75):.1f} KM/H."
    return fig, interp

def plot_accident_severity_distribution(accidents_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Pie/Donut chart of accident severities."""
    if accidents_df is None or "Accident_Severity" not in accidents_df.columns:
        return None, "Accident severity data not available."
        
    counts = accidents_df["Accident_Severity"].value_counts().reset_index()
    counts.columns = ["Accident_Severity", "Count"]
    
    color_map = {"Fatal": "#d9534f", "Serious": "#f0ad4e", "Minor": "#5bc0de"}
    
    fig = px.pie(
        counts,
        names="Accident_Severity",
        values="Count",
        hole=0.45,
        title="Recorded Accident Events by Severity Level",
        color="Accident_Severity",
        color_discrete_map=color_map
    )
    fig.update_layout(template="plotly_white", height=400)
    
    fatal_pct = (counts[counts['Accident_Severity'] == 'Fatal']['Count'].sum() / counts['Count'].sum()) * 100 if 'Fatal' in counts['Accident_Severity'].values else 0
    interp = f"Recorded accidents comprise {fatal_pct:.1f}% fatal incidents, with the remainder categorized as serious or minor incidents requiring varying maintenance and safety interventions."
    return fig, interp

def plot_pothole_severity_distribution(potholes_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Bar chart of pothole/complaint severity."""
    if potholes_df is None or "Severity" not in potholes_df.columns:
        return None, "Complaint severity data not available."
        
    counts = potholes_df["Severity"].value_counts().reset_index()
    counts.columns = ["Severity", "Count"]
    
    fig = px.bar(
        counts,
        x="Severity",
        y="Count",
        color="Severity",
        title="Observed Road Distress / Pothole Severity Breakdown",
        labels={"Severity": "Severity Grade", "Count": "Reported Complaints"},
        text="Count",
        color_discrete_map={"Critical": "#d9534f", "High": "#f0ad4e", "Medium": "#337ab7", "Low": "#5cb85c"}
    )
    fig.update_traces(textposition='outside')
    fig.update_layout(template="plotly_white", height=400)
    
    crit_count = counts[counts['Severity'].isin(['Critical', 'High'])]['Count'].sum()
    interp = f"A combined total of {crit_count:,} complaints ({(crit_count/len(potholes_df))*100:.1f}%) represent High or Critical distress levels on road pavements."
    return fig, interp

def plot_correlation_heatmap(integrated_summary_df: pd.DataFrame) -> Tuple[go.Figure, str]:
    """Plot numerical correlation matrix for integrated road metrics."""
    if integrated_summary_df is None or integrated_summary_df.empty:
        return None, "Integrated summary data not available."
        
    candidate_cols = [
        "Road_Length_KM", "Number_of_Lanes", "Road_Width_M", "Road_Age_Years",
        "Avg_Vehicle_Count", "Avg_Speed", "Avg_Occupancy", "Avg_Travel_Time",
        "Accident_Count", "Fatalities", "Total_Casualties",
        "Complaint_Count", "Total_Pothole_Count", "Avg_Road_Damage_Percent"
    ]
    valid_cols = [c for c in candidate_cols if c in integrated_summary_df.columns]
    
    if len(valid_cols) < 2:
        return None, "Insufficient numerical columns for correlation analysis."
        
    corr = integrated_summary_df[valid_cols].corr().round(2)
    
    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pearson Correlation Matrix of Observed Infrastructure & Operational Indicators"
    )
    fig.update_layout(template="plotly_white", height=550)
    
    interp = "The correlation matrix illustrates linear bivariate relationships among observed metrics. Note that statistical correlation indicates empirical co-variation and does not establish causality."
    return fig, interp
