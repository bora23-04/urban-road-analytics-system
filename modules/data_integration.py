"""
Data Integration Module for Urban Road Analytics and Maintenance Assessment System.
Constructs Road-Day analytical datasets and comprehensive Road Master integrated views.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional

def build_road_day_dataset(
    traffic_df: Optional[pd.DataFrame] = None,
    accidents_df: Optional[pd.DataFrame] = None,
    potholes_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Builds the Road-Day analytical dataset integrated at (Road_ID + Date).
    Carefully merges event aggregations without duplicating records.
    """
    if traffic_df is None or traffic_df.empty:
        return pd.DataFrame()
        
    # Aggregate traffic observations to Road_ID + Date
    trf = traffic_df.copy()
    if 'Date' in trf.columns:
        trf['Date'] = pd.to_datetime(trf['Date'], errors='coerce')
        
    trf_day = trf.groupby(['Road_ID', 'Date']).agg(
        Avg_Vehicle_Count=('Vehicle_Count', 'mean'),
        Max_Vehicle_Count=('Vehicle_Count', 'max'),
        Avg_Speed=('Average_Speed_KMH', 'mean'),
        Avg_Occupancy=('Road_Occupancy_Percent', 'mean'),
        Avg_Travel_Time=('Travel_Time_Min', 'mean')
    ).reset_index()
    
    road_day = trf_day
    
    # Merge accident events at Road_ID + Date
    if accidents_df is not None and not accidents_df.empty and 'Date' in accidents_df.columns:
        acc = accidents_df.copy()
        acc['Date'] = pd.to_datetime(acc['Date'], errors='coerce')
        
        acc_day = acc.groupby(['Road_ID', 'Date']).agg(
            Accident_Count=('Accident_ID', 'count'),
            Fatal_Accident_Count=('Accident_Severity', lambda s: (s.astype(str).str.title() == 'Fatal').sum()),
            Total_Casualties=('Number_of_Casualties', 'sum'),
            Fatalities=('Fatalities', 'sum'),
            Injuries=('Injuries', 'sum')
        ).reset_index()
        
        road_day = road_day.merge(acc_day, on=['Road_ID', 'Date'], how='left')
    else:
        for col in ['Accident_Count', 'Fatal_Accident_Count', 'Total_Casualties', 'Fatalities', 'Injuries']:
            road_day[col] = 0
            
    # Merge pothole / road condition complaints at Road_ID + Date
    date_col = 'Complaint_Date' if potholes_df is not None and 'Complaint_Date' in potholes_df.columns else ('Date' if potholes_df is not None and 'Date' in potholes_df.columns else None)
    if potholes_df is not None and not potholes_df.empty and date_col:
        poth = potholes_df.copy()
        poth['Date'] = pd.to_datetime(poth[date_col], errors='coerce')
        
        poth_day = poth.groupby(['Road_ID', 'Date']).agg(
            Complaint_Count=('Complaint_ID', 'count'),
            Total_Pothole_Count=('Pothole_Count', 'sum') if 'Pothole_Count' in poth.columns else ('Complaint_ID', 'count'),
            Avg_Pothole_Size=('Pothole_Size_M2', 'mean') if 'Pothole_Size_M2' in poth.columns else ('Complaint_ID', lambda x: 0.0),
            Avg_Road_Damage_Percent=('Road_Damage_Percent', 'mean') if 'Road_Damage_Percent' in poth.columns else ('Complaint_ID', lambda x: 0.0),
            High_Severity_Complaints=('Severity', lambda s: (s.astype(str).str.title().isin(['High', 'Critical'])).sum()) if 'Severity' in poth.columns else ('Complaint_ID', lambda x: 0),
            Waterlogging_Complaints=('Waterlogging_Present', lambda s: (s.astype(str).str.title() == 'Yes').sum()) if 'Waterlogging_Present' in poth.columns else ('Complaint_ID', lambda x: 0),
            Drainage_Issues=('Drainage_Issue', lambda s: (s.astype(str).str.title() == 'Yes').sum()) if 'Drainage_Issue' in poth.columns else ('Complaint_ID', lambda x: 0)
        ).reset_index()
        
        road_day = road_day.merge(poth_day, on=['Road_ID', 'Date'], how='left')
    else:
        for col in ['Complaint_Count', 'Total_Pothole_Count', 'Avg_Pothole_Size', 'Avg_Road_Damage_Percent', 
                    'High_Severity_Complaints', 'Waterlogging_Complaints', 'Drainage_Issues']:
            road_day[col] = 0.0
            
    # Fill event count nulls with zero for non-events
    count_cols = ['Accident_Count', 'Fatal_Accident_Count', 'Total_Casualties', 'Fatalities', 'Injuries',
                  'Complaint_Count', 'Total_Pothole_Count', 'High_Severity_Complaints', 
                  'Waterlogging_Complaints', 'Drainage_Issues']
    for col in count_cols:
        if col in road_day.columns:
            road_day[col] = road_day[col].fillna(0).astype(int)
            
    if 'Avg_Road_Damage_Percent' in road_day.columns:
        road_day['Avg_Road_Damage_Percent'] = road_day['Avg_Road_Damage_Percent'].fillna(0.0)
    if 'Avg_Pothole_Size' in road_day.columns:
        road_day['Avg_Pothole_Size'] = road_day['Avg_Pothole_Size'].fillna(0.0)
        
    return road_day

def build_road_summary_dataset(
    road_master_df: pd.DataFrame,
    traffic_df: Optional[pd.DataFrame] = None,
    accidents_df: Optional[pd.DataFrame] = None,
    potholes_df: Optional[pd.DataFrame] = None
) -> pd.DataFrame:
    """
    Builds the complete road-level master dataset integrating all traffic, accident,
    and road condition indicators for priority indexing and GIS analysis.
    """
    if road_master_df is None or road_master_df.empty:
        return pd.DataFrame()
        
    summary = road_master_df.copy()
    
    # 1. Traffic Aggregations
    if traffic_df is not None and not traffic_df.empty and 'Road_ID' in traffic_df.columns:
        trf_agg = traffic_df.groupby('Road_ID').agg(
            Avg_Vehicle_Count=('Vehicle_Count', 'mean'),
            Max_Vehicle_Count=('Vehicle_Count', 'max'),
            Avg_Speed=('Average_Speed_KMH', 'mean'),
            Avg_Occupancy=('Road_Occupancy_Percent', 'mean'),
            Avg_Travel_Time=('Travel_Time_Min', 'mean'),
            Traffic_Observations_Count=('Traffic_ID' if 'Traffic_ID' in traffic_df.columns else 'Road_ID', 'count')
        ).reset_index()
        summary = summary.merge(trf_agg, on='Road_ID', how='left')
    else:
        summary['Avg_Vehicle_Count'] = 0.0
        summary['Max_Vehicle_Count'] = 0.0
        summary['Avg_Speed'] = 0.0
        summary['Avg_Occupancy'] = 0.0
        summary['Avg_Travel_Time'] = 0.0
        summary['Traffic_Observations_Count'] = 0
        
    # 2. Accident Aggregations
    if accidents_df is not None and not accidents_df.empty and 'Road_ID' in accidents_df.columns:
        agg_dict = {
            'Accident_Count': ('Accident_ID' if 'Accident_ID' in accidents_df.columns else 'Road_ID', 'count'),
            'Fatalities': ('Fatalities', 'sum') if 'Fatalities' in accidents_df.columns else ('Road_ID', lambda x: 0),
            'Total_Casualties': ('Number_of_Casualties', 'sum') if 'Number_of_Casualties' in accidents_df.columns else ('Road_ID', lambda x: 0),
            'Injuries': ('Injuries', 'sum') if 'Injuries' in accidents_df.columns else ('Road_ID', lambda x: 0),
        }
        if 'Latitude' in accidents_df.columns and 'Longitude' in accidents_df.columns:
            agg_dict['Acc_Lat'] = ('Latitude', 'mean')
            agg_dict['Acc_Lon'] = ('Longitude', 'mean')
            
        acc_agg = accidents_df.groupby('Road_ID').agg(**agg_dict).reset_index()
        summary = summary.merge(acc_agg, on='Road_ID', how='left')
    else:
        summary['Accident_Count'] = 0
        summary['Fatalities'] = 0
        summary['Total_Casualties'] = 0
        summary['Injuries'] = 0
        
    # 3. Pothole / Condition Aggregations
    if potholes_df is not None and not potholes_df.empty and 'Road_ID' in potholes_df.columns:
        agg_dict = {
            'Complaint_Count': ('Complaint_ID' if 'Complaint_ID' in potholes_df.columns else 'Road_ID', 'count'),
            'Total_Pothole_Count': ('Pothole_Count', 'sum') if 'Pothole_Count' in potholes_df.columns else ('Road_ID', lambda x: 0),
            'Avg_Pothole_Size': ('Pothole_Size_M2', 'mean') if 'Pothole_Size_M2' in potholes_df.columns else ('Road_ID', lambda x: 0.0),
            'Avg_Road_Damage_Percent': ('Road_Damage_Percent', 'mean') if 'Road_Damage_Percent' in potholes_df.columns else ('Road_ID', lambda x: 0.0),
            'High_Severity_Complaints': ('Severity', lambda s: (s.astype(str).str.title().isin(['High', 'Critical'])).sum()) if 'Severity' in potholes_df.columns else ('Road_ID', lambda x: 0),
            'Waterlogging_Complaints': ('Waterlogging_Present', lambda s: (s.astype(str).str.title() == 'Yes').sum()) if 'Waterlogging_Present' in potholes_df.columns else ('Road_ID', lambda x: 0),
            'Drainage_Issues': ('Drainage_Issue', lambda s: (s.astype(str).str.title() == 'Yes').sum()) if 'Drainage_Issue' in potholes_df.columns else ('Road_ID', lambda x: 0)
        }
        if 'Latitude' in potholes_df.columns and 'Longitude' in potholes_df.columns:
            agg_dict['Poth_Lat'] = ('Latitude', 'mean')
            agg_dict['Poth_Lon'] = ('Longitude', 'mean')
            
        poth_agg = potholes_df.groupby('Road_ID').agg(**agg_dict).reset_index()
        summary = summary.merge(poth_agg, on='Road_ID', how='left')
    else:
        summary['Complaint_Count'] = 0
        summary['Total_Pothole_Count'] = 0
        summary['Avg_Pothole_Size'] = 0.0
        summary['Avg_Road_Damage_Percent'] = 0.0
        summary['High_Severity_Complaints'] = 0
        summary['Waterlogging_Complaints'] = 0
        summary['Drainage_Issues'] = 0
        
    # Impute zeros for discrete count metrics
    zero_fill_cols = [
        'Accident_Count', 'Fatalities', 'Total_Casualties', 'Injuries',
        'Complaint_Count', 'Total_Pothole_Count', 'High_Severity_Complaints',
        'Waterlogging_Complaints', 'Drainage_Issues', 'Traffic_Observations_Count'
    ]
    for c in zero_fill_cols:
        if c in summary.columns:
            summary[c] = summary[c].fillna(0).astype(int)
            
    summary['Avg_Road_Damage_Percent'] = summary.get('Avg_Road_Damage_Percent', pd.Series(0, index=summary.index)).fillna(0.0)
    summary['Avg_Vehicle_Count'] = summary.get('Avg_Vehicle_Count', pd.Series(0, index=summary.index)).fillna(0.0)
    summary['Avg_Speed'] = summary.get('Avg_Speed', pd.Series(0, index=summary.index)).fillna(0.0)
    summary['Avg_Occupancy'] = summary.get('Avg_Occupancy', pd.Series(0, index=summary.index)).fillna(0.0)
    summary['Avg_Travel_Time'] = summary.get('Avg_Travel_Time', pd.Series(0, index=summary.index)).fillna(0.0)
    
    # Establish Geographic Coordinates
    # Check if native Latitude/Longitude exists in Road Master
    if 'Latitude' not in summary.columns or summary['Latitude'].isnull().all():
        acc_lat = summary.get('Acc_Lat', pd.Series(np.nan, index=summary.index))
        poth_lat = summary.get('Poth_Lat', pd.Series(np.nan, index=summary.index))
        summary['Latitude'] = acc_lat.combine_first(poth_lat)
        
    if 'Longitude' not in summary.columns or summary['Longitude'].isnull().all():
        acc_lon = summary.get('Acc_Lon', pd.Series(np.nan, index=summary.index))
        poth_lon = summary.get('Poth_Lon', pd.Series(np.nan, index=summary.index))
        summary['Longitude'] = acc_lon.combine_first(poth_lon)
        
    # Drop intermediate columns
    summary = summary.drop(columns=['Acc_Lat', 'Acc_Lon', 'Poth_Lat', 'Poth_Lon'], errors='ignore')
    
    return summary
