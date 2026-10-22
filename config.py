"""
Configuration file for Urban Road Analytics and Maintenance Assessment System.
Academic B.Tech Project Configuration.
"""

import os

# Project Metadata
PROJECT_TITLE = "Urban Road Analytics & Maintenance Assessment System"
PROJECT_SUBTITLE = "A Transparent, Rule-Based Infrastructure Analytics and Geospatial Maintenance Assessment Framework"
PROJECT_TYPE = "Academic Final-Year B.Tech Project Deliverable"
DEFAULT_CITY = "Delhi Urban Area (NCT)"

# Base Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_RAW_DIR = os.path.join(BASE_DIR, "data", "raw")
DATA_CLEANED_DIR = os.path.join(BASE_DIR, "data", "cleaned")
DATA_OUTPUTS_DIR = os.path.join(BASE_DIR, "data", "outputs")

# Default Sample File Names
DEFAULT_FILES = {
    "road_master": os.path.join(DATA_RAW_DIR, "road_master.csv"),
    "traffic": os.path.join(DATA_RAW_DIR, "traffic.csv"),
    "accidents": os.path.join(DATA_RAW_DIR, "accidents.csv"),
    "potholes": os.path.join(DATA_RAW_DIR, "pothole_complaints.csv"),
}

# Column Signatures for Automatic Dataset Detection
DATASET_SIGNATURES = {
    "road_master": ["road_id", "road_name", "road_type", "road_category", "road_length_km"],
    "traffic": ["traffic_id", "vehicle_count", "average_speed_kmh", "congestion_level"],
    "accidents": ["accident_id", "accident_severity", "number_of_casualties", "fatalities"],
    "potholes": ["complaint_id", "pothole_count", "road_damage_percent", "issue_category"],
    "weather": ["weather_id", "temperature_c", "rainfall_mm", "relative_humidity_percent"],
    "integrated": ["maintenance_score", "priority_level", "traffic_score", "accident_score"]
}

# Default Maintenance Index Scoring Weights (Sum = 100%)
DEFAULT_WEIGHTS = {
    "traffic_weight": 0.25,      # 25%
    "accident_weight": 0.25,     # 25%
    "damage_weight": 0.25,       # 25%
    "complaint_weight": 0.15,    # 15%
    "drainage_weight": 0.10      # 10%
}

# Accident Severity Multipliers (Project-defined parameters)
ACCIDENT_WEIGHTS = {
    "base_count": 1.0,
    "fatalities_multiplier": 3.0,
    "casualties_multiplier": 0.5
}

# Maintenance Priority Thresholds
DEFAULT_PRIORITY_THRESHOLDS = {
    "low_cutoff": 33.33,
    "medium_cutoff": 66.66
}

# Default Map Center (Delhi Coordinates)
MAP_DEFAULT_CENTER = [28.6275, 77.1709]
MAP_DEFAULT_ZOOM = 11

# Color Schemes for Priority
PRIORITY_COLORS = {
    "High": "#d9534f",     # Red / Critical
    "Medium": "#f0ad4e",   # Amber / Moderate
    "Low": "#5cb85c"       # Green / Normal
}

SEVERITY_COLORS = {
    "Fatal": "#d9534f",
    "Serious": "#f0ad4e",
    "Minor": "#5bc0de",
    "Critical": "#d9534f",
    "High": "#f0ad4e",
    "Medium": "#5bc0de",
    "Low": "#5cb85c"
}
