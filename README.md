# Urban Road Analytics and Maintenance Assessment System
### Final-Year B.Tech Academic Engineering Project Software Deliverable

---

## 📌 Executive Abstract & Project Overview

The **Urban Road Analytics and Maintenance Assessment System** is a modular Python-based software application designed to evaluate municipal road networks, vehicular operational characteristics, recorded safety incidents, and pavement distress conditions. 

Unlike black-box machine learning or predictive forecasting models, this system adheres to **strict academic transparency**: all analytical findings, bivariate diagnostic evaluations, spatial GIS mappings, and maintenance prioritization indices are computed **directly from observed historical infrastructure records** using deterministic, explainable mathematical rules.

---

## 🛠️ Technology Stack & System Architecture

| Technology | Role in System Architecture |
| :--- | :--- |
| **Python 3.10+** | Core programming language and mathematical runtime |
| **Streamlit** | Interactive multi-page software interface & state management |
| **Pandas & NumPy** | High-performance data cleaning, aggregation, and integration |
| **GeoPandas & Shapely** | Geospatial vector processing, WGS84 CRS management, and spatial bounds |
| **Folium & Branca** | Interactive web maps, MarkerClusters, and styled HTML popups |
| **Plotly Express & Graph Objects** | Interactive analytical visualizations and speed-flow diagrams |
| **OpenPyXL** | Multi-sheet Excel workbook generation and export |

---

## 📂 Project Directory Structure

```text
urban_road_system/
│
├── app.py                      # Main Streamlit Software Application
├── requirements.txt            # System dependencies
├── README.md                   # Technical documentation & project manual
├── config.py                   # System constants, weights, and map configurations
│
├── data/
│   ├── raw/                    # Raw uploaded datasets (Road Master, Traffic, Accidents, Potholes)
│   ├── cleaned/                # Standardized datasets
│   └── outputs/                # Exported CSV, Excel, and GeoJSON files
│
├── modules/
│   ├── data_loader.py          # File ingestion and dataset auto-detection
│   ├── data_validation.py      # 18-point data quality & referential integrity audit
│   ├── data_cleaning.py        # Safe, non-destructive data cleaning & type casting
│   ├── data_integration.py     # Road-Day and Road-Master integrated analytical views
│   ├── eda.py                  # Exploratory data analysis & distribution charts
│   ├── traffic_analysis.py     # Volume profiles, peak analysis, speed-flow diagnostics
│   ├── accident_analysis.py    # Crash frequency, severity, vulnerable users (VRU)
│   ├── road_condition.py       # Pothole frequency, pavement damage, repair latency
│   ├── relationship_analysis.py# Bivariate correlation & non-causal diagnostic analysis
│   ├── gis_analysis.py         # GeoPandas vector processing & Folium GIS mapping
│   ├── maintenance_priority.py # Rule-based Maintenance Priority Index calculation
│   └── testing.py              # Automated academic unit & integrity test suite
│
├── maps/
│   └── map_utils.py            # Folium map stylers, HTML popups, and legends
│
└── utils/
    ├── helpers.py              # Normalization, string cleaners, and data helpers
    ├── formatting.py           # HTML badges, number formatting, metric cards
    └── export.py               # Excel, CSV, GeoJSON, and text report generators
```

---

## ⚙️ Mathematical Index Formulation (Rule-Based Maintenance Priority)

The **Maintenance Priority Index ($MPI$)** is computed on a scale of $0.0$ to $100.0$ using a deterministic linear weighted sum of five normalized sub-indices:

$$MPI = w_{\text{traffic}} \cdot S_{\text{traffic}} + w_{\text{accident}} \cdot S_{\text{accident}} + w_{\text{damage}} \cdot S_{\text{damage}} + w_{\text{complaint}} \cdot S_{\text{complaint}} + w_{\text{drainage}} \cdot S_{\text{drainage}}$$

Where:
- $\sum w_i = 1.0$ (Default Weights: Traffic = 25%, Safety/Accidents = 25%, Pavement Damage = 25%, Complaints/Potholes = 15%, Drainage = 10%)
- Each sub-index $S_i$ is normalized between 0 and 100 via Min-Max scaling:
  $$S_i = \frac{x_i - \min(X)}{\max(X) - \min(X)} \times 100$$
- **Safety / Accident Burden Sub-Formula:**
  $$\text{Raw Accident Burden} = \text{Accident Count} + 3.0 \cdot \text{Fatalities} + 0.5 \cdot \text{Casualties}$$

### Priority Categorization
- **High Priority:** $MPI \ge 66.67$ (Immediate intervention required)
- **Medium Priority:** $33.33 \le MPI < 66.67$ (Scheduled maintenance required)
- **Low Priority:** $MPI < 33.33$ (Routine monitoring)

---

## 🧪 Testing & Validation Suite

The system includes an automated 11-point testing suite verifying:
1. **Catalog Completeness:** Validates non-empty road master records.
2. **Primary Key Uniqueness:** Confirms zero duplicate `Road_ID` keys.
3. **Foreign Key Integrity:** Asserts that 100% of traffic records map to existing master roads.
4. **Accident Referential Integrity:** Asserts zero orphaned accident records.
5. **Complaint Referential Integrity:** Asserts zero orphaned complaint records.
6. **WGS84 Coordinate Range:** Validates $-90 \le \text{Lat} \le 90$ and $-180 \le \text{Lon} \le 180$.
7. **Volume Bounds:** Validates non-negative vehicular counts ($\ge 0$).
8. **Damage Domain:** Asserts $0\% \le \text{Damage} \le 100\%$.
9. **Normalization Bounds:** Ensures all $MPI$ scores fall strictly within $[0.0, 100.0]$.
10. **Classification Domain:** Validates priority levels match `{'High', 'Medium', 'Low'}`.
11. **Algorithmic Determinism:** Verifies purely transparent arithmetic without black-box ML.

---

## 🚀 Installation & Execution Guide

### Prerequisites
- Python 3.10, 3.11, 3.12, or 3.13
- Git

### 1. Clone or Navigate to Project Directory
```bash
cd urban_road_system
```

### 2. Install Required Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```

The application will launch and open in your default browser at `http://localhost:8501`.

---

## 🛡️ Academic Non-Predictive Disclaimer

> **Methodological Statement:** This software is an academic engineering tool developed for urban road infrastructure evaluation. It does not use machine learning, neural networks, or artificial intelligence prediction models. All calculations, spatial patterns, and priority rankings reflect observed empirical data. Statistical correlation between variables indicates co-variation and must not be interpreted as causation.
