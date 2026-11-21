# Crude Oil Market & Inventory Forecasting System

A machine learning and GenAI pipeline designed to predict weekly changes in U.S. commercial crude oil inventories (excluding SPR) one week ahead. The system features an automated data ingestion pipeline, multiple model evaluations (Baselines, SARIMAX, LightGBM, LSTM), and an interactive Streamlit dashboard with custom scenario simulations and automated Wednesday data updates.

---

## 📌 Project Overview
- **Team Name:** Tech Fem (Group 11)
- **Program:** Samsung Innovation Campus — AI Course
- **Target Variable:** Weekly change in U.S. commercial crude oil stocks (excluding SPR) measured in thousand barrels.
- **Data Range:** Weekly data from February 1991 to Present (~1,855 rows).

---

## ⚙️ Key Features
- **Automated Data Ingestion:** Real-time collection and integration of EIA Petroleum Status Report data and FRED Broad U.S. Dollar Index via REST APIs.
- **Leakage-Free Feature Engineering:** 45+ engineered features including time lags, rolling averages, seasonal indicators, and market momentum variables.
- **Robust Model Evaluation:** Time-series walk-forward validation comparing Naive/Seasonal Naive baselines against SARIMAX, LightGBM, and LSTM.
- **Selected Model:** **LightGBM** (optimized for minimum MAE/RMSE and superior directional accuracy over a 104-week holdout test period).
- **Interactive Streamlit Dashboard:** 
  - **Forecast & Volatility Engine:** Displays 1-week-ahead predictions with uncertainty bounds.
  - **Interactive What-If Scenarios:** User-controlled sliders for imports and refinery utilization adjustments.
  - **Automated Pipeline:** Automatic refresh triggering every Wednesday upon EIA publication.
  - **GenAI Explainer:** Structured insight generator utilizing approved model factors to explain inventory movements.

---

## 🏗️ System Architecture & Data Flow

```
[EIA API (WPSR)] + [FRED API (USD Index)]
                  │
                  ▼
   [Data Preprocessing & Feature Engineering]
                  │  (Parquet Storage Format)
                  ▼
   [Time-Series Walk-Forward Validation]
                  │
                  ▼
   [Model Inference Engine (LightGBM)]
                  │
                  ▼
       [Streamlit Web Dashboard]
       ├── Live Wednesday Auto-Refresh
       ├── Interactive Scenario Engine
       └── GenAI Factor Explainer
```

---

## 📊 Methodology & Workflow

1. **Data Acquisition & Alignment:** Pulls weekly EIA metrics (stocks, production, imports, exports, refinery input/utilization, Cushing stocks, WTI/Brent prices) and FRED U.S. Dollar Index. Left-joined on `week_ending` date.
2. **Preprocessing & Feature Engineering:** Implements strict cutoff logic to prevent data leakage. Formats output to optimized Parquet files.
3. **Exploratory Data Analysis (EDA):** Analyzes historical supply shocks (e.g., COVID-19 disruption, 2022 market shocks), seasonal volatility, and cross-feature correlations.
4. **Model Benchmark & Selection:**
   - *Baselines:* Naive, Seasonal Naive
   - *Statistical:* SARIMAX
   - *Gradient Boosting:* LightGBM (Primary Selected Model)
   - *Deep Learning:* LSTM
5. **Dashboard & Deployment:** Multi-page Streamlit application hosting risk matrices, interactive scenario models, and error distribution visualizers.

---

## 👥 Team Roles & Responsibilities

| Team Member | Role & Key Responsibilities |
| :--- | :--- |
| **Shora Wael** *(Leader)* | Baseline Models, Deep Learning Architecture (LSTM), SARIMAX & Selection |
| **Rania Fathi** | Data Ingestion Pipeline, API Integration, Quality & Preprocessing |
| **Shaima Abdulsalam**| Feature Engineering, Disruption EDA, Technical Documentation |
| **Gezlan Akram** | Machine Learning (LightGBM), Model Evaluation & Walk-Forward Metrics |
| **Rawan Eltheni** | Web Architecture, Streamlit Scenario Engine, Auto-Refresh Pipeline |
| **Hind Khmaj** | UI/UX Visualizations, Dynamic Charts, Presentation Strategy |

---

## 🛠️ Tech Stack & Requirements

- **Language:** Python 3.9+
- **Data Acquisition:** `requests`, EIA Open Data API v2, FRED API
- **Data Manipulation:** `pandas`, `numpy`, `pyarrow` (Parquet format)
- **Modeling & ML:** `scikit-learn`, `lightgbm`, `statsmodels`, `tensorflow` / `pytorch`
- **Visualization:** `plotly`, `matplotlib`, `seaborn`
- **Dashboard & Web:** `streamlit`

---

## 📜 Copyright & License

© 2025 SAMSUNG. All rights reserved.  
Developed under the **Samsung Innovation Campus** curriculum by Team Tech Fem. Redistribution or reproduction without prior written consent is strictly prohibited.