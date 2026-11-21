# Crude Oil Market & Inventory Forecasting Pipeline

An end-to-end weekly crude oil inventory forecasting pipeline, model benchmark, and prediction API built by **Tech Fem (Group 11)** for the Samsung Innovation Campus (SIC) AI Capstone Project.

## Project Structure
- `src/` — Data acquisition script (`step1_data_acquisition.py`) pulling EIA and FRED macroeconomic series.
- `data/` — Raw and processed weekly datasets (`.csv` and `.parquet`).
- `notebooks/` — End-to-end workflow covering preprocessing, feature selection, model training, walk-forward validation, and evaluation.
- `models/` — Trained model artifacts (`SARIMAX`, `LightGBM`, `LSTM`) and feature scalers.
- `results/` — Evaluation metrics, feature importances, and model predictions.
- `api/` — Backend prediction service (`api.py`, `forecast_service.py`) and precomputed JSON payloads.

## Models Benchmarked
- Naive & Seasonal Naive Baselines
- SARIMAX (Statistical)
- LightGBM (Gradient Boosting with Walk-Forward Validation)
- LSTM (PyTorch Neural Network)

## Quick Start
1. Install dependencies:
   ```bash
   pip install -r requirements.txt