# 🌊 AI-Based Environmental Risk Monitoring & Water Quality Intelligence Platform

An end-to-end machine-learning decision-support prototype for river/water-quality monitoring. It extends the original River Health Prediction project with **water-quality scoring, pollution-risk prediction, historical analytics, explainability, geospatial visualization, automated risk alerts, and what-if simulation**.

> **Important:** The included fallback dataset is synthetic and the map coordinates are simulated. This is a portfolio/educational prototype, not a regulatory water-quality system.

## Features

- **Water-quality prediction:** Random Forest pipeline using pH, nitrate, temperature, turbidity, dissolved oxygen, conductivity and industry type.
- **Pollution classification:** Probability-based low/moderate/high/critical risk labels.
- **Water Quality Index (WQI):** Transparent composite score based on water parameters.
- **Historical trends:** Daily WQI and parameter analytics with Plotly.
- **Geographic visualization:** Interactive Folium monitoring map with risk-colored stations.
- **Risk alerts:** Automatically surfaces high-risk monitoring stations.
- **Model explainability:** Model-agnostic permutation feature importance; optional SHAP support can be added.
- **What-if simulator:** Change environmental conditions and observe the predicted risk delta.
- **Model benchmarking:** `train_models.py` compares Random Forest and an ANN-style MLP baseline.

## Run locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

The application can run even without a CSV because it generates a reproducible synthetic dataset automatically. To use the original dataset, place it at:

`data/synthetic_river_health_data.csv`

## Train/evaluate models

```bash
python train_models.py
```

This saves the deployable Random Forest pipeline and evaluation metrics under `models/`.

## Project structure

```text
River-Health-Intelligence/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── data/
│   └── synthetic_river_health_data.csv   # optional
├── models/
│   └── .gitkeep
├── src/
│   ├── data_pipeline.py
│   ├── model.py
│   ├── risk_engine.py
│   └── explainability.py
└── notebooks/
```

## Resume-ready description

**AI-Based Environmental Risk Monitoring & Water Quality Intelligence Platform** — Developed an ML-driven environmental monitoring platform that classifies pollution risk from multi-parameter water-quality data; built an interactive Streamlit dashboard with historical analytics, geospatial risk visualization, automated alerts, model explainability, WQI scoring and what-if environmental simulations.

## Future upgrades

1. Replace synthetic observations with a verified public water-quality dataset.
2. Connect real IoT sensor streams through MQTT/API ingestion.
3. Add SHAP beeswarm/waterfall explanations.
4. Add temporal LSTM/Transformer forecasting on station histories.
5. Add PostgreSQL/PostGIS for production-scale geospatial storage.
6. Deploy with authentication and scheduled alert delivery.
