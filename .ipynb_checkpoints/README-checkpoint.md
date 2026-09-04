# ⚽ Predicting EPL Player Market Values: Global vs. Position-Specific ML Models

An end-to-end machine learning pipeline and interactive Streamlit dashboard for predicting English Premier League (EPL) player market values across the 2021/22 to 2024/25 seasons. This project evaluates whether position-specific segmentation improves model accuracy over a global model architecture.

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://epl-player-valuation-prediction.streamlit.app/)

---

## 📌 Executive Summary

* **Best Overall Model:** Global LightGBM ($R^2 = 0.700$, $\text{MAE} = 0.532$, $\text{RMSE} = 0.742$).
* **Core Takeaway:** Position-specific segmentation revealed that optimal algorithms vary by role (e.g., Random Forest for Attackers, XGBoost for Defenders), but overall positional splitting provided no consistent accuracy gain over a well-tuned global LightGBM model.
* **Primary Value Drivers:** Minutes played, age, team points per match, and team goals scored carry the highest global predictive weight.

---

## 📊 Interactive Dashboard Overview

The repository includes a production-ready **Streamlit** web application for exploring model performance metrics, feature importances, diagnostic residual plots, and real-time player valuations.

### Key Features
1. **Executive Overview:** High-level metrics summarising overall thesis results.
2. **Model Performance & Comparison:** Interactive Plotly comparison tables and charts benchmarking Linear Regression, Random Forest, XGBoost, and LightGBM across Global and Positional models.
3. **Interpretability & Diagnostics:** Feature importance rankings alongside Actual vs. Predicted scatter plots and residual distributions.
4. **Live Value Predictor:** Real-time single-player valuation estimator based on custom player stats and squad contextual metrics.

> 💡 **Live Demo:** [Access the Streamlit Web Application](https://epl-player-valuation-prediction.streamlit.app/)

---

## 🛠 Tech Stack & Infrastructure

* **Language:** Python
* **Data Processing & Engineering:** Pandas, NumPy, `soccerdata`
* **Machine Learning:** scikit-learn, XGBoost, LightGBM, `joblib`
* **Interactive UI & Visualisation:** Streamlit, Plotly Express
* **Dataset Scope:** 2,116 player-season observations (FBref performance stats + Transfermarkt market valuations across 2021/22–2024/25 seasons).

---

## 📈 Model Performance & Evaluation

| Model Scope | Best Algorithm | $R^2$ | MAE (Log) | RMSE (Log) |
| :--- | :--- | :---: | :---: | :---: |
| **Global Baseline** | **LightGBM** | **0.700** | **0.532** | **0.742** |
| Attackers | Random Forest | 0.721 | 0.512 | 0.655 |
| Midfielders | LightGBM | 0.689 | 0.541 | 0.690 |
| Defenders | XGBoost | 0.705 | 0.528 | 0.672 |
| Goalkeepers | Linear Regression | 0.735 | 0.495 | 0.630 |

*Note: Evaluation metrics are based on log-transformed market values ($\log(\text{Market Value})$).*

---

## 🚀 Quickstart Guide

### 1. Clone the Repository
```bash
git clone [https://github.com/YOUR_USERNAME/epl-player-valuation-ml.git](https://github.com/YOUR_USERNAME/epl-player-valuation-ml.git)
cd epl-player-valuation-ml