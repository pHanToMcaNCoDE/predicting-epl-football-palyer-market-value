# Predicting EPL Football Player Market Value Using Position-Specific ML Models

An end-to-end data science pipeline for predicting English Premier League player market values (2021/22 to 2024/25 seasons), comparing global machine learning models against position-specific models to evaluate whether positional segmentation improves predictive accuracy.

## The Core Infrastructure

- Data Sources: Real-world player performance statistics (FBref) merged with market valuations and demographic data (Transfermarkt); 2,116 player season observations after cleaning.
- Architecture: Modular Python preprocessing pipeline covering missing data analysis, outlier detection (IQR), log transformation of the target variable, and position-based dataset splitting (attackers, midfielders, defenders, goalkeepers).
- Machine Learning Models: Benchmarked Linear Regression against tuned Random Forest, XGBoost, and LightGBM (GridSearchCV, 5-fold CV), trained both globally and per position.

## Tech Stack

- Language: Python
- Data & ML Engineering: Pandas, NumPy, scikit-learn, XGBoost, LightGBM, soccerdata
- Development Workflow: Git, modular reusable model training functions

## Model Evaluation & Insights
The global LightGBM model achieved the strongest overall predictive performance (MAE 0.532, RMSE 0.742, R² 0.700). Position-specific results showed the best model varied by position: Random Forest for attackers, LightGBM for midfielders, XGBoost for defenders, and Linear Regression for goalkeepers. Positional segmentation did not consistently outperform the global model, suggesting playing position influences model behaviour without providing a decisive accuracy gain.
