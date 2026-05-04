# Predicting-Appliance-Energy-Use
Predicting residential appliance energy consumption (Wh) using the UCI Appliances Energy Prediction dataset (19,735 observations, 10-min intervals from a Belgian home, Jan–May 2016).

## Problem
Forecast short-term household appliance energy use to support smart-home platforms, grid optimization, and demand-response strategies.

## Data
- **Source:** UCI Appliances Energy Prediction dataset
- **Features:** 9 indoor temperature/humidity sensor pairs, outdoor weather (temp, humidity, pressure, wind, visibility), and timestamp
- **Target:** Appliance energy consumption (Wh)

## Approach
**Feature engineering:** 10/20/30/60-min lag features, 1-hour rolling mean, cyclical hour/day encodings, momentum, weekend flag, and `log1p` target transform.
**Validation:** TimeSeriesSplit (5 folds) and chronological 80/20 train/test split to preserve temporal order.
**Models trained:** Linear Regression, Ridge, Lasso, SVR (RBF), Random Forest (500 trees), XGBoost.
**Clustering:** KMeans (k=3) with PCA (25→11 components) to segment household energy regimes.

## Results

| Model           | R²     | MAE (Wh) |
|-----------------|--------|----------|
| XGBoost         | 0.5827 | 24.20    |
| Random Forest   | 0.5780 | 24.13    |
| SVR             | 0.4386 | 34.61    |
| Linear models   | ~0.31  | ~28.7    |

## Requirements
Python 3.x, pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn.

## Usage
Open `Group 5 - Predicting Energy Use Notebook.ipynb` in Jupyter and run cells sequentially.