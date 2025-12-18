# Objective 3 – Project 1: Smart Temperature Predictor

**Objective 3:** Demonstrate proficiency in designing and developing machine learning systems using industry approaches and patterns.

## What This Does
This project trains a machine learning model to predict temperature based on recent historical readings. It builds lag and rolling-window features, uses a time-based train/test split, evaluates performance, and saves the trained model and metrics.

## Why This Meets Objective 3
This is a full ML system workflow: data ingestion, feature engineering, model training, evaluation with real metrics (MAE/RMSE), and model persistence for repeatable predictions. The design follows an industry pattern where models are trained offline and then loaded for inference.

To demonstrate that this machine learning system works reliably and is not hard-coded to a single result, the model was trained and evaluated using multiple generated datasets. By changing the random seed used during data generation, the training process produced slightly different temperature patterns while still following realistic trends. Each training run resulted in different evaluation metrics and predictions, confirming that the system learns from the data provided and generalizes correctly. This approach reflects industry practices where models are validated across multiple datasets to ensure consistent and dependable performance.

## How to Run
```bash
pip install -r requirements.txt
cd src
python make_data.py
python train.py
python predict.py
