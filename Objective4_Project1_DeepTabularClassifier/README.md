# Objective 4 – Project 1: Deep Tabular Classifier

## Why this meets Objective 4
This project designs and employs a deep learning system using best practices and conventions. It generates a realistic tabular dataset, applies standardized preprocessing (scaling), trains a multi-layer neural network with regularization (dropout), evaluates performance with industry metrics (accuracy, F1, ROC-AUC), and saves the trained model so it can be reused for inference on new data.

This project meets Objective 4 by demonstrating the design and implementation of a deep learning system using industry best practices. The system generates and processes realistic tabular data, applies proper feature scaling, and trains a multi-layer neural network with regularization to prevent overfitting. Model performance is evaluated using standard metrics such as accuracy, F1 score, and ROC-AUC, and the trained model is saved and reused to make predictions on new data. By completing both training and inference runs and validating the model with different input scenarios, this project shows how deep learning systems are built, evaluated, and deployed in real-world applications.

## Run
```bash
pip install -r requirements.txt
cd src
python make_data.py
python train.py
python predict.py
