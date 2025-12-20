# Objective 5 – Project 3: Explainable AI Demo

## Why this meets Objective 5
This project demonstrates a deep understanding of how models make decisions by quantifying which features drive predictions. It connects neural-network-era thinking to responsible intelligent systems by focusing on interpretability and validation, not just accuracy.

## Run
```bash
pip install -r requirements.txt
cd src
python explainable_ai.py


# Objective 5 – Project 3: Explainable AI Demo

## What this project does
This project trains a machine learning model (RandomForestClassifier) and then explains its decisions using **Permutation Feature Importance**. The goal is to show which input features most strongly affect model performance and predictions.

## Why this meets Objective 5
This project demonstrates a deep understanding of intelligent systems by focusing on *how and why* a model makes decisions, not just whether it is accurate. Explainable AI is important because many ML models behave like black boxes. By measuring feature importance, I can show which inputs drive predictions, improve trust in the system, and make the model easier to debug and validate. This demonstrates understanding of model behavior, interpretability, and responsible use of intelligent systems.

This project meets Objective 5 by showing that I understand not only how intelligent systems make predictions, but how to explain why they make those decisions. I trained a RandomForest classification model and achieved a test accuracy of 0.9580, then used permutation feature importance to measure which input features most influenced the model’s performance. Instead of treating the model like a black box, this project identifies the top features driving predictions and saves the results as a chart and data file for validation. This demonstrates a deeper understanding of how models “think,” how to verify their behavior, and why explainability is important for trust, debugging, and responsible use of intelligent systems.

## Run
```bash
pip install -r requirements.txt
cd src
python explainable_ai.py
