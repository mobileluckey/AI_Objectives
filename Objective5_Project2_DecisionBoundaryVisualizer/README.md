# Objective 5 – Project 2: Decision Boundary Visualizer

## What this project does
This program generates a non-linear dataset (two moons), trains two different models, and visualizes the decision boundary each model learns:
- Logistic Regression (mostly linear boundary)
- Neural Network (MLP) (non-linear boundary)

The plots are saved into the `models/` folder as PNG files so they can be uploaded to GitHub and embedded on a website.

## Why this meets Objective 5
This project demonstrates deep understanding of neural networks by showing how model architecture affects learned behavior. The decision boundary is the visible “rule” the model uses to classify data. Logistic regression struggles to form curved boundaries, while a neural network can learn non-linear regions because hidden layers and activation functions transform the input space. This makes the learning process visible and explains why neural networks are important building blocks for intelligent systems.

## Run
```bash
pip install -r requirements.txt
cd src
python decision_boundary_visualizer.py
