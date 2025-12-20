# Objective 6 – Project 1: Genetic Algorithm Parameter Tuner (Reboot)

## Why this meets Objective 6
This project integrates a genetic algorithm to optimize ML system parameters (regularization strength and classification threshold). It demonstrates evolutionary search, fitness evaluation, selection, crossover, and mutation, which is a core intelligent-software technique used when brute-force tuning is expensive.

Paragraph for README + website (based on YOUR run)

This project meets Objective 6 by using a genetic algorithm to tune machine learning parameters through evolutionary search instead of manual trial and error. I represented each candidate solution as an individual containing two genes, Logistic Regression regularization strength (C) and a probability threshold, and evaluated fitness using F1 score to account for class imbalance. Across generations, the algorithm applied selection, crossover, and mutation to improve fitness, and the best F1 score increased from 0.8432 to 0.8463 before stabilizing, showing convergence on a strong solution. The final best individual (C=2.7047, threshold=0.508) demonstrates how genetic algorithms can efficiently search parameter spaces and improve intelligent software behavior without brute-force tuning.

## Run
```bash
pip install -r requirements.txt
cd src
python ga_tuner.py
