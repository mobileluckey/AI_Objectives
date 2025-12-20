# Objective 4 – Project 3: Deep Emotion Detector

## Why this meets Objective 4
This project uses a deep neural network with embeddings to classify emotions from text. It follows best practices: train/validation split, regularization, clear class mapping, and saved model artifacts for repeatable inference.

This project meets Objective 4 by demonstrating the design and implementation of a deep learning system that performs multi-class natural language classification. The system processes raw text input, converts it into numerical representations using text vectorization and word embeddings, and trains a multi-layer neural network to identify emotional intent such as happy, sad, angry, or neutral. The model is trained, evaluated, and saved for reuse, then loaded to perform inference on new text samples. By running multiple training epochs and validating predictions on unseen sentences, this project shows how deep learning systems are built, tested, and applied using industry-standard workflows.

## Run
```bash
pip install -r requirements.txt
cd src
python train.py
python predict.py
