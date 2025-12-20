# Objective 4 – Project 2: Deep Spam Detector

## Why this meets Objective 4
This is a deep learning NLP classifier built with TensorFlow using best practices: TextVectorization, embeddings, a neural architecture with regularization, evaluation, and saved model artifacts for repeatable inference.

This project meets Objective 4 by demonstrating the design and use of a deep learning system for natural language classification using industry-standard techniques. The system processes raw text messages, converts them into numerical representations using text vectorization and word embeddings, and trains a multi-layer neural network to classify messages as spam or legitimate. The model is trained, evaluated, and saved for reuse, then loaded to perform inference on new, unseen messages. By running multiple training and prediction passes and validating the model with different input examples, this project shows how deep learning systems are built, tested, and deployed to solve real-world text classification problems.

## Run
```bash
pip install -r requirements.txt
cd src
python train.py
python predict.py
