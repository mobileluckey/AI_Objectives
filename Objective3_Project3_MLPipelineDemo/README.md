# Objective 3 – Project 3: ML Pipeline Demo (End-to-End)

This project demonstrates a complete machine learning workflow using an industry approach: a single pipeline that handles preprocessing and prediction consistently.

## Why This Meets Objective 3
It shows an end-to-end ML system pattern: load data, split, preprocess (encoding + scaling), train, evaluate, and save the pipeline artifact for repeatable inference. This is the same basic pattern used in real ML deployments to avoid training-serving skew.

To demonstrate that this end-to-end machine learning pipeline functions correctly, the system was tested using multiple prediction scenarios rather than regenerating the training data. After training the pipeline once, different input records were provided during inference, resulting in different classification outcomes and probability scores. This confirms that the saved pipeline correctly applies preprocessing, feature encoding, and model inference each time it is used. This approach reflects real-world machine learning systems, where models are trained offline and then deployed to make reliable predictions on new data without retraining.

## Run
```bash
pip install -r requirements.txt
cd src
python train.py
python predict.py
