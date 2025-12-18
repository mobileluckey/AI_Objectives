# Objective 3 – Project 2: Language Mastery Progress Model

This project predicts whether a learner will get a vocabulary word correct next time based on learning signals like attempts, correct rate, time since last seen, and streak.

## Why This Meets Objective 3
This is an ML system built using an industry pattern: a single saved pipeline that includes preprocessing (encoding and scaling) plus the trained model. It trains, evaluates using multiple metrics (accuracy, F1, ROC AUC), and saves artifacts for repeatable inference.

This project was trained and tested using multiple generated datasets to verify that the learning model behaves consistently across different inputs. By regenerating the dataset with a different random seed, the training process produced slightly different accuracy, F1, and ROC-AUC scores, along with different prediction probabilities. These variations confirm that the model is responding to changes in learner behavior data rather than relying on fixed values. Running the system multiple times demonstrates that the pipeline correctly preprocesses data, retrains the model, and produces valid predictions, which aligns with real-world machine learning development practices.

## Run
```bash
pip install -r requirements.txt
cd src
python make_data.py
python train.py
python predict.py
