# Objective 2 – Project 1: Chat Classifier

**Objective 2:** Create and evolve natural language processing systems.

## Project Overview

This project is a simple **chat classifier** that looks at a user’s message and tries to understand two things:

- **Sentiment** – Is the message positive, negative, or neutral?
- **Intent** – Is the user asking a question, giving a command, or making a statement?

The program is written in Python and uses **scikit-learn** with a TF-IDF vectorizer and a linear model. I keep a small, readable training set of example sentences inside the code, which makes it easy for me to grow the system over time by adding more examples.

## How It Meets Objective 2

This project fits **Objective 2: Create and evolve natural language processing systems** because it shows the full path from raw text to meaningful categories using NLP:

- It **processes natural language** input (whatever the user types).
- It applies **machine learning models** to classify sentiment and intent.
- The training data is intentionally small and visible in the code so I can **evolve the system** by adding more examples, adjusting labels, and retraining as I learn more.
- Each time I expand the training set and retrain, the classifier becomes a little smarter and more accurate, which demonstrates how NLP systems grow over time with better data.

This is the type of tool that could eventually support my TutiTech language tutor by helping it understand how the learner is feeling or what kind of message they are sending.

## How to Run It

1. Make sure you have Python 3 installed.

2. Install the required Python package (preferably in a virtual environment):

   ```bash
   pip install -r requirements.txt
