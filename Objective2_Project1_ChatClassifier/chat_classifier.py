"""
Objective 2 - Project 1: Chat Classifier
Create and evolve natural language processing systems.

This program is a small NLP demo that:
- Classifies a user message for SENTIMENT (positive, negative, neutral)
- Classifies a user message for INTENT (question, command, statement)

It uses scikit-learn with a TF-IDF vectorizer and a linear model.
The training data is small but easy to extend, which shows how the
system can evolve over time as I add more examples.
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC


def build_sentiment_classifier():
    """
    Build a simple sentiment classifier.

    Labels:
        - positive
        - negative
        - neutral
    """
    # Small starter training set. This can be expanded over time.
    training_texts = [
        # positive
        "I love this",
        "This is great",
        "I am very happy with the result",
        "Nice job, this is awesome",
        "That makes me excited",
        "This is exactly what I wanted",
        "Everything is working perfectly",
        "That was really helpful",
        "I appreciate this a lot",
        "This feels amazing",
        "This is so much fun!",# I entered more positive training data.

        # negative
        "I hate this",
        "This is terrible",
        "I am very unhappy",
        "This is not what I wanted",
        "I am frustrated and annoyed",
        "This is broken and useless",
        "Nothing is working",
        "I am disappointed",
        "This is a bad result",
        "That made things worse",

        # neutral
        "I will test this now",
        "The system is running",
        "I am checking the output",
        "The result is okay",
        "I am not sure what to think",
        "It is fine",
        "This is an example sentence",
        "We can look at this later",
        "I typed this message",
        "This is normal behavior",
    ]

    training_labels = [
        # first 10 -> positive
        "positive", "positive", "positive", "positive", "positive",
        "positive", "positive", "positive", "positive", "positive",
        "positive",# I had to also add a label for the extra positive data.
        # next 10 -> negative
        "negative", "negative", "negative", "negative", "negative",
        "negative", "negative", "negative", "negative", "negative",
        # last 10 -> neutral
        "neutral", "neutral", "neutral", "neutral", "neutral",
        "neutral", "neutral", "neutral", "neutral", "neutral",
    ]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    model.fit(training_texts, training_labels)
    return model


def build_intent_classifier():
    """
    Build a simple intent classifier.

    Labels:
        - question  (user is asking something)
        - command   (user is telling the system to do something)
        - statement (user is describing or expressing something)
    """
    training_texts = [
        # questions
        "What is the status of my project",
        "Can you help me with this",
        "How do I fix this error",
        "Why is this not working",
        "Where should I start",
        "When will this be finished",
        "Could you explain that again",
        "What does this message mean",
        "How can I improve this",
        "Is this the correct result",

        # commands
        "Run the test again",
        "Please restart the system",
        "Show me the logs",
        "Save this to a file",
        "Start a new session",
        "Stop the program",
        "Generate a report",
        "Open the next lesson",
        "Close the current window",
        "Reset everything",

        # statements
        "The program is running",
        "I think this looks good",
        "I am still learning this",
        "The output is different now",
        "I see a new error message",
        "The device is connected",
        "The lesson is complete",
        "I am getting better at this",
        "This result is interesting",
        "The code compiled successfully",
    ]

    training_labels = [
        # 10 questions
        "question", "question", "question", "question", "question",
        "question", "question", "question", "question", "question",
        # 10 commands
        "command", "command", "command", "command", "command",
        "command", "command", "command", "command", "command",
        # 10 statements
        "statement", "statement", "statement", "statement", "statement",
        "statement", "statement", "statement", "statement", "statement",
    ]

    model = Pipeline([
        ("tfidf", TfidfVectorizer()),
        ("clf", LinearSVC())
    ])

    model.fit(training_texts, training_labels)
    return model


def classify_message(text, sentiment_model, intent_model):
    """
    Run both classifiers on the given text and return their predictions.
    """
    sentiment = sentiment_model.predict([text])[0]
    intent = intent_model.predict([text])[0]
    return sentiment, intent


def main():
    print("=" * 60)
    print("Objective 2 - Project 1: Chat Classifier")
    print("Create and evolve natural language processing systems.")
    print("=" * 60)
    print("This tool classifies your message by:")
    print("  - Sentiment: positive / negative / neutral")
    print("  - Intent:    question / command / statement")
    print()
    print("Type a message and press Enter to classify it.")
    print("Type 'quit' or 'exit' to leave.")
    print()

    # Build models once at startup
    sentiment_model = build_sentiment_classifier()
    intent_model = build_intent_classifier()

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ("quit", "exit"):
            print("Goodbye! Thanks for testing the chat classifier.")
            break

        if not user_input:
            print("Please type something for me to classify.")
            continue

        sentiment, intent = classify_message(user_input, sentiment_model, intent_model)

        print(f"  -> Sentiment: {sentiment}")
        print(f"  -> Intent:    {intent}")
        print("-" * 60)
        print("Hint: To 'evolve' this system over time, you can add more")
        print("training examples in the build_sentiment_classifier() and")
        print("build_intent_classifier() functions and re-run the program.")
        print("-" * 60)


if __name__ == "__main__":
    main()

