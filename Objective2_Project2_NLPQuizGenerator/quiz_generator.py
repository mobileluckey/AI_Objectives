"""
Objective 2 - Project 2: NLP Quiz Generator

This program takes a passage of text and automatically builds
a small quiz based on it. It does three main things:

1. Extracts the most important-looking words (keywords) by
   counting how often they appear and filtering out common
   stopwords.
2. Creates fill-in-the-blank questions by hiding those
   keywords inside sentences from the passage.
3. Prints some generic comprehension questions that a
   teacher or student can use to think about the passage.

This shows an evolving natural language processing system:
as I improve the keyword logic, stopwords, or question
templates, the quality of the quiz gets better over time.
"""

import re
from collections import Counter
from textwrap import fill


# A small built-in stopword list to filter common words.
STOPWORDS = {
    "the", "and", "a", "an", "of", "to", "in", "it", "is", "for", "on",
    "that", "this", "with", "as", "at", "by", "from", "or", "if", "be",
    "are", "was", "were", "can", "could", "would", "should", "has", "have",
    "had", "will", "not", "no", "yes", "you", "your", "i", "we", "they",
    "he", "she", "them", "our", "us", "their", "my", "me", "so", "but",
    "about", "into", "out", "up", "down", "over", "under", "then", "than",
    "when", "where", "how", "what", "why", "who"
}


def read_passage_from_user():
    """
    Ask the user to paste or type a passage of text.

    The user can enter multiple lines. When they are finished,
    they type a single line containing only 'END' and press Enter.
    """
    print("=" * 60)
    print("NLP Quiz Generator")
    print("=" * 60)
    print("Paste or type your passage below.")
    print("When you are done, type a single line with: END")
    print()

    lines = []
    while True:
        line = input()
        if line.strip().upper() == "END":
            break
        lines.append(line)

    passage = "\n".join(lines).strip()
    return passage


def tokenize_words(passage):
    """
    Extract words from the passage using a simple regular expression.

    Returns a list of lowercase tokens like: ['example', 'words', 'here'].
    """
    tokens = re.findall(r"[A-Za-z']+", passage)
    tokens = [t.lower() for t in tokens]
    return tokens


def split_sentences(passage):
    """
    Split the passage into sentences using a simple regex.

    This is not perfect, but good enough for this small project.
    """
    # Split on ., ?, or ! followed by whitespace
    raw_sentences = re.split(r"[.!?]\s+", passage)
    sentences = [s.strip() for s in raw_sentences if s.strip()]
    return sentences


def extract_keywords(tokens, max_keywords=8, min_length=4):
    """
    Pick the most frequent non-stopword tokens as keywords.

    - max_keywords: maximum number of keywords to return
    - min_length: minimum length of each keyword
    """
    filtered = [
        t for t in tokens
        if t not in STOPWORDS and len(t) >= min_length
    ]

    counts = Counter(filtered)
    # Most common returns tuples like (word, count)
    most_common = counts.most_common(max_keywords)
    keywords = [word for word, count in most_common]
    return keywords, counts


def create_fill_in_blank_questions(sentences, keywords, max_questions=5):
    """
    Build fill-in-the-blank questions by hiding keywords inside sentences.

    For each keyword, we find the first sentence that contains it and
    replace the word with '_____' (case-insensitive match).
    """
    questions = []
    used_sentences = set()

    for keyword in keywords:
        keyword_lower = keyword.lower()
        found_sentence = None

        for idx, sentence in enumerate(sentences):
            # Skip sentences we've already used
            if idx in used_sentences:
                continue

            # Case-insensitive check
            if re.search(rf"\b{re.escape(keyword_lower)}\b", sentence, flags=re.IGNORECASE):
                found_sentence = (idx, sentence)
                break

        if found_sentence is None:
            continue

        idx, sentence = found_sentence
        used_sentences.add(idx)

        # Replace the first occurrence of the keyword with a blank
        blank_sentence = re.sub(
            rf"\b{re.escape(keyword_lower)}\b",
            "_____",
            sentence,
            count=1,
            flags=re.IGNORECASE
        )

        questions.append({
            "keyword": keyword,
            "question": blank_sentence
        })

        if len(questions) >= max_questions:
            break

    return questions


def print_keywords_section(keywords, counts):
    """
    Print a simple vocabulary list with word frequencies.
    """
    print("=" * 60)
    print("SECTION 1: KEYWORDS / VOCABULARY")
    print("=" * 60)
    if not keywords:
        print("No strong keywords found. Try a longer or more detailed passage.")
        return

    for word in keywords:
        print(f"- {word} (frequency: {counts[word]})")
    print()
    print("Tip: You can ask the student to define these words, use them in")
    print("a sentence, or match them with images or translations.")
    print()


def print_fill_in_blank_section(questions):
    """
    Print the fill-in-the-blank questions and an answer key.
    """
    print("=" * 60)
    print("SECTION 2: FILL-IN-THE-BLANK QUESTIONS")
    print("=" * 60)

    if not questions:
        print("No fill-in-the-blank questions could be generated.")
        print("Try including more descriptive words in your passage.")
        print()
        return

    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {fill(q['question'], width=70)}")
        print()

    print("-" * 60)
    print("Answer Key:")
    for i, q in enumerate(questions, start=1):
        print(f"Q{i}: {q['keyword']}")
    print()


def print_comprehension_section():
    """
    Print some generic comprehension questions that work for any passage.
    """
    print("=" * 60)
    print("SECTION 3: COMPREHENSION QUESTIONS")
    print("=" * 60)
    questions = [
        "1. In your own words, what is the main idea of this passage?",
        "2. List two important details that support the main idea.",
        "3. Why do you think this information is important or useful?",
        "4. Is there anything in the passage that you found confusing? What is it?",
        "5. How could you apply what you learned from this passage in real life?"
    ]
    for q in questions:
        print(q)
    print()


def main():
    passage = read_passage_from_user()

    if not passage:
        print("No passage was entered. Exiting.")
        return

    tokens = tokenize_words(passage)
    sentences = split_sentences(passage)

    keywords, counts = extract_keywords(tokens, max_keywords=8, min_length=4)
    fill_in_questions = create_fill_in_blank_questions(sentences, keywords, max_questions=5)

    print_keywords_section(keywords, counts)
    print_fill_in_blank_section(fill_in_questions)
    print_comprehension_section()

    print("=" * 60)
    print("Quiz generation complete.")
    print("You can copy these questions into a worksheet, lesson plan,")
    print("or into your TutiTech language tutor content.")
    print("=" * 60)


if __name__ == "__main__":
    main()
