# text_mining_ai.py
# Simple "AI-style" text mining tool
# This program reads a text file and analyzes my writing.
# It finds how many sentences and words I use, the most common words,
# and which words I might be overusing.

# text_mining_ai.py
# Text-Mining AI Tool with text + CSV export
# This version generates:
#   - text_report.txt  (human-readable report)
#   - text_report.csv  (structured analysis)

import sys
from collections import Counter
import string
from pathlib import Path
import re
import csv


def load_text(path):
    return Path(path).read_text(encoding="utf-8", errors="ignore")


def split_sentences(text):
    raw_sentences = re.split(r"[.!?]+", text)
    return [s.strip() for s in raw_sentences if s.strip()]


def tokenize(text):
    translator = str.maketrans("", "", string.punctuation)
    clean = text.lower().translate(translator)
    return [w for w in clean.split() if w]


def write_report_text(contents):
    Path("text_report.txt").write_text(contents, encoding="utf-8")
    print("\nSaved text report: text_report.txt")


def write_report_csv(data_dict, top_words, overused_list):
    """
    Saves structured data to text_report.csv
    CSV will include:
    - Basic stats
    - Top 20 words
    - Overused words
    """
    with open("text_report.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Category", "Metric", "Value"])

        # Basic statistics
        for key, value in data_dict.items():
            writer.writerow(["Basic Stats", key, value])

        # Top 20 most common words
        writer.writerow([])
        writer.writerow(["Category", "Word", "Count"])
        for word, count in top_words:
            writer.writerow(["Top 20 Words", word, count])

        # Overused words
        writer.writerow([])
        writer.writerow(["Category", "Word", "Count"])
        for word, count in overused_list:
            writer.writerow(["Overused Words", word, count])

    print("Saved CSV report: text_report.csv")


def main():
    if len(sys.argv) < 2:
        print("Usage: python text_mining_ai.py your_file.txt")
        sys.exit(1)

    filepath = sys.argv[1]
    text = load_text(filepath)

    sentences = split_sentences(text)
    words = tokenize(text)
    sentence_lengths = [len(tokenize(s)) for s in sentences]

    # === Basic Stats ===
    basic_stats = {
        "Sentence Count": len(sentences),
        "Word Count": len(words),
        "Average Sentence Length (words)": round(sum(sentence_lengths) / len(sentence_lengths), 2)
        if sentences else 0,
        "Shortest Sentence (words)": min(sentence_lengths) if sentences else 0,
        "Longest Sentence (words)": max(sentence_lengths) if sentences else 0,
        "Average Word Length (characters)": round(sum(len(w) for w in words) / len(words), 2)
        if words else 0,
    }

    # Word frequency
    counts = Counter(words)
    top20 = counts.most_common(20)

    # Overused words
    stopwords = {
        "the", "and", "a", "to", "of", "in", "it", "is", "that", "for",
        "on", "with", "as", "this", "i", "you", "my", "at", "be", "are"
    }
    overused = [(w, c) for w, c in counts.items() if c >= 5 and w not in stopwords]
    overused.sort(key=lambda x: -x[1])

    # === Generate Text Report ===
    report = []
    report.append(f"AI Text Analysis Report for: {filepath}\n")
    report.append("=" * 60 + "\n\n")

    report.append("--- Basic Counts ---\n")
    for key, value in basic_stats.items():
        report.append(f"{key}: {value}\n")

    report.append("\n--- Top 20 Most Common Words ---\n")
    for word, count in top20:
        report.append(f"{word:15} {count}\n")

    report.append("\n--- Overused Words ---\n")
    if overused:
        for word, count in overused:
            report.append(f"{word:15} {count}\n")
    else:
        report.append("None detected\n")

    # Save reports
    write_report_text("".join(report))
    write_report_csv(basic_stats, top20, overused)


if __name__ == "__main__":
    main()
