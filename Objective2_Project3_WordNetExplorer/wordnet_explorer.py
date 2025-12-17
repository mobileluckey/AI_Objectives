"""
Objective 2 - Project 3: WordNet Explorer
Create and evolve natural language processing systems.

This tool uses NLTK's WordNet to explore words:
- Definitions (glosses)
- Example sentences
- Synonyms
- Antonyms
- Related word forms (lemmas)

It also auto-downloads required NLTK data the first time you run it.
"""

import re
import nltk
from nltk.corpus import wordnet as wn


def ensure_wordnet_downloaded():
    """
    Make sure WordNet and the Open Multilingual WordNet data are available.
    If not, download them.
    """
    try:
        _ = wn.synsets("test")
    except LookupError:
        print("NLTK WordNet data not found. Downloading now...")
        nltk.download("wordnet")
        nltk.download("omw-1.4")


def normalize_word(text: str) -> str:
    """
    Normalize input to a clean word form:
    - strip whitespace
    - lowercase
    - keep only letters/apostrophes
    """
    text = text.strip().lower()
    # keep letters and apostrophes
    text = re.sub(r"[^a-z']", "", text)
    return text


def get_synonyms_antonyms(word: str):
    """
    Pull synonyms and antonyms using WordNet lemmas.
    """
    synonyms = set()
    antonyms = set()

    for syn in wn.synsets(word):
        for lemma in syn.lemmas():
            synonyms.add(lemma.name().replace("_", " "))
            for ant in lemma.antonyms():
                antonyms.add(ant.name().replace("_", " "))

    return sorted(synonyms), sorted(antonyms)


def get_definitions_examples(word: str, max_senses=5):
    """
    Get WordNet senses with definitions and examples.
    """
    senses = wn.synsets(word)
    results = []

    for syn in senses[:max_senses]:
        definition = syn.definition()
        examples = syn.examples()  # list
        pos = syn.pos()  # n, v, a, r
        results.append({
            "pos": pos,
            "name": syn.name(),
            "definition": definition,
            "examples": examples
        })

    return results


def pretty_pos(pos_code: str) -> str:
    """
    Convert WordNet POS codes to readable text.
    """
    mapping = {
        "n": "Noun",
        "v": "Verb",
        "a": "Adjective",
        "s": "Adjective Satellite",
        "r": "Adverb"
    }
    return mapping.get(pos_code, pos_code)


def print_results(word: str):
    """
    Print WordNet exploration results for the given word.
    """
    print("=" * 60)
    print(f"WordNet Explorer: '{word}'")
    print("=" * 60)

    senses = get_definitions_examples(word, max_senses=6)
    synonyms, antonyms = get_synonyms_antonyms(word)

    if not senses and not synonyms and not antonyms:
        print("No results found. Try a different word.")
        return

    # Definitions + examples
    if senses:
        print("\nDEFINITIONS / SENSES")
        print("-" * 60)
        for i, s in enumerate(senses, start=1):
            print(f"{i}. {pretty_pos(s['pos'])}  ({s['name']})")
            print(f"   Definition: {s['definition']}")
            if s["examples"]:
                # Only print up to 2 examples to keep output readable
                for ex in s["examples"][:2]:
                    print(f"   Example:    \"{ex}\"")
            print()

    # Synonyms
    print("SYNONYMS")
    print("-" * 60)
    if synonyms:
        print(", ".join(synonyms[:40]))
        if len(synonyms) > 40:
            print(f"... and {len(synonyms) - 40} more")
    else:
        print("(No synonyms found)")
    print()

    # Antonyms
    print("ANTONYMS")
    print("-" * 60)
    if antonyms:
        print(", ".join(antonyms[:40]))
        if len(antonyms) > 40:
            print(f"... and {len(antonyms) - 40} more")
    else:
        print("(No antonyms found)")
    print()

    print("=" * 60)
    print("Tip: This is a simple NLP lexical knowledge tool.")
    print("You can evolve it by adding: word similarity, phrase support,")
    print("language translation, or a mini flashcard mode for TutiTech.")
    print("=" * 60)


def main():
    ensure_wordnet_downloaded()

    print("=" * 60)
    print("Objective 2 - Project 3: WordNet Explorer")
    print("Create and evolve natural language processing systems.")
    print("=" * 60)
    print("Type a word to explore its meaning, synonyms, and antonyms.")
    print("Type 'quit' or 'exit' to leave.\n")

    while True:
        user_input = input("Enter a word: ")
        if user_input.strip().lower() in ("quit", "exit"):
            print("Goodbye! Thanks for exploring WordNet.")
            break

        word = normalize_word(user_input)
        if not word:
            print("Please enter a real word (letters only).")
            continue

        print_results(word)


if __name__ == "__main__":
    main()
