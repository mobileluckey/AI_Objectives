# Objective 2 – Project 2: NLP Quiz Generator

**Objective 2:** Create and evolve natural language processing systems.

## Project Overview

This project is a simple **NLP-based quiz generator**. I give it a passage of text, and it automatically builds a small quiz from that passage. The program:

1. **Analyzes the text** to find important words (keywords) by counting how often words appear and ignoring common stopwords.
2. Uses those keywords to create **fill-in-the-blank questions** by hiding the words inside sentences from the passage.
3. Prints a short list of **generic comprehension questions** that can be used with any reading.

The output is something I can copy into a worksheet, a lesson plan, or eventually into my TutiTech language tutor content.

## How It Meets Objective 2

This project supports **Objective 2: Create and evolve natural language processing systems** because it shows how I can:

- Take **raw natural language input** from the user.
- Use simple NLP techniques (tokenizing, stopword filtering, word frequency) to **extract structure and meaning**.
- Automatically generate **educational content** (vocabulary lists and questions) from that processed text.
- Improve and **evolve the system over time** by:
  - tuning the stopword list,
  - changing how keywords are selected,
  - and adding better question templates.

Even though the techniques are simple, this is exactly how many real-world NLP tools start: with basic text processing that can be refined and expanded as the system grows.

## How to Run It

1. Make sure you have Python 3 installed.

2. (Optional) Verify there are no external dependencies:

   ```bash
   type requirements.txt
