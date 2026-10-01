"""
Keyword Search Baseline Module

Purpose:
    This module implements the keyword-search baseline used to compare
    against the RAG semantic retrieval system.

Main responsibilities:
    - Load HR source documents
    - Load the fixed evaluation questions
    - Match question keywords against document text
    - Select the best matching document
    - Save retrieval results for evaluation

Input:
    - HR documents stored in the HR_Documents folder
    - Evaluation/questions.csv

Output:
    - Evaluation/keyword_results.csv

Why this module exists:
    This baseline provides a simple lexical-search comparison so the
    project can measure whether embedding-based semantic retrieval
    improves performance on paraphrased employee questions.
"""

import os
import re
import pandas as pd


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_FOLDER = os.path.join(BASE_DIR, "HR_Documents")
QUESTION_FILE = os.path.join(BASE_DIR, "Evaluation", "questions.csv")
OUTPUT_FILE = os.path.join(BASE_DIR, "Evaluation", "keyword_results.csv")


def load_documents():
    documents = []

    for filename in sorted(os.listdir(DOC_FOLDER)):
        if filename.endswith(".txt"):
            path = os.path.join(DOC_FOLDER, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append(
                {
                    "filename": filename,
                    "text": text
                }
            )

    return documents


def tokenize(text):
    return re.findall(r"\b[a-zA-Z]+\b", text.lower())


def keyword_score(question, document_text):
    question_words = set(tokenize(question))
    document_words = tokenize(document_text)

    score = 0

    for word in question_words:
        score += document_words.count(word)

    return score


def retrieve_document(question, documents):
    best_document = None
    best_score = -1

    for document in documents:
        score = keyword_score(
            question,
            document["text"]
        )

        if score > best_score:
            best_score = score
            best_document = document

    return best_document, best_score


def main():
    documents = load_documents()
    questions = pd.read_csv(QUESTION_FILE)

    results = []

    for _, row in questions.iterrows():
        document, score = retrieve_document(
            row["question"],
            documents
        )

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "keyword_document": document["filename"],
                "keyword_score": score
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUTPUT_FILE, index=False)

    print("Keyword search evaluation completed.")
    print(f"Total questions: {len(result_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("First 5 results:")
    print(result_df.head())


if __name__ == "__main__":
    main()
