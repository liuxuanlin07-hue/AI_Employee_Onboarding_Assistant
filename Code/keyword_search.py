import os
import re
import pandas as pd

# Paths
DOC_FOLDER = "../HR_Documents"
QUESTION_FILE = "../Evaluation/questions.csv"
OUTPUT_FILE = "../Evaluation/keyword_results.csv"


def load_documents():
    documents = []

    for filename in os.listdir(DOC_FOLDER):
        if filename.endswith(".txt"):
            path = os.path.join(DOC_FOLDER, filename)

            with open(path, "r", encoding="utf-8") as file:
                text = file.read()

            documents.append({
                "filename": filename,
                "text": text
            })

    return documents


def keyword_search(query, documents):
    query_words = re.findall(r"\b\w+\b", query.lower())

    best_document = None
    best_score = 0

    for document in documents:
        text = document["text"].lower()

        score = 0

        for word in query_words:
            if word in text:
                score += 1

        if score > best_score:
            best_score = score
            best_document = document

    return best_document, best_score


# Load HR documents
documents = load_documents()

# Load the 40 fixed questions
questions = pd.read_csv(QUESTION_FILE)

results = []

for _, row in questions.iterrows():
    question = row["question"]

    result, score = keyword_search(question, documents)

    if result:
        document_name = result["filename"]
    else:
        document_name = "No result"

    results.append({
        "id": row["id"],
        "category": row["category"],
        "question": question,
        "ground_truth": row["ground_truth"],
        "keyword_document": document_name,
        "keyword_score": score
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("Keyword search evaluation completed.")
print(f"Total questions: {len(results_df)}")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nFirst 5 results:")
print(results_df.head())
