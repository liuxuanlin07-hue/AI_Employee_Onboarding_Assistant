"""
RAG Retrieval Evaluation Module

Purpose:
    This module evaluates semantic document retrieval using embeddings
    and FAISS.

Main responsibilities:
    - Load HR source documents
    - Generate document embeddings
    - Build a FAISS vector index
    - Load the fixed evaluation questions
    - Retrieve the most relevant HR document for each question
    - Save retrieval results for later evaluation

Input:
    - HR documents stored in the HR_Documents folder
    - Evaluation/questions.csv

Output:
    - Evaluation/rag_retrieval_results.csv
"""

import os
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_FOLDER = os.path.join(BASE_DIR, "HR_Documents")
QUESTION_FILE = os.path.join(BASE_DIR, "Evaluation", "questions.csv")
OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "Evaluation",
    "rag_retrieval_results.csv"
)


OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    raise ValueError(
        "OPENROUTER_API_KEY is not set. "
        "Please set the environment variable before running this script."
    )


client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


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


def get_embedding(text):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


def build_index(documents):
    embeddings = []

    for document in documents:
        embeddings.append(
            get_embedding(document["text"])
        )

    embeddings = np.array(
        embeddings
    ).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def retrieve_document(question, documents, index):
    question_embedding = get_embedding(question)

    question_embedding = np.array(
        [question_embedding]
    ).astype("float32")

    distances, indices = index.search(
        question_embedding,
        1
    )

    document_index = int(indices[0][0])
    distance = float(distances[0][0])

    return documents[document_index], distance


def main():
    print("Creating document embeddings...")

    documents = load_documents()
    index = build_index(documents)

    print("Document embeddings ready.")

    questions = pd.read_csv(QUESTION_FILE)

    print(f"Running {len(questions)} questions...")

    results = []

    for _, row in questions.iterrows():
        document, distance = retrieve_document(
            row["question"],
            documents,
            index
        )

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "rag_document": document["filename"],
                "distance": distance
            }
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("RAG retrieval evaluation completed.")
    print(f"Total questions: {len(result_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("First 5 results:")
    print(result_df.head())


if __name__ == "__main__":
    main()
