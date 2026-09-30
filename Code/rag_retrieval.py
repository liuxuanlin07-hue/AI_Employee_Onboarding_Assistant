import os
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

DOC_FOLDER = "../HR_Documents"
QUESTION_FILE = "../Evaluation/questions.csv"
OUTPUT_FILE = "../Evaluation/rag_retrieval_results.csv"

client = OpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)


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


def get_embedding(text):
    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# Load HR documents
documents = load_documents()

print("Creating document embeddings...")

document_embeddings = []

for document in documents:
    embedding = get_embedding(document["text"])
    document_embeddings.append(embedding)

document_embeddings = np.array(document_embeddings).astype("float32")

# Create FAISS index
dimension = document_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(document_embeddings)

print("Document embeddings ready.")

# Load 40 fixed questions
questions = pd.read_csv(QUESTION_FILE)

results = []

print("Running 40 questions...")

for _, row in questions.iterrows():
    question = row["question"]

    query_embedding = get_embedding(question)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, 1)

    best_index = indices[0][0]
    best_document = documents[best_index]["filename"]
    distance = distances[0][0]

    results.append({
        "id": row["id"],
        "category": row["category"],
        "question": question,
        "ground_truth": row["ground_truth"],
        "rag_document": best_document,
        "distance": distance
    })

# Save results
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_FILE, index=False)

print("\nRAG retrieval evaluation completed.")
print(f"Total questions: {len(results_df)}")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nFirst 5 results:")
print(results_df.head())
