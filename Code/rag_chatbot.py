import os
import numpy as np
import pandas as pd
import faiss
from openai import OpenAI

DOC_FOLDER = "../HR_Documents"
QUESTION_FILE = "../Evaluation/questions.csv"
OUTPUT_FILE = "../Evaluation/rag_answers.csv"

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


def build_index(documents):
    embeddings = []

    for document in documents:
        embedding = get_embedding(document["text"])
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return index


def retrieve_document(query, documents, index):
    query_embedding = get_embedding(query)
    query_vector = np.array([query_embedding]).astype("float32")

    distances, indices = index.search(query_vector, 1)

    best_index = indices[0][0]

    return documents[best_index]


def generate_answer(query, document):
    prompt = f"""
You are an HR onboarding assistant.

Answer the employee's question using ONLY the HR document below.

If the document does not contain enough information, say:
"I cannot find enough information in the available HR documents. Please contact HR."

Keep the answer short and clear.

HR Document:
{document["text"]}

Employee Question:
{query}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response.choices[0].message.content


documents = load_documents()

print("Preparing HR knowledge base...")

index = build_index(documents)

questions = pd.read_csv(QUESTION_FILE)

results = []

print("Running 40 questions through the RAG chatbot...")

for _, row in questions.iterrows():
    question = row["question"]

    document = retrieve_document(question, documents, index)

    answer = generate_answer(question, document)

    results.append({
        "id": row["id"],
        "category": row["category"],
        "question": question,
        "ground_truth": row["ground_truth"],
        "rag_answer": answer,
        "source": document["filename"]
    })

    print(f"Completed question {row['id']}")

results_df = pd.DataFrame(results)

results_df.to_csv(OUTPUT_FILE, index=False)

print("\nRAG answer evaluation file created.")
print(f"Total questions: {len(results_df)}")
print(f"Results saved to: {OUTPUT_FILE}")

print("\nFirst 5 results:")
print(results_df.head())
