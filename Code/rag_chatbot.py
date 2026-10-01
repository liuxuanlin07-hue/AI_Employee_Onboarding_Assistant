"""
RAG Answer Generation and Evaluation Module

Purpose:
    This module runs the full RAG question-answering pipeline across
    the fixed evaluation dataset.

Main responsibilities:
    - Load HR source documents
    - Create document embeddings
    - Build a FAISS retrieval index
    - Retrieve the most relevant HR document
    - Generate grounded answers using an LLM through OpenRouter
    - Save final answers and their source documents

Input:
    - HR documents stored in the HR_Documents folder
    - Evaluation/questions.csv

Output:
    - Evaluation/rag_answers.csv

Safety:
    The model is instructed to answer only from the retrieved HR
    document and to escalate unsupported questions to Human HR.
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
    "rag_answers.csv"
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

    return documents[document_index]


def generate_answer(question, document):
    prompt = f"""
You are an AI Employee Onboarding Assistant.

Answer the employee question using ONLY the HR document provided below.

Rules:
1. Do not use outside knowledge.
2. Do not invent company policies.
3. Keep the answer short and clear.
4. If the document does not contain enough information to answer the
   question, respond exactly with:

I cannot find enough information in the available HR documents. Please contact HR.

HR DOCUMENT:
{document["text"]}

EMPLOYEE QUESTION:
{question}
"""

    response = client.chat.completions.create(
        model="openai/gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You answer employee onboarding questions using "
                    "only the supplied HR document."
                )
            },
            {
                "role": "user",
                "content": prompt
            }
        ],
        temperature=0
    )

    return response.choices[0].message.content.strip()


def main():
    print("Preparing HR knowledge base...")

    documents = load_documents()
    index = build_index(documents)

    questions = pd.read_csv(QUESTION_FILE)

    print(
        f"Running {len(questions)} questions "
        "through the RAG chatbot..."
    )

    results = []

    for _, row in questions.iterrows():
        document = retrieve_document(
            row["question"],
            documents,
            index
        )

        answer = generate_answer(
            row["question"],
            document
        )

        results.append(
            {
                "id": row["id"],
                "category": row["category"],
                "question": row["question"],
                "ground_truth": row["ground_truth"],
                "rag_answer": answer,
                "source": document["filename"]
            }
        )

        print(
            f"Completed question {row['id']}"
        )

    result_df = pd.DataFrame(results)
    result_df.to_csv(
        OUTPUT_FILE,
        index=False
    )

    print()
    print("RAG answer evaluation file created.")
    print(f"Total questions: {len(result_df)}")
    print(f"Results saved to: {OUTPUT_FILE}")
    print()
    print("First 5 results:")
    print(result_df.head())


if __name__ == "__main__":
    main()
