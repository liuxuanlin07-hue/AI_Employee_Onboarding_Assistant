"""
AI Employee Onboarding Assistant - Streamlit Application

Purpose:
    This module provides the user-facing Streamlit interface for the
    AI Employee Onboarding Assistant.

Main responsibilities:
    - Load HR knowledge documents
    - Create document embeddings
    - Build a FAISS semantic retrieval index
    - Accept natural-language employee questions
    - Retrieve the most relevant HR document
    - Generate a grounded answer using an LLM through OpenRouter
    - Display the source document when an answer is supported
    - Escalate unsupported questions to Human HR

Input:
    - User question entered in the Streamlit interface
    - HR documents stored in the HR_Documents folder

Output:
    - Generated HR answer
    - Source document when supported
    - Human HR escalation when evidence is insufficient

Safety:
    The assistant is instructed to use only the retrieved HR document
    and not guess when the available documents do not contain enough
    information.
"""

import os
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI


# ---------------------------------------------------------
# Project paths
# ---------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DOC_FOLDER = os.path.join(BASE_DIR, "HR_Documents")


# ---------------------------------------------------------
# OpenRouter client
# ---------------------------------------------------------

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

if not OPENROUTER_API_KEY:
    st.error(
        "OPENROUTER_API_KEY is not set. "
        "Please set the environment variable before running the app."
    )
    st.stop()

client = OpenAI(
    api_key=OPENROUTER_API_KEY,
    base_url="https://openrouter.ai/api/v1"
)


# ---------------------------------------------------------
# Load HR documents
# ---------------------------------------------------------

def load_documents():
    """
    Load all .txt files from the HR_Documents folder.

    Returns:
        list: A list of dictionaries containing filename and text.
    """

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


# ---------------------------------------------------------
# Embedding
# ---------------------------------------------------------

def get_embedding(text):
    """
    Create an embedding for the supplied text using OpenRouter.

    Args:
        text (str): Text to embed.

    Returns:
        list: Embedding vector.
    """

    response = client.embeddings.create(
        model="openai/text-embedding-3-small",
        input=text
    )

    return response.data[0].embedding


# ---------------------------------------------------------
# FAISS index
# ---------------------------------------------------------

@st.cache_resource
def prepare_knowledge_base():
    """
    Load HR documents, create embeddings, and build the FAISS index.

    Returns:
        tuple:
            documents - loaded HR documents
            index - FAISS retrieval index
    """

    documents = load_documents()

    embeddings = []

    for document in documents:
        embedding = get_embedding(document["text"])
        embeddings.append(embedding)

    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]

    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)

    return documents, index


# ---------------------------------------------------------
# Retrieve best document
# ---------------------------------------------------------

def retrieve_document(question, documents, index):
    """
    Retrieve the most relevant HR document for a user question.

    Args:
        question (str): User question.
        documents (list): HR document collection.
        index: FAISS index.

    Returns:
        tuple:
            selected document
            retrieval distance
    """

    question_embedding = get_embedding(question)

    question_embedding = np.array(
        [question_embedding]
    ).astype("float32")

    distances, indices = index.search(question_embedding, 1)

    document_index = int(indices[0][0])
    distance = float(distances[0][0])

    return documents[document_index], distance


# ---------------------------------------------------------
# Generate grounded answer
# ---------------------------------------------------------

def generate_answer(question, document):
    """
    Generate a grounded answer using only the retrieved HR document.

    Args:
        question (str): Employee question.
        document (dict): Retrieved HR document.

    Returns:
        str: Generated answer.
    """

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
                    "You answer employee onboarding questions using only "
                    "the supplied HR document."
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


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------

st.set_page_config(
    page_title="AI Employee Onboarding Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Employee Onboarding Assistant")

st.write(
    "Ask questions about annual leave, employee benefits, or IT support."
)

st.divider()


# ---------------------------------------------------------
# Prepare knowledge base
# ---------------------------------------------------------

with st.spinner("Preparing HR knowledge base..."):
    documents, index = prepare_knowledge_base()


# ---------------------------------------------------------
# User question
# ---------------------------------------------------------

question = st.text_input(
    "Ask an HR question:",
    placeholder="Example: How many days of annual leave do I get?"
)


if st.button("Ask"):

    if not question.strip():

        st.warning("Please enter a question.")

    else:

        try:

            with st.spinner("Searching HR documents..."):

                document, distance = retrieve_document(
                    question,
                    documents,
                    index
                )

                answer = generate_answer(
                    question,
                    document
                )

            st.divider()

            st.subheader("Answer")

            st.write(answer)

            fallback_text = (
                "I cannot find enough information in the available "
                "HR documents. Please contact HR."
            )

            if fallback_text.lower() in answer.lower():

                st.warning("👤 Escalated to Human HR")

                st.caption(
                    "The available HR documents do not contain enough "
                    "information to answer this question safely."
                )

            else:

                st.subheader("Source")

                st.info(document["filename"])

                with st.expander("View retrieved document"):
                    st.text(document["text"])

        except Exception as error:

            st.error("The assistant could not complete the request.")

            st.code(str(error))
