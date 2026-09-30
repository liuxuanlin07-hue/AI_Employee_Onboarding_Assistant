import os
import numpy as np
import faiss
import streamlit as st
from openai import OpenAI

DOC_FOLDER = "../HR_Documents"

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

If the document does not contain enough information, say exactly:
"I cannot find enough information in the available HR documents. Please contact HR."

Do not guess.
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


# -------------------------
# Streamlit Interface
# -------------------------

st.set_page_config(
    page_title="AI Employee Onboarding Assistant",
    page_icon="🤖",
    layout="centered"
)

st.title("🤖 AI Employee Onboarding Assistant")

st.write(
    "Ask questions about annual leave, employee benefits, or IT support."
)

st.caption(
    "The assistant answers using the company's HR documents and shows the source when available."
)

st.divider()

st.markdown("### Supported Topics")

col1, col2, col3 = st.columns(3)

with col1:
    st.info("🏖️ Leave")

with col2:
    st.info("🩺 Benefits")

with col3:
    st.info("💻 IT Support")


@st.cache_resource
def prepare_knowledge_base():
    documents = load_documents()
    index = build_index(documents)
    return documents, index


with st.spinner("Preparing HR knowledge base..."):
    documents, index = prepare_knowledge_base()


st.divider()

question = st.text_input(
    "Ask an HR question:",
    placeholder="Example: How much holiday can I take each year?"
)

ask_button = st.button("Ask", type="primary")


if ask_button:

    if question.strip() == "":
        st.warning("Please enter a question.")

    else:
        with st.spinner("Searching HR documents..."):

            document = retrieve_document(
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
            "I cannot find enough information in the available HR documents. "
            "Please contact HR."
        )

        if fallback_text.lower() in answer.lower():

            st.warning("👤 Escalated to Human HR")

            st.caption(
                "The available HR documents do not contain enough information to answer this question safely."
            )

        else:

            st.subheader("Source")
            st.info(f"📄 {document['filename']}")

            with st.expander("View retrieved HR document"):
                st.text(document["text"])


st.divider()

st.caption(
    "Prototype scope: Leave Policies, Employee Benefits, and IT Support. "
    "The assistant does not process leave approvals, payroll changes, or other HR transactions."
)
