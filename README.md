# AI Employee Onboarding Assistant

This project is a Retrieval-Augmented Generation (RAG) based HR onboarding assistant.

The system allows new employees to ask questions in natural language about:

- Annual leave
- Employee benefits
- IT support

The assistant retrieves information from simulated HR documents, generates an answer using an LLM, and shows the source document.

## Project Scope

The prototype covers three areas:

1. Leave policies
2. Employee benefits
3. IT support

The prototype does not perform HR transactions such as leave approval, payroll changes, or performance management.

## System Architecture

Employee Question  
→ Embedding  
→ Vector Search using FAISS  
→ Retrieve Relevant HR Document  
→ LLM Generates Answer  
→ Display Answer and Source

If the available HR documents do not contain enough information, the system escalates the question to Human HR.

## Technologies Used

- Python
- Streamlit
- FAISS
- OpenRouter API
- OpenAI-compatible API
- Pandas
- NumPy

## HR Knowledge Base

The prototype uses simulated HR documents:

- Annual_Leave_Policy.txt
- Employee_Benefits_Guide.txt
- IT_Support_Guide.txt

## Evaluation

A fixed set of 40 manually prepared questions was used for evaluation.

Results:

- Keyword Search Retrieval Accuracy: 92.5%
- RAG Retrieval Accuracy: 100.0%
- RAG Final Answer Score: 97.5%

The evaluation uses:

- Correct = 1
- Partly Correct = 0.5
- Incorrect = 0

## Example

Question:

How much holiday can I take each year?

Answer:

You are entitled to 18 days of paid annual leave per year.

Source:

Annual_Leave_Policy.txt

## Human Escalation

If the system cannot find enough supporting information in the HR documents, it responds:

"I cannot find enough information in the available HR documents. Please contact HR."

## How to Run

Install the required packages:

```bash
pip install openai streamlit pandas numpy faiss-cpu

