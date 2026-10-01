# AI Employee Onboarding Assistant

## Project Overview

The AI Employee Onboarding Assistant is a RAG-based chatbot designed to help new employees quickly find information about annual leave, employee benefits, and IT support.

Employees can ask questions in natural language. The system retrieves the most relevant HR document and uses an LLM to generate a short answer based on that document.

If there is not enough information in the available documents, the system does not guess. It asks the employee to contact Human HR.

## Persona

Primary users are new employees who need quick answers during onboarding.

Secondary users are HR and IT support teams who may receive repeated questions from employees.

## Input

The input is a natural-language employee question.

Example:

```text
How much holiday can I take each year?
```

The knowledge base contains three simulated documents:

```text
Annual_Leave_Policy.txt
Employee_Benefits_Guide.txt
IT_Support_Guide.txt
```

## Output

The system returns a short answer and the source document when the answer is supported.

Example:

```text
Answer:
You are entitled to 18 days of paid annual leave per year.

Source:
Annual_Leave_Policy.txt
```

If the available documents do not support the question, the system returns:

```text
I cannot find enough information in the available HR documents.
Please contact HR.
```

## Product Architecture

```text
Employee Question
        |
        v
Query Embedding
        |
        v
FAISS Semantic Retrieval
        |
        v
Relevant HR Document
        |
        v
LLM via OpenRouter
        |
        v
Grounded Answer
        |
        +----------------------+
        |                      |
        v                      v
Answer + Source        Human HR Escalation
```

## How the System Works

1. The employee enters a question.
2. The question is converted into an embedding.
3. FAISS searches for the most relevant HR document.
4. The retrieved document is provided to the LLM.
5. The LLM generates an answer using only the retrieved information.
6. The system displays the answer and source.
7. Unsupported questions are escalated to Human HR.

## Why RAG

A keyword-search baseline was built for comparison.

Keyword search achieved 92.5% retrieval accuracy, while RAG semantic retrieval achieved 100%.

The main improvement appeared on paraphrased questions. For example, an employee may use the word "holiday" while the company policy uses "annual leave".

Semantic retrieval can identify this similarity better than simple keyword matching.

## Data

The prototype uses three simulated HR documents instead of real confidential company data.

More information about the dataset is available in:

```text
HR_Documents/README.md
```

## Evaluation

The project uses a fixed set of 40 evaluation questions covering Leave, Benefits, and IT Support.

The same questions were used for the keyword baseline and RAG evaluation.

Final answers were scored using:

```text
Correct = 1
Partly Correct = 0.5
Incorrect = 0
```

More details are available in:

```text
Evaluation/README.md
```

## Metrics Targeted

The original project target was at least 85% answer accuracy.

## Metrics Reached

| Metric | Result |
| --- | --- |
| Keyword Retrieval Accuracy | 92.5% (37/40) |
| RAG Retrieval Accuracy | 100.0% (40/40) |
| RAG Final Answer Score | 97.5% |

The results show that semantic retrieval performed better than keyword matching on the current evaluation set.

However, perfect retrieval did not guarantee perfect generation. One final answer was still incorrect even though the correct document had been retrieved.

## Safety and Human Escalation

The assistant is instructed to answer only from the retrieved HR document.

If there is not enough evidence, the system escalates the question to Human HR instead of guessing.

The prototype does not:

Approve leave

Change payroll

Modify employee records

Make employment decisions

## Project Structure

```text
AI_Employee_Onboarding_Assistant/
|
|-- Code/
|   |-- app.py
|   |-- keyword_search.py
|   |-- rag_retrieval.py
|   |-- rag_chatbot.py
|
|-- Evaluation/
|   |-- questions.csv
|   |-- keyword_results.csv
|   |-- rag_retrieval_results.csv
|   |-- rag_answers.csv
|   |-- rag_scores.csv
|   |-- evaluation_summary.csv
|   |-- README.md
|
|-- HR_Documents/
|   |-- Annual_Leave_Policy.txt
|   |-- Employee_Benefits_Guide.txt
|   |-- IT_Support_Guide.txt
|   |-- README.md
|
|-- README.md
|-- requirements.txt
|-- .gitignore
```

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the OpenRouter API key:

```bash
export OPENROUTER_API_KEY="YOUR_OPENROUTER_KEY"
```

Do not store the API key directly in the source code or GitHub repository.

Run the application:

```bash
streamlit run Code/app.py
```

Then open:

```text
http://localhost:8501
```

## Limitations

The current prototype uses only three short simulated documents and 40 evaluation questions.

It does not currently test large enterprise document collections, conflicting policy versions, user permissions, long multi-turn conversations, or high production traffic.

The system also depends on an external API, so latency, pricing, and provider availability can affect performance.

## Future Improvements

Future versions could add document chunking, top-k retrieval, metadata filtering, persistent embeddings, role-based access control, larger evaluation datasets, cost monitoring, document version management, and stronger confidence-based escalation.

The next step should focus on better evaluation, document management, access control, and human oversight before adding more autonomous HR functions.
